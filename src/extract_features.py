import re
import time
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

LENGTH_CONFOUND_THRESHOLD = 0.7  # |corr| at or above this against query length -> flagged

client = anthropic.Anthropic()

FEATURE_MODEL = "claude-haiku-4-5"  # cheap classification/counting task, doesn't need judge-tier quality
FEATURE_MAX_TOKENS = 150

PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}
BATCH_DISCOUNT = 0.5

_DIGIT_RE = re.compile(r"\d+")
_CAPITALIZED_WORD_RE = re.compile(r"(?<!^)(?<!\. )(?<!\.\n)\b[A-Z][a-z]+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


# --- Non-LLM (free) features -------------------------------------------------

def build_length_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["query_char_count"] = df["query"].str.len()
    out["query_sentence_count"] = df["query"].apply(
        lambda q: max(1, len([s for s in _SENTENCE_SPLIT_RE.split(q) if s.strip()]))
    )
    out["query_avg_word_length"] = df["query"].apply(
        lambda q: (sum(len(w) for w in q.split()) / len(q.split())) if q.split() else 0.0
    )
    out["query_has_question_mark"] = df["query"].str.contains(r"\?", regex=True).astype(int)
    return out


def build_entity_density_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regex-based proxies for numeric/technical density, not a full NER model:
    digit and mid-sentence-capitalized-word counts, normalized by word count."""
    out = pd.DataFrame(index=df.index)

    def count_digit_tokens(q: str) -> int:
        return len(_DIGIT_RE.findall(q))

    def count_capitalized_words(q: str) -> int:
        return len(_CAPITALIZED_WORD_RE.findall(q))

    word_counts = df["query"].str.split().str.len().clip(lower=1)
    out["query_number_count"] = df["query"].apply(count_digit_tokens)
    out["query_number_density"] = out["query_number_count"] / word_counts
    out["query_capitalized_word_count"] = df["query"].apply(count_capitalized_words)
    out["query_capitalized_density"] = out["query_capitalized_word_count"] / word_counts
    return out


# --- LLM-extracted features ---------------------------------------------------

FEATURE_PROMPT_TEMPLATE = """Analyze this query and classify it along four dimensions. Do not answer the query itself.

Query:
{query}

1. reasoning_step_count: estimate how many distinct reasoning/lookup steps are needed to answer this well (integer, 1-10).
2. ambiguity_score: how underspecified/ambiguous is this query on its own, with no other context (integer 1-10; 1 = fully unambiguous, 10 = severely underspecified).
3. domain: the primary subject domain -- one of: coding, math, writing, factual, creative, other.
4. question_type: the shape of the task -- one of: single_fact_lookup, multi_step_reasoning, open_ended."""

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_step_count": {"type": "integer"},
        "ambiguity_score": {"type": "integer"},
        "domain": {"type": "string", "enum": ["coding", "math", "writing", "factual", "creative", "other"]},
        "question_type": {"type": "string", "enum": ["single_fact_lookup", "multi_step_reasoning", "open_ended"]},
    },
    "required": ["reasoning_step_count", "ambiguity_score", "domain", "question_type"],
    "additionalProperties": False,
}


def estimate_llm_feature_cost(df: pd.DataFrame, sample_size: int = 30) -> dict:
    sample = df["query"].sample(n=min(sample_size, len(df)), random_state=42)
    counts = [
        client.messages.count_tokens(
            model=FEATURE_MODEL,
            messages=[{"role": "user", "content": FEATURE_PROMPT_TEMPLATE.format(query=q)}],
        ).input_tokens
        for q in sample
    ]
    avg_input = sum(counts) / len(counts)
    n = len(df)
    input_cost = avg_input * n / 1_000_000 * PRICE_PER_MTOK["input"] * BATCH_DISCOUNT
    expected_output_cost = 0.5 * FEATURE_MAX_TOKENS * n / 1_000_000 * PRICE_PER_MTOK["output"] * BATCH_DISCOUNT
    worst_output_cost = FEATURE_MAX_TOKENS * n / 1_000_000 * PRICE_PER_MTOK["output"] * BATCH_DISCOUNT
    return {
        "n_requests": n,
        "avg_input_tokens": round(avg_input, 1),
        "expected_cost_usd": round(input_cost + expected_output_cost, 2),
        "worst_case_cost_usd": round(input_cost + worst_output_cost, 2),
    }


def build_llm_feature_batch_requests(df: pd.DataFrame) -> list[Request]:
    requests = []
    for row in df.itertuples():
        prompt = FEATURE_PROMPT_TEMPLATE.format(query=row.query)
        requests.append(
            Request(
                custom_id=row.query_id,
                params=MessageCreateParamsNonStreaming(
                    model=FEATURE_MODEL,
                    max_tokens=FEATURE_MAX_TOKENS,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": FEATURE_SCHEMA}},
                ),
            )
        )
    return requests


def submit_llm_feature_batch(df: pd.DataFrame):
    return client.messages.batches.create(requests=build_llm_feature_batch_requests(df))


def wait_for_batch(batch_id: str, poll_seconds: int = 45):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def _extract_text(result):
    return next((b.text for b in result.result.message.content if b.type == "text"), None)


def parse_llm_feature_batch_results(batch_id: str) -> tuple[pd.DataFrame, list[dict]]:
    import json

    rows = []
    failed = []
    for result in client.messages.batches.results(batch_id):
        query_id = result.custom_id
        text = _extract_text(result) if result.result.type == "succeeded" else None
        if text is not None:
            parsed = json.loads(text)
            rows.append({
                "query_id": query_id,
                "reasoning_step_count": parsed.get("reasoning_step_count"),
                "ambiguity_score": parsed.get("ambiguity_score"),
                "domain": parsed.get("domain"),
                "question_type": parsed.get("question_type"),
            })
        else:
            rows.append({
                "query_id": query_id,
                "reasoning_step_count": None,
                "ambiguity_score": None,
                "domain": None,
                "question_type": None,
            })
            error = result.result.type if result.result.type != "succeeded" else f"empty_text:{result.result.message.stop_reason}"
            failed.append({"query_id": query_id, "error": error})

    return pd.DataFrame(rows), failed


def retry_failed_feature_batch(df: pd.DataFrame, failed_df: pd.DataFrame, llm_features_df: pd.DataFrame):
    import json

    query_by_id = df.set_index("query_id")["query"].to_dict()
    requests = []
    for row in failed_df.itertuples():
        prompt = FEATURE_PROMPT_TEMPLATE.format(query=query_by_id[row.query_id])
        requests.append(
            Request(
                custom_id=row.query_id,
                params=MessageCreateParamsNonStreaming(
                    model=FEATURE_MODEL,
                    max_tokens=FEATURE_MAX_TOKENS,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": FEATURE_SCHEMA}},
                ),
            )
        )

    batch = client.messages.batches.create(requests=requests)
    batch = wait_for_batch(batch.id)

    still_failed = []
    llm_features_df = llm_features_df.set_index("query_id")
    for result in client.messages.batches.results(batch.id):
        query_id = result.custom_id
        text = _extract_text(result) if result.result.type == "succeeded" else None
        if text is not None:
            parsed = json.loads(text)
            for k in ("reasoning_step_count", "ambiguity_score", "domain", "question_type"):
                llm_features_df.loc[query_id, k] = parsed.get(k)
        else:
            error = result.result.type if result.result.type != "succeeded" else f"empty_text:{result.result.message.stop_reason}"
            still_failed.append({"query_id": query_id, "error": error})

    return llm_features_df.reset_index(), still_failed


# --- Length confound check ----------------------------------------------------

def length_confound_check(
    features_df: pd.DataFrame,
    length_col: str = "query_word_count",
    threshold: float = LENGTH_CONFOUND_THRESHOLD,
) -> pd.DataFrame:
    """Correlate every numeric feature against raw query length. A feature at or above
    `threshold` |corr| is likely just re-encoding length, not adding real signal."""
    numeric_cols = [
        c for c in features_df.select_dtypes(include="number").columns
        if c != length_col and not c.endswith("_id")
    ]
    rows = []
    for col in numeric_cols:
        corr = features_df[col].corr(features_df[length_col])
        rows.append({
            "feature": col,
            "corr_with_length": round(float(corr), 3) if pd.notna(corr) else None,
            "length_confounded": bool(abs(corr) >= threshold) if pd.notna(corr) else False,
        })
    return pd.DataFrame(rows).sort_values("corr_with_length", key=lambda s: s.abs(), ascending=False)


# Dropped for failing the length confound check; density-normalized counterparts kept instead.
LENGTH_CONFOUNDED_COLS = [
    "query_char_count", "query_sentence_count",
    "query_capitalized_word_count", "query_number_count",
]


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "query_dataset.parquet")

    length_feats = build_length_features(df)
    entity_feats = build_entity_density_features(df)
    free_feats = pd.concat([df[["query_id"]], length_feats, entity_feats], axis=1)
    free_feats.to_parquet(DATA_DIR / "phase2_free_features.parquet", index=False)
    print(f"Built {len(free_feats.columns) - 1} free features for {len(free_feats)} queries.")

    estimate = estimate_llm_feature_cost(df)
    print(f"Pre-flight LLM-feature cost estimate: {estimate}")

    batch = submit_llm_feature_batch(df)
    print(f"Submitted feature batch {batch.id}, polling until complete...")
    batch = wait_for_batch(batch.id)

    llm_feats, failed = parse_llm_feature_batch_results(batch.id)
    print(f"Batch complete. {len(failed)} failed queries.")

    retries = 0
    while failed and retries < 2:
        retries += 1
        print(f"Retry round {retries}: resubmitting {len(failed)} failed queries...")
        llm_feats, failed = retry_failed_feature_batch(df, pd.DataFrame(failed), llm_feats)
        print(f"After retry {retries}: {len(failed)} still failed.")

    llm_feats.to_parquet(DATA_DIR / "phase2_llm_features_raw.parquet", index=False)

    combined = df[["query_id", "query", "query_word_count", "source_category", "label"]].merge(
        free_feats, on="query_id"
    ).merge(llm_feats, on="query_id")

    report = length_confound_check(combined, length_col="query_word_count")
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    report.to_csv(PROJECT_ROOT / "reports" / "phase2_length_confound_check.csv", index=False)
    print(report.to_string(index=False))

    final = combined.drop(columns=[c for c in LENGTH_CONFOUNDED_COLS if c in combined.columns])
    final.to_parquet(DATA_DIR / "features.parquet", index=False)
    print(f"Saved data/features.parquet: {len(final)} rows, {len(final.columns)} columns.")

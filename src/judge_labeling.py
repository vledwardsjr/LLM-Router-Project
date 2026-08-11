import json
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

JUDGE_MODEL = "claude-opus-5"
JUDGE_MAX_TOKENS = 180  # 4 ints + a 1-sentence reasoning fits comfortably

VALIDATION_SAMPLE_SIZE = 75
VALIDATION_RANDOM_STATE = 42
AGREEMENT_WITHIN1_THRESHOLD = 0.80
AGREEMENT_CORR_THRESHOLD = 0.70

SCORE_GAP_THRESHOLD = 1  # points, on the 1-10 scale

FINAL_COLUMNS = [
    "query_id", "query", "source_category", "query_word_count",
    "small_model_output", "large_model_output",
    "small_model_score", "large_model_score",
    "small_correctness_score", "small_completeness_score", "small_quality_score", "small_reasoning",
    "large_correctness_score", "large_completeness_score", "large_quality_score", "large_reasoning",
    "label",
]

client = anthropic.Anthropic()

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator scoring how well an AI assistant's response answers a user's query.

Query:
{query}

Response:
{response}

Score this response on three dimensions, each a 1-10 integer (1 = completely fails, 10 = excellent):
- correctness: is the information/reasoning factually and logically correct?
- completeness: does the response fully address everything the query asked for?
- quality: is the response well-written, clear, and appropriately detailed (not padded, not truncated)?

Also give an overall_score (1-10) reflecting your holistic judgment of whether this response
is "good enough" - weight correctness most heavily, then completeness, then quality.

Give your reasoning as ONE concise sentence (max 20 words) justifying your scores."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer"},
        "correctness_score": {"type": "integer"},
        "completeness_score": {"type": "integer"},
        "quality_score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["overall_score", "correctness_score", "completeness_score", "quality_score", "reasoning"],
    "additionalProperties": False,
}


def wait_for_batch(batch_id: str, poll_seconds: int = 60):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def build_judge_batch_requests(df: pd.DataFrame) -> list[Request]:
    requests = []
    for row in df.itertuples():
        for slot, output_col in (("small", row.small_model_output), ("large", row.large_model_output)):
            if output_col is None or (isinstance(output_col, float) and pd.isna(output_col)):
                continue
            prompt = JUDGE_PROMPT_TEMPLATE.format(query=row.query, response=output_col)
            requests.append(
                Request(
                    custom_id=f"{row.query_id}__{slot}_judge",
                    params=MessageCreateParamsNonStreaming(
                        model=JUDGE_MODEL,
                        max_tokens=JUDGE_MAX_TOKENS,
                        # Opus 5 thinks by default, which can eat the token budget before
                        # any JSON is written. Disable for this short structured task.
                        thinking={"type": "disabled"},
                        messages=[{"role": "user", "content": prompt}],
                        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                    ),
                )
            )
    return requests


def submit_judge_batch(df: pd.DataFrame):
    return client.messages.batches.create(requests=build_judge_batch_requests(df))


def _clip_score(value) -> int | None:
    if value is None:
        return None
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return None


def parse_judge_batch_results(
    batch_id: str,
    out_path: Path = DATA_DIR / "step3_judge_scores_long.parquet",
) -> pd.DataFrame:
    rows = []
    failed_rows = []
    for result in client.messages.batches.results(batch_id):
        query_id, rest = result.custom_id.split("__")
        model_type = rest.replace("_judge", "")
        text = None
        if result.result.type == "succeeded":
            text = next((b.text for b in result.result.message.content if b.type == "text"), None)

        if text is not None:
            scores = json.loads(text)
            rows.append({
                "query_id": query_id,
                "model_type": model_type,
                "overall_score": _clip_score(scores.get("overall_score")),
                "correctness_score": _clip_score(scores.get("correctness_score")),
                "completeness_score": _clip_score(scores.get("completeness_score")),
                "quality_score": _clip_score(scores.get("quality_score")),
                "reasoning": scores.get("reasoning"),
            })
        else:
            rows.append({
                "query_id": query_id,
                "model_type": model_type,
                "overall_score": None,
                "correctness_score": None,
                "completeness_score": None,
                "quality_score": None,
                "reasoning": None,
            })
            error = result.result.type if result.result.type != "succeeded" else f"empty_text:{result.result.message.stop_reason}"
            failed_rows.append({"query_id": query_id, "model_type": model_type, "error": error})

    judge_long_df = pd.DataFrame(rows)
    judge_long_df.to_parquet(out_path, index=False)

    failed_path = DATA_DIR / "step3_failed_judge.csv"
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()

    return judge_long_df, failed_rows


def retry_failed_judge_batch(
    model_outputs_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    out_path: Path = DATA_DIR / "step3_judge_scores_long.parquet",
):
    output_lookup = model_outputs_df.set_index("query_id")
    requests = []
    for row in failed_df.itertuples():
        output_text = output_lookup.loc[row.query_id, f"{row.model_type}_model_output"]
        prompt = JUDGE_PROMPT_TEMPLATE.format(query=output_lookup.loc[row.query_id, "query"], response=output_text)
        requests.append(
            Request(
                custom_id=f"{row.query_id}__{row.model_type}_judge",
                params=MessageCreateParamsNonStreaming(
                    model=JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                ),
            )
        )

    batch = client.messages.batches.create(requests=requests)
    batch = wait_for_batch(batch.id)

    judge_long_df = pd.read_parquet(out_path)
    still_failed = []
    for result in client.messages.batches.results(batch.id):
        query_id, rest = result.custom_id.split("__")
        model_type = rest.replace("_judge", "")
        text = _extract_text(result) if result.result.type == "succeeded" else None
        if text is not None:
            scores = json.loads(text)
            mask = (judge_long_df["query_id"] == query_id) & (judge_long_df["model_type"] == model_type)
            for field, key in (
                ("overall_score", "overall_score"), ("correctness_score", "correctness_score"),
                ("completeness_score", "completeness_score"), ("quality_score", "quality_score"),
            ):
                judge_long_df.loc[mask, field] = _clip_score(scores.get(key))
            judge_long_df.loc[mask, "reasoning"] = scores.get("reasoning")
        else:
            error = result.result.type if result.result.type != "succeeded" else f"empty_text:{result.result.message.stop_reason}"
            still_failed.append({"query_id": query_id, "model_type": model_type, "error": error})

    judge_long_df.to_parquet(out_path, index=False)

    failed_path = DATA_DIR / "step3_failed_judge.csv"
    if still_failed:
        pd.DataFrame(still_failed).to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()

    return judge_long_df, still_failed


def _extract_text(result) -> str | None:
    return next((b.text for b in result.result.message.content if b.type == "text"), None)


def export_validation_sample(
    judge_long_df: pd.DataFrame,
    model_outputs_df: pd.DataFrame,
    out_path: Path = DATA_DIR / "step3_validation_sample.csv",
    n: int = VALIDATION_SAMPLE_SIZE,
    random_state: int = VALIDATION_RANDOM_STATE,
) -> pd.DataFrame:
    scored = judge_long_df.dropna(subset=["overall_score"])
    sample = scored.sample(n=min(n, len(scored)), random_state=random_state).copy()

    output_lookup = model_outputs_df.set_index("query_id")
    def get_output(row):
        col = f"{row['model_type']}_model_output"
        return output_lookup.loc[row["query_id"], col]

    sample["query"] = sample["query_id"].map(model_outputs_df.set_index("query_id")["query"])
    sample["output"] = sample.apply(get_output, axis=1)
    sample = sample[["query_id", "model_type", "query", "output"]].copy()
    sample["human_score"] = ""  # judge's score deliberately withheld -- blind scoring
    sample.to_csv(out_path, index=False)
    return sample


def compute_agreement(
    judge_long_df: pd.DataFrame,
    scored_csv_path: Path = DATA_DIR / "step3_validation_sample_scored.csv",
    out_path: Path = DATA_DIR / "step3_agreement_report.json",
) -> dict:
    scored = pd.read_csv(scored_csv_path)
    merged = scored.merge(
        judge_long_df[["query_id", "model_type", "overall_score"]],
        on=["query_id", "model_type"],
    )
    merged = merged.dropna(subset=["human_score", "overall_score"])

    exact_match_rate = (merged["human_score"] == merged["overall_score"]).mean()
    within_1_rate = (merged["human_score"] - merged["overall_score"]).abs().le(1).mean()
    correlation = merged["human_score"].corr(merged["overall_score"])

    gate_passed = bool(
        within_1_rate >= AGREEMENT_WITHIN1_THRESHOLD
        and correlation >= AGREEMENT_CORR_THRESHOLD
    )

    report = {
        "n_validated": int(len(merged)),
        "exact_match_rate": float(exact_match_rate),
        "within_1_rate": float(within_1_rate),
        "pearson_correlation": float(correlation),
        "within_1_threshold": AGREEMENT_WITHIN1_THRESHOLD,
        "correlation_threshold": AGREEMENT_CORR_THRESHOLD,
        "gate_passed": gate_passed,
    }
    out_path.write_text(json.dumps(report, indent=2))
    return report


def derive_labels_and_save(
    model_outputs_df: pd.DataFrame,
    judge_long_df: pd.DataFrame,
    out_path: Path = DATA_DIR / "query_dataset.parquet",
) -> pd.DataFrame:
    judge_wide = judge_long_df.pivot(index="query_id", columns="model_type")
    judge_wide.columns = [
        f"{model}_model_score" if stat == "overall_score" else f"{model}_{stat}"
        for stat, model in judge_wide.columns
    ]
    judge_wide = judge_wide.reset_index()

    labeled = model_outputs_df.merge(judge_wide, on="query_id", how="inner")
    labeled = labeled.dropna(subset=["small_model_score", "large_model_score"])

    labeled["label"] = (
        (labeled["large_model_score"] - labeled["small_model_score"]) <= SCORE_GAP_THRESHOLD
    ).astype(int)

    final_df = labeled[FINAL_COLUMNS]
    final_df.to_parquet(out_path, index=False)
    return final_df

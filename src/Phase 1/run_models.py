import hashlib
import time
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

SUBSAMPLE_SIZE = 900
RANDOM_STATE = 42

SMALL_MODEL = "claude-haiku-4-5"
LARGE_MODEL = "claude-sonnet-5"
MAX_TOKENS_RESPONSE = 600

client = anthropic.Anthropic()


def build_query_id(query: str) -> str:
    return hashlib.sha1(query.encode()).hexdigest()[:12]


def make_subsample(
    raw_path: Path = DATA_DIR / "raw_queries_parquet",
    out_path: Path = DATA_DIR / "subsampled_queries.parquet",
    n: int = SUBSAMPLE_SIZE,
) -> pd.DataFrame:
    df = pd.read_parquet(raw_path)
    df["query_id"] = df["query"].apply(build_query_id)

    df_subsampled, _ = train_test_split(
        df,
        train_size=n,
        stratify=df["source_category"],
        random_state=RANDOM_STATE,
    )
    df_subsampled = df_subsampled.reset_index(drop=True)
    df_subsampled.to_parquet(out_path, index=False)
    return df_subsampled


def _request_kwargs(model_name: str) -> dict:
    # Sonnet 5 thinks by default, which can eat the token budget before any
    # answer is written. Haiku 4.5 doesn't support disabling thinking, so only
    # apply this to the model that needs it.
    if model_name == LARGE_MODEL:
        return {"thinking": {"type": "disabled"}}
    return {}


def run_single_query(query: str, model_name: str) -> dict:
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=MAX_TOKENS_RESPONSE,
            messages=[{"role": "user", "content": query}],
            **_request_kwargs(model_name),
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        return {"output": text, "error": None}
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        return {"output": None, "error": str(e)}


def build_model_batch_requests(df: pd.DataFrame) -> list[Request]:
    requests = []
    for row in df.itertuples():
        for slot, model in (("small", SMALL_MODEL), ("large", LARGE_MODEL)):
            requests.append(
                Request(
                    custom_id=f"{row.query_id}__{slot}",
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=MAX_TOKENS_RESPONSE,
                        messages=[{"role": "user", "content": row.query}],
                        **_request_kwargs(model),
                    ),
                )
            )
    return requests


def submit_model_batch(df: pd.DataFrame):
    return client.messages.batches.create(requests=build_model_batch_requests(df))


def wait_for_batch(batch_id: str, poll_seconds: int = 60):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def _extract_text(result) -> str | None:
    return next((b.text for b in result.result.message.content if b.type == "text"), None)


def parse_model_batch_results(
    batch_id: str,
    df: pd.DataFrame,
    out_path: Path = DATA_DIR / "step2_model_outputs.parquet",
    failed_path: Path = DATA_DIR / "step2_failed_queries.csv",
):
    outputs: dict[str, dict] = {}
    failed_rows = []
    for result in client.messages.batches.results(batch_id):
        query_id, slot = result.custom_id.split("__")
        outputs.setdefault(query_id, {})
        text = _extract_text(result) if result.result.type == "succeeded" else None
        outputs[query_id][f"{slot}_model_output"] = text
        if result.result.type != "succeeded":
            failed_rows.append({"query_id": query_id, "slot": slot, "error": result.result.type})
        elif text is None:
            # Succeeded but no text block -- still needs a retry.
            failed_rows.append({"query_id": query_id, "slot": slot, "error": f"empty_text:{result.result.message.stop_reason}"})

    outputs_df = pd.DataFrame.from_dict(outputs, orient="index").reset_index(names="query_id")
    out_df = df.merge(outputs_df, on="query_id", how="left")
    out_df.to_parquet(out_path, index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()

    return out_df, failed_rows


def build_retry_requests(df: pd.DataFrame, failed_df: pd.DataFrame) -> list[Request]:
    query_by_id = df.set_index("query_id")["query"].to_dict()
    model_by_slot = {"small": SMALL_MODEL, "large": LARGE_MODEL}
    requests = []
    for row in failed_df.itertuples():
        model = model_by_slot[row.slot]
        requests.append(
            Request(
                custom_id=f"{row.query_id}__{row.slot}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=MAX_TOKENS_RESPONSE,
                    messages=[{"role": "user", "content": query_by_id[row.query_id]}],
                    **_request_kwargs(model),
                ),
            )
        )
    return requests


def retry_failed_batch(
    df: pd.DataFrame,
    failed_df: pd.DataFrame,
    out_path: Path = DATA_DIR / "step2_model_outputs.parquet",
    failed_path: Path = DATA_DIR / "step2_failed_queries.csv",
):
    batch = client.messages.batches.create(requests=build_retry_requests(df, failed_df))
    batch = wait_for_batch(batch.id)

    out_df = pd.read_parquet(out_path)
    still_failed = []
    for result in client.messages.batches.results(batch.id):
        query_id, slot = result.custom_id.split("__")
        col = f"{slot}_model_output"
        text = _extract_text(result) if result.result.type == "succeeded" else None
        out_df.loc[out_df["query_id"] == query_id, col] = text
        if result.result.type != "succeeded":
            still_failed.append({"query_id": query_id, "slot": slot, "error": result.result.type})
        elif text is None:
            still_failed.append({"query_id": query_id, "slot": slot, "error": f"empty_text:{result.result.message.stop_reason}"})

    out_df.to_parquet(out_path, index=False)
    if still_failed:
        pd.DataFrame(still_failed).to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()

    return out_df, still_failed


if __name__ == "__main__":
    df_sub = make_subsample()
    print(f"Subsampled {len(df_sub)} queries across {df_sub['source_category'].nunique()} categories")

    batch = submit_model_batch(df_sub)
    print(f"Submitted batch {batch.id}, polling until complete...")
    batch = wait_for_batch(batch.id)

    out_df, failed = parse_model_batch_results(batch.id, df_sub)
    print(f"Batch complete. {len(failed)} failed (query_id, slot) pairs.")

    retries = 0
    while failed and retries < 2:
        retries += 1
        print(f"Retry round {retries}: resubmitting {len(failed)} failed pairs...")
        out_df, failed = retry_failed_batch(df_sub, pd.DataFrame(failed))
        print(f"After retry {retries}: {len(failed)} still failed.")

    print(f"Done. Final failed count: {len(failed)}. Saved to data/step2_model_outputs.parquet")

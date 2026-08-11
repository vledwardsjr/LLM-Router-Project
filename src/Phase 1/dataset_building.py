from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
df = ds.to_pandas()


def build_query(row):
    if row["context"]:
        return f"{row['instruction']}\n\nContext:\n{row['context']}"
    return row['instruction']


df["query"] = df.apply(build_query, axis=1)
df = df.rename(columns={'category': 'source_category'})
df = df[["query", "source_category"]].copy()

df = df.drop_duplicates(subset="query").reset_index(drop=True)
df["query_word_count"] = df["query"].str.split().str.len()

df_sampled = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)

print(f"Sampled {len(df_sampled)} queries across {df_sampled['source_category'].nunique()} categories")
print(df_sampled["source_category"].value_counts())
print(df_sampled["query_word_count"].describe())

df_sampled.to_parquet(DATA_DIR / "raw_queries_parquet", index=False)

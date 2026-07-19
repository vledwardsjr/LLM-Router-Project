import pandas as pd
from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
df = ds.to_pandas()

#Returns the instructions + context(if applicable) from each row
def build_query(row):
    if row["context"]:
        return f"{row['instruction']}\n\nContext:\n{row['context']}"
    return row['instruction']

#changes the category name from "category" -> "source category", condenses dataframe to two columns ("query", "source_category")
df["query"] = df.apply(build_query, axis = 1)
df = df.rename(columns={'category' : 'source_category'})
df = df[["query", "source_category"]].copy()

#Remove duplicates form query
df = df.drop_duplicates(subset="query").reset_index(drop=True)

#gets the query word count
df["query_word_count"] = df["query"].str.split().str.len()

#Samples 3000 queries over 8 categories
df_sampled = df.sample(n = min(3000, len(df)), random_state=42).reset_index(drop=True)

#Prints the value counts for each sampled category, and stats about their word counts
print(f"Sampled {len(df_sampled)} queries across {df_sampled['source_category'].nunique()} categories")
print(df_sampled["source_category"].value_counts())
print(df_sampled["query_word_count"].describe())

df_sampled.to_parquet('/Users/vledwards09/Desktop/personal projects/LLM-Router-Project/data/raw_queries_parquet', index=False)

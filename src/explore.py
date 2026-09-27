import pandas as pd
import numpy as np
import gc

UP = "/mnt/user-data/uploads"

def profile_source(path, name):
    print(f"\n{'='*60}\n{name}: {path}\n{'='*60}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("\nnull/empty rate:")
    print((df.isna().sum() / len(df)).round(4))
    print("\ncountry distribution:")
    print(df["country"].value_counts(dropna=False).head(10))
    print("\nentity_id duplicated?", df["entity_id"].duplicated().sum())
    name_len = df["business_name"].fillna("").str.len()
    print("\nbusiness_name length stats:", name_len.describe()[["mean","50%","min","max"]].to_dict())
    addr_len = df["business_address"].fillna("").str.len()
    print("business_address length stats:", addr_len.describe()[["mean","50%","min","max"]].to_dict())
    non_ascii_name = df["business_name"].fillna("").apply(lambda s: any(ord(c) > 127 for c in s))
    print("non-ascii business_name rate:", non_ascii_name.mean().round(4))
    print("\nsample rows:")
    print(df.sample(min(5, len(df)), random_state=1).to_string())
    del df
    gc.collect()

for split in ["train", "test"]:
    for src in ["source1", "source2", "source3"]:
        profile_source(f"{UP}/{split}_{src}.tsv", f"{split}_{src}")

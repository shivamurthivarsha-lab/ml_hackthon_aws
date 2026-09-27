import pandas as pd
import numpy as np

UP = "/mnt/user-data/uploads"
gt = pd.read_csv(f"{UP}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
print("rows:", len(gt))
print("dup source1_entity_id:", gt["source1_entity_id"].duplicated().sum())

def parse(x):
    if x == "" or pd.isna(x):
        return []
    return x.split(",")

gt["matches"] = gt["matched_entity_ids"].apply(parse)
gt["n_matches"] = gt["matches"].apply(len)
print("\nn_matches distribution:")
print(gt["n_matches"].value_counts().sort_index().head(20))
print("\nsingleton (0 matches) rate:", (gt["n_matches"]==0).mean())
print("mean matches (incl 0):", gt["n_matches"].mean())
print("max matches:", gt["n_matches"].max())

# how many from S2 vs S3 per row
def count_src(matches, prefix):
    return sum(1 for m in matches if m.startswith(prefix))
gt["n_s2"] = gt["matches"].apply(lambda m: count_src(m, "S2-"))
gt["n_s3"] = gt["matches"].apply(lambda m: count_src(m, "S3-"))
print("\nmean n_s2:", gt["n_s2"].mean(), "mean n_s3:", gt["n_s3"].mean())
print("rows with n_s2>0:", (gt["n_s2"]>0).mean())
print("rows with n_s3>0:", (gt["n_s3"]>0).mean())

# check for duplicate ids across rows (same S2/S3 id matched to multiple S1 -> many-to-many?)
all_matches = [m for lst in gt["matches"] for m in lst]
print("\ntotal match references:", len(all_matches))
print("unique matched ids:", len(set(all_matches)))
from collections import Counter
c = Counter(all_matches)
multi = sum(1 for v in c.values() if v>1)
print("matched ids appearing in >1 S1 row (many-to-many):", multi)

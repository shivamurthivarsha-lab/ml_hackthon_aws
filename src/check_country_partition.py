import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
UP = str(ROOT / "data" / "uploads")
gt = pd.read_csv(f"{UP}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["matched_entity_ids"]!=""].sample(5000, random_state=3)

need_s1 = set(gt["source1_entity_id"])
need_s2, need_s3 = set(), set()
for m in gt["matched_entity_ids"]:
    for x in m.split(","):
        (need_s2 if x.startswith("S2-") else need_s3).add(x)

def load_needed(path, ids):
    out = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
        sub = chunk[chunk["entity_id"].isin(ids)]
        if len(sub): out.append(sub)
    return pd.concat(out).set_index("entity_id")

s1 = load_needed(f"{UP}/train_source1.tsv", need_s1)
s2 = load_needed(f"{UP}/train_source2.tsv", need_s2)
s3 = load_needed(f"{UP}/train_source3.tsv", need_s3)

mismatch = 0
total = 0
for _, row in gt.iterrows():
    s1c = s1.loc[row["source1_entity_id"], "country"]
    for m in row["matched_entity_ids"].split(","):
        total += 1
        src = s2 if m.startswith("S2-") else s3
        if m in src.index:
            mc = src.loc[m, "country"]
            if mc != s1c:
                mismatch += 1

print("total matches checked:", total, "country mismatches:", mismatch)

import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
UP = str(ROOT / "data" / "uploads")
gt = pd.read_csv(f"{UP}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["matched_entity_ids"]!=""].sample(5000, random_state=3)

s1 = pd.read_csv(f"{UP}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")["country"]

need_s2, need_s3 = set(), set()
pairs = []
for _, row in gt.iterrows():
    for m in row["matched_entity_ids"].split(","):
        pairs.append((row["source1_entity_id"], m))
        (need_s2 if m.startswith("S2-") else need_s3).add(m)

def load_needed(path, ids):
    chunks=[]
    for c in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
        sub = c[c["entity_id"].isin(ids)]
        if len(sub): chunks.append(sub)
    return pd.concat(chunks).set_index("entity_id")["country"] if chunks else pd.Series(dtype=str)

s2 = load_needed(f"{UP}/train_source2.tsv", need_s2)
s3 = load_needed(f"{UP}/train_source3.tsv", need_s3)

mismatch = 0
total = 0
for s1id, m in pairs:
    if s1id not in s1.index: continue
    c1 = s1.loc[s1id]
    src = s2 if m.startswith("S2-") else s3
    if m not in src.index: continue
    c2 = src.loc[m]
    total += 1
    if c1 != c2:
        mismatch += 1
        if mismatch <= 10:
            print("MISMATCH:", s1id, c1, m, c2)

print(f"\ntotal checked: {total}, country mismatches: {mismatch} ({mismatch/total*100:.3f}%)")

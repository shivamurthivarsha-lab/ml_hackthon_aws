import pandas as pd, numpy as np, random
random.seed(0)
UP = "/mnt/user-data/uploads"

gt = pd.read_csv(f"{UP}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["matched_entity_ids"]!=""]
sample_gt = gt.sample(3000, random_state=1)

need_s2, need_s3 = set(), set()
pairs = []
for _, row in sample_gt.iterrows():
    s1 = row["source1_entity_id"]
    for m in row["matched_entity_ids"].split(","):
        pairs.append((s1, m))
        if m.startswith("S2-"): need_s2.add(m)
        else: need_s3.add(m)

s1_ids = set(sample_gt["source1_entity_id"])

def load_needed(path, ids_needed):
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
        sub = chunk[chunk["entity_id"].isin(ids_needed)]
        if len(sub): chunks.append(sub)
    return pd.concat(chunks) if chunks else pd.DataFrame(columns=["entity_id","business_name","business_address","country"])

s1_df = load_needed(f"{UP}/train_source1.tsv", s1_ids).set_index("entity_id")
s2_df = load_needed(f"{UP}/train_source2.tsv", need_s2).set_index("entity_id")
s3_df = load_needed(f"{UP}/train_source3.tsv", need_s3).set_index("entity_id")

print("loaded s1:", len(s1_df), "s2:", len(s2_df), "s3:", len(s3_df))

rows = []
for s1, m in pairs:
    if s1 not in s1_df.index: continue
    src = s2_df if m.startswith("S2-") else s3_df
    if m not in src.index: continue
    r1 = s1_df.loc[s1]
    r2 = src.loc[m]
    rows.append({
        "s1_name": r1["business_name"], "m_name": r2["business_name"],
        "s1_addr": r1["business_address"], "m_addr": r2["business_address"],
        "country": r1["country"]
    })

pairdf = pd.DataFrame(rows)
print("total true pairs sampled:", len(pairdf))
pairdf.to_csv("/home/claude/er_challenge/true_pairs_sample.tsv", sep="\t", index=False)
print(pairdf.sample(15, random_state=2).to_string())

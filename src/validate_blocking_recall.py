import sys, time, random, gc
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from blocking import build_index, get_candidates_for

UP = str(ROOT / "data" / "uploads")
random.seed(0)

COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "India"
N_SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

t0 = time.time()
print(f"[{COUNTRY}] loading source2/source3 (country-filtered, streamed by chunk)...")

def load_country_chunked(path, country, cols=("entity_id", "business_name", "business_address")):
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=300000):
        sub = chunk.loc[chunk["country"] == country, list(cols)]
        if len(sub):
            parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return out

s2 = load_country_chunked(f"{UP}/train_source2.tsv", COUNTRY)
s3 = load_country_chunked(f"{UP}/train_source3.tsv", COUNTRY)
print(f"  s2={len(s2)} s3={len(s3)}  ({time.time()-t0:.1f}s)")

t0 = time.time()
print("building indexes...")
idx2 = build_index(s2)
del s2
gc.collect()
idx3 = build_index(s3)
del s3
gc.collect()
print(f"  done ({time.time()-t0:.1f}s)")

t0 = time.time()
print("loading source1 + ground truth (country-filtered)...")
s1 = pd.read_csv(f"{UP}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = s1[s1["country"] == COUNTRY]
gt = pd.read_csv(f"{UP}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
gt = gt[gt["matched_entity_ids"] != ""]  # focus recall check on entities that DO have matches
sample = gt.sample(min(N_SAMPLE, len(gt)), random_state=0)
s1_lookup = dict(zip(s1["entity_id"], zip(s1["business_name"], s1["business_address"])))
print(f"  s1={len(s1)}  gt(with matches)={len(gt)}  sampled={len(sample)}  ({time.time()-t0:.1f}s)")

t0 = time.time()
total_true = 0
total_found = 0
fully_recalled = 0
missed_examples = []

for _, row in sample.iterrows():
    s1id = row["source1_entity_id"]
    if s1id not in s1_lookup:
        continue
    name, addr = s1_lookup[s1id]
    true_matches = set(row["matched_entity_ids"].split(","))
    true_s2 = {m for m in true_matches if m.startswith("S2-")}
    true_s3 = {m for m in true_matches if m.startswith("S3-")}

    cand2 = get_candidates_for(name, addr, idx2)
    cand3 = get_candidates_for(name, addr, idx3)

    found = (true_s2 & cand2) | (true_s3 & cand3)
    total_true += len(true_matches)
    total_found += len(found)
    if found == true_matches:
        fully_recalled += 1
    else:
        if len(missed_examples) < 10:
            missed_examples.append((s1id, name, addr, true_matches - found))

print(f"scored {len(sample)} entities in {time.time()-t0:.1f}s")
print(f"\nMICRO recall (matches found / all true matches): {total_found/total_true:.4f}  ({total_found}/{total_true})")
print(f"entity-level fully-recalled rate: {fully_recalled/len(sample):.4f}  ({fully_recalled}/{len(sample)})")

if missed_examples:
    print("\nsample of entities with a missed match:")
    for s1id, name, addr, missed in missed_examples:
        print(f"  {s1id} | {name!r} | {addr!r} -> missed {missed}")

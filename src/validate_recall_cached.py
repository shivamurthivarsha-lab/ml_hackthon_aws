import sys, time, random, argparse, pickle, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from blocking import get_candidates_for

ap = argparse.ArgumentParser()
ap.add_argument("--country", default="India")
ap.add_argument("--n", type=int, default=5000)
args = ap.parse_args()
CACHE = str(ROOT / "data" / "cache")
prefix = f"{CACHE}/train_{args.country}"

random.seed(0)
t0 = time.time()
with open(f"{prefix}_s1lookup.pkl", "rb") as f:
    s1_lookup = pickle.load(f)
with open(f"{prefix}_gtlookup.pkl", "rb") as f:
    gt_lookup = pickle.load(f)
print(f"loaded s1/gt ({time.time()-t0:.1f}s)")

matched_ids = [k for k, v in gt_lookup.items() if v]
sample = random.sample(matched_ids, min(args.n, len(matched_ids)))
true_s2 = {s1id: {m for m in gt_lookup[s1id].split(",") if m.startswith("S2-")} for s1id in sample}
true_s3 = {s1id: {m for m in gt_lookup[s1id].split(",") if m.startswith("S3-")} for s1id in sample}
found_s2 = {}
found_s3 = {}

# --- S2 pass (load only idx2, discard before loading idx3) ---
t0 = time.time()
with open(f"{prefix}_s2_idx.pkl", "rb") as f:
    idx2 = pickle.load(f)
print(f"loaded idx2 ({time.time()-t0:.1f}s)")
t0 = time.time()
for s1id in sample:
    name, addr = s1_lookup[s1id]
    found_s2[s1id] = true_s2[s1id] & get_candidates_for(name, addr, idx2)
print(f"S2 pass: {time.time()-t0:.1f}s")
del idx2
gc.collect()

# --- S3 pass ---
t0 = time.time()
with open(f"{prefix}_s3_idx.pkl", "rb") as f:
    idx3 = pickle.load(f)
print(f"loaded idx3 ({time.time()-t0:.1f}s)")
t0 = time.time()
for s1id in sample:
    name, addr = s1_lookup[s1id]
    found_s3[s1id] = true_s3[s1id] & get_candidates_for(name, addr, idx3)
print(f"S3 pass: {time.time()-t0:.1f}s")
del idx3
gc.collect()

total_true = total_found = fully = 0
for s1id in sample:
    tm = true_s2[s1id] | true_s3[s1id]
    fm = found_s2[s1id] | found_s3[s1id]
    total_true += len(tm)
    total_found += len(fm)
    if fm == tm:
        fully += 1

print(f"\nMICRO recall: {total_found/total_true:.4f}  ({total_found}/{total_true})")
print(f"entity fully-recalled: {fully/len(sample):.4f}  ({fully}/{len(sample)})")

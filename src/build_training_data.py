"""
Build a labeled (feature, label) dataset for the matching classifier, using
pre-built cached indexes (see cache_indexes.py).

Two-pass design (S2 fully processed and discarded before S3 is loaded): holding
idx2 AND idx3 in memory at once OOM-killed this container at India's scale
(~2M rows/source) once names were romanized (unidecode expands overlap between
scripts, which grows some buckets). Processing one source at a time keeps peak
memory to roughly one index's worth instead of two.

For each sampled S1 entity:
  1. Blocking (same function used at inference) -> candidate ids for this source
  2. Label: 1 if in ground-truth match list, else 0
  3. Compute pairwise similarity features (features.py)

Negative sampling: candidate sets are mostly non-matches by design (blocking is
recall-first), so we keep ALL positives and cap negatives per entity to bound
runtime/imbalance.

IMPORTANT for generalization: every row keeps s1_entity_id so the trainer can
split by ENTITY (not by row) into train/val - all pairs for one entity stay on
one side of the split, so the model learns transferable string-similarity
patterns instead of memorizing entity-specific quirks.
"""
import sys, os, time, random, argparse, pickle, gc
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking import get_candidates_for
from features import compute_features

DEFAULT_CACHE_DIR = ROOT / "data" / "cache"


def process_source(prefix, source_label, sample_ids, s1_lookup, gt_lookup, neg_cap, rows_out):
    with open(f"{prefix}_{source_label.lower()}_idx.pkl", "rb") as f:
        idx = pickle.load(f)
    with open(f"{prefix}_{source_label.lower()}_lookup.pkl", "rb") as f:
        lookup = pickle.load(f)

    t0 = time.time()
    for i, s1id in enumerate(sample_ids):
        if i and i % 3000 == 0:
            print(f"    [{source_label}] {i}/{len(sample_ids)}  ({time.time()-t0:.1f}s)")
        s1_name, s1_addr = s1_lookup[s1id]
        m = gt_lookup.get(s1id, "")
        true_matches = set(m.split(",")) if m else set()

        cands = get_candidates_for(s1_name, s1_addr, idx)
        pos_ids = [c for c in cands if c in true_matches]
        neg_ids = [c for c in cands if c not in true_matches]
        random.shuffle(neg_ids)
        neg_ids = neg_ids[:neg_cap]

        for cid in pos_ids + neg_ids:
            crec = lookup.get(cid)
            if crec is None:
                continue
            cname, caddr = crec
            feats = compute_features(s1_name, s1_addr, cname, caddr, source_label == "S2")
            feats["s1_entity_id"] = s1id
            feats["cand_entity_id"] = cid
            feats["label"] = int(cid in true_matches)
            rows_out.append(feats)

    print(f"    [{source_label}] done ({time.time()-t0:.1f}s, {len(rows_out)} rows so far)")
    del idx, lookup
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="directory containing cached lookup and index files")
    ap.add_argument("--n-entities", type=int, default=20000)
    ap.add_argument("--neg-cap", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    prefix = str(cache_dir / f"train_{args.country}")
    with open(f"{prefix}_s1lookup.pkl", "rb") as f:
        s1_lookup = pickle.load(f)
    with open(f"{prefix}_gtlookup.pkl", "rb") as f:
        gt_lookup = pickle.load(f)
    print(f"s1={len(s1_lookup)} gt={len(gt_lookup)}")

    s1_ids = list(s1_lookup.keys())
    random.shuffle(s1_ids)
    sample_ids = s1_ids[:min(args.n_entities, len(s1_ids))]
    print(f"sampled {len(sample_ids)} entities")

    rows = []
    print("  processing S2...")
    process_source(prefix, "S2", sample_ids, s1_lookup, gt_lookup, args.neg_cap, rows)
    print("  processing S3...")
    process_source(prefix, "S3", sample_ids, s1_lookup, gt_lookup, args.neg_cap, rows)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_parent = out_path.parent
    if str(out_parent) and str(out_parent) != ".":
        out_parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), sep="\t", index=False)
    print(f"\nwrote {len(df)} rows ({df['label'].mean():.4f} positive rate) to {args.out}")


if __name__ == "__main__":
    main()

"""
Run blocking + trained classifier over TEST data for one country and emit
this country's slice of matching_results.tsv / candidate_pairs.tsv.

Two-pass (S2 then S3) for the same memory reason as build_training_data.py.
Every S1 entity gets exactly one output row, including entities with zero
blocking candidates (empty matched_entity_ids / candidate_entity_ids) -
required by the README's "every S1 entity must appear exactly once" rule.
"""
import sys, os, time, argparse, pickle, gc
from pathlib import Path
import pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from blocking import get_candidates_for
from features import compute_features, precompute_s1, compute_features_fast, FEATURE_NAMES

DEFAULT_CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_MODEL = ROOT / "data" / "model_india.txt"
DEFAULT_OUT_DIR = ROOT / "output" / "predictions"


MAX_CANDIDATES = 30  # cap per-entity candidate count fed to the model for maximum throughput


import multiprocessing as mp

_worker_idx = None
_worker_lookup = None
_worker_booster = None
_worker_threshold = None
_worker_is_s2 = None


def _init_worker(idx_path, lookup_path, model_path, threshold, is_s2):
    global _worker_idx, _worker_lookup, _worker_booster, _worker_threshold, _worker_is_s2
    with open(idx_path, "rb") as f:
        _worker_idx = pickle.load(f)
    with open(lookup_path, "rb") as f:
        _worker_lookup = pickle.load(f)
    _worker_booster = lgb.Booster(model_file=model_path)
    _worker_threshold = threshold
    _worker_is_s2 = is_s2


def _score_chunk(chunk):
    res_cand = []
    res_match = []
    flat_s1ids = []
    flat_cids = []
    flat_feats = []

    for s1id, s1_name, s1_addr in chunk:
        cands = get_candidates_for(s1_name, s1_addr, _worker_idx)
        if not cands:
            continue
        if len(cands) > MAX_CANDIDATES:
            cands = set(list(cands)[:MAX_CANDIDATES])

        s1_prep = precompute_s1(s1_name, s1_addr)
        cand_list = []
        for cid in cands:
            crec = _worker_lookup.get(cid)
            if crec is None:
                continue
            cname, caddr = crec
            feats = compute_features_fast(s1_prep, cname, caddr, _worker_is_s2)

            flat_s1ids.append(s1id)
            flat_cids.append(cid)
            flat_feats.append([feats[fn] for fn in FEATURE_NAMES])
            cand_list.append(cid)

        if cand_list:
            res_cand.append((s1id, cand_list))

    if not flat_feats:
        return res_cand, res_match

    import numpy as np
    X_arr = np.array(flat_feats, dtype=np.float32)
    probs = _worker_booster.predict(X_arr)

    matches_by_s1 = {}
    for s1id, cid, prob in zip(flat_s1ids, flat_cids, probs):
        if prob >= _worker_threshold:
            matches_by_s1.setdefault(s1id, []).append(cid)

    for s1id, m_cids in matches_by_s1.items():
        res_match.append((s1id, m_cids))

    return res_cand, res_match


def process_source(prefix, source_label, s1_ids, s1_lookup, booster_path, threshold,
                   candidates_out, matches_out, n_workers=3, chunk_size=2500):
    idx_path = f"{prefix}_{source_label.lower()}_idx.pkl"
    lookup_path = f"{prefix}_{source_label.lower()}_lookup.pkl"

    records = [(s1id, s1_lookup[s1id][0], s1_lookup[s1id][1]) for s1id in s1_ids]
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

    t0 = time.time()
    print(f"    [{source_label}] scoring {len(s1_ids)} entities across {len(chunks)} chunks with {n_workers} workers...")

    if n_workers <= 1:
        # single process fallback for low RAM
        _init_worker(idx_path, lookup_path, str(Path(booster_path).resolve()), threshold, source_label == "S2")
        for idx, chunk in enumerate(chunks):
            if idx and idx % 20 == 0:
                print(f"    [{source_label}] {idx * chunk_size}/{len(s1_ids)} entities completed ({time.time()-t0:.1f}s)")
            c_cand, c_match = _score_chunk(chunk)
            for s1id, cids in c_cand:
                candidates_out.setdefault(s1id, []).extend(cids)
            for s1id, m_cids in c_match:
                matches_out.setdefault(s1id, []).extend(m_cids)
    else:
        initargs = (idx_path, lookup_path, str(Path(booster_path).resolve()), threshold, source_label == "S2")
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=initargs) as pool:
            for idx, (c_cand, c_match) in enumerate(pool.imap_unordered(_score_chunk, chunks, chunksize=1)):
                if idx and idx % 20 == 0:
                    print(f"    [{source_label}] {idx * chunk_size}/{len(s1_ids)} entities completed ({time.time()-t0:.1f}s)")
                for s1id, cids in c_cand:
                    candidates_out.setdefault(s1id, []).extend(cids)
                for s1id, m_cids in c_match:
                    matches_out.setdefault(s1id, []).extend(m_cids)

    print(f"    [{source_label}] done ({time.time()-t0:.1f}s)")
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="directory containing test indexes and lookups")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--n-entities", type=int, default=None,
                     help="limit to first N test S1 entities (for a quick demo run); omit for full run")
    ap.add_argument("--n-workers", type=int, default=3, help="number of parallel worker processes (default: 3)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    prefix = str(Path(args.cache_dir) / f"test_{args.country}")
    with open(f"{prefix}_s1lookup.pkl", "rb") as f:
        s1_lookup = pickle.load(f)
    s1_ids = list(s1_lookup.keys())
    if args.n_entities:
        s1_ids = s1_ids[:args.n_entities]
    print(f"country={args.country}  s1 entities to score: {len(s1_ids)} (workers={args.n_workers})")

    candidates_out, matches_out = {}, {}
    print("  processing S2...")
    process_source(prefix, "S2", s1_ids, s1_lookup, args.model, args.threshold, candidates_out, matches_out, n_workers=args.n_workers)
    print("  processing S3...")
    process_source(prefix, "S3", s1_ids, s1_lookup, args.model, args.threshold, candidates_out, matches_out, n_workers=args.n_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)
    cand_path = out_dir / f"candidate_pairs_{args.country}.tsv"
    match_path = out_dir / f"matching_results_{args.country}.tsv"

    with open(cand_path, "w") as fc, open(match_path, "w") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for s1id in s1_ids:  # every S1 entity gets exactly one row, even with no candidates
            cand_ids = candidates_out.get(s1id, [])
            match_ids = matches_out.get(s1id, [])
            fc.write(f"{s1id}\t{','.join(cand_ids)}\n")
            fm.write(f"{s1id}\t{','.join(match_ids)}\n")

    n_matched = sum(1 for s1id in s1_ids if matches_out.get(s1id))
    print(f"\nwrote {cand_path}")
    print(f"wrote {match_path}")
    print(f"{n_matched}/{len(s1_ids)} entities got >=1 match "
          f"({n_matched/len(s1_ids)*100:.1f}%)")


if __name__ == "__main__":
    main()

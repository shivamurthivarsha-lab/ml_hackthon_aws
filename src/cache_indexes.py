import sys, os, time, pickle, gc, argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import resource
except ImportError:
    resource = None

from blocking import build_index

DEFAULT_UPLOAD_DIR = ROOT / "data" / "uploads"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"


def rss_mb():
    if resource is None:
        return 0.0

    usage = resource.getrusage(resource.RUSAGE_SELF)
    maxrss = getattr(usage, "ru_maxrss", 0)
    if sys.platform == "darwin":
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def load_country_chunked(path, country, cols=("entity_id", "business_name", "business_address")):
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=300000):
        sub = chunk.loc[chunk["country"] == country, list(cols)]
        if len(sub):
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=list(cols))
    return pd.concat(parts, ignore_index=True)


def process_one_source(path, country, out_prefix, source_label):
    if not os.path.exists(path):
        print(f"  [{source_label}] file not found: {path}")
        print("  Check that the upload TSV exists and pass --upload-dir with the correct folder.")
        return

    t0 = time.time()
    df = load_country_chunked(path, country)
    print(f"  [{source_label}] loaded {len(df)} rows ({time.time()-t0:.1f}s, rss={rss_mb():.0f}MB)")
    if df.empty:
        print(f"  [{source_label}] no rows for country={country}; skipping")
        return

    t0 = time.time()
    idx = build_index(df)
    print(f"  [{source_label}] index built ({time.time()-t0:.1f}s, rss={rss_mb():.0f}MB)")

    lookup = dict(zip(df["entity_id"], zip(df["business_name"], df["business_address"])))
    del df
    gc.collect()

    with open(f"{out_prefix}_idx.pkl", "wb") as f:
        pickle.dump(idx, f, protocol=4)
    del idx
    gc.collect()
    with open(f"{out_prefix}_lookup.pkl", "wb") as f:
        pickle.dump(lookup, f, protocol=4)
    n = len(lookup)
    del lookup
    gc.collect()
    print(f"  [{source_label}] cached ({time.time()-t0:.1f}s total, rss={rss_mb():.0f}MB, n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--upload-dir", default=str(DEFAULT_UPLOAD_DIR),
                    help="directory containing source TSV files")
    ap.add_argument("--out-dir", default=str(DEFAULT_CACHE_DIR),
                    help="directory for cached indexes and lookups")
    ap.add_argument("--sources", default="s1,s2,s3,gt",
                     help="comma list of which pieces to (re)build: s1,s2,s3,gt")
    args = ap.parse_args()
    upload_dir = Path(args.upload_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / f"{args.split}_{args.country}")
    todo = set(args.sources.split(","))

    if "s2" in todo:
        print("processing source2...")
        process_one_source(str(upload_dir / f"{args.split}_source2.tsv"), args.country, f"{prefix}_s2", "S2")

    if "s3" in todo:
        print("processing source3...")
        process_one_source(str(upload_dir / f"{args.split}_source3.tsv"), args.country, f"{prefix}_s3", "S3")

    if "s1" in todo:
        path = str(upload_dir / f"{args.split}_source1.tsv")
        if not os.path.exists(path):
            print(f"  [S1] file not found: {path}")
            print("  Check that the upload TSV exists and pass --upload-dir with the correct folder.")
        else:
            print("processing source1...")
            s1 = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            s1 = s1[s1["country"] == args.country]
            s1_lookup = dict(zip(s1["entity_id"], zip(s1["business_name"], s1["business_address"])))
            with open(f"{prefix}_s1lookup.pkl", "wb") as f:
                pickle.dump(s1_lookup, f, protocol=4)
            print(f"  s1={len(s1_lookup)} cached")

            if args.split == "train" and "gt" in todo:
                gt_path = str(upload_dir / "train_ground_truth.tsv")
                if not os.path.exists(gt_path):
                    print(f"  [GT] file not found: {gt_path}")
                    print("  Check that the upload TSV exists and pass --upload-dir with the correct folder.")
                else:
                    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
                    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
                    gt_lookup = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))
                    with open(f"{prefix}_gtlookup.pkl", "wb") as f:
                        pickle.dump(gt_lookup, f, protocol=4)
                    print(f"  gt={len(gt_lookup)} cached")

    print(f"\ndone. rss={rss_mb():.0f}MB")


if __name__ == "__main__":
    main()

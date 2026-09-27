"""
Combine per-country prediction TSV files into a single, leaderboard-compliant
submission file matching the exact row order and count of test_source1.tsv.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_S1 = ROOT / "data" / "uploads" / "test_source1.tsv"
DEFAULT_OUT_DIR = ROOT / "output" / "predictions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-s1", default=str(DEFAULT_TEST_S1),
                    help="path to test_source1.tsv to get exact entity ID sequence")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="directory containing matching_results_<country>.tsv")
    ap.add_argument("--countries", default="India,US,France")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    matches_map = {}
    candidates_map = {}

    for c in countries:
        m_file = out_dir / f"matching_results_{c}.tsv"
        c_file = out_dir / f"candidate_pairs_{c}.tsv"

        if not m_file.exists():
            print(f"[WARNING] Missing matching results file for {c}: {m_file}")
        else:
            print(f"Loading {m_file}...")
            df_m = pd.read_csv(m_file, sep="\t", dtype=str, keep_default_na=False)
            matches_map.update(dict(zip(df_m["source1_entity_id"], df_m["matched_entity_ids"])))

        if not c_file.exists():
            print(f"[WARNING] Missing candidate pairs file for {c}: {c_file}")
        else:
            print(f"Loading {c_file}...")
            df_c = pd.read_csv(c_file, sep="\t", dtype=str, keep_default_na=False)
            candidates_map.update(dict(zip(df_c["source1_entity_id"], df_c["candidate_entity_ids"])))

    print(f"Loaded total matches for {len(matches_map)} entities across countries.")

    print(f"Reading entity list and order from {args.test_s1}...")
    s1_df = pd.read_csv(args.test_s1, sep="\t", dtype=str, usecols=["entity_id", "country"])

    final_match_path = out_dir / "matching_results.tsv"
    final_cand_path = out_dir / "candidate_pairs.tsv"

    print(f"Writing {final_match_path} and {final_cand_path} in exact order...")
    total_count = len(s1_df)
    matched_count = 0
    missing_count = 0

    country_stats = {c: {"total": 0, "matched": 0} for c in countries}

    with open(final_match_path, "w", encoding="utf-8") as fm, open(final_cand_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for _, row in s1_df.iterrows():
            eid = row["entity_id"]
            cntry = row["country"]

            if cntry in country_stats:
                country_stats[cntry]["total"] += 1

            m_val = matches_map.get(eid)
            c_val = candidates_map.get(eid)

            if m_val is None:
                missing_count += 1
                m_val = ""
            if c_val is None:
                c_val = ""

            if m_val:
                matched_count += 1
                if cntry in country_stats:
                    country_stats[cntry]["matched"] += 1

            fm.write(f"{eid}\t{m_val}\n")
            fc.write(f"{eid}\t{c_val}\n")

        fm.flush()
        fc.flush()

    print("\n--- SANITY CHECK & VALIDATION ---")
    print(f"Total entities processed: {total_count}")
    print(f"Total entities with >=1 match: {matched_count} ({matched_count/total_count*100:.2f}%)")
    if missing_count > 0:
        print(f"[ERROR] {missing_count} entities were missing from country predictions!")
    else:
        print("[OK] Every single S1 entity ID is present in the final output file!")

    for c in countries:
        tot = country_stats[c]["total"]
        mch = country_stats[c]["matched"]
        pct = (mch / tot * 100) if tot > 0 else 0
        print(f"  Country {c:6s}: {mch}/{tot} matched ({pct:.2f}%)")

    print(f"\nWrote final submission file: {final_match_path}")
    print(f"Wrote final candidates file: {final_cand_path}")


if __name__ == "__main__":
    main()

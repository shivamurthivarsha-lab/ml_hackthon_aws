"""
Fast, parallelized master pipeline runner for Entity Resolution challenge.
Runs France, US, and India predictions sequentially in isolated processes using
multi-core worker pools, then combines all results into the final submission file.
"""
import sys
import time
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "data" / "model_combined.txt")
THRESHOLD_PATH = ROOT / "data" / "threshold_combined.json"
OUT_DIR = str(ROOT / "output" / "predictions")
CACHE_DIR = str(ROOT / "data" / "cache")
TEST_S1 = str(ROOT / "data" / "uploads" / "test_source1.tsv")
N_WORKERS = "3"


def main():
    with open(THRESHOLD_PATH, encoding="utf-8") as f:
        threshold = str(json.load(f)["threshold"])

    print("=" * 60)
    print("STARTING FAST PARALLEL PREDICTION AND COMBINATION PIPELINE")
    print(f"Model: {MODEL_PATH}")
    print(f"Threshold: {threshold}")
    print(f"Workers per country: {N_WORKERS}")
    print("=" * 60)

    t_start = time.time()
    countries = ["France", "US", "India"]

    for country in countries:
        m_file = Path(OUT_DIR) / f"matching_results_{country}.tsv"
        c_file = Path(OUT_DIR) / f"candidate_pairs_{country}.tsv"
        if m_file.exists() and m_file.stat().st_size > 1000 and c_file.exists() and c_file.stat().st_size > 1000:
            print(f"\n>>> SKIPPING {country} (already completed: {m_file}) <<<")
            continue

        print(f"\n>>> PROCESSING COUNTRY: {country} <<<")
        t_c = time.time()
        cmd = [
            sys.executable, "src/predict.py",
            "--country", country,
            "--cache-dir", CACHE_DIR,
            "--model", MODEL_PATH,
            "--threshold", threshold,
            "--n-workers", N_WORKERS,
            "--out-dir", OUT_DIR
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] Prediction failed for {country} with code {res.returncode}")
            sys.exit(res.returncode)
        print(f">>> COMPLETED {country} in {time.time() - t_c:.1f}s <<<")

    print("\n>>> COMBINING ALL COUNTRY PREDICTIONS INTO FINAL SUBMISSION <<<")
    t_m = time.time()
    cmd_combine = [
        sys.executable, "src/combine_results.py",
        "--test-s1", TEST_S1,
        "--out-dir", OUT_DIR,
        "--countries", "India,US,France"
    ]
    res = subprocess.run(cmd_combine)
    if res.returncode != 0:
        print(f"[ERROR] Combination failed with code {res.returncode}")
        sys.exit(res.returncode)

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t_start:.1f}s!")
    print("=" * 60)


if __name__ == "__main__":
    main()

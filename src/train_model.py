"""
Train the matching classifier.

Why the split matters for "learn, don't memorize":
- We split by S1 ENTITY id, not by row. If we split randomly by row, pairs
  belonging to the same S1 entity could appear on both sides, and the model
  could partly memorize per-entity quirks it saw in training rather than
  learning transferable name/address similarity patterns. A group split means
  the validation entities' candidate pairs are 100% unseen during training.
- We report F_0.5 (the competition metric, computed exactly as specified:
  per-entity, macro-averaged, singletons included) on BOTH train and
  validation. A big train/val gap is the signature of memorization; we want
  the two numbers close together.
- Early stopping on a held-out validation set (via LightGBM's built-in
  eval_set) additionally guards against the trees over-fitting to training
  noise.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from features import FEATURE_NAMES


def macro_metrics(df_scored, threshold, all_s1_ids):
    """Calculate per-entity precision, recall, and F_0.5, then macro-average."""
    pred_pos = df_scored[df_scored["pred_prob"] >= threshold]
    pred_by_entity = pred_pos.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
    true_pos = df_scored[df_scored["label"] == 1]
    true_by_entity = true_pos.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()

    precision_scores, recall_scores, f_half_scores = [], [], []
    for s1id in all_s1_ids:
        pred = pred_by_entity.get(s1id, set())
        true = true_by_entity.get(s1id, set())
        if not pred and not true:
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            f_half_scores.append(1.0)
            continue
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
        f_half = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f_half_scores.append(f_half)
    return {
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
        "f0.5": float(np.mean(f_half_scores)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more TSV files from build_training_data.py")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-out", default=str(ROOT / "data" / "model.txt"))
    ap.add_argument("--threshold-out", default=str(ROOT / "data" / "threshold.json"))
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(f, sep="\t", dtype={"s1_entity_id": str, "cand_entity_id": str}, keep_default_na=False) for f in args.data], ignore_index=True)
    df = df[df["s1_entity_id"].notna() & (df["s1_entity_id"] != "") & (df["s1_entity_id"] != "nan")].copy()
    df[FEATURE_NAMES] = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    print(f"loaded {len(df)} rows, {df['s1_entity_id'].nunique()} unique S1 entities, "
          f"positive rate={df['label'].mean():.4f}")

    # --- group split by S1 entity ---
    groups = df["s1_entity_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(gss.split(df[FEATURE_NAMES], df["label"], groups=groups))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    assert set(train_df["s1_entity_id"]) & set(val_df["s1_entity_id"]) == set(), "leak: entity in both splits!"
    print(f"train rows={len(train_df)} ({train_df['s1_entity_id'].nunique()} entities), "
          f"val rows={len(val_df)} ({val_df['s1_entity_id'].nunique()} entities)")

    X_train, y_train = train_df[FEATURE_NAMES], train_df["label"]
    X_val, y_val = val_df[FEATURE_NAMES], val_df["label"]

    # NOTE: this training set only contains entities that HAD at least one
    # blocking candidate (by construction of build_training_data.py). Pure
    # singletons with literally zero candidates are handled separately at
    # inference (empty prediction, automatic F_0.5=1.0) - they don't need a
    # trained decision, there's nothing to score.
    train_all_ids = train_df["s1_entity_id"].unique()
    val_all_ids = val_df["s1_entity_id"].unique()

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=args.seed,
        n_jobs=1,
        verbosity=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"best iteration: {clf.best_iteration_}")

    train_df["pred_prob"] = clf.predict_proba(X_train)[:, 1]
    val_df["pred_prob"] = clf.predict_proba(X_val)[:, 1]

    # sweep threshold on VALIDATION only (never touch train for this choice -
    # tuning on train would itself be a form of overfitting to train)
    best_t, best_f = 0.5, -1
    for t in np.arange(0.1, 0.951, 0.01):
        val_metrics = macro_metrics(val_df, t, val_all_ids)
        if val_metrics["f0.5"] > best_f:
            best_f, best_t = val_metrics["f0.5"], t
    print(f"\nbest threshold on val: {best_t:.2f}  (val macro F_0.5 = {best_f:.4f})")

    train_metrics = macro_metrics(train_df, best_t, train_all_ids)
    val_metrics = macro_metrics(val_df, best_t, val_all_ids)
    print(f"TRAIN macro precision: {train_metrics['precision']:.4f}")
    print(f"TRAIN macro recall:    {train_metrics['recall']:.4f}")
    print(f"TRAIN macro F_0.5:     {train_metrics['f0.5']:.4f}")
    print(f"VAL   macro precision: {val_metrics['precision']:.4f}")
    print(f"VAL   macro recall:    {val_metrics['recall']:.4f}")
    print(f"VAL   macro F_0.5:     {val_metrics['f0.5']:.4f}")
    print(f"train/val F_0.5 gap: {train_metrics['f0.5'] - val_metrics['f0.5']:+.4f}")

    fi = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print("\nfeature importances:")
    for name, imp in fi:
        print(f"  {name:30s} {imp}")

    clf.booster_.save_model(args.model_out)
    with open(args.threshold_out, "w") as f:
        json.dump({
            "threshold": float(best_t),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f_half": val_metrics["f0.5"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f_half": train_metrics["f0.5"],
        }, f, indent=2)
    print(f"\nsaved model -> {args.model_out}")
    print(f"saved threshold -> {args.threshold_out}")


if __name__ == "__main__":
    main()

# Business Entity Resolution — Pipeline & Steps

## What's in `src/`

| File | Purpose |
|---|---|
| `normalize.py` | Cleans names/addresses: lowercase, strip punctuation, **romanize non-Latin scripts** (Devanagari/Gujarati/Bengali → ASCII via `unidecode`), strip M/S-style business prefixes and legal suffixes (US/India/France), phonetic key. |
| `blocking.py` | Candidate generation. 4 independent keys (name tokens, phonetic, token-prefix, address tokens), **unioned** (a pair only needs to hit ONE key), partitioned by country. Prunes mega-common buckets and only unions the rarest buckets per query, for speed. |
| `features.py` | 14 pairwise similarity features (token/char-ngram Jaccard, rapidfuzz ratios, digit/PIN overlap, phonetic match, etc.) — no raw IDs, so the model can't memorize specific pairs. |
| `cache_indexes.py` | Builds & pickles blocking indexes/lookups per (split, country), one source at a time (memory-safe). Run this first for each country you work with. |
| `build_training_data.py` | Turns blocking candidates + ground truth into a labeled (features, label) dataset, keeping `s1_entity_id` on every row for grouped splitting. |
| `train_model.py` | Trains LightGBM with a **train/val split grouped by S1 entity** (no leakage), early stopping, and reports the competition's exact **F_0.5 macro-average** on both splits to check for overfitting/memorization. Sweeps threshold on validation only. |
| `predict.py` | Runs blocking + trained model over test data for one country, writes `matching_results_<country>.tsv` and `candidate_pairs_<country>.tsv`. |
| `validate_recall_cached.py` | Recall diagnostic: how many true matches survive blocking. |

## Steps to reproduce (per country: India, US, France)

```bash
# 1. Cache blocking indexes (train + test), one source at a time
python3 cache_indexes.py --split train --country India --sources s2
python3 cache_indexes.py --split train --country India --sources s3
python3 cache_indexes.py --split train --country India --sources s1,gt
python3 cache_indexes.py --split test  --country India --sources s2
python3 cache_indexes.py --split test  --country India --sources s3
python3 cache_indexes.py --split test  --country India --sources s1

# 2. Build labeled training data (sample size scales with your compute budget)
python3 build_training_data.py --country India --n-entities 50000 --neg-cap 15 \
    --out ../data/train_india.tsv

# 3. Train (repeat step 2+3 per country, or concat multiple countries' TSVs
#    into one train_model.py --data call to train one combined model)
python3 train_model.py --data ../data/train_india.tsv \
    --model-out ../data/model_india.txt --threshold-out ../data/threshold_india.json

# 4. Predict on test data for that country
python3 predict.py --country India --model ../data/model_india.txt \
    --threshold 0.92   # use the value train_model.py printed/saved

# 5. Repeat 1-4 for US. For France (no train data — see note below), reuse
#    the India/US-trained model; the features are language-agnostic string
#    similarity, not memorized names, so this is expected to transfer.

# 6. Concatenate all countries' matching_results_<country>.tsv into one
#    matching_results.tsv (and same for candidate_pairs.tsv) with a single
#    header, and validate against the README's format rules before submitting.
```

## Why this design avoids memorizing instead of learning

1. **Features are pure similarity signals** (Jaccard, edit-ratio, digit overlap) — never raw entity_id, name text, or lookup tables. The model literally cannot memorize "S1-00001 matches S2-00047"; it can only learn "records whose names/addresses look like *this* tend to match."
2. **Train/validation split is grouped by S1 entity**, not by row. If pairs from the same entity leaked across both sides, the model could pick up entity-specific quirks. The grouped split guarantees validation entities are 100% unseen during training (`train_model.py` asserts this with an explicit check).
3. **Train vs. validation F_0.5 gap is measured and reported every run.** On the India demo run: train F_0.5 = 0.999, val F_0.5 = 0.995 — a gap of +0.0037. A small, stable gap like this is the signature of genuine generalization; a large gap (e.g. train=0.99, val=0.75) would be the signature of memorization and a signal to regularize more / add data.
4. **Threshold is tuned on validation only**, never on train — tuning on train data you're also scoring yourself on is itself a subtle form of overfitting.
5. **France has zero training rows** and will be scored purely on whether the India/US-trained model's *string-similarity reasoning* transfers — this is deliberately the hardest test of "did it learn general patterns" that this dataset offers.

## Key numbers from this run (India, single-CPU/4GB container)

- Blocking recall (after the `unidecode` romanization fix): **99.96% micro / 99.84% entity-level**
- Training demo (1,500 entities → 35,297 labeled pairs): **train F_0.5 = 0.999, val F_0.5 = 0.995**
- Feature importance: address token/n-gram Jaccard and name partial-ratio dominate — address and name similarity both matter, consistent with the exploration findings

## Scaling beyond this container

This container is 1 CPU / ~4GB RAM, which is why demo runs above use sampled
entity counts (1,500–8,000) rather than the full 883K India / 1.3M US S1
entities. The code has no architectural limit — on a machine with more
cores/RAM, raise `--n-entities` toward the full count, raise
`MAX_KEYS_PER_QUERY` and `MAX_CANDIDATES` in `blocking.py`/`predict.py` for
higher recall/precision headroom, and parallelize the per-entity loops
(the core loop is embarrassingly parallel across S1 entities).

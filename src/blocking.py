"""
Blocking / candidate generation.

Recall-first design: a candidate pair only needs to match on ONE of several
independent keys (union, not intersection). Precision is cleaned up later by
the classifier - this stage's only job is to not lose true matches.

Keys used (all computed on normalized name tokens, per-country partition):
  1. token key       - each significant name token (catches word-order changes,
                        catches "M S Ramayya" vs "Ramayya" style partial overlaps)
  2. phonetic key     - Soundex-lite of the first token (catches spelling typos
                        even when zero tokens match exactly)
  3. ngram key        - character 4-grams of the full normalized name, joined with
                        an index (catches typos anywhere in the name, not just first word)
  4. city key         - first alphabetic token found in the normalized address
                        (safety net when name is badly mangled but location matches)

Partitioning by country first (verified 0 cross-country matches in ground truth)
cuts the search space ~3x before any of the above keys are even applied.
"""
import pandas as pd
from collections import defaultdict
from normalize import normalize_name, normalize_address, phonetic_key, ADDRESS_STOPWORDS

MIN_TOKEN_LEN = 2          # ignore 1-char tokens as a blocking key (too common/noisy)
PREFIX_N = 4               # per-token prefix length used as a cheap typo-tolerant key
MAX_BUCKET = 200000        # drop any index key whose posting list exceeds this after
                            # build (e.g. common words / city names hit 100k+ records
                            # in this data) - a bucket that big carries almost no
                            # discriminative signal on its own and is what made
                            # candidate generation extremely slow; other keys
                            # (the union of 4) still catch these pairs


def token_prefix_keys(tokens):
    """Cheap typo-tolerant key: first PREFIX_N chars of each token (length >= PREFIX_N).
    Much smaller footprint than a full-name character-ngram index (which OOM'd on
    ~2-5M row sources at 4GB RAM) while still catching typos *inside* a token, since
    most single-character typos/spellings still share the token's first few chars."""
    keys = set()
    for t in tokens:
        if len(t) >= PREFIX_N:
            keys.add(t[:PREFIX_N])
    return keys


def extract_city_token(addr_tokens):
    """First alphabetic token of length >= 4 in the address, used only as a weak
    extra blocking key - not a geocoder, just a string heuristic."""
    for t in addr_tokens:
        if len(t) >= 4 and t.isalpha():
            return t
    return None


ADDR_MIN_TOKEN_LEN = 3
# A curated stopword list (states, generic address filler words) rather than a
# corpus-wide document-frequency pass: computing/storing per-token DF over
# 4M+ rows with many high-cardinality address tokens was the actual cause of the
# OOM kill during development (see conversation) - a fixed-size stopword set gives
# the same effect (don't block on generic, non-discriminative words) at O(1) memory
# instead of O(vocabulary size).
_ADDR_STOP = ADDRESS_STOPWORDS


def build_index(df: pd.DataFrame):
    """df must have columns entity_id, business_name, business_address.
    Single pass, streaming - never retains per-record token lists, only the
    resulting inverted indexes, to stay memory-safe on large sources."""
    token_idx = defaultdict(list)
    phon_idx = defaultdict(list)
    prefix_idx = defaultdict(list)
    addr_idx = defaultdict(list)

    for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
        norm_name, name_tokens = normalize_name(name)
        norm_addr, addr_tokens = normalize_address(addr)

        for tok in set(name_tokens):
            if len(tok) >= MIN_TOKEN_LEN:
                token_idx[tok].append(eid)

        if name_tokens:
            phon_idx[phonetic_key(name_tokens[0])].append(eid)

        for pk in token_prefix_keys(name_tokens):
            prefix_idx[pk].append(eid)

        for tok in set(addr_tokens):
            if len(tok) >= ADDR_MIN_TOKEN_LEN and tok not in _ADDR_STOP:
                addr_idx[tok].append(eid)

    # prune mega-buckets (see MAX_BUCKET comment above)
    for idx in (token_idx, phon_idx, prefix_idx, addr_idx):
        for k in [k for k, v in idx.items() if len(v) > MAX_BUCKET]:
            del idx[k]

    return {"token": token_idx, "phon": phon_idx, "prefix": prefix_idx, "addr": addr_idx}


MAX_KEYS_PER_QUERY = 10    # per key-type, only union the N *rarest* (smallest)
                            # buckets for this query, instead of every matching
                            # key. A true match virtually always shares more than
                            # one token (see exploration: mean name-token-Jaccard
                            # ~0.66 for true pairs), so we don't need to pull every
                            # shared token's full posting list - just enough rare
                            # ones to catch it, which is what actually made
                            # set.update() the dominant cost (profiled).


def _smallest_buckets(index, keys, limit):
    """Return up to `limit` keys (from `keys`) with the smallest posting lists
    in `index`, cheapest-first, without materializing large lists early."""
    sized = [(k, len(index[k])) for k in keys if k in index]
    sized.sort(key=lambda kv: kv[1])
    return [k for k, _ in sized[:limit]]


def get_candidates_for(name, addr, indexes, max_cands=30):
    """Given a query record's raw name/address, return the union of candidate
    entity_ids across all four keys. Only the rarest buckets per key-type are
    unioned (see MAX_KEYS_PER_QUERY) to keep this fast at millions-of-rows scale.
    Early candidate set capping (max_cands=30) prevents set explosion from generic tokens."""
    norm_name, name_tokens = normalize_name(name)
    norm_addr, addr_tokens = normalize_address(addr)

    cands = set()

    name_keys = [t for t in set(name_tokens) if len(t) >= MIN_TOKEN_LEN]
    for tok in _smallest_buckets(indexes["token"], name_keys, MAX_KEYS_PER_QUERY):
        needed = max_cands - len(cands)
        cands.update(indexes["token"][tok][:needed])
        if len(cands) >= max_cands:
            return cands

    if name_tokens:
        pk_list = indexes["phon"].get(phonetic_key(name_tokens[0]), ())
        needed = max_cands - len(cands)
        cands.update(pk_list[:needed])
        if len(cands) >= max_cands:
            return cands

    prefix_keys = token_prefix_keys(name_tokens)
    for pk in _smallest_buckets(indexes["prefix"], prefix_keys, MAX_KEYS_PER_QUERY):
        needed = max_cands - len(cands)
        cands.update(indexes["prefix"][pk][:needed])
        if len(cands) >= max_cands:
            return cands

    addr_keys = [t for t in set(addr_tokens) if len(t) >= ADDR_MIN_TOKEN_LEN and t not in _ADDR_STOP]
    for tok in _smallest_buckets(indexes["addr"], addr_keys, MAX_KEYS_PER_QUERY):
        needed = max_cands - len(cands)
        cands.update(indexes["addr"][tok][:needed])
        if len(cands) >= max_cands:
            return cands

    return cands

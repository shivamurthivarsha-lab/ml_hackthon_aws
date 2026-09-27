"""
Pairwise feature engineering.

Every feature here is a *similarity signal*, never a raw ID or a memorized
lookup — that's what keeps the downstream model able to generalize to S1/S2/S3
records it has never seen (including France, which has zero training rows),
instead of memorizing specific entity_id pairs.
"""
import re
from rapidfuzz import fuzz
from normalize import normalize_name, normalize_address, phonetic_key

DIGIT_RE = re.compile(r"\d+")


def char_ngrams(s: str, n: int = 3) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return set(s[i:i + n] for i in range(len(s) - n + 1))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def digit_tokens(s: str) -> set:
    """Numeric tokens (PIN codes, street numbers, unit numbers). Strong signal
    when present, since these rarely match by chance."""
    return set(DIGIT_RE.findall(s))


FEATURE_NAMES = [
    "name_token_jaccard", "name_core_token_jaccard", "name_ngram_jaccard",
    "name_token_sort_ratio", "name_partial_ratio", "name_first_token_match",
    "name_len_diff_ratio", "name_phonetic_match",
    "addr_token_jaccard", "addr_ngram_jaccard", "addr_token_sort_ratio",
    "addr_digit_jaccard", "addr_digit_overlap_count",
    "is_source2",
]


def precompute_s1(s1_name: str, s1_addr: str):
    name1_str, name1_tok = normalize_name(s1_name)
    addr1_str, addr1_tok = normalize_address(s1_addr)
    t1 = set(name1_tok)
    at1 = set(addr1_tok)
    ng1 = char_ngrams(name1_str)
    addr_ng1 = char_ngrams(addr1_str)
    d1 = digit_tokens(addr1_str)
    len1 = len(name1_str)
    first_tok1 = name1_tok[0] if name1_tok else None
    phon1 = phonetic_key(first_tok1) if first_tok1 else None
    return (name1_str, name1_tok, t1, ng1, addr1_str, addr1_tok, at1, addr_ng1, d1, len1, first_tok1, phon1)


def compute_features_fast(s1_prep, cand_name: str, cand_addr: str, cand_is_s2: bool) -> dict:
    name1_str, name1_tok, t1, ng1, addr1_str, addr1_tok, at1, addr_ng1, d1, len1, first_tok1, phon1 = s1_prep

    name2_str, name2_tok = normalize_name(cand_name)
    addr2_str, addr2_tok = normalize_address(cand_addr)

    t2 = set(name2_tok)
    at2 = set(addr2_tok)
    ng2 = char_ngrams(name2_str)
    addr_ng2 = char_ngrams(addr2_str)
    d2 = digit_tokens(addr2_str)

    len2 = len(name2_str)
    len_diff_ratio = abs(len1 - len2) / max(len1, len2, 1)

    first_tok2 = name2_tok[0] if name2_tok else None
    first_match = bool(first_tok1 and first_tok2 and first_tok1 == first_tok2)
    phon2 = phonetic_key(first_tok2) if first_tok2 else None
    phon_match = bool(phon1 and phon2 and phon1 == phon2)

    return {
        "name_token_jaccard": jaccard(t1, t2),
        "name_core_token_jaccard": jaccard(t1, t2),
        "name_ngram_jaccard": jaccard(ng1, ng2),
        "name_token_sort_ratio": fuzz.token_sort_ratio(name1_str, name2_str) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(name1_str, name2_str) / 100.0,
        "name_first_token_match": float(first_match),
        "name_len_diff_ratio": len_diff_ratio,
        "name_phonetic_match": float(phon_match),
        "addr_token_jaccard": jaccard(at1, at2),
        "addr_ngram_jaccard": jaccard(addr_ng1, addr_ng2),
        "addr_token_sort_ratio": fuzz.token_sort_ratio(addr1_str, addr2_str) / 100.0,
        "addr_digit_jaccard": jaccard(d1, d2),
        "addr_digit_overlap_count": float(len(d1 & d2)),
        "is_source2": float(cand_is_s2),
    }


def compute_features(s1_name, s1_addr, cand_name, cand_addr, cand_is_s2: bool) -> dict:
    s1_prep = precompute_s1(s1_name, s1_addr)
    return compute_features_fast(s1_prep, cand_name, cand_addr, cand_is_s2)


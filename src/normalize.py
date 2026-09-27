"""
Normalization utilities for business_name and business_address fields.

Design goals (see conversation with user):
- Strip business-entity prefixes/suffixes (M/S, Ltd, Inc, Pvt Ltd ...) unconditionally
  -- these are NOT personal honorifics, they are pure legal/naming noise.
- Do NOT attempt any geocoding / external lookup - purely string-level normalization.
- Be script-agnostic: works the same way on Latin, Devanagari, Bengali, Telugu, etc.
  text since we never translate, only compare strings against strings of the same kind.
- Cheap and vectorizable - this runs over ~10M+ rows total, on a single CPU.
"""
import re
import unicodedata
from unidecode import unidecode

# --- Business-entity suffixes (legal form indicators) -----------------------------
# Stripped from the END of a name (after tokenizing), regardless of language of the
# rest of the name. Covers US, India and France legal forms per the README's scope.
LEGAL_SUFFIXES = {
    # generic / US
    "inc", "incorporated", "llc", "l l c", "ltd", "limited", "corp", "corporation",
    "co", "company", "llp", "l l p", "plc", "lp", "pc", "pllc",
    # India
    "pvt", "private", "pvtltd", "pltd",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "ste", "societe",
}

# --- Business-entity prefixes -------------------------------------------------------
# "M/S" = Messrs (India), means roughly "the firm/proprietors of" - pure noise,
# NOT a personal honorific like Mr/Mrs/Dr. Stripped unconditionally from the front.
ENTITY_PREFIXES = {"m/s", "ms", "m s", "messrs", "mss"}

# Personal honorifics - handled separately, NOT blindly stripped the same way,
# since they may carry weak proprietor-identity signal. We still normalize their
# punctuation/casing but we don't drop them from the token set.
PERSONAL_HONORIFICS = {"mr", "mrs", "dr", "shri", "smt", "md"}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_LEADING_JUNK_RE = re.compile(r"^[\.\-<>~`'\"\s]+")


def strip_accents(s: str) -> str:
    """Remove diacritics from Latin text (café -> cafe) without touching non-Latin
    scripts (Devanagari etc. pass through unchanged, since NFKD does not decompose
    them the same way and we only strip combining marks)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def romanize(s: str) -> str:
    """Romanize non-Latin scripts (Devanagari, Gujarati, Bengali, Telugu, ...) to
    an ASCII approximation via unidecode, so S2/S3 records written in a local
    script become character-comparable to S1's Latin-script reference name
    (verified in exploration: S1 is always Latin, S2/S3 sometimes aren't, for
    the SAME business - it's a transliteration, not a different business).
    No-op / near no-op for text that's already Latin."""
    return unidecode(s)


def basic_clean(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _LEADING_JUNK_RE.sub("", s)
    s = romanize(s)
    s = strip_accents(s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_name(raw: str):
    """Returns (normalized_string, token_list) with entity prefixes/suffixes removed."""
    s = basic_clean(raw)
    if not s:
        return "", []
    tokens = s.split(" ")

    # strip entity prefixes (M/S, Messrs...) from the front, unconditionally.
    # "M/S" survives punctuation-stripping as two tokens ["m","s"], so check both
    # the single-token form and the two-token form.
    while tokens:
        if tokens[0] in ENTITY_PREFIXES:
            tokens = tokens[1:]
            continue
        if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" in ENTITY_PREFIXES:
            tokens = tokens[2:]
            continue
        break

    # strip trailing junk that isn't part of the business name: long pure-digit
    # tokens (phone numbers etc.) and stray leftover punctuation-only tokens.
    while tokens and (tokens[-1].isdigit() and len(tokens[-1]) >= 6):
        tokens = tokens[:-1]

    # strip legal suffixes from the end (can be more than one, e.g. "pvt ltd"),
    # re-checking after removing trailing digit junk above.
    changed = True
    while changed and tokens:
        changed = False
        if tokens and tokens[-1] in LEGAL_SUFFIXES:
            tokens = tokens[:-1]
            changed = True

    norm = " ".join(tokens)
    return norm, tokens


def normalize_address(raw: str):
    """Returns (normalized_string, token_list). No geocoding - just cleanup."""
    s = basic_clean(raw)
    if not s:
        return "", []
    tokens = s.split(" ")
    return s, tokens


# Common US state abbreviation <-> full name pairs, and a few India state
# abbreviations, used only to help address token overlap - not a geocoder.
US_STATE_ABBR = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "newhampshire",
    "nj": "newjersey", "nm": "newmexico", "ny": "newyork", "nc": "northcarolina",
    "nd": "northdakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhodeisland", "sc": "southcarolina", "sd": "southdakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "westvirginia", "wi": "wisconsin", "wy": "wyoming",
}


INDIA_STATES = {
    "andhra", "pradesh", "arunachal", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal", "jharkhand", "karnataka", "kerala",
    "madhya", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamil", "nadu",
    "telangana", "tripura", "uttar", "uttarakhand", "bengal", "delhi",
    "chandigarh", "puducherry", "jammu", "kashmir", "ladakh",
}

ADDRESS_STOPWORDS = INDIA_STATES | set(US_STATE_ABBR.values()) | set(US_STATE_ABBR.keys()) | {
    # generic filler present all over addresses in every country - not discriminative
    "road", "rd", "street", "st", "nagar", "colony", "layout", "floor", "phase",
    "block", "sector", "near", "opp", "opposite", "behind", "main", "cross", "ward",
    "taluka", "tk", "district", "dist", "village", "po", "ps", "pin", "apartment",
    "apt", "unit", "suite", "ste", "drive", "dr", "avenue", "ave", "lane", "ln",
    "court", "ct", "circle", "cir", "way", "blvd", "boulevard", "highway", "hwy",
    "india", "us", "usa", "france", "city", "town", "north", "south", "east", "west",
    # France
    "rue", "chemin", "allee", "impasse", "place", "cedex",
}


def phonetic_key(token: str) -> str:
    """Cheap phonetic key for a single Latin token (Soundex-lite). Non-Latin tokens
    are returned as-is (first 4 chars) since Soundex only makes sense for Latin script."""
    if not token:
        return ""
    if not re.match(r"^[a-z]+$", token):
        return token[:4]
    codes = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6",
    }
    first = token[0]
    tail = "".join(codes.get(c, "") for c in token[1:])
    # collapse consecutive duplicate codes
    collapsed = []
    for c in tail:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    return (first + "".join(collapsed) + "000")[:4]

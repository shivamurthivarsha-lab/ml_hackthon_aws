import pandas as pd, numpy as np, re, random
random.seed(0)

def norm(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokens(s):
    return set(norm(s).split())

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def char_ngrams(s, n=3):
    s = norm(s).replace(" ", "")
    if len(s) < n: return {s} if s else set()
    return set(s[i:i+n] for i in range(len(s)-n+1))

pairdf = pd.read_csv("/home/claude/er_challenge/true_pairs_sample.tsv", sep="\t", dtype=str, keep_default_na=False)

def feats(row, sname, mname, saddr, maddr):
    nt1, nt2 = tokens(row[sname]), tokens(row[mname])
    ng1, ng2 = char_ngrams(row[sname]), char_ngrams(row[mname])
    at1, at2 = tokens(row[saddr]), tokens(row[maddr])
    return pd.Series({
        "name_tok_jacc": jaccard(nt1, nt2),
        "name_ngram_jacc": jaccard(ng1, ng2),
        "name_first_tok_match": (norm(row[sname]).split()[:1] == norm(row[mname]).split()[:1]) if norm(row[sname]) and norm(row[mname]) else False,
        "addr_tok_jacc": jaccard(at1, at2),
    })

true_feats = pairdf.apply(lambda r: feats(r, "s1_name","m_name","s1_addr","m_addr"), axis=1)
print("TRUE PAIRS feature stats:")
print(true_feats.describe())

# build random negative pairs within same country for comparison
s1_sample = pairdf[["s1_name","s1_addr","country"]].drop_duplicates().reset_index(drop=True)
m_sample = pairdf[["m_name","m_addr","country"]].reset_index(drop=True)
neg_rows = []
for i in range(3000):
    a = s1_sample.sample(1).iloc[0]
    cand = m_sample[m_sample["country"]==a["country"]]
    if len(cand)==0: continue
    b = cand.sample(1).iloc[0]
    neg_rows.append({"s1_name": a["s1_name"], "s1_addr": a["s1_addr"], "m_name": b["m_name"], "m_addr": b["m_addr"]})
negdf = pd.DataFrame(neg_rows)
neg_feats = negdf.apply(lambda r: feats(r, "s1_name","m_name","s1_addr","m_addr"), axis=1)
print("\nRANDOM (mostly non-match) PAIRS feature stats:")
print(neg_feats.describe())

print("\nTRUE: name_ngram_jacc >= 0.3 rate:", (true_feats["name_ngram_jacc"]>=0.3).mean())
print("NEG:  name_ngram_jacc >= 0.3 rate:", (neg_feats["name_ngram_jacc"]>=0.3).mean())
print("TRUE: addr_tok_jacc >= 0.2 rate:", (true_feats["addr_tok_jacc"]>=0.2).mean())
print("NEG:  addr_tok_jacc >= 0.2 rate:", (neg_feats["addr_tok_jacc"]>=0.2).mean())
print("TRUE: name_first_tok_match rate:", true_feats["name_first_tok_match"].mean())
print("NEG:  name_first_tok_match rate:", neg_feats["name_first_tok_match"].mean())

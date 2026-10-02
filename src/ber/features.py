"""Pairwise feature construction for (S1, candidate) pairs.

All features are country-agnostic similarity / agreement measures, so the model
transfers to countries unseen in training (e.g. France). Country itself is never a
feature (blocking already restricts pairs to the same country label).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer

from .normalize import phon

W = -1  # all cores


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32) / 100.0


_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20,
                        alternate_sign=False, norm="l2", dtype=np.float32)


def _hv_chunk(texts):
    return _HV.transform(texts)


def _char_vecs(texts, workers=10, chunk=200_000):
    """Hashed char-3gram L2-normalised vectors, computed in parallel."""
    jobs = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    with Pool(workers) as pool:
        mats = pool.map(_hv_chunk, jobs)
    return sp.vstack(mats).tocsr()


def _char_cos(texts, i1, io, chunk=2_000_000):
    """Row-wise cosine between record i1[k] and io[k]; each record vectorised once."""
    uniq, inv = np.unique(np.concatenate([i1, io]), return_inverse=True)
    X = _char_vecs(list(texts[uniq]))
    p1, p2 = inv[:len(i1)], inv[len(i1):]
    out = np.empty(len(i1), dtype=np.float32)
    for s in range(0, len(i1), chunk):
        out[s:s + chunk] = np.asarray(X[p1[s:s + chunk]].multiply(X[p2[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def _set_feats(a, b):
    """Token-set Jaccard, containment (min-side), shared count for space-joined strings."""
    jac = np.empty(len(a), np.float32)
    cont = np.empty(len(a), np.float32)
    inter = np.empty(len(a), np.int16)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        k = len(sx & sy)
        u = len(sx | sy)
        jac[i] = k / u if u else 0.0
        m = min(len(sx), len(sy))
        cont[i] = k / m if m else 0.0
        inter[i] = k
    return jac, cont, inter


def _num_feats(a, b):
    """House/building number agreement between numeric-token strings."""
    n = len(a)
    jac = np.full(n, -1, np.float32)
    first_eq = np.full(n, -1, np.int8)
    conflict = np.zeros(n, np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        lx, ly = x.split(), y.split()
        sx, sy = set(lx), set(ly)
        k = len(sx & sy)
        jac[i] = k / len(sx | sy)
        first_eq[i] = int(lx[0] == ly[0])
        conflict[i] = int(k == 0)
    return jac, first_eq, conflict


BASE_COLS = ["n_tset", "n_cc_ratio", "a_tset", "a_tsort"]


def _col(recs, col, idx):
    return recs[col].take(idx).to_numpy(dtype=object)


def base_features(cand, recs, chunk=4_000_000):
    """Cheap similarities for ALL pairs (used for group-context features)."""
    out = {c: np.empty(len(cand), np.float32) for c in BASE_COLS}
    for s in range(0, len(cand), chunk):
        i1, io = cand.i1.values[s:s + chunk], cand.io.values[s:s + chunk]
        c1, c2 = _col(recs, "core", i1), _col(recs, "core", io)
        a1, a2 = _col(recs, "addr_c", i1), _col(recs, "addr_c", io)
        cc1 = np.array([x.replace(" ", "") for x in c1], dtype=object)
        cc2 = np.array([x.replace(" ", "") for x in c2], dtype=object)
        out["n_tset"][s:s + chunk] = _cp(c1, c2, fuzz.token_set_ratio)
        out["n_cc_ratio"][s:s + chunk] = _cp(cc1, cc2, fuzz.ratio)
        out["a_tset"][s:s + chunk] = _cp(a1, a2, fuzz.token_set_ratio)
        out["a_tsort"][s:s + chunk] = _cp(a1, a2, fuzz.token_sort_ratio)
    return pd.DataFrame(out)


def pair_features(cand, recs, ctx):
    """Full feature matrix for a chunk of pairs.

    cand: DataFrame[i1, io, bscore, rank_rev, rank_fwd] (chunk); ctx: matching rows of the
    global context-feature frame (see context_features)."""
    i1, io = cand.i1.values, cand.io.values
    n1, n2 = _col(recs, "name_c", i1), _col(recs, "name_c", io)
    c1, c2 = _col(recs, "core", i1), _col(recs, "core", io)
    a1, a2 = _col(recs, "addr_c", i1), _col(recs, "addr_c", io)
    m1, m2 = _col(recs, "nums", i1), _col(recs, "nums", io)
    l1, l2 = _col(recs, "legal", i1), _col(recs, "legal", io)
    cc1 = np.array([x.replace(" ", "") for x in c1], dtype=object)
    cc2 = np.array([x.replace(" ", "") for x in c2], dtype=object)
    p1 = np.array([" ".join(phon(t) for t in x.split()) for x in c1], dtype=object)
    p2 = np.array([" ".join(phon(t) for t in x.split()) for x in c2], dtype=object)

    f = pd.DataFrame(index=cand.index)
    f["bscore"] = cand.bscore.values
    f["rank_rev"] = cand.rank_rev.values
    f["rank_fwd"] = cand.rank_fwd.values
    f["src"] = recs.src.take(io).to_numpy().astype(np.int8)
    f["web"] = recs.web.take(io).to_numpy()
    # name
    f["n_ratio"] = _cp(c1, c2, fuzz.ratio)
    f["n_tsort"] = _cp(c1, c2, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(c1, c2, fuzz.partial_ratio)
    f["n_wratio"] = _cp(n1, n2, fuzz.WRatio)
    f["n_full_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["n_cc_partial"] = _cp(cc1, cc2, fuzz.partial_ratio)
    f["n_jw"] = cpdist(c1, c2, scorer=JaroWinkler.similarity, workers=W, dtype=np.float32)
    f["n_phon_tset"] = _cp(p1, p2, fuzz.token_set_ratio)
    f["n_phon_jac"], _, _ = _set_feats(p1, p2)
    f["n_char3"] = _pair_char_cos(cc1, cc2)
    f["n_jac"], f["n_cont"], f["n_inter"] = _set_feats(c1, c2)
    f["n_exact"] = (c1 == c2).astype(np.int8)
    f["n_len1"] = np.array([len(x) for x in c1], np.int16)
    f["n_len2"] = np.array([len(x) for x in c2], np.int16)
    f["n_ntok1"] = np.array([x.count(" ") + 1 if x else 0 for x in c1], np.int8)
    f["n_ntok2"] = np.array([x.count(" ") + 1 if x else 0 for x in c2], np.int8)
    f["legal_eq"] = (l1 == l2).astype(np.int8)
    f["legal_any2"] = (l2 != "").astype(np.int8)
    f["legal_jac"], _, _ = _set_feats(l1, l2)
    # address
    f["a_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["a_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["a_jac"], f["a_cont"], f["a_inter"] = _set_feats(a1, a2)
    f["a_exact"] = (a1 == a2).astype(np.int8)
    f["a_empty2"] = (a2 == "").astype(np.int8)
    f["a_len1"] = np.array([len(x) for x in a1], np.int16)
    f["a_len2"] = np.array([len(x) for x in a2], np.int16)
    f["num_jac"], f["num_first_eq"], f["num_conflict"] = _num_feats(m1, m2)
    # cross field: candidate-name tokens appearing in the S1 address (landmark / C/O refs)
    f["x_n2_in_a1"], _, _ = _set_feats(c2, a1)
    for c in ctx.columns:
        f[c] = ctx[c].values
    return f


def _pair_char_cos(a, b):
    """Row-wise cosine of hashed char-3gram vectors (serial; chunks are moderate)."""
    xa = _HV.transform(a)
    xb = _HV.transform(b)
    return np.asarray(xa.multiply(xb).sum(axis=1)).ravel().astype(np.float32)


def context_features(base, cand):
    """Group-level features: how this pair compares with competing candidates, both
    among the S1's candidates (g1_*) and among the S1s competing for the same record
    (go_*). Uses only unlabeled candidate structure, so it is valid at test time.
    Returns a DataFrame aligned with cand (includes the base similarity columns)."""
    f = base.copy()
    comp = (base.n_tset + base.n_cc_ratio + base.a_tset + base.a_tsort) / 4.0
    f["comp"] = comp.astype(np.float32)
    d = pd.DataFrame({"i1": cand.i1.values, "io": cand.io.values, "comp": comp.values,
                      "n": base.n_cc_ratio.values, "a": base.a_tsort.values})
    for key, pre in (("i1", "g1"), ("io", "go")):
        grp = d.groupby(key, sort=False)
        mx = grp.comp.transform("max")
        f[pre + "_max"] = mx.values.astype(np.float32)
        f[pre + "_gap"] = (mx - d.comp).values.astype(np.float32)
        f[pre + "_rank"] = grp.comp.rank(ascending=False, method="min").values.astype(np.float32)
        f[pre + "_cnt"] = grp.comp.transform("size").values.astype(np.float32)
        f[pre + "_nmax"] = (grp.n.transform("max") - d.n).values.astype(np.float32)
        f[pre + "_amax"] = (grp.a.transform("max") - d.a).values.astype(np.float32)
    # best competing comp for the same record, excluding this pair
    top2 = d.sort_values(["io", "comp"], ascending=[True, False]).groupby("io", sort=False).comp
    first = top2.nth(0); second = top2.nth(1)
    fi = pd.Series(first.values, index=d.loc[first.index, "io"].values)
    se = pd.Series(second.values, index=d.loc[second.index, "io"].values)
    b1 = d.io.map(fi).values
    b2 = d.io.map(se).fillna(0).values
    f["go_other_best"] = np.where(d.comp.values >= b1, b2, b1).astype(np.float32)
    for thr in (0.6, 0.75):
        f[f"g1_n{int(thr * 100)}"] = (d.comp > thr).groupby(d.i1).transform("sum").values.astype(np.float32)
    return f


# ---------------------------------------------------------------------------------------
# Extra features (v2), computed as a separate cached block (feat2_*), from error analysis:
# true matches often carry small house-number edits (934->933, 4625->462) or an extra /
# dropped generic name word ("Center", "Corp"), while hard negatives differ by an
# informative word ("Services" vs "Travels") or a different street.
# ---------------------------------------------------------------------------------------
from rapidfuzz.distance import Levenshtein  # noqa: E402

EXTRA_COLS = ["num_first_lev", "num_first_prefix", "num_best_lev", "a_street_tset", "a_street_ratio",
              "n_miss_idf1", "n_extra_idf2", "n_miss_max1", "n_extra_max2",
              "a_miss_idf1", "a_extra_idf2", "a_miss_max1", "a_extra_max2", "n_diff_cnt"]


IDF_CAP = 10.0


def token_idf(texts, countries):
    """Scale-invariant IDF: log(N_c / df_c) within each country c (all sources), capped at
    IDF_CAP. Using relative document frequency makes the value of a token independent of
    how many records a split has for that country (test US has half the S1 records of
    train US), which is what lets the model trained on train transfer to test.
    Returns ({(country, token): idf}, IDF_CAP for unseen tokens)."""
    from collections import Counter
    out = {}
    texts = np.asarray(texts, dtype=object)
    countries = np.asarray(countries, dtype=object)
    for c in pd.unique(countries):
        sub = texts[countries == c]
        cnt = Counter()
        for t in sub:
            cnt.update(set(t.split()))
        n = len(sub)
        for k, v in cnt.items():
            out[(c, k)] = min(float(np.log(n / v)), IDF_CAP)
    return out, IDF_CAP


def _idf_diff(a, b, idf, unk, ctry):
    """Sum/max IDF of tokens of a missing from b, and of b missing from a."""
    n = len(a)
    ms, xs, mm, xm = (np.zeros(n, np.float32) for _ in range(4))
    for i, (x, y, c) in enumerate(zip(a, b, ctry)):
        sx, sy = set(x.split()), set(y.split())
        miss = [idf.get((c, t), unk) for t in sx - sy]
        extra = [idf.get((c, t), unk) for t in sy - sx]
        if miss:
            ms[i] = sum(miss); mm[i] = max(miss)
        if extra:
            xs[i] = sum(extra); xm[i] = max(extra)
    return ms, xs, mm, xm


def _num_fuzzy(a, b):
    n = len(a)
    fl = np.full(n, -1, np.float32)
    fp = np.full(n, -1, np.int8)
    bl = np.full(n, -1, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        lx, ly = x.split(), y.split()
        fl[i] = Levenshtein.normalized_similarity(lx[0], ly[0])
        fp[i] = int(lx[0].startswith(ly[0]) or ly[0].startswith(lx[0]))
        bl[i] = max(Levenshtein.normalized_similarity(u, v) for u in lx[:4] for v in ly[:4])
    return fl, fp, bl


def _strip_digits(arr):
    return np.array([" ".join(t for t in x.split() if not any(ch.isdigit() for ch in t)) for x in arr], dtype=object)


def extra_features(cand, recs, name_idf, addr_idf):
    """name_idf/addr_idf: (idf dict, unknown-token idf) from token_idf over Source 1."""
    i1, io = cand.i1.values, cand.io.values
    c1, c2 = _col(recs, "core", i1), _col(recs, "core", io)
    a1, a2 = _col(recs, "addr_c", i1), _col(recs, "addr_c", io)
    m1, m2 = _col(recs, "nums", i1), _col(recs, "nums", io)
    f = pd.DataFrame(index=cand.index)
    f["num_first_lev"], f["num_first_prefix"], f["num_best_lev"] = _num_fuzzy(m1, m2)
    s1, s2 = _strip_digits(a1), _strip_digits(a2)
    f["a_street_tset"] = _cp(s1, s2, fuzz.token_set_ratio)
    f["a_street_ratio"] = _cp(s1, s2, fuzz.token_sort_ratio)
    ctry = _col(recs, "country", i1)
    f["n_miss_idf1"], f["n_extra_idf2"], f["n_miss_max1"], f["n_extra_max2"] = _idf_diff(c1, c2, *name_idf, ctry)
    f["a_miss_idf1"], f["a_extra_idf2"], f["a_miss_max1"], f["a_extra_max2"] = _idf_diff(s1, s2, *addr_idf, ctry)
    f["n_diff_cnt"] = np.array([len(set(x.split()) ^ set(y.split())) for x, y in zip(c1, c2)], np.int8)
    return f


# ---------------------------------------------------------------------------------------
# Corroboration features (v3 block, feat3_*). Hard negatives behave like *sibling*
# businesses (same name, neighbouring house number / changed name word) whose several
# noisy copies CONSISTENTLY carry the sibling's values, while typos in true copies are
# random per copy. For a pair (A, r) we therefore count how many other records support
# r's value versus A's value -- inside A's candidate pool and over the whole corpus.
# ---------------------------------------------------------------------------------------
SUPPORT_COLS = []


def record_keys(recs):
    """Per-record int64 hash keys: first house number, sorted core name, number+street."""
    nums = recs.nums.to_numpy(dtype=object)
    core = recs.core.to_numpy(dtype=object)
    addr = recs.addr_c.to_numpy(dtype=object)
    ctry = recs.country.to_numpy(dtype=object)
    first = np.array([x.split(" ", 1)[0] if x else "" for x in nums], dtype=object)
    name = np.array([" ".join(sorted(set(x.split()))) for x in core], dtype=object)
    street = np.array([next((t for t in a.split() if not any(ch.isdigit() for ch in t)), "") for a in addr],
                      dtype=object)
    h = lambda arr: pd.util.hash_array(arr, categorize=True).astype(np.int64)
    num_k = np.where(first == "", 0, h(ctry + "|" + first))
    name_k = np.where(name == "", 0, h(ctry + "|" + name))
    ns_k = np.where((first == "") | (street == ""), 0, h(ctry + "|" + first + "|" + street))
    return {"num": num_k, "name": name_k, "ns": ns_k}


def support_features(cand, recs):
    keys = record_keys(recs)
    i1, io = cand.i1.values, cand.io.values
    other = recs.src.to_numpy() != 1
    f = pd.DataFrame(index=cand.index)
    for nm, k in keys.items():
        kr, ka = k[io], k[i1]
        valid = (kr != 0) & (ka != 0)
        f[f"sup_{nm}_eq"] = np.where(valid, (kr == ka).astype(np.int8), -1).astype(np.int8)
        # within A's candidate pool
        d = pd.DataFrame({"i1": i1, "k": kr})
        cnt = d.groupby(["i1", "k"], sort=False).k.transform("size").values
        pool_r = np.where(kr != 0, cnt - 1, -1)
        tab = d[kr != 0].groupby(["i1", "k"]).size()
        pool_a = pd.Series(tab.reindex(pd.MultiIndex.from_arrays([i1, ka])).values).fillna(0).values
        # as fractions of the S1's pool size (test pools are larger than train pools)
        psize = pd.Series(i1).groupby(i1).transform("size").values.astype(np.float32)
        f[f"sup_{nm}_pool_r"] = np.where(kr != 0, pool_r / psize, -1).astype(np.float32)
        f[f"sup_{nm}_pool_a"] = np.where(ka != 0, pool_a / psize, -1).astype(np.float32)
        # over all S2/S3 records of the split
        vc = pd.Series(k[other & (k != 0)]).value_counts()
        g_r = pd.Series(kr).map(vc).fillna(0).values
        g_a = pd.Series(ka).map(vc).fillna(0).values
        f[f"sup_{nm}_glob_r"] = np.where(kr != 0, g_r, -1).astype(np.float32)
        f[f"sup_{nm}_glob_a"] = np.where(ka != 0, g_a, -1).astype(np.float32)
    return f

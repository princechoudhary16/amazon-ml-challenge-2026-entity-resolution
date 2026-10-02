"""Candidate generation (blocking).

Scalable, country-partitioned sparse retrieval:
  * every record -> bag of blocking tokens: name-core unigrams + bigrams, the
    concatenated core name (catches "acmetech.com" vs "Acme Tech"), address unigrams +
    bigrams (house-number + street bigrams are highly selective);
  * name-token x address-token "cross" tokens (e.g. classic|surat): many businesses
    share a name and differ only by locality, and locality words alone are too common
    to index; the cross tokens are rare, cheap and disambiguating;
  * phonetic consonant-skeleton name tokens (apex/epeks -> pks) for transliteration noise;
  * tokens are feature-hashed (2^26 buckets); IDF is fit on Source 1 of that country and
    buckets whose Source-1 document frequency exceeds `max_df` are dropped (little
    identity information, dominate compute), which keeps retrieval ~linear;
  * top-k cosine retrieval in BOTH directions with multithreaded sparse top-n matmul:
      - reverse: each S2/S3 record retrieves its top-`k_rev` Source-1 entities
      - forward: each S1 entity retrieves its top-`k_fwd` records of each source
  * the candidate set is the union; retrieval score and both ranks are kept as features.
Records are only compared within the same `country` label (open set; any label works).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Pool

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .normalize import ADDR_STOP, phon

N_THREADS = 10


# blocking token recipe (tuned on training recall; see work/ experiments)
CFG = {"raw_name": True, "phon_name": True, "addr_bigrams": True, "cross": True}


def block_tokens(core, addr_c, cfg=None):
    """Build the whitespace-joined blocking token string for one record."""
    cfg = cfg or CFG
    ct = core.split()
    toks = []
    if cfg["raw_name"]:
        toks += ["n:" + t for t in ct]
        toks += ["nb:" + a + "_" + b for a, b in zip(ct, ct[1:])]
        if ct:
            toks.append("c:" + "".join(ct))
    if cfg["phon_name"]:
        pt = [phon(t) for t in ct]
        toks += ["p:" + t for t in pt]
        toks += ["pb:" + a + "_" + b for a, b in zip(pt, pt[1:])]
        if pt:
            toks.append("pc:" + "".join(pt))
    at = [t for t in addr_c.split() if t not in ADDR_STOP and (len(t) > 1 or t.isdigit())]
    toks += ["a:" + t for t in at]
    if cfg["addr_bigrams"]:
        toks += ["ab:" + a + "_" + b for a, b in zip(at, at[1:])]
    if cfg["cross"]:
        # name-token x address-token: "same name word in the same locality". Keeps
        # disambiguating city/locality evidence that is too common to index alone.
        pn = sorted(set(phon(t) for t in ct if len(t) > 1))
        an = sorted(set(t for t in at if not t.isdigit()))
        toks += ["x:" + a + "|" + b for a in pn for b in an]
    return " ".join(toks)


_HASH_BITS = 26


def _hash_chunk(args):
    """Worker: build blocking tokens for a chunk of records, hash them (binary) and, if
    an IDF vector is given, weight + prune + L2-normalise in the worker so only the
    small surviving matrix is sent back (memory)."""
    cores, addrs, cfg, idf = args
    hv = HashingVectorizer(token_pattern=r"\S+", lowercase=False, n_features=2 ** _HASH_BITS,
                           alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    x = hv.transform([block_tokens(c, a, cfg) for c, a in zip(cores, addrs)])
    if idf is not None:
        x = _weight(x, idf)
    return x


def _hash_matrix(cores, addrs, idf=None, workers=6, chunk=50_000):
    """Token strings are built inside the workers so the full corpus of token strings
    never materialises in the parent process (memory)."""
    jobs = ((cores[i:i + chunk], addrs[i:i + chunk], CFG, idf) for i in range(0, len(cores), chunk))
    with Pool(workers, initializer=_set_idf, initargs=(idf,)) as pool:
        mats = pool.map(_hash_chunk, ((c, a, g, None) for c, a, g, _ in jobs), chunksize=1) \
            if idf is None else pool.map(_hash_chunk_idf, ((c, a, g) for c, a, g, _ in jobs), chunksize=1)
    return sp.vstack(mats, format="csr")


_IDF = None


def _set_idf(idf):
    global _IDF
    _IDF = idf


def _hash_chunk_idf(args):
    cores, addrs, cfg = args
    return _hash_chunk((cores, addrs, cfg, _IDF))


def _weight(x, idf):
    """In-place IDF weighting + L2 normalisation; drops zero-weight entries."""
    x.data *= idf[x.indices]
    x.eliminate_zeros()
    return normalize(x, norm="l2", copy=False)


def _vectorize(core, addr, i1, io, max_df):
    """Hashed binary token vectors -> IDF (fit on Source 1) -> drop buckets with S1 df
    above max_df (or absent from S1) -> L2 normalise."""
    x1 = _hash_matrix(core[i1].tolist(), addr[i1].tolist())
    df = np.bincount(x1.indices, minlength=x1.shape[1])
    idf = (np.log((1 + x1.shape[0]) / (1 + df)) + 1.0).astype(np.float32)
    idf[(df > max_df) | (df == 0)] = 0.0
    del df
    x1 = _weight(x1, idf)
    xo = _hash_matrix(core[io].tolist(), addr[io].tolist(), idf=idf)
    return x1, xo


def _i32(m):
    """CSR with contiguous int32 indices/indptr and float32 data: sparse_dot_topn's C++
    kernel segfaults on mixed index dtypes (scipy may upcast indptr to int64)."""
    m = m.tocsr()
    m.sort_indices()
    return sp.csr_matrix((np.ascontiguousarray(m.data, dtype=np.float32),
                          np.ascontiguousarray(m.indices, dtype=np.int32),
                          np.ascontiguousarray(m.indptr, dtype=np.int32)), shape=m.shape)


def _topn(a, b_t, k, chunk=400_000):
    """Row-wise top-k of a @ b_t (b_t already transposed, CSR). Returns (row, col, val, rank)."""
    rows, cols, vals, ranks = [], [], [], []
    b_t = _i32(b_t)
    for s in range(0, a.shape[0], chunk):
        m = sp_matmul_topn(_i32(a[s:s + chunk]), b_t, top_n=k, threshold=1e-4, sort=True, n_threads=N_THREADS)
        m = m.tocsr()
        cnt = np.diff(m.indptr)
        r = np.repeat(np.arange(m.shape[0], dtype=np.int64) + s, cnt)
        rk = np.arange(m.nnz) - np.repeat(m.indptr[:-1], cnt)
        rows.append(r); cols.append(m.indices.astype(np.int64)); vals.append(m.data); ranks.append(rk)
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32),
            np.concatenate(ranks).astype(np.int16))


def generate_candidates(recs, k_rev=3, k_fwd=5, max_df=2000, log=print):
    """recs: normalised records (columns entity_id, src, country, core, addr_c).

    Returns DataFrame of pairs with integer positions into `recs`:
      i1 (S1 row), io (S2/S3 row), bscore, rank_rev (rank of S1 for the record),
      rank_fwd (rank of record among that source's results for the S1); -1 == not retrieved.
    """
    import time
    t0 = time.time()
    core = recs.core.to_numpy(dtype=object)
    addr = recs.addr_c.to_numpy(dtype=object)
    country_arr = recs.country.to_numpy(dtype=object)
    src_arr = recs.src.to_numpy()
    out = []
    for country in pd.unique(country_arr):
        i1 = np.flatnonzero((country_arr == country) & (src_arr == 1))
        io = np.flatnonzero((country_arr == country) & (src_arr != 1))
        if len(i1) == 0 or len(io) == 0:
            continue
        x1, xo = _vectorize(core, addr, i1, io, int(max_df))
        log(f"  {country}: vectorised ({time.time() - t0:.0f}s)")
        # reverse: each other record -> top k_rev S1
        r, c, v, rk = _topn(xo, x1.T.tocsr(), k_rev)
        rev = pd.DataFrame({"i1": i1[c], "io": io[r], "s_rev": v, "rank_rev": rk})
        del r, c, v, rk
        log(f"  {country}: reverse done ({time.time() - t0:.0f}s)")
        # forward: each S1 -> top k_fwd per source
        fw = []
        for s in (2, 3):
            ios = np.flatnonzero(src_arr[io] == s)
            r, c, v, rk = _topn(x1, xo[ios].T.tocsr(), k_fwd)
            fw.append(pd.DataFrame({"i1": i1[r], "io": io[ios[c]], "s_fwd": v, "rank_fwd": rk}))
        fwd = pd.concat(fw, ignore_index=True)
        del x1, xo
        log(f"  {country}: forward done ({time.time() - t0:.0f}s)")
        m = rev.merge(fwd, on=["i1", "io"], how="outer")
        m["bscore"] = m.s_rev.fillna(m.s_fwd).astype(np.float32)
        m["rank_rev"] = m.rank_rev.fillna(-1).astype(np.int16)
        m["rank_fwd"] = m.rank_fwd.fillna(-1).astype(np.int16)
        m = m.drop(columns=["s_rev", "s_fwd"])
        log(f"  blocking {country}: S1={len(i1)} other={len(io)} pairs={len(m)} "
            f"({len(m) / len(i1):.1f}/S1)")
        out.append(m)
    return pd.concat(out, ignore_index=True)

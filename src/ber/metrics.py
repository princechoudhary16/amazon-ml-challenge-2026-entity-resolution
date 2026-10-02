"""Competition metric: macro-averaged per-Source-1 F_0.5 (singletons included)."""
import numpy as np


def f05(pred, true):
    """F_0.5 for one Source-1 entity. pred/true: sets of matched ids."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth, ids=None):
    """pred/truth: dict s1_id -> set. ids: evaluation S1 ids (default: truth keys)."""
    ids = list(truth) if ids is None else ids
    return float(np.mean([f05(pred.get(i, set()), truth.get(i, set())) for i in ids]))


def truth_from_pairs(pairs, s1_ids):
    """pairs: DataFrame[source1_entity_id, ids] -> dict covering every id in s1_ids."""
    t = {i: set() for i in s1_ids}
    for a, b in zip(pairs.source1_entity_id.values, pairs.ids.values):
        if a in t:
            t[a].add(b)
    return t


def fast_macro_f05(i1, y, keep, n_true, s1_rows):
    """Vectorised macro F0.5 over pairs (identical to macro_f05 for S1s in s1_rows).

    i1: S1 row index per pair; y: 1 if pair is a true match; keep: predicted mask;
    n_true: array (len = max row + 1) with the number of true matches per S1 row
    (including matches outside the candidate set); s1_rows: S1 rows to average over.
    For one S1: F0.5 = 1.25*TP / (k + 0.25*T); T=0 -> 1 if k=0 else 0; k=0,T>0 -> 0."""
    m = len(n_true)
    k = np.bincount(i1[keep], minlength=m)[s1_rows]
    tp = np.bincount(i1[keep & (y == 1)], minlength=m)[s1_rows]
    t = n_true[s1_rows]
    f = np.where(t == 0, (k == 0).astype(float),
                 np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * t, 1e-9)))
    return float(f.mean())

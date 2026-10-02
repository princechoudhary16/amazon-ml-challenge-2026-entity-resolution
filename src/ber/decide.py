"""Decision policy: turn pair probabilities into per-S1 match lists.

Structural fact from training data: every S2/S3 record matches at most ONE Source-1
entity. So each candidate record is first assigned to its single most probable S1
("exclusive assignment"); the link is kept only if that probability clears a threshold
tuned on out-of-fold macro F_0.5.
"""
import numpy as np
import pandas as pd

from .metrics import fast_macro_f05, macro_f05


def assign(i1, io, p, thr, exclusive=True, margin=0.0):
    """Return boolean mask of kept pairs."""
    keep = p >= thr
    if exclusive:
        d = pd.DataFrame({"io": io, "p": p})
        best = d.groupby("io", sort=False).p.transform("max").values
        keep &= p >= best
        if margin > 0:
            # require a margin over the best competing S1 for the same record
            second = d.assign(r=d.groupby("io", sort=False).p.rank(ascending=False, method="first"))
            sec = second[second.r == 2].set_index("io").p
            comp = pd.Series(io).map(sec).fillna(0).values
            keep &= (p - comp) >= margin
    return keep


def to_lists(i1, io, keep, eid):
    """dict S1 entity_id -> set of matched entity_ids."""
    out = {}
    for a, b in zip(eid[i1[keep]], eid[io[keep]]):
        out.setdefault(a, set()).add(b)
    return out


def tune(i1, io, p, y, n_true, s1_rows, grid=None):
    """Grid-search (threshold, margin) of the exclusive-assignment policy on OOF macro F0.5."""
    grid = grid if grid is not None else np.round(np.arange(0.2, 0.96, 0.025), 3)
    d = pd.DataFrame({"io": io, "p": p})
    best = d.groupby("io", sort=False).p.transform("max").values
    srt = d.sort_values(["io", "p"], ascending=[True, False])
    sec = srt.groupby("io", sort=False).p.nth(1)
    second = pd.Series(sec.values, index=srt.loc[sec.index, "io"].values)
    comp = pd.Series(io).map(second).fillna(0).values
    rows = []
    for m in (0.0, 0.05, 0.1, 0.2, 0.3):
        base = (p >= best) & ((p - comp) >= m)
        for t in grid:
            keep = base & (p >= t)
            rows.append({"thr": float(t), "margin": m, "f05": fast_macro_f05(i1, y, keep, n_true, s1_rows)})
    tab = pd.DataFrame(rows).sort_values("f05", ascending=False)
    return tab.iloc[0].to_dict(), tab


def expected_f_select(i1, io, p, min_p=0.05, t_scale=1.0):
    """Per-S1 expected-F0.5-optimal subset selection.

    For one S1 with k predicted links: F0.5 = 1.25*TP / (k + 0.25*T) (T = true matches).
    Candidates are first restricted to those for which this S1 is the record's most
    probable S1 (exclusive assignment). Sorting them by p, for every k we estimate
    E[F] ~= 1.25 * sum(p_top_k) / (k + 0.25 * E[T]), with E[T] = t_scale * sum of the
    exclusive-winner probabilities; k = 0 scores P(no match) = prod(1 - p). The k with
    the highest expected score is chosen. Returns boolean mask of kept pairs."""
    d = pd.DataFrame({"i1": i1, "io": io, "p": p})
    best = d.groupby("io", sort=False).p.transform("max").values
    win = (p >= best) & (p >= min_p)
    w = d[win].sort_values(["i1", "p"], ascending=[True, False])
    g = w.groupby("i1", sort=False)
    k = g.cumcount().values + 1
    cum = g.p.cumsum().values
    et = g.p.transform("sum").values * t_scale
    ef = 1.25 * cum / (k + 0.25 * et)
    lq = pd.Series(np.log1p(-np.clip(w.p.values, 0, 0.999999)), index=w.index)
    p0 = np.exp(lq.groupby(w.i1.values).transform("sum").values)
    w = w.assign(ef=ef, p0=p0, k=k)
    bestk = w.groupby("i1", sort=False).ef.transform("max").values
    # choose largest-EF k; keep rows up to that k, unless empty set is better
    kstar = w.assign(isbest=w.ef.values >= bestk).query("isbest").groupby("i1").k.min()
    w["kstar"] = w.i1.map(kstar).values
    keep_rows = (w.k.values <= w.kstar.values) & (bestk > w.p0.values)
    mask = np.zeros(len(p), bool)
    mask[w.index.values[keep_rows]] = True
    return mask


def tune_ef(i1, io, p, y, n_true, s1_rows):
    rows = []
    for ts in (0.8, 1.0, 1.2, 1.5):
        for mp in (0.05, 0.2, 0.35, 0.5):
            keep = expected_f_select(i1, io, p, min_p=mp, t_scale=ts)
            rows.append({"t_scale": ts, "min_p": mp, "f05": fast_macro_f05(i1, y, keep, n_true, s1_rows)})
    tab = pd.DataFrame(rows).sort_values("f05", ascending=False)
    return tab.iloc[0].to_dict(), tab

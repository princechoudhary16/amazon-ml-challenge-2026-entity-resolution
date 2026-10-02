"""Country-holdout validation: a proxy for generalisation to a country absent from training
(the test set contains France, which has no training examples).

Train on one training country only, evaluate macro F0.5 on the other (never seen), for two
feature sets: the base block only vs. base + extra (feat2) + corroboration (feat3) blocks.
A feature set whose held-out-country score collapses relative to its in-distribution score
is overfit to country-specific patterns.

Usage: python -m ber.holdout --art <artifacts> --train-country US --eval-country India
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from . import decide, train
from .metrics import fast_macro_f05


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_rows(art, mask, blocks):
    out, off = [], 0
    for p in sorted(glob.glob(os.path.join(art, "feat_train", "part-*.parquet"))):
        f = pd.read_parquet(p)
        m = mask[off:off + len(f)]
        off += len(f)
        if not m.any():
            continue
        f = f[m]
        for blk in blocks:
            p2 = p.replace("feat_train", f"{blk}_train")
            f = pd.concat([f, pd.read_parquet(p2)[m].set_index(f.index)], axis=1)
        out.append(f)
    X = pd.concat(out, ignore_index=True)
    return X[[c for c in X.columns if "_glob_" not in c]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--train-country", default="US")
    ap.add_argument("--eval-country", default="India")
    ap.add_argument("--train-frac", type=float, default=0.3)
    args = ap.parse_args()
    art = args.art
    recs = pd.read_parquet(os.path.join(art, "train_norm.parquet"), columns=["entity_id", "src", "country"])
    eid = recs.entity_id.to_numpy(dtype=object)
    ctry = recs.country.to_numpy(dtype=object)
    pairs = pd.read_parquet(os.path.join(art, "train_pairs.parquet"))
    cand = pd.read_parquet(os.path.join(art, "train_cand.parquet"), columns=["i1", "io"])
    y = train.label_pairs(cand, eid, pairs)
    c = ctry[cand.i1.values]
    h = train.hash_frac(eid[cand.i1.values], 2)
    tr_m = (c == args.train_country) & (h < args.train_frac)
    es_m = (c == args.train_country) & (h >= 0.9)
    ev_m = c == args.eval_country
    pos = pd.Series(np.arange(len(eid)), index=eid)
    n_true = np.bincount(pos[pairs.source1_entity_id.values].values, minlength=len(eid))
    s1_rows = np.flatnonzero((recs.src.values == 1) & (ctry == args.eval_country))
    i1, io = cand.i1.values[ev_m], cand.io.values[ev_m]
    for name, blocks in (("base", []), ("base+feat2+feat3", ["feat2", "feat3"])):
        X = load_rows(art, tr_m, blocks)
        Xe = load_rows(art, es_m, blocks)
        b = train.fit(X.values, y[tr_m], Xe.values, y[es_m])
        del X, Xe
        Xv = load_rows(art, ev_m, blocks)
        p = b.predict(Xv.values, num_threads=10)
        del Xv
        # fixed policy (as chosen in-distribution) and best-possible policy on the held-out country
        fixed = fast_macro_f05(cand.i1.values[ev_m], y[ev_m],
                               decide.assign(i1, io, p, 0.65, margin=0.3), n_true, s1_rows)
        best, _ = decide.tune(i1, io, p, y[ev_m], n_true, s1_rows)
        rate = decide.assign(i1, io, p, 0.65, margin=0.3).sum() / len(s1_rows)
        log(f"[{name}] train={args.train_country} -> eval={args.eval_country}: "
            f"F0.5@0.65/0.3={fixed:.4f}  best={best['f05']:.4f} (thr={best['thr']}, m={best['margin']})  "
            f"links/S1={rate:.3f}  iters={b.best_iteration}")


if __name__ == "__main__":
    main()

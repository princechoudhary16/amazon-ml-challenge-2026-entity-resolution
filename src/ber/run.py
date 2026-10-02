"""End-to-end pipeline (resumable stages; every stage caches its output in --art).

  block   : candidate generation for train and test           -> {split}_cand.parquet
  ctx     : cheap similarities + group-context features        -> {split}_ctx.parquet
  feat    : full pairwise features, written in chunks          -> feat_{split}/part-*.parquet
  train   : S1-grouped 3-fold LightGBM, OOF predictions,
            decision-policy tuning on OOF macro F0.5           -> model_fold*.txt, oof.npy, config.json
  predict : average of fold models on test, decision policy,
            output/matching_results.tsv + candidate_pairs.tsv

Usage (after `python -m ber.prep ...`):
  python -m ber.run --art <artifacts> --out <output dir>

The defaults reproduce the FINAL submission. Experimental (not used in the final
submission, see Documentation section 5.2): stages feat2 / feat3, --blocks feat2,feat3,
--admin-strip 1, --lr / --rounds / --train-frac variations.
"""
import argparse
import glob
import json
import os
import time

import subprocess
import sys

import numpy as np
import pandas as pd

from . import decide
from .metrics import fast_macro_f05, macro_f05, truth_from_pairs

# NOTE: blocking (sparse_dot_topn) and LightGBM bundle different OpenMP runtimes that
# segfault when loaded into the same process, so each stage runs in its own subprocess
# and heavy modules are imported inside the stage that needs them.

REC_COLS = ["entity_id", "src", "country", "name_c", "core", "legal", "web", "addr_c", "nums"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_recs(art, split, cols=REC_COLS):
    return pd.read_parquet(os.path.join(art, f"{split}_norm.parquet"), columns=cols)


ADMIN_DF = 0.02
ADMIN_STRIP = False           # --admin-strip 1: experimental admin-token stripping (not in final)
BLOCKS = ()                   # --blocks feat2,feat3: experimental feature blocks (not in final)


def admin_strip(recs):
    """Remove admin-level address tokens before any address comparison.

    A token is 'admin-level' if it occurs in more than ADMIN_DF of that country's records
    (all sources): region / departement / state names, big-city names, street-type words.
    Sources disagree on these (e.g. French S1 gives the region, S2/S3 the departement), so
    comparing them injects noise that does not transfer across countries. The rule is
    unsupervised, relative to each country's own size, and applies to any country label."""
    from collections import Counter
    addr = recs.addr_c.to_numpy(dtype=object)
    ctry = recs.country.to_numpy(dtype=object)
    out = addr.copy()
    for c in pd.unique(ctry):
        idx = np.flatnonzero(ctry == c)
        cnt = Counter()
        for a in addr[idx]:
            cnt.update(set(a.split()))
        lim = ADMIN_DF * len(idx)
        admin = {t for t, v in cnt.items() if v > lim and not t.isdigit()}
        log(f"  admin tokens {c}: {len(admin)} e.g. {sorted(admin, key=lambda t: -cnt[t])[:12]}")
        out[idx] = [" ".join(t for t in a.split() if t not in admin) for a in addr[idx]]
    return out


def load_feat_recs(art, split, cols):
    """Records for feature computation, with admin-level address tokens stripped (cached)."""
    if not ADMIN_STRIP:
        return load_recs(art, split, cols)
    path = os.path.join(art, f"{split}_addrf.parquet")
    if not os.path.exists(path):
        r = load_recs(art, split, ["country", "addr_c"])
        pd.DataFrame({"addr_f": admin_strip(r)}).to_parquet(path)
    recs = load_recs(art, split, cols)
    if "addr_c" in cols:
        recs["addr_c"] = pd.read_parquet(path).addr_f.values
    return recs


def stage_block(art, split, args):
    path = os.path.join(art, f"{split}_cand.parquet")
    if os.path.exists(path):
        return
    from . import blocking
    recs = load_recs(art, split, ["entity_id", "src", "country", "core", "addr_c"])
    cand = blocking.generate_candidates(recs, k_rev=args.k_rev, k_fwd=args.k_fwd, max_df=args.max_df, log=log)
    cand = cand.sort_values(["i1", "io"]).reset_index(drop=True)
    cand.to_parquet(path)
    n1 = int((recs.src == 1).sum())
    log(f"{split}: {len(cand)} candidate pairs ({len(cand) / n1:.2f} per S1)")


def stage_ctx(art, split):
    path = os.path.join(art, f"{split}_ctx.parquet")
    if os.path.exists(path):
        return
    from . import features
    recs = load_feat_recs(art, split, ["core", "addr_c"])
    cand = pd.read_parquet(os.path.join(art, f"{split}_cand.parquet"), columns=["i1", "io"])
    base = features.base_features(cand, recs)
    ctx = features.context_features(base, cand)
    ctx.to_parquet(path)
    log(f"{split}: context features {ctx.shape}")


def stage_feat(art, split, chunk=1_000_000):
    from . import features
    out_dir = os.path.join(art, f"feat_{split}")
    os.makedirs(out_dir, exist_ok=True)
    recs = load_feat_recs(art, split, REC_COLS)
    cand = pd.read_parquet(os.path.join(art, f"{split}_cand.parquet"))
    ctx = pd.read_parquet(os.path.join(art, f"{split}_ctx.parquet"))
    n = (len(cand) + chunk - 1) // chunk
    for k in range(n):
        path = os.path.join(out_dir, f"part-{k:04d}.parquet")
        if os.path.exists(path):
            continue
        sl = slice(k * chunk, (k + 1) * chunk)
        f = features.pair_features(cand.iloc[sl], recs, ctx.iloc[sl])
        f.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)  # atomic: a crash never leaves a half-written part
        log(f"{split}: feature part {k + 1}/{n}")


def stage_feat2(art, split, chunk=1_000_000):
    """Extra feature block (v2) written alongside the base feature parts."""
    from . import features
    out_dir = os.path.join(art, f"feat2_{split}")
    os.makedirs(out_dir, exist_ok=True)
    recs = load_feat_recs(art, split, REC_COLS)
    # IDF over ALL records (S1+S2+S3) of each country, relative to the country's size
    # (scale-invariant; see features.token_idf).
    ctry = recs.country.to_numpy(dtype=object)
    name_idf = features.token_idf(recs.core.to_numpy(dtype=object), ctry)
    addr_idf = features.token_idf(features._strip_digits(recs.addr_c.to_numpy(dtype=object)), ctry)
    cand = pd.read_parquet(os.path.join(art, f"{split}_cand.parquet"), columns=["i1", "io"])
    n = (len(cand) + chunk - 1) // chunk
    for k in range(n):
        path = os.path.join(out_dir, f"part-{k:04d}.parquet")
        if os.path.exists(path):
            continue
        f = features.extra_features(cand.iloc[k * chunk:(k + 1) * chunk], recs, name_idf, addr_idf)
        f.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        log(f"{split}: extra feature part {k + 1}/{n}")


def stage_feat3(art, split, chunk=1_000_000):
    """Corroboration (support) features over the full candidate graph, saved in parts
    aligned with the other feature blocks."""
    from . import features
    out_dir = os.path.join(art, f"feat3_{split}")
    if os.path.exists(os.path.join(out_dir, "_DONE")):
        return
    os.makedirs(out_dir, exist_ok=True)
    # admin-stripped address, so the street key is the street name ("kervegan"), not "rue"
    recs = load_feat_recs(art, split, ["src", "country", "nums", "core", "addr_c"])
    cand = pd.read_parquet(os.path.join(art, f"{split}_cand.parquet"), columns=["i1", "io"])
    f = features.support_features(cand, recs)
    n = (len(cand) + chunk - 1) // chunk
    for k in range(n):
        f.iloc[k * chunk:(k + 1) * chunk].reset_index(drop=True).to_parquet(
            os.path.join(out_dir, f"part-{k:04d}.parquet"))
    open(os.path.join(out_dir, "_DONE"), "w").close()
    log(f"{split}: support features {f.shape}")


def iter_parts(art, split):
    """Yield feature chunks; the v2 extra block is joined column-wise when present."""
    base = sorted(glob.glob(os.path.join(art, f"feat_{split}", "part-*.parquet")))
    for p in base:
        f = pd.read_parquet(p)
        for blk in BLOCKS:
            p2 = p.replace(f"feat_{split}", f"{blk}_{split}")
            if os.path.exists(p2):
                f = pd.concat([f, pd.read_parquet(p2).set_index(f.index)], axis=1)
        yield f


def stage_train(art, args):
    from . import train
    train.PARAMS["learning_rate"] = args.lr
    recs = load_recs(art, "train", ["entity_id", "src"])
    pairs = pd.read_parquet(os.path.join(art, "train_pairs.parquet"))
    cand = pd.read_parquet(os.path.join(art, "train_cand.parquet"), columns=["i1", "io"])
    eid = recs.entity_id.to_numpy(dtype=object)
    y = train.label_pairs(cand, eid, pairs)
    log(f"train pairs={len(y)} positives={int(y.sum())} pair-recall={y.sum() / len(pairs):.4f}")
    s1_ids = eid[cand.i1.values]
    fold = (train.hash_frac(s1_ids, 1) * args.folds).astype(np.int8)
    samp = train.hash_frac(s1_ids, 2) < args.train_frac
    del s1_ids

    # training sample (all pairs of a random subset of S1 entities)
    Xs, off = [], 0
    for f in iter_parts(art, "train"):
        m = samp[off:off + len(f)]
        Xs.append(f[m])
        off += len(f)
    Xs = pd.concat(Xs, ignore_index=True)
    # corpus-wide support counts scale with each country's corpus size / numbering
    # conventions and do not transfer to unseen countries -> excluded (pool-relative
    # support features are kept)
    cols = [c for c in Xs.columns if "_glob_" not in c]
    Xs = Xs[cols]
    ys, fs = y[samp], fold[samp]
    log(f"training sample: {Xs.shape}, positives={int(ys.sum())}")

    models = []
    for k in range(args.folds):
        tr, va = fs != k, fs == k
        b = train.fit(Xs.values[tr], ys[tr], Xs.values[va], ys[va], rounds=args.rounds)
        b.save_model(os.path.join(art, f"model_fold{k}.txt"))
        models.append(b)
        log(f"fold {k}: best_iter={b.best_iteration}")
        if k == 0:
            imp = pd.Series(b.feature_importance("gain"), index=cols).sort_values(ascending=False)
            imp.to_csv(os.path.join(art, "feature_importance.tsv"), sep="\t")
    del Xs

    # OOF prediction for EVERY training pair (fold model that never saw that S1)
    oof = np.empty(len(y), np.float32)
    off = 0
    for f in iter_parts(art, "train"):
        fk = fold[off:off + len(f)]
        for k, b in enumerate(models):
            m = fk == k
            if m.any():
                oof[off:off + len(f)][m] = b.predict(f[cols].values[m], num_threads=10)
        off += len(f)
    np.save(os.path.join(art, "oof.npy"), oof)

    s1_all = eid[recs.src.values == 1]
    s1_rows = np.flatnonzero(recs.src.values == 1)
    pos = pd.Series(np.arange(len(eid)), index=eid)
    n_true = np.bincount(pos[pairs.source1_entity_id.values].values, minlength=len(eid))
    i1, io = cand.i1.values, cand.io.values
    ceiling = fast_macro_f05(i1, y, y.astype(bool), n_true, s1_rows)
    # sanity: vectorised metric == reference implementation on a subsample
    chk = s1_rows[:20000]
    keep0 = oof >= 0.5
    ref = macro_f05(decide.to_lists(i1, io, keep0 & np.isin(i1, chk), eid),
                    truth_from_pairs(pairs, eid[chk]), list(eid[chk]))
    assert abs(ref - fast_macro_f05(i1, y, keep0, n_true, chk)) < 1e-9, "metric mismatch"
    best, tab = decide.tune(i1, io, oof, y, n_true, s1_rows)
    tab.to_csv(os.path.join(art, "threshold_table.tsv"), sep="\t", index=False)
    best["policy"] = "threshold"
    best_ef, tab_ef = decide.tune_ef(i1, io, oof, y, n_true, s1_rows)
    tab_ef.to_csv(os.path.join(art, "expected_f_table.tsv"), sep="\t", index=False)
    log(f"OOF macro F0.5 threshold-policy={best} | expected-F policy={best_ef} | "
        f"ceiling with perfect matcher on candidates={ceiling:.4f}")
    if best_ef["f05"] > best["f05"]:
        best = dict(best_ef, policy="expected_f")
    json.dump({"best": best, "ceiling": ceiling, "features": cols, "args": vars(args)},
              open(os.path.join(art, "config.json"), "w"), indent=1)


def write_outputs(cand, keep, recs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eid = recs.entity_id.to_numpy(dtype=object)
    s1 = pd.Series(eid[recs.src.values == 1])
    a, b = eid[cand.i1.values], eid[cand.io.values]
    cands = pd.Series(b).groupby(a).agg(",".join)
    match = pd.Series(b[keep]).groupby(a[keep]).agg(",".join)
    pd.DataFrame({"source1_entity_id": s1, "matched_entity_ids": s1.map(match).fillna("").values}) \
        .to_csv(os.path.join(out_dir, "matching_results.tsv"), sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": s1, "candidate_entity_ids": s1.map(cands).fillna("").values}) \
        .to_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", index=False)


def stage_predict(art, out_dir, args):
    import lightgbm as lgb
    cfg = json.load(open(os.path.join(art, "config.json")))
    models = [lgb.Booster(model_file=p) for p in sorted(glob.glob(os.path.join(art, "model_fold*.txt")))]
    ps = []
    for f in iter_parts(art, "test"):
        X = f[cfg["features"]].values
        ps.append(np.mean([b.predict(X, num_threads=10) for b in models], axis=0).astype(np.float32))
    p = np.concatenate(ps)
    np.save(os.path.join(art, "test_p.npy"), p)
    recs = load_recs(art, "test", ["entity_id", "src"])
    cand = pd.read_parquet(os.path.join(art, "test_cand.parquet"), columns=["i1", "io"])
    assert len(cand) == len(p)
    best = cfg["best"]
    if best["policy"] == "expected_f":
        keep = decide.expected_f_select(cand.i1.values, cand.io.values, p,
                                        min_p=best["min_p"], t_scale=best["t_scale"])
    else:
        keep = decide.assign(cand.i1.values, cand.io.values, p, best["thr"], margin=best["margin"])
    write_outputs(cand, keep, recs, out_dir)
    n1 = int((recs.src == 1).sum())
    log(f"test: {int(keep.sum())} matches; {len(cand) / n1:.2f} candidates/S1; "
        f"{(pd.Series(keep).groupby(cand.i1.values).any()).sum()} of {n1} S1 have >=1 match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="block,ctx,feat,train,predict")
    ap.add_argument("--k-rev", type=int, default=2)
    ap.add_argument("--k-fwd", type=int, default=4)
    ap.add_argument("--max-df", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--admin-strip", type=int, default=0)
    ap.add_argument("--blocks", default="")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args()
    global ADMIN_STRIP, BLOCKS
    ADMIN_STRIP = bool(args.admin_strip)
    BLOCKS = tuple(b for b in args.blocks.split(",") if b)
    st = args.stages.split(",")
    if len(st) > 1:  # run every stage in a fresh process (OpenMP isolation, memory release)
        for stage in st:
            argv, skip = [], False
            for a in sys.argv[1:]:
                if skip:
                    skip = False
                elif a == "--stages":
                    skip = True
                elif not a.startswith("--stages="):
                    argv.append(a)
            cmd = [sys.executable, "-m", "ber.run"] + argv
            log(f"=== stage {stage}")
            subprocess.run(cmd + ["--stages", stage], check=True)
        return
    for split in ("train", "test"):
        if "block" in st:
            stage_block(args.art, split, args)
        if "ctx" in st:
            stage_ctx(args.art, split)
        if "feat" in st:
            stage_feat(args.art, split)
        if "feat2" in st:
            stage_feat2(args.art, split)
        if "feat3" in st:
            stage_feat3(args.art, split)
    if "train" in st:
        stage_train(args.art, args)
    if "predict" in st:
        stage_predict(args.art, args.out, args)


if __name__ == "__main__":
    main()

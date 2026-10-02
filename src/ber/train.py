"""Labels, deterministic S1-level splitting and LightGBM fitting for the pair matcher.

Folds and training subsamples are defined over Source-1 entities (all candidate pairs
of one S1 always stay together), using a stable hash of the entity id so that runs are
reproducible without storing split files.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=10, verbose=-1, seed=42)


def hash_frac(ids, seed):
    """Deterministic pseudo-random number in [0,1) per entity id."""
    h = pd.util.hash_array(np.asarray(ids, dtype=object), hash_key=f"ber-seed-{seed:07d}")
    return (h % 1_000_003) / 1_000_003.0


def label_pairs(cand, eid, pairs):
    """1 if (S1, candidate) is a ground-truth match, else 0. Pairs never observed in the
    ground truth are negatives (Source 1 is complete and deduplicated)."""
    key = pd.MultiIndex.from_arrays([pairs.source1_entity_id.values, pairs.ids.values])
    ck = pd.MultiIndex.from_arrays([eid[cand.i1.values], eid[cand.io.values]])
    return ck.isin(key).astype(np.int8)


def fit(X, y, Xv=None, yv=None, rounds=3000):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is None:
        return lgb.train(PARAMS, dtr, rounds)
    dv = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(PARAMS, dtr, rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

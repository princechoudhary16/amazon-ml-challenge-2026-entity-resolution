"""Data loading and parallel normalisation of all source records."""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import normalize as N


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")


def load_split(data_dir, split):
    """Return (records DataFrame with a 'src' column in {1,2,3}, ground truth or None)."""
    parts = []
    for s in (1, 2, 3):
        d = read_tsv(os.path.join(data_dir, split, f"{split}_source{s}.tsv"))
        d["src"] = np.int8(s)
        parts.append(d)
    recs = pd.concat(parts, ignore_index=True)
    gt_path = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
    gt = read_tsv(gt_path) if os.path.exists(gt_path) else None
    return recs, gt


def explode_gt(gt):
    ids = gt.matched_entity_ids.str.split(",")
    ex = pd.DataFrame({"source1_entity_id": gt.source1_entity_id, "ids": ids}).explode("ids")
    ex = ex[ex.ids.notna() & (ex.ids != "")]
    return ex.reset_index(drop=True)


def _init(tdict):
    N.set_translit_dict(tdict)


def _norm_chunk(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        canon, core, legal, web = N.norm_name(n)
        acanon, atoks, nums = N.norm_addr(a)
        out.append((canon, " ".join(core), " ".join(legal), web, acanon, " ".join(nums)))
    return out


def normalise_records(recs, tdict, workers=None, chunk=50000):
    """Add normalised columns: name_c, core, legal, web, addr_c, nums."""
    workers = workers or os.cpu_count()
    names = recs.business_name.tolist()
    addrs = recs.business_address.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(workers, initializer=_init, initargs=(tdict,)) as pool:
        res = pool.map(_norm_chunk, jobs)
    flat = [r for part in res for r in part]
    cols = ["name_c", "core", "legal", "web", "addr_c", "nums"]
    df = pd.DataFrame(flat, columns=cols, index=recs.index)
    df["web"] = df.web.astype(np.int8)
    return pd.concat([recs, df], axis=1)

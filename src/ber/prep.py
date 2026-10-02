"""Step 1: ingest, sanity-check, learn transliteration dictionary (train only), normalise.

Usage: python -m ber.prep --data-dir <student_resource/dataset> --art <artifacts dir>
"""
import argparse
import os
import time

from . import data, translit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--art", required=True)
    args = ap.parse_args()
    os.makedirs(args.art, exist_ok=True)
    t = time.time()

    tr, gt = data.load_split(args.data_dir, "train")
    te, _ = data.load_split(args.data_dir, "test")
    for name, d in (("train", tr), ("test", te)):
        assert not d.entity_id.duplicated().any(), f"duplicate ids in {name}"
        print(name, len(d), d.groupby(["src", "country"]).size().to_dict(), flush=True)
    pairs = data.explode_gt(gt)
    pairs.to_parquet(os.path.join(args.art, "train_pairs.parquet"))

    ti = tr.set_index("entity_id")
    tdict = translit.learn_translit(ti[ti.src == 1], ti[ti.src != 1], pairs)
    translit.save_dict(tdict, os.path.join(args.art, "translit_dict.tsv"))
    print(f"translit dict: {len(tdict)} tokens ({time.time() - t:.0f}s)", flush=True)

    for name, d in (("train", tr), ("test", te)):
        nd = data.normalise_records(d, tdict)
        nd.to_parquet(os.path.join(args.art, f"{name}_norm.parquet"))
        print(f"normalised {name} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

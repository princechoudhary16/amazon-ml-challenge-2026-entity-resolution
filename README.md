# Business Entity Resolution at Scale

[![tests](https://github.com/princechoudhary16/amazon-ml-challenge-2026-entity-resolution/actions/workflows/tests.yml/badge.svg)](https://github.com/princechoudhary16/amazon-ml-challenge-2026-entity-resolution/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

Linking noisy business records from multiple sources to a deduplicated reference set:
**12.5M records, three countries, eight Indic scripts, no shared identifiers**. Built for
the **Amazon ML Challenge 2026** (Business Entity Resolution), where it qualified for the
Top-500 round.

For every Source-1 business, the pipeline finds all matching Source-2 / Source-3 records.
A business may have zero, one or many matches. Scoring is **macro F0.5** per Source-1
entity, which weights precision twice as much as recall and gives singletons full credit
when they are correctly left unmatched.

| Metric | Value |
| --- | --- |
| Out-of-fold macro F0.5 (2.21M training entities) | **0.9777** |
| Public leaderboard macro F0.5 | **0.968** |
| Blocking pair recall | **97.9%** |
| Candidates per Source-1 entity | **13.6** (vs. 10M+ possible) |
| Hardware | one 10-core / 16 GB laptop, no GPU, no external data |

---

## How it works

```
raw TSVs
  │  prep      normalise names/addresses; learn an Indic→Latin token dictionary from training pairs
  ▼
  │  block     country-partitioned, feature-hashed TF-IDF retrieval, top-k in both directions
  ▼            (≈13 candidates per entity, ~linear time)
  │  ctx/feat  59 pairwise string, house-number, retrieval and group-context features
  ▼
  │  train     LightGBM, 3 folds grouped by Source-1 entity, out-of-fold predictions for every pair
  ▼
  │  decide    each record is assigned to at most one entity; threshold + margin tuned on OOF macro F0.5
  ▼
matching_results.tsv  +  candidate_pairs.tsv
```

**Key ideas**

- **Name × locality cross tokens.** Many different businesses share an identical name and
  differ only by city or locality. Locality words are too common to index on their own, but
  `name-token|locality` pairs are rare and disambiguating. They lifted top-1 retrieval
  accuracy in India from 90.1% to 94.5%.
- **Learned transliteration plus a phonetic key.** Names written in Devanagari, Gujarati,
  Tamil and other scripts are mapped to Latin tokens with a dictionary learned from training
  pairs alone, backed by a consonant-skeleton phonetic key (`apex`/`epeks` → `pks`).
  Recall on Indic names rose from 86% to 98.5%.
- **Exclusive assignment.** In the training data a record never belongs to two entities, so
  each record is linked only to its most probable entity, and only if it clearly beats the
  runner-up.
- **Metric-exact tuning.** The decision rule is chosen on out-of-fold predictions using the
  real competition metric, unit-tested and cross-checked against a reference implementation.
- **Built for a laptop.** Feature hashing, streamed feature chunks, resumable stages, and
  process isolation for libraries whose OpenMP runtimes conflict.

The full write-up, with EDA, ablations, error analysis and generalisation lessons, is in
[`docs/methodology.md`](docs/methodology.md).

---

## Quickstart

### 1. Install

Requires Python 3.12. On macOS, LightGBM also needs OpenMP: `brew install libomp`.

```bash
git clone https://github.com/princechoudhary16/amazon-ml-challenge-2026-entity-resolution.git
cd amazon-ml-challenge-2026-entity-resolution
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # exact pinned versions used for the submission
pip install -e .                   # makes the `ber` package importable
```

### 2. Get the data

The challenge dataset is not redistributed in this repository. Download `student_resource`
from the challenge page and place it at the repository root:

```
student_resource/
├── dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
├── dataset/test/{test_source1,test_source2,test_source3}.tsv
└── utils/validate_submission.py
```

### 3. Run the pipeline

```bash
# normalise records + learn the transliteration dictionary              (~4 min)
python -m ber.prep --data-dir student_resource/dataset --art artifacts

# blocking -> features -> training -> prediction                        (~2.5 h)
python -m ber.run --art artifacts --out output
```

Each stage runs in its own process and caches its result in `artifacts/`, so an interrupted
run picks up where it stopped. The defaults reproduce the competition submission exactly.
On a 16 GB machine you can run it under a memory cap:

```bash
scripts/run_with_memory_cap.sh 9.5 run.log python -m ber.run --art artifacts --out output
```

### 4. Validate and package

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test

scripts/package_submission.sh <team_name>   # builds <team_name>_submission.zip in the challenge layout
```

### Tests

```bash
pytest -q
```

---

## Project layout

```
src/ber/
├── prep.py         ingestion, integrity checks, parallel normalisation
├── normalize.py    Unicode/script folding, legal-form and street canonicalisation, phonetic key
├── translit.py     Indic→Latin token dictionary learned from training pairs
├── blocking.py     hashed TF-IDF candidate generation (sparse top-k, both directions)
├── features.py     pairwise and group-context features
├── train.py        labels, deterministic entity-level folds, LightGBM
├── decide.py       exclusive assignment + decision-rule tuning (and an expected-F0.5 variant)
├── metrics.py      macro F0.5 (reference + vectorised)
├── run.py          stage orchestration and output writing
└── holdout.py      leave-one-country-out generalisation check
tests/              metric edge cases, normalisation, phonetic key
scripts/            memory-capped runner, submission packaging
docs/               methodology write-up, submission-package README
```

## Configuration

`python -m ber.run --help` lists every option. The defaults are the submitted configuration:

| Option | Default | Meaning |
| --- | --- | --- |
| `--k-rev` / `--k-fwd` | 2 / 4 | retrieval depth: entities per record / records per entity and source |
| `--max-df` | 2000 | drop hashed tokens more frequent than this among Source-1 records |
| `--folds` / `--train-frac` | 3 / 0.3 | entity-grouped folds / share of entities used to train each fold |
| `--lr` / `--rounds` | 0.05 / 3000 | LightGBM learning rate / max boosting rounds (early stopping) |
| `--blocks`, `--admin-strip` | off | experimental feature blocks (see `docs/methodology.md` §5.2) |

## License

[MIT](LICENSE)

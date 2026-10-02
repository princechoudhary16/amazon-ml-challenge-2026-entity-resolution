# Business Entity Resolution: reproducible pipeline

```
raw TSVs ──► prep (normalise, learned Indic→Latin dictionary)
         ──► block (country-partitioned hashed TF-IDF top-k retrieval, both directions)
         ──► ctx + feat (59 pairwise / group-context features)
         ──► train (LightGBM, 3 folds grouped by Source-1 entity, OOF tuning of the decision rule)
         ──► predict (mean of the 3 fold models + exclusive assignment) ──► output TSVs
```

No external data, APIs, geocoders or pretrained models are used. The only learned
artefacts (the transliteration dictionary and the LightGBM models) are fit on the provided
training files.

## 1. Environment

- Python 3.12 (tested 3.12.14 on macOS 15 / Apple M5, 10 cores, 16 GB RAM)
- `pip install -r requirements.txt` (exact pinned versions)
- macOS only: LightGBM needs the OpenMP runtime → `brew install libomp`

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Reproduce the submission end to end

Set `DATA` to the challenge `dataset/` folder, which contains `train/` and `test/`. Run the
commands from `src/`:

```bash
cd src
DATA=/path/to/student_resource/dataset
ART=../artifacts      # cache / models (created)
OUT=../output         # final outputs (created)

# Step 1: ingest + sanity checks, learn the Indic->Latin token dictionary from TRAIN
#         ground-truth pairs only, normalise every record                       (~4 min)
python -m ber.prep --data-dir $DATA --art $ART

# Step 2: all remaining stages with the final configuration (the defaults).   (~2.5 h)
#         Each stage runs in its own process and caches its result in $ART, so
#         an interrupted run resumes where it stopped.
python -m ber.run --art $ART --out $OUT

# Step 3: validate the two output files
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir $DATA/test
```

Final configuration (all defaults):
`--k-rev 2 --k-fwd 4 --max-df 2000 --folds 3 --train-frac 0.3 --lr 0.05 --rounds 3000 --admin-strip 0 --blocks ""`.
The decision rule (threshold 0.65, margin 0.30) is selected automatically on out-of-fold
macro F0.5 and written to `$ART/config.json`.

### What each stage writes

| stage | output in `$ART` | content |
| --- | --- | --- |
| `prep` | `translit_dict.tsv`, `{train,test}_norm.parquet`, `train_pairs.parquet` | normalised records, exploded ground truth |
| `block` | `{split}_cand.parquet` | **exact candidate set** scored by the model (= `candidate_pairs.tsv`) |
| `ctx` | `{split}_ctx.parquet` | cheap similarities + group-context features for all pairs |
| `feat` | `feat_{split}/part-*.parquet` | full pairwise features in 1M-pair chunks |
| `train` | `model_fold{0,1,2}.txt`, `oof.npy`, `config.json`, `threshold_table.tsv`, `feature_importance.tsv` | models, out-of-fold predictions, decision-rule tuning |
| `predict` | `test_p.npy` + `$OUT/*.tsv` | test probabilities and the two submission files |

### Determinism
LightGBM uses `seed=42` with fixed thread count. Folds and training subsamples come from
a stable hash of the Source-1 entity id (`ber/train.py:hash_frac`), so no split files are
needed. We checked reproducibility by re-running the `predict` stage of this code with the
saved fold models. It regenerates `matching_results.tsv` byte-for-byte (identical MD5) to
the file submitted to the leaderboard.

### Resources
Peak memory stays below about 9 GB: feature chunks are streamed from disk, and token
strings are built and hashed inside worker processes. Blocking (`sparse_dot_topn`) and
LightGBM bundle different OpenMP runtimes that segfault when loaded into one process,
so every stage runs in its own subprocess. Approximate wall-clock times on a 10-core
laptop: prep 4 min, block 30 min, ctx 5 min, feat 15 min, train + OOF 40 min,
predict 35 min.

## 3. Code layout (`src/ber/`)

| module | role |
| --- | --- |
| `prep.py` | load TSVs (explicit `sep="\t"`), integrity checks, translit dictionary, parallel normalisation |
| `normalize.py` | Unicode/accent/script folding, legal-form and street-type canonicalisation, number extraction, phonetic key |
| `translit.py` | Indic token → Latin token dictionary learned from training pairs (positional alignment + Dice) |
| `blocking.py` | candidate generation: hashed TF-IDF over name / phonetic / address / name×locality tokens, sparse top-k both directions |
| `features.py` | pairwise string, number and retrieval features; group-context features |
| `train.py` | labels, deterministic S1-level folds, LightGBM fitting |
| `decide.py` | exclusive assignment of each record to one S1 + threshold/margin tuning; expected-F0.5 alternative |
| `metrics.py` | competition metric (macro per-S1 F0.5 incl. singletons) + vectorised version (cross-checked) |
| `run.py` | stage orchestration, output writing |
| `holdout.py` | leave-one-country-out validation (diagnostic used for model selection) |

Unit tests (metric edge cases, normalisation, phonetic key): `PYTHONPATH=src python -m pytest -q tests`

## 4. Experimental options (not used in the final submission)

These options are kept for transparency. Methodology §5.2 describes their results and why
they were not selected.

```bash
# extra features (v2 family) + corroboration features, admin-token stripping
python -m ber.run --art $ART --out $OUT_EXP --stages ctx,feat,feat2,feat3,train,predict \
    --admin-strip 1 --blocks feat2,feat3 --lr 0.08 --rounds 4000 --train-frac 0.4
# leave-one-country-out generalisation check
python -m ber.holdout --art $ART --train-country US --eval-country India
```

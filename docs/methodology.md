# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** YumMail
**Team Members:** Nischit Hosamani, Prince Choudhary, Saksham Gupta
**Submission Date:** 27 September 2026 (final leaderboard file); package submitted October 2026

---

## 1. Executive Summary

We link Source-2/Source-3 records to the deduplicated Source-1 reference with a three-stage,
laptop-scale pipeline:

1. **Blocking.** Country-partitioned, feature-hashed TF-IDF retrieval runs in both
   directions. It uses name, phonetic-name, address and *name × locality cross* tokens,
   plus an Indic→Latin dictionary learned from the training pairs.
2. **Matching.** A LightGBM pair classifier uses 59 string, house-number, retrieval and
   group-context features.
3. **Decision.** Each record is assigned to at most one Source-1 entity, and the threshold
   and margin are tuned directly on out-of-fold macro F0.5.

Results: out-of-fold macro F0.5 of **0.9777** over all 2.21M training Source-1 entities,
**0.968 on the public leaderboard**, with **13.6 candidates per Source-1 entity** at
97.9% pair recall. No external data, APIs or pretrained models are used.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the full training set (2,206,821 S1; 5,034,616 S2; 5,285,603 S3; 7,638,365 true links):

| Finding | Value | Consequence |
| --- | --- | --- |
| S1 singletons (no match) | 5.6% | empty predictions are rarely right |
| Matches per S1 | mean 3.46, max 11 (≤5 from S2, ≤6 from S3) | multi-match lists are the norm |
| S2/S3 records linked to more than one S1 | **0** | each record can be assigned to at most one S1 |
| S2/S3 records linked to no S1 | about 26% | many distractors |
| Cross-country true links | **0** | compare records only within the same `country` label |
| Indic-script names in S2/S3 | 5–9% (train), 6–11% (test), 8 scripts | transliteration is needed |
| Empty address in S2/S3 | 3.3% | some records carry name-only evidence |
| Test: S2+S3 records per S1 | 5.5–5.8 (train 4.7) | denser test pools; see §5.3 |

Observed noise:
- legal-suffix variants and moved suffixes (`LLC 4220 First Ave`)
- honorifics (`Sri`, `Smt`, `M/s`)
- junk tokens (`#77599`, `[..]`, `<<`, `--`) and appended words (`Center`, `Services`)
- website forms (`4220firstave.Com`, `Acme \| www.acme.com`)
- OCR-like typos (`8uilding`, `MILLWOSD`)
- names and state names in Devanagari, Gujarati, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi and Odia scripts
- state written as an abbreviation or in full; component reordering
- zero padding (`0020718`); house-number edits even in true matches (`4625`→`462`)
- French région vs département labels (`Nouvelle-Aquitaine` vs `Gironde`)

The decisive finding came from analysing blocking misses. **Many distinct S1 entities share
an identical name**, for example "Classic Foundation Private Limited" in Noida and in Surat.
Only the locality tells them apart, and locality words are individually too frequent to
index cheaply.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pair classifier + metric-aware exclusive assignment

**Core Innovation:**
1. **Name × locality cross tokens** inside hashed TF-IDF retrieval. They are rare, cheap and
   disambiguating, and they raised India top-1 retrieval accuracy from 90.1% to 94.5%.
2. **Indic→Latin dictionary learned from training pairs**, plus a phonetic consonant-skeleton
   key. Together they raised Indic-name recall from 86% to 98.5%.
3. **Exclusive assignment of each record to one S1**, with the decision rule chosen on the
   exact competition metric computed out of fold.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`normalize.py`, `translit.py`):
- **Unicode and script handling.** Text is NFKC-normalised. Indic tokens are replaced by a
  dictionary of 1,363 tokens learned only from training ground-truth pairs: name tokens are
  aligned to S1 name tokens by position or Dice co-occurrence, and address tokens to S1
  address tokens. Examples: प्राइवेट→pvt, ଲିମିଟେଡ୍→ltd, टेक्नोलॉजीज→technologies.
  Separating name and address alignment fixed a bug in which "Private" in every script
  mapped to a state name. Remaining non-ASCII text is transliterated and accent-folded.
- **Names.** Legal forms are canonicalised (pvt, ltd, llc, llp, inc, corp, sarl, sas,
  eurl, …) and separated from the core name. Honorifics are dropped, and web domains are
  reduced to their stem.
- **Addresses.** Generic street-type and direction abbreviations are mapped (road→rd,
  r→rue, …), US and Indian state names become codes, and numbers are de-padded and
  extracted.

**Blocking tokens per record** (`blocking.py`):
- name-core unigrams, bigrams and concatenated core (`c:acmetech` matches `acmetech.com`)
- phonetic consonant-skeleton tokens (apex/epeks→`pks`, properties/puraapprttttiis→`prprts`), with bigrams and concatenation
- address unigrams and bigrams (house number + street bigrams are very selective)
- **cross tokens** `name-skeleton | address-word` (for example `klsk|surat`)

**Retrieval:** each country label is handled separately, as an open set, so France needed
no special handling. Tokens are feature-hashed into 2^26 buckets. IDF is fit on Source 1,
and buckets with Source-1 document frequency above 2,000 are dropped to keep cost linear.
Rows are L2-normalised and cosine top-k is computed with multithreaded sparse top-n
matrix multiplication.
- **reverse:** every S2/S3 record retrieves its top-2 S1 entities
- **forward:** every S1 retrieves its top-4 records from S2 and its top-4 from S3

The candidate set is the union of both directions. The retrieval score and both ranks are
kept as features. There is no all-pairs comparison; blocking the 12.5M training records
takes about 18 minutes on a laptop.

- **Blocking keys used:** hashed TF-IDF over name, phonetic-name, address and name×locality cross tokens; country partition
- **Candidate pairs generated:** **23,601,025 for the test set (13.62 per S1: US 13.4, India 13.8, France 13.5)**; train 26,719,066 (12.11 per S1)
- **Recall ceiling (train):** pair recall **97.94%** (India 97.45%, US 98.27%). A perfect matcher on our candidates would score macro F0.5 **0.9941**.
- **How we ensured true matches were not lost:** every change was A/B-tested for pair
  recall by segment (Indic names, empty addresses, web names), and samples of missed true
  pairs were inspected after each round.

| Blocking variant (India, train) | Pair recall | Candidates / S1 |
| --- | --- | --- |
| word TF-IDF, top-3 rev + top-6 fwd | 94.9% | 19.3 |
| + phonetic name tokens | 95.3% | 19.7 |
| + corrected translit + cross tokens, top-1 rev + top-4 fwd | 97.1% | 8.4 |
| **+ cross tokens, top-2 rev + top-4 fwd (final)** | **97.5%** | **12.3** |

Compared with the first version, the final blocking has higher recall
(97.9% vs 96.9% overall) and **37% fewer candidates** (12.1 vs 19.1 per S1).

---

## 4. Matching Model

**Features used** (`features.py`, 59 in total). Country is never a feature.

- **Name:**
  - RapidFuzz ratio, token-sort, token-set, partial and WRatio (on canonical and core names)
  - Jaro-Winkler
  - ratio and partial ratio of the concatenated core, which catches website forms
  - char-3gram cosine
  - phonetic-skeleton token-set ratio and Jaccard
  - token Jaccard, containment and intersection count; exact flag
  - lengths and token counts
  - legal-form equality, Jaccard and presence
- **Address:**
  - ratio, token-sort, token-set and partial ratio
  - token Jaccard, containment and intersection count
  - exact flag, empty-address flag, lengths
- **House numbers:** Jaccard of numeric tokens, first-number agreement, and **number conflict** (both present and disjoint)
- **Cross-field:** candidate-name tokens appearing in the S1 address (C/O and landmark references)
- **Retrieval provenance:** cosine score, reverse rank, forward rank, source (S2/S3), web flag
- **Group context** (no labels involved, so valid at test time):
  - within the S1's candidate pool: max, gap to max, rank, pool size, name and address gaps, count of strong candidates
  - within the S1s competing for the same record: the same statistics, plus the best competing score

Top features by gain: composite similarity (43%), gap to the best competing S1 for the same
record (11%), number Jaccard (8%), number conflict (7%), retrieval score (4%), rank among
competing S1s (4%).

**Model type:** LightGBM binary classifier.

| Parameter | Value |
| --- | --- |
| objective | binary |
| learning_rate | 0.05 |
| num_leaves | 127 |
| min_data_in_leaf | 100 |
| feature_fraction | 0.8 |
| bagging_fraction / bagging_freq | 0.8 / 1 |
| lambda_l2 | 1.0 |
| rounds | up to 3,000, early stopping (patience 100) on the held-out fold |
| seed | 42 |

Validation used 3 folds **grouped by Source-1 entity**, assigned by a deterministic hash of
the S1 id. Each fold trains on a 30% S1 subsample (about 5.3M pairs). Out-of-fold
predictions are produced for all 26.7M training pairs. Test scores are the mean of the 3
fold models.

**Threshold selection method:** the exact competition metric (macro per-S1 F0.5 with
singletons, unit-tested and cross-checked against a reference implementation) is
maximised on out-of-fold predictions.

1. **Exclusive assignment** (selected). A record is linked only to its most probable S1,
   only if p ≥ threshold, and only if p beats the second-best S1 for that record by a
   margin. Selected: **threshold 0.65, margin 0.30**.
2. **Expected-F0.5 selection** (compared). For each S1, choose the k that maximises
   1.25·Σp_topk / (k + 0.25·E[T]), or the empty list when P(no match) is higher. Out of
   fold this scores 0.97750, against 0.97766 for exclusive assignment, so it was not used.

**Robustness check:** the out-of-fold F0.5 surface is flat. Every threshold from 0.55 to
0.75 with margin 0.3 scores within 0.0005 of the optimum, so the chosen point is not a
noise-fitted peak.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold, all 2.21M training S1):** **0.9777** (US 0.9803, India 0.9737)
- **Public leaderboard (final submission):** **0.968**
- **Test output:** 5,782,728 links; 5.7% of S1 left empty (train singleton rate 5.6%)

### 5.1 Error analysis (out of fold)

| Loss source | Share of F0.5 lost |
| --- | --- |
| missed matches only (no false link) | 0.0166 |
| S1s with a false link (non-singleton) | 0.0045 |
| false links on singletons | 0.0012 |

- **Common false positives (wrong merges):** sibling businesses, meaning the same or a
  nearly identical name at a neighbouring house number (`964` vs `965 Mayfair Ct`,
  `Best Ashford` `1128` vs `1139 Bo Fork`) or with one changed name word
  (`Finance Lokenath Services` vs `Travels`); also name-only records whose name occurs
  for several S1s.
- **Common false negatives (missed matches):** 97% of in-candidate misses are scored
  below the threshold rather than lost to another S1. They include true copies with house
  numbers edited by noise (`934`→`933`, `4625`→`462`), which look like sibling negatives;
  name-only records (empty address); and records whose name is replaced by an opaque
  string (`Umbralumarc`), so only the address links them. A further 2.1% of true pairs
  never reach the candidate set, most of them name-only records (76% recall for empty
  addresses).

### 5.2 Experiments not selected for the final submission (transparency)

| Version | Change | OOF F0.5 | Public LB | Outcome |
| --- | --- | --- | --- | --- |
| **v1 (final)** | as described above | **0.9777** | **0.968** | selected |
| v2 | + 14 features (fuzzy house numbers, street-only similarity, IDF-weighted missing/extra tokens with IDF from S1), lr 0.08, 40% sample | 0.9826 | 0.962 | rejected: better OOF, worse LB |
| hybrid | v2 for US/India + v1 for France | – | 0.960 | diagnostic only |
| v6 | v2 features with scale-invariant IDF + admin-token stripping + pool corroboration features | 0.9830 | not submitted | not selected (see below) |

### 5.3 Lessons on generalisation

The three leaderboard scores together show that v2 lost about 0.008 on **US/India**, not on
France. Its IDF-weighted features depend on corpus size, and the test split has a
different composition: half as many US S1 records, and 23% more S2/S3 records per S1. So
the model's calibration shifted between train and test. Grouped K-fold validation cannot
reveal this, because all folds share the training corpus. We therefore added two
corpus-shift diagnostics:

1. **Leave-one-country-out validation** (`holdout.py`): train on US only, evaluate on India
   as a stand-in for an unseen country like France. The base feature set scores 0.919;
   the v6 features score 0.936. A stricter threshold (about 0.90) is optimal on an unseen
   country.
2. **Predicted-link-rate comparison** between out-of-fold and test, per country.

We kept v1 as the final submission because it had the best verified leaderboard score and
transferred consistently: 3.31 links per S1 out of fold vs 3.34 on test.

---

## 6. Conclusion

EDA-driven blocking (name×locality cross tokens, phonetic keys, a learned transliteration
dictionary) produced most of the gain while keeping the candidate set small, at 13.6 per
S1 with 97.9% recall. Exploiting the one-S1-per-record structure, with metric-exact
out-of-fold tuning, turned a strong pair classifier into a precision-oriented macro-F0.5
decision. The main lesson is that train/test corpus-composition shift, and not only an
unseen country, can reverse offline gains. Corpus-size-dependent features need
scale-invariant definitions, and validation should include leave-one-country-out and
link-rate checks. Natural next steps:
- scale-invariant corroboration features (v6), validated with the holdout protocol
- a compact multilingual cross-encoder (MIT/Apache, ≤8B) as a re-ranker
- a name-only blocking pass for empty-address records

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: see its `README.md` for the exact environment and
commands. The defaults reproduce this submission:

```
cd src
python -m ber.prep --data-dir <dataset> --art ../artifacts          # ingest, translit dictionary, normalise
python -m ber.run  --art ../artifacts --out ../output               # block, ctx, feat, train, predict
```

`ber/prep.py` ingestion · `ber/normalize.py` · `ber/translit.py` · `ber/blocking.py` ·
`ber/features.py` · `ber/train.py` · `ber/decide.py` · `ber/metrics.py` · `ber/run.py` ·
`ber/holdout.py`; unit tests in `tests/`.

- **Determinism.** LightGBM uses seed 42, and splits come from a hash of the entity id.
  Re-running `predict` with the saved fold models regenerates the submitted
  `matching_results.tsv` byte-for-byte.
- **Resumability.** Every stage caches to disk and resumes after an interruption.
- **Process isolation.** Blocking and LightGBM run in separate processes because their
  OpenMP runtimes conflict.

**Models and licences.** The only model is our LightGBM classifier (LightGBM: MIT), trained
from scratch on the provided data. **No pretrained model is used.** Libraries: pandas,
numpy, scipy, scikit-learn, pyarrow (BSD / Apache-2.0); RapidFuzz (MIT); sparse_dot_topn
(Apache-2.0); Unidecode (GPL-2.0+), used only as a character-transliteration utility.

### B. Additional Results

- **Out-of-fold F0.5 surface** (exclusive assignment). Rows are thresholds; columns are margins.

| thr \ margin | 0.00 | 0.10 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- |
| 0.55 | 0.9767 | 0.9771 | 0.9773 | 0.9775 |
| 0.60 | 0.9771 | 0.9773 | 0.9775 | 0.9776 |
| **0.65** | 0.9773 | 0.9775 | 0.9776 | **0.9777** |
| 0.70 | 0.9773 | 0.9774 | 0.9775 | 0.9775 |
| 0.75 | 0.9771 | 0.9771 | 0.9772 | 0.9772 |

- **Pipeline wall-clock on a 10-core / 16 GB laptop:** prep 4 min, block 30 min, features 20 min, train + OOF 40 min, predict 35 min.

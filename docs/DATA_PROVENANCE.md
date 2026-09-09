# Data Provenance

**Audited:** 2026-09-09
**Method:** direct filesystem and git audit of `C:\Projects\Hiver`, not recollection.

This document exists so that no reader — reviewer, interviewer, or future maintainer — can
mistake a synthetic test fixture for real corpus data. Every dataset touching this project is
classified into exactly one of four categories, and the boundaries between them are enforced.

---

## Headline statement

> **No real TWCS data has been downloaded or processed. Zero real tweets have passed through any
> module in this repository. All 137 passing tests run on synthetic, in-memory fixtures.**

Anything in this repository that reads as a result is a result **about code behaviour on
constructed inputs**, not a finding about the Customer Support on Twitter corpus.

---

## The four categories

### Category 1 — Real TWCS / Kaggle data

**Status: NOT PRESENT. Never downloaded, never processed.**

| Property | Value |
|---|---|
| Source | Kaggle `thoughtvector/customer-support-on-twitter` |
| Location when acquired | `data/raw/` |
| Committed to git? | **Never** — `.gitignore` excludes `data/raw/`, `data/processed/`, `data/interim/`, `*.csv` |
| Acquisition | `python scripts/fetch_data.py` |
| Currently on disk? | **No.** `data/` does not exist |
| Rows processed to date | **0** |

`scripts/fetch_data.py` is the only file in the entire repository that reads a data file from
disk (`pd.read_csv`, line 81). It has never been executed — the directory it would populate does
not exist.

Reviewers reproduce this data by running the script against their own Kaggle credentials. We
never redistribute the corpus.

### Category 2 — Public repository data / forensic artifacts

**Status: PRESENT ON THE MACHINE, OUTSIDE THIS REPOSITORY. Never copied in.**

The seven public Hiver repositories were cloned for the forensic analysis in
`PUBLIC_REPO_COMPARISON.md`. They live in the session scratchpad:

```
C:\Users\nitis\AppData\Local\Temp\claude\...\scratchpad\repos\
```

| Property | Value |
|---|---|
| Purpose | Read-only forensic evidence for `PUBLIC_REPO_COMPARISON.md` |
| Inside `C:\Projects\Hiver`? | **No** |
| Any file copied into this project? | **No** — verified by file audit |
| Their golden labels, metrics, reports used? | **No** |
| Licence status | Six of seven unlicensed (all rights reserved); one MIT |

Their committed CSVs (e.g. `golden_labeled.csv`, `apple_support_golden_set_200.csv`,
`amazonhelp_conversations.csv`) were **parsed to audit their claims** and are cited as evidence
of *their* methodology. None of that data is used as input to our system, and none of it will
appear in our golden set, our corpus, or our results. See `DECISION_LOG.md` D1.

### Category 3 — Synthetic test fixtures

**Status: PRESENT. This is the only data the 137 tests use.**

Every fixture is constructed in memory, inside the test file, by a Python helper. There are no
fixture *files* — `tests/fixtures/` exists but is **empty**, and nothing reads from it.

| Test file | Tests | Fixture source | What it is |
|---|---|---|---|
| `test_thread_reconstruction.py` | 23 | `_rows()`, `_simple_thread()` | Hand-written `pd.DataFrame` rows imitating the documented TWCS schema |
| `test_pii_handling.py` | 29 | Inline string literals | Hand-written messages containing invented emails, phones, cards, order ids |
| `test_normalisation.py` | 31 | Inline string literals | Hand-written messages with URLs, mentions, entities, emoji |
| `test_leakage.py` | 32 | `_pair()`, `_corpus()`, `_golden()` | Constructed `SupportPair` objects with invented text |
| `test_temporal_split.py` | 22 | `_pair()`, `_corpus()`, `_long_threads()` | Constructed pairs from a fixed vocabulary of subjects/problems |
| **Total** | **137** | | **100% synthetic** |

Every identifier, timestamp, handle, email, phone number, card number and order reference in
these fixtures is **invented**. `4111-1111-1111-1111` is the standard non-issued test card;
`@AmazonHelp`, `@AppleSupport` and `@sprintcare` appear only as brand-handle *strings* to test
handle preservation, and carry no data from those brands.

Fixtures are *modelled on* documented TWCS quirks (comma-separated `response_tweet_id`, absent
`in_response_to_tweet_id`, float-coerced ids, Twitter's date format). **Modelled on is not
drawn from.** The schema assumptions behind them are unverified against the real file, which is
precisely why Milestone 1 is still marked NOT COMPLETE.

### Category 4 — Generated / intermediate development artifacts

**Status: NONE EXIST.**

| Artifact | Status |
|---|---|
| Processed conversations | Not generated |
| Retrieval index / embeddings | Not generated |
| Trained models (`.pkl`, `.joblib`) | Not generated |
| Brand profile tables | Not generated |
| Golden set | Not created |
| Evaluation results / metrics | **None — nothing has been evaluated** |
| LLM response cache | None — no LLM call has been made |

A file audit confirms **zero** `.csv`, `.json`, `.jsonl`, `.parquet`, `.pkl` or `.joblib` files
anywhere in the repository. All 22 tracked files are source code, tests, or documentation.

---

## Language rules (binding on all code, docs and the final report)

To keep categories 1 and 3 from blurring, these phrasings are fixed:

| Permitted | Meaning |
|---|---|
| "137 unit/integration tests pass on synthetic fixtures" | Category 3 |
| "Verified on hand-written inputs modelled on documented TWCS quirks" | Category 3 |
| "Forensic finding from repository X's committed artifacts" | Category 2 |

| Forbidden until the corpus is actually processed |
|---|
| "validated against the real TWCS dataset" |
| "verified on real data" |
| "tested on the corpus" |
| any accuracy, coverage, or distribution figure attributed to TWCS |

**No metric derived from a fixture may ever be presented as a property of the dataset.** Fixture
counts describe the tests; they describe nothing about customer support on Twitter.

---

## What would change this document

Running `python scripts/fetch_data.py` creates Category 1 data. At that point:

1. Milestone 1's outstanding criterion — schema validation against real rows — can be attempted.
   The reconstruction code may well need changes; the fixtures encode assumptions, not findings.
2. `scripts/analyse_brands.py` produces the first Category 4 artifacts.
3. This document and `VERIFICATION.json` are updated **in the same commit** as the first real-data
   processing, so the claim state never lags the code state.

Until then, every statement this project makes about the Customer Support on Twitter corpus is a
statement about *documentation of* that corpus, not about the corpus itself.

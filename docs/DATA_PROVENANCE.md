# Data Provenance

**Audited:** 2026-09-09 | **Updated:** 2026-09-10 (corpus acquired; Category 1 now present)
**Method:** direct filesystem and git audit of `C:\Projects\Hiver`, not recollection.

This document exists so that no reader — reviewer, interviewer, or future maintainer — can
mistake a synthetic test fixture for real corpus data. Every dataset touching this project is
classified into exactly one of four categories, and the boundaries between them are enforced.

---

## Headline statement

> **The real corpus has been downloaded and processed. 2,811,774 records were read,
> 798,197 conversations reconstructed and 1,149,717 pairs extracted. Of 243 passing tests,
> 224 run on synthetic in-memory fixtures and 19 read the real corpus.**

The distinction still matters and is still enforced. `tests/test_real_data.py` is the **only**
module permitted to read the corpus; it skips entirely when the corpus is absent, so a
reviewer without 493 MB sees skips rather than silent passes. Everything else proves code
behaviour on constructed inputs and proves nothing about the dataset.

Statements about the corpus are legitimate **only** where they cite a measured artifact in
`reports/`. Statements about test behaviour are statements about tests.

---

## The four categories

### Category 1 — Real TWCS / Kaggle data

**Status: PRESENT locally as of 2026-09-10. Never committed.**

| Property | Value |
|---|---|
| Source | Kaggle `thoughtvector/customer-support-on-twitter` |
| Location when acquired | `data/raw/` |
| Committed to git? | **Never** — `.gitignore` excludes `data/raw/`, `data/processed/`, `data/interim/`, `*.csv` |
| Acquisition | `python scripts/fetch_data.py` |
| Currently on disk? | **Yes** — `data/raw/twcs/twcs.csv`, 493 MB |
| Records read | **2,811,774** |
| Conversations reconstructed | 798,197 |
| Customer/support pairs | 1,149,717 |
| Modified by this project? | **No** — opened read-only; `corpus_modified: false` in every provenance block |
| Tracked by git? | **No** — `git check-ignore` confirms `.gitignore:2:data/raw/` |

Verified ignored at the 2026-09-10 checkpoint:

```
$ git check-ignore -v data/raw/twcs/twcs.csv
.gitignore:2:data/raw/   data/raw/twcs/twcs.csv
$ git ls-files data/ | wc -l
0
```

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

**Status: PRESENT. The data 224 of the 243 tests use.**

Every fixture is constructed in memory, inside the test file, by a Python helper. There are no
fixture *files* — `tests/fixtures/` exists but is **empty**, and nothing reads from it.

**One sanctioned exception:** `tests/test_real_data.py` (19 tests) reads the corpus and is
named explicitly in `test_data_provenance.py` so the boundary stays visible rather than
eroding. It skips when the corpus is absent.

| Test file | Tests | Fixture source | What it is |
|---|---|---|---|
| `test_thread_reconstruction.py` | 23 | `_rows()`, `_simple_thread()` | Hand-written `pd.DataFrame` rows imitating the documented TWCS schema |
| `test_pii_handling.py` | 29 | Inline string literals | Hand-written messages containing invented emails, phones, cards, order ids |
| `test_normalisation.py` | 31 | Inline string literals | Hand-written messages with URLs, mentions, entities, emoji |
| `test_leakage.py` | 32 | `_pair()`, `_corpus()`, `_golden()` | Constructed `SupportPair` objects with invented text |
| `test_temporal_split.py` | 24 | `_pair()`, `_corpus()`, `_long_threads()` | Constructed pairs from a fixed vocabulary of subjects/problems |
| `test_credentials.py` | 20 | Fake tokens, fake home directories | No real credential is ever read |
| `test_reply_classification.py` | 54 | Verbatim corpus reply **text** as string literals | Reads no file; text quoted as evidence |
| `test_data_provenance.py` | 14 | git / filesystem audit | Enforces these boundaries |
| **Subtotal (synthetic)** | **224** | | |
| `test_real_data.py` | **19** | **the real corpus** | Skipped when absent |
| **Total** | **243** | | |

Every identifier, timestamp, handle, email, phone number, card number and order reference in
these fixtures is **invented**. `4111-1111-1111-1111` is the standard non-issued test card;
`@AmazonHelp`, `@AppleSupport` and `@sprintcare` appear only as brand-handle *strings* to test
handle preservation, and carry no data from those brands.

Fixtures are *modelled on* documented TWCS quirks (comma-separated `response_tweet_id`, absent
`in_response_to_tweet_id`, float-coerced ids, Twitter's date format). **Modelled on is not
drawn from.**

Those assumptions have since been **checked against the real file** and held:
`test_real_data.py` confirms fan-out lists, orphan roots and float-coerced ids all occur, and
reconstruction on real rows loses no rows. The fixtures were a good model — but they were a
model, and only the real-data tests license any statement about the corpus.

### Category 4 — Generated / intermediate development artifacts

**Status: brand-analysis artifacts exist. Everything downstream does not.**

| Artifact | Status |
|---|---|
| Brand profile tables | **Generated** — `reports/brand_profiles.json`, `brand_profiles_all.csv` |
| Brand decision | **Generated** — `reports/brand_decision.json`, `brand_selection.md` |
| Processed conversations | Not persisted (reconstructed in memory per run) |
| Retrieval index / embeddings | Not generated |
| Trained models (`.pkl`, `.joblib`) | Not generated |
| Golden set | **Not created — no labels exist** |
| Evaluation results / metrics | **None — no system has been evaluated** |
| LLM response cache | None — **no LLM call has ever been made** |

The committed artifacts are small (largest 103 KB), reproducible from the corpus via
`scripts/analyse_brands.py`, and carry full provenance. `test_data_provenance.py` enforces
that no tracked file exceeds 2 MB and that no bulk corpus data is tracked.

**No metric of system quality exists**, because no system has been built. Every number in
`reports/` is a descriptive statistic about the corpus, not a measure of performance.

---

## Language rules (binding on all code, docs and the final report)

To keep categories 1 and 3 from blurring, these phrasings are fixed:

| Permitted | Meaning |
|---|---|
| "224 tests pass on synthetic fixtures" | Category 3 |
| "Verified on hand-written inputs modelled on documented TWCS quirks" | Category 3 |
| "Reconstruction validated against the real corpus (19 real-data tests)" | Category 1 — **now true** |
| "2,811,774 records -> 1,149,717 pairs" citing `reports/brand_profiles.json` | Category 1 |
| "Forensic finding from repository X's committed artifacts" | Category 2 |

| Still forbidden |
|---|
| presenting a fixture result as a property of the dataset |
| reporting a **skipped** real-data test as a pass |
| any accuracy, F1 or quality figure attributed to **this system** — none has been measured |
| any claim about golden-set labels — none exist |

**No metric derived from a fixture may ever be presented as a property of the dataset.** Fixture
counts describe the tests; they describe nothing about customer support on Twitter.

---

## What would change this document

The next change to the claim state is the **intent taxonomy** and, after it, the **golden set**.
When labelling begins, Category 4 gains the golden set — the one derived artifact that
*should* be committed, because it is small, hand-made, and cannot be reproduced any other way.
`tests/test_data_provenance.py` already permits `data/golden/`.

At that point this document must record: how many labels exist, who produced them, and that
they are human rather than model-generated. Three of the seven audited public repositories
presented machine-generated labels as human ones; the guard against repeating that is writing
down what was actually done, at the time it is done.

**Rule: update this file and `VERIFICATION.json` in the same commit as any change to the claim
state, so the claim never lags the code.**

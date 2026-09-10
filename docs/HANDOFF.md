# HANDOFF — Continuation Document

**Last updated:** 2026-09-10
**Status:** Milestone 5 complete — end-to-end agent runs on real AppleSupport data.
**Next:** golden set. Golden set NOT started. The agent is NOT evaluated.

This document is written for a **different coding agent, on a different machine, with no
access to the conversation that produced this project**. It should be sufficient on its own.
If it conflicts with memory or assumption, this document and the files it points to win.

---

## 1. Project purpose

### What is being built

A **support resolution engine**: an AI customer-support agent that, given an incoming customer
message for one brand,

1. **classifies** it into a brand-specific intent taxonomy derived from that brand's own data,
2. **drafts a reply grounded** in how that brand historically resolved similar issues,
3. **decides `AUTO_HANDLE` vs `ESCALATE`** with a stated, machine-readable reason.

### Where it originated

A take-home assignment for an SDE intern role, built on the public *Customer Support on
Twitter* dataset. The assignment's explicit emphasis is: *"whether you can turn a messy
real-world dataset into a working AI system and prove it works. The proof is worth more than
the system."*

That emphasis drives every design decision here. **Evaluation integrity outranks model
performance throughout.** If you find yourself trading evaluation credibility for a better
number, you are working against the project's purpose.

Deliverables the assignment requires: a runnable pipeline, a 150–250 example hand-labelled
golden set, an evaluation harness with automated metrics and an LLM-as-judge, evidence of
judge–human agreement, results against at least two baselines, a top-5 failure analysis, a
mandatory *"what is misleading about my headline number?"* section, a one-week next-step plan,
and a 10–15 item decision log.

### Eventual product direction

The repository is named `support-resolution-engine` rather than after the assignment,
deliberately: the architecture is not employer-specific and the natural product expansion is
multi-brand support, real action execution (currently out of scope), and live deployment.
The current system is a **single-brand prototype**. Do not describe it as production-ready.

---

## 2. Current status

### Completed

| Milestone | Content | State |
|---|---|---|
| 0 | Public-repository forensics (7 competing repos audited) | ✅ COMPLETE |
| 1 | Scaffold + conversation reconstruction | ✅ COMPLETE (closed by real-data validation in M2) |
| 1b | PII masking at the API boundary | ✅ COMPLETE |
| 1c | Leakage guards (6, all raising) | ✅ COMPLETE |
| 1d | Text normalisation + deterministic temporal split | ✅ COMPLETE |
| 2 | Real-data validation + brand selection | ✅ COMPLETE (brand re-decided in M3, see below) |
| 3 | **Intent taxonomy — FROZEN v0.3.0** | ✅ **COMPLETE** |
| 4 | Classifier subsystem (weak labels, dev only) | ✅ COMPLETE — see `CLASSIFIER.md` |
| 5 | **End-to-end agent** (retrieval, generation, grounding, routing) | ✅ COMPLETE — see `AGENT.md` |

### Next milestone (not started)

**Build the intent classifier under TDD against the frozen taxonomy (v0.3.0).** The taxonomy
is frozen: changing it requires a new version and a decision-log entry, and model
performance may never motivate a change.

### Explicitly NOT done yet

- ✅ Intent taxonomy — **FROZEN v0.3.0**, hash `613f5dfec125...`, 10 intents + 2 attributes
- ✅ Intent classifier — built (`tfidf_logreg`); **all dev figures are rule-recovery scores against weak labels, NOT accuracy**
- ✅ Retrieval — hybrid BM25 + embeddings, leakage- and temporally-guarded
- ✅ Reply generation — deterministic evidence template; LLM adapter ready, unconfigured
- ✅ Grounding check — deterministic, runs independently of the generator
- ✅ Escalation policy — implemented, fail-closed, typed reasons
- ✅ End-to-end pipeline — runs on real data; **not evaluated**
- ❌ Baselines — not built
- ❌ **Golden set — not created, no labels exist**
- ❌ Evaluation harness — not built
- ❌ LLM judge — not built
- ❌ Human calibration / judge agreement — not measured
- ❌ Failure analysis — not performed
- ❌ Final report — not written
- ❌ **No LLM API call has ever been made. Zero cost incurred.**

`VERIFICATION.json` is the machine-readable version of this list and is authoritative.

---

## 3. Architecture

### Directory structure

```
support-resolution-engine/
├── AGENTS.md                     # operating manual for coding agents — read first
├── VERIFICATION.json             # machine-readable claim state: what IS and IS NOT demonstrated
├── pyproject.toml                # pytest config, src layout
├── requirements.txt              # pinned dependencies
├── docs/
│   ├── HANDOFF.md                # this file
│   ├── SPEC.md                   # target system specification (authoritative design)
│   ├── CURRENT_STATE.md          # starting position + provenance declaration
│   ├── MILESTONES.md             # per-milestone plan + result + defects found
│   ├── DECISION_LOG.md           # 15 non-obvious decisions (CAPPED — see §10)
│   ├── DATA_PROVENANCE.md        # four-category data separation + language rules
│   └── PUBLIC_REPO_COMPARISON.md # forensic audit of 7 competing public repos
├── src/hiver_support/            # package name is internal; NOT a branding decision
│   ├── kaggle_auth.py            # credential DETECTION (never reads credential values)
│   ├── leakage.py                # 6 leakage guards, all raising
│   └── data/
│       ├── schema.py             # Tweet, Conversation, SupportPair (frozen dataclasses)
│       ├── threads.py            # parse / reconstruct / brand ID / pair extraction
│       ├── normalise.py          # pipeline text normalisation (preserves signal)
│       ├── pii.py                # PII masking before any API boundary
│       ├── reply_classify.py     # deflection / substantive / actionable / language
│       └── split.py              # deterministic temporal split
├── scripts/
│   ├── fetch_data.py             # Kaggle download + schema verification
│   ├── analyse_brands.py         # 13-feature brand profile for every candidate
│   └── select_brand.py           # applies pre-registered criteria, emits artifacts
├── tests/                        # 244 tests
└── reports/                      # generated artifacts (committed; small)
    ├── brand_profiles.json       # 83 brand profiles + provenance
    ├── brand_profiles_all.csv    # same, tabular
    ├── brand_decision.json       # selection decision + thresholds + ranking
    └── brand_selection.md        # human-readable comparison, all 83 brands
```

> **Note on the package name.** The Python package is `hiver_support`. Renaming it is
> explicitly *not* required and should not be done casually — it would touch every import and
> every test for no technical gain. The repository name is the only naming decision that was
> made deliberately.

### Data flow (implemented)

```
data/raw/twcs/twcs.csv  (read-only, never modified)
  → parse_tweets()               normalise ids, parse Twitter dates, drop unusable rows
  → reconstruct_conversations()  union-find over both reply-link directions
  → identify_brand()             most frequent outbound author, deterministic tie-break
  → extract_support_pairs()      anchored on the brand reply
  → temporal_split()             train / dev / test_pool, guard-satisfying by construction
```

### Data flow (specified, NOT built)

```
CUSTOMER MESSAGE
  → normalise / PII-mask
  → INTENT CLASSIFIER          → (intent, confidence)
  → RISK GATE                  → high-risk intent?  → ESCALATE (reason: policy)
  → HISTORICAL RETRIEVAL       (temporal-filtered, leakage-guarded)
      → empty / low confidence → ESCALATE (reason: no_evidence)
  → REPLY GENERATION           (grounded in retrieved evidence only)
  → GROUNDING CHECK            → fail → ESCALATE (reason: ungrounded)
  → EXPECTED-COST DECISION     → AUTO_HANDLE | ESCALATE + typed reason
```

Full contract and invariants: `SPEC.md` §6.

### Evaluation flow (specified, NOT built)

```
locked golden set (200 hand-labelled examples)
  → agent + baseline A (trivial) + baseline B (TF-IDF/BM25), identical harness
  → intent metrics | routing metrics | retrieval metrics
  → LLM judge (cached, versioned prompt, blind to gold and to system identity)
  → human calibration on 40–60 cases → judge–human agreement
  → bootstrap 95% CIs on every headline number
  → risk–coverage curve + expected-cost sweep
  → top-5 failure modes → regression tests
```

---

## 4. Data

### Dataset

| Field | Value |
|---|---|
| Name | Customer Support on Twitter |
| Source | Kaggle: `thoughtvector/customer-support-on-twitter` |
| Local path | `data/raw/twcs/twcs.csv` |
| Size | 493 MB |
| Records | 2,811,774 |
| Committed to git? | **No — deliberately** |

### Expected schema

```
tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
```

`created_at` uses Twitter's format: `Tue Oct 31 22:10:47 +0000 2017`.

**Read it with `dtype=str`.** With inferred dtypes pandas returns `response_tweet_id` as
`float64`, so tweet id `2` becomes `2.0` — see §7.1.

### Why the raw dataset is excluded from GitHub

1. 493 MB is far beyond sensible source control; it would make every clone painful.
2. The corpus is redistributable from Kaggle under its own terms — we should not mirror it.
3. The assignment explicitly says reviewers will not run the full dataset.
4. Two of the seven audited competing repositories committed hundreds of megabytes of raw
   corpus (one committed the same 95 MB file twice). Not repeating that is deliberate.

`tests/test_data_provenance.py` **enforces** this: no bulk data may be tracked, and no tracked
file may exceed 2 MB.

### How another machine obtains it

See §12. Summary: authenticate with your own Kaggle credentials, then
`python scripts/fetch_data.py`. Credentials are never committed.

### Current real-data counts (measured, not estimated)

| Metric | Value |
|---|---|
| Records read | 2,811,774 |
| Conversations reconstructed | 798,197 |
| Customer→support pairs | 1,149,717 |
| Brands present | 108 |
| Brands profiled (≥1,000 pairs) | 83 |
| Rows lost in parsing | 0 |
| Max conversation length | 261 tweets |

### Provenance files

| File | Records |
|---|---|
| `reports/brand_profiles.json` | corpus path, size, SHA-256 of first 64 MB, `corpus_modified: false`, record counts, seed, sample size, analysis git SHA, timestamp, elapsed |
| `reports/brand_decision.json` | selected brand, thresholds, rubric weights, full ranking, `model_performance_used: false`, plus the profile provenance block |
| `VERIFICATION.json` | machine-readable statement of what has and has **not** been demonstrated |
| `docs/DATA_PROVENANCE.md` | four-category separation and the binding language rules |

---

## 5. Brand selection

### Selected: `AppleSupport`

### Pre-registered criteria

Frozen in `SPEC.md` §3.2.3 **before any profile was computed**. Six filters:

| Filter | Threshold |
|---|---|
| `pair_count` | ≥ 5,000 |
| `distinct_intents_at_3pct` | ≥ 6 |
| `usable_grounding_evidence_pairs` | ≥ 1,500 |
| `escalation_sensitive_count` | ≥ 300 |
| `escalation_sensitive_rate` | within (0.01, 0.40) |
| `distinct_months` | ≥ 6 |

Survivors are ranked on five equally-weighted dimensions (0.2 each): volume, intent diversity,
substantive resolutions, retrieval potential, evaluation coverage. **Equal weighting is
deliberate** — choosing weights after seeing profiles is how a rubric becomes a way to justify
a preferred answer.

**5 of 83 brands passed all six filters.**

| Rank | Brand | Score | Pairs | Usable grounding | Intents | RetrNN | EscN |
|---|---|---|---|---|---|---|---|
| **1** | **AppleSupport** | **0.6551** | 105,368 | **9,388** | 7 | 0.387 | 2,689 |
| 2 | AmazonHelp | 0.5610 | 152,903 | 2,645 | 11 | 0.332 | 9,436 |
| 3 | SpotifyCares | 0.4551 | 41,353 | 1,707 | 11 | 0.398 | 1,922 |
| 4 | AskLyft | 0.4009 | 11,403 | 1,686 | 11 | 0.359 | 1,253 |
| 5 | TMobileHelp | 0.3362 | 33,128 | 2,749 | 10 | 0.353 | 2,487 |

### Why AppleSupport

The decisive feature is **usable grounding evidence** (SPEC §3.2.2 feature 13 — the
conjunction of actionable *and* non-duplicate). AppleSupport has **9,388 such pairs, 3.4× the
next brand**. AmazonHelp has 45% more raw pairs but only 2,645 usable: its replies are largely
non-actionable (1.8%) and **16.6% are not in English**. Uber_Support shows why deflection
matters — 55,150 pairs but 60.2% are "DM us", leaving only 507 usable.

### Important tradeoffs (do not omit these from the report)

- **AppleSupport ranks *last* of the five survivors on intent diversity** (7 intents at ≥3%
  versus 11–12 for others). A narrower taxonomy was accepted in exchange for grounding
  evidence. This is a real cost.
- Its DM-deflection rate is 34.5% — high in absolute terms; roughly a third of its replies
  carry nothing groundable. It wins despite this, not because of low deflection.
- Escalation-sensitive volume (2,689) is well below AmazonHelp's 9,436, which slightly limits
  how many hard routing cases the golden set can draw on.

### Selection integrity

Selection used **descriptive corpus statistics only**. No model was trained, no agent was run,
and no downstream performance existed at the time. `reports/brand_decision.json` records
`model_performance_used: false`, and `tests/test_real_data.py` asserts it rather than leaving
it as a prose claim.

**There is no re-selection.** If AppleSupport proves hard, that is a reported finding, not a
reason to switch. Changing brand after seeing results would be a garden-of-forking-paths error
guaranteeing an inflated, non-replicating number. See `DECISION_LOG.md` D14.

Full artifact: `reports/brand_selection.md` (all 83 brands, including every rejected one and
which criteria it failed).

---

## 6. Test status

**325 passing, 3 skipped, 0 failing.**

| Module | Tests | Data |
|---|---|---|
| `test_thread_reconstruction.py` | 23 | synthetic |
| `test_pii_handling.py` | 29 | synthetic |
| `test_normalisation.py` | 31 | synthetic |
| `test_leakage.py` | 32 | synthetic |
| `test_temporal_split.py` | 24 | synthetic |
| `test_data_provenance.py` | 15 | repo/git audit |
| `test_credentials.py` | 20 | synthetic (fake tokens, fake home dirs) |
| `test_reply_classification.py` | 54 | synthetic *(verbatim corpus reply text as literals; reads no file)* |
| `test_real_data.py` | **19** | **real corpus — skipped when absent** |

`tests/test_real_data.py` is the **only** module that reads the corpus. It skips entirely when
the corpus is missing, so a reviewer without 493 MB sees skips, never silent passes.
`test_data_provenance.py` mechanically enforces this boundary — it is the single sanctioned
exception to "no test reads a data file from disk".

### Commands

```bash
pytest                                  # full suite (244)
pytest -m "not slow"                    # skip slow tests
pytest tests/test_real_data.py -v       # real-data validation only
pytest --collect-only -q                # per-file counts
```

---

## 7. Important bugs found and fixed

Each entry records the **resolution**, not just the symptom.

### 7.1 pandas float coercion of tweet ids — *the highest-impact defect*

**Symptom.** With inferred dtypes, pandas reads an integer column containing blanks as
`float64`, so tweet id `2` becomes `2.0`. `str(2.0) == "2.0" != "2"`.

**Impact.** Every reply link in the file breaks. There is no error — the corpus silently
becomes a pile of single-tweet conversations, and every downstream number is quietly wrong.

**Resolution.** `_normalise_id()` in `data/threads.py` coerces via a float check so `1`,
`1.0` and `"1.0"` all become `"1"`. Analysis scripts additionally read with `dtype=str`.
**Confirmed on the real file**, not hypothetical:
`test_ids_are_float_coerced_when_pandas_infers_dtypes`. See `DECISION_LOG.md` D3.

### 7.2 Reply-link reconstruction

**Symptom.** Reply structure lives in two partially redundant columns
(`in_response_to_tweet_id`, `response_tweet_id`), either of which can be missing. Some
malformed rows point at each other, forming cycles; a recursive parent-walk hangs.

**Resolution.** Union-find over **both** link directions — cycle-safe by construction and
recovers threads where only one column is populated. `DECISION_LOG.md` D2.

### 7.3 Fan-out reply lists

**Symptom.** `response_tweet_id` holds a comma-separated list (`"2,3"`) when a reply fans out.
**26,987 occurrences in the first 300k rows alone.**

**Resolution.** `_split_response_ids()` splits and normalises each element. Pinned by
`test_fan_out_reply_lists_exist` on real data.

### 7.4 Orphan roots

**Symptom.** `in_response_to_tweet_id` is absent for conversation roots and for replies whose
parent was not sampled — **76,918 occurrences in the first 300k rows**. A naive implementation
drops these tweets entirely.

**Resolution.** Links pointing outside the supplied set are ignored rather than fatal, so an
orphan becomes a single-tweet conversation instead of vanishing.

### 7.5 Temporal split boundary — *found by a test written specifically to distrust the fixtures*

**Symptom.** Conversations were placed by their **earliest** turn, but threads have duration:
a train thread starting before the cut can end well after it, leaving a resolution in the
retrieval corpus that postdates the question it should precede.

```
temporal leakage: retrieval corpus extends to 2017-01-06T18:00,
at or beyond the earliest golden example 2017-01-05T16:00
```

**Why tests missed it.** Every fixture had one turn per conversation, so no thread *could*
straddle a boundary. Single-turn fixtures are structurally incapable of expressing this bug.

**Resolution.** Boundaries are computed once, before any drop; conversations crossing one are
discarded whole and counted. Regression test: `TestOverlappingConversations`.

### 7.6 Empty dev split — *found by manual inspection, not by any test*

**Symptom.** On overlapping multi-turn data the manifest showed `dev: 0` — an entire split
annihilated by boundary drops, 25% of pairs discarded. Threshold calibration and every metric
would have run to completion against nothing and reported valid-looking numbers.

**Resolution.** `temporal_split()` raises `SplitError` naming the empty split, and `drop_rate`
is surfaced in the manifest because a high rate means threads are long relative to the split
cadence and the retained sample may no longer represent the corpus.
Regression test: `TestDegenerateSplits`.

### 7.7 Reply classification / deflection — *found by manual inspection of real replies*

**Symptom.** The first lexicon matched only explicit "DM" phrasings, so real deflections were
counted as substantive resolutions:

```
[deflection=0 substantive=1] a canned "contact us directly at [URL]" brand reply (tweet `227421`)
[deflection=0 substantive=1] "please request a callback here: [URL]"
[deflection=0 substantive=1] "Send us a note here, [URL] and our team will be in touch."
```

**Impact.** Inflated `substantive_resolution_rate` and deflated `dm_deflection_rate` — on the
exact feature the brand-selection rubric depends on most.

**Resolution.** Extracted into the tested module `data/reply_classify.py`, with **verbatim
corpus replies as test cases**. The obvious over-correction was explicitly rejected: treating
*any* redirect phrase as deflection would discard replies that redirect **and** inform
(an explanation that pending charges may be authorisations; tweet `74659`), throwing away real grounding
evidence. A redirect makes a reply a deflection only when the reply carries no information of
its own. Language detection was added in the same pass — AmazonHelp is 16.6% non-English.

**Note on adjudication.** Two replies initially labelled deflections were classified as
substantive by the module. On review the module was right ("The Echo Show is supported, please
reach us..." answers the question before redirecting), so **the test labels were corrected, not
the code**. The failing tests were measuring a hasty judgement, not a defect.

### 7.8 Provenance manifest drift

**Symptom.** `VERIFICATION.json` declared a test count that no longer matched the suite after
new tests were added.

**Resolution.** `test_manifest_lists_exactly_the_test_files_that_exist` asserts the manifest
lists exactly the test files present. It caught its own drift on the first run and has caught
every subsequent one. **Update `VERIFICATION.json` in the same commit as any change to the
claim state.**

---

## 8. Evaluation methodology

Preserved from `SPEC.md`. **Everything in this section is PLANNED. None of it is built.**

### What "good" means (SPEC §1.2)

Not accuracy — **expected cost**. Support operates under asymmetric loss: a bad auto-reply to
a billing dispute costs far more than an unnecessary escalation. The objective is to maximise
safely-absorbed volume subject to a bound on harm:

```
E[cost] = C_bad · P(wrong ∧ auto-handled) · |A| + C_human · |¬A|
```

`C_bad / C_human` is **unknowable from tweets**, so it is **swept (2, 4, 8, 12, 20), never
asserted**. The headline routing deliverable is a **risk–coverage curve**, not a single number.

### Golden set (SPEC §9) — PLANNED

- **200 examples, hand-labelled by a human.** Target confirmed with the project owner,
  including a self-agreement recheck.
- Stratified over intent, difficulty, thread length, time bin; hard strata over-sampled.
- Drawn **only** from the held-out test pool.
- **160 suggested / 40 blind** — comparing human-vs-suggestion agreement between the two groups
  measures anchoring bias.
- Every labelling action logged so override rate is computable.
- ≥40 examples re-labelled after ≥24h to measure intra-annotator agreement.
- Then **locked and hash-pinned**.

> **Critical framing rule.** Self-agreement measures **annotation consistency (reliability)**,
> **not** label accuracy and **not** a ceiling on achievable model performance. A consistent
> annotator can be consistently wrong. Do **not** write "ceiling on achievable accuracy". The
> defensible claim is narrower: intents whose labels are unstable on re-labelling carry
> uncertainty beyond the sampling error in the confidence interval. See `DECISION_LOG.md` D12.

> **Honesty constraint.** If labelling cannot reach the target, report the size actually
> labelled and the CIs it implies. **Never backfill with model labels.** The harness must
> refuse to score unlabelled columns.

### Baselines (SPEC §11) — PLANNED

- **A (trivial):** majority intent; always-escalate; canned reply. Always-escalate is a
  *strong* safety baseline (zero false auto-handles) and must be presented as such, not
  strawmanned. If the system cannot beat it on expected cost, say so.
- **B (simple ML):** TF-IDF + logistic regression for intent; BM25/TF-IDF nearest historical
  reply for generation.

Both evaluated on the identical locked golden set through the identical harness.

### Metrics (SPEC §12) — PLANNED

- **Intent:** accuracy, macro-F1, per-intent P/R, confusion matrix.
- **Routing:** escalation precision/recall, **false auto-handle rate** (safety-critical),
  false escalation rate, risk–coverage curve, expected cost across the sweep.
- **Reply:** groundedness, relevance, resolution match, actionability, tone.
- **Retrieval:** recall@k and MRR **against human relevance judgements** — never against a
  signal the reranker itself optimises. One audited public repo scored the reranker's own sort
  key and reported the mathematically impossible `recall@1 == recall@3 == recall@5 == MRR`.
- **Annotation:** intra-annotator self-agreement; anchoring bias (blind vs suggested).
- **Bootstrap 95% CIs on every headline number.** At n=200 the CI on accuracy near 0.9 is
  roughly ±4 points; a bare point estimate at this sample size is misleading.

### LLM-as-judge (SPEC §13) — PLANNED

Structured schema, versioned prompt, cached responses (model, prompt version, input hash,
output, parse status). Judge does **not** see the gold label or which system produced a reply.
Position/verbosity bias controlled by randomising presentation order.

### Human calibration — PLANNED

40–60 golden cases scored independently by a human. Report exact agreement, within-1
agreement, Spearman, weighted kappa. **If judge–human agreement is poor, that is a headline
result, not a footnote** — it bounds how far any reply-quality number can be trusted.

### Anti-circularity (SPEC §7.3) — IMPLEMENTED

Enforced structurally, not by convention: the pre-annotator must not share a model family with
the system under test, and the judge must not share one with the generator.
`leakage.assert_independent_models()` **raises**. This is the guard that would have caught the
worst failure found in the audited field.

### "What is misleading about my headline number?" — PLANNED, MANDATORY

Grounded in measured quantities, not generic caveats. Already-known material: annotator
self-agreement, judge–human agreement, anchoring bias, escalation-rate skew, the 34.5%
deflection rate of the chosen brand, and the intent-diversity tradeoff accepted at selection.

### Failure analysis — PLANNED

Top-5 modes from **actual** evaluation output. Each: frequency, real example, expected vs
actual, root cause, why tests missed it, fix, **new regression test**. Do not manufacture
failure modes before running the system.

---

## 9. Provenance and integrity

| Field | Value |
|---|---|
| Branch | `main` |
| Random seed (analysis) | `20260910` |
| Corpus SHA-256 (first 64 MB) | `6de454c514d54f1b2b994a04c0413aef...` (full value in `brand_profiles.json`) |
| Corpus modified | **false** |
| LLM API calls made | **0** |
| Cost incurred | **$0.00** |

Current git SHA: see `git log -1 --format=%H`. The analysis git SHA at profile-generation time
is recorded inside `reports/brand_profiles.json`.

### Public-repository forensics — material to provenance

Seven public repositories implementing this same assignment were cloned and audited
file-by-file (`docs/PUBLIC_REPO_COMPARISON.md`). Findings that matter here:

- **All seven were created within a five-hour window** on the same day — they are concurrent
  submissions by competing candidates, not prior art.
- **Six of seven carry no license**, which under default copyright means all rights reserved.
  The seventh is MIT.
- **No code, data, labels, metrics, report text, or decision-log content was copied from any
  of them, including the MIT-licensed one.** Copying a direct competitor's submission is an
  integrity problem independent of licensing.
- Ideas and architectural patterns adopted are **enumerated and cited** in
  `PUBLIC_REPO_COMPARISON.md` §5–6, `VERIFICATION.json` (`ideas_adopted_with_citation`), and
  `DECISION_LOG.md` D1.
- The clones live in a session scratchpad **outside this repository** and were never copied in.
  `tests/test_data_provenance.py` asserts the repository contains no such artifacts.

**If you continue this project, preserve this position.** Do not vendor competitor code.

---

## 10. Decision log

**`docs/DECISION_LOG.md` is the authoritative decision history.** 15 entries, written as
decisions were made rather than reconstructed afterwards.

**The log is CAPPED at 15**, matching the assignment's 10–15 requirement. Do **not** append a
sixteenth for routine implementation choices — those belong in module docstrings and
`MILESTONES.md`. A genuinely material architectural, methodological, evaluation or scope
decision is recorded by **consolidating it into the existing entry it belongs with**.

Entries most likely to matter next: **D12** (self-agreement framing), **D13** (deterministic
before paid), **D14** (frozen brand criteria, no re-selection), **D1** (no competitor code).

---

## 11. File map

| File | Purpose |
|---|---|
| `AGENTS.md` | Operating manual for coding agents — read before changing anything |
| `VERIFICATION.json` | Machine-readable claim state; **update in the same commit as any claim change** |
| `docs/HANDOFF.md` | This document |
| `docs/SPEC.md` | Authoritative target-system design, 17 sections |
| `docs/CURRENT_STATE.md` | Starting position, provenance declaration |
| `docs/MILESTONES.md` | Per-milestone plan, result, and defects found |
| `docs/DECISION_LOG.md` | 15 non-obvious decisions (capped) |
| `docs/DATA_PROVENANCE.md` | Four-category data separation, binding language rules |
| `docs/PUBLIC_REPO_COMPARISON.md` | Forensic audit of 7 competing repos |
| `src/hiver_support/leakage.py` | 6 leakage guards; all raise, never warn |
| `src/hiver_support/kaggle_auth.py` | Credential *detection* — returns mechanism name, never a value |
| `src/hiver_support/data/schema.py` | `Tweet`, `Conversation`, `SupportPair` (frozen) |
| `src/hiver_support/data/threads.py` | Reconstruction pipeline |
| `src/hiver_support/data/normalise.py` | Pipeline normalisation (preserves casing/emoji/punctuation) |
| `src/hiver_support/data/pii.py` | PII masking; over-masking treated as a failure |
| `src/hiver_support/data/reply_classify.py` | Deflection / substantive / actionable / language |
| `src/hiver_support/data/split.py` | Deterministic temporal split + manifest |
| `scripts/fetch_data.py` | Kaggle download + schema verification |
| `scripts/analyse_brands.py` | 13-feature profile per brand (~8 min full corpus) |
| `scripts/select_brand.py` | Applies frozen criteria, emits all artifacts |
| `reports/brand_selection.md` | Human-readable comparison, all 83 brands |
| `reports/brand_decision.json` | Decision, thresholds, ranking, integrity flags |

---

## 12. How to resume on a fresh machine

```bash
# 1. Clone
git clone https://github.com/NITISH-R-G/support-resolution-engine.git
cd support-resolution-engine

# 2. Dependencies (Python 3.11+; developed on 3.14.3)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -r requirements.txt

# 3. Kaggle authentication — NEVER commit credentials.
#    Easiest (recommended by the Kaggle CLI itself):
kaggle auth login
#    Or generate a token at https://www.kaggle.com/settings/api and either:
#      - save it to ~/.kaggle/access_token   (Windows may append .txt — supported)
#      - or export KAGGLE_API_TOKEN=...
#    Five mechanisms are supported; see src/hiver_support/kaggle_auth.py.
#    Verify detection without revealing any value:
python -c "import sys; sys.path.insert(0,'src'); from hiver_support.kaggle_auth import credential_source; print(credential_source() or 'NONE')"

# 4. Obtain the dataset (~493 MB, one time; lands in data/raw/, git-ignored)
python scripts/fetch_data.py
python scripts/fetch_data.py --check     # verify an existing copy

# 5. Run the full suite (244 tests; real-data tests skip if the corpus is absent)
pytest

# 6. Verify environment and claim state
cat VERIFICATION.json
git log --oneline | head -5

# 7. Inspect current state
cat reports/brand_selection.md          # brand decision + all 83 profiles
cat docs/MILESTONES.md                  # what was done and what broke

# 8. OPTIONAL — reproduce the brand analysis (~8.5 min full corpus)
python scripts/analyse_brands.py        # or --limit 300000 for a quick pass
python scripts/select_brand.py
```

Expected after step 5: **244 passed** with the corpus present; on a fresh clone without the
corpus, 226 passed + 18 skipped (a skip is a skip — never report it as a pass).
it. A skip is a skip — never report it as a pass.

---

## 13. Next action

> **Next milestone: derive and freeze the AppleSupport intent taxonomy from the real data,
> then build the classifier under TDD.**

Procedure (SPEC §5): embed a sample of AppleSupport customer messages → cluster → a human reads
cluster exemplars → names and merges clusters → writes a codebook with inclusion/exclusion
rules and boundary cases → **freeze and version-hash the taxonomy before any golden-set
labelling begins**.

Requirements: 6–10 intents plus explicit `other`/`unclear`; each grounded in ≥20 real examples;
documented tie-breaks for known-ambiguous pairs; **routing-useful** — merge two intents that
always route identically and are always handled identically. The taxonomy exists to drive a
decision, not to be a taxonomy. Deliverable: `docs/TAXONOMY.md`.

### What must NOT happen before that milestone is complete

- ❌ Do **not** build the classifier before the taxonomy is frozen and hash-pinned.
- ❌ Do **not** start the golden set — its labels depend on the frozen taxonomy. Changing the
  taxonomy after labelling invalidates the golden set.
- ❌ Do **not** train models or run evaluation.
- ❌ Do **not** make LLM API calls without first presenting provider, model, role, expected call
  count and a cost estimate to the project owner.
- ❌ Do **not** re-open brand selection.
- ❌ Do **not** modify evaluation methodology.

---

## 14. Agent rules

Binding on anyone continuing this project.

**Development methodology**
1. **TDD, non-negotiable.** Write the failing test first; confirm it fails for the *right
   reason*; then implement. Never write implementation first and add tests afterwards.
2. **One logical change at a time.** Do not stack unfinished work on unfinished work.
3. **Run the full regression suite after every meaningful change.**
4. **Inspect actual output**, not just green tests. Two of the defects in §7 were invisible to
   a passing suite and were found only by reading real output.
5. **Every discovered bug becomes a permanent regression test.**
6. **Stop at milestone boundaries for review.**

**Evaluation integrity — the project's core value**
7. **No fabricated metrics.** Ever. If something was not measured, say it was not measured.
8. **No fake human labels.** A script that applies keyword rules is not a human annotator. One
   audited repo shipped exactly that in a file named `human_review_pass.py`.
9. **No leakage.** Guards raise, never warn. Do not weaken or disable one to make a pipeline
   pass — a disabled guard is worse than none because it still reads as protection.
10. **No benchmark contamination.** The model that pre-annotates must never be the model under
    test; the judge must never share a family with the generator.
11. **No changing criteria after seeing results.** Thresholds are fitted on dev, never on the
    locked test set. Brand selection is frozen.
12. **No metric without provenance.** Every reported number carries seed, data version, code
    version, and what it was computed against.
13. **Prefer underclaiming.** When a statistic does not support a claim, make the narrower
    claim. Reliability is not validity.
14. **Report weaknesses.** A lower number that means something beats a higher number that does
    not.

**Data handling**
15. **Never commit the raw corpus, credentials, tokens, `.env` files, caches, or venvs.**
16. **Never modify `data/raw/`.** It is read-only input.
17. **Keep the fixture/real-data boundary explicit.** `tests/test_real_data.py` is the only
    module permitted to read the corpus.
18. **Never copy competitor code, data, labels, or report text.**

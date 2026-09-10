# Milestone Log

> **Claim state (current: 2026-09-10).** Milestones 1–1d were produced on **synthetic,
> in-memory fixtures**; their result blocks record the suite size *at that time* and are left
> unedited as historical record. **Milestone 2 onward uses the real corpus**, which has been
> downloaded, schema-verified and validated. The current suite is **325 passed, 3 skipped**, of
> which 19 are real-data tests that skip when the corpus is absent — a skip is never a pass.
> `../VERIFICATION.json` and `DATA_PROVENANCE.md` are authoritative over any prose here.

Each milestone records its plan before implementation and its result after. A milestone is complete
only when its acceptance criteria pass, the full regression suite is green, and output has been
manually inspected.

---

## MILESTONE 1 — Project scaffold + conversation reconstruction

**Goal:** A tested project skeleton that turns raw TWCS tweet rows into reconstructed conversation
threads and per-brand customer→support pairs.

**SPEC:** `SPEC.md` §3.1 (pipeline: raw tweets → thread reconstruction → brand extraction →
customer/support pairing).

The TWCS schema is `tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id`. Reconstruction must handle the corpus's real quirks: `response_tweet_id`
is a comma-separated list; `in_response_to_tweet_id` is frequently absent; replies can be orphaned
(pointing at a tweet not present in a subsample); ids appear as both ints and strings; and
malformed rows can form cycles.

**ACCEPTANCE CRITERIA:**
1. Raw rows parse into a typed `Tweet` model, with TWCS date format and multi-valued
   `response_tweet_id` handled.
2. Conversations are reconstructed by reply-linkage; every tweet belongs to exactly one
   conversation; tweets within a conversation are chronologically ordered.
3. Cyclic reply links terminate rather than hang.
4. Orphan replies (dangling `in_response_to`) become their own conversation root, not a crash.
5. Brand is identified per conversation from the non-inbound author; multi-brand and brand-less
   conversations are handled explicitly.
6. Customer→support pairs are extracted with preceding turns as context; a pair always has an
   inbound customer tweet and a following outbound brand reply.
7. Subsampling is deterministic under a fixed seed.

**TEST PLAN:** `tests/test_thread_reconstruction.py`, `tests/test_brand_filtering.py` — unit tests
per criterion plus adversarial fixtures for cycles, orphans, mixed id types, and empty input.

**FILES EXPECTED TO CHANGE:** `pyproject.toml`, `requirements.txt`, `src/hiver_support/__init__.py`,
`src/hiver_support/config.py`, `src/hiver_support/data/schema.py`,
`src/hiver_support/data/threads.py`, `tests/*`.

**RISKS:** Assuming a schema without seeing real rows — mitigated by validating against the real
Kaggle file before declaring the milestone complete.

**DONE WHEN:** all criteria pass, suite green, and reconstruction verified by eye on real rows.

**RESULT:**

```
Tests:      23 passed, 0 failed  (tests/test_thread_reconstruction.py)
Regression: PASS (52 passed overall after PII work)
Manual:     PENDING — blocked on dataset access
```

| # | Criterion | Status |
|---|---|---|
| 1 | Typed parse, TWCS dates, multi-valued response ids | PASS |
| 2 | Reply-linked, exactly-once membership, chronological | PASS |
| 3 | Cycles terminate | PASS |
| 4 | Orphan replies survive as own conversation | PASS |
| 5 | Brand identification incl. multi-brand + brand-less | PASS |
| 6 | Pair extraction with context | PASS |
| 7 | Deterministic subsampling | NOT STARTED — needs real data |

**RED/GREEN/REFACTOR:**
- *RED:* `ModuleNotFoundError: hiver_support.data.schema` — contract defined before implementation.
- *GREEN:* `schema.py` + `threads.py`; union-find grouping, id normalisation, reply-anchored pairing.
- *REFACTOR:* replaced a `pd.concat` fixture that emitted a pandas `FutureWarning`; suite is
  warning-clean.

**Non-obvious decisions made here** (for the decision log):
1. **Union-find rather than a recursive parent-walk.** The corpus contains reply cycles; a walk
   either hangs or needs ad-hoc cycle detection. Union-find is cycle-safe by construction.
2. **Ids normalised through a float check.** pandas reads an int column containing blanks as
   floats, so tweet id `1` becomes `1.0` and `"1.0" != "1"` silently breaks every reply link.
   This is the single highest-impact bug in this module and it is invisible without a test.
3. **Pairs anchored on the brand reply, not the customer message.** Brands routinely send
   consecutive tweets; anchoring on the reply yields one pair instead of duplicating the
   customer turn across two.
4. **Conversation id = hash of sorted membership**, so it is stable under row order and defined
   even for cyclic threads that have no root.

**KNOWN ISSUES:** criterion 7 and real-data validation are blocked on Kaggle credentials. The
schema was written against the documented TWCS columns and must be re-validated against the real
file before Milestone 2 builds on it.

**DECISION AT THE TIME: NOT COMPLETE** — logic green, real-data validation outstanding.

> **STATUS UPDATE (closed in Milestone 2, 2026-09-10):** the corpus was downloaded and
> reconstruction validated on real rows — 0 rows lost, exactly-once membership,
> chronological ordering, threads up to 261 tweets. Criterion 7 is satisfied and the
> float-coercion assumption behind D3 was **confirmed** in the real file.
> **MILESTONE 1 IS COMPLETE.** The record above is left unedited deliberately: it shows
> what was known at the time rather than being retrofitted to the outcome.

---

## MILESTONE 1b — PII masking (brought forward)

Pulled ahead of its place in the order because it is schema-independent and therefore unblocked
by the dataset gap, and because it guards the boundary to third-party APIs.

**SPEC:** `SPEC.md` §3.4.

**ACCEPTANCE CRITERIA:** mask emails, phones, order ids, tracking numbers, card numbers; do **not**
mask ordinary support text; idempotent; auditable report; loud failure on non-string input.

**RESULT:**

```
Tests:      29 passed, 0 failed  (tests/test_pii_handling.py)
Regression: 52 passed, 0 failed
Manual:     PASS — inspected on 5 hand-written synthetic messages (NOT real TWCS rows)
```

All criteria PASS. Manual inspection (synthetic inputs) confirmed `iPhone 7`, `iOS 11.0.1`, `$9.99`, `Oct 31 2017`,
`2 weeks` and `5 star` survive untouched while every PII class is masked.

**Non-obvious decision:** over-masking is treated as a *failure*, not as caution. Masking "iPhone 7"
or "$9.99" would destroy the exact features the intent classifier depends on, so phone detection
validates digit count (7–15) rather than matching any digit run, and hash-prefixed order references
must contain a digit so ordinary uppercase hashtags survive. `TestDoesNotOverMask` pins this.

**DECISION: COMPLETE**

---

## MILESTONE 1c — Leakage guards (brought forward)

Pulled ahead because it is schema-independent, and because it is the module that makes the
field's dominant failure mode structurally impossible here rather than merely discouraged.

**SPEC:** `SPEC.md` §4, §7.3.

**ACCEPTANCE CRITERIA:**
1. Guards for id overlap (pair / conversation / customer), normalised-text duplicates,
   near-duplicates, response leakage, and temporal ordering.
2. A model-independence guard: judge must not share a family with the generator; pre-annotator
   must not share a family with the system under test.
3. Every guard **raises**; each is demonstrated failing on a purpose-built violating fixture.
4. Aggregate runner reports all violations at once, not just the first.
5. Guards stay silent on clean data (no false positives on canned brand replies).

**RESULT:**

```
Tests:      32 passed, 0 failed  (tests/test_leakage.py)
Regression: 84 passed, 0 failed  (2.9s)
Manual:     PASS — all five data guards fired together with actionable messages
```

| # | Criterion | Status |
|---|---|---|
| 1 | Five data guards | PASS |
| 2 | Model-independence guard | PASS |
| 3 | Each guard observed to raise | PASS |
| 4 | Aggregate reporting | PASS |
| 5 | No false positives on canned replies | PASS |

**RED/GREEN/REFACTOR:**
- *RED:* `ModuleNotFoundError: hiver_support.leakage`.
- *GREEN:* `src/hiver_support/leakage.py`.
- *REFACTOR:* none needed; sklearn import made lazy so the cheap guards run without the ML stack.

**Non-obvious decisions:** see `DECISION_LOG.md` D8 (response leakage defined as question+answer,
because a reply-only check fires constantly on canned replies and would get disabled), D9 (customer
id overlap is leakage), D10 (model-family independence, not name equality), D11 (aggregate and
raise).

**Verification against the audited failures:** the model guard raises on exactly the shubham
configuration (gold prefilled by the model under test); the near-duplicate and id guards raise on
the paraphrase/duplicate case. Both failure modes are now unreachable without an explicit,
loud test failure.

**DECISION: COMPLETE**

---

## MILESTONE 1d — Text normalisation + deterministic temporal splitting

**Goal:** Pipeline-grade text normalisation, and a split that satisfies every leakage guard by
construction rather than by inspection.

**SPEC:** `SPEC.md` §3.3, §3.5, §4.

**ACCEPTANCE CRITERIA:**
1. Normalisation removes noise (URLs, foreign handles, HTML entities, encoding artefacts) and
   preserves signal (casing, punctuation, emphasis, hashtags, emoji); brand handle survives.
2. Split is temporal, groups by conversation *and* customer, and is deterministic under input
   reordering.
3. Split output passes `run_all_checks` for train-vs-test and train-vs-dev.
4. Discards are counted and explained in a JSON-serialisable manifest.
5. Degenerate splits fail loudly.

**RESULT:**

```
Tests:      31 normalisation + 24 split = 55 passed, 0 failed
Regression: 137 passed, 0 failed (2.9s)
Manual:     PASS — manifests inspected on single-turn and overlapping multi-turn fixtures
```

All five criteria PASS.

**RED/GREEN/REFACTOR — two defects found here, both worth recording:**

1. *RED (real defect).* `test_overlapping_threads_still_satisfy_the_temporal_guard` failed:
   `retrieval corpus extends to 2017-01-06T18:00, at or beyond the earliest golden example
   2017-01-05T16:00`. Conversations were placed by their **earliest** turn, but a thread has
   duration — one starting before the cut can end well after it, leaving a resolution in the
   corpus that postdates the question it should precede. An index-based cut does not deliver
   strict temporal ordering.
   *GREEN:* boundaries are computed once, before any drop, and conversations crossing one are
   discarded whole and counted.
   **Why existing tests missed it:** every fixture had one turn per conversation, so no thread
   could straddle anything. Single-turn fixtures cannot express this bug.

2. *Found by manual inspection, not by any test.* On the overlapping fixture the manifest showed
   `dev: 0` — an entire split annihilated by boundary drops, with 25% of pairs discarded. Every
   downstream step would have run to completion against nothing and reported numbers that looked
   valid. Now raises `SplitError` naming the empty split, and `drop_rate` is surfaced in the
   manifest because a high rate means threads are long relative to the split cadence and the
   retained sample may no longer represent the corpus.

Both are now permanent regression tests (`TestOverlappingConversations`, `TestDegenerateSplits`).

**Fixture defect also fixed:** templated text (`"issue number 11"` vs `"issue number 111"`) is a
genuine char-level near-duplicate at cosine 0.905, so the near-duplicate guard fired on the
fixture rather than on the splitter. The guard was right; the fixture was unrealistic. Fixtures
now draw from a varied vocabulary.

**No new decision-log entries** — these are implementation defects, not material decisions, and
the log is at its 15-entry cap.

**DECISION: COMPLETE**

---

## MILESTONE 2 — Real-data analysis and brand selection

**Goal:** Validate reconstruction against the real corpus, compute the frozen brand-selection
profile for every candidate, and select a brand using the pre-registered criteria only.

**SPEC:** `SPEC.md` §3.1–3.2.

**RESULT:**

```
Tests:      244 passed, 0 failed (19 real-data validation tests added)
Regression: PASS
Manual:     PASS — reply classifier inspected on real replies across 3 brands;
                   reconstructed pairs inspected by eye
Corpus:     2,811,774 records | 798,197 conversations | 1,149,717 pairs | 108 brands
Selected:   AppleSupport (5 of 83 profiled brands passed all six filters)
```

**Milestone 1's outstanding criterion is now closed.** Reconstruction was validated on real
rows: 0 rows lost, exactly-once conversation membership, chronological ordering, multi-turn
threads up to 261 tweets, and every extracted pair has an inbound question followed by an
outbound reply from the identified brand.

**Real-data confirmation of D3.** `response_tweet_id` genuinely arrives as `float64` under
pandas' inferred dtypes, so tweet id 2 reads as `2.0`. The id-normalisation decision was not
hypothetical; without it every reply link in the file breaks silently.
Pinned by `test_ids_are_float_coerced_when_pandas_infers_dtypes`.

**Defect found by manual inspection, not by tests.** The first reply-classification lexicon
matched only explicit "DM" phrasings, so real deflections were counted as substantive
resolutions:

```
[deflection=0 substantive=1] a canned "contact us directly at [URL]" brand reply (tweet `227421`)
[deflection=0 substantive=1] "please request a callback here: [URL]"
[deflection=0 substantive=1] "Send us a note here, [URL] and our team will be in touch."
```

This inflated `substantive_resolution_rate` and deflated `dm_deflection_rate` — on the exact
feature the selection rubric depends on. Classification was extracted into the tested module
`hiver_support.data.reply_classify`, whose tests use verbatim corpus replies. The naive
over-correction (treat any redirect phrase as deflection) was rejected too: a reply can
redirect *and* inform, and discarding those would throw away real grounding evidence.

Language detection was added in the same pass: AmazonHelp's replies are 16.6% non-English,
which would otherwise have been counted as English resolutions.

**Two test labels adjudicated against the code, not the reverse.** Two replies I had labelled
pure deflections during inspection were classified as substantive by the module. On review the
module was right — a reply confirming a device is supported before redirecting (tweet `639`) answers the question before
redirecting — so the labels were corrected. The failing tests were measuring a hasty judgement,
not a defect.

**Selection integrity.** Criteria were frozen in `SPEC.md` §3.2.3 before any profile existed;
inputs are descriptive corpus statistics only; no agent was run and no model trained. Rubric
weights are equal by construction, since choosing weights after seeing profiles is how a rubric
becomes a way to justify a preferred answer. `test_real_data.py` asserts
`model_performance_used is False` rather than leaving it as a prose claim.

**Provenance.** `reports/brand_profiles.json` and `brand_decision.json` carry corpus path,
size, SHA-256 prefix, `corpus_modified: false`, record counts, seed, sample size, analysis git
SHA and timestamp. The raw corpus was opened read-only and is unmodified.

**DECISION: COMPLETE** — stopping here for review before taxonomy work, as instructed.

---

## MILESTONE 3 — AppleSupport intent taxonomy (FROZEN)

**Result:** taxonomy **FROZEN at v0.3.0**, hash
`613f5dfec1253168c8f2d01db141923c9e41363b6158c79bab6f067df4a9ee4d`, approved 2026-09-10 after
three review rounds.

```
Tests:      325 passed, 3 skipped, 0 failed
Data:       AppleSupport train split only, n=60,817 English. Dev/test/golden never inspected
Outcome:    10 intents + 2 orthogonal attributes
```

**Round 1 (rejected).** Candidate merged `account_access` into security *because security was
rare*. Correctly rejected: that optimises the label set for measurement convenience, not for
what support must do.

**Round 2.** Replaced the reasoning with an empirical test — *does support actually handle
these differently?* — using reply behaviour with bootstrap CIs and no verdict below n=30.
Found `battery_vs_update` has **zero** significant differences, killing `software_update_issue`
as a causal-attribution label; and `repair_order_replacement` as the one unambiguous boundary
(0.221–0.288).

**Round 3.** Two defects in `reply_classify` found while building handling profiles —
stem-only verbs missed "restarting", and the English gate rejected terse navigation
instructions, systematically discarding the most actionable replies. Fixing them moved the
brand-selection inputs materially (brands passing filters 5 → 27) and produced an exact tie.
Re-decided by lexicographic tie-break on criterion 1, absolute usable grounding evidence
(31,241 vs 20,076) → AppleSupport, independently.

**Security became an attribute, not an intent**, on measurement: **306 of 340 (90%)**
security-sensitive messages sit outside the account topic, so an account-shaped label would
have captured a tenth of the safety signal.

**Defects found during this milestone**

| Defect | How found | Resolution |
|---|---|---|
| `restart` missed "restarting" | Building handling profiles | Verb stems take `\w*` |
| English gate rejected terse instructions | Same | Strong-marker fallback; the bias ran against actionable replies |
| Bootstrap declared significance on n=2 | Own test | `MIN_GROUP_FOR_VERDICT = 30` |
| `confusions=( "x")` was a **string**, not a tuple | Tie-break coverage test | Added trailing comma; it had been iterating characters |
| Manifest conflated collected with passing | Provenance test | `total_collected` = passing + skipped |
| `nohup &` reported success while the job died | Comparison showed impossible zero deltas | Re-ran tracked; nearly reported a false "no impact" |

**Honest limitation carried forward:** 13 of 36 label pairs show no material handling
difference. Deflection is 0.28–0.55 for every label — the 2017 Twitter channel was
deflection-dominated and the instrument has low power. Handling similarity is treated as
absence of evidence, not evidence of absence; most boundaries rest on resolution-evidence type
and escalation policy. **This goes in the report's "misleading headline number" section.**

**DECISION: COMPLETE — taxonomy frozen.**

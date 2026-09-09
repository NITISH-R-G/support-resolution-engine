# Milestone Log

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

**DECISION: NOT COMPLETE** — logic is green, real-data validation outstanding.

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
Manual:     PASS — inspected on 5 realistic messy messages
```

All criteria PASS. Manual inspection confirmed `iPhone 7`, `iOS 11.0.1`, `$9.99`, `Oct 31 2017`,
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

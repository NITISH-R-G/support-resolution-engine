# AGENTS.md — Operating Manual

Machine-readable operating manual for coding agents working on this repository.

---

## STOP — read these first

Before making **any** change, read in this order:

1. **`docs/HANDOFF.md`** — canonical continuation document. Start here. It alone should be
   enough to continue the project correctly.
2. **`docs/SPEC.md`** — authoritative target-system design.
3. **`docs/CURRENT_STATE.md`** — starting position and provenance declaration.
4. **`docs/MILESTONES.md`** — what was done per milestone, and what broke.
5. **`docs/DATA_PROVENANCE.md`** — the four data categories and binding language rules.
6. **`docs/DECISION_LOG.md`** — 15 non-obvious decisions and their reasoning.

Also read **`VERIFICATION.json`** — the machine-readable statement of what has and has **not**
been demonstrated. It is authoritative over any prose claim, including in this file.

There is no `CLAUDE.md` in this repository. If one is added later, it must be reconciled with
this file rather than contradicting it; on conflict, `docs/HANDOFF.md` wins.

---

## Project in one paragraph

A **support resolution engine**: classify an incoming customer message, draft a reply grounded
in how the brand historically resolved similar issues, and decide `AUTO_HANDLE` vs `ESCALATE`
with a stated reason. Built on the public *Customer Support on Twitter* corpus, single brand
(**AppleSupport**, selected by pre-registered criteria). It originated as a take-home whose
explicit emphasis is *"the proof is worth more than the system"* — **evaluation integrity
outranks model performance throughout.**

---

## Current state (2026-09-10)

| | |
|---|---|
| Milestone complete | **2 — real-data validation + brand selection** |
| Milestone next | **AppleSupport intent taxonomy, then classifier under TDD** |
| Tests | **244 passing** (19 real-data, skipped without the corpus) |
| Corpus | 2,811,774 records → 798,197 conversations → 1,149,717 pairs → 108 brands |
| Brand | **AppleSupport** (5 of 83 passed all six filters) |
| Golden set | **Not created.** No labels exist |
| Models | **None trained. Zero LLM API calls. $0.00 spent** |

---

## Hard rules

### Evaluation integrity
- **No fabricated metrics.** If it was not measured, say so.
- **No fake human labels.** A keyword script is not an annotator.
- **No leakage.** Guards raise, never warn. Do not weaken one to make a pipeline pass.
- **No benchmark contamination.** Pre-annotator ≠ model under test; judge family ≠ generator
  family. Enforced by `leakage.assert_independent_models()`.
- **No changing criteria after seeing results.** Thresholds fit on dev, never on the locked
  test set. Brand selection is frozen — no re-selection.
- **No metric without provenance** — seed, data version, code version, and what it scored.
- **Prefer underclaiming.** Reliability is not validity. Self-agreement is *annotation
  consistency*, never a "ceiling on achievable accuracy" (`DECISION_LOG.md` D12).

### Development
- **TDD is non-negotiable.** Failing test first; confirm it fails for the right reason; then
  implement. Never retrofit tests.
- **One logical change at a time.** Do not stack unfinished work.
- **Full regression suite after every meaningful change.**
- **Inspect real output, not just green tests.** Two of the worst defects in this project were
  invisible to a passing suite (`HANDOFF.md` §7.5, §7.6).
- **Every bug becomes a permanent regression test.**
- **Stop at milestone boundaries for review.**

### Data
- **Never commit** the raw corpus, credentials, tokens, `.env`, caches, or virtualenvs.
- **Never modify `data/raw/`** — read-only input.
- **`tests/test_real_data.py` is the only module permitted to read the corpus.** Enforced by
  `tests/test_data_provenance.py`.
- **Never copy competitor code, data, labels, or report text.** Seven competing repositories
  were audited for methodology; six carry no license. Ideas are cited, code is not vendored.

### Claim hygiene
- **Update `VERIFICATION.json` in the same commit** as any change to the claim state.
- Permitted: *"225 tests pass on synthetic fixtures"*, *"validated against the real corpus"*
  (only where true).
- Forbidden: presenting fixture behaviour as a property of the dataset; reporting a skipped
  test as a pass.
- **`docs/DECISION_LOG.md` is capped at 15 entries.** Consolidate into an existing entry;
  do not append a sixteenth for routine choices.

---

## Commands

```bash
pytest                              # full suite (244)
pytest tests/test_real_data.py -v   # real-data validation only
python scripts/fetch_data.py        # obtain corpus (needs your own Kaggle credentials)
python scripts/analyse_brands.py    # brand profiles (~8.5 min full corpus)
python scripts/select_brand.py      # apply frozen criteria, emit artifacts
```

Full fresh-machine setup: **`docs/HANDOFF.md` §12**.

---

## Next action

> **Derive and freeze the AppleSupport intent taxonomy from the real data, then build the
> classifier under TDD.**

**Do NOT, before that milestone is complete:** build the classifier before the taxonomy is
frozen; start the golden set; train models; run evaluation; make LLM API calls without first
presenting provider, model, role, call count and cost estimate to the project owner; re-open
brand selection; or modify evaluation methodology.

Details and the full procedure: **`docs/HANDOFF.md` §13**.

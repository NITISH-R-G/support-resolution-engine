# Current State — Starting Position

**Date:** 2026-09-09
**Status at the time of writing:** Greenfield. No pre-existing local implementation.

> **CURRENT STATUS (2026-09-10): superseded as a status document.** This file records the
> project's *starting position* and its provenance declaration, both of which remain accurate
> and are worth preserving. For where the project stands **now**, read
> [`HANDOFF.md`](HANDOFF.md) — Milestone 2 is complete, the corpus is acquired, AppleSupport is
> selected, and 244 tests pass. The environment questions listed in §4 below have all been
> resolved.

---

## 1. Statement of starting position

`C:\Projects\Hiver` was **empty** at the start of this project, apart from the `docs/` directory
created by this analysis phase. Verified by direct listing of the directory (and only that
directory — no filesystem-wide search was performed).

There is **no prior local codebase, no fork, no inherited implementation, and no vendored code**
from any source. This project is written from scratch.

This document exists to make that explicit, because the standard "repository forensics" audit
assumes an existing implementation to inventory. There is none. The audit that *was* performed is
the forensic comparison of seven public repositories in
[`PUBLIC_REPO_COMPARISON.md`](PUBLIC_REPO_COMPARISON.md), which serves the same purpose: establishing
what already exists, what is worth learning from, and what must be built independently.

---

## 2. Component inventory

The standard audit table, completed for a greenfield start:

| Component | Current behavior | Quality | Hiver relevance | Reusable? | Needs modification? | Reason | Tests? | Risk |
|---|---|---|---|---|---|---|---|---|
| *(all components)* | Does not exist | n/a | n/a | n/a | n/a | Greenfield | None | n/a |

Nothing is inherited. Nothing needs removal. Nothing carries technical debt. There is no fork
history, no provenance to untangle, and no dangerous-to-reuse component — because there is no code.

**This is a favourable starting position, not a deficit.** Six of the seven public repositories
carry no license, which makes their code legally unusable; and their dominant failure mode
(circular evaluation, documented in the comparison) is *structural* — inherited by anyone who
adapts their harness. Starting clean avoids importing both problems.

---

## 3. Provenance declaration

For the final report's citation section, the position we can truthfully assert:

- **No code** is copied from any of the seven public Hiver repositories, including the one that is
  MIT-licensed. Rationale in `PUBLIC_REPO_COMPARISON.md` §1.
- **Ideas and architectural patterns** taken from those repositories are enumerated in
  `PUBLIC_REPO_COMPARISON.md` §5–6 and will each be cited in `docs/DECISION_LOG.md`.
- **No golden labels, evaluation results, report text, decision-log entries, human annotations, or
  benchmark artifacts** are taken from any other candidate. All are produced independently.
- Third-party **libraries** (scikit-learn, pandas, pytest, etc.) are used normally under their own
  licenses and pinned in `requirements.txt`.
- The **dataset** is the public Kaggle "Customer Support on Twitter" corpus, used under its own
  terms, with acquisition documented in the README rather than committed to the repository.

---

## 4. Environment

Confirmed present on the development machine:

| Tool | Version | Notes |
|---|---|---|
| git | 2.53.0.windows.1 | |
| gh CLI | 2.95.0 | authenticated |
| OS | Windows 11 | paths and scripts must be Windows-compatible |

**Resolved since (2026-09-10):**

| Question | Resolution |
|---|---|
| Python version | 3.14.3; `requirements.txt` pinned; `>=3.11` supported |
| Kaggle credentials | Resolved by the project owner; five auth mechanisms supported (`src/hiver_support/kaggle_auth.py`). Corpus acquired |
| LLM API access | **Still open** — deliberately. No provider configured, no call made, $0 spent. Required only from the generation/judge milestones onward (`DECISION_LOG.md` D13) |

---

## 5. What happens next

1. ✅ `PUBLIC_REPO_COMPARISON.md` — forensic comparison of the seven public repositories
2. ✅ `CURRENT_STATE.md` — this document
3. ✅ `SPEC.md` — target system specification
4. ✅ Blocking environment questions resolved (dataset acquired; LLM access deferred by design)
5. ✅ Milestones 1, 1b, 1c, 1d — scaffold, PII, leakage guards, normalisation and splitting
6. ✅ Milestone 2 — real-data validation and brand selection (**AppleSupport**)
7. ⏳ **Next:** derive and freeze the AppleSupport intent taxonomy, then the classifier under TDD

See [`MILESTONES.md`](MILESTONES.md) for per-milestone results and [`HANDOFF.md`](HANDOFF.md)
for the canonical continuation document.

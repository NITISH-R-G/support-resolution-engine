# Current State — Starting Position

**Date:** 2026-09-09
**Status:** Greenfield. No pre-existing local implementation.

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

**Not yet confirmed — blocking for later milestones:**

- Python version and virtual-environment strategy
- LLM API access (which provider(s), which keys) — required for Milestones 7, 12, 13
- Kaggle credentials or a manual download path for the dataset — required for Milestone 2

These are resolved in `SPEC.md` §2 and raised with the project owner before Milestone 1 begins.

---

## 5. What happens next

1. ✅ `PUBLIC_REPO_COMPARISON.md` — forensic comparison of the seven public repositories
2. ✅ `CURRENT_STATE.md` — this document
3. ✅ `SPEC.md` — target system specification
4. ⏳ Confirm blocking environment questions (LLM provider, dataset acquisition, brand choice)
5. ⏳ Milestone 1 — repository scaffold, test harness, CI, leakage-guard tests (TDD from the first
   commit)

No implementation code is written before step 4 completes.

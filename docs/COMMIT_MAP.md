# Commit map (history rewrite, 2026-09-17)

The repository history was rewritten to remove tweet text (`docs/RELEASE_AUDIT.md` §19). Every
commit was kept, with its message, author, dates and order. Commit ids changed wherever a commit's
content or message changed.

**Where old ids still appear.** Prose, scripts and `VERIFICATION.json` were updated to the new ids.
Generated provenance was left byte-unchanged as immutable evidence and still cites the old ids:
- the `git_sha` fields in `GOLDEN_LOCK*.json`, `manifest.json` and `metrics.json`;
- the report JSON;
- `reports/golden_eval/summary.md` and the `evaluation_boundary*` reports.

Use this table to resolve them.

| Committed | Old id | New id | Subject |
|---|---|---|---|
| 2026-09-09 | `c008b87` | `c008b87` (unchanged) | Milestone 1: scaffold + conversation reconstruction (TDD) |
| 2026-09-09 | `15d973a` | `15d973a` (unchanged) | Milestone 1b: PII masking at the API boundary (TDD) |
| 2026-09-09 | `51fd4cb` | `51fd4cb` (unchanged) | docs: correct self-agreement framing (reliability != validity) |
| 2026-09-09 | `b00f9e8` | `b00f9e8` (unchanged) | Milestone 1c: leakage guards (TDD) + decision log |
| 2026-09-09 | `7f97599` | `7f97599` (unchanged) | docs: multi-criteria brand selection; retract unsupported claim |
| 2026-09-09 | `59c3f17` | `59c3f17` (unchanged) | Milestone 1d: text normalisation + deterministic temporal split (TDD) |
| 2026-09-09 | `e010e0a` | `e010e0a` (unchanged) | Data provenance: hard separation between fixtures and real corpus data |
| 2026-09-09 | `40680aa` | `40680aa` (unchanged) | Fix Kaggle credential detection (real integration bug) |
| 2026-09-10 | `5f73d9a` | `2435687` | Milestone 2: real-data validation and brand selection -> AppleSupport |
| 2026-09-10 | `434bee5` | `85671cd` | chore: checkpoint real-data pipeline and brand selection |
| 2026-09-10 | `49b831b` | `23678e1` | fix: two provenance-guard defects caught by fresh-clone verification |
| 2026-09-10 | `b337f0b` | `30dca50` | Milestone 3 (candidate): AppleSupport intent taxonomy, not frozen |
| 2026-09-10 | `1645ba5` | `ada599e` | Milestone 3 round 2: taxonomy adjudication + reply_classify defect fixes |
| 2026-09-10 | `f5f66dc` | `8e9df79` | Milestone 3 round 3: corrected brand decision + final taxonomy candidate |
| 2026-09-10 | `cd75638` | `9053e70` | Milestone 3 FROZEN: AppleSupport intent taxonomy v0.3.0 |
| 2026-09-10 | `4be4e59` | `dee2c5c` | Milestone 4: classifier subsystem against frozen taxonomy v0.3.0 |
| 2026-09-10 | `c29afe3` | `d036a98` | docs: sync milestone banner to 485 tests and flag weak-label framing |
| 2026-09-11 | `d256ef0` | `e9c423e` | Milestone 5: end-to-end support resolution agent on real data |
| 2026-09-11 | `6a1fcfc` | `d449b02` | Milestone 6: golden candidate set (unlabelled) + real LLM provider |
| 2026-09-11 | `18855f9` | `2540aa0` | Validate the LLM provider against a live API (Groq) |
| 2026-09-11 | `fcc4e1a` | `302207e` | Milestone 7 Part 0: opaque evidence labels, upstream provenance, honest cache key |
| 2026-09-11 | `ae713c6` | `86e8b3a` | Milestone 7: semantic security gate, policy validator, relevance contract |
| 2026-09-11 | `7adfca8` | `87999f4` | Milestone 8 prep: annotation workflow verified, freeze tooling ready |
| 2026-09-11 | `84cf6a3` | `8db4c3b` | Add annotation retraction: withdraw a label without destroying it |
| 2026-09-11 | `3822b44` | `a7ff1c1` | Model-assisted annotation: suggestions a human adjudicates, never labels |
| 2026-09-11 | `2e209f5` | `c5a5e14` | Inline corrections and queue filtering for assisted annotation |
| 2026-09-14 | `fbbe88f` | `3914f9d` | Plain-English annotation screens for the human annotator |
| 2026-09-14 | `6772707` | `9b9e6f0` | Golden-set evaluation: agent vs two baselines, metrics with CIs, LLM judge |
| 2026-09-14 | `b1429da` | `0135294` | Risk-coverage curves: the automation/safety tradeoff, with no threshold chosen |
| 2026-09-14 | `8586a29` | `9e73901` | Top-5 failure analysis from committed evaluation artifacts |
| 2026-09-14 | `409622d` | `9c4efdd` | Release audit: dependency failures fail closed; audit scripts and findings |
| 2026-09-14 | `a536b45` | `7bae8d2` | README with verified reproduction commands; pin matplotlib |
| 2026-09-14 | `81bafe4` | `7ec606d` | Freeze the golden set: GOLDEN_LOCK.json |
| 2026-09-14 | `07f7ca8` | `0bc9df7` | Release audit: evaluation boundary, reproducibility timings, privacy and cost reconciliation |
| 2026-09-14 | `fe9cfe0` | `b9c545c` | Upgrade pytest 8.3.4 -> 9.0.3 (CVE-2025-71176, GHSA-6w46-j5rx-g56g) |
| 2026-09-14 | `1aae20c` | `d50e610` | Release boundary: Python 3.12 + evaluated-environment lock, verified; remediation plan |
| 2026-09-17 | `ddf088e` | `d469684` | Remove tweet text from the repository; gold v1/v2; masker and line-ending fixes |
| 2026-09-17 | `74660ed` | `42ea7fd` | History-rewrite tooling, local prompt-injection audit, docs for the text-free repository |
| 2026-09-17 | `d4cdce4` | `102ef95` | Verifier: scan quoted spans in commit messages; fix corrupted glyph in MILESTONES.md |
| 2026-09-17 | `6e05ee1` | `659ac42` | Close gaps found by a full-corpus history scan: probes exemption, brand-reply test literals |
| 2026-09-17 | `f7d6abd` | `7228094` | Replace two further brand-reply variants in reply-classification tests; verifier probe scope |

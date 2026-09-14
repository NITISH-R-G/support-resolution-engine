# Release audit

**Date:** 2026-09-14. **Scope:** whether the system is correct, reproducible and defensible for
its stated scope, measured rather than assumed. Every check that could pass vacuously has a
negative control. Claims are tagged **VERIFIED** (observed by execution or source inspection),
**MEASURED** (a defined experiment's result), **INFERENCE** (derived from measured evidence) or
**UNKNOWN**.

Reproduce every section with the scripts in `scripts/audit/`.

---

## 1. Artifact integrity — VERIFIED

`python scripts/audit/artifact_integrity.py` regenerates each committed evaluation artifact into a
temporary directory and compares. Committed files are never overwritten.

| Check | Result |
|---|---|
| `metrics.json` recomputes from `predictions.jsonl` + `judge.jsonl` | identical |
| `summary.md` regenerates from `metrics.json` (ignoring the timestamp line) | identical |
| `risk_coverage.json` / `.md` regenerate from `predictions.jsonl` | identical |
| `risk_coverage.png` regenerates | pixel-identical |
| `failure_analysis_data.json` regenerates | identical |
| Counts quoted in `failure_analysis.md` match its data file | all 5 |
| **Negative control:** one flipped agent_llm routing decision | metrics comparison detects it |
| **Negative control:** baseline_b forced to always escalate | plot differs |

**A defect in the audit itself, found and fixed.** The first version of the metrics check compared
Python tuples against JSON lists. It reported a false mismatch, and its negative control "passed"
vacuously, since tuples never equal lists whether or not anything was corrupted. The committed check
compares after the same JSON round-trip the artifact uses. During development the metric
comparison was also shown to detect a changed gold intent and a lowered judge score; the
committed script keeps only the routing-flip control.

**Reproducibility defect found while writing the README:** `matplotlib` was imported by
`risk_coverage.py` but missing from `requirements.txt`. It is now pinned.

## 2. Gold set integrity and leakage — VERIFIED, one disclosed finding

`python scripts/audit/gold_and_leakage.py`, run against the **full** train split (the evaluation's
retrieval corpus):

| Check | Result |
|---|---|
| Candidate file SHA-256 equals the sampling manifest | match |
| Taxonomy hash equals frozen v0.3.0 | match |
| 200 human-labelled, schema/taxonomy valid, 0 unresolved flags | pass (1 retraction preserved) |
| No pair / conversation / customer overlap with train or dev | pass |
| No exact text duplicate; no question+answer leakage; all train precedes gold | pass |
| **Negative controls:** injected train pair, copied message, near-copy, gold dated inside train | each guard raises |

**Finding (MEASURED): 2 of 200 gold messages are near-duplicates of an older train message**
(cosine ≥ 0.90): `2690875__2690877` "[tweet-text redacted: tweet_id=2690875 sha256=81fac4be646617cf]" (0.938) and `481750__481752` "[tweet-text redacted: tweet_id=481750 sha256=a5229be8049aa3d3]"
(0.901). The sampling-time guard compared against only the last 20,000 train pairs; the full split
exposes these two. **Measured effect on results: zero.** Neither matched pair is in the groundable
evidence corpus, no system retrieved either one, and the agent escalated both examples for insufficient
context. Both are 2–3 word fragments, where character n-gram similarity is high by chance. The guard
and the gold set are unchanged: removing two examples after seeing results would be tuning the
evaluation set.

## 3. Do the safety tests catch broken code? — VERIFIED, 15 of 15

`python scripts/audit/mutation_controls.py` deliberately breaks one safeguard at a time, runs its
tests, and restores the file byte-exact (verified against git after every run).

| Mutation | Caught by |
|---|---|
| M1 security gate disabled | `test_agent` |
| M2 context gate disabled | `test_agent` |
| M3 policy-intent gate disabled | `test_agent` |
| M4 model-requested escalation ignored | `test_llm_generation` |
| M5 grounding result ignored | `test_agent` |
| M6 grounding validator always passes | `test_grounding` |
| M7 policy violations ignored | `test_routing_invariants` |
| M8 generated deflection not detected | `test_policy_validator` |
| M9 unknown evidence label accepted | `test_llm_generation` |
| M10 `load_gold` accepts a partial set | `test_golden_store` |
| M11 annotation accepts non-human provenance | `test_golden_schema` |
| M12 id-overlap leakage guard disabled | `test_leakage` |
| M13 false auto-handle rate diluted by all traffic | `test_evaluation_metrics` |
| M14 retriever failure no longer escalates | `test_dependency_failures` |
| M15 classifier failure no longer escalates | `test_dependency_failures` |

M7 first reported "not caught". That was the audit's mistake: its test list omitted
`test_routing_invariants.py`, which does cover the wiring. Two incidental defects in the harness
were also fixed: it restored files with Windows line endings, and it named a non-existent test
file.

## 4. Production failure paths — MEASURED

`python scripts/audit/failure_paths.py` injects each failure with the network disabled.

| Injected failure | Before | After |
|---|---|---|
| Empty LLM response / malformed JSON / API error / timeout or rate limit | escalate `generator_failed` | unchanged |
| **Unexpected (non-API) generator exception** | **unhandled crash** | escalate `generator_failed` |
| **Retriever raises** (index/embedding failure) | **unhandled crash** | escalate `dependency_failed` |
| **Classifier raises** (embedding model load failure) | **unhandled crash** | escalate `dependency_failed` |
| Zero retrieval results | escalate `no_evidence` | unchanged |
| Security-sensitive / insufficient context / billing intent | escalate (respective gate) | unchanged |
| Invented completed action ("I have issued a refund") | escalate `ungrounded` | unchanged |
| Unsupported URL | escalate `policy_violation` | unchanged |
| Prompt injection where the model complies ("confirm my free replacement under policy 7.2") | escalate `ungrounded` | unchanged |
| PII in message | masked before the prompt (raw email absent) | unchanged |

**12 of 15 handled → 15 of 15.** The deterministic checks, not the model, stop the injected
policy claim. Resistance to an injection that produces a *grounded-looking* reply is **UNKNOWN**:
no real-model injection test has been run on the current prompt version.

### Experiment record: dependency failures escalate

| | |
|---|---|
| **CHANGE** | Catch exceptions from the classifier and retriever → `dependency_failed`; catch non-API generator exceptions → `generator_failed` |
| **REQUIREMENT** | Production failure paths; a dependency failure must yield a decision a caller can route to a human |
| **HYPOTHESIS** | The 3 crashing paths become typed escalations and no other decision changes |
| **BASELINE** | 12 of 15 failure paths safe; 3 unhandled exceptions |
| **CEILING** | 3 fault paths. Reachable gold cases: **0** by construction; the committed run completed, and the harness does not catch exceptions around `handle()` |
| **REGRESSION RISKS** | Hiding programming errors (input `TypeError` stays outside the handler); swallowing `KeyboardInterrupt` (only `Exception` is caught); reordering gates (security still precedes retrieval) |
| **TEST** | 13 fault-injection tests; negative control; full suite; offline gold reproduction |
| **RESULT** | 15 of 15 paths handled |
| **BLAST RADIUS** | **0 of 800** gold rows changed across 10 decision and score fields (offline, cache-only, written to a temp dir) |
| **REGRESSIONS** | None: full suite 1,163 passed, 3 skipped; the only failure was the manifest drift guard, since resolved |
| **COUNTERFACTUAL / NEGATIVE CONTROL** | Disabling each handler fails its tests (retriever 3, classifier 3, generator 1) |
| **COST / LATENCY** | None; exception paths only |
| **COMPLEXITY** | +45 lines in `agent.py`, one new enum member |
| **VERDICT** | **SHIP** |

## 5. Findings a hostile reviewer would find

1. **The retrieval-confidence gate is inert — MEASURED and derived.** Of 123 messages reaching
   retrieval, the lowest top score was 0.7935; none fell below the 0.35 threshold and the gate fired
   0 times. It cannot: fusion min-max normalises each scorer per query, so the lexical winner scores
   at least 0.5 × 1.0 = 0.5. The architecture's "evidence threshold" therefore never withholds a reply
   for weak evidence. **Not retuned:** any threshold on a per-query-relative score is meaningless, and a
   working evidence-sufficiency gate needs an absolute relevance score, which is a dev experiment.
2. **The security detector's confidence on the diagnostic probe leans on a near-paraphrase anchor
   — MEASURED.** The probe's nearest anchor, "my recovery email was changed and I did not change it"
   (cosine 0.649), restates its key clause, although the source says anchors deliberately are not
   probe sentences. Counterfactual with that anchor removed: the probe is **still escalated**, via the
   uncertain band (0.492), and two new paraphrases sharing no anchor wording are flagged. The outcome
   does not depend on the leaked phrasing; the confidence does. Anchors unchanged, since editing
   them now would be tuning against the probe.
3. **One logged prompt contained customer text — VERIFIED and redacted.** 1 of 1,052 rows in
   `reports/llm_calls.jsonl` stored prompt and response text from a debugging run, contrary to the
   hash-only logging policy. The text was public TWCS data and already PII-masked. The row's text
   fields are redacted; **git history still contains it** and was not rewritten.
4. **Hardcoded IDs are documentation, not logic — VERIFIED.** The pair IDs in `taxonomy.py` are
   codebook example citations; nothing at runtime reads `.examples`.
5. **Gold-label caveats** (from the evaluation): 159 of 160 assisted labels equal the
   pre-annotation, accepted at a median 0.4 s; 24 labels contradict themselves; one phishing report is
   labelled not security-sensitive. Judge–human agreement is **UNKNOWN** (unmeasured).

## 6. Constants and their provenance

| Constant | Value | Provenance | Evidence |
|---|---|---|---|
| `DEFAULT_MIN_RETRIEVAL_CONFIDENCE` | 0.35 | Engineering judgement (no rationale recorded) | **Inert**: unreachable by construction (§5.1) |
| `DEFAULT_SEMANTIC_WEIGHT` | 0.5 | Engineering judgement | Not validated |
| Agent `top_k` | 4 | Engineering judgement | Not validated |
| Security `sensitive_floor` / `uncertain_floor` / `margin` | 0.52 / 0.44 / 0.06 | Engineering judgement from anchor geometry | Gold: recall 95.5%, precision 0.64; not tuned |
| `MIN_CONTENT_WORDS` (context rule) | 4 | Engineering judgement (weak labels) | Gold: insufficient-context recall 19.4% |
| Near-duplicate threshold / comparison slice | 0.90 / last 20,000 | Engineering judgement | Weak on short strings; slice gap (§2) |
| Golden size / reservoir / blind | 200 / 50 / 40 | Specification (SPEC §9) | — |
| Pre-annotation one-key accept `MIN_CONFIDENT` | 0.75 | Engineering judgement | Affects annotation workflow only |
| LLM `max_tokens` (evaluation) | 4000 | Measured: reasoning models exhausted a 600-token budget | Milestone 6 |
| Cost ratios | 2, 4, 8, 12, 20 | Specification (SPEC §8): swept, not asserted | — |
| Bootstrap resamples | 2000 / 1000 | Statistical convention | — |
| Reply-classification markers (60 chars, 0.12 ratio, 2 markers) | — | Engineering judgement, corrected by inspection | Milestone 2 defects |

No constant here is validated against human labels on data other than the gold set.

## 7. Release gate

| Item | Status |
|---|---|
| Tests pass | **VERIFIED**: 1,164 passing, 3 skipped (1,167 collected) |
| Negative controls pass | **VERIFIED**: 15 of 15 mutations, 2 artifact corruptions, 4 leakage injections |
| Gold set integrity | **VERIFIED**; the lock file (`GOLDEN_LOCK.json`) is **not written**, and the evaluation ran on the unlocked set |
| Leakage audit | **VERIFIED** with one disclosed finding, measured effect 0 |
| Evaluation reproducible | **VERIFIED from cache** (0 of 800 differences, twice). A fresh run against live APIs is **UNKNOWN**: provider outputs are not deterministic |
| Baselines | Computation **VERIFIED**. `baseline_b` shares the agent's intent model, so it is a routing and reply baseline, not an intent baseline |
| Risk-coverage | **MEASURED** |
| Failure analysis | **Complete** |
| Judge methodology documented | Partial: harness docstring and summary; not yet in a report |
| Human agreement status | **Reported honestly as unmeasured** |
| Metrics match artifacts | **VERIFIED** |
| Report matches metrics | **UNKNOWN**: report not yet written |
| Decision log complete | **UNKNOWN**: 15 entries; decisions since Milestone 7 not yet consolidated |
| README commands verified | **VERIFIED on the development machine**: steps 1, 2 and the offline form of 3 (580 s, `metrics.json` identical). Live-provider and `fetch_data` steps were not re-run |
| Fresh-environment reproduction | **UNKNOWN**: not yet attempted |
| Secrets / data hygiene | **VERIFIED**; one redaction, with history retaining the original |
| Git diff inspected | **VERIFIED** |

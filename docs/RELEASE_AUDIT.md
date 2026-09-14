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

## 7. Gold freeze — VERIFIED

`python scripts/freeze_golden.py` was run once, after `--check` passed.

- **Validation:** 200/200 human-labelled; 0 missing fields, 0 taxonomy mismatches, 0 non-human
  annotations; 40 blind, 160 assisted (159 accepted, 1 corrected); 1 retraction kept.
- **Leakage:** 5 guards re-run against the real split, all passing.
- **Lock:** `data/golden/GOLDEN_LOCK.json`, content sha256 `6d78823ac42c95c6…`, committed alone
  in `81bafe4`.
- **Immutability:** all 5 gold files byte-identical before and after (sha256).
- **Lock check:** `verify_lock` passes. **Negative control:** one label changed in memory → raises.
- **Evaluated labels = locked labels:** the gold fields in all 800 prediction rows match the locked
  annotations, 0 mismatches.
- **Disclosed:** the lock was written *after* the evaluation ran. It certifies the same labels;
  it did not gate that run.

## 8. Evaluation boundary — VERIFIED (decisions), with a precise scope

**Original evaluated system.**
- Recorded in `metrics.json` provenance: `original_generated_at` 2026-09-14 09:14 UTC,
  `original_git_sha` `fbbe88f`.
- At that moment the evaluation harness (`scripts/evaluate_golden.py`,
  `evaluation/metrics.py`) and the 200 annotations were **uncommitted**; they were first
  committed in `6772707`.
- `fbbe88f → 6772707` adds only evaluation code and tests (997 lines); no agent code changed.
- The agent that produced the headline numbers is therefore the `src/` tree at `fbbe88f`. The
  harness that ran is **not byte-verifiable** (**UNKNOWN**). The strongest evidence that the
  committed harness is the one that ran: every replay through it reproduces the original
  predictions exactly.

**Post-evaluation changes** (`6772707 → HEAD`):
- `evaluate_golden.py`: `--offline`, `--out`, and score fields recorded on each row.
- `metrics.py`: risk-coverage curve.
- `agent.py`: fail-closed handling of dependency failures.

**What the 800-row comparison shows.** `python scripts/audit/evaluation_boundary.py` replays the
current code from cache in-process, with sockets blocked and the Hugging Face hub forced offline.
- **Network:** 0 connection attempts. Control: a deliberate connection is refused.
- **Cache misses:** 6, the same 6 generator failures as the original run (failed responses are
  never cached).
- **Rows:** the original (git show `6772707`), committed and hardened rows are identical on all
  800: 0 decision, 0 routing, 0 intent and 0 reply differences.
- **Negative control:** one flipped decision is reported as exactly 1 difference.

**What it does not show.**
- The source is **not** byte-identical; the diffs are listed in
  `reports/golden_eval/evaluation_boundary.md`.
- The dependency handlers never execute on these inputs: no classifier or retriever raised. Zero
  differences proves the hardening is inert on the evaluated inputs, not that it is correct. Its
  correctness evidence is §3–§4.
- Score fields did not exist at the evaluation commit, so they are compared only between the
  committed and hardened states.

**Artifact note.** One `judge.jsonl` row (`489714__489713`, baseline_b) differs between
`6772707` and HEAD in `usage` only: `cost_usd` 0.00077 → 0.0, `from_cache` false → true. This
was written by the offline replay in `b1429da`. Scores are unchanged, and `metrics.json` is
identical. The original row is recoverable from `6772707`, and the file was not edited back.

## 9. Cost and call accounting — MEASURED, one unexplained gap

Recomputed from every row of `reports/llm_calls.jsonl` (1,052 calls). Cache hits are not calls
and are not logged. "Retries" counts extra attempts inside a call (rate limits).

| Date (UTC) | Provider | Model | Role | Calls | OK | Failed | Retries | In tok | Out tok | Cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| 09-11 | openrouter | llama-3.3-70b-instruct | pre-annotator | 160 | 160 | 0 | 0 | 108,672 | 10,592 | 0.01352 |
| 09-11 | groq | gpt-oss-120b | smoke tests | 65 | 54 | 11 | 106 | 43,676 | 23,795 | 0.02037 |
| 09-11 | groq | gpt-oss-20b | smoke tests | 37 | 29 | 8 | 65 | 23,487 | 15,636 | 0.00645 |
| 09-11 | openrouter | gpt-oss-120b | smoke test | 3 | 3 | 0 | 0 | 2,394 | 4,952 | 0.00093 |
| 09-14 | openrouter | gpt-oss-120b | generator (4 smoke + 117 eval) | 121 | 115 | 6 | 0 | 86,082 | 48,117 | 0.01136 |
| 09-14 | openrouter | claude-sonnet-5 | judge, abandoned | 337 | 18 | 319 | 0 | 11,463 | 3,942 | 0.06235 |
| 09-14 | groq | qwen3.8-27b | judge | 329 | 328 | 1 | 726 | 144,759 | 32,168 | 0.24448 |
| | | | **Total** | **1,052** | **707** | **345** | **897** | **420,533** | **139,202** | **0.35946** |

Failures: 317 HTTP 402 (Claude, no credit), 18 HTTP 429 after retries, 8 empty generations
(6 gpt-oss-120b in the evaluation, 2 Claude), 2 HTTP 400. Failed calls carry no cost; if any
were billed, the total is a lower bound.

**Reconciliation against reported figures.**

| Reported | Source | Log | Verdict |
|---|---|---|---|
| Generator: 111 requests, 4 cache hits, 6 failures, $0.01081 | `metrics.json` | Eval cluster 08:22–08:38: 117 calls = 111 OK + 6 failed, $0.01081. A 4-call smoke run at 08:15 ($0.00056) produced the 4 cache hits | **Reconciled exactly** |
| Judge: 336 judged | `metrics.json` | 328 OK calls. The 336 auto-handled replies contain 8 duplicate (message, reply) prompts, served from cache | **Reconciled** |
| Judge: 148,110 in / 32,892 out tokens | `metrics.json` | 144,759 / 32,168. Differences of 3,351 and 724 are about 8 × one judge prompt (419 / 90 tokens) | **Explained (INFERENCE)**: summary tokens include the 8 cache hits |
| Judge: **$0.237** | `metrics.json`, README, summary | **$0.24448** (= log tokens × $0.80 / $4.00 per million) | **Discrepancy $0.0075 (3%), cause UNKNOWN.** The summary's own tokens at the same price give $0.2501, so $0.237 matches neither. The call log is the authority; reported judge cost should read $0.244 |
| VERIFICATION.json `cost_incurred_usd` (earlier 0.3156, 102 calls) | manual | 0.35946, 1,052 calls | **Was stale**; now recomputed from the log |

No paid call was made during this audit.

## 10. Privacy: customer text in the tree and in history — MEASURED

`python scripts/audit/privacy_scan.py`.
- **Method:** 8-word shingles over every AppleSupport customer and brand tweet (including thread
  context), matched against every tracked file and all 272 blobs reachable from any ref. Output
  is counts and paths only (`reports/privacy_scan.json`).
- **Controls, all pass:** a planted customer message is detected; invented text is not; a planted
  email and phone are counted; ids and floats are not counted as phones.
- **Blind spot:** messages under 8 words and paraphrases are invisible to this method.

**Current tree.** Real TWCS customer text is present in 18 tracked files:

| File(s) | Customer msgs matched | Why it is there | Class |
|---|---|---|---|
| `data/golden/candidates.jsonl` | 280 (200 messages + thread context) | The golden set: a required deliverable | **SENSITIVE**: dataset content, pseudonymised by the source, not PII-masked at rest |
| `reports/golden_eval/predictions.jsonl` | 429 | The `message` field of the evaluation rows | **SENSITIVE**: dataset content |
| `reports/taxonomy_{discovery,probes,adjudication,candidate}.json` | 177 / 132 / 98 / 27 | Cluster exemplars for taxonomy derivation | **SENSITIVE**: dataset content |
| `reports/classifier_dev_errors.json` | 27 | Dev-set error inspection | **SENSITIVE**: dataset content |
| `reports/llm_smoke_*.json` (3), `reports/agent_demo_run.json` | 23 / 23 / 23 / 5 | Behavioural smoke runs on train messages | **SENSITIVE**: dataset content |
| `src/hiver_support/taxonomy.py`, `docs/ANNOTATION_GUIDE.md`, `docs/INTENT_TAXONOMY.md`, `docs/TAXONOMY_ADJUDICATION.md`, `docs/AGENT.md`, `docs/CLASSIFIER.md`, `failure_analysis.md` | 24, 10, 4, 3, 1, 1, 1 | Codebook examples and quoted failures | **SENSITIVE**: dataset content, short excerpts |
| `tests/*` | 0 customer; brand phrases in 5 files | Canned brand phrases in classifier tests | **SAFE**: public brand boilerplate |

**Identifier-shaped content in the tree.**
- **Emails:** 0 in any dataset-bearing file. The 4 in `tests/test_pii_handling.py` are synthetic
  fixtures (**SAFE**, synthetic).
- **Phone-shaped strings:** 5 (4 rows of one message in `predictions.jsonl`, 1 in
  `candidates.jsonl`) are the same toll-free 8xx number, quoted by a customer inside their
  message. **Class: SENSITIVE (dataset content).** A business number is not personal data, but it
  still reproduces a message from a real person, and the committed message is not PII-masked
  (masking runs only at the API boundary). The 7 in `docs/PUBLIC_REPO_COMPARISON.md` and 1 in
  `summary.md` are dates and version numbers (false positives).
- **Twitter handles:** TWCS already replaces user handles with numeric author ids. Brand handles
  (e.g. `@AppleSupport`) are public. Whether any real user handle survived inside message text
  was **not measured**: **UNKNOWN**.
- **Dataset licence:** the terms under which excerpts may be redistributed were not verified in
  this audit: **UNKNOWN**.

**History.** 9 blobs reachable from `main` contain customer text and are not present at HEAD:

| Blob | Path | Customer msgs | Exposure | HEAD status |
|---|---|---|---|---|
| `d0a31f28`, `bfd0ed5f`, `9b60f3dd` | `reports/llm_calls.jsonl` | 5 each | Earlier versions; one row stored prompt and response text from a debug run | **REDACTED** at HEAD (`409622d`); history retains it |
| `65598a9a` | `reports/golden_eval/predictions.jsonl` | 429 | The `6772707` version, before score fields | Same messages still at HEAD |
| `3175975d` | `reports/agent_demo_run.json` | 33 | Earlier, larger demo run | 5 match at HEAD; **up to 28 exist only in history** |
| `db71f9b0` | `src/hiver_support/taxonomy.py` | 27 | Earlier codebook examples | 24 at HEAD; up to 3 only in history |
| `7a752c16`, `d61579c2`, `6879304a` | `ANNOTATION_GUIDE.md`, `INTENT_TAXONOMY.md`, `AGENT.md` | 10 / 4 / 1 | Earlier doc versions | Equivalent excerpts at HEAD |

**Repository visibility (verified with `gh repo view`):** **PRIVATE**, 0 forks. No blob in
history exceeds 2 MB, and no raw dataset file (`twcs*`, `.pkl`, `.zip`) was ever committed. The
one tracked CSV, `reports/brand_profiles_all.csv`, holds per-brand aggregate statistics.

**Assessment and remediation.**
- **Not required under the current data policy:** a history rewrite. The policy forbids
  committing the *raw dataset*, credentials and caches, and none was found in any blob. The
  dataset excerpts in history are the same kind of content HEAD deliberately contains: the golden
  set requires the messages. Rewriting history while HEAD still holds 280+ messages would remove
  no exposure category.
- **Required only if the owner decides the repository must hold no dataset excerpts** (for
  example before making it public):
  1. Replace message text at HEAD with pair ids, re-materialised from the Kaggle download by a
     script (all 18 files above, including the golden candidates).
  2. Rewrite history for the same paths with `git filter-repo`, then force-push.
  3. Ask GitHub support to purge cached views.
  4. Accept that existing clones and forks keep the content.

  This is **a decision for the project owner, not taken in this audit.** No history was rewritten.

## 11. Secrets — VERIFIED

Every revision was searched for OpenRouter, Groq, Anthropic, OpenAI project, Kaggle, AWS and
GitHub token shapes and private-key headers. No matched text was printed.
- **Only hits:** 2 strings in `tests/test_llm_provider.py` (every revision since `6a1fcfc`). Both
  are obvious placeholders (bodies begin `abcdef…` and `deadbeef…`) used by redaction tests:
  **SAFE, synthetic**.
- **Paths:** no `.env`, `kaggle.json`, access token or key file has ever been committed.
  (`tests/test_credentials.py` is a test module.)
- **Ignored directories:** nothing under `cache/`, `data/raw/`, `data/interim/`, `exports/` or
  `.venv/` is tracked.
- **Caveat:** a secret in a format outside these patterns would not be found.

## 12. Documentation audit — stale claims found and resolved

| Old claim | Current verified value | File | Action taken |
|---|---|---|---|
| Current state 2026-09-10: Milestone 5; golden set "Not created"; "Zero LLM API calls"; 563 tests | Gold frozen; evaluation done; 1,052 logged calls; 1,164 passing | `AGENTS.md` | Current-state table rewritten |
| `pytest  # full suite (244)`; next action "build the golden set" | README commands; next action: the report | `AGENTS.md` | Replaced |
| "on conflict, HANDOFF.md wins" | HANDOFF is stale | `AGENTS.md` | VERIFICATION.json and RELEASE_AUDIT.md now win |
| Status "Milestone 6 … agent still NOT evaluated"; 1,003 / 244 tests | Evaluated; 1,164 | `docs/HANDOFF.md` | Dated release banner; body kept as history |
| Banner "839 passed … agent remains unevaluated"; log ends at M6 | M7–release exist | `docs/MILESTONES.md` | Banner with commits for later work |
| Pass 1 "fully blind" for all 200; passes 2–3 planned | 40 blind / 160 assisted; passes 2–3 not done | `docs/GOLDEN_SET.md` §7 | "As executed" note |
| "Not evaluated", "zero API calls"; evidence gate "empty or thin" | Evaluated; thin-evidence threshold inert | `docs/AGENT.md` | Banner and gate labels corrected |
| "244 tests pass" | 1,164 | `docs/CURRENT_STATE.md` | Pointer updated |
| `_agent_note` "NOT evaluated"; `golden_candidates_label_status` UNLABELED; `golden_annotation_pass_1_blind` true; `golden_set_locked` false; 102 calls / $0.3156; OpenRouter "not exercised"; lexical defect "left unfixed"; `readme_present` false | See §7–§9; OpenRouter used; defect addressed in M7 | `VERIFICATION.json` | Fields updated from artifacts |
| 15 decisions, all from Milestones 1–2 | 15 decisions covering taxonomy through release | `docs/DECISION_LOG.md` | Consolidated (§13) |
| Judge cost $0.237 | $0.2445 from the call log (§9) | README, `summary.md`, `metrics.json` | README corrected; evaluated artifacts left as recorded, discrepancy documented |
| "a fresh clone must call the providers"; "stops at start-up"; an offline run implied reproducible | Four targets with measured timings (§14) | README | Reproduction section rewritten from measurements |
| "Gold lock … not written" | Frozen | README | Corrected |

Historical milestone result blocks (for example "481 passed" in `CLASSIFIER.md`) are dated
snapshots, left as-is. `DATA_PROVENANCE.md` "325 passing" is labelled as a headline and is
**stale, not yet updated**.

## 13. Decision log — consolidated

Now 15 entries (D1–D15) covering the decisions the assignment asks to justify:
- brand selection, taxonomy, reconstruction;
- leakage and model independence, PII;
- the safety boundary, retrieval and evidence labels;
- assisted/blind gold, judge selection and judge-provider failure;
- risk-coverage methodology;
- fail-closed dependencies and the evaluation freeze boundary.

IDs cited elsewhere in the repository (D1, D2, D3, D8, D12, D13, D14) keep their meaning; each
entry cites commits or artifacts. The cited test names, `LLM_EXTRA_BODY` and the security
figures (recall 21/22, precision 21/33) were checked against source and `metrics.json`. One
unverifiable test count was removed from the draft.

## 14. Reproducibility — MEASURED

"Reproduce the headline results" can mean four different things. Evidence for one is not
evidence for another.

| Target | Credentials | Network | Measured | Bit-identical | README |
|---|---|---|---|---|---|
| **A. Artifact verification**: metrics, curves, plot and failure counts from committed predictions and judge scores | none | package install | Fresh clone, light dependencies: clone 5 s + venv 22 s + install 201 s (warm pip cache) + scripts 65 s = **295 s**. Cold pip cache: install **683 s** (numpy and pandas compiled on Python 3.14), about **775 s** total. With full `requirements.txt`: **1,437 s** | **Yes**: 9/9 integrity checks including 2 negative controls; `git status` clean | Step 2, both timings |
| **B. Offline replay**: 800 predictions from cached responses | Kaggle, plus dummy LLM keys (defect 1) | dataset download | Development machine **580 s** (all stages). Fresh-clone environment with a *copied* cache: **450 s** (predictions). **Not possible from a clone:** the cache is not distributed | Decisions, reasons, intents, replies, evidence: **0 of 800 differ** in both environments; 9 `retrieval_confidence` values differ by < 1e-6 under torch 2.14 | Step 3 note |
| **C. Fresh evaluation**: live models | Kaggle + OpenRouter + Groq | yes | **Not re-run** (no paid calls in this audit). Original call-log span: generator 08:22–08:38 UTC (~16 min), judge 08:41–09:13 (~32 min, 726 rate-limit retries) | **No**: provider output is non-deterministic | Step 3 |
| **D. Full end-to-end** from a fresh clone | Kaggle + OpenRouter + Groq | yes | clone 6 + venv 71 + install 1,273 + tests 311 + analyses 87 + data 558 + evaluation setup 614 = **2,922 s before the first prediction** (stopped at the missing key), then C | No | Step 3 timing table |

**Fresh-clone conditions.**
- New clone of `origin/main` at `81bafe4` and a new venv.
- All `*_API_KEY` variables unset; no `.env`; no LLM cache.
- Kaggle credentials from `~/.kaggle`, the documented mechanism.
- **Lower-bound caveats:** 50 of 78 wheels came from pip's local cache (including previously
  compiled numpy and pandas), and the embedding model came from the local Hugging Face cache.
  The test step ran before the corpus was downloaded: 1,146 passed, 21 skipped.

**15-minute requirement.**
- **Target A: PASS** from a fresh clone (295 s warm, about 775 s cold) with the light
  dependencies; 1,437 s with the full `requirements.txt`.
- **Target D: FAIL** (≥ 2,922 s before any prediction). The bottlenecks are:
  1. a 124 MB torch download plus a 42,225-file, 1.24 GB install;
  2. a 169 MB dataset download;
  3. about 10 min of CPU setup (reconstruction, fitting, embedding 14,921 evidence cases);
  4. about 48 min of rate-limited API calls in the original run.

  Items 1 and 3 are the agent's own requirements: embeddings drive retrieval and the semantic
  security gate. Items 2 and 4 are bound by network and provider.

**Options considered (ceiling before change).**

| Option | Benefit | Blast radius | Verdict |
|---|---|---|---|
| Light dependency set for target A (documented pip command) | Install 1,273 s → 201 s warm / 683 s cold; 1.24 GB → 377 MB | None: `requirements.txt` unchanged; the scripts never import torch, transformers or kaggle (verified) | **Adopted as documentation only** |
| Drop torch, transformers or sentence-transformers | torch + transformers = 667 MB, 54% of the install | Breaks retrieval embeddings and the semantic security gate, so changes evaluated behaviour | **Rejected** |
| Move pytest and kaggle to a dev file | pytest < 1% of install; kaggle is needed by `fetch_data.py` | Marginal | **Rejected**: cannot bring D under 15 min |
| Commit the LLM cache so a clone can replay offline | B from a clone in about 10 min | Publishes prompts containing customer messages | **Rejected** under the data policy |
| Headline on a `--limit` subset | Faster | Not the headline results | **Rejected** as reproduction; remains a smoke test |

**Defects found (recorded, not fixed).**
1. `--offline` requires API key variables although no call is made: providers are constructed
   before the cache-only wrapper. Workaround documented.
2. The generator's key and endpoint come from `LLM_PROVIDER` instead of being resolved for
   OpenRouter the way the judge's are. Safe with `.env.example` as shipped; documented.
3. The README said a missing key stops the script "at start-up"; it stops after about 614 s of
   setup. Corrected.

## 15. Dependencies — MEASURED

**CVE-2025-71176** (GHSA-6w46-j5rx-g56g, Dependabot alert #1, medium).
- **Package:** `pytest==8.3.4`, vulnerable `< 9.0.3`, fixed in 9.0.3.
- **Issue:** on UNIX, predictable `/tmp/pytest-of-{user}` directories allow local denial of
  service or possibly privilege escalation.
- **Classification:** Dependabot labels it "runtime" only because `requirements.txt` has no dev
  split. **VERIFIED development-only:** nothing under `src/` or `scripts/` imports pytest.
- **Surface:** 18 test files use `tmp_path` (none use legacy `tmpdir`). The development machine
  is Windows, where the UNIX path in the advisory does not apply.

| | BASELINE | CANDIDATE |
|---|---|---|
| pytest | 8.3.4 | 9.0.3 |
| Environment | throwaway fresh clone, corpus present | same venv; pip changed nothing else |
| Result | 1,164 passed, 3 skipped, rc 0 | 1,164 passed, 3 skipped, rc 0; 0 failures; 0 deprecation or removal warnings |
| Runtime | 256 s | 254 s |

- **Blast radius:** one line in `requirements.txt`; no evaluation code path.
- **VERDICT: SHIP.** Not yet applied to `main`; it should land as its own commit.

**Python 3.14 wheel gap — MEASURED.**
- `numpy==2.1.3` and `pandas==2.2.3` have no Python 3.14 wheels, so every fresh install on the
  project's own Python version compiles them. That needs a C toolchain, and fails without one.
- Options, each needing its own experiment: pin wheel-available versions (changes the evaluated
  environment, and requires the 800-row replay to show 0 decision differences), or document
  Python 3.12/3.13 (untested).
- **UNRESOLVED; owner decision.**

**Unpinned transitive dependencies — MEASURED.**
- A fresh install resolves torch 2.14.0 / transformers 5.17.0 / tokenizers 0.23.2 /
  huggingface-hub 1.31.0; the evaluation ran on 2.10.0 / 5.5.0 / 0.22.2 / 1.12.0.
- **Measured effect on the evaluated inputs:** 0 of 800 decisions differ; 9 retrieval scores
  differ by < 1e-6.
- A lock file would remove the drift: **proposed, not applied**.

## 16. Release gate

| Check | Status | Measured value | Evidence | Risk / unknown |
|---|---|---|---|---|
| Runnable repository | **PASS** | Fresh clone: install, tests, analyses all rc 0 | §14 | Python 3.14 needs a compiler for numpy/pandas |
| README verified | **PASS for A and local B; UNKNOWN for C** | Commands run from a fresh clone | §14 | Paid live run not executed |
| < 15-minute reproduction | **PASS for A; FAIL for D** | A 295 s warm / ~775 s cold; D ≥ 2,922 s before predictions | §14 | Headline *metrics* verify in < 15 min; *regenerating predictions* does not |
| Golden set 150–250 | **PASS** | 200 | `GOLDEN_LOCK.json` | — |
| Golden set frozen | **PASS** | sha `6d78823a…`; `verify_lock` + negative control | §7, `81bafe4` | Frozen after the evaluation (disclosed) |
| Assisted/blind provenance disclosed | **PASS** | 160 / 40; 159 accepted; median 0.4 s | README, `metrics.json` | Anchoring and self-agreement unmeasured |
| Leakage audit | **PASS with finding** | 5 guards pass; 2/200 near-duplicates vs full train, effect 0 | §2, §7 | Near-duplicate slice |
| Automated metrics | **PASS** | intent, security, escalation, false auto-handle rate, cost, CIs | `metrics.json` | — |
| Two baselines | **PASS** | always-escalate; TF-IDF + rules + nearest reply | `summary.md` | baseline_b shares the intent model |
| LLM judge | **PASS** | 336 replies, `qwen/qwen3.8-27b`, 0 failures | D10 | Smaller judge than planned |
| Judge–human agreement | **UNMEASURED (disclosed)** | no human ratings | `metrics.json` | Judge scores uncalibrated |
| Risk-coverage | **PASS** | full grid; no threshold selected | `risk_coverage.md` | — |
| Top-5 failure analysis | **PASS** | counts regenerate | `failure_analysis.md`, §1 | — |
| Misleading-headline section | **UNKNOWN** | report not written | — | Must be in the report |
| One-week plan | **UNKNOWN** | report not written | — | — |
| Decision log 10–15 | **PASS** | 15 | §13 | — |
| Fail-closed behaviour | **PASS** | 15/15 failure paths | §4 | — |
| Structured output validation | **PASS** | malformed / empty / unknown label → escalate | §4, M9 | — |
| PII handling | **PASS at API boundary; FINDING at rest** | email absent from prompt; stored messages unmasked | §4, §10 | Handles in text: UNKNOWN |
| Prompt-injection handling | **PARTIAL** | deterministic checks stop an injected policy claim | §4 | Grounded-looking injection with a real model: UNKNOWN |
| Dependency-failure handling | **PASS** | 12 → 15 of 15; M14, M15 caught | §4 | — |
| Artifact integrity | **PASS** | 9/9 in the dev tree and in a fresh clone (light deps) | §1, §14 | — |
| 800-row evaluation boundary | **PASS (decisions)** | 0 decision, routing, intent, reply diffs in 3 comparisons; 0 network attempts; plus 0 under the fresh environment | §8, §14 | Source not byte-identical (by design); original harness not byte-verifiable |
| Data privacy (tree) | **FINDING** | dataset text in 18 tracked files; 0 emails; 5 rows with one quoted toll-free number | §10 | Licence terms and handles: UNKNOWN |
| Git history | **FINDING, no rewrite** | 9 history-only blobs with dataset text; repo PRIVATE, 0 forks | §10 | Owner decision before any publication |
| Secrets | **PASS** | 0 real credentials in any revision | §11 | Formats outside the scanned patterns |
| Dependency vulnerability | **OPEN; fix verified** | pytest 9.0.3 identical results | §15 | Not applied |
| Python 3.14 wheel gap | **OPEN** | numpy/pandas compiled from source | §15 | Fails without a compiler |
| Cost accounting | **PASS with discrepancy** | $0.35946 / 1,052 calls; judge reported $0.237 vs logged $0.2445 | §9 | $0.0075 gap cause UNKNOWN |
| Stale documentation | **PASS with 1 open item** | 11 stale claims corrected | §12 | `DATA_PROVENANCE.md` headline |
| Full regression suite | **PASS** | 1,164 passed, 3 skipped (dev tree, corpus present) | this commit | — |
| Negative-control audits | **PASS** | 15 mutations, 2 artifact corruptions, 4 leakage injections, lock tamper, boundary flip, network block, 4 privacy-scan controls | §1–§4, §7, §8, §10 | — |
| Final git diff / commit / push | recorded in the commit message | | | |

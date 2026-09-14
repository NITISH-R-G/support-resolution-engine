# Top-5 failure analysis

**Scope.** Built only from committed evaluation artifacts (`predictions.jsonl`, `judge.jsonl`,
`metrics.json`, and `data/golden/candidates.jsonl` for thread context). Nothing was re-run and
no fix is implemented. Every count is recomputed by `scripts/failure_analysis.py` into
`failure_analysis_data.json`. System under analysis: `agent_llm`, all 200 golden examples.

**Ranking rule, fixed before ranking:** severity tier first (customer harmed by an unsafe answer
> customer given a degraded answer > automation lost > metric-only), then frequency, with
production relevance breaking ties.

**Gold-label caveat, applies throughout.** 159 of the 160 assisted labels equal the Llama
pre-annotation, accepted in a median 0.4 s. "Unsafe" means "the adjudicated label says escalate",
and on the assisted examples that is mostly the pre-annotator's judgement. See Limitations.

## Summary table

| Rank | Failure mode | Count | Severity | Representative case | Root cause | Fix (not implemented) | Needs new eval? |
|---|---|---:|---|---|---|---|---|
| 1 | Messages needing Apple to act were auto-handled with self-serve advice | **28** of 29 unsafe auto-handles (42 counting every `human_action_required` label) | Critical: customer harmed | `2580545__2633234` (blind): *[text redacted: sha256=45aaef8344669c9a]* → *[text redacted: sha256=8e17b78ed6eee9ed]* | **Policy.** Escalation rules key on *intent*, and only billing and repair are escalation-sensitive. Nothing represents "this message needs a human to act". The same 29 are auto-handled by `agent_template` and `baseline_b`, so the LLM is not the cause. | A message-level gate on predicted resolution kind: escalate when a human action is required | **Yes.** No training labels exist for resolution kind; it needs new non-gold labels and a dev experiment |
| 2 | Vague requests not recognised as insufficient context | **25** of 31 missed (recall 19.4%); 9 became unsafe auto-handles | High: answers without knowing the request | `2410161__2410160`: *[text redacted: sha256=ccca00feae7ddca1]* → *[text redacted: sha256=b82f9f227ad90aa2]* | **Model (rule).** The context detector is a word-count/lexical rule. A short but word-rich vague message passes. Not explained by missing thread context (only 4 of 25 have prior turns, below the 37% base rate). | Replace the lexical rule with a trained detector | **Yes.** Dev experiment; thresholds must not be set on gold |
| 3 | Replies that deflect instead of help | **11** of 88 answered (12.5%): 6 "go to DM", 5 link-only; 4 are also unsafe | Medium: degraded answer, automation defeated | `550643__550645`: *[text redacted: sha256=b065946d69b53dcb]* | **Retrieval data + policy, one shared rule.** The evidence filter and the output policy validator use the *same* regex (`_DEFLECTION_TAIL_RE`, reused at `policy.py:164`). It matches "in dm" but not "in **a** DM", "head to DM", "join us in a DM", "start the DM", "DM link". All 6 DM replies were copied from evidence that passed the filter (0 of 6 matched, 6 of 6 mention DM), so groundedness scored 5.0. The 5 link-only replies came from evidence whose substance was a URL, masked to `[URL]`. | Detect DM phrasing on the token, not fixed phrases, in both layers; drop evidence that is only a link after masking | **Yes to report a gain.** The pattern gap is a label-independent bug, but it was found on gold and changes the retrieval corpus. Any improvement needs a fresh evaluation, disclosed as post-hoc if run on gold |
| 4 | Over-escalation of automatable messages | **66** of 125 (52.8%) | Low: cost, no harm | `126381__126383`: *[text redacted: sha256=79e949b194e27a1e]*, a reply within an ongoing thread, escalated as insufficient context | **Design + detector precision.** The agent is **single-turn**. All 21 context over-flags have prior thread turns (vs 37% base), which the annotator saw and the agent never does. Also: `model_requested` 15, security false positives 12 (precision 0.64), `policy_intent` 10, `policy_violation` 5, empty provider responses 3. | Pass prior thread turns to the context detector and classifier | **Yes.** Architectural change; dev experiment |
| 5 | Intent misclassification | **114** of 200 wrong (accuracy 43.0%) | Low: little routing effect | `143558__143556`: *[text redacted: sha256=8a8c065307928729]* → `other_unclear` (gold `device_malfunction`) | **Model + design.** 26 of 26 `other_unclear` errors have thread context: the pipeline abstains to `other_unclear` whenever the context rule fires. The rest are mostly `device_malfunction` ↔ `apps_and_services`, a boundary the taxonomy itself marks as fuzzy. The classifier was trained on weak labels. Routing effect: 7 of 9 billing/repair messages detected, 1 auto-handled. | Thread-aware input; train intent on human-labelled non-gold data | **Yes.** Retraining; dev experiment |

## How the failures overlap

| | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| **F1** human action auto-handled (28) | – | 9 | 4 | disjoint (F1 is auto-handled, F4 escalated) | 15 of 29 unsafe had the wrong intent |
| **F2** context misses (25) | 9, all also F1 | – | includes `2410161` | – | – |

- **F1 and F2 are the same harm seen from two sides.** All 9 context misses that became unsafe are also human-action-required.
- **F3 ∩ F1 = 4.** A deflecting reply to a message that needed a human is the worst combination seen.
- **F4 and F5 share one root cause:** single-turn design. 21/21 over-flags and 26/26 `other_unclear` errors have thread context.
- **Intent is not what drives F1.** 15 of the 29 unsafe had the wrong intent, but only 1 had an escalation-sensitive gold intent that was missed. Fixing intent would not fix F1.

## Inspected in full, as required

**All 29 unsafe auto-handles.** 28 are `human_action_required`. Gold intents: `device_malfunction` 17, `apps_and_services` 6, `connectivity` 2, `other_unclear` 2, `complaint_feedback` 1, `repair_order_replacement` 1. **All 29 had model confidence ≥ 0.8**, consistent with the risk-coverage finding that the LLM's self-reported confidence is not a safety signal. The judge rated these replies relevance 4.21 and helpfulness 3.48 (0 fabricated actions): **a reply can read well and still be the wrong routing decision, and reply-quality scores cannot see that.** Also auto-handled by `agent_template` and `baseline_b`: 29 of 29.

**All 11 deflecting replies.** DM-style: `127151__127150`, `2410161__2410160`, `2858282__2858281`, `550643__550645`, `559077__559076`, `600389__600387`. Link-only: `2291596__2291595`, `2948212__2948211`, `487886__487884`, `505063__505061`, `620060__620058`. 4 overlap the unsafe set.

**All 6 empty provider responses** (`2557324`, `2562325`, `2640993`, `2686713`, `2782053`, `2914647`). These are 6 of 115 generator calls (5.2%). Every one escalated fail-closed as designed. 3 were gold-automatable (lost automation, counted in F4); 3 were gold should-escalate (right outcome, incidental reason). The deterministic template generator would have auto-handled all 6. **The cause beyond "provider returned empty content" cannot be determined from the artifacts**; no raw response was logged, and I have not guessed at one. Below the top five because the safety design contained every one. Suggested fix: one retry on empty content before failing closed. That is a reliability change independent of gold labels, checkable with unit tests, though it would change results and so needs re-evaluation to report.

## Evaluation and data limitations (not system failures)

1. **The gold provenance shapes the failure counts.** 25 of the 29 unsafe cases are assisted labels (Llama suggestion accepted), 4 are blind. The unsafe rate is 15.6% on assisted and 10.0% on blind. With n = 40 on the blind side that difference is not a finding.
2. **The gold labels contradict themselves.** 24 examples are labelled `human_action_required` yet `should_escalate = false`. The unsafe count is 29 by the escalation label and 42 by the resolution label. The two fields disagree, and the evaluation uses the escalation label.
3. **Some security labels look wrong.** `2696074__2696073`, *[text redacted: sha256=b0067c324d375d43]*, is labelled not security-sensitive, although the codebook defines phishing as security-sensitive. It counts as one of the agent's 12 false positives. The blind security miss `2893324__2893325` shows no security content in the message itself. Gold is frozen, so these are recorded, not corrected.
4. **The judge cannot see routing errors,** and judge–human agreement is unmeasured. On 17 replies scored by both judges, Qwen was more lenient than Claude (relevance 3.53 vs 2.71).
5. **Annotators and agent saw different inputs.** Annotators saw prior thread turns; the agent receives only the message. That depresses the agent's context and intent scores (F4, F5) independently of model quality.

## Which fixes are safe to make now

None of the five should be tuned against these results. F1, F2, F4 and F5 each need new labels or a dev experiment. F3's pattern gap and the empty-response retry are genuine reliability bugs definable without gold labels, but both change agent behaviour. Fixing either means the committed numbers no longer describe the shipped system, so a fix must come with a fresh, disclosed re-evaluation or be left for after submission.

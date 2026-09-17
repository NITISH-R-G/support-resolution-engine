# Final Taxonomy Candidate — Milestone 3, Round 3

**Status: NOT FROZEN. Awaiting approval.**
**Data:** AppleSupport **train split only**, English, n = 60,817. Dev, test pool and golden
candidates untouched. **Seed:** 20260910.

---

## 1. Corrected brand decision

### The tie

The `reply_classify` defect fix changed the selection evidence materially (details §2). On the
corrected evidence, **AmazonHelp and AppleSupport tie at rubric score 0.7246, exact to six
decimal places**. The composite score cannot separate them, and sort order is not a decision.

### Tie-break: definitions, and why each is relevant

Applied **lexicographically** — criterion 1 decides unless exactly equal, then 2, and so on.
No new composite score; no weights.

| # | Criterion | Exact definition | Why it is relevant here |
|---|---|---|---|
| 1 | **usable_grounding_evidence_pairs** | Absolute count of pairs whose reply is actionable **and** non-duplicate (SPEC §3.2.2 feature 13) | The assignment requires replies *grounded in how the brand historically resolved similar issues*. This is the direct count of pairs that can serve as grounding evidence. It is the binding constraint on the core deliverable |
| 2 | **English coverage** | `1 − non_english_reply_rate` | The system, judge and annotator all operate in English. Non-English pairs are unusable as grounding evidence and unlabellable by our annotator |
| 3 | **Intent diversity** | `distinct_intents_at_3pct` (clusters ≥3% share at fixed k, seed-pinned) | A taxonomy needs several distinguishable intents to be worth building |
| 4 | **Temporal / conversation coverage** | `distinct_months`, `distinct_conversations` | A temporal split needs spread; conversations are the unit of independence |

**Confirmation that none depends on model or evaluation performance.** All four are descriptive
corpus statistics computed before any agent exists. No classifier we evaluate, no retrieval
score, no generated reply, no judge output is an input.

One caveat stated plainly: criterion 1 depends on `reply_classify`, which *is* a classifier —
a deterministic lexical instrument used to measure the corpus. It is **not** the system under
test and its output is never scored. Its correctness is nevertheless load-bearing for this
decision, which is exactly why the defect that triggered this re-decision mattered so much.

### Result

| Criterion | AmazonHelp | AppleSupport | Winner |
|---|---|---|---|
| **1. usable_grounding_evidence_pairs** | **20,076** | **31,241** | **AppleSupport (1.56×, +11,165)** |
| 2. English coverage | 0.8342 | 0.9997 | *(not reached — would also favour AppleSupport)* |
| 3. Intent diversity | 11 | 7 | *(not reached — would favour AmazonHelp)* |
| 4. Months / conversations | 25 / 82,476 | 16 / 80,625 | *(not reached)* |

> ### **Corrected decision: `AppleSupport`** — decided at criterion 1.

**This is a re-decision, not a retention.** The prior decision was set aside; the corrected
evidence was re-run and the tie-break applied from scratch. It happens to name the same brand,
for a different and now-authoritative reason. Had criterion 1 favoured AmazonHelp, the project
would have switched.

**What is being traded away.** AmazonHelp offers 45% more raw volume, 11 intents vs 7, 25
months vs 16, and 2× the escalation-sensitive cases. AppleSupport is chosen because it has
1.56× the grounding evidence and is effectively monolingual (99.97% vs 83.4%). That trade is
real and belongs in the report.

---

## 2. Corrected brand-selection provenance

| | |
|---|---|
| **Original decision** | AppleSupport, 5 of 83 brands passed filters, rubric 0.6551 — `reports/brand_decision.json`, commit `2435687` |
| **Defect discovered** | `reply_classify` (a) stem-only verb patterns missed inflected forms (`restart` vs `restarting`); (b) the English gate rejected terse navigation instructions, systematically discarding the most actionable replies |
| **How found** | Building operational-handling profiles for this milestone — not by a failing test |
| **Impact** | `actionable_resolution_rate` +0.092 mean; `usable_grounding_evidence_pairs` +1,487 mean (max +21,853); brands passing filters **5 → 27**; top two tie exactly |
| **Corrected evidence** | `reports/brand_profiles_reclassified.json`, `reports/brand_decision_reclassified.json` |
| **Corrected decision** | **AppleSupport**, by lexicographic tie-break on criterion 1 |
| **Status of original** | **Preserved historically, superseded.** Not deleted, not rewritten |

This is recorded as a **methodology correction caused by a discovered defect**, not as
re-selection. `DECISION_LOG.md` D14 forbids re-selecting on downstream performance; correcting
a measurement instrument and re-deciding on the corrected inputs is a different act, and the
tie-break used no performance data.

---

## 3–5. Final proposed taxonomy

### Substantive intents (9 + fallback)

| # | Intent | Auto-handle candidate | Prevalence (floor) |
|---|---|---|---|
| 1 | `device_malfunction` | Yes | ≥ 9.28% |
| 2 | `battery_charging` | Yes | ≥ 7.70% |
| 3 | `apps_and_services` | Yes | ≥ 4.20% |
| 4 | `billing_and_subscription` | No | ≥ 4.05% |
| 5 | `connectivity` | Yes | ≥ 3.21% |
| 6 | `howto_information` | Yes | ≥ 3.13% |
| 7 | `complaint_feedback` | Varies | ≥ 2.37% |
| 8 | `account_access` | Yes, cautiously | ≥ 1.58% |
| 9 | `repair_order_replacement` | No | ≥ 1.53% |
| 10 | `other_unclear` | No | — |

Prevalences are **regex-probe floors, not estimates**: 62.9% of the split matched no probe
under priority-first assignment. They indicate relative scale only.

### Orthogonal attributes

```
intent             : exactly one of the 10 labels above
security_sensitive : bool   -> forces ESCALATE
context_sufficient : bool   -> ESCALATE / ask when false
```

### Fallback states — three distinct concepts, two axes

| Concept | Representation | Why |
|---|---|---|
| Substantive intent | intent label | The request is identifiable |
| **Insufficient context** | **`context_sufficient = false`** (attribute) | Not a *kind of request* but a failure to determine one. ~8% are bare acknowledgements ("@AppleSupport Ok"). A message can be `battery_charging` **and** context-insufficient — unrepresentable on one axis |
| Out-of-distribution | **`other_unclear`** (intent) | A real annotatable category: not a support request at all. The classifier must be able to predict it |

---

## 6. Operational-boundary audit

### The uncomfortable finding, stated first

Measuring all nine substantive labels against each other: **13 of 36 pairs show no material
handling difference** (max |diff| < 0.10 on every metric).

| Pair | max \|diff\| |
|---|---|
| billing ~ account_access | 0.048 |
| billing ~ complaint_feedback | 0.048 |
| battery ~ complaint_feedback | 0.049 |
| account_access ~ howto | 0.055 |
| billing ~ howto | 0.061 |
| complaint ~ device_malfunction | 0.067 |
| account_access ~ complaint | 0.072 |
| howto ~ device_malfunction | 0.079 |
| billing ~ battery | 0.088 |
| connectivity ~ apps_and_services | 0.088 |
| connectivity ~ device_malfunction | 0.091 |
| howto ~ complaint_feedback | 0.095 |
| apps_and_services ~ device_malfunction | 0.098 |

Only `repair_order_replacement` separates decisively from everything (0.221–0.288).

**Applied mechanically, "merge when handling is materially identical" collapses this taxonomy
to roughly two labels: `repair_order_replacement` and everything else.**

### Why I am not doing that

Because of what the instrument actually measures. Deflection rates sit between 0.28 and 0.55
for *every* label: on Twitter in 2017, Apple's dominant response to almost everything was
*acknowledge, ask one question, move to DM*. **The channel is the dominant variable and it
swamps intent-level differences.**

That gives handling-profile comparison **asymmetric evidential power**:

- When it **does** separate (repair, 0.288), that is strong positive evidence of a real
  operational boundary.
- When it **fails** to separate, that is **weak** evidence of sameness — absence of evidence,
  not evidence of absence — because the instrument has low discriminative power in a
  deflection-dominated channel.

Treating the null result as proof of sameness would be over-reading a low-powered instrument.
So handling profile is used as a **positive** criterion for keeping labels apart, and the
remaining labels are justified on the other two consumers SPEC §5 names — **resolution evidence
type** and **escalation policy** — not on public-channel behaviour alone.

**This limitation belongs in the report's "what is misleading about my headline number?"
section, and I will carry it there.**

### Per-label audit

| Intent | Routing | Retrieval | Resolution evidence | Escalation | Observable from message? | Handling-identical to |
|---|---|---|---|---|---|---|
| `device_malfunction` | Troubleshooting flow | Device symptom cases | Diagnostic steps | Only if safety-flagged | Yes | connectivity, apps, howto, complaint |
| `battery_charging` | Troubleshooting flow | Battery/power cases | Battery-specific steps, health checks | No | Yes — distinct vocabulary | billing, complaint |
| `connectivity` | Troubleshooting flow | Network cases | Network-reset steps | No | Yes | apps, device_malfunction |
| `apps_and_services` | Service-specific flow | Named-service cases | Per-service guidance | No | Yes — named service | connectivity, device_malfunction |
| `howto_information` | Information flow | How-to articles | Documentation link | No | Yes — interrogative, no fault | account, billing, complaint, device |
| `account_access` | Account flow | Recovery cases | Account-recovery procedure | If `security_sensitive` | Yes | billing, howto, complaint |
| `billing_and_subscription` | Financial flow | Billing cases | Charge explanation, refund policy | **Yes — money** | Yes | account, battery, howto, complaint |
| `repair_order_replacement` | **Logistics — out of channel** | Warranty/repair cases | Appointment, warranty status | **Yes — physical/financial** | Yes | **none (0.221 min)** |
| `complaint_feedback` | Feedback channel | Rarely groundable | Acknowledgement / feedback route | Sometimes | Yes — evaluative, no request | battery, howto, device, account, billing |
| `other_unclear` | No automated reply | None | None | Yes — never auto-handle | Yes | — |

### What justifies each surviving boundary

| Boundary | Justification | Type |
|---|---|---|
| `repair_order_replacement` vs all | 0.221–0.288 handling difference; routed out of channel at 55% | **Measured** |
| `billing_and_subscription` separate | Money is an escalation-policy category (SPEC §8) regardless of channel behaviour | **Policy** |
| `account_access` separate | Distinct resolution evidence (recovery procedure); gates the security attribute | **Resolution evidence** |
| `battery`/`connectivity`/`apps` vs `device_malfunction` | Different resolution evidence — battery-health checks, network resets and per-service guidance are not interchangeable, even where channel behaviour matches | **Resolution evidence** |
| `howto_information` separate | No fault reported; resolution is documentation, not diagnosis | **Resolution evidence** |
| `complaint_feedback` separate | Frequently no groundable resolution exists; routed to feedback | **Resolution evidence** |
| `other_unclear` separate | Must never be auto-answered | **Policy** |

---

## 7. Security design — attribute wins, decisively

The evidence favours the attribute model over a separate `account_security_compromise` intent,
and more strongly than I expected.

| Finding | Number | Consequence |
|---|---|---|
| Security-flagged messages, all intents | **340** (0.56%) | **4.1× larger** than the 82 an account-scoped intent would capture |
| **Share sitting OUTSIDE the account topic** | **306 of 340 — 90.0%** | An account-scoped intent captures only **10%** of security-sensitive traffic |
| Within account topic: flagged vs not | actionable **−0.214**, substantive **−0.256**, links out **+0.172** | All **material**; a stronger signal than the earlier access-vs-compromise intent split, whose CIs nearly touched zero |

The 90% figure is decisive. Security concern appears as stolen devices (`device_malfunction`),
fraudulent charges (`billing_and_subscription`), scam apps (`apps_and_services`) and phishing
questions (`howto_information`). **Encoding safety inside an intent label would lose nine tenths
of the safety signal**, because those messages would be labelled by topic and silently drop
their flag.

It also resolves the rare-label problem the previous round flagged: 0.56% is thin but
**measurable**; 0.13% was not.

> **Recommendation: drop `account_security_compromise` as an intent. A forgotten password and a
> suspected takeover are both `account_access`, separated by `security_sensitive`.** SPEC §8's
> hard rule fires on the attribute, across every intent.

---

## 8. Labels merged, rejected, reframed

| Label | Decision | Basis |
|---|---|---|
| `software_update_issue` | **Rejected** | Causal attribution. `battery_vs_update` showed **zero** significant differences on any metric |
| `account_security_compromise` | **Reframed → attribute** | 90% of security traffic sits outside the account topic |
| `privacy_data` | **Merged → `complaint_feedback`** | Routed to the Feedback channel in the data |
| `phishing_scam_verification` | **Merged → `howto_information`** | Lowest deflection (0.274); answered in-channel with an article. It is an information request |
| `needs_more_context` | **Reframed → routing state** | Not a kind of request; ~8% bare acknowledgements |
| `security_sensitive` | **New attribute** | Cuts across intents |

---

## 9. Remaining rare-label concerns

1. **`security_sensitive` at 0.56%** → ~1–2 examples in a 200-item golden set at natural
   prevalence. **Deliberate stratified over-sampling is required**, with prevalence reweighted
   at reporting time and the over-sampling documented. Without it the safety-critical path is
   untested.
2. **`repair_order_replacement` (≥1.53%) and `account_access` (≥1.58%)** → roughly 3 examples
   each. Per-class metrics will have very wide intervals and must be reported with them, never
   as bare point estimates.
3. **`other_unclear` prevalence is unmeasured** — probes cannot detect "not a support request".
   Only human labelling will establish it.
4. **62.9% of the split matched no probe.** All prevalences are floors.

---

## 10. Annotation implications

- Annotators assign **one intent plus two booleans**, not a single label. The guide must show
  the axes are independent, with worked examples of each combination.
- **The brand's reply must never be consulted when labelling.** It was used here for taxonomy
  *design* on train data; using it to assign labels would contaminate gold.
- `security_sensitive` needs a bright-line rule: flag on *suspicion expressed by the customer*,
  not on confirmed compromise, since confirmation is unavailable from the message.
- `context_sufficient` needs its own rule: false when the specific request cannot be
  determined, even if the topic is clear.
- Known confusion pairs requiring explicit tie-breaks: `device_malfunction` vs
  `apps_and_services` (0.098), `connectivity` vs `apps_and_services` (0.088),
  `howto_information` vs `device_malfunction` (0.079), `complaint_feedback` vs any topic label.

---

## 11–12. Tests and results

New this round: `tests/test_handling_profile.py` (19) and 6 regression tests in
`tests/test_reply_classification.py` for the two defects.

```
306 passed, 0 failed
```

No classifier built. No golden set. No model evaluation. No LLM API call — **$0.00 spent**.

---

## 13. Exact taxonomy that would be frozen

```yaml
taxonomy_version: 0.3.0-candidate      # NOT frozen
brand: AppleSupport                    # corrected decision, tie-break criterion 1
derived_from: train split only, n=60,817 English
seed: 20260910

intents:                               # exactly one per message
  - device_malfunction
  - battery_charging
  - apps_and_services
  - billing_and_subscription
  - connectivity
  - howto_information
  - complaint_feedback
  - account_access
  - repair_order_replacement
  - other_unclear

attributes:                            # orthogonal, independently annotated
  security_sensitive:  bool            # true -> ESCALATE (SPEC 8 hard rule)
  context_sufficient:  bool            # false -> ESCALATE / ask

rejected:
  software_update_issue:        causal attribution; zero handling difference
  account_security_compromise:  reframed as security_sensitive attribute
  privacy_data:                 merged into complaint_feedback
  phishing_scam_verification:   merged into howto_information
  needs_more_context:           reframed as context_sufficient attribute
```

**10 intents + 2 attributes.** Within SPEC §5's 6–10 guidance counting `other_unclear` as the
required fallback.

---

## 14. What needs approval before freezing

1. **Corrected brand decision: AppleSupport**, by lexicographic tie-break on absolute usable
   grounding evidence (31,241 vs 20,076), with the original decision preserved and marked
   superseded.
2. **Security as an attribute, not an intent** — dropping `account_security_compromise`.
3. **Acceptance of the low-power finding**: most label boundaries are *not* justified by
   measured handling differences, because the deflection-dominated channel cannot resolve them.
   They rest on resolution-evidence type and escalation policy instead. This is a real
   limitation and I would carry it into the report's "misleading headline number" section
   rather than leave it implicit.

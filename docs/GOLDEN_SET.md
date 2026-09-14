# The Golden Evaluation Set — Protocol

**Status:** protocol FROZEN 2026-09-11, before any example was drawn from the test pool.
**Taxonomy binding:** v0.3.0, hash `613f5dfec125...`.
**Labels in existence at freeze time: zero.**

This document is the protocol, written and frozen *first*. Everything downstream — the
sampler, the annotation tool, the harness — implements what is written here. Deciding the
protocol after seeing which examples came out is how a sampling plan becomes a way to justify
a number.

---

## 0. The one rule everything else serves

> **A label is gold only if a human typed it.**

No weak label, classifier prediction, LLM output, retrieval result or heuristic may ever be
promoted into the gold column. The repository therefore keeps **four provenance classes**, and
every labelled quantity in every artifact carries exactly one of them:

| Provenance | Meaning | May be used as gold? |
|---|---|---|
| `UNLABELED` | No label exists. The field is absent, not empty-string. | — |
| `WEAKLY_LABELED` | Produced by deterministic labelling functions (`classifier/weak_labels.py`) | **Never** |
| `MODEL_GENERATED` | Produced by a model — classifier, LLM, judge | **Never** |
| `HUMAN_LABELED` | Typed by a named human annotator through `scripts/annotate_golden.py` | **Yes, only this** |

`golden.store.load_gold()` **raises** when asked for gold and finds anything else. The harness
cannot be talked into scoring an unlabelled column; it fails instead.

---

## 1. Population

**The held-out test pool, and nothing else.** `temporal_split()` reserves the chronologically
last ~20% of AppleSupport conversations. Train and dev are excluded by construction, not by
convention: the sampler is handed `splits.test_pool` and never sees the other two.

The test pool was **untouched** until this protocol was frozen. Nothing has been trained on
it, no threshold fitted on it, no error analysis run over it.

---

## 2. Size

**200 examples.** Within the assignment's 150–250 band; the midpoint is chosen because the
binding constraint is human annotation hours, and 200 blind labels plus a 40-example
re-labelling pass is what one annotator can do carefully.

At n=200 a bootstrap 95% CI on an accuracy near 0.9 is roughly ±4 points. That is a limitation
of the deliverable, not a detail: **any headline number from this set is a point estimate with
a four-point band around it**, and reporting it bare would be misleading.

---

## 3. Sampling design

Two disjoint components, drawn in this order, from one seeded RNG (`seed = 20260911`).

### 3.1 Random reservoir — 50 examples

Uniform, unstratified, without replacement from the whole test pool.

This is the **unbiased core**. Its purpose is to survive the failure of everything in §3.2: if
the stratification proxies are wrong about what is hard, the reservoir is still a random
sample of real traffic and still supports an honest — if wide — estimate of production
behaviour.

### 3.2 Stratified over-sample — 150 examples

Drawn from the test pool *minus* the reservoir, allocated across six strata:

| Order | Stratum | n | Why it is over-sampled |
|---|---|---|---|
| 1 | `security_signal` | 25 | Safety-critical and rare (~3% of traffic). A representative sample would contain ~6, too few to say anything about the gate that matters most. |
| 2 | `thin_context` | 25 | The weak rules judge the request undeterminable. Tests `context_sufficient`, the attribute that drives ~8% of escalations. |
| 3 | `multi_signal` | 20 | Two or more conflicting labelling-function votes — the multi-intent cases the taxonomy's tie-breaks exist for. |
| 4 | `rule_abstain` | 40 | The labelling functions produced no intent vote *and* none of the above explains why. This is the region where every dev figure the project has reported is blind, so it carries the most information about what the classifier actually does. |
| 5 | `long_message` | 15 | 40 words or more. Long messages bury the request and are where retrieval degrades. |
| 6 | `typical` | 25 | Everything else, so the hard strata are not compared against nothing. |

Strata are assigned in **that priority order, first match wins**, so every example belongs to
exactly one and the allocation is well defined. The order runs rarest-and-most-critical first:
`security_signal` must not be starved by a broader stratum absorbing its members, and
`thin_context` is placed above `rule_abstain` deliberately — a message the rules abstain on
*because it is too short* is a different and less interesting phenomenon than one they abstain
on despite having plenty to read, and separating them keeps `rule_abstain` pointed at the
genuine blind spot. If a stratum holds fewer eligible examples than its
quota, the shortfall is recorded in the manifest and redistributed to `typical` — never
silently absorbed.

### 3.3 The stratification proxies are weak labels, and that is load-bearing

`rule_abstain`, `security_signal`, `thin_context` and `multi_signal` are computed from
`classifier/weak_labels.py` — the same deterministic rules that produced the training labels.

**This is a deliberate, bounded use, and it is the only place a weak signal touches the golden
set.** Three constraints make it safe:

1. It is frozen **before** sampling begins and never revisited. No example is added, removed or
   re-stratified after the first draw. Adjusting a sampling frame once you can see what it
   caught is a garden-of-forking-paths error.
2. It affects **which examples are shown**, never **what they are labelled**. The stratum is
   not written into the annotation file the annotator reads, and the annotator never learns it.
3. The 50-example reservoir is drawn **first and independently**, so the weak rules cannot make
   the whole set unrepresentative.

What it *can* still do is leave a blind spot: a phenomenon invisible to the rules is
over-represented in `rule_abstain` only by accident. That is a real limitation and belongs in
the report.

### 3.4 Inclusion probabilities are recorded, so representative estimates stay computable

A deliberately hard-skewed sample does not estimate production performance. Every example
therefore carries its known inclusion probability and the inverse-probability weight
`1 / p_inclusion`, computed from the design rather than fitted.

Three estimates are then reportable from one annotation effort, and the report must give all
three rather than the flattering one:

- **Reservoir-only** — unbiased for real traffic, n=50, wide CIs.
- **Per-stratum** — precise where it matters, generalises to nothing on its own.
- **IPW-reweighted whole set** — the point estimate for production, with the caveat that
  weights magnify variance.

### 3.5 Reproducibility

`python scripts/build_golden_candidates.py` is deterministic: fixed seed, sorted iteration,
no set iteration in any ordering path. The manifest records seed, git SHA, taxonomy hash,
test-pool size, per-stratum eligible and drawn counts, and the SHA-256 of the candidate file.
Re-running it on the same corpus reproduces the same 200 ids, and the script **refuses to
overwrite** an existing candidate file — re-sampling after annotation has started would
silently discard human work.

### 3.6 Eligibility filter — added after the guards fired on a real draw

The first real draw tripped two leakage guards. Both were right, and neither was weakened.

```
LeakageError: duplicate customer message across splits
  (golden pair '2592237__2592238'): 'yes'

LeakageError: near-duplicate across splits at cosine 0.979 >= 0.9
  golden: '7plus ios 11 1 2'
  corpus: '7plus ios 11 1'
```

A test-pool message is therefore **eligible only if** its normalised text neither appears
verbatim in the retrieval corpus nor is a near-duplicate of one at cosine ≥ 0.90 — checked
with the same analyser the guard uses, so the filter genuinely pre-empts the guard rather than
approximating it.

**The filter runs before the draw, never after.** Removing examples once you can see which
ones came out is how a sampling frame becomes a way to pick a result. Measured on the real
test pool:

| | |
|---|---|
| Test pool | 22,378 |
| Excluded, exact duplicate | 546 (2.44%) |
| Excluded, near duplicate | 232 |
| **Eligible** | **21,600** |
| Exclusions by stratum | `thin_context` 333, `rule_abstain` 193, `typical` 15, `security_signal` 3, `multi_signal` 2 |

**This biases the set, and the bias is reported rather than absorbed.** It removes roughly
42% of the `thin_context` stratum — every customer types "yes", "ok" and "thanks" in the same
words, and bare device-plus-version fragments collide too. So the golden set
**under-represents ultra-short messages**, and therefore under-represents
`context_sufficient = false`, relative to production traffic. Any escalation rate estimated
from this set is an underestimate of the rate driven by thin context in real traffic.

The stratum still holds 452 eligible members against a quota of 25, so no stratum was starved;
the cost is representativeness, not coverage.

---

## 4. What the annotator sees, and what is structurally withheld

**Shown:**

- an opaque `pair_id`
- the customer message, normalised and PII-masked exactly as the agent sees it
- the conversation turns that **precede** that message, if any
- the message timestamp

**Withheld — and withheld structurally, not by discipline:**

| Withheld | How |
|---|---|
| The brand's actual reply to this message | **Never written into the candidate file.** It stays in the corpus and is joined back by `pair_id` at evaluation time. The annotation tool cannot show what it does not have. |
| Weak label, stratum, inclusion weight | Stored in the manifest and the sampling record, not in the file the tool reads. |
| Classifier prediction and confidence | Never computed during annotation; the tool imports no classifier. |
| Retrieved evidence, generated reply | Never computed during annotation; the tool imports no retriever or generator. |

The brand's reply is withheld because it is *downstream* of the intent. Reading "let us take
this to DM" and inferring "so it must be escalation-worthy" would make the gold label a
function of the very behaviour the agent is being scored against. `tests/test_golden_schema.py`
asserts the candidate record has no reply field and no label field at all.

---

## 5. Annotation schema

Per example, the annotator supplies:

| Field | Type | Notes |
|---|---|---|
| `intent` | one of taxonomy v0.3.0's 10 | Validated on entry; an out-of-taxonomy value is rejected, never coerced |
| `security_sensitive` | bool | Flag on the customer's *suspicion*, never on confirmation |
| `context_sufficient` | bool | Whether the specific request is determinable |
| `should_escalate` | bool | **Independent human routing judgement**, not derived from the two above |
| `expected_resolution_kind` | `self_serve_steps` / `information` / `human_action_required` / `no_resolution_possible` / `unclear` | The evidence expectation: what *kind* of thing would resolve this |
| `reference_resolution` | free text, optional | In the annotator's own words, what a good reply must contain. Written **without** seeing the brand's reply, so it is a genuine reference rather than a paraphrase of the system's target. |
| `is_ambiguous` | bool | See §6 |
| `alternative_intent` | intent or null | The runner-up, when ambiguous |
| `label_confidence` | `high` / `medium` / `low` | |
| `notes` | free text, optional | |

Automatically recorded, never typed: `annotator_id`, `timestamp_utc`, `annotation_version`,
`taxonomy_version`, `taxonomy_hash`, `seconds_spent`, `pass_number`, `provenance =
HUMAN_LABELED`.

### Why `should_escalate` is annotated independently

The deterministic policy already *derives* escalation from intent and attributes. If gold
escalation were derived the same way, the routing evaluation would be checking the policy
against itself and would be guaranteed to score well. Annotating it independently is the only
way the question "is this policy right?" has an answer.

Expect disagreement between the human judgement and the derived policy. **That disagreement is
a finding, not a labelling error**, and the rate is reported.

---

## 6. Ambiguity: an explicit mechanism, not a forced choice

Automated clustering of this data separates poorly (silhouette < 0.09). Forcing a single label
on a genuinely two-sided message manufactures false precision and then scores a model against
it.

So an annotator who cannot decide sets `is_ambiguous = true`, records the runner-up in
`alternative_intent`, and still gives a primary intent — the best available reading, not a
coin flip dressed as a decision.

Metrics are then reported **three ways**: strict (primary only), lenient (credit for either
reading on ambiguous examples), and excluding ambiguous examples entirely. The gap between
strict and lenient is itself a measurement of how much of the task is genuinely underdetermined.

Ambiguous examples are **never dropped from the set**. Dropping the hard cases is the most
common way an evaluation flatters itself.

---

## 7. Passes, disagreement, and what self-agreement does and does not mean

> **As executed (2026-09-14) — this differs from the plan below.** Pass 1 was **not** blind for
> all 200. Commit `3822b44` restored SPEC §9.2: 160 examples showed a
> `meta-llama/llama-3.3-70b-instruct` suggestion (159 accepted, 1 corrected) and 40 were entered
> blind. **Pass 2 (self-agreement) and pass 3 were not performed**, so intra-annotator
> reliability is unmeasured. Anchoring is not measured either: the blind and assisted examples
> are different messages, and no within-subject comparison exists. Authoritative counts are in
> `data/golden/GOLDEN_LOCK.json`.

| Pass | What | Blind? |
|---|---|---|
| **1** | All 200, one annotator | **Fully blind.** No suggestion, no model output of any kind. |
| **2** | 40 or more re-labelled at least 24h later, first-pass labels not shown | Blind |
| **3** *(optional)* | 60 re-labelled **with** a weak-label suggestion shown | Not blind — that is the point |

**Disagreement between passes 1 and 2** is adjudicated by the annotator on a third viewing,
seeing both prior labels. The adjudicated label becomes gold; both original labels are kept in
the log, so the adjudication can be audited and the disagreement rate cannot be quietly
deleted.

**Deviation from SPEC §9.2, recorded deliberately.** SPEC specified 160 suggested / 40 blind —
a *between-subject* anchoring measurement. Pass 1 here is blind for all 200 instead, which is
strictly better data, and anchoring bias is measured *within-subject* in pass 3 by comparing
the same annotator's suggested labels against their own blind labels on the same 60 examples.
Within-subject removes the confound that the two groups are different examples. The cost is
that pass 3 must actually happen; if it does not, **anchoring bias is unmeasured and is
reported as unmeasured**, not estimated.

### The framing rule on self-agreement — binding

Self-agreement measures **annotation consistency (reliability)**. It is **not** label accuracy
and **not** a ceiling on achievable model performance. A consistent annotator can be
consistently wrong.

Do not write "measured ceiling on achievable accuracy". The defensible claim is narrower:
*intents whose labels are unstable on re-labelling carry uncertainty beyond the sampling error
in the confidence interval.* See `DECISION_LOG.md` D12.

---

## 8. Leakage controls

Enforced by guards that **raise**, run by the sampler at build time and by the harness at
evaluation time:

| Control | Guard |
|---|---|
| No golden id in the retrieval corpus | `assert_no_id_overlap` (pair, conversation **and customer**) |
| No identical normalised text across the boundary | `assert_no_text_duplicates` |
| No near-duplicate query | `assert_no_near_duplicates` |
| The golden example's own reply absent from the corpus | `assert_no_response_leakage` |
| Every corpus item predates its golden query | `assert_temporal_split` |
| Judge/pre-annotator never shares a family with the system under test | `assert_independent_models` |

Two further controls are structural rather than asserted:

- The candidate file **contains no reply text**, so the answer cannot leak into annotation.
- The candidate file **contains no label field**, so there is nothing for a script to fill in.

---

## 9. Lock

Once pass 1 is complete the set is locked: the candidate file's SHA-256 is pinned in the
manifest and in `VERIFICATION.json`. Post-lock changes require a version bump and a documented
reason. Thresholds are fitted on **dev**, never here.

---

## 10. Honesty constraint

If annotation stops short of 200, the report states the number actually labelled and the CIs
that size implies. **Missing labels are never backfilled** — not with weak labels, not with an
LLM, not with the classifier. An unlabelled example stays `UNLABELED` and the harness refuses
to score it.

`scripts/build_golden_candidates.py` writes **200 explicitly unlabelled records**. That file is
not a golden set. It becomes one only as a human fills it in.

---

## 11. Known limitations

1. **One annotator.** Inter-annotator agreement cannot be measured; only intra-annotator
   consistency. A second annotator is the single highest-value addition.
2. **The annotator is not a support agent** and has no access to Apple's internal policy, so
   `should_escalate` reflects an informed outsider's judgement, not operational ground truth.
3. **Deliberately unrepresentative by design** — see §3.4. Only the reservoir and the
   IPW-reweighted estimate speak to production rates.
4. **Stratification inherits the weak rules' blind spots** (§3.3).
5. **Single-turn framing.** Thread context is shown but the agent currently ignores it, so
   gold labels may reflect context the system cannot use — biasing *against* the system, which
   is the safe direction.
6. **2017 Twitter data during the iOS 11 launch.** Intent distribution is event-dominated and
   does not transfer to another period or channel.

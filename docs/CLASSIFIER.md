# Classifier — Milestone 4

**Status:** built and evaluated on dev. **Not approved.** Taxonomy v0.3.0 untouched.
**Test pool:** never read. **Golden set:** not started. **LLM API calls:** zero, $0.00.

---

## 0. Read this before reading any number

Every metric in this milestone is a **rule-recovery score, not accuracy.**

No human labels exist. The golden set is drawn from the held-out test pool and has not been
built, so training *and* dev labels both come from the same deterministic labelling functions
in `classifier/weak_labels.py`. A model trained on those labels and scored against those labels
is being measured on **agreement with my own heuristics**.

Two consequences follow, and both matter more than the headline figure:

1. **The comparison is structurally biased toward lexical models.** The labelling functions are
   regexes, so TF-IDF is rewarded for reproducing them while a semantic model that correctly
   generalises to a paraphrase the regex misses is *penalised* for it. The dev scores cannot
   establish which model classifies intent better — only which better imitates the labeller.
2. **The labeller is demonstrably wrong on real cases** (§8). Where the model disagrees with
   it, the model is sometimes right. True accuracy is therefore unknown and could be higher or
   lower than 0.93.

This is the same circularity documented in `PUBLIC_REPO_COMPARISON.md` §2.2, where a competing
repository reported 0.955 accuracy that turned out to equal its gold-vs-model agreement rate.
The difference here is that it is labelled, measured, and reported as the limitation it is.

---

## 1. Classifier contract

`classifier/contract.py`. Every model returns a `Prediction`:

```python
intent: str                 # exactly one frozen-taxonomy label
confidence: float           # [0, 1], finite
security_sensitive: bool    # strict bool
context_sufficient: bool    # strict bool
model_name / model_version  # provenance
taxonomy_version / hash     # binds v0.3.0
source: MODEL | RULE | ABSTAINED
evidence: tuple[str, ...]   # deterministic explanation
```

**Fails closed.** An out-of-taxonomy intent, a confidence outside [0,1], NaN, infinity, or a
non-boolean attribute raises `ContractError`. There is **no default intent**: omitting it is a
`TypeError` rather than a silent fall back to `other_unclear`, because a wrong-but-plausible
label flowing into a routing decision is worse than a crash.

**Binds the taxonomy.** Every prediction carries the frozen hash. A prediction made against
different label definitions is not comparable to one made against these, and the mismatch
raises rather than passing unnoticed.

**Attributes dominate.** `must_escalate` returns True whenever `security_sensitive`,
`not context_sufficient`, abstention, or an escalation-sensitive intent applies. Confidence
never overrides it — tested at 0.0, 0.5, 0.99 and 1.0 across every intent.

## 2. Training-label methodology — WEAK SUPERVISION

| | |
|---|---|
| Source | Deterministic labelling functions, `classifier/weak_labels.py` |
| Human labels | **None. Zero.** |
| Marking | Every label carries `is_weak=True`, unsettable; every artifact repeats the warning |
| Train coverage | **40.5%** (27,379 of 67,626) — 59.5% abstained |
| Conflict rate | 9.6% |
| Abstentions | Dropped, not defaulted |

Abstentions are **excluded** from training rather than labelled `other_unclear`: teaching the
model that "unrecognised" is a category would make it predict `other_unclear` for anything
unfamiliar, which is not what that label means.

**Double circularity, stated plainly.** The frozen taxonomy was itself derived from the train
split, and these labelling functions were written against that taxonomy and applied to that
split. Nothing here is independent evidence about the corpus.

**Weak label distribution (train):** `device_malfunction` 30.1%, `battery_charging` 18.4%,
`apps_and_services` 12.9%, `billing_and_subscription` 9.1%, `connectivity` 8.5%,
`howto_information` 7.8%, `complaint_feedback` 6.2%, `repair_order_replacement` 3.7%,
`account_access` 3.5%, **`other_unclear` 0.0%**.

`other_unclear` receives **zero** training examples — the rules never assign it. The model
therefore cannot predict it, and it cannot be evaluated. See §14.

## 3–5. Results — agreement with weak labels, not accuracy

| Model | Agreement | Macro-F1 | Fit (s) | ms/msg | ECE |
|---|---|---|---|---|---|
| **A: majority** | 0.3273 | 0.0548 | 0.0 | 1.3 | — |
| **B: TF-IDF + LogReg** | **0.9298** | **0.9144** | 20.4 | 8.6 | 0.266 |
| Candidate: embeddings + LogReg | 0.8170 | 0.7974 | 636.4 | 42.7 | **0.067** |

Dev: 9,804 messages, 3,990 evaluable after the coverage filter.

Baseline A's split — 0.327 accuracy against 0.055 macro-F1 — is the clearest illustration of
why accuracy alone is misleading on a skewed distribution.

**Per-class, TF-IDF:**

| Intent | P | R | F1 | n |
|---|---|---|---|---|
| device_malfunction | 0.951 | 0.969 | 0.960 | 1306 |
| battery_charging | 0.969 | 0.917 | 0.942 | 516 |
| apps_and_services | 0.918 | 0.927 | 0.923 | 607 |
| account_access | 0.911 | 0.931 | 0.921 | 188 |
| billing_and_subscription | 0.962 | 0.879 | 0.919 | 348 |
| connectivity | 0.936 | 0.903 | 0.919 | 309 |
| complaint_feedback | 0.891 | 0.913 | 0.902 | 241 |
| howto_information | 0.844 | 0.932 | 0.886 | 308 |
| repair_order_replacement | 0.866 | 0.850 | 0.858 | 167 |
| **other_unclear** | **0.000** | **0.000** | **0.000** | **0** |

## 6. Chosen classifier: TF-IDF + logistic regression

**Not chosen because it scored highest.** That score is structurally inflated for lexical
models (§0), so it is explicitly *not* the deciding factor. Chosen on operational grounds:

| Criterion | TF-IDF | Embeddings |
|---|---|---|
| **Explainability** | Token-level evidence from actual coefficients | "embedding-similarity" only |
| **Latency** | 8.6 ms/msg | 42.7 ms/msg (**5×**) |
| **Fit time** | 20 s | 636 s (**31×**) |
| **Determinism** | Fully | Depends on the model build |
| **Cost / deps** | None | 90 MB model, torch |
| **Calibration** | ECE 0.266 | **ECE 0.067** |

Embeddings win decisively on calibration, and that is a real cost of this choice. It is
accepted because miscalibration is fixable post-hoc (§9) while missing explanations are not —
a support agent that escalates must be able to say *why*, and token evidence provides that
where embedding proximity does not.

**This decision should be revisited once human labels exist**, since the metric that currently
favours TF-IDF is precisely the one that is biased.

## 7. Security classifier — independent of intent

`classifier/attributes.py::SecurityDetector`. Deterministic rule, **never sees the intent**
(asserted by a signature test). Detects across account, billing, device, apps, phishing/scam,
stolen goods and fraud — verified on all eight categories.

**Why a rule and not a model.** The only training signal available is the rule itself, so a
model trained to imitate it could only lose recall while adding the appearance of learning. On
a path where a miss means auto-answering a live account takeover, a rule with known behaviour
beats a model with unknown recall. Revisit once human labels exist.

Tuned for **recall over precision**: a false flag costs one unnecessary escalation, a miss can
mean auto-answering a compromise.

**Dev security positive rate: 0.65%.** The reported precision/recall of 1.0 is a **tautology** —
the detector and the weak labeller share their patterns, so this confirms the detector
reproduces the labelling function and nothing more. It is reported so the tautology is visible
rather than mistaken for validation.

## 8. Context classifier

`ContextDetector`. Fires on bare acknowledgements, isolated version strings, lone device names,
and messages under four content words. Dev context-insufficient rate: **3.5%**.

When context is insufficient the pipeline **abstains** rather than inheriting a confident label
from the training prior — "iPhone 7" must not become `battery_charging` at high confidence.
Same tautology caveat as §7 applies to its 1.0 figures.

## 9. Calibration — dev only, never the test pool

TF-IDF ECE is **0.266**, but the direction matters more than the magnitude: it is
**under-confident** in every bin.

| Confidence bin | Mean confidence | Agreement | Gap |
|---|---|---|---|
| [0.2,0.3) | 0.261 | 0.704 | **+0.443** |
| [0.4,0.5) | 0.451 | 0.894 | +0.443 |
| [0.6,0.7) | 0.649 | 0.956 | +0.307 |
| [0.9,1.0) | 0.952 | 0.992 | +0.041 |

Under-confidence is the **safe** direction for this system: it abstains and escalates more
often than strictly needed. Over-confidence would silently auto-handle cases it should not.

**Selective prediction (TF-IDF):**

| Threshold | Coverage | Agreement |
|---|---|---|
| 0.0 | 100.0% | 0.930 |
| 0.5 | 75.5% | 0.967 |
| 0.7 | 46.6% | 0.981 |
| 0.9 | 16.5% | 0.992 |

**No threshold is being fixed yet.** Choosing one requires the expected-cost model in SPEC §8,
whose `C_bad/C_human` ratio is swept rather than assumed, and it must be fitted against labels
that mean something. Fixing a threshold against weak labels would optimise for agreement with
a regex.

## 10. Dev disagreement analysis

**280 disagreements, 7.0% of 3,990 evaluable.** Called disagreements, not errors: both sides
are fallible.

| Category | n | Share |
|---|---|---|
| multi_intent | 168 | 60.0% |
| semantic_overlap (documented in tie-breaks) | 74 | 26.4% |
| low_confidence_ambiguous | 18 | 6.4% |
| rare_intent | 12 | 4.3% |
| lexical_trap | 5 | 1.8% |
| security_adjacent | 3 | 1.1% |

That 26.4% fall on pairs the frozen taxonomy **already documents** as confusable is mild
evidence the tie-breaks anticipated the right boundaries.

### The important finding: the labeller is wrong on real cases

> `weak=billing_and_subscription  model=complaint_feedback  conf=0.40`
> *"[tweet-text redacted: tweet_id=2285175 sha256=e890b43c8ea7e218]"*

**"two charges" is battery charging.** My billing regex matches `charg\w*` and fired on it. The
model is right; the weak label is wrong. Similarly:

> `weak=complaint_feedback  model=howto_information`
> *"[tweet-text redacted: tweet_id=2402186 sha256=3e88aed1efcb0626]"*

**"Police complaint"** tripped the complaint pattern. Both labels are wrong — this is a stolen
device. The safety attribute *did* fire, so the escalation path held even though the intent was
wrong: exactly the behaviour the orthogonal-attribute design exists to produce.

**My adjudication heuristic is uninformative and I am not reporting its output as a finding.**
It returned "labeller looks right" 42.5% and "unclear" 57.5%, and "model looks right"
**zero** times — which is structurally impossible to avoid, because it reuses the same patterns
the labeller uses. It cannot detect the very cases shown above. Recorded as a defect in the
analysis, not as evidence.

**No taxonomy change was made.** Nothing here indicates a taxonomy problem; the defects are in
the labelling functions, which are not frozen.

## 11–12. Tests

| File | Tests |
|---|---|
| `test_classifier_contract.py` | 44 |
| `test_weak_labels.py` | 46 |
| `test_classifier_models.py` | 32 (2 slow) |
| `test_classifier_attributes.py` | 38 |

Suite total: **481 passed, 3 skipped, 0 failed.**

## 13. Files

`src/hiver_support/classifier/{contract,weak_labels,models,attributes,pipeline}.py` ·
`tests/test_classifier_{contract,models,attributes}.py`, `tests/test_weak_labels.py` ·
`scripts/{train_classifier,evaluate_dev,analyse_dev_errors}.py` ·
`reports/classifier_{dev_results,dev_errors,training_manifest}.json`

## 14. Known limitations

1. **Every number is a rule-recovery score.** No statement about real classification quality is
   possible until human labels exist. This is the dominant limitation.
2. **The weak labeller has demonstrable defects** (§10) — `charg\w*` collides between billing
   and battery; `complain\w*` fires on "Police complaint". These corrupt training labels. Not
   patched here: fixing one regex invites the question of how many others are wrong, and the
   answer is unknowable without human labels. **That is the point about weak supervision, and
   the reason the golden set matters.**
3. **`other_unclear` is untrained and unevaluated** (0 examples). The model cannot predict it.
   The pipeline reaches it only via context-abstention.
4. **Weak coverage is 40.5%.** The remaining 59.5% is unlabelled and unmodelled; the corpus may
   look different there.
5. **Attribute metrics are tautologies** (§7, §8) — detector and labeller share patterns.
6. **Model selection rests on a biased metric**, so it was made on operational grounds instead.
7. **TF-IDF is poorly calibrated** (ECE 0.266), mitigated only by being wrong in the safe
   direction.
8. **No threshold is fixed**, so selective prediction is not yet operational.
9. **The 2017 iOS 11 event dominates the corpus**; lexical features may not transfer.

## 15. Next milestone

**Not the classifier.** The blocking dependency is labels: everything above is measured against
heuristics. The highest-value next step is the **hand-labelled golden set** from the held-out
test pool, which would for the first time make an honest accuracy statement possible, allow the
model choice to be re-examined on an unbiased metric, and let a confidence threshold be fitted
against something meaningful.

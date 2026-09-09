# Public Repository Forensics — Hiver SDE Intern Take-Home

**Date of analysis:** 2026-09-09
**Analyst:** lead engineer (this project)
**Method:** all seven repositories cloned in full and inspected file-by-file. Every claim below is
backed by a file path and line reference, or by re-parsing the repository's own committed artifacts
with `pandas`. **No claim in this document is taken from a README.**

---

## 0. Headline conclusion

> **None of the seven candidates actually produced a hand-labelled golden set.**

That is the assignment's centrepiece deliverable (#2), and it is unmet in all seven repositories.
Three repositories are *honest* about this. Three present machine-generated labels as human labels.
One is partially honest.

Because every reported headline metric in the field is computed against machine-generated labels,
**every published accuracy number in these seven repositories is either circular or unverifiable.**
In two cases I was able to prove the circularity numerically.

This is the single largest differentiation opportunity available to us: doing the human labelling
honestly, and measuring what it costs us, beats the entire field on the dimension the assignment
says it cares about most ("The proof is worth more than the system").

---

## 1. Provenance and licensing — read this first

| Repo | Created | License | Reuse status |
|---|---|---|---|
| ChetanChavan45/hiver-support-agent | 2026-09-09 12:39Z | **none** | ❌ All rights reserved |
| shubhamsharmass0001/hiver-sde-intern-assignment | 2026-09-09 10:29Z | **none** | ❌ All rights reserved |
| AshrafAhmed9/hiver-support-agent-takehome | 2026-09-09 10:50Z | **none** | ❌ All rights reserved |
| MitDesai9118/hiver-apple-support-agent | 2026-09-09 13:50Z | **none** | ❌ All rights reserved |
| Himanshu0518/hiver-ai-support-agent | 2026-09-09 14:29Z | **none** | ❌ All rights reserved |
| Vishwateja-123/hiver-support-agent | 2026-09-09 12:53Z | **MIT** | ⚠️ Legally reusable with attribution |
| anujksharma03/hiver-support-agent | 2026-09-09 09:36Z | **none** | ❌ All rights reserved |

Two facts dominate the reuse strategy:

1. **All seven repositories were created on the same day this analysis was run** — within a
   five-hour window. These are not prior art or reference implementations. They are *concurrent
   submissions by competing candidates for the same role*.
2. **Six of seven carry no license file.** Under the Berne Convention and GitHub's own terms,
   absence of a license means **all rights reserved**: public visibility grants the right to view
   and fork within GitHub, not to copy code into a derivative work.

### Reuse ruling (binding on this project)

**We copy no code from any of the seven — including the MIT-licensed one.**

The MIT repository is legally reusable, but copying a direct competitor's submission code into our
own submission is an integrity problem independent of the license, and it is exactly the kind of
thing the assignment's "Cite anything you borrowed" rule is designed to surface. The cost of
writing our own `check_leakage` is ten minutes; the cost of being asked "why is your code
byte-identical to another candidate's?" in a live interview is the offer.

What we *do* take is at the **IDEA** and **ARCHITECTURAL PATTERN** level, cited explicitly in
`docs/DECISION_LOG.md`. See §6 for the itemised list. Per the five-way classification:

| Level | Policy |
|---|---|
| 1. IDEA | ✅ Adopt freely, cite the source |
| 2. ARCHITECTURAL PATTERN | ✅ Adopt freely, cite the source |
| 3. ALGORITHM | ✅ Adopt if standard/published, re-implement from the definition |
| 4. CODE IMPLEMENTATION | ❌ Never copy |
| 5. DATA / LABELS / EVAL ARTIFACT | ❌ Never copy — ours must be independently produced |

---

## 2. Golden-set forensics — the decisive dimension

I parsed every committed golden set rather than trusting its README.

| Repo | Rows | Labelled? | How labels were actually produced | Evidence |
|---|---|---|---|---|
| **chetan** | 200 | Yes | **Keyword rulebook** in a file *named* `human_review_pass.py` | `scripts/human_review_pass.py` |
| **shubham** | 200 | Yes | **LLM pre-fill**, `human_final_*` identical to model 95.5% of the time | `eval/golden_set/build_golden_set.py:206` |
| **himanshu** | 206 | Yes | **Keyword matching**; reference replies are a fixed template string | `intent_notes: "Auto-classified by keyword matching"` |
| **mitdesai** | 200 | Yes | Deterministic rules — **and says so plainly** | `PROJECT_STATUS.md:22` |
| **anuj** | 200 | **No** | Sampled but never labelled; all 4 label columns 0/200 filled | re-parsed CSV |
| **ashraf** | 200 | **No** | Candidate pool built; labelling tool written but never run | `golden_v1.jsonl` absent |
| **vishwateja** | 3 | **No** | `label_status: "unlabelled"` in its own manifest | `data/demo/manifest.json` |

### 2.1 chetan — machine rules presented as human review

`scripts/human_review_pass.py` is docstringed *"Perform human-level review and verification across
all 200 golden-set examples"* and then implements a keyword cascade:

```python
elif any(k in m_low for k in ["refund", "money back", "reimburse", ...]):
    intent = "refund_request"
    action = "ESCALATE"
    notes  = "Direct refund demand"
```

Proof this produced the shipped labels: the committed `golden_set.csv` contains exactly 13 non-empty
`annotation_notes`, and every one is a verbatim string literal from that script (e.g. `"Prime
membership cancellation request"`). A human annotator does not write the same note 13 times
character-for-character.

The escalation labels are then **circular with the system under test**: the label generator says
`refund → ESCALATE, billing → ESCALATE, account_issue → ESCALATE`, and
`src/escalation/policy.py:18` documents `high_risk_intent : intent is in the risk list (billing,
refund, etc.)`. The policy is graded against a rulebook that encodes the same policy.

### 2.2 shubham — numerically provable circular evaluation

This is the clearest methodological failure in the field, and it is provable arithmetic.

- `eval/golden_set/build_golden_set.py:206-208` instantiates `LLMClassifier()` and
  `EscalationPolicy(classifier=classifier)` and writes their outputs to `model_suggested_intent`
  and `model_suggested_escalate`.
- `eval/harness.py:59,95` instantiates **the same `LLMClassifier`** and scores it with
  `accuracy_score(gold_intents, llm_preds)`.
- Re-parsing the committed CSV: `gold_intent == model_suggested_intent` for **95.5%** of rows.
- The reported headline in `scorecard.json`: `accuracy = 0.955`.

Those two numbers are the same number. The classifier scores 95.5% because the answer key is its
own output with a 4.5% edit rate. It is grading its own homework, and the headline metric measures
annotation-edit-rate, not accuracy.

Two supporting signals: `human_annotator_comments` is filled **0/200** while the four adjacent
`human_final_*` columns are filled 200/200 — a pattern consistent with programmatic fill, not
human work. And `docs/REPORT.md:129` simultaneously claims *"overall judge-human agreement was
κ_w = 1.0"* and that the kappa denominator *"collapse[d] to zero (κ_w = NaN)"*. κ_w = 1.0 and
κ_w = NaN cannot both hold; a degenerate statistic is being reported as a perfect one.

*(Minor, but telling about review discipline: `eval/golden_set/README.md` ships absolute local paths
— `file:///Users/shubhamsharma/Desktop/hiver%20assignment/...`.)*

### 2.3 himanshu — circularity visible in its own results file

`src/evaluation/build_golden_set.py:178` stamps every row `"Auto-classified by keyword matching"`,
and `generate_reference_response()` emits a fixed template — the shipped set is full of
`"Thank you for reaching out. Please let us know how we can help you today."` as the *reference
reply*. Reply quality cannot be meaningfully measured against a constant.

The circularity then shows up in their own `artifacts/evaluation_results.json`:

```
majority   accuracy 0.558   macro_f1 0.051
keyword    accuracy 0.937   macro_f1 0.779   ← beats the ML model
tfidf_lr   accuracy 0.864   macro_f1 0.841
```

A keyword baseline beating a trained TF-IDF+LR model by 7 accuracy points is not a finding about
keywords. The keyword baseline *is the labelling function*, so it scores ~94% by construction.

**Plus a genuine metric bug.** Their retrieval block reports:

```
recall_at_1 = recall_at_3 = recall_at_5 = mrr = 0.8155339805825242
```

Recall@k is monotonically non-decreasing in k, so `r@1 == r@5` only if every hit sits at rank 1 —
and MRR equals r@1 only under the same condition. Root cause, at
`src/retrieval/reranker.py:82-95`: `rerank()` adds an `intent_weight * intent_score` bonus for
cases whose intent matches the query intent, then `src/evaluation/run.py:107` defines a retrieval
"hit" as *the query intent appearing among the reranked intents*. The metric rewards precisely what
the reranker sorts by, so matches are forced to rank 1. It measures "did the reranker obey its own
sort key" — a tautology, not retrieval quality. `groundedness: 1.0` and `safety: 1.0` are further
degenerate-metric signals; real systems do not score exactly 1.0.

### 2.4 mitdesai — the honest one

`PROJECT_STATUS.md:22`, `README.md:142` and `reports/final_report.md:42` all state the same thing
without being asked:

> "The labels were curated using deterministic sampling/rules and cleanup ... they were not
> independently hand-labelled from scratch by a second human annotator. **This limitation is
> disclosed rather than falsely claiming human annotation.**"

The deliverable is still unmet, but the disclosure is exactly the posture the assignment's
"What is misleading about my headline number?" section is testing for. It also ships the field's
only committed judge-vs-human agreement output — and the numbers are *bad*:

```
overall_exact_match_rate  = 0.178
overall_mean_absolute_error = 1.85   (on a 1–5 scale)
safety exact match        = 0.067
```

Publishing an MAE of 1.85/5 — a judge that is nearly two points off the human on average — instead
of burying it is the most credible single act in the seven repositories. The correct reading is
that **their reply-quality headline is not supported by their own judge**, and their report should
say so more loudly than it does.

### 2.5 ashraf — best methodology, never executed

No labels shipped, but the *design* is the strongest in the field by a distance:

- **Three different model families, deliberately separated** (`src/config.py:12-14`):
  generator `gpt-oss-120b` (Groq), judge `gemini-2.5-pro` (Google), pre-annotator `qwen3.8-27b`
  (Groq). This is a real defence against LLM self-preference bias, which every other repo ignores.
- **Anchoring-bias measurement built into the sampling design** (`src/preannotate.py`): 150 of 200
  candidates get a pre-annotator suggestion; **50 are held blind**. Human-vs-suggestion agreement
  on the suggested items can then be compared against the blind items to *quantify how much the
  pre-annotation biased the annotator*. I have not seen this done in a take-home before.
- **A decision-theoretic cost model instead of an arbitrary threshold** (`src/config.py:18-25`):
  `COST_BAD_AUTO_REPLY = 8.0` vs `COST_HUMAN_TOUCH = 1.0`, with `COST_RATIO_SWEEP` and
  `TARGET_SAFE_AUTO_REPLY_RATE = 0.95`. Escalation becomes an expected-cost minimisation with a
  sensitivity analysis, not a magic number.
- **Refuses to auto-write gold** (`src/preannotate.py` docstring): *"It never writes to
  golden_v1.jsonl or labeling_log.jsonl — those only get written by a human running
  src/label_tui.py."*

The gap is execution: `golden_v1.jsonl` does not exist, so there are no results, no baselines
scored, no judge agreement. Best blueprint, unbuilt.

### 2.6 anuj — honest, and refuses to fabricate

Golden set is 200 rows with **all four label columns 0/200 filled**. Rather than inventing numbers,
`report.md:27` states:

> "Accuracy, macro F1, both baselines, escalation accuracy, and LLM-judge agreement are **not
> reported here yet** — they require the golden set to actually be labelled first."

And `evaluation/evaluate.py:109-114` hard-refuses to run on unlabelled data and warns below 150
rows. Very little was built, but nothing was faked.

### 2.7 vishwateja — a machine-readable honesty manifest

Only 3 test examples and a thin system, but it ships `results/VERIFICATION.json`:

```json
{ "unit_tests_passed": 15,
  "demo_split_hash_reproduction": "exact_match",
  "synthetic_600_pair_integration": "passed; fixture not retained; not real evaluation evidence",
  "human_gold_available": false,
  "real_llm_judge_run": false }
```

A machine-readable claims manifest that pre-emptively states what has *not* been demonstrated. This
is a genuinely good idea and we are adopting the pattern (see §6). It also has the field's only
hard-failing leakage guard (`support_agent/data.py:128-135`, raises `ValueError`) and the only CI
workflow.

---

## 3. Engineering quality

| Repo | Test files | Test fns | CI | Leakage guard | Notes |
|---|---|---|---|---|---|
| chetan | 7 | **38** | ❌ | Retrieval-corpus exclusion by golden ID | Best-tested; leakage-aware retrieval |
| ashraf | 8 | 27 | ❌ | Split-level, incl. a cost-model test | Tests the *decision theory*, not just plumbing |
| vishwateja | 1 | 15 | ✅ | **Hard-fail `ValueError`** on id/group/customer/text overlap | Only CI in the field |
| anuj | 0 | **0** | ❌ | none | |
| shubham | 0 | **0** | ❌ | none (14 grep hits are "chemical leak" in the safety policy) | |
| mitdesai | 0 | **0** | ❌ | **none** | Zero occurrences of "leak" anywhere |
| himanshu | 0 | **0** | ❌ | none | |

**Four of seven repositories contain zero tests.** The assignment says candidates will be asked to
"explain and modify your own code live" — an untested 35MB pickle-shipping repo is a bad place to
be in that conversation.

Note also what is being *committed*: `mitdesai` ships a 35 MB `nn.pkl` and a 26 MB `pairs.pkl`;
`himanshu` commits **~500 MB** of raw CSVs including the same 95 MB file twice
(`data/amazonhelp_conversations.csv` and `data/raw/amazonhelp_conversations.csv`). Both violate the
"we will not run your code on the full dataset — a subsample is expected" instruction, and pickles
are an arbitrary-code-execution vector for a reviewer.

---

## 4. Scorecard

Scored 1–5 on evidence, not claims. **Bold = field best.**

| Dimension | chetan | shubham | ashraf | mitdesai | himanshu | anuj | vishwateja |
|---|---|---|---|---|---|---|---|
| Assignment coverage | **4** | **4** | 2 | 3 | 3 | 2 | 2 |
| Architecture | **4** | 3 | **4** | 2 | 3 | 3 | 3 |
| Data pipeline | **4** | **4** | 3 | 2 | 3 | 2 | 3 |
| Classifier | 3 | 3 | 3 | 3 | 3 | 2 | 2 |
| Retrieval | 3 | 3 | 3 | 2 | 2 (broken metric) | 2 | 2 |
| Generation | 3 | **4** | 3 | 2 | 3 | 2 | 2 |
| Grounding | 3 | 3 | 3 | 2 | 2 | 2 | 2 |
| Escalation | 3 | 3 | **5** | 2 | 3 | 3 | 2 |
| **Evaluation rigor** | 2 | **1** | 4 | 3 | 1 | 3 | 3 |
| **Golden-set methodology** | 1 | 1 | **5** | 2 | 1 | 2 | 2 |
| **Leakage prevention** | 3 | 1 | 3 | **1** | 1 | 1 | **5** |
| LLM judge | 3 | 2 | **4** | 3 | 2 | 2 | 2 |
| Human calibration | 2 | 1 | 4 | **4** | 1 | 1 | 1 |
| Failure analysis | 3 | 3 | 1 | 3 | 2 | 2 | 1 |
| Testing | **5** | 1 | 4 | 1 | 1 | 1 | 3 |
| Reproducibility | 3 | 2 | **4** | 2 | 1 | 3 | **4** |
| Code quality | **4** | 3 | **4** | 2 | 3 | 3 | 2 |
| Documentation | 3 | **4** | **4** | 3 | 2 | 3 | 3 |
| License/provenance | 1 | 1 | 1 | 1 | 1 | 1 | **5** |
| **Scientific integrity** | **1** | **1** | 4 | **5** | 1 | 4 | **5** |
| Interview defensibility | 2 | 1 | 4 | 4 | 1 | 3 | 3 |

### Category winners

| Category | Winner | Why |
|---|---|---|
| **Best finished submission** | **chetan** | Only repo with a complete, coherent end-to-end story *and* a real test suite (38 tests). Fatally undermined by fake "human review" — but it is the most *complete* artifact. |
| **Best engineering foundation** | **chetan** / **ashraf** | chetan for test density and leakage-aware retrieval; ashraf for module boundaries and typed config. |
| **Best evaluation methodology** | **ashraf** | Model-family separation, blind-subset anchoring measurement, cost-ratio sweep. Unexecuted, but correct. |
| **Best data / leakage practices** | **vishwateja** | Only hard-failing leakage guard; hash-pinned split manifest; honest `label_status: unlabelled`. |
| **Best architectural ideas** | **ashraf** | Expected-cost escalation is the one idea here that would survive a staff-level design review. |
| **Most scientifically honest** | **mitdesai** / **vishwateja** | Both published numbers that make themselves look worse. mitdesai shipped a judge MAE of 1.85/5 rather than hiding it. |

---

## 5. What each repo teaches us

**Adopt (idea/pattern level, cited):**

1. **Expected-cost escalation with a cost-ratio sweep** — *ashraf*. Replaces an arbitrary confidence
   threshold with `argmin E[cost]`, plus sensitivity analysis over the ratio we cannot know exactly.
2. **Separate model families for generator / judge / pre-annotator** — *ashraf*. Defends against
   self-preference bias.
3. **Blind annotation subset to quantify anchoring bias** — *ashraf*. We will hold out ~40 of 200.
4. **Hard-failing leakage assertions** — *vishwateja*. `raise`, never `warn`. Matches our §6.
5. **Machine-readable verification manifest** — *vishwateja*. States what was *not* demonstrated.
6. **Hash-pinned split manifest for reproducibility** — *vishwateja*.
7. **Golden IDs excluded from the retrieval corpus** — *chetan*. Retrieval-side leakage control.
8. **Stratifying on difficulty and "hard case" type** — *chetan*, *shubham*.
9. **Refuse-to-fabricate harness behaviour** — *anuj*, *chetan*. Error out on unlabelled input
   rather than silently scoring nothing.
10. **Disclose label provenance in the report unprompted** — *mitdesai*.

**Avoid (anti-patterns, all observed):**

1. Naming a rule script `human_review_pass.py` — *chetan*.
2. Grading a classifier against labels that classifier generated — *shubham*.
3. Defining a retrieval metric that rewards the reranker's own sort key — *himanshu*.
4. Constant-string reference replies — *himanshu*.
5. Reporting κ = 1.0 from a degenerate zero-variance kappa — *shubham*.
6. Committing 500 MB of raw data and 35 MB pickles — *himanshu*, *mitdesai*.
7. Shipping a golden set with 0/200 labels filled — *anuj*.

---

## 6. Reuse strategy (binding)

| Component | Source of inspiration | Level taken | Implementation |
|---|---|---|---|
| Escalation policy | ashraf | IDEA + ALGORITHM | Ours, from decision theory first principles |
| Judge/generator separation | ashraf | IDEA | Ours |
| Blind annotation subset | ashraf | ARCHITECTURAL PATTERN | Ours |
| Leakage guards | vishwateja | ARCHITECTURAL PATTERN | Ours, written from scratch |
| Verification manifest | vishwateja | IDEA | Ours |
| Retrieval-corpus exclusion | chetan | IDEA | Ours |
| Difficulty stratification | chetan, shubham | IDEA | Ours |
| Intent taxonomy | **nobody** | — | **Derived from our own brand's data** |
| Golden labels | **nobody** | — | **Hand-labelled by us** |
| Metrics / results / report / decision log | **nobody** | — | **Ours** |

Every row in the "source of inspiration" column will be cited in `docs/DECISION_LOG.md` and in the
report's citations section, satisfying "Cite anything you borrowed."

---

## 7. Implication for our build

The field's collective weakness is concentrated in one place: **nobody paid the cost of real
annotation, and most hid that fact.** Consequently:

1. **We hand-label. Actually.** 200 examples, by a human, with a written codebook, a logged
   session, and a measured self-agreement rate on a re-labelled subset. This alone puts us ahead of
   all seven on deliverable #2.
2. **We make circularity structurally impossible**, and we write the test that proves it: the model
   that pre-annotates must never be the model under test, and the harness must fail loudly if the
   two are ever configured to be the same.
3. **We adopt ashraf's cost-based escalation** — the field's best idea, which its author never got
   to run.
4. **We adopt vishwateja's honesty instruments** — hard-fail leakage, hash-pinned manifest,
   machine-readable statement of what we did *not* demonstrate.
5. **Our "misleading headline" section writes itself**, because we will have measured the things
   that make it misleading: annotator self-agreement, judge-human agreement, anchoring bias from
   pre-annotation, and escalation-rate skew in the gold distribution.

The bar to beat is not chetan's 95.5%. It is **an honest number with a credible error bar**, which
no one in this field currently has.

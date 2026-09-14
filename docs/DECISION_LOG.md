# Decision Log

Non-obvious decisions, with the reasoning and evidence behind each. Trivial implementation
choices are excluded; they live in module docstrings and `MILESTONES.md`.

**Scope discipline.** Capped at 15 entries, matching the assignment's 10–15 requirement. A new
material decision is consolidated into the entry it belongs with.

**Consolidation (2026-09-14, release audit).** The first 15 entries were written during
Milestones 1–2 and left the decisions a reviewer most needs to audit — taxonomy, safety
boundary, gold-set design, judge, risk-coverage and the evaluation freeze — without an entry.
Entries still cited elsewhere in the repository keep their numbers and meaning (D1, D2, D3, D8,
D12, D13, D14). Former D4 and D5 are now in D2; former D7 in D6; former D9, D10 and D11 in D8;
former D15 in D14. The freed numbers carry new decisions. Nothing was invented after the fact:
each entry cites the commit, document or artifact where the decision was made.

| | Decision | Area |
|---|---|---|
| D1 | Reuse no code from any public Hiver repository | integrity |
| D2 | Union-find reconstruction; pairs anchored on the brand reply | data |
| D3 | Normalise ids through an explicit float check | data |
| D4 | Taxonomy derived from reply behaviour, security as an attribute, frozen before labelling | taxonomy |
| D5 | Retrieve over groundable, earlier, train-split questions only | retrieval |
| D6 | PII: over-masking is a failure; phones by validated digit count | privacy |
| D7 | The model may add an escalation, never clear one; safety checks independent of it | safety |
| D8 | Leakage guards raise, aggregate and cover questions+answers, customers and model families | leakage |
| D9 | The model cites opaque evidence labels, mapped exactly | grounding |
| D10 | Independent-family judge; on provider failure, switch whole-run, keep the partial | evaluation |
| D11 | Risk-coverage as a sensitivity analysis; no threshold chosen from gold | evaluation |
| D12 | Assisted 160 / blind 40 gold, with provenance kept separate from review action | gold set |
| D13 | Deterministic pipeline before API spend; the model stays untrusted | process |
| D14 | Brand chosen on a frozen multi-criteria profile, rejected profiles published | scope |
| D15 | Dependency failures fail closed; the evaluated system and the hardened system are kept distinct | release |

---

### D1 — Reuse no code from any public Hiver repository

**Context.** Seven public repositories implement this assignment, all created on 2026-09-09
within five hours: concurrent submissions, not prior art.

**Chosen.** Take ideas only; implement everything.

**Why.** Six carry no license (all rights reserved). The MIT one is legally reusable, but
copying a competing submission is an integrity problem regardless. Their dominant failure,
circular evaluation, is structural and would be inherited by adapting their harness.

**Tradeoff.** Substantially more work.

**Evidence.** `docs/PUBLIC_REPO_COMPARISON.md` §1. **Date.** 2026-09-09

---

### D2 — Union-find reconstruction; pairs anchored on the brand reply

**Context.** TWCS encodes threads in two partially redundant link columns, either of which can
be missing, and some rows form cycles. Brands often send several tweets answering one question.

**Chosen.** Union-find over both link directions. Conversation id is a hash of sorted member ids.
A pair is each brand reply with the customer message immediately before it.

**Why.** A parent-walk hangs or needs bolted-on cycle detection; union-find is cycle-safe by
construction. Cyclic threads have no root, so a root-based id is undefined exactly where it is
needed. Pairing each customer message with the "next reply" maps several replies onto one turn,
duplicating it in the corpus and double-counting it in evaluation.

**Tradeoff.** Opaque conversation ids. A question answered after intervening chatter is not
paired.

**Evidence.** `test_cyclic_reply_links_terminate`, `test_conversation_id_is_stable_regardless_of_row_order`,
`test_consecutive_brand_replies_pair_only_the_first`. **Date.** 2026-09-09

---

### D3 — Normalise all ids through an explicit float check

**Context.** pandas reads an integer column with blanks as floats, so id `1` becomes `1.0`.

**Chosen.** One float-aware coercion mapping `1`, `1.0` and `"1.0"` to `"1"`.

**Why.** `str(1.0) != "1"`, so the naive cast silently breaks every reply link and produces a
corpus of single-tweet conversations with no error. Confirmed in the real file.

**Evidence.** `test_ids_are_normalised_to_strings_regardless_of_input_type`; `tests/test_real_data.py`.
**Date.** 2026-09-09

---

### D4 — Taxonomy derived from reply behaviour, security as an attribute, frozen before labelling

**Context.** Intents must drive a routing decision, and AppleSupport's data decides which
distinctions support actually makes.

**Options.** (a) A generic support taxonomy. (b) Clusters named by inspection. (c) Candidate
labels kept only where support measurably handles them differently, frozen and hash-pinned before
any gold label exists.

**Chosen.** (c). v0.3.0: 10 intents plus two orthogonal attributes (security-sensitive,
context-sufficient). Hash `613f5dfe...`, frozen 2026-09-10 after three review rounds.

**Why.** Round 1 merged account access into security *because security was rare*; that was
rejected as optimising labels for measurement convenience. Round 2 tested handling differences
with bootstrap CIs and no verdict below n=30, which removed `software_update_issue`. Security
became an attribute on measurement: 306 of 340 security-sensitive messages sit outside the
account topic, so an intent would have captured a tenth of the safety signal. Freezing before
labelling means gold results cannot motivate a taxonomy change.

**Tradeoff.** 13 of 36 label pairs show no material handling difference, since the 2017 channel
was deflection-dominated. That is treated as absence of evidence, and disclosed.

**Evidence.** `docs/MILESTONES.md` Milestone 3; `docs/TAXONOMY_ADJUDICATION.md`; commit `cd75638`.
**Date.** 2026-09-10

---

### D5 — Retrieve over groundable, earlier, train-split questions only

**Context.** Replies must be grounded in how the brand resolved similar issues.

**Chosen.** Hybrid BM25 plus local sentence embeddings over customer *questions* (not answers).
Only actionable, non-deflecting resolutions are indexed. Only the train split is used, and only
resolutions strictly earlier than the query.

**Why.** BM25 misses paraphrases and embeddings miss exact strings ("iOS 11.0.3", "Error 4013"),
and support text has both. Matching a question against answers rewards shared vocabulary rather
than shared problems. Of 8,000 real pairs only 1,195 (14.9%) can ground an answer; indexing the
rest would let the agent retrieve "please DM us". The temporal and split filters stop a
resolution that postdates the question, or belongs to it, from answering it.

**Tradeoff, measured afterwards.** Scores are min-max normalised per query, so the top fused
score is at least 0.5 by construction. The 0.35 retrieval-confidence gate never fires (0 of 123
gold messages), so it provides no evidence-sufficiency check. Recorded, not retuned: an absolute
relevance score would be a dev-set experiment.

**Evidence.** `docs/AGENT.md` §2; `docs/RELEASE_AUDIT.md` §5.1. **Date.** 2026-09-10

---

### D6 — PII: over-masking is a failure; phones detected by validated digit count

**Context.** Masking runs before any text reaches a third-party API.

**Chosen.** Tight patterns pinned by `TestDoesNotOverMask`. A phone-shaped match is accepted only
with 7–15 digits.

**Why.** "iPhone 7", "$9.99", "iOS 11.0.1" and "2 weeks" are exactly what the classifier needs;
aggressive masking looks prudent while degrading the system invisibly. A digit-run regex masks
dates and versions, and an exception list grows forever. Digit count is the actual
distinguishing property.

**Tradeoff.** Exotic formats and long numbers with extensions may slip through.

**Evidence.** `tests/test_pii_handling.py`. **Date.** 2026-09-09

---

### D7 — The model may add an escalation, never clear one; safety checks independent of it

**Context.** A described Apple ID takeover, using none of the lexical trigger words, was
AUTO_HANDLED, and `gpt-oss-120b` reported `should_escalate: false` at 0.95 on the same message.
Two layers were wrong at once.

**Chosen.** Deterministic gates (security, context, policy intent) run before generation. A local
semantic security detector is unioned with the unchanged lexical rule, so it can only add
detections. Grounding and a policy validator (deflection, unsupported URLs, invented actions) run
after generation, independent of the model. The model's `should_escalate: true` escalates;
`false` is ignored. A malformed response escalates. There is no repair loop.

**Why.** A generator that can clear a gate sets its own safety policy. A local encoder is
independent of the generator family, cannot be disabled by a provider outage, and costs nothing.
On 24 security descriptions the lexical rule caught 2 and the composite 24, with 0 of 10 benign
false positives. A repair loop would make the generator its own judge.

**Tradeoff.** Similarity-based detection misses phrasings far from every anchor, which is why
the uncertain band escalates. The takeover probe's nearest anchor turned out to paraphrase it;
without that anchor it still escalates, at lower confidence (`RELEASE_AUDIT.md` §5.2). On gold,
security recall is 95.5% but precision 0.64, and the system still over-escalates.

**Evidence.** `docs/AGENT.md` §10; `test_routing_invariants.py` (exhaustive over intents and
confidence); commits `ae713c6`, `d256ef0`. **Date.** 2026-09-11

---

### D8 — Leakage guards raise, aggregate, and cover question+answer, customers and model families

**Context.** One public repository's headline accuracy (0.955) equalled its gold-vs-model
agreement (0.955), because the gold labels were prefilled by the classifier being scored.

**Chosen.**
- Response leakage means the (question, answer) pair reappearing, not the answer alone.
- Splits are separated by conversation *and* customer.
- Model roles (generator, judge, pre-annotator) must come from distinct families.
- Every guard runs, then raises once listing all violations.

**Why.** Brands send identical canned replies thousands of times, so an answer-only guard fires
constantly and gets disabled. A shared customer leaks phrasing and repeat issues even without a
shared conversation. Name inequality is too weak for independence, since `gpt-4o` and
`gpt-4o-mini` share lineage. A warning scrolls past; fail-fast reports one violation per full run.

**Tradeoff.** Family detection is heuristic; unknown models map to a family of their own name. A
reworded question with the same answer is left to the near-duplicate guard. Measured gap: that
guard compared against the last 20,000 train pairs, and the full split finds 2 of 200 short
fragments (effect on results: 0).

**Evidence.** `tests/test_leakage.py` (`TestModelIndependence`, `test_reports_all_failures_not_merely_the_first`);
`RELEASE_AUDIT.md` §2. **Date.** 2026-09-09

---

### D9 — The model cites opaque evidence labels, mapped exactly

**Context.** The prompt printed canonical ids like `387511__387510`. Groq echoed the wrapper and
OpenRouter truncated at `__`; both were refused as fabricated, so a third of one run measured the
id format rather than the model.

**Chosen.** The model sees only `E1..En`. The mapping lives outside the model; lookup is exact
after normalising brackets and case; an unknown label escalates. The prompt version was bumped so
the old cache entries cannot be served.

**Why.** Prefix or fuzzy matching could bind a citation to the wrong case: a reply attributed to
evidence it was not built from, undetectable afterwards. Removing the failure mode beats
tolerating it.

**Evidence.** `agent/generation.py`; `tests/test_llm_generation.py`; mutation M9 caught; commit `fcc4e1a`.
**Date.** 2026-09-11

---

### D10 — Independent-family judge; on provider failure, switch for the whole run and keep the partial

**Context.** Reply quality needs a judge from a different family than the generator
(`gpt-oss-120b`) and the pre-annotator (Llama 3.3 70B). The planned `anthropic/claude-sonnet-5`
via OpenRouter returned HTTP 402 (no credit) after 17 of 336 judgements.

**Options.** (a) Report the 17. (b) Mix Claude and a second judge. (c) Re-judge all 336 with one
independent judge and keep the partial file separate.

**Chosen.** (c). `qwen/qwen3.8-27b` on Groq judged all 336 with 0 failures ($0.237). The
independence guard runs against the judge actually used. The Claude partial is kept as
`judge_claude_partial_402.jsonl` and is never merged.

**Why.** Mixing judges confounds judge with example. 17 is too few for any interval. A different
family still avoids self-preference.

**Tradeoff.** A smaller judge than planned, and **judge–human agreement is unmeasured**: no human
reply ratings exist. It is reported as unmeasured, not estimated.

**Evidence.** `reports/golden_eval/summary.md`; `scripts/evaluate_golden.py` (`judge`); commit
`6772707`. **Date.** 2026-09-14

---

### D11 — Risk-coverage as a sensitivity analysis; no threshold chosen from gold

**Context.** The native operating point has a false auto-handle rate of 38.7%. A threshold looks
like the obvious fix.

**Chosen.** A fixed grid: at threshold t a message is auto-handled only if the system auto-handled
it *and* its score is at least t, so a threshold can withhold automation, never add it. Expected
cost is swept over error-to-human cost ratios 2, 4, 8, 12 and 20, with always-escalate as the
reference. No row is recommended.

**Why.** Choosing the best row on 200 gold examples is tuning on the evaluation set. The cost of
an unsafe reply cannot be inferred from tweets, so it is swept, not assumed. The measurement
showed the LLM's self-confidence is not a safety signal. Rows cheaper than always-escalate: 143
of 147 at ratio 2, 36 at 4, 6 at 8, and 0 at 12 or 20.

**Tradeoff.** No single recommended threshold. That is the honest output.

**Evidence.** `reports/golden_eval/risk_coverage.md`; `metrics.risk_coverage_curve`; commit
`b1429da`. **Date.** 2026-09-14

---

### D12 — Assisted 160 / blind 40 gold, with provenance kept separate from review action

**Context.** 200 labels by one annotator with limited hours. SPEC §9.2 specified 160 with a
pre-annotator suggestion and 40 blind. An earlier design made all 200 blind and was reverted to
the SPEC.

**Chosen.**
- Weak rules choose *which* examples are sampled, never *what* they are labelled. A 50-example
  unstratified reservoir is included, with inclusion probabilities recorded.
- Suggestions from `meta-llama/llama-3.3-70b-instruct` (a different family from the system under
  test) are a separate type in a separate file with `MODEL_GENERATED` provenance, which cannot be
  gold.
- A label becomes gold only through a recorded human action. Review action (entered, accepted,
  corrected) is stored separately from provenance.
- One-key accept is disabled for safety-relevant suggestions.
- A mistaken pass was retracted, not deleted.

**Why.** Blind labels on 40 examples are what make anchoring measurable. Conflating review action
with provenance would have destroyed that measurement. The same principle drove the self-agreement
framing: agreement measures reliability, never an accuracy ceiling.

**Tradeoff, disclosed.** 159 of 160 assisted labels equal the suggestion, accepted at a median
0.4 s. All-200 metrics therefore measure agreement with human-accepted pre-annotation, and the
blind 40 are reported separately. The planned self-agreement re-label pass was not performed, so
intra-annotator reliability is unmeasured.

**Evidence.** `data/golden/GOLDEN_LOCK.json` (counts, warning); `docs/GOLDEN_SET.md`; commits
`51fd4cb`, `84cf6a3`, `3822b44`. **Date.** 2026-09-09 to 2026-09-14

---

### D13 — Build the deterministic pipeline before spending API budget; the model stays untrusted

**Context.** No keys existed at the start.

**Chosen.** Data, taxonomy, retrieval, routing, baselines, harness and guards were built
deterministically first. Model calls were introduced only where needed, through an
environment-configured provider with a full-request cache key and hash-only call logging.

**Why.** Most of the system needs no LLM, and paying an API to find bugs a unit test catches is
waste. The cache makes re-evaluation free. The deterministic template generator remained as an
ablation (`agent_template`).

**Tradeoff.** Model-quality questions were answered later. Upstream host variance was found only
against real APIs, and generation is now pinned to one host (`LLM_EXTRA_BODY`).

**Evidence.** `docs/LLM_PROVIDER.md`; commits `6a1fcfc`, `18855f9`, `fcc4e1a`. Total recorded
evaluation spend: generator $0.0108, judge $0.237. **Date.** 2026-09-09

---

### D14 — Brand chosen on a frozen multi-criteria profile; rejected profiles published

**Context.** The brand determines the taxonomy, the corpus and every downstream number.

**Options.** Convention (AmazonHelp), a single criterion, the best agent result across brands, or
a frozen profile of corpus statistics fixed before any agent runs.

**Chosen.** The frozen profile: 13 features and a five-part rubric. A later defect fix changed the
inputs and produced an exact tie, re-decided by the pre-declared lexicographic tie-break
(absolute groundable evidence, 31,241 vs 20,076) → AppleSupport. Profiles of all 83 brands are
published, including rejected ones. No comparative claim about other candidates' choices is made
without measurement.

**Why.** Picking the brand with the best downstream metric guarantees an inflated result that
does not replicate. Publishing rejected profiles makes the trade visible.

**Tradeoff.** Our numbers may be lower than a cherry-picked brand's.

**Evidence.** `reports/brand_selection.md`; `SPEC.md` §3.2; commits `5f73d9a`, `f5f66dc`.
**Date.** 2026-09-09

---

### D15 — Dependency failures fail closed; the evaluated and hardened systems are kept distinct

**Context.** Fault injection during the release audit found that a classifier, retriever or
unexpected generator exception crashed `handle()`: no reply, but no decision either. The gold set
was frozen only after the evaluation had run.

**Chosen.** Classifier and retriever exceptions escalate as `dependency_failed`; any other
generator exception escalates as `generator_failed`. `TypeError` from invalid input and
`KeyboardInterrupt` still propagate. The committed evaluation artifacts are not re-generated
under the hardened code. Instead:
- the original system is identified by commit `6772707`;
- the hardened code is replayed from cache with the network blocked, and compared field by field;
- the lock is committed as its own step, with the evaluated labels shown to equal the locked labels.

**Why.** Fail-closed has to include failures of our own dependencies. Overwriting the evaluated
artifacts with a later system's output would blur which system produced the headline numbers.

**Tradeoff.** Two named system states to explain instead of one.

**Evidence.** `tests/test_dependency_failures.py`; mutations M14–M15; failure paths 12/15 → 15/15;
`reports/golden_eval/evaluation_boundary.md`; commits `409622d`, `81bafe4`. **Date.** 2026-09-14

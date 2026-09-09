# Decision Log

Non-obvious decisions, with the reasoning and evidence behind each. Written as they were made,
not reconstructed afterwards. Trivial implementation choices are deliberately excluded.

---

### D1 — Reuse no code from any public Hiver repository

**Context.** Seven public repositories implement this assignment. All were created on 2026-09-09,
within a five-hour window — they are concurrent submissions by competing candidates, not prior art.

**Options.** (a) Fork the strongest and adapt. (b) Copy the MIT-licensed one. (c) Copy specific
functions with attribution. (d) Take ideas only, implement everything ourselves.

**Chosen.** (d).

**Why.** Six of seven carry no license, which under default copyright means all rights reserved —
public visibility permits viewing and forking within GitHub, not copying into a derivative work.
The seventh is MIT and therefore legally reusable, but copying a direct competitor's submission is
an integrity problem independent of licensing, and "why is your code identical to another
candidate's?" is not a question worth risking in a live interview. Separately, their dominant
failure mode (circular evaluation) is *structural* and would be inherited by anyone adapting their
harness.

**Tradeoff.** Substantially more implementation work. Accepted.

**Evidence.** `gh repo view` license metadata for all seven; `docs/PUBLIC_REPO_COMPARISON.md` §1.

**Date.** 2026-09-09

---

### D2 — Union-find for thread reconstruction, not a recursive parent-walk

**Context.** TWCS encodes reply structure in two partially redundant columns, either of which can
be missing. Some malformed rows point at each other, forming cycles.

**Options.** (a) Recursive walk up `in_response_to`. (b) Union-find over both link directions.

**Chosen.** (b).

**Why.** A parent-walk either hangs or needs bolted-on cycle detection; union-find is cycle-safe by
construction. Using both link directions also recovers threads where only one column is populated.

**Tradeoff.** Conversation has no canonical "root" tweet, so ids derive from membership (see D4).

**Evidence.** `tests/test_thread_reconstruction.py::test_cyclic_reply_links_terminate`.

**Date.** 2026-09-09

---

### D3 — Normalise all ids through an explicit float check

**Context.** pandas reads an integer column containing blanks as floats, so tweet id `1` is read as
`1.0`.

**Options.** (a) Cast with `str()`. (b) Read all id columns as strings via `dtype`. (c) Normalise
through a float-aware coercion that maps `1.0`, `"1.0"` and `1` to `"1"`.

**Chosen.** (c).

**Why.** `str(1.0) == "1.0" != "1"`, so option (a) silently breaks *every* reply link in the file
while appearing to work. Option (b) helps on load but not for values arriving from elsewhere.
This is the highest-impact defect in the data layer and it produces no error — only a corpus of
single-tweet conversations.

**Tradeoff.** A small amount of defensive parsing on a hot path.

**Evidence.** `test_ids_are_normalised_to_strings_regardless_of_input_type`.

**Date.** 2026-09-09

---

### D4 — Conversation id derives from sorted membership

**Context.** Conversation ids must be stable across runs and row orderings.

**Options.** (a) Root tweet id. (b) Row index. (c) Hash of sorted member tweet ids.

**Chosen.** (c) — `conv_` + first 16 hex of SHA-1.

**Why.** Cyclic threads have no root, so (a) is undefined for exactly the malformed cases we must
handle. (b) is not stable under reordering. (c) is well-defined and order-independent.

**Tradeoff.** Ids are opaque rather than traceable to a tweet.

**Evidence.** `test_conversation_id_is_stable_regardless_of_row_order`.

**Date.** 2026-09-09

---

### D5 — Anchor customer/support pairs on the brand reply

**Context.** Brands routinely send several consecutive tweets answering one question.

**Options.** (a) For each customer message, find the next brand reply. (b) For each brand reply,
take the immediately preceding customer message.

**Chosen.** (b).

**Why.** (a) maps several replies onto the same customer turn, duplicating it in the corpus and
double-counting it in evaluation. (b) yields exactly one pair per answered question and naturally
ignores follow-on brand tweets.

**Tradeoff.** A question answered only after intervening chatter is not paired. Acceptable — those
are lower-quality training signal anyway.

**Evidence.** `test_consecutive_brand_replies_pair_only_the_first`.

**Date.** 2026-09-09

---

### D6 — Treat over-masking of PII as a failure, not as caution

**Context.** PII masking runs before any text reaches a third-party API.

**Options.** (a) Aggressive masking of anything digit-shaped. (b) Tight patterns validated against
false positives.

**Chosen.** (b), with `TestDoesNotOverMask` pinning the behaviour.

**Why.** "iPhone 7", "$9.99", "iOS 11.0.1", "2 weeks" and "5 stars" are precisely the features an
intent classifier depends on. Aggressive masking looks prudent while quietly degrading the system,
and the damage is invisible in a masking test suite that only checks that PII disappears.

**Tradeoff.** Tighter patterns risk missing exotic PII formats. Mitigated by testing each class
explicitly and by masking being one layer, not the only one.

**Evidence.** `tests/test_pii_handling.py::TestDoesNotOverMask`; manual inspection on 5 realistic
messages.

**Date.** 2026-09-09

---

### D7 — Detect phone numbers by validated digit count, not by digit-run matching

**Context.** A regex matching digit runs classifies "Oct 31 2017" and "iOS 11.0.1" as phone numbers.

**Options.** (a) Hand-maintained exception list. (b) Match a phone-shaped candidate, then accept
only if its digit count falls in 7–15.

**Chosen.** (b).

**Why.** An exception list grows forever and encodes no principle. Digit count is the actual
distinguishing property of a phone number and generalises to formats not yet seen.

**Tradeoff.** Very long international numbers with extensions may slip through.

**Evidence.** `TestPhoneMasking`, `TestDoesNotOverMask`.

**Date.** 2026-09-09

---

### D8 — Define response leakage as the question *and* answer, never the answer alone

**Context.** A guard was needed to stop a golden example's own resolution appearing in the
retrieval corpus.

**Options.** (a) Flag when the golden reply text appears in the corpus. (b) Flag when the
(question, answer) combination appears.

**Chosen.** (b).

**Why.** Brands send identical canned replies ("please DM us") thousands of times, so (a) fires
constantly on legitimate data. A guard that always fires gets disabled, and a disabled guard is
worse than none because it still reads as protection. (b) catches genuine reposts and duplicated
rows while staying silent on canned replies.

**Tradeoff.** A corpus entry with the same answer to a *differently worded* version of the golden
question is not caught here — that is the near-duplicate guard's job.

**Evidence.** `test_silent_when_only_the_canned_reply_repeats`.

**Date.** 2026-09-09

---

### D9 — Treat shared customer ids across splits as leakage

**Context.** Splitting is usually done at the example or conversation level.

**Options.** (a) Split by example. (b) By conversation. (c) By conversation *and* customer.

**Chosen.** (c).

**Why.** The same customer phrases complaints the same way and frequently raises the same issue
more than once, so a shared customer leaks both style and content across the split even when no
conversation is shared.

**Tradeoff.** Reduces usable data and complicates stratification.

**Evidence.** `test_raises_on_shared_customer_id`.

**Date.** 2026-09-09

---

### D10 — Enforce model-role independence as a raising guard

**Context.** The audit proved that one public repository's headline accuracy (0.955) was exactly
its gold-vs-model agreement rate (0.955), because gold labels were prefilled by the same classifier
the harness then scored.

**Options.** (a) Document the convention. (b) Assert distinct model *names*. (c) Assert distinct
model *families*, raising on violation.

**Chosen.** (c).

**Why.** A convention in a README does not survive a late-night configuration change. Name equality
is too weak — `gpt-4o` and `gpt-4o-mini` share training lineage and exhibit the same
self-preference. Family-level independence is the property actually required.

**Tradeoff.** Family detection is heuristic and needs updating as new models appear. An unknown
model maps to a family unique to its own name, so unknowns never collide by accident.

**Evidence.** `TestModelIndependence`; `docs/PUBLIC_REPO_COMPARISON.md` §2.2.

**Date.** 2026-09-09

---

### D11 — Leakage checks raise and aggregate, rather than warning or failing fast

**Context.** Leakage guards need a failure mode.

**Options.** (a) Log a warning. (b) Raise on the first violation. (c) Run all checks, raise once
with every violation listed.

**Chosen.** (c).

**Why.** (a) scrolls past unnoticed — the assignment brief is explicit that detection must fail
loudly. (b) is correct but wasteful: leakage arrives in clusters from one bad split, and fixing one
error message per full pipeline run is slow. (c) fails loudly and reports the whole problem at once.

**Tradeoff.** Slightly more code than a bare assertion.

**Evidence.** `test_reports_all_failures_not_merely_the_first`; manual inspection showed all five
guards firing together with per-guard detail.

**Date.** 2026-09-09

---

### D12 — Self-agreement is reported as annotation consistency, never as an accuracy ceiling

**Context.** The plan includes re-labelling ≥40 golden examples after ≥24h to measure
intra-annotator agreement. An earlier draft of `SPEC.md` described this as a "ceiling on achievable
accuracy" and inferred that a system scoring above it was overfitting the annotator.

**Options.** (a) Keep the ceiling framing — it is rhetorically strong. (b) Report it strictly as a
reliability measure.

**Chosen.** (b).

**Why.** Self-agreement measures *reliability*, not *validity*. A consistent annotator can be
consistently wrong, so the statistic bounds neither true label accuracy nor achievable model
performance, and the overfitting inference does not follow. The claim would not survive a
methodologically careful reviewer, and the cost of being caught overclaiming in an
evaluation-focused assignment is far higher than the rhetorical benefit.

**Tradeoff.** A weaker-sounding headline. Accepted deliberately: better to underclaim than to put a
technically questionable claim in the report.

**Evidence.** `SPEC.md` §9.2; corrected in commit `51fd4cb`.

**Date.** 2026-09-09

---

### D13 — Build the entire deterministic pipeline before spending any API budget

**Context.** No API keys are currently provisioned; the project owner can obtain them.

**Options.** (a) Provision keys immediately and build against live models. (b) Build data,
taxonomy, retrieval, routing, baselines, harness and leakage guards deterministically first, and
introduce model calls only where a model is genuinely required.

**Chosen.** (b).

**Why.** Most of this system does not need an LLM: reconstruction, PII, splitting, leakage,
TF-IDF/BM25 baselines and the metric harness are all deterministic and unit-testable.
`sentence-transformers` runs locally, so retrieval embeddings cost nothing. Paying an API to
discover bugs that a unit test catches for free is waste, and cached calls make repeated evaluation
free thereafter.

**Tradeoff.** Model-dependent quality questions stay unanswered for longer.

**Evidence.** 84 tests passing with zero API spend at the time of writing.

**Date.** 2026-09-09

---

### D14 — Brand selected on a frozen multi-criteria profile, never on achievable metric

**Context.** One brand must be chosen from dozens. The choice determines the taxonomy, the
retrieval corpus and every downstream number.

**Options.** (a) Pick a high-volume brand (AmazonHelp) by convention. (b) Optimise a single
criterion such as lowest DM-deflection rate. (c) Run the agent on several brands and keep the best
result. (d) Score every candidate on a frozen multi-criteria profile computed from corpus
statistics alone, and lock the choice before any agent is evaluated.

**Chosen.** (d), with 13 measured features and a five-part rubric (`SPEC.md` §3.2).

**Why.** (c) is the serious hazard and the tempting one: with a dozen candidates, selecting the
brand with the best downstream metric guarantees an inflated, non-replicating result — a
garden-of-forking-paths error, and a subtler cousin of the circularity found in the public field.
(b) is nearly as bad in a different direction: optimising DM-deflection alone selects for the
easiest benchmark rather than the most informative one, and a brand with low deflection but only
two real intents cannot exercise a classifier at all. (a) substitutes convention for evidence.

Under (d) the profile is descriptive statistics only, so no system performance influences the
choice. Two commitments make this enforceable: criteria and weights frozen before evaluation, and
no re-selection afterwards — if the chosen brand proves hard, that is a reported finding, not a
reason to switch.

**Tradeoff.** We may select a brand on which our headline numbers are lower than they could have
been. Accepted deliberately: a lower number that means something beats a higher number that does
not, and the selection procedure is itself defensible in the interview.

**Evidence.** `SPEC.md` §3.2; profile table for all candidates to be published in
`reports/brand_selection.md`, including brands not chosen.

**Date.** 2026-09-09

---

### D15 — Publish brand profiles for rejected candidates, and make no claim about others' choices

**Context.** An earlier draft of `SPEC.md` asserted that DM-deflection rate is "under-examined in
the public field" and that several candidates had picked brands whose replies are overwhelmingly
deflections, capping achievable groundedness.

**Options.** (a) Keep the claim. (b) Measure it first, then state it if supported. (c) Drop any
comparative claim and publish only our own analysis.

**Chosen.** (b) as the standard for making the claim at all, with (c) as the default until
measurements exist.

**Why.** The claim was unsupported when written: no deflection rate had been computed for
AmazonHelp, AppleSupport or SpotifyCares, and none of those repositories published a brand profile
to compare against. Asserting it would have been the same error this project audits others for —
a confident statement outrunning its evidence. If our analysis later shows a commonly chosen brand
scores poorly on grounding evidence, it will be reported as an empirical finding from our dataset
analysis with numbers attached, not as a criticism of reasoning we cannot see.

Publishing the profile for every candidate, including rejected ones, is what makes the selection
auditable: a reviewer can see what was traded away rather than only the winner's numbers.

**Tradeoff.** A less striking narrative. Correct.

**Evidence.** `SPEC.md` §3.2.4; retraction of the claim previously at `SPEC.md:99`.

**Date.** 2026-09-09

# AppleSupport Intent Taxonomy — CANDIDATE v0.1.0

> **STATUS: CANDIDATE. NOT FROZEN.** `CANDIDATE_TAXONOMY.frozen is False`.
> Freezing requires review and approval, and is recorded as a decision — not a side effect of
> drafting. No golden-set labelling may begin until this is frozen (`SPEC.md` §5).

**Derived from:** the real AppleSupport **train split only** (67,626 customer→support pairs).
**Evidence:** `reports/taxonomy_discovery.json`, `reports/taxonomy_probes.json`.
**Implementation:** `src/hiver_support/taxonomy.py` · **Tests:** `tests/test_taxonomy.py` (33)
**Annotation guide:** `docs/ANNOTATION_GUIDE.md`

---

## 1. Method, and why discovery used the train split only

The golden set is drawn from the held-out **test pool** (`SPEC.md` §9.1). Clustering the whole
corpus would fit the label design to the very data the system is later evaluated on — a quiet
form of contamination that would weaken every downstream claim. So discovery ran on **train
only** (67,626 pairs); dev (9,804) and test pool (22,378) were never inspected.

| Parameter | Value |
|---|---|
| Brand | AppleSupport |
| Split used | train (67,626 pairs, 51,677 distinct customers) |
| Random seed | 20260910 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, run locally (no API, $0) |
| Embedding sample | 20,000 messages |
| Cluster sweep | k ∈ {8, 10, 12, 14, 16, 20, 24} |
| Discovery runtime | 873.5s |

**No cluster assignment is a label, and no probe is a classifier.** Both are exploratory
evidence for a human decision. One audited public repository generated its gold labels by
keyword matching and then reported that its keyword baseline beat a trained model — the
baseline *was* the labelling function (`PUBLIC_REPO_COMPARISON.md` §2.3). That failure mode is
the reason this separation is stated so bluntly.

---

## 2. What the data actually looks like

### 2.1 Clustering does not yield a routing taxonomy here

| k | silhouette (cosine) | largest cluster | clusters ≥3% |
|---|---|---|---|
| 8 | 0.0885 | 19.8% | 8 |
| **10** | **0.0892** | 12.3% | 10 |
| 12 | 0.0759 | 11.4% | 11 |
| 16 | 0.0667 | 9.7% | 14 |
| 24 | 0.0602 | 6.1% | 19 |

**Silhouette never exceeds 0.09.** These messages do not separate cleanly in embedding space at
any granularity. That is a property of short, noisy, emotionally-charged tweets, not a tuning
failure, and it is reported rather than tuned away. It has two consequences worth carrying
forward: annotation will be genuinely hard at the boundaries, and a classifier should not be
expected to reach the accuracy this task's surface simplicity might suggest.

**More important: the clusters are dominated by a single event.** At k=12, six of twelve
clusters are variants of "the iOS 11 update broke my phone" (paraphrase) — including two devoted to the
iOS 11 autocorrect bug that rendered the letter "I" as "I⍰":

> a customer asking why typing the letter I produces something strange *(tweet `1509390`; text removed)*
> a second customer reporting that the letter I changes as they type *(tweet `1973498`; text removed)*

Clustering surfaces **topic of the moment**, not **support intent**. Labels here are therefore
defined by *the support action a message calls for*, which is what `SPEC.md` §5 requires
("two intents that always route identically … should be merged").

### 2.2 What clustering missed entirely

No cluster emerged at any k for billing, account access, orders, repair, or security. Those
categories exist but are drowned out by the event. They were measured directly instead
(§2.3) — which is why probes exist at all.

### 2.3 Prevalence floors

Measured by deliberately **high-precision, low-recall** lexical probes over the train split.

> **These are FLOORS, not estimates.** 45.7% of the split matches no probe. True prevalence is
> unknown until the golden set is hand-labelled, and is deliberately not estimated here.

| Hypothesis probe | Count | Floor | In a 200-item set |
|---|---|---|---|
| software_update | 16,084 | 23.8% | ~48 |
| battery_charging | 6,217 | 9.2% | ~18 |
| apps_services | 4,194 | 6.2% | ~12 |
| howto_information | 2,573 | 3.8% | ~8 |
| connectivity | 2,366 | 3.5% | ~7 |
| complaint_feedback | 1,611 | 2.4% | ~5 |
| billing_payment | 1,267 | 1.9% | ~4 |
| repair_replacement | 811 | 1.2% | ~2 |
| account_access | 802 | 1.2% | ~2 |
| security_privacy | 277 | 0.4% | **~1** |
| order_delivery | 224 | 0.3% | **~1** |
| hardware_physical | 224 | 0.3% | **~1** |
| subscription | 201 | 0.3% | **~1** |

### 2.4 Structural findings that shaped the design

| Finding | Value | Consequence |
|---|---|---|
| Thread-opening messages | 72.5% | What the agent really receives first |
| Mid-thread continuations | 27.5% | Often answers to Apple's diagnostic questions |
| Mid-thread **and** <8 words | 7.5% | Not classifiable standalone → `needs_more_context` |
| Non-English | 10.8% | Out of scope; handled by a language gate (§5) |
| Messages matching ≥2 probes | 10.1% | Multi-intent is real; tie-breaks required |
| Only 29% contain a question mark | — | Most messages are complaints, not questions |
| Reply deflection rate | 37.1% | Over a third of Apple's replies are "DM us" |

### 2.5 Coverage check on the residual

45.7% of the split matches no probe. To test whether that hides a **missing category**, 22
random residual messages were read. Every one mapped to an existing candidate label; the
probes simply miss keyword-free phrasing:

| Residual message (abridged) | Maps to | Why the probe missed it |
|---|---|---|
| *"This phone used to last 12 hrs … now under two hours"* | `battery_charging` | never says "battery" |
| *"the silent mode switch always malfunctions"* | `device_malfunction` | hardware control, no keyword |
| *"[tweet-text redacted: tweet_id=816622 sha256=77e3a592956afb7c]"* | `software_update_issue` | "iOS" without a digit |
| *"[tweet-text redacted: tweet_id=1638793 sha256=c4f2a650d5fd42fd]"* | `howto_information` | pricing question |
| *"have i been compromised??"* | `account_security` | "compromised" not in probe |

**Conclusion: no missing category was found.** This is coverage evidence, not proof — the
sample is 22 messages, and true coverage is only established once the golden set is labelled.

---

## 3. The candidate taxonomy — 12 labels

Ordered as declared in code; declaration order is the deterministic fallback precedence.

| # | Label | Floor | Escalation-sensitive | Auto-handleable |
|---|---|---|---|---|
| 1 | `account_security` | 1.4% | **Yes** | No |
| 2 | `billing_and_subscription` | 1.8% | **Yes** | No |
| 3 | `repair_order_replacement` | 1.1% | **Yes** | No |
| 4 | `connectivity` | 2.9% | No | Yes |
| 5 | `battery_charging` | 7.3% | No | Yes |
| 6 | `apps_and_services` | 4.7% | No | Yes |
| 7 | `software_update_issue` | 15.8% | No | Yes |
| 8 | `device_malfunction` | 5.9% | No | Yes |
| 9 | `howto_information` | 2.0% | No | Yes |
| 10 | `complaint_feedback` | 0.7% | No | No |
| 11 | `needs_more_context` | 7.5% | No | No |
| 12 | `other_unclear` (catch-all) | — | No | No |

Full definitions, inclusion/exclusion criteria and three traceable real examples per label live
in `src/hiver_support/taxonomy.py` and are reproduced for annotators in `ANNOTATION_GUIDE.md`.
Every example carries its `pair_id`, so any of them can be traced back to the corpus.

### 3.1 Why each label exists

**`account_security`** — Identity is the one thing an agent must never guess at. Sign-in
failures, password resets, lockouts and suspected phishing all demand verification before any
action. Kept despite a 1.4% floor because routing is categorically different, not merely
different in degree.

**`billing_and_subscription`** — Money requires account-specific action the system cannot
perform (`SPEC.md` §1.3: no action execution). Merges one-off charges with subscription
lifecycle: both route identically (escalate), and neither is separately evaluable at 200
examples.

**`repair_order_replacement`** — Physical logistics: repair, warranty, AppleCare, orders,
delivery, physical damage. Merges three hypotheses that were each ~0.3% and each unevaluable
alone, all of which route to the same place.

**`connectivity`** — Wi-Fi, Bluetooth, cellular. Distinct diagnostic path (network settings
reset), and the iOS 11 Wi-Fi/Bluetooth toggle behaviour makes it materially common.

**`battery_charging`** — Highest-volume specific symptom after updates (7.3% floor). Its own
diagnostic path, and can escalate to hardware replacement.

**`apps_and_services`** — A named Apple service failing (Music, iCloud, iMessage, Siri, Safari)
is resolved differently from a device-wide fault: the evidence retrieved is service-specific.

**`software_update_issue`** — By far the largest label (15.8% floor, and dominant in the
residual). Retained separately from `device_malfunction` because explicit update attribution
changes the resolution: a known OS regression is answered with "known issue, update to 11.1",
while an unattributed fault needs a diagnostic flow.

**`device_malfunction`** — The unattributed fault. Distinct from the above precisely by the
absence of attribution.

**`howto_information`** — Nothing is broken; the customer wants instructions, a capability
answer, or pricing. The most straightforwardly auto-handleable label.

**`complaint_feedback`** — Expressive dissatisfaction with no actionable request. Routing
differs fundamentally: acknowledge, do not attempt resolution. Its 0.7% floor badly undercounts
— pure venting is common in the residual and rarely uses the probe's vocabulary.

**`needs_more_context`** — Not a topic but a *classifiability state*, and it is 7.5% of the
split. Mid-thread fragments answering Apple's questions (*"@AppleSupport Both 11.0.3"*,
*"@AppleSupport iPhone 7"*) have no standalone intent. Without this label an annotator would be
forced to guess, and guesses become noise in the gold labels.

**`other_unclear`** — Explicit catch-all, required by `SPEC.md` §5.

---

## 4. Merge and split decisions

| Decision | Hypotheses | Outcome | Reason |
|---|---|---|---|
| **MERGE** | `billing_payment` + `subscription` | `billing_and_subscription` | Identical routing (escalate, account-specific). Separately 1.9% and 0.3% → neither evaluable. |
| **MERGE** | `repair_replacement` + `order_delivery` + `hardware_physical` | `repair_order_replacement` | All physical/logistical, all escalate. Each ~0.3% alone → unevaluable. |
| **MERGE** | `account_access` + `security_privacy` | `account_security` | See §4.1 — this is the contested one. |
| **SPLIT** | device problems | `software_update_issue` vs `device_malfunction` | Attribution changes the retrieved evidence and the reply. |
| **SPLIT** | non-classifiable | `needs_more_context` vs `other_unclear` | "Ask a clarifying question" ≠ "cannot serve". |
| **KEEP separate** | `connectivity` from `device_malfunction` | — | Distinct diagnostic path despite being a malfunction. |
| **KEEP separate** | `apps_and_services` from `device_malfunction` | — | Service-specific vs device-wide evidence. |
| **REJECTED as a label** | non-English (10.8%) | Scope exclusion | Language is an attribute, not an intent. See §5. |
| **REJECTED as a label** | pre-sales pricing | Folded into `howto_information` | Too rare to evaluate; no separate routing. |

### 4.1 The contested merge: `account_access` + `security_privacy`

**Argument for keeping them separate.** They do *not* always route identically. "I forgot my
password" can plausibly be auto-handled with reset guidance; "someone has taken over my
account" must always escalate. `SPEC.md` §5 merges only labels that *always* route identically.

**Argument for merging.** `security_privacy` has a 0.4% floor — roughly **one example** in a
200-item golden set. A label that cannot be measured cannot support any claim about it, and
per-class metrics on n=1 are meaningless.

**Chosen: merge, and mark the combined label escalation-sensitive and never auto-handleable.**
The safety-conservative reading wins: treating every account-access request as escalation-worthy
costs some automation coverage but cannot cause an identity-related error. This is the strictest
of the available options and the easiest to defend.

**This is the decision most in need of review**, because it deliberately trades a routing
distinction for evaluability. See §7.

---

## 5. Scope exclusion: non-English messages (10.8%)

10.8% of the train split is not English — predominantly Spanish, with a visible cluster at
k=12 (2.2% of the embedded sample) of iOS 11 battery complaints in Spanish.

Language is an **attribute**, not an intent: "my battery drains" in Spanish is the same intent
as in English. Making it a label would conflate two orthogonal dimensions.

**Decision:** the system is scoped to English. A language gate runs *before* intent
classification and routes non-English messages to a human. The 10.8% figure is disclosed as a
coverage limitation in the final report, and the golden set is drawn from English messages
only — which must be stated whenever a headline number is quoted, since it means every metric
describes 89.2% of real traffic, not 100%.

---

## 6. Major ambiguity and confusion pairs

Sixteen tie-break rules are implemented and tested acyclic. The governing principle:

> **The label naming the ACTION SUPPORT MUST TAKE beats the label naming the customer's
> explanation of the cause.**

| Confusion | Resolution | Real example |
|---|---|---|
| battery vs billing — **"charged"** | *Lexical trap, not a tie-break.* "charged my phone" is battery; only money is billing | `1507298__1507297` "Literally have charged my phone 4 times today" |
| account vs connectivity — **"password"** | *Lexical trap.* A Wi-Fi network password is connectivity | `52265__52263` "will connect to unsecured wifi but not those requiring a password" |
| symptom vs update attribution | Named symptom wins | `2080925__2080924` "[tweet-text redacted: tweet_id=2080925 sha256=31bc6807db178c4a]" |
| account access vs update | Account wins — the blocked action is sign-in | `1714094__1714093` "ever since iOS 11 I haven't been able to log into my Apple ID" |
| app vs device-wide | Named app wins | `229207__229206` "the podcast app not working" |
| repair vs malfunction | Physical damage → logistics | `972534__972533` "[tweet-text redacted: tweet_id=972534 sha256=0d2a2f46570911b9]" |
| complaint vs real fault | Fault wins; anger is tone, not intent | `1718188__1718187` "[tweet-text redacted: tweet_id=1718188 sha256=befdf022de9ea0b4]" |
| software "replacement" | Not physical | `1405943__1405945` "keyboard replacement" = the iOS keyboard |

**Lexical traps are handled in `excludes`, not by tie-breaks.** Tie-breaks resolve *genuine
overlap* — a message really about both money and battery. A misleading word is a definitional
matter. Conflating the two was a real defect in the first draft of these rules, caught by
manual inspection of resolution output and corrected.

---

## 7. Open decisions requiring review before freeze

1. **`account_access` + `security_privacy` merged (§4.1).** Trades a real routing distinction
   for evaluability. Reversible only before freeze.
2. **12 labels exceeds `SPEC.md` §5's "6–10 plus other".** The brief for this milestone asked
   for 8–12. Twelve sits at the top of one range and outside the other. Merge candidates if
   fewer are wanted: `device_malfunction` into `software_update_issue` (→11), or
   `needs_more_context` into `other_unclear` (→11, but loses a routing distinction).
3. **Non-English excluded as scope (§5)**, not modelled as a label.
4. **`complaint_feedback` retained on a 0.7% floor**, on the argument that the floor badly
   undercounts. If the golden set shows otherwise it may prove unevaluable.

---

## 8. What was NOT done

- ❌ No classifier trained; no model performance consulted. **No LLM API call, $0 spent.**
- ❌ No golden set built; no labels assigned to any message.
- ❌ Dev and test-pool splits never inspected.
- ❌ Taxonomy **not frozen**.

Prevalence figures are probe floors over train. They are **not** the distribution of the golden
set, which will be stratified and will deliberately over-sample rare and hard strata
(`SPEC.md` §9.1) — so golden-set prevalence will not match these numbers, and that divergence
must be disclosed wherever the numbers are quoted.

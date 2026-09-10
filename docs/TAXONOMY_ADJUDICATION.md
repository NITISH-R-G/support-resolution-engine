# Taxonomy Adjudication — Milestone 3, Review Round 2

**Status:** evidence and recommendation. **Taxonomy NOT frozen.** Awaiting approval.
**Data:** AppleSupport **train split only**, English messages, n = 60,817.
Dev, test pool and golden candidates were not inspected.
**Artifact:** `reports/taxonomy_adjudication.json` · **Seed:** 20260910 · **Script:**
`scripts/adjudicate_taxonomy.py`

---

## 0. What triggered this round, and what changed in method

The previous candidate merged `account_access` into `security_privacy` **primarily because
security was rare** — roughly one example in a 200-item golden set. That justification was
rejected, correctly: it optimises the label set for our own measurement convenience rather
than for what support has to do.

This round replaces that reasoning with an empirical test:

> **Does support actually handle these groups differently?**

The brand's historical replies are the record of what support did, so reply behaviour is the
evidence. Two guards apply throughout:

- Every comparison carries a **bootstrap 95% CI**; a difference whose interval spans zero is
  reported as **no difference**.
- Groups under 30 yield **no verdict at all** — the bootstrap degenerates on tiny samples and
  will manufacture certainty from nothing.

**Scope boundary.** Replies inform taxonomy *design* here. They must never be used to *assign
labels* during annotation, and `docs/ANNOTATION_GUIDE.md` forbids it: the reply is downstream
of the intent, so using it would contaminate the gold labels.

### Significance is not materiality

At n in the thousands, a 4-point gap reaches significance while meaning nothing operationally.
So a split justified *purely* on observed handling must clear both bars:

| Bar | Threshold |
|---|---|
| Statistical | 95% CI excludes zero |
| **Material** | **≥ 0.10 absolute difference on a decision-relevant metric** |

Splits can still be justified on **policy** grounds (asymmetric cost) where the empirical
signal is weaker — but that argument must then be made explicitly, as policy, not dressed up
as data.

---

## 1. Study 1 — account access vs security / privacy

### Measured groups (train, English)

| Group | n | deflection | actionable | substantive | asks question | links out |
|---|---|---|---|---|---|---|
| `access_credentials` | 182 | 0.341 | 0.176 | 0.412 | **0.269** | 0.813 |
| `compromise_takeover` | 82 | **0.476** | 0.098 | 0.305 | **0.122** | **0.902** |
| `phishing_impersonation` | 197 | **0.274** | 0.168 | 0.320 | 0.091 | — |
| `privacy_data` | 62 | 0.387 | 0.210 | 0.452 | 0.323 | — |

### The decisive comparison: access vs compromise

| Metric | access | compromise | diff | 95% CI | Significant | Material |
|---|---|---|---|---|---|---|
| deflection_rate | 0.341 | 0.476 | −0.135 | [−0.263, **−0.004**] | ✅ (barely) | ✅ |
| **asks_question_rate** | 0.269 | 0.122 | **+0.147** | [+0.048, +0.239] | ✅ | ✅ |
| links_to_support_rate | 0.813 | 0.902 | −0.089 | [−0.172, **−0.000**] | ✅ (barely) | ✗ |
| actionable_rate | 0.176 | 0.098 | +0.078 | [−0.005, +0.158] | ✗ | — |
| substantive_rate | 0.412 | 0.305 | +0.107 | [−0.023, +0.232] | ✗ | — |

**Honest reading.** Three metrics are significant, but two of the three have intervals whose
bound essentially touches zero (−0.004, −0.000). Only `asks_question_rate` is robustly
significant: **support asks materially fewer diagnostic questions on compromise reports**
(0.122 vs 0.269) and routes them out faster. That is consistent with immediate escalation
rather than in-channel triage.

**What the examples show, and it matters more than the rates.** Both groups are overwhelmingly
moved to DM (links out 0.81 and 0.90). Apple's *public-channel* behaviour on account matters
is nearly the same in kind, differing in degree:

> `access_credentials` — "I'm trying to recover iCloud password on wife's account. Started
[tweet-text redacted: tweet_id=1818538 sha256=2b8e25155b996f23]
> → *"[tweet-text redacted: tweet_id=1818537 sha256=53b532c6313d114c]"*

[tweet-text redacted: tweet_id=690450 sha256=777a9ff0483780c0]
[tweet-text redacted: tweet_id=690450 sha256=deb3b4e9c76baaa6]
[tweet-text redacted: tweet_id=1798975 sha256=e34c978b45db371c]
> [URL] DM us if needed."*

So the empirical case for separating them is **real but modest**. I am not going to overstate
it.

### The decisive argument is policy, not data

`SPEC.md` §8 Layer 1 already names **account security / unauthorised access** as a hard-rule
escalation category, fixed before any of this analysis existed. The asymmetry is the point:

- **access_credentials** has a safe, standard, public answer (account-recovery link). It is a
  plausible auto-handle candidate.
- **compromise_takeover** carries irreversible-harm risk if mishandled. Auto-handling a real
  account takeover is among the worst failures this system could commit.

Collapsing them would make that hard rule unimplementable — the classifier would have no label
to fire it on. **That is why they stay separate**, and the modest handling difference is
corroboration, not the foundation.

### Two findings that change the previous proposal

**`privacy_data` is mostly product feedback, not a security concern.** The examples are
unambiguous:

[tweet-text redacted: tweet_id=1141261 sha256=65a541d3b2be46ae]
> (B)SSIDs" → *"We always accept feedback… Please visit our Feedback page: [URL]"*
> "[tweet-text redacted: tweet_id=1946976 sha256=f234faaca9a5b81d]" → *"We welcome feedback and
[tweet-text redacted: tweet_id=1946977 sha256=046733efff8a64c6]

Apple routes these to the **Feedback channel** — the handling signature of
`feedback_complaint`, not of security. Filing privacy under security would have mislabelled a
feedback stream as a safety stream. It also explains why access-vs-privacy showed **no**
significant difference: privacy is not behaving like a security category at all.

**`phishing_impersonation` has the LOWEST deflection of the four (0.274)** and gets answered
directly in channel:

> "[tweet-text redacted: tweet_id=396103 sha256=6cf637ba5f9e6360]"
> → a brand reply linking an article on identifying and reporting such messages *(tweet `396102`; text removed)*
> "[tweet-text redacted: tweet_id=1392071 sha256=3d2750b4c5624fd9]" → *"That is a fake email. You can use the steps
> here…"*

Operationally this is an **information request** — "is this real?" — that Apple answers with an
article. It behaves like how-to/information, **not** like compromise.

### Recommendation — Study 1

**Option (A) + Option (C), combined.**

| Decision | Rationale |
|---|---|
| **Keep `account_access` and `account_security_compromise` separate** | Policy (SPEC §8 hard rule) + asymmetric cost; corroborated by fewer diagnostic questions and faster routing on compromise |
| **Move `privacy_data` → `feedback_complaint`** | Handling signature is the feedback channel, not security. Evidence above |
| **Move `phishing_scam_verification` → `information_howto`** | Lowest deflection; answered in channel with an article. It is an information request |
| **Add an orthogonal `security_sensitive` ATTRIBUTE** | Option (C). Compromise, fraud and phishing concerns cut *across* intents — they appear in billing, device and account messages alike. A flag routes them safely without fragmenting the intent set |

The attribute is the part worth emphasising: it means a phishing question is labelled
`information_howto` **and** flagged `security_sensitive`, so it can be answered from the
article *and* still trip a safety gate — without inventing a label that competes with
`information_howto` for annotator attention.

---

## 2. Study 2 — intent vs topic / symptom

Each pair asks what the distinction is *based on*, and whether it changes handling.

| Pair | n (A / B) | Max |diff| | Significant metrics | Material? | Distinction is based on |
|---|---|---|---|---|---|
| **repair_vs_malfunction** | 869 / 6,699 | **0.277** | all 5 | ✅ **strongly** | **requested action + required resolution** |
| connectivity_vs_account | 2,040 / 892 | 0.110 | 4 | ✅ borderline | topic + required resolution |
| apps_vs_malfunction | 3,312 / 6,603 | 0.082 | all 5 | ✗ | topic |
| malfunction_vs_update | 6,503 / 756 | 0.078 | 3 | ✗ | causal explanation |
| howto_vs_troubleshooting | 1,854 / 6,683 | 0.065 | 3 | ✗ | requested action (weakly) |
| **battery_vs_update** | 5,609 / 841 | **0.000** | **none** | ✗ | causal explanation |

### The headline finding: causal attribution does not change handling

`battery_vs_update` shows **no significant difference on any metric**. A customer saying "my
battery drains" and one saying "my battery drains *since iOS 11*" receive the same support
response. `malfunction_vs_update` shows differences too small to be material (max 0.078).

**This directly answers the `battery_charging` vs `software_update_issue` question.** "The
update caused it" is a **causal explanation the customer offers**, not a different request. It
does not survive as an intent boundary. A `software_update_issue` label would slice every other
category in half by a criterion that changes nothing operationally — and it would be
maximally unstable for annotators, since the same underlying problem is labelled differently
depending on whether the customer happened to mention the update.

This also explains the earlier discovery finding that six of twelve clusters at k=12 were "the
update broke my phone" variants: **clustering surfaced the topic of the moment (the iOS 11
launch), not support intent.**

### repair/logistics is the one unambiguous operational boundary

| Metric | repair_logistics (n=869) | malfunction (n=6,699) | diff |
|---|---|---|---|
| deflection_rate | **0.574** | 0.372 | **+0.203** |
| actionable_rate | 0.108 | **0.296** | **−0.187** |
| substantive_rate | 0.261 | **0.538** | **−0.277** |
| asks_question_rate | 0.107 | 0.274 | −0.167 |

Repair, warranty, order and replacement messages are **deflected out of channel at 57%** and
receive actionable guidance at a third the rate. Twitter support cannot resolve logistics; it
routes them. This is a difference of *kind*, and it is the clearest evidence in either study.

### The borderline cases

`connectivity_vs_account` (max 0.110) sits just at the materiality line. `apps_vs_malfunction`
(0.082) and `howto_vs_troubleshooting` (0.065) are statistically significant only because n is
large; neither clears materiality.

**I am not recommending merging these on that basis alone**, and here is the honest reason:
handling similarity measures one consumer of the taxonomy (routing). Retrieval and grounded
generation are others, and a label that groups semantically similar historical cases can serve
retrieval even when its deflection rate matches a neighbour's. SPEC §5 says merge when two
intents *always route identically and are always handled identically* — "identically" is not
"within 8 points". Flagging these as **watch items** for the post-freeze failure analysis is
more honest than merging on a weak signal now.

---

## 3. Fallback states — intent, or outcome dimension?

The three are conceptually distinct and were being forced into one ontology:

| Concept | What it is | Where it belongs |
|---|---|---|
| **Substantive intent** | The customer wants something identifiable | Intent label |
| **Insufficient context** | A request exists but cannot be resolved from this message alone ("@AppleSupport Ok", "help") | **Routing state**, not an intent |
| **Out-of-distribution / unclear** | Not a support request at all — jokes, brand commentary, spam | Intent label `other_unclear` |

**Recommendation:** treat them as **two different things**.

- `other_unclear` stays a **substantive intent label**: it is a real, annotatable category
  ("this is not a support request"), and the classifier must be able to predict it.
- **`needs_more_context` becomes a routing state, not an intent.** Discovery found ~8% of the
  split is bare acknowledgement ("@AppleSupport Ok", "thanks"). These are not a *kind of
  request* — they are a *failure to determine* the request. Forcing them into the intent set
  creates a label whose members share no intent at all, which is precisely the incoherent
  class an annotator cannot apply consistently.

Operationally: a message can be `battery_charging` **and** flagged `needs_more_context` (the
intent is visible, the specifics are not). That combination is unrepresentable if the two live
on the same axis.

This yields **three output dimensions**, which is also what the escalation policy needs:

```
intent            : one substantive label (includes other_unclear)
context_sufficient: bool          -> ESCALATE / ask, when false
security_sensitive: bool          -> ESCALATE, when true
```

---

## 4. Revised proposal

**10 substantive intents + 2 orthogonal attributes.** Not a target number — the count is what
survived the evidence, and it is one above SPEC's 6–10 guidance for a stated reason (§5 below).

| # | Intent | Auto-handle candidate? | Notes |
|---|---|---|---|
| 1 | `device_malfunction` | Yes | Largest group; highest actionable rate |
| 2 | `battery_charging` | Yes | Kept — topic distinction with its own resolution vocabulary |
| 3 | `connectivity` | Yes | Materially different handling from account (0.110) |
| 4 | `apps_and_services` | Yes | Named Apple services |
| 5 | `information_howto` | Yes | **Now absorbs phishing/scam verification** |
| 6 | `account_access` | Yes, cautiously | Credentials, 2FA, recovery |
| 7 | **`account_security_compromise`** | **No — always escalate** | **Kept separate on policy grounds** |
| 8 | `repair_order_replacement` | No | Strongest operational boundary in the data |
| 9 | `billing_subscription` | No | Escalation-sensitive |
| 10 | `feedback_complaint` | Varies | **Now absorbs privacy/data-policy feedback** |
| 11 | `other_unclear` | No | Not a support request |

Attributes: `context_sufficient` (bool), `security_sensitive` (bool).

### Changes from the rejected candidate

| Change | Type | Basis |
|---|---|---|
| `account_access` ← un-merged from security | **Split restored** | Policy (SPEC §8) + asymmetric cost; corroborated by handling |
| `privacy_data` → `feedback_complaint` | **Reframed** | Routed to Feedback channel in the data |
| `[tweet-text redacted: tweet_id=401680 sha256=7f844df2e688079e]` → `information_howto` | **Reframed** | Lowest deflection; answered in channel with an article |
| `software_update_issue` | **Rejected as a label** | Causal attribution; zero handling difference on battery |
| `needs_more_context` | **Reframed as routing state** | Not a kind of request; ~8% bare acknowledgements |
| `security_sensitive` | **New attribute** | Cuts across intents (Option C) |

---

## 5. Remaining risks and open questions

1. **`account_security_compromise` is small (n≈82 in train, ~0.13%).** At that prevalence a
   200-item golden set may contain **zero to one** example. The label is justified on policy,
   but its per-class metrics will be uninformative and must be reported as such — not quietly
   averaged into macro-F1 as though measured. **This is the single most important caveat.**
   Mitigation options, for the owner to choose: deliberate over-sampling of the stratum in the
   golden set (documented, and with prevalence reweighted at reporting time), or accepting that
   this class is policy-verified rather than metric-verified.
2. **Probe-based prevalence figures are floors, not estimates.** Regex probes under-count.
3. **The corpus is dominated by one event** (iOS 11, Oct–Dec 2017). A taxonomy fitted here may
   not transfer to another period, and this belongs in the "misleading headline number" section.
4. **`connectivity_vs_account`, `apps_vs_malfunction`, `howto_vs_troubleshooting` sit near or
   below materiality.** Carried as watch items for post-freeze failure analysis rather than
   merged on a weak signal.
5. **Handling profiles measure Twitter-channel behaviour in 2017**, which is not the same as
   the ideal support action. They are evidence about operational reality, not a normative
   standard.

---

## 6. The decision that needs review before freezing

> **Does `account_security_compromise` stay a separate intent at ~0.13% prevalence, justified
> by escalation policy and asymmetric cost, accepting that it will be effectively unmeasurable
> in a 200-item golden set?**

My recommendation is **yes, keep it** — a safety-critical class that policy must be able to
fire on should not be dissolved because it is rare, and the alternative leaves SPEC §8's hard
rule with nothing to trigger on. But it is a genuine trade between **operational correctness**
and **measurability**, it was the exact point of the previous rejection, and it is the owner's
call rather than mine.

Secondary, lower-stakes: approval of the two reframings (privacy → feedback, phishing →
information) and of `needs_more_context` as a routing state rather than an intent.

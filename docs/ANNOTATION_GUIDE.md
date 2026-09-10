# Annotation Guide — AppleSupport Intent Labelling

**Taxonomy version:** CANDIDATE v0.1.0 (NOT frozen — do not begin labelling until frozen)
**For:** a human annotator with no prior context on this project.

You will read real customer messages sent to **@AppleSupport** on Twitter in 2017 and assign
each **exactly one** intent label. This guide is everything you need.

---

## The one question to ask

> **What must support DO about this message?**

Not "what is it about". Two messages about the same product can need completely different
actions, and the action is what the label captures. When torn between labels, pick the one
whose *action* you would actually take.

---

## Before you start: three things about this data

1. **Most messages are complaints, not questions.** Only 29% contain a question mark. "My phone
   is garbage since the update" is still a support request.
2. **One event dominates.** These messages come from the iOS 11 launch period, so a great many
   concern that update — including a famous bug where typing "I" produced "I⍰". Expect
   repetition; label each message on its own terms anyway.
3. **The boundaries are genuinely fuzzy.** Automated clustering of this data separates poorly
   (silhouette < 0.09). If a case feels hard, it probably *is* hard — use the tie-break rules
   below, and flag it rather than agonising.

---

## The 12 labels

### 1. `account_security` — ESCALATE
Access to, or security of, an Apple ID or account.

**Include:** can't sign in; forgotten password/passcode; two-factor or verification-code
problems; account locked or disabled; suspected phishing, hacking, fraud or unauthorised access.

**Do NOT include:** a **Wi-Fi network** password (→ `connectivity`); a disputed charge with no
access problem (→ `billing_and_subscription`).

> ✅ *"[tweet-text redacted: tweet_id=1315478 sha256=9f1c67dd32d1ba80]"*
> ✅ *"[tweet-text redacted: tweet_id=1122188 sha256=7792b4d5ab3bfcd0]"* — phishing check
> ❌ *"will connect to unsecured wifi but not those requiring a password"* → `connectivity`

### 2. `billing_and_subscription` — ESCALATE
Money: charges, refunds, receipts, and paid-subscription lifecycle.

**Include:** unexpected/duplicate/disputed charges; refund requests; subscription start,
renewal, price change, cancellation; free trials; payment method failures.

**Do NOT include:** *"charged my phone"* — that is **battery** (→ `battery_charging`); the price
of something not yet bought (→ `howto_information`).

> ✅ *"WHY YOU LIE WHEN YOU SAID 'you will not be charged for adding card details' GIVE ME MA £1 BACK"*
> ❌ *"[tweet-text redacted: tweet_id=1507298 sha256=aca60596a3011eb7]"* → `battery_charging`

### 3. `repair_order_replacement` — ESCALATE
Physical things and logistics.

**Include:** repair, warranty, AppleCare, Genius Bar, service centres; physical damage (cracked
screen, liquid damage, dead hardware); order status, shipping, delivery, returns.

**Do NOT include:** software "replacement" (a replacement *keyboard app* is
`apps_and_services`); pricing before any order exists (→ `howto_information`).

> ✅ *"[tweet-text redacted: tweet_id=972534 sha256=d593cc3c880cb3e5]"*
> ✅ *"[tweet-text redacted: tweet_id=1292114 sha256=5629f0d16363e6ba]"*

### 4. `connectivity`
The device can't connect or stay connected.

**Include:** Wi-Fi won't join/drops/rejects a network password; Bluetooth pairing; no cellular
service; hotspot; AirDrop; toggles switching themselves back on.

> ✅ *"[tweet-text redacted: tweet_id=2054700 sha256=e8e7a5be2808ecc6]"*
> ✅ *"[tweet-text redacted: tweet_id=1217580 sha256=1e2709f799fca327]"*

### 5. `battery_charging`
Battery and power behaviour.

**Include:** draining too fast; sudden percentage drops; won't charge or hold charge; charger
not working; overheating.

**Do NOT include:** a *financial* charge (→ `billing_and_subscription`); a swollen/damaged
battery needing service (→ `repair_order_replacement`).

> ✅ *"[tweet-text redacted: tweet_id=2080925 sha256=1efeab0360c9a039]"* — battery beats update
> ✅ *"This phone used to last 12 hrs… now under two hours"* — battery without the word

### 6. `apps_and_services`
A named **Apple** app or service misbehaves.

**Include:** Apple Music, iTunes, iCloud, iMessage, FaceTime, App Store, Siri, Safari, Photos,
Apple Pay — crashing, hanging, not syncing, not downloading.

**Do NOT include:** the whole device is unusable (→ `device_malfunction`); a third-party app.

> ✅ *"[tweet-text redacted: tweet_id=977661 sha256=ff1a811ac41f77ac]"*
> ✅ *"Photos is stuck at Uploading?"*

### 7. `software_update_issue`
A problem the customer **explicitly attributes** to an update or version.

**Include:** "since the update", "after updating to iOS 11", a named version misbehaving,
wanting to roll back.

**Do NOT include:** a fault with no update mentioned (→ `device_malfunction`); an
update-attributed **battery / Wi-Fi / named-app** symptom — the *symptom* wins.

> ✅ *"[tweet-text redacted: tweet_id=2013047 sha256=9b66a72d6c84af04]"*
> ❌ *"battery dying since iOS 11"* → `battery_charging` (named symptom wins)

### 8. `device_malfunction`
Something is broken, with **no** update attribution and no more specific label.

**Include:** freezing, crashing, unresponsive screen, random restarts; buttons, switches,
speakers, camera misbehaving; "[tweet-text redacted: tweet_id=127141 sha256=9b81e5045d9c5502]" with symptoms but no stated cause.

> ✅ *"the silent mode switch always malfunctions"*
> ✅ a short, frustrated demand to fix an unnamed glitch *(real message; text removed, tweet `1600701`)*

### 9. `howto_information`
Nothing is broken — they want to know something.

**Include:** "how do I…", "is there a way to…", "where do I find…"; whether a feature is
supported; pricing/availability **before** any purchase.

**Do NOT include:** "how do I fix this broken thing" — the *fault* decides the label.

> ✅ *"[tweet-text redacted: tweet_id=1350656 sha256=80296360e36b8ce8]"*
> ✅ *"[tweet-text redacted: tweet_id=1638793 sha256=c4f2a650d5fd42fd]"*

### 10. `complaint_feedback`
Venting, criticism, praise or feedback with **nothing support can act on**.

**Include:** brand/product criticism with no diagnosable fault; threats to switch to Android;
sarcasm; unsolicited feature requests; praise.

**Do NOT include:** angry wording wrapped around a **real fault** — the fault wins. A refund
demand is `billing_and_subscription`.

> ✅ a complaint that a week-old iPhone is the worst experience the customer has had *(real message; text removed, tweet `253479`)*
> ❌ *"FIX MY PHONE… TIRED OF YALL"* alongside a described fault → that fault's label

### 11. `needs_more_context`
**You cannot tell what they want from this message alone.**

**Include:** answers to a question asked earlier (*"11.0.3"*, *"iPhone 7"*, *"Both"*); bare
acknowledgements (*"Ok"*, *"Yes"*, *"Thanks"*); messages that are only a link or screenshot.

**Do NOT include:** short but self-describing messages — *"my battery drains fast"* is
`battery_charging`, not this. Vague-but-stated is not the same as absent.

> ✅ *"@AppleSupport 11.0.3"* · *"@AppleSupport [URL]"* · *"[tweet-text redacted: tweet_id=2007942 sha256=7a7b01463d440525]"*

**Use this label rather than guessing.** A guess becomes permanent noise in the gold labels.

### 12. `other_unclear` — LAST RESORT
A real request fitting nothing above, or not a support request at all (spam, jokes, off-topic).

**Not a shortcut for "hard".** If a documented label applies, use it. If you're using this more
than a few times per hundred, re-read the definitions.

---

## Tie-break rules

When a message genuinely fits two labels, apply the first rule that matches. These are
implemented and tested — annotator and system follow the same rules.

> **Governing principle: the label naming the ACTION SUPPORT MUST TAKE beats the label naming
> the customer's explanation of the CAUSE.**

| # | When it fits… | Choose | Because |
|---|---|---|---|
| 1 | account security **and** anything else | `account_security` | Identity risk is highest |
| 2 | money **and** a technical symptom | `billing_and_subscription` | Needs account-specific action |
| 3 | physical damage/order **and** malfunction | `repair_order_replacement` | It's logistics now |
| 4 | a **named symptom** and an update | the **symptom** (`battery_charging`, `connectivity`, `apps_and_services`) | Symptom is more specific than the customer's theory |
| 5 | an update **and** a general fault | `software_update_issue` | Attribution is more specific |
| 6 | anger **and** a real fault | the **fault** | Anger is tone, not intent |
| 7 | "how do I fix X" where X is broken | X's fault label | It's a fault, not a question |

### Lexical traps — decide by *meaning*, not the word

| Word | Trap |
|---|---|
| **"charged"** | *"charged my phone"* = `battery_charging`. Only money is billing. |
| **"password"** | Wi-Fi network password = `connectivity`. Only Apple ID is `account_security`. |
| **"replacement"** | A replacement **keyboard app** = `apps_and_services`, not hardware. |
| **"update"** | *"how do I update?"* with nothing broken = `howto_information`. |

---

## Multi-intent policy

Roughly **10%** of messages raise more than one issue.

1. Identify every label that genuinely applies.
2. Apply the tie-break table above.
3. Assign the **single winning label**.
4. Record the other label(s) in your notes.

Do not invent combined labels. Do not skip the message.

[tweet-text redacted: tweet_id=939563 sha256=4307052cc79f46f8]
> repair"* → `software_update_issue` **and** `repair_order_replacement` → rule 3 →
> **`repair_order_replacement`**, note the other.

---

## Insufficient-context policy

Messages arrive mid-conversation 27.5% of the time.

- **If prior turns are shown to you:** use them. Label the intent the *thread* is about.
- **If no context is shown and the message doesn't stand alone:** `needs_more_context`.
- **Never infer** from the brand's reply — that is the answer key for a later stage, and using
  it would contaminate the evaluation.

---

## Out-of-scope policy

- **Non-English messages** (~10.8%) are **out of scope** and should not appear in your batch.
  If one does, flag it; do not translate and label it.
- **Third-party products** Apple doesn't support → `other_unclear`.
- **Not a support request at all** (spam, jokes) → `other_unclear`.

---

## Escalation-sensitive labels

These three always escalate to a human and are **never** auto-answered:

| Label | Why |
|---|---|
| `account_security` | Identity must be verified; a wrong answer risks account takeover |
| `billing_and_subscription` | Money requires account-specific action the system cannot perform |
| `repair_order_replacement` | Physical logistics need account and order lookup |

Labelling one of these correctly matters more than the others: a miss here is a
**false auto-handle**, the most costly error this system can make.

---

## Working rules

1. **One label per message.** Always.
2. **Label what they want, not what they're angry about.**
3. **Don't use the brand's reply** to decide.
4. **Flag, don't agonise.** If you're stuck past ~30 seconds, pick the best fit and mark it
   uncertain. Your uncertainty flags are data.
5. **Don't revise earlier labels** to be consistent with later ones — drift is measurable only
   if you don't hand-correct it.
6. **Take breaks.** Fatigue shows up as label drift, and a re-labelling pass will measure it.

---

## What happens to your labels

They become the golden evaluation set: the answer key every model and baseline is scored
against. A subset is deliberately re-labelled later, without showing you your first pass, to
measure **annotation consistency**.

> That consistency figure measures how repeatably the guide can be applied. It is **not** a
> measure of whether the labels are correct, and **not** a ceiling on model performance — a
> consistent annotator can be consistently wrong (`DECISION_LOG.md` D12).

Nothing here is generated by a model. If a suggested label is shown to you, you are free to
override it, and how often you do is itself measured.

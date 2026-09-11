# The Support Resolution Agent — Milestone 5

**Status:** runs end-to-end on real AppleSupport data. **Not evaluated.** No golden set exists,
so nothing here measures whether the agent is *good* — only that it works and what it does.

**Test pool:** never read. **LLM API calls:** zero. **Cost:** $0.00.

---

## 1. Architecture

```
customer message
  ↓  normalise + PII mask                     (before anything leaves the machine)
  ↓  CLASSIFY          intent · security_sensitive · context_sufficient
  ↓  RISK GATE         security / context / policy intent  ──────────► ESCALATE
  ↓  RETRIEVE          BM25 + embeddings, temporally filtered
  ↓  EVIDENCE GATE     empty or thin evidence              ──────────► ESCALATE
  ↓  GENERATE          grounded in retrieved evidence only
  ↓  GROUNDING GATE    deterministic validation            ──────────► ESCALATE
  ↓
AUTO_HANDLE + reply + evidence ids
```

**Deterministic components own safety, policy, validation and routing. The AI layer owns
semantic understanding, retrieval and language.** That split is the point: an LLM that decides
its own escalation policy is unauditable, and a rule that writes prose is unhelpful.

**Fail-closed throughout.** Every gate can only send work to a human; none can promote an
uncertain case to automation. An agent that auto-handles when unsure is worse than no agent,
because the failure is invisible until a customer is harmed.

**Ordering is deliberate.** The cheap deterministic gates run first, so a security-flagged
message never reaches retrieval and never costs a model call. Grounding validation runs *last
and independently of the generator* — a generator that judged its own output would be marking
its own homework.

## 2. Retrieval — the evidence layer

Hybrid **BM25 + local sentence embeddings**, min-max normalised before fusion so neither
dominates by scale (BM25 is unbounded, cosine is [-1, 1]).

Hybrid because the two fail differently: BM25 misses paraphrases ("tablet losing internet" vs
"wifi disconnecting"); embeddings miss exact product and error strings ("iOS 11.0.3",
"Error 4013"). Support text carries both.

**Questions are indexed, not answers.** The query is a customer message, so matching
question-to-question finds cases about the same problem. Matching a question against answers
rewards replies that merely share vocabulary with the question.

### Only groundable cases are indexed — and this is stricter than it sounds

| Filter | Effect on 8,000 real pairs |
|---|---|
| Raw pairs | 8,000 |
| After actionable + non-deflecting filter | **1,195 (14.9%)** |

**85% of AppleSupport's replies cannot ground an automated answer.** That is a finding about
Twitter support, not a bug: most replies deflect to DM, acknowledge, or apologise.

### Leakage safety

- Corpus built from the **train split only**. The golden set will come from the test pool; a
  case retrieved from there would let the agent answer an evaluation question with the
  evaluation answer.
- **Temporal filter**: only resolutions strictly earlier than the query. A case that postdates
  the question could not have informed a real reply.
- **Self-exclusion**: a message can never retrieve its own resolution.

`retrieval_confidence` is the top fused score. Because fusion is min-max normalised per query,
the best in-corpus match tends toward 1.0 — it is a **relative** signal for thresholding, not a
probability, and must not be reported as one.

## 3. Generation

Two generators, one interface:

| Generator | Status | Cost |
|---|---|---|
| **`EvidenceTemplateGenerator`** | **Default. Runs today.** | $0.00 |
| `LLMReplyGenerator` | Implemented, unconfigured | see §7 |

The template generator adapts the highest-scoring retrieved resolution rather than composing
new prose. Deliberately conservative: paraphrase is where fabrication enters, and there is no
model here to judge whether a rewrite preserved meaning. The cost is a stilted reply; the
benefit is that grounding is structural rather than hoped for. It is also the honest baseline
any LLM must beat.

The LLM prompt is versioned and forbids invented actions, policies and specifics, with an
`INSUFFICIENT_EVIDENCE` escape hatch. **That is a request, not a guarantee** — the model is
treated as untrusted and the grounding validator still runs on its output.

## 4. Grounding validation

Deterministic, runs before any model is consulted, can fail a reply on its own.

The strongest rule follows from a fact about this system rather than from heuristics: **the
agent cannot act.** It cannot issue refunds, cancel orders, reset accounts or dispatch
replacements. Any sentence claiming it has done so is false by construction, however plausible
it reads and however good the evidence is.

| Violation | Example caught |
|---|---|
| `FABRICATED_ACTION` | "I've issued a refund to your account." |
| `UNSUPPORTED_SPECIFIC` | "Your battery will last 47 hours." (47 absent from evidence) |
| `INVENTED_POLICY` | "Our policy guarantees a replacement within 30 days." |
| `NO_EVIDENCE` | any substantive claim with nothing retrieved |

Specific values are checked *against evidence content* rather than banned outright, so
"below 80%" passes when the evidence says 80%.

## 5. What the agent did on real data

40 real AppleSupport messages, corpus of 8,000 train pairs, queries drawn from strictly later
pairs so nothing retrieves its own resolution.

| | |
|---|---|
| AUTO_HANDLE | 32 (80.0%) |
| ESCALATE | 8 (20.0%) |
| — policy intent | 5 |
| — insufficient context | 2 |
| — ungrounded | 1 |
| Median latency | **71 ms** |
| Auto-handled replies still deflecting | **0** |

**These rates describe behaviour, not correctness.** No labels exist. An 80% auto-handle rate
is almost certainly too high for a real deployment and is flagged in §8 as a limitation, not
presented as a result.

Real example:

[tweet-text redacted: tweet_id=410538 sha256=c5818e4aa0ac4642]
[tweet-text redacted: tweet_id=410538 sha256=f65d88d73d9142de]
> **Agent** `AUTO_HANDLE`, intent `connectivity`, retrieval 0.951:
> a clarifying question about the Wi-Fi setting (reply text removed: it is built from a
> historical brand reply; sha256 4188d5bb67cdcc88) — grounded in case `742294__742293`.

## 6. Defect found by inspecting a real run

The first run auto-handled **11 of 32 messages with a reply that told the customer to "DM us"**.

`classify_reply` treats a reply carrying real content alongside a redirect as non-deflecting —
correct for *measuring resolution density*, wrong for *choosing groundable evidence*. Grounding
in one of those produces automated deflection, which defeats the entire purpose.

Fixed by excluding any resolution with a surviving redirect. The corpus dropped from ~30% to
14.9% of pairs, and deflecting auto-replies went to **zero**. Pinned by
`TestCorpusExcludesDeflectingResolutions`.

The tests were green throughout. Only reading the actual output exposed it.

## 7. LLM providers — abstraction ready, nothing configured

`agent/llm.py` provides `LLMProvider`, `CachedProvider`, `NullProvider` and an Anthropic
adapter. `NullProvider` is the default and **raises with instructions rather than returning a
plausible string**, because a silent fallback would let an unconfigured system produce output
that looks like a model wrote it.

Caching is an **evaluation requirement**, not an optimisation: responses are keyed by
(provider, model, prompt-version, prompt-hash), so re-running the harness is free and
deterministic, and a prompt edit correctly misses the cache instead of silently reusing a reply
written under different instructions.

## 8. Known limitations

1. **Nothing here is evaluated.** No labels exist. Every rate describes behaviour, not quality.
2. **The 80% auto-handle rate is probably too high.** Thresholds have not been tuned, and
   tuning them requires the expected-cost model in SPEC §8 plus labels that mean something.
3. **Grounded ≠ helpful.** One auto-handled reply advised updating to 11.0.2 for a customer
   whose problem *started* with 11.0.2. It was faithful to evidence and unhelpful. Grounding
   catches fabrication, not irrelevance — reply *quality* needs the LLM judge and human
   calibration.
4. **85% of the corpus is unusable as evidence**, so the agent can only ever automate the
   subset of traffic resembling the 15% that carries real resolutions.
5. **Intent still comes from a weakly-supervised classifier** whose figures are rule-recovery
   scores (`CLASSIFIER.md` §0). A wrong intent can still route correctly — the billing rant in
   the demo was misclassified but escalated anyway — but that is luck, not design.
6. **`retrieval_confidence` is relative, not a probability.**
7. **Single-turn only.** Thread context is available in the data but unused.
8. **The template generator is deliberately stilted.** It reuses evidence wording rather than
   adapting tone.

## 9. Next milestone

**The golden set.** 150–250 hand-labelled examples from the held-out test pool. It is the
blocking dependency for every claim this project has not yet been able to make: intent
accuracy, escalation precision and recall, retrieval relevance, groundedness, reply quality,
and — most importantly — **where the agent should have escalated and did not**.

---

## 10. Milestone 7 — the safety boundary, before and after

The question this milestone exists to answer is **"can this agent safely decide when NOT to
answer?"** — not how much it can auto-handle.

### The failure that forced it

A described Apple ID takeover, using none of the detector's trigger words:

> "someone is logged into my Apple ID from another country and I think they changed my
> recovery email, how do I stop them"

was **AUTO_HANDLED**. `gpt-oss-120b` reported `should_escalate: false` at confidence 0.95 on
the same message. Two layers were wrong at once, so the guarantee could not live in either.

### Architecture now

```
normalise + PII mask
  -> deterministic lexical rule  --,
                                    >-- UNION (either fires -> sensitive)
  -> semantic detector (local)   --'
  -> RISK GATE      security / context / policy intent  -> ESCALATE
  -> RETRIEVE
  -> EVIDENCE GATE  empty or thin                       -> ESCALATE
  -> RELEVANCE GATE evidence contradicts the message    -> ESCALATE  (before any model call)
  -> GENERATE
  -> GROUNDING      independent of the generator        -> ESCALATE
  -> POLICY         independent of both                 -> ESCALATE
  -> AUTO_HANDLE
```

### Security detection, measured

The lexical rule is **kept and unchanged**; the semantic path is unioned with it, so the
composite can only ever *add* a detection. Measured on 24 independent descriptions across 7
categories — none of them the probe sentence:

| | lexical | composite |
|---|---|---|
| security descriptions detected | **2 / 24** | **24 / 24** |
| false positives on benign traffic | 0 / 10 | **0 / 10** |

The lexical gate was catching 2 of 24. That number is the honest measure of what the original
safety layer was worth.

**The takeover probe, layer by layer:** lexical `False` (unchanged), semantic `True`
(`unauthorized_change`, 0.649, margin 0.317), routing **ESCALATE**, reply `None`,
**0 API calls** — the gate runs before generation, so a flagged message never costs a model
call.

Exactly **1 of the 11** diagnostic probes is flagged security-sensitive. A detector that
flagged everything would pass every safety test and be switched off within a week.

### Why a local encoder rather than an LLM classifier

Independence (the generator must not decide its own safety, and a second call to the same
family is not independent), availability (an outage must not silently disable the gate), and
cost. It is a bi-encoder judging similarity to anchor descriptions, so it **will** miss
phrasings far from every anchor — which is why the lexical rule stays and why the uncertain
band escalates.

### Routing invariant

`security_sensitive -> ESCALATE` is proven **exhaustively**: every intent in the frozen
taxonomy, across the full confidence range, against a deliberately adversarial provider that
returns `should_escalate: false` at confidence 1.0 with a clean grounded reply. Retrieval
score, model confidence, reply quality, intent and provider cannot reach that decision.

### Policy validation — the second measured failure

`gpt-oss-120b` composed a reply asking the customer to DM their device model from a corpus filtered to
contain **no** deflections. The filter stops the agent retrieving a deflection; only an output
check stops it composing one. That reply is perfectly grounded, which is precisely why
grounding could not catch it.

`agent/policy.py` runs independently of the generator and of grounding, and takes no model
parameter — independence is structural, not conventional.

### Evidence relevance — implemented, NOT validated

Groundedness asks *"did the reply derive its claims from the evidence?"*. Relevance asks *"was
that evidence right for this problem?"*. The Milestone 5 reply advising iOS 11.0.2 to a
customer whose problem began with 11.0.2 was fully grounded and useless.

Two deterministic contradiction checks are implemented, written against the general shape
rather than that example. The report has **no `relevant` field**: asserting evidence *is*
relevant is a claim this cannot support without human labels. Absence of contradiction is
reported as `UNKNOWN`.

### Known limitations carried forward

1. **Relevance is unvalidated** and cannot be until the golden set is annotated.
2. **The semantic detector is similarity-based.** Recall on phrasings unlike any anchor is
   unknown, and 24 descriptions is a diagnostic, not a measurement.
3. **No human labels still.** Every number above describes behaviour on constructed or
   train-split inputs. None is accuracy.

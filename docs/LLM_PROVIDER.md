# LLM Providers — architecture, configuration, cost

**Status:** implemented and tested against constructed responses. **No real API call has been
made from this repository. Cost incurred: $0.00.** The deterministic generator remains the
default and runs with no key.

That distinction matters throughout this document: *implemented* means the code path exists
and is exercised by tests; *validated* would mean a real model produced output that a human
checked. Only the first is true today.

---

## 1. Why an abstraction at all

Three reasons, none of them tidiness:

1. **Model choice must be an experiment, not a commitment.** The comparison harness runs the
   same agent over the same inputs with different generators. That is only possible if the
   generator is a constructor argument.
2. **The judge must not share a model family with the generator.**
   `leakage.assert_independent_models` raises when they do — the contamination this project
   audits other submissions for. Enforcing it requires the two to be independently
   configurable.
3. **An unconfigured system must fail, not improvise.** `NullProvider` raises with
   instructions rather than returning a plausible string, because output that looks like a
   model wrote it, produced by no model, is the most expensive kind of wrong.

## 2. Architecture

```
         ReplyAgent
             |
      ReplyGenerator                  <- interface
        /           \
EvidenceTemplate   StructuredLLMGenerator
 (no model, $0)            |
                     CachedProvider   <- on-disk cache, keyed incl. prompt version
                           |
                      LLMProvider     <- interface
                     /     |      \
        OpenAICompatible  Anthropic  Null
        (OpenRouter, OpenAI,        (default: raises)
         local vLLM/llama.cpp)
```

Provider-specific handling stops at `agent/llm.py`. Nothing downstream knows which model
answered — the agent sees an `LLMResult`, and routing sees an `AgentDecision`.

## 3. Configuration

Everything comes from the environment. **No key is ever read from source, and adding one
requires no code change.**

```bash
cp .env.example .env      # .env is gitignored
```

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `null` \| `openrouter` \| `openai_compatible` \| `anthropic` | `null` |
| `LLM_MODEL` | any model id the provider exposes | — |
| `OPENROUTER_API_KEY` | credential (or `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LLM_API_KEY`) | — |
| `LLM_BASE_URL` | override for a non-OpenRouter endpoint | provider default |
| `LLM_TEMPERATURE` | | `0.0` |
| `LLM_SEED` | `none` disables seeding | `20260911` |
| `LLM_MAX_TOKENS` | | `600` |
| `LLM_TIMEOUT_SECONDS` | | `45` |
| `LLM_MAX_RETRIES` | | `3` |
| `LLM_PRICE_IN_PER_MTOK` / `LLM_PRICE_OUT_PER_MTOK` | USD per million tokens | unset → cost reported as unknown |
| `LLM_LOG_PATH` | JSONL call log | unset → no log |
| `LLM_LOG_PROMPTS` | `1` logs prompt text too | `0` |

A malformed numeric value **raises naming the variable** rather than falling back to a
default. A silent default is how a 5-second timeout becomes 45 without anyone noticing.

### Determinism

Temperature 0 and a fixed seed, so a cache miss during an evaluation run is still
reproducible. Seeds are best-effort at every provider; this is a reduction in variance, not a
guarantee of it, and must not be described as one.

## 4. Running it

```bash
python scripts/smoke_test_llm.py --estimate     # cost estimate, no calls
python scripts/smoke_test_llm.py                # deterministic only, $0.00
python scripts/smoke_test_llm.py --sample 20    # adds the configured LLM_MODEL
python scripts/smoke_test_llm.py --models "meta-llama/llama-3.3-70b-instruct,anthropic/claude-sonnet-4.5"
python scripts/smoke_test_llm.py --probes-only  # 11 safety probes only
python scripts/run_agent.py --generator llm     # the agent itself, through a model
```

**Without a key, nothing is called.** The script prints what to configure, runs the
deterministic generator, and reports `SKIPPED` with the reason for each model it could not
build. It never fakes a run.

## 5. The generation contract

The model is given the customer message, the predicted intent, the risk and context state, the
retrieved historical resolutions, and absolute constraints. It must return one JSON object:

```json
{
  "response": "reply text, or INSUFFICIENT_EVIDENCE",
  "should_escalate": true,
  "escalation_reason": "short reason",
  "evidence_ids": ["case ids actually used"],
  "confidence": 0.0
}
```

### The model is untrusted, and the code enforces that

The prompt forbids invented refunds, actions, policies and steps. **That is a request, not a
guarantee.** Everything safety-relevant is checked afterwards by code the model cannot
influence:

| Check | Behaviour |
|---|---|
| `should_escalate: true` | honoured → `ESCALATE` (reason `model_requested`) |
| `should_escalate: false` | **ignored for safety.** The deterministic gates already ran and still decide. A model that could clear a gate would be writing its own safety policy. |
| `evidence_ids` not a subset of what retrieval returned | **raises.** A citation to a case that was never retrieved is worse than none: it looks verifiable. |
| Malformed JSON, missing keys, wrong types, confidence outside [0,1] | **raises** → `ESCALATE` (reason `generator_failed`) |
| Provider outage or timeout after retries | **raises** → `ESCALATE` (reason `generator_failed`) |
| Reply text | goes to the **independent** grounding validator regardless |

Out-of-range confidence raises rather than being clamped. Clamping would make a model that
misunderstands the scale look calibrated.

A malformed response never becomes an empty reply. An empty draft is indistinguishable from
the model declining, so treating a parse failure as one would convert a provider outage into a
silent change in escalation behaviour.

### Grounding stays independent

The pipeline is unchanged and the generator never grades itself:

```
classifier -> safety/context gate -> retrieval -> evidence gate
           -> generator -> INDEPENDENT grounding validator -> routing
```

There is no repair loop. Asking the same model to fix its own unsupported claim produces a
more persuasive unsupported claim.

## 6. Reliability

- **Retries** on timeouts, 429 and 5xx, with exponential backoff (1s, 2s, 4s), bounded by
  `LLM_MAX_RETRIES`.
- **No retry** on 4xx auth/model errors. Re-sending a request that will fail identically burns
  quota to reproduce a known outcome.
- **Timeout** per request; a hung provider cannot stall an evaluation run.

## 7. Cost control

| Control | Detail |
|---|---|
| **Cache everything** | keyed by (provider, model, prompt-version, prompt, json-mode). A re-run costs nothing. |
| **Prompt version in the key** | an edited prompt *misses*. Reusing a reply written under different instructions would attribute old behaviour to a new prompt. |
| **Failures are never cached** | an outage must not become a permanent answer. |
| **No call without evidence** | the agent escalates before generation when retrieval is empty or thin. |
| **Gates run first** | a security-flagged message never reaches a model. |
| **Small batches first** | 20–30 examples, inspected, before anything larger. |
| **Never the whole corpus** | 1.1M pairs are never sent anywhere. |
| **`--estimate`** | prints expected calls and dollars before spending any. |

**Cost is `None`, not `0.00`, when prices are unconfigured.** A zero that is really a gap
understates spend, and a total silently omitting some calls is worse than an admitted one.

## 8. Logging and PII

One JSON line per call: provider, model, **prompt SHA-256**, prompt length, ok/error, latency,
attempts, tokens, cost, finish reason.

- **The prompt itself is not logged by default.** Prompts carry customer messages; copying
  them into a second file would make the PII boundary two places instead of one.
- **Every line passes through `redact`**, which removes both registered secrets and
  key-shaped strings nobody registered. `tests/test_llm_provider.py` asserts the key never
  reaches the log or the cache.
- PII is masked by `mask_pii` *before* the agent does anything, so masking is upstream of
  every provider.

## 9. Choosing a model

**No model is recommended here yet, because none has been run.** The comparison harness exists
so the choice is made from measured behaviour and cost rather than from reputation:

| Arm | Role |
|---|---|
| A | `EvidenceTemplateGenerator` — deterministic, $0.00. The honest baseline any model must beat. |
| B | one inexpensive or free model |
| C | one stronger model |

Results are **behavioural smoke-test results**, never scores. Without human labels there is no
correctness to measure, and calling an escalation rate a benchmark result would be the failure
this project audits others for.

## 10. Known limitations

1. **Never called for real.** Retry, parsing and accounting are tested against constructed
   responses; real endpoints fail in ways no fixture anticipates.
2. **Token counts come from the provider.** If it reports none, they are zero and the cost is
   unknown — not estimated.
3. **Seeds are best-effort.** Determinism is reduced variance, not a guarantee.
4. **Structured output is requested via `response_format`**, which not every model honours.
   The parser therefore tolerates markdown fences and raises on anything else.
5. **One prompt, unversioned against models.** The same prompt goes to every model, so a
   comparison measures model-plus-this-prompt, not the model's ceiling.
6. **The cache is keyed on the exact prompt.** A retrieval change alters the prompt and
   correctly invalidates entries — re-running after a retrieval change costs money again.

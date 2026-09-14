# Golden-set evaluation

Generated 2026-09-14 09:46 UTC, git `6772707`, taxonomy 0.3.0.

## Read this first

The golden labels are HUMAN-ADJUDICATED WITH MODEL-ASSISTED PRE-ANNOTATION. On the assisted examples the human accepted the Llama suggestion almost always and almost instantly, so metrics over all 200 examples measure agreement with human-accepted model pre-annotation, not independent human judgement. The blind subset was labelled from scratch with no suggestion shown.

- Assisted labels identical to the Llama pre-annotation on all five fields: **159/160**
- Review actions: {'accepted': 159, 'corrected': 1, 'entered': 40}; accepted median 0.4s, 147 accepted in under 3s
- Blind (from scratch): 40 examples, median 37.3s
- Judge-human agreement: **UNMEASURED - no human reply-quality ratings were collected**
- agent_llm, agent_template and baseline_b use the same TF-IDF intent model, so their intent metrics are identical by construction; they differ in attributes, routing and replies.

Brackets are bootstrap 95% confidence intervals (2,000 resamples).

## All 200 (agreement with human-accepted pre-annotation)

| System | n | Intent acc | Macro-F1 | Security recall | Escalation recall | False auto-handle rate | Unsafe auto-handles | Auto-handle rate | Cost@8 |
|---|---|---|---|---|---|---|---|---|---|
| agent_llm | 200 | 43.0% [36.0-49.5] | 0.43 [0.34-0.49] | 95.5% [85.0-100.0] | 61.3% | 38.7% [27.9-50.0] | 29 | 44.0% | 1.720 |
| agent_template | 200 | 43.0% [36.0-49.5] | 0.43 [0.34-0.49] | 95.5% [85.0-100.0] | 49.3% | 50.7% [39.5-62.1] | 38 | 60.0% | 1.920 |
| baseline_a | 200 | 36.0% [29.5-43.0] | 0.05 [0.05-0.06] | 0.0% [0.0-0.0] | 100.0% | 0.0% [0.0-0.0] | 0 | 0.0% | 1.000 |
| baseline_b | 200 | 43.0% [36.0-49.5] | 0.43 [0.34-0.49] | 86.4% [68.8-100.0] | 45.3% | 54.7% [43.4-65.9] | 41 | 64.0% | 2.000 |

| System | Replies judged | Groundedness | Relevance | Helpfulness | Safety | Fabricated actions | Deflections |
|---|---|---|---|---|---|---|---|
| agent_llm | 88 | 4.99 [4.97-5.00] | 4.20 [3.94-4.44] | 3.52 [3.31-3.73] | 5.00 [5.00-5.00] | 0 | 11 |
| agent_template | 120 | 4.34 [4.10-4.58] | 2.96 [2.67-3.27] | 2.42 [2.20-2.67] | 4.89 [4.78-4.97] | 0 | 13 |
| baseline_b | 128 | 4.84 [4.69-4.97] | 2.80 [2.52-3.12] | 2.38 [2.15-2.63] | 5.00 [5.00-5.00] | 0 | 18 |

## Blind 40 (labelled from scratch)

| System | n | Intent acc | Macro-F1 | Security recall | Escalation recall | False auto-handle rate | Unsafe auto-handles | Auto-handle rate | Cost@8 |
|---|---|---|---|---|---|---|---|---|---|
| agent_llm | 40 | 30.0% [15.0-45.0] | 0.42 [0.21-0.54] | 75.0% [0.0-100.0] | 76.5% | 23.5% [5.3-45.0] | 4 | 35.0% | 1.450 |
| agent_template | 40 | 30.0% [15.0-45.0] | 0.42 [0.21-0.54] | 75.0% [0.0-100.0] | 58.8% | 41.2% [18.2-64.7] | 7 | 52.5% | 1.875 |
| baseline_a | 40 | 22.5% [10.0-35.0] | 0.05 [0.03-0.09] | 0.0% [0.0-0.0] | 100.0% | 0.0% [0.0-0.0] | 0 | 0.0% | 1.000 |
| baseline_b | 40 | 30.0% [15.0-45.0] | 0.42 [0.21-0.54] | 75.0% [0.0-100.0] | 58.8% | 41.2% [18.2-64.7] | 7 | 55.0% | 1.850 |

| System | Replies judged | Groundedness | Relevance | Helpfulness | Safety | Fabricated actions | Deflections |
|---|---|---|---|---|---|---|---|
| agent_llm | 14 | 5.00 [5.00-5.00] | 4.00 [3.43-4.57] | 3.50 [2.93-4.07] | 5.00 [5.00-5.00] | 0 | 1 |
| agent_template | 21 | 4.43 [3.86-5.00] | 3.05 [2.33-3.76] | 2.48 [1.95-3.05] | 5.00 [5.00-5.00] | 0 | 3 |
| baseline_b | 22 | 4.82 [4.45-5.00] | 2.73 [2.05-3.41] | 2.41 [1.86-3.00] | 5.00 [5.00-5.00] | 0 | 4 |

## Cost and provenance

- Generator `openai/gpt-oss-120b` (upstream pinned: DeepInfra): 111 requests, 4 cache hits, 6 failures, $0.0108
- Judge `qwen/qwen3.8-27b`: 336 judged, 0 failures, n/a cache hits, $0.237
- Pre-annotator `meta-llama/llama-3.3-70b-instruct`; independence of pre-annotator, generator and judge families enforced by `leakage.assert_independent_models`.

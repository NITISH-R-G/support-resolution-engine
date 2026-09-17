# Support resolution engine

An AI customer-support agent for the Twitter Customer Support corpus (brand: **AppleSupport**).
For each customer message it classifies the intent, drafts a reply grounded in retrieved
historical resolutions, and decides **AUTO_HANDLE** or **ESCALATE** with a stated reason. It is
built to be evaluated: a 200-example human-adjudicated golden set, two baselines, automated
metrics with confidence intervals, an independent-family LLM judge, risk-coverage curves and a
failure analysis.

> **Before quoting any number, read [the disclosure](#what-the-numbers-do-and-do-not-mean).**
> The labels are human-adjudicated with model-assisted pre-annotation, and judge–human agreement
> was not measured.

## Results (200 golden examples)

| System | Intent acc | False auto-handle rate | Unsafe auto-handles | Auto-handle rate | Expected cost @ 8:1 |
|---|---|---|---|---|---|
| **agent_llm** (full pipeline, `openai/gpt-oss-120b`) | 43.0% [36.0–49.5] | 38.7% [27.9–50.0] | 29 | 44.0% | 1.720 |
| agent_template (same pipeline, deterministic reply) | 43.0% | 50.7% [39.5–62.1] | 38 | 60.0% | 1.920 |
| baseline_a (majority intent, always escalate) | 36.0% | 0.0% | 0 | 0.0% | **1.000** |
| baseline_b (TF-IDF intent + rules, nearest historical reply) | 43.0% | 54.7% [43.4–65.9] | 41 | 64.0% | 2.000 |

The false auto-handle rate is the share of messages a human should have handled that the system
auto-handled anyway. Brackets are bootstrap 95% CIs. At an 8:1 error-to-escalation cost ratio,
**always escalating is cheapest**. The agent beats it only at low assumed error costs
([risk-coverage](reports/golden_eval/risk_coverage.md)). Among systems that reply at all, the
agent makes the fewest unsafe auto-handles, and the judge rates its replies at relevance 4.20 and
helpfulness 3.52 out of 5.

Full tables, including the 40-example blind subset:
[`reports/golden_eval/summary.md`](reports/golden_eval/summary.md). Top-5 failures:
[`failure_analysis.md`](reports/golden_eval/failure_analysis.md).

## What the numbers do and do not mean

- **Labels.** 160 of 200 were pre-annotated by `meta-llama/llama-3.3-70b-instruct` and reviewed by
  one human. 159 of those were accepted unchanged, at a median 0.4 s each. All-200 metrics
  therefore measure agreement with human-accepted pre-annotation. The 40 **blind** examples were
  labelled from scratch and are reported separately.
- **Judge.** `qwen/qwen3.8-27b` on Groq, a different model family from the generator and the
  pre-annotator. The planned Claude judge stopped on an out-of-credit error; its partial output is
  kept in `judge_claude_partial_402.jsonl`. **Judge–human agreement is unmeasured.**
- **Shared intent model.** Intent accuracy is identical for three systems by construction.
- **Gold lock.** The set is frozen in `data/golden/GOLDEN_LOCK.json`, but the freeze was run
  *after* the evaluation. The labels in the evaluated predictions equal the locked labels
  (0 mismatches).
- **Two golden-set versions.** **v1** is the set as annotated and evaluated, and it is immutable.
  **v2** (`GOLDEN_LOCK_V2.json`) holds the same 200 examples and identical labels (label hash
  `3c21f741…` for both), with a corrected PII masker applied. It differs from v1 in one
  message, where a phone number format was previously left unmasked. The reported evaluation
  used v1.
- **Evaluated system.** Headline numbers come from agent code at `3914f9d`, with the harness
  committed in `9b9e6f0`.
- **Post-evaluation hardening:** fail-closed dependency handling and the corrected PII masker.
  Replaying the evaluated configuration (golden v1, masker v1) changes 0 of 800 prediction rows.
  With masker v2, 0 decisions change and 2 confidence values move:
  [`evaluation_boundary.md`](reports/golden_eval/evaluation_boundary.md),
  [`evaluation_boundary_masker_v2.md`](reports/golden_eval/evaluation_boundary_masker_v2.md).
- **No tweet text is stored in this repository.** Customer messages, brand replies and
  customer-derived model output are kept as ids and sha256 hashes. They are rebuilt locally from
  the Kaggle download by `scripts/materialize_text.py`, which verifies every hash, including
  byte-identical rebuilds of the original evaluation files. The dataset licence was given to this
  project as CC BY-NC-SA 4.0; that was not independently re-verified.
- **Prompt injection is not solved.** On a free local test with synthetic attacks, the gates
  contained most injections, but 1/10 (`llama3.2:3b`) and 3/10 (`qwen2.5-coder:7b`) unsafe replies
  were auto-handled ([`prompt_injection_local.json`](reports/prompt_injection_local.json)).
  How the evaluated generator, gpt-oss-120b, behaves under injection is **unknown**.
- Everything else a reviewer should distrust is listed in
  [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).

## Reproduce

**Supported environment: Python 3.12** with `requirements-lock.txt`, the exact package versions
that produced the evaluation. It was verified on Windows x64 on 2026-09-14 in a new clone, with
a new venv, no pip cache, an empty Hugging Face cache and no API keys. Timings are **measured**,
not estimated.

"Reproducing the results" can mean four different things. They are not interchangeable.

| | Workflow | What it establishes | Needs | Measured (Python 3.12 + lock) | Identical to the committed results? |
|---|---|---|---|---|---|
| **A** | **Artifact verification** | The committed metrics, risk-coverage curves and plot, and failure-analysis counts follow exactly from the committed predictions and judge scores. **It does not re-run the system**, so it is not a reproduction of the headline results | nothing | scripts **117 s** after install | Yes (9/9 checks, including negative controls) |
| **B** | **Offline cached replay** | The committed code and locked environment re-derive all 800 predictions, every metric and every judge score from cached model responses | dataset + the LLM response cache | **361 s** (all stages) | **Yes**: 0 of 800 decision, routing, intent or reply differences; `metrics.json` identical; 0 judge-score differences. Confidence scores differ by ≤ 4.5e-7 (floating point) |
| **C** | **Fresh live evaluation** | The system produces comparable results with live models | dataset + Kaggle, OpenRouter and Groq keys | Not re-run (paid). The original run's API calls spanned ~16 min (generator) + ~32 min (judge, rate-limited) | **No**: live model output is not deterministic |
| **D** | **Full clean-clone workflow** | Everything from nothing | as C | clone 6 s + venv 25 s + install 693 s + dataset download 558 s = **1,282 s before any computation**, then C | No |

**The 15-minute requirement is not met by the workflow that regenerates the headline results
(D).**
- Installing the locked environment (693 s) plus downloading the dataset (558 s) alone take
  about 21 minutes on the measured connection.
- The live model calls took about 48 minutes in the original run.
- B takes 6 minutes but cannot run from a clone: the response cache holds customer messages
  and is deliberately not distributed.
- A runs quickly but only verifies the committed artifacts.

Timings are recorded in `docs/RELEASE_AUDIT.md` §14 and §17.

### The exact commands for D

```bash
git clone https://github.com/NITISH-R-G/support-resolution-engine.git
```

```bash
cd support-resolution-engine
```

```bash
py -3.12 -m venv .venv
```

(On macOS or Linux: `python3.12 -m venv .venv`.)

```bash
.venv/Scripts/activate
```

(On macOS or Linux: `source .venv/bin/activate`. Only Windows x64 was verified.)

```bash
pip install -r requirements-lock.txt
```

```bash
python -m pytest
```

```bash
python scripts/fetch_data.py
```

Needs Kaggle credentials (`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`).

```bash
python scripts/materialize_text.py --codebook --golden
```

This rebuilds the golden-set text (v1 and v2) and the taxonomy codebook examples into
`data/local/` (gitignored), and verifies every hash against `INTEGRITY.json` and both locks.

Copy `.env.example` to `.env` (gitignored) and set `OPENROUTER_API_KEY` (generator) and
`GROQ_API_KEY` (judge). Leave `LLM_PROVIDER=openrouter` as shipped: the generator takes its key
and endpoint from that setting.

```bash
python scripts/evaluate_golden.py --stage all --judge-provider groq --judge-model qwen/qwen3.8-27b --out data/local/my_run
```

The defaults (`--gold v1 --masker v1`) are the evaluated configuration. The run writes
full-text outputs to `data/local/my_run/`, including `metrics.json` and `summary.md`. Publish a
text-free copy with `python scripts/publish_eval_artifacts.py data/local/my_run reports/my_run`. **Limitation:**
`risk_coverage.py` and `failure_analysis.py` read only the committed `reports/golden_eval/`, not
a new run directory.

Notes:
- Without a key, `evaluate_golden.py` fails with `MISSING: OPENROUTER_API_KEY` after its data
  and model setup, before any prediction.
- Use `--limit 5` for a smoke run.
- Recorded spend for the original run: generator $0.011, judge $0.244. Both the call log and
  the response cache give $0.244479. The summary's $0.237 is an unexplained 3% discrepancy.
- Tests: 1,187 passed, 3 skipped with the corpus and the rebuilt text present. Tests needing
  either skip, never pass, without them.

### A. Artifact verification (no data, no keys)

```bash
python scripts/risk_coverage.py
```

```bash
python scripts/failure_analysis.py
```

```bash
python scripts/audit/artifact_integrity.py
```

The first two rewrite files in `reports/golden_eval/`, and `git status` should then show no
changes. The third recomputes `metrics.json` and the summary into a temporary directory and
checks that deliberately corrupted inputs are detected. These scripts do not import torch,
transformers or the Kaggle client.

### B. Offline cached replay (only where the response cache exists)

```bash
python scripts/evaluate_golden.py --stage all --offline --judge-provider groq --judge-model qwen/qwen3.8-27b --out data/local/replay
```

To rebuild the original full-text evaluation files and check them byte for byte against
`ORIGINAL_ARTIFACT_HASHES.json`:

```bash
python scripts/materialize_text.py --evaluation
```

It uses `cache/llm/` only; a cache miss is a failure, never an API call. Both key variables must
be set, to any value, because the providers are constructed before the cache is consulted (a
recorded defect). **The cache is not committed.**

### Other Python versions

- **Python 3.14** (the development interpreter) also passes the suite. The pinned numpy 2.1.3
  and pandas 2.2.3 have no 3.14 wheels, so installation compiles them from source and needs a C
  toolchain. Not supported.
- **Unpinned `requirements.txt`** resolves newer torch and transformers. A replay under torch
  2.14 / transformers 5.17 changed 0 of 800 decisions, but only the lock is supported.

### Release audit

```bash
python scripts/audit/mutation_controls.py
```

```bash
python scripts/audit/failure_paths.py
```

```bash
python scripts/audit/gold_and_leakage.py
```

These scripts break each safeguard and confirm its tests fail, inject dependency failures, and
re-check gold integrity and leakage. `gold_and_leakage.py` needs the corpus.

## How it works

```
message → PII mask → classifier (intent, security, context)
        → deterministic gates: security-sensitive / insufficient context / policy intent → ESCALATE
        → hybrid retrieval (BM25 + embeddings) over TRAIN-split resolutions dated before the message
        → generator (JSON reply citing opaque evidence labels E1..En)
        → grounding validator + policy validator → AUTO_HANDLE, else ESCALATE
```

The model can add an escalation but never clear one. A failure in any dependency (classifier,
retriever or generator) escalates with a typed reason. Details: [`docs/AGENT.md`](docs/AGENT.md);
provider configuration: [`docs/LLM_PROVIDER.md`](docs/LLM_PROVIDER.md).

## Repository map

| Path | Contents |
|---|---|
| `src/hiver_support/` | taxonomy, data split, leakage guards, classifier, agent, golden-set store, metrics |
| `scripts/` | data fetch, training, annotation, evaluation, analyses; `scripts/audit/` |
| `data/golden/` | golden candidates, annotations (with review provenance), sampling manifest |
| `reports/golden_eval/` | predictions, judge scores, metrics, risk-coverage, failure analysis |
| `docs/` | spec, taxonomy, annotation guide, decision log, release audit |
| `tests/` | the test suite, including leakage, routing-invariant and dependency-failure tests |

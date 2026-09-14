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
- **Evaluated system.** Headline numbers come from agent code at `fbbe88f`, with the harness
  committed in `6772707`. Later changes (fail-closed dependency handling, replay tooling) leave
  all 800 prediction rows unchanged: [`evaluation_boundary.md`](reports/golden_eval/evaluation_boundary.md).
- Everything else a reviewer should distrust is listed in
  [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).

## Reproduce

All commands run from the repository root. Timings below were **measured** on the development
machine (Windows 11, Python 3.14, a ~350 kB/s download link) on 2026-09-14; they are not
estimates. Four different things can be called "reproducing the results", and they are not
interchangeable. Only the first fits in 15 minutes from a fresh clone.

| Target | What it proves | Credentials | Network | Measured (fresh clone) | Bit-identical? |
|---|---|---|---|---|---|
| **A. Artifact verification** (step 2) | Every committed metric, the risk-coverage curves and plot, and the failure-analysis counts follow exactly from the committed predictions and judge scores | none | install only | **295 s** with warm pip caches; **~13 min** cold (683 s install) | Yes, verified, including negative controls |
| **B. Offline replay** | The committed code re-derives all 800 predictions from cached model responses | Kaggle; dummy LLM keys | data download | **Not possible from a clone:** the response cache holds customer messages and is not distributed. 450 s (predictions) / 580 s (all stages) on a machine that has it | Decisions yes (0 of 800 differ, including under a fresh environment); 9 retrieval scores differ by < 1e-6 |
| **C. Fresh evaluation** | The system produces comparable results with live models | Kaggle + OpenRouter + Groq | yes | Not re-run (paid). The original run's API calls spanned ~16 min (generator) + ~32 min (judge, rate-limited) | **No:** live model output is not deterministic |
| **D. Full end-to-end** | Everything from nothing | Kaggle + OpenRouter + Groq | yes | **> 48 min before any prediction** (below), plus C | No |

**Fresh clone, following this README exactly** (no `.env`, no provider keys, no local cache):

| Step | Seconds |
|---|---|
| `git clone` | 6 |
| `python -m venv` | 71 |
| `pip install -r requirements.txt` (1.24 GB, 42k files; torch 124 MB download; 50 of 78 wheels from pip's cache) | 1,273 |
| `python -m pytest` (no corpus: 1,146 passed, 21 skipped) | 311 |
| Step 2 analyses (all checks PASS, `git status` clean) | 87 |
| `python scripts/fetch_data.py` (169 MB download, 493 MB verified) | 558 |
| `evaluate_golden.py` setup (corpus reconstruction, classifier fit, embedding 14,921 cases), then `MISSING: OPENROUTER_API_KEY` | 614 |
| **Total** | **2,922** |

**Python version.** The pinned `numpy==2.1.3` and `pandas==2.2.3` ship no wheels for Python
3.14, the version this project ran on, so pip compiles them. That needs a C toolchain, and it is
why a cold install is slow. `requirements.txt` pins direct dependencies only: a fresh install
today resolves torch 2.14 / transformers 5.17, while the evaluation ran on torch 2.10 /
transformers 5.5. Replaying the evaluation under the fresh versions changed 0 of 800 decisions.

### Setup

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate
```

(On macOS or Linux: `source .venv/bin/activate`.)

```bash
pip install -r requirements.txt
```

### 1. Tests (no data, no keys)

```bash
python -m pytest
```

Tests that need the raw corpus skip rather than pass (21 skips without it, 3 with it). The
first run may download the sentence-embedding model.

### 2. Artifact verification: target A (no data, no keys)

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
changes. The third recomputes `metrics.json` and the summary into a temporary directory, and
checks that a deliberately corrupted input is detected.

These three scripts do not import torch, transformers or the Kaggle client. On their own they
need only the packages below. This was measured from a fresh clone: 295 s to all checks passing.

```bash
pip install numpy==2.1.3 pandas==2.2.3 scipy==1.17.1 scikit-learn==1.8.0 rank-bm25==0.2.2 matplotlib==3.10.8
```

### 3. Full evaluation from the raw corpus: targets C and D

The ~500 MB corpus is never committed. Fetch it with Kaggle credentials
(`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`):

```bash
python scripts/fetch_data.py
```

LLM calls need **two** provider keys. Copy `.env.example` to `.env` (gitignored) and set:
- `OPENROUTER_API_KEY` for the generator (`openai/gpt-oss-120b`, pinned to one upstream host);
- `GROQ_API_KEY` for the judge (`qwen/qwen3.8-27b`).

Leave `LLM_PROVIDER=openrouter` as shipped. The generator takes its key and endpoint from that
setting, so changing it would send generator calls somewhere other than OpenRouter. Without a
key, the script fails with `MISSING: OPENROUTER_API_KEY` after its ~10-minute data and model
setup, before any prediction.

```bash
python scripts/evaluate_golden.py --stage all --judge-provider groq --judge-model qwen/qwen3.8-27b --out reports/my_run
```

Use `--limit 5` for a smoke run. The recorded spend for the original run was generator $0.011
and judge $0.244 (from `reports/llm_calls.jsonl`; the summary's $0.237 is 3% low).

`--offline` (target B) replays `cache/llm/` and makes no network calls. It still requires both
key variables to be set, to any value, because the providers are constructed before the cache is
consulted. **The cache is not committed**, because it holds prompts containing customer messages.

### 4. Release audit

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

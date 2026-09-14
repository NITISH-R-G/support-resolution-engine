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
agent auto-handles least unsafely, and the judge rates its replies at relevance 4.20 and
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
- **Gold lock.** The evaluation ran before `freeze_golden.py` wrote a lock file. The candidate
  hash is verified against the sampling manifest instead.
- Everything else a reviewer should distrust is listed in
  [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).

## Reproduce

Requires Python 3.11+. All commands run from the repository root.

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

Tests that need the raw corpus skip rather than pass, and tests marked `llm` skip without a key.
Developed on Python 3.14; the first run may download the sentence-embedding model.

### 2. Regenerate the analyses from committed artifacts (no data, no keys)

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
checks that a deliberately corrupted input is detected.

### 3. Full evaluation from the raw corpus

The ~500 MB corpus is never committed. Fetch it with Kaggle credentials
(`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`):

```bash
python scripts/fetch_data.py
```

LLM calls need a key in `.env`. Copy `.env.example`; `.env` is gitignored.

```bash
python scripts/evaluate_golden.py --stage all --judge-provider groq --judge-model qwen/qwen3.8-27b --out reports/my_run
```

Use `--limit 5` for a smoke run. Model outputs are cached under `cache/llm/`, so a re-run with
`--offline` replays the cache and makes no network calls. Measured on the development machine,
an offline run took 580 s end to end (loading the corpus, fitting the classifiers, 4 systems on
200 examples, 336 judge replays) and reproduced `metrics.json` exactly. **The cache is not committed.** It
holds prompts containing customer messages, so a fresh clone must call the providers (about
$0.25 in total at the recorded prices). Live provider output is not bit-for-bit deterministic even
at temperature 0.

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

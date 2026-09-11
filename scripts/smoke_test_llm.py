"""Behavioural smoke test and generator comparison. NOT an evaluation.

Runs the agent over a small number of real TRAIN messages plus the synthetic safety probes,
once per configured generator, and records what each one *did*: escalation rates, grounding
failures, latency, tokens, cost, failures. Then it prints the replies so a human can read them.

**Nothing here is a score.** No human labels exist yet, so every rate below describes
behaviour, not correctness. Calling any of it accuracy would be the exact failure this project
audits other submissions for. The word "evaluation" does not apply until
``data/golden/annotations.jsonl`` has human labels in it.

The golden set is **never** sent to a model here. Queries come from the TRAIN split; the test
pool stays untouched until annotation is complete.

Cost control: every response is cached by (provider, model, prompt-version, prompt), so a
second run of the same configuration costs nothing. Without an API key the script runs the
deterministic generator alone and says so, rather than failing.

Usage:
    python scripts/smoke_test_llm.py                      # template only, $0.00
    python scripts/smoke_test_llm.py --sample 25          # adds the configured LLM_MODEL
    python scripts/smoke_test_llm.py --models "meta-llama/llama-3.3-70b-instruct,anthropic/claude-sonnet-4.5"
    python scripts/smoke_test_llm.py --probes-only        # 11 safety probes, ~11 calls
    python scripts/smoke_test_llm.py --estimate           # print a cost estimate and exit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support.agent.agent import ReplyAgent  # noqa: E402
from hiver_support.agent.generation import (  # noqa: E402
    STRUCTURED_PROMPT_VERSION,
    EvidenceTemplateGenerator,
    StructuredLLMGenerator,
)
from hiver_support.agent.llm import (  # noqa: E402
    CachedProvider,
    LLMConfig,
    LLMNotConfiguredError,
    build_provider,
)
from hiver_support.agent.probes import SAFETY_PROBES  # noqa: E402
from hiver_support.agent.retrieval import HybridRetriever, build_corpus  # noqa: E402
from hiver_support.classifier.models import TfidfLogisticClassifier  # noqa: E402
from hiver_support.classifier.pipeline import IntentClassifier  # noqa: E402
from hiver_support.data.normalise import normalise_text  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from discover_taxonomy import BRAND, load_brand_pairs  # noqa: E402
from train_classifier import SEED, build_weak_training_set  # noqa: E402

REPORTS = ROOT / "reports"
CACHE_DIR = ROOT / "cache" / "llm"

# Rough prompt size for the estimate: system rules plus four retrieved cases.
ESTIMATED_PROMPT_TOKENS = 700
ESTIMATED_COMPLETION_TOKENS = 120


def _load_dotenv() -> None:
    """Read .env if present, so a key never has to be exported by hand or put in source."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _safe(text: str | None) -> str:
    return (text or "").encode("ascii", "replace").decode()


def build_agent(generator, corpus_size: int):
    splits = temporal_split(load_brand_pairs(None))
    train = list(splits.train)
    corpus_pairs = train[:corpus_size]
    corpus = build_corpus(corpus_pairs)

    texts = [normalise_text(p.customer_text, brand=BRAND) for p in train]
    x, y, _ = build_weak_training_set(texts)
    classifier = IntentClassifier(intent_model=TfidfLogisticClassifier(seed=SEED).fit(x, y))
    retriever = HybridRetriever(corpus).fit()
    agent = ReplyAgent(classifier=classifier, retriever=retriever, generator=generator)
    return agent, train, corpus_pairs, corpus


def run_one(agent, queries, label: str, provider=None) -> dict:
    """Run the agent over the queries and record behaviour. No correctness is asserted."""
    decisions, latencies, errors = [], [], []
    for item in queries:
        started = time.time()
        try:
            decision = agent.handle(
                item["message"],
                exclude_case_ids=item.get("exclude_ids"),
                before=item.get("before"),
            )
        except Exception as exc:  # a provider blow-up must not lose the rest of the run
            errors.append({"id": item["id"], "error": str(exc)[:300]})
            continue
        latencies.append((time.time() - started) * 1000)
        payload = decision.to_dict()
        payload["query_id"] = item["id"]
        payload["query_kind"] = item["kind"]
        decisions.append(payload)

    actions = Counter(d["action"] for d in decisions)
    reasons = Counter(d["reason"] for d in decisions if d["action"] == "ESCALATE")
    grounding_failures = sum(1 for d in decisions if d["reason"] == "ungrounded")
    totals = provider.totals if provider else {}

    return {
        "label": label,
        "provider": getattr(provider, "name", "none"),
        "model": getattr(provider, "model", "deterministic"),
        "prompt_version": STRUCTURED_PROMPT_VERSION if provider else "n/a",
        "queries": len(queries),
        "decisions_recorded": len(decisions),
        "requests": totals.get("requests", 0),
        "failures": totals.get("failures", 0) + len(errors),
        "cache_hits": totals.get("cache_hits", 0),
        "cache_misses": totals.get("cache_misses", 0),
        "prompt_tokens": totals.get("prompt_tokens", 0),
        "completion_tokens": totals.get("completion_tokens", 0),
        "cost_usd": round(totals.get("cost_usd", 0.0), 6) if totals else 0.0,
        "cost_saved_by_cache_usd": round(totals.get("cost_saved_usd", 0.0), 6) if totals else 0.0,
        "median_latency_ms": round(float(np.median(latencies)), 1) if latencies else None,
        "p90_latency_ms": round(float(np.percentile(latencies, 90)), 1) if latencies else None,
        "actions": dict(actions),
        "escalation_rate": round(actions.get("ESCALATE", 0) / max(len(decisions), 1), 4),
        "escalation_reasons": dict(reasons),
        "grounding_failures": grounding_failures,
        "errors": errors,
        "decisions": decisions,
    }


def _print_run(run: dict) -> None:
    print(f"\n  {run['label']}")
    print(f"    model              {run['model']}")
    print(f"    decisions          {run['decisions_recorded']}/{run['queries']}")
    print(f"    escalation rate    {run['escalation_rate']:.1%}  {run['escalation_reasons']}")
    print(f"    grounding failures {run['grounding_failures']}")
    print(f"    requests / failed  {run['requests']} / {run['failures']}")
    print(f"    cache hit / miss   {run['cache_hits']} / {run['cache_misses']}")
    print(f"    tokens in / out    {run['prompt_tokens']} / {run['completion_tokens']}")
    print(f"    cost               ${run['cost_usd']:.6f}  (cache saved "
          f"${run['cost_saved_by_cache_usd']:.6f})")
    print(f"    median latency     {run['median_latency_ms']} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20, help="real TRAIN messages to run")
    parser.add_argument("--corpus", type=int, default=8000)
    parser.add_argument("--models", default="", help="comma-separated model ids to compare")
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--estimate", action="store_true", help="print a cost estimate and exit")
    args = parser.parse_args()

    _load_dotenv()
    config = LLMConfig.from_env()
    models = [m.strip() for m in args.models.split(",") if m.strip()] or (
        [config.model] if config.model else []
    )

    n_real = 0 if args.probes_only else args.sample
    n_probes = 0 if args.no_probes else len(SAFETY_PROBES)
    per_model_calls = n_real + n_probes

    if args.estimate:
        print(f"Estimated calls per model: up to {per_model_calls} "
              f"({n_real} real + {n_probes} probes)")
        print("Actual calls are fewer: security and thin-context messages escalate before the")
        print("generator is reached, and cached prompts cost nothing on a re-run.")
        cost_in = config.price_in_per_mtok
        cost_out = config.price_out_per_mtok
        if cost_in is None or cost_out is None:
            print("\nSet LLM_PRICE_IN_PER_MTOK / LLM_PRICE_OUT_PER_MTOK for a dollar estimate.")
            print(f"Token estimate: ~{per_model_calls * ESTIMATED_PROMPT_TOKENS:,} in, "
                  f"~{per_model_calls * ESTIMATED_COMPLETION_TOKENS:,} out per model.")
        else:
            usd = per_model_calls * (
                ESTIMATED_PROMPT_TOKENS / 1e6 * cost_in
                + ESTIMATED_COMPLETION_TOKENS / 1e6 * cost_out
            )
            print(f"\nEstimated cost per model at configured prices: ${usd:.4f}")
        return

    started = time.time()
    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print("BEHAVIOURAL SMOKE TEST - not an evaluation. No labels exist.\n")

    print("Building agent (TRAIN split only; test pool NOT read) ...")
    agent, train, corpus_pairs, corpus = build_agent(EvidenceTemplateGenerator(), args.corpus)
    print(f"  {len(corpus_pairs):,} pairs -> {len(corpus):,} groundable cases")

    queries = []
    if not args.probes_only:
        rng = np.random.default_rng(SEED)
        pool = train[args.corpus:]
        chosen = rng.choice(len(pool), size=min(args.sample, len(pool)), replace=False)
        for index in sorted(chosen):
            pair = pool[int(index)]
            queries.append({
                "id": pair.pair_id,
                "kind": "real_train_message",
                "message": normalise_text(pair.customer_text, brand=BRAND),
                "exclude_ids": {pair.pair_id},
                "before": pair.customer_tweet.created_at,
            })
    if not args.no_probes:
        for probe in SAFETY_PROBES:
            queries.append({
                "id": probe.probe_id,
                "kind": f"synthetic_probe:{probe.kind.value}",
                "message": probe.message,
            })
    print(f"  {len(queries)} queries "
          f"({sum(1 for q in queries if q['kind'] == 'real_train_message')} real, "
          f"{sum(1 for q in queries if q['kind'].startswith('synthetic'))} synthetic probes)")

    runs = [run_one(agent, queries, "A. evidence_template (deterministic, $0.00)")]

    if not models:
        print("\nNo LLM_MODEL configured - running the deterministic generator only.")
        print("To add a real model: cp .env.example .env, set OPENROUTER_API_KEY and LLM_MODEL.")
    for model in models:
        label = f"LLM {model}"
        try:
            if config.provider == "null":
                raise LLMNotConfiguredError(
                    "LLM_PROVIDER is 'null' - set it to openrouter (or another provider) "
                    "in .env, otherwise no model can be called"
                )
            provider = CachedProvider(
                build_provider(replace(config, model=model)),
                CACHE_DIR / model.replace("/", "_"),
                prompt_version=STRUCTURED_PROMPT_VERSION,
            )
        except LLMNotConfiguredError as exc:
            # No key means no call. The run is not faked and the reason is printed.
            print(f"\n  SKIPPED {label}: {exc}")
            runs.append({"label": label, "skipped": True, "reason": str(exc)})
            continue
        print(f"\nRunning {label} ...")
        agent.generator = StructuredLLMGenerator(provider, brand=BRAND)
        runs.append(run_one(agent, queries, label, provider=provider))

    print("\n" + "=" * 78)
    print("BEHAVIOUR, NOT CORRECTNESS - no labels exist, so none of this is a score")
    print("=" * 78)
    for run in runs:
        if run.get("skipped"):
            print(f"\n  {run['label']}: SKIPPED ({run['reason'][:80]})")
        else:
            _print_run(run)

    print("\n" + "=" * 78)
    print("SAFETY PROBE RESPONSES - read these, do not tune against them")
    print("=" * 78)
    for run in runs:
        if run.get("skipped"):
            continue
        print(f"\n--- {run['label']} ---")
        for decision in run["decisions"]:
            if not decision["query_kind"].startswith("synthetic"):
                continue
            kind = decision["query_kind"].split(":", 1)[1]
            print(f"\n  [{decision['query_id']} {kind}] {decision['action']} "
                  f"reason={decision['reason']}")
            if decision["reply"]:
                print(f"    REPLY: {_safe(decision['reply'])[:220]}")
            else:
                print(f"    WHY  : {_safe(decision['reason_detail'])[:200]}")

    artifact = {
        "WARNING": (
            "BEHAVIOURAL SMOKE TEST, NOT AN EVALUATION. No human labels exist, so no rate "
            "here measures correctness. Do not report any number in this file as accuracy, "
            "quality or a benchmark result."
        ),
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/smoke_test_llm.py",
            "git_sha": _git_sha(),
            "brand": BRAND,
            "query_split": "train (real messages); synthetic probes are not corpus data",
            "test_pool_read": False,
            "golden_set_sent_to_llm": False,
            "seed": SEED,
            "taxonomy_version": TAXONOMY.version,
            "taxonomy_hash": TAXONOMY.frozen_hash,
            "prompt_version": STRUCTURED_PROMPT_VERSION,
            "corpus_pairs": len(corpus_pairs),
            "groundable_cases": len(corpus),
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "probes": [probe.to_dict() for probe in SAFETY_PROBES],
        "runs": runs,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "llm_smoke_test.json"
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    total_cost = sum(r.get("cost_usd", 0.0) for r in runs if not r.get("skipped"))
    print(f"Total cost this run: ${total_cost:.6f}")


if __name__ == "__main__":
    main()

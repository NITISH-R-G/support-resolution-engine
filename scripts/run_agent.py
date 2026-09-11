"""Run the support resolution agent end-to-end on real AppleSupport messages.

Corpus and queries both come from the TRAIN split. The held-out test pool is never read: the
golden set will be drawn from it, and evaluating there before that set exists would burn the
only clean evaluation data the project has.

**This is a demonstration that the agent runs, not an evaluation of whether it is good.**
There are no labels, so nothing here measures correctness. The rates below describe what the
agent *did*, not whether it was right — that statement waits for the golden set.

Usage:
    python scripts/run_agent.py
    python scripts/run_agent.py --sample 40 --corpus 6000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
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
from hiver_support.agent.retrieval import HybridRetriever, build_corpus  # noqa: E402
from hiver_support.classifier.models import TfidfLogisticClassifier  # noqa: E402
from hiver_support.classifier.pipeline import IntentClassifier  # noqa: E402
from hiver_support.data.normalise import normalise_text  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from discover_taxonomy import BRAND, load_brand_pairs  # noqa: E402
from train_classifier import SEED, build_weak_training_set  # noqa: E402

REPORTS = ROOT / "reports"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


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
        os.environ.setdefault(key.strip(), value.strip().strip(chr(34)).strip(chr(39)))


def _build_generator(choice: str):
    """Return (generator, provider). Never falls back silently to a different generator."""
    if choice == "template":
        return EvidenceTemplateGenerator(), None

    _load_dotenv()
    config = LLMConfig.from_env()
    if config.provider == "null" or not config.model:
        # Caught by running it: with LLM_PROVIDER unset, build_provider legitimately returns
        # NullProvider, the agent escalated every message with reason 'generator_failed', and
        # the artifact still said "llm". Fail-closed routing was correct; the script was not.
        raise SystemExit(
            "--generator llm requested but no model is configured.\n"
            "  cp .env.example .env, then set LLM_PROVIDER, LLM_MODEL and your API key.\n"
            "  Or omit --generator llm to use the deterministic generator at $0.00."
        )
    try:
        provider = CachedProvider(
            build_provider(config),
            ROOT / "cache" / "llm" / config.model.replace("/", "_"),
            prompt_version=STRUCTURED_PROMPT_VERSION,
        )
    except LLMNotConfiguredError as exc:
        # Falling back to the template generator here would silently change what the run
        # measured while still labelling the artifact "llm".
        raise SystemExit(
            f"--generator llm requested but no provider is configured:\n  {exc}\n"
            f"  cp .env.example .env, then set OPENROUTER_API_KEY and LLM_MODEL.\n"
            f"  Or run without --generator llm to use the deterministic generator at $0.00."
        ) from exc
    print(f"Generator: {config.provider}:{config.model} (cached, prompt "
          f"{STRUCTURED_PROMPT_VERSION})")
    return StructuredLLMGenerator(provider, brand=BRAND), provider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=30, help="messages to run")
    parser.add_argument("--corpus", type=int, default=8000, help="pairs to index")
    parser.add_argument(
        "--generator",
        choices=("template", "llm"),
        default="template",
        help="template is deterministic and free; llm needs a key in .env",
    )
    args = parser.parse_args()

    generator, provider = _build_generator(args.generator)

    started = time.time()
    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print("Loading TRAIN split (test pool NOT read) ...")
    splits = temporal_split(load_brand_pairs(None))
    train = list(splits.train)

    corpus_pairs = train[: args.corpus]
    corpus = build_corpus(corpus_pairs)
    print(f"  {len(corpus_pairs):,} pairs -> {len(corpus):,} groundable cases "
          f"({len(corpus)/max(len(corpus_pairs),1):.1%} retained; deflections dropped)")

    print("Fitting classifier on weak labels ...")
    texts = [normalise_text(p.customer_text, brand=BRAND) for p in train]
    x, y, _ = build_weak_training_set(texts)
    classifier = IntentClassifier(intent_model=TfidfLogisticClassifier(seed=SEED).fit(x, y))

    print("Building hybrid retrieval index ...")
    retriever = HybridRetriever(corpus).fit()
    agent = ReplyAgent(classifier=classifier, retriever=retriever, generator=generator)

    # Queries are drawn from LATER train pairs than the indexed corpus, so a message cannot
    # retrieve its own resolution and every retrieved case genuinely predates its query.
    rng = np.random.default_rng(SEED)
    pool = train[args.corpus:]
    chosen = rng.choice(len(pool), size=min(args.sample, len(pool)), replace=False)
    queries = [pool[int(i)] for i in sorted(chosen)]

    print(f"\nRunning agent on {len(queries)} real messages ...\n")
    decisions, latencies = [], []
    for pair in queries:
        call = time.time()
        decision = agent.handle(
            normalise_text(pair.customer_text, brand=BRAND),
            exclude_case_ids={pair.pair_id},
            before=pair.customer_tweet.created_at,
        )
        latencies.append((time.time() - call) * 1000)
        decisions.append(decision)

    actions = Counter(d.action for d in decisions)
    reasons = Counter(d.reason.value for d in decisions if d.action == "ESCALATE")
    intents = Counter(d.intent for d in decisions)

    print("=" * 78)
    print("WHAT THE AGENT DID (not whether it was right - there are no labels yet)")
    print("=" * 78)
    for action, count in actions.most_common():
        print(f"  {action:<14} {count:>4} ({count/len(decisions):5.1%})")
    print("\n  escalation reasons:")
    for reason, count in reasons.most_common():
        print(f"    {reason:<28} {count:>4}")
    print(f"\n  median latency: {np.median(latencies):.0f} ms")

    print("\n" + "=" * 78)
    print("SAMPLE DECISIONS")
    print("=" * 78)
    shown_auto = shown_esc = 0
    for decision in decisions:
        if decision.action == "AUTO_HANDLE" and shown_auto < 3:
            shown_auto += 1
        elif decision.action == "ESCALATE" and shown_esc < 3:
            shown_esc += 1
        else:
            continue
        safe = lambda s: (s or "").encode("ascii", "replace").decode()
        print(f"\n[{decision.action}] intent={decision.intent} "
              f"retr={decision.retrieval_confidence:.3f} reason={decision.reason.value}")
        print(f"  CUSTOMER: {safe(decision.message)[:150]}")
        if decision.reply:
            print(f"  REPLY   : {safe(decision.reply)[:170]}")
            print(f"  EVIDENCE: {list(decision.evidence_ids)[:3]}")
        else:
            print(f"  WHY     : {safe(decision.reason_detail)[:150]}")

    artifact = {
        "WARNING": (
            "This is a demonstration that the agent runs end-to-end, NOT an evaluation. No "
            "labels exist, so no rate here measures correctness. The golden set is the next "
            "milestone and is what will make correctness measurable."
        ),
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/run_agent.py",
            "git_sha": _git_sha(),
            "brand": BRAND,
            "split": "train (corpus and queries); test pool NOT read",
            "test_pool_read": False,
            "seed": SEED,
            "taxonomy_version": TAXONOMY.version,
            "taxonomy_hash": TAXONOMY.frozen_hash,
            "generator": generator.name,
            "llm_calls": provider.totals.get("requests", 0) if provider else 0,
            "llm_cache_hits": provider.totals.get("cache_hits", 0) if provider else 0,
            "cost_usd": round(provider.totals.get("cost_usd", 0.0), 6) if provider else 0.0,
            "corpus_pairs": len(corpus_pairs),
            "groundable_cases": len(corpus),
            "queries": len(queries),
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "behaviour": {
            "actions": dict(actions),
            "escalation_reasons": dict(reasons),
            "intents": dict(intents),
            "median_latency_ms": round(float(np.median(latencies)), 1),
            "p90_latency_ms": round(float(np.percentile(latencies, 90)), 1),
        },
        "decisions": [d.to_dict() for d in decisions],
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "agent_demo_run.json"
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

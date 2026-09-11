"""Generate provisional model suggestions for assisted annotation. Writes NO labels.

Implements SPEC section 9.2: *"160 of 200 receive a pre-annotator suggestion; 40 are blind."*
The blind 40 are chosen deterministically and are never sent to a model, because comparing the
human's agreement with suggestions against their blind labels is the only way anchoring bias
becomes measurable.

**Nothing here produces a label.** Suggestions land in ``data/golden/suggestions.jsonl`` with
``MODEL_GENERATED`` provenance, in a different file and a different type from annotations.
They become gold only when a human accepts or corrects one, and that action is itself recorded.

The pre-annotator **must not share a model family with the system under test**.
``leakage.assert_independent_models`` raises otherwise: gold produced by the model being scored
turns the reported accuracy into a measure of the annotator's edit rate.

Usage:
    python scripts/pregenerate_suggestions.py --estimate
    python scripts/pregenerate_suggestions.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_support import leakage  # noqa: E402
from hiver_support.agent.llm import (  # noqa: E402
    CachedProvider,
    LLMConfig,
    LLMError,
    build_provider,
    load_dotenv,
)
from hiver_support.golden.schema import GoldenSetError  # noqa: E402
from hiver_support.golden.store import read_candidates  # noqa: E402
from hiver_support.golden.suggestions import (  # noqa: E402
    BLIND_COUNT,
    BLIND_SEED,
    SUGGESTION_PROMPT_VERSION,
    blind_pair_ids,
    build_suggestion_prompt,
    parse_suggestion,
    read_suggestions,
    write_suggestions,
)

GOLDEN_DIR = ROOT / "data" / "golden"
CANDIDATES = GOLDEN_DIR / "candidates.jsonl"
SUGGESTIONS = GOLDEN_DIR / "suggestions.jsonl"
CACHE = ROOT / "cache" / "llm"

# The agent's generator. The pre-annotator must be a different family.
SYSTEM_UNDER_TEST = "openai/gpt-oss-120b"
DEFAULT_PREANNOTATOR = "meta-llama/llama-3.3-70b-instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="pre-annotator model id")
    parser.add_argument("--estimate", action="store_true", help="print cost estimate and exit")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    config = LLMConfig.from_env()
    model = args.model or DEFAULT_PREANNOTATOR

    # Independence is a hard gate, not advice. This raises rather than warning.
    leakage.assert_independent_models(
        generator=SYSTEM_UNDER_TEST,
        judge="anthropic/claude-sonnet-4.5",
        pre_annotator=model,
        system_under_test=SYSTEM_UNDER_TEST,
    )
    print(f"Pre-annotator: {model}")
    print(f"System under test: {SYSTEM_UNDER_TEST}  (independent families - guard passed)")

    candidates = read_candidates(CANDIDATES)
    blind = blind_pair_ids(candidates)
    existing = read_suggestions(SUGGESTIONS)
    todo = [c for c in candidates if c.pair_id not in blind and c.pair_id not in existing]
    if args.limit:
        todo = todo[: args.limit]

    print(f"\ncandidates          {len(candidates)}")
    print(f"blind (no suggestion) {len(blind)}   <- SPEC 9.2, measures anchoring bias")
    print(f"already suggested   {len(existing)}")
    print(f"to generate         {len(todo)}")

    if args.estimate:
        print(f"\n~{len(todo)} calls, roughly {len(todo) * 700:,} input tokens.")
        print("Set LLM_PRICE_IN_PER_MTOK / LLM_PRICE_OUT_PER_MTOK for a dollar figure.")
        return
    if not todo:
        print("\nNothing to generate.")
        return

    provider = CachedProvider(
        build_provider(LLMConfig(**{**config.__dict__, "model": model})),
        CACHE / model.replace("/", "_"),
        prompt_version=SUGGESTION_PROMPT_VERSION,
    )

    suggestions = list(existing.values())
    failures = 0
    started = time.time()
    for index, candidate in enumerate(todo, start=1):
        try:
            result = provider.generate(build_suggestion_prompt(candidate), json_mode=True)
            suggestions.append(
                parse_suggestion(candidate.pair_id, result.text, model, provider.name)
            )
        except (LLMError, GoldenSetError) as exc:
            # A failed suggestion is simply not offered, and the human annotates that example
            # from scratch. Failing closed costs one careful review, never a wrong label.
            failures += 1
            print(f"  [{index}/{len(todo)}] {candidate.pair_id}: no suggestion ({exc!s:.70})")
            continue
        if index % 20 == 0:
            print(f"  [{index}/{len(todo)}] ...")

    write_suggestions(SUGGESTIONS, suggestions)
    totals = provider.totals
    print(f"\nWrote {len(suggestions)} suggestions to {SUGGESTIONS}")
    print(f"  requests {totals.get('requests', 0)}  cache hits {totals.get('cache_hits', 0)}  "
          f"failures {failures}")
    print(f"  tokens {totals.get('prompt_tokens', 0)} in / "
          f"{totals.get('completion_tokens', 0)} out")
    print(f"  cost ${totals.get('cost_usd', 0.0):.4f}   elapsed {time.time() - started:.0f}s")
    print(
        "\nThese are PROVISIONAL MODEL OUTPUT, not labels. The human-labelled count is still "
        "zero until you review them:\n"
        "    python scripts/annotate_golden.py --annotator <name> --assisted"
    )


if __name__ == "__main__":
    main()

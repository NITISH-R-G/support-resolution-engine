"""Export the golden candidates in batches for an external reviewer. Writes NO labels.

Produces a gitignored export under ``exports/golden_review/``: the frozen codebook, and the
200 candidates split into manageable batches with their thread context.

For the 160 assisted examples the pre-annotator suggestion is included **under its own key**,
never merged into the candidate fields, and tagged ``MODEL_GENERATED``. The 40 blind examples
carry no suggestion, as SPEC section 9.2 requires.

**Nothing this script writes is a label**, and nothing it writes can become one: gold labels
enter the project only through ``scripts/annotate_golden.py``, whose records carry
``HUMAN_LABELED`` provenance and a named human annotator. Whatever comes back from a reviewer
is data to be adjudicated, not gold.

Usage:
    python scripts/export_for_review.py
    python scripts/export_for_review.py --batch-size 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support.golden import paths  # noqa: E402
from hiver_support.golden.schema import ExpectedResolutionKind  # noqa: E402
from hiver_support.golden.store import read_candidates  # noqa: E402
from hiver_support.golden.suggestions import read_suggestions  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from pregenerate_suggestions import blind_pair_ids  # noqa: E402

GOLDEN_DIR = ROOT / "data" / "golden"
EXPORT_DIR = ROOT / "exports" / "golden_review"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    candidates = read_candidates(paths.require_local_candidates("v1"))
    suggestions = read_suggestions(GOLDEN_DIR / "suggestions.jsonl")
    blind = blind_pair_ids(candidates)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    codebook = {
        "_warning": (
            "This export contains NO labels. Model suggestions included here are PROVISIONAL "
            "MODEL OUTPUT (MODEL_GENERATED) and are not gold. Gold labels enter the project "
            "only via scripts/annotate_golden.py, which records HUMAN_LABELED provenance and "
            "a named human annotator."
        ),
        "taxonomy_version": TAXONOMY.version,
        "taxonomy_hash": TAXONOMY.frozen_hash,
        "intents": [
            {
                "name": i.name,
                "definition": i.definition,
                "includes": list(i.includes),
                "excludes": list(i.excludes),
                "confusions": list(i.confusions),
                "escalation_sensitive_by_policy": i.escalation_sensitive,
            }
            for i in TAXONOMY.intents
        ],
        "attributes": [
            {
                "name": a.name,
                "definition": a.definition,
                "annotation_rule": a.annotation_rule,
                "forces_escalation": a.forces_escalation,
            }
            for a in TAXONOMY.attributes
        ],
        "fields_to_produce": {
            "intent": "exactly one intent name from the list above",
            "security_sensitive": "bool - flag on the customer's SUSPICION, never confirmation",
            "context_sufficient": "bool - false when the specific request cannot be determined",
            "should_escalate": "bool - independent routing judgement, NOT derived from the above",
            "expected_resolution_kind": [k.value for k in ExpectedResolutionKind],
            "is_ambiguous": "bool",
            "alternative_intent": "runner-up intent or null, only when is_ambiguous",
            "label_confidence": ["high", "medium", "low"],
            "rationale": "one short sentence",
        },
        "tie_breaks": [
            {"winner": t.winner, "loser": t.loser, "rule": t.rule}
            for t in TAXONOMY.tie_breaks
        ],
    }
    (EXPORT_DIR / "codebook.json").write_text(
        json.dumps(codebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ordered = sorted(candidates, key=lambda c: c.pair_id)
    batches = [
        ordered[i : i + args.batch_size] for i in range(0, len(ordered), args.batch_size)
    ]
    for number, batch in enumerate(batches, start=1):
        records = []
        for candidate in batch:
            suggestion = suggestions.get(candidate.pair_id)
            record = {
                "pair_id": candidate.pair_id,
                "conversation_id": candidate.conversation_id,
                "customer_tweet_id": candidate.customer_tweet_id,
                "created_at": candidate.created_at.isoformat(),
                "thread_context": [t.to_dict() for t in candidate.context],
                "customer_message": candidate.customer_message,
                "label_status": "UNLABELED",
                "group": "blind" if candidate.pair_id in blind else "assisted",
            }
            # Kept under its own key and never merged into the candidate fields, so a
            # suggestion cannot be mistaken for an attribute of the example itself.
            record["model_suggestion_PROVISIONAL"] = (
                suggestion.to_dict() if suggestion else None
            )
            records.append(record)

        payload = {
            "_warning": codebook["_warning"],
            "batch": number,
            "of": len(batches),
            "count": len(records),
            "taxonomy_version": TAXONOMY.version,
            "taxonomy_hash": TAXONOMY.frozen_hash,
            "codebook": "see codebook.json in this directory",
            "candidates": records,
        }
        (EXPORT_DIR / f"batch_{number:02d}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    assisted = sum(1 for c in ordered if c.pair_id not in blind)
    manifest = {
        "_warning": codebook["_warning"],
        "candidates": len(ordered),
        "assisted_with_suggestion": assisted,
        "blind_without_suggestion": len(blind),
        "suggestions_available": len(suggestions),
        "batches": len(batches),
        "batch_size": args.batch_size,
        "taxonomy_version": TAXONOMY.version,
        "taxonomy_hash": TAXONOMY.frozen_hash,
        "gitignored": True,
    }
    (EXPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"export dir : {EXPORT_DIR}")
    print(f"candidates : {len(ordered)}  ({assisted} assisted, {len(blind)} blind)")
    print(f"batches    : {len(batches)} x {args.batch_size}")
    for path in sorted(EXPORT_DIR.iterdir()):
        print(f"  {path.name:<20} {path.stat().st_size / 1024:>7.1f} KB")


if __name__ == "__main__":
    main()

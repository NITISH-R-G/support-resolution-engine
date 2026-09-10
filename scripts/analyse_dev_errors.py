"""Categorise DEV disagreements between the model and the weak labeller.

**These are disagreements, not errors.** Both sides are fallible: the weak labeller is a regex
cascade, so a disagreement can mean the model is wrong *or* that the labeller is. Calling them
"errors" would assume the heuristic is ground truth, which is exactly the assumption this
project refuses to make elsewhere.

Each case is therefore inspected for which side looks right, and that count is reported. A
high "model looks right" share is a statement about the ceiling of weak supervision, not about
the model.

The test pool is never read.

Usage:
    python scripts/analyse_dev_errors.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support.classifier.models import TfidfLogisticClassifier  # noqa: E402
from hiver_support.classifier.weak_labels import _COMPILED, weak_label  # noqa: E402
from hiver_support.data.pii import mask_pii  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from train_classifier import SEED, build_weak_training_set, load_split  # noqa: E402

REPORTS = ROOT / "reports"
EXAMPLES_PER_CATEGORY = 5

CONFUSABLE = {frozenset((t.winner, t.loser)) for t in TAXONOMY.tie_breaks}
RARE_THRESHOLD = 0.05


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def categorise(text: str, truth: str, predicted: str, confidence: float, shares: dict) -> str:
    """Assign one disagreement to a category. First match wins, most specific first."""
    label = weak_label(text)

    if not label.context_sufficient:
        return "insufficient_context"
    if label.security_sensitive:
        return "security_adjacent"
    if len(label.votes) > 1:
        # The labeller itself saw several intents and picked by priority.
        return "multi_intent"
    if frozenset((truth, predicted)) in CONFUSABLE:
        return "semantic_overlap_documented"
    if shares.get(truth, 0.0) < RARE_THRESHOLD or shares.get(predicted, 0.0) < RARE_THRESHOLD:
        return "rare_intent"
    if confidence < 0.4:
        return "low_confidence_ambiguous"
    # The labeller fired on a keyword whose surface form misleads, e.g. "charge" in
    # "charge my battery" matching the billing pattern.
    if any(name == truth and pattern.search(text) for name, pattern in _COMPILED):
        return "lexical_trap"
    return "unexplained"


def which_side_looks_right(text: str, truth: str, predicted: str) -> str:
    """A conservative heuristic for adjudicating a disagreement.

    Deliberately abstains ('unclear') unless the signal is strong, because guessing here would
    manufacture a statistic about a question only a human can settle.
    """
    lowered = text.lower()
    truth_hits = sum(
        1 for name, pattern in _COMPILED if name == truth and pattern.search(lowered)
    )
    predicted_hits = sum(
        1 for name, pattern in _COMPILED if name == predicted and pattern.search(lowered)
    )
    if predicted_hits and not truth_hits:
        return "model"
    if truth_hits and not predicted_hits:
        return "labeller"
    return "unclear"


def main() -> None:
    print("Loading splits (test pool NOT read) ...")
    train_texts, dev_texts = load_split("train"), load_split("dev")
    x_train, y_train, _ = build_weak_training_set(train_texts)
    x_dev, y_dev, _ = build_weak_training_set(dev_texts)

    model = TfidfLogisticClassifier(seed=SEED).fit(x_train, y_train)
    probabilities = model.predict_proba(x_dev)
    predicted = [str(model.classes[i]) for i in probabilities.argmax(axis=1)]
    confidences = probabilities.max(axis=1)

    shares = {k: v / len(y_train) for k, v in Counter(y_train).items()}
    disagreements = [
        (i, x_dev[i], y_dev[i], predicted[i], float(confidences[i]))
        for i in range(len(x_dev))
        if predicted[i] != y_dev[i]
    ]
    print(f"dev evaluable: {len(x_dev):,}   disagreements: {len(disagreements):,} "
          f"({len(disagreements)/len(x_dev):.1%})")

    buckets: dict[str, list] = {}
    adjudication = Counter()
    for index, text, truth, prediction, confidence in disagreements:
        category = categorise(text, truth, prediction, confidence, shares)
        side = which_side_looks_right(text, truth, prediction)
        adjudication[side] += 1
        buckets.setdefault(category, []).append(
            {
                "weak_label": truth,
                "model_prediction": prediction,
                "confidence": round(confidence, 4),
                "looks_right": side,
                "text": mask_pii(text).text[:200],
            }
        )

    print("\n=== DISAGREEMENT CATEGORIES ===")
    summary = {}
    for category, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        share = len(items) / len(disagreements)
        summary[category] = {
            "count": len(items),
            "share_of_disagreements": round(share, 4),
            "examples": items[:EXAMPLES_PER_CATEGORY],
        }
        print(f"  {category:<32} {len(items):>5} ({share:6.1%})")

    print("\n=== WHICH SIDE LOOKS RIGHT (conservative heuristic) ===")
    for side, count in adjudication.most_common():
        print(f"  {side:<12} {count:>5} ({count/len(disagreements):6.1%})")

    artifact = {
        "WARNING": (
            "These are DISAGREEMENTS between the model and a weak labeller, not errors. The "
            "labeller is a regex cascade and is frequently the wrong side. No conclusion about "
            "classifier accuracy can be drawn from them."
        ),
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/analyse_dev_errors.py",
            "git_sha": _git_sha(),
            "split": "dev",
            "test_pool_read": False,
            "seed": SEED,
            "taxonomy_version": TAXONOMY.version,
            "taxonomy_hash": TAXONOMY.frozen_hash,
            "dev_evaluable": len(x_dev),
            "disagreements": len(disagreements),
            "disagreement_rate": round(len(disagreements) / len(x_dev), 4),
        },
        "categories": summary,
        "adjudication": dict(adjudication),
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "classifier_dev_errors.json"
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

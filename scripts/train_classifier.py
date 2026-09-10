"""Fit intent models on the AppleSupport TRAIN split against frozen taxonomy v0.3.0.

**Training labels are WEAK SUPERVISION, not human annotation.** No hand-labelled data exists:
the golden set is drawn from the held-out test pool and has not been built. Labels come from
the deterministic labelling functions in `classifier/weak_labels.py`, and every artifact this
script writes repeats that warning.

The held-out test pool is never read here. Dev is used only by `evaluate_dev.py`.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --model tfidf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support.classifier.models import MajorityBaseline, TfidfLogisticClassifier  # noqa: E402
from hiver_support.classifier.weak_labels import WEAK_PROVENANCE, label_batch  # noqa: E402
from hiver_support.data.normalise import normalise_text  # noqa: E402
from hiver_support.data.split import temporal_split  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from discover_taxonomy import BRAND, load_brand_pairs  # noqa: E402

ARTIFACTS = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
SEED = 20260910


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def load_split(name: str) -> list[str]:
    """Return normalised customer messages for one split. Never returns the test pool."""
    if name == "test_pool":
        raise ValueError(
            "the held-out test pool must not be read during classifier development"
        )
    splits = temporal_split(load_brand_pairs(None))
    pairs = {"train": splits.train, "dev": splits.dev}[name]
    return [normalise_text(p.customer_text, brand=BRAND) for p in pairs]


def build_weak_training_set(texts: list[str]) -> tuple[list[str], list[str], dict]:
    """Apply labelling functions and keep only covered examples.

    Abstentions are dropped rather than assigned a default: training a model on
    ``other_unclear`` for every message the rules did not understand would teach it that
    "unrecognised" is a category, which it is not.
    """
    labels, stats = label_batch(texts)
    kept_texts, kept_labels = [], []
    for text, label in zip(texts, labels):
        if label.intent is not None:
            kept_texts.append(text)
            kept_labels.append(label.intent)
    return kept_texts, kept_labels, stats.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["majority", "tfidf", "all"], default="all")
    args = parser.parse_args()

    started = time.time()
    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")
    print("Loading TRAIN split (test pool is never read) ...")
    train_texts = load_split("train")
    print(f"  {len(train_texts):,} train messages")

    texts, labels, stats = build_weak_training_set(train_texts)
    print(f"\n  WEAK LABEL COVERAGE: {stats['coverage']:.1%} "
          f"({stats['covered']:,} of {stats['total']:,})")
    print(f"  conflict rate: {stats['conflict_rate']:.1%}   abstained: {stats['abstained']:,}")
    print(f"  security-flagged: {stats['security_count']:,}   "
          f"context-insufficient: {stats['context_insufficient_count']:,}")
    print("\n  weak label distribution:")
    for name, count in sorted(Counter(labels).items(), key=lambda kv: -kv[1]):
        print(f"    {name:<26} {count:>7,} ({count/len(labels)*100:5.2f}%)")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trained = {}
    for key in (["majority", "tfidf"] if args.model == "all" else [args.model]):
        model = MajorityBaseline() if key == "majority" else TfidfLogisticClassifier(seed=SEED)
        print(f"\nFitting {model.name} ...")
        model.fit(texts, labels)
        trained[key] = model
        print(f"  fitted {model.name} v{model.version}")

    # Models are re-fitted by evaluate_dev.py from the same seed rather than pickled: a pickle
    # is an arbitrary-code-execution vector for a reviewer, and test_data_provenance.py
    # forbids committing one.
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_classifier.py",
        "git_sha": _git_sha(),
        "brand": BRAND,
        "split": "train",
        "test_pool_read": False,
        "seed": SEED,
        "taxonomy_version": TAXONOMY.version,
        "taxonomy_hash": TAXONOMY.frozen_hash,
        "label_source": "WEAK SUPERVISION",
        "label_warning": WEAK_PROVENANCE,
        "train_messages": len(train_texts),
        "training_examples_after_coverage_filter": len(texts),
        "weak_label_stats": stats,
        "label_distribution": dict(sorted(Counter(labels).items())),
        "models_fitted": [m.name for m in trained.values()],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "classifier_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {REPORTS / 'classifier_training_manifest.json'}")


if __name__ == "__main__":
    main()

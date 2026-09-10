"""Evaluate classifiers on the AppleSupport DEV split. The test pool is never read.

**READ THIS BEFORE READING ANY NUMBER BELOW.**

Both the training labels and the dev labels come from the same deterministic labelling
functions (`classifier/weak_labels.py`). A model trained on those labels and scored against
those labels is being measured on **agreement with the heuristics, not on intent accuracy**.
Every figure this script produces is a *rule-recovery score*.

That has a specific consequence for model selection: the labelling functions are lexical, so
TF-IDF has a structural advantage. A semantic model that correctly generalises to a paraphrase
the regex misses is *penalised* for doing so. These numbers therefore cannot establish which
model classifies intent better — only which better imitates the labeller.

The honest statement of classifier quality remains unavailable until a human-labelled set
exists. `docs/CLASSIFIER.md` states this, and the artifact repeats it in its own header.

Usage:
    python scripts/evaluate_dev.py
    python scripts/evaluate_dev.py --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hiver_support.classifier.attributes import ContextDetector, SecurityDetector  # noqa: E402
from hiver_support.classifier.models import (  # noqa: E402
    EmbeddingClassifier,
    MajorityBaseline,
    TfidfLogisticClassifier,
)
from hiver_support.classifier.weak_labels import WEAK_PROVENANCE, label_batch  # noqa: E402
from hiver_support.taxonomy import TAXONOMY  # noqa: E402
from train_classifier import SEED, build_weak_training_set, load_split  # noqa: E402

REPORTS = ROOT / "reports"

HEADER_WARNING = (
    "RULE-RECOVERY SCORES, NOT ACCURACY. Training labels and dev labels come from the same "
    "deterministic labelling functions, so these figures measure agreement with those "
    "heuristics. The labelling functions are lexical, which gives TF-IDF a structural "
    "advantage over any semantic model. No statement about real classification quality can "
    "be made until a human-labelled set exists."
)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def intent_metrics(truth: list[str], predicted: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = sorted({i.name for i in TAXONOMY.intents})
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(truth, predicted, average="weighted", zero_division=0)), 4
        ),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def binary_metrics(truth: list[bool], predicted: list[bool]) -> dict:
    """Precision, recall and — for safety paths — the false-negative rate.

    FNR is reported explicitly because on a safety attribute a miss is the costly error, and
    it is the number most easily hidden behind a high accuracy on an imbalanced class.
    """
    truth_array, predicted_array = np.array(truth), np.array(predicted)
    tp = int((truth_array & predicted_array).sum())
    fp = int((~truth_array & predicted_array).sum())
    fn = int((truth_array & ~predicted_array).sum())
    tn = int((~truth_array & ~predicted_array).sum())
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "positive_rate": round(float(truth_array.mean()), 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "false_negative_rate": round(fn / (tp + fn), 4) if (tp + fn) else None,
    }


def calibration_report(probabilities: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict:
    """Reliability curve plus expected calibration error, fitted on DEV only.

    Thresholds are never chosen on the test pool. Selective-prediction coverage is reported
    across candidate thresholds so the accuracy/coverage trade is visible rather than asserted.
    """
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= 1.0)
        if not mask.any():
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(bin_accuracy - bin_confidence)
        curve.append(
            {
                "bin": f"[{low:.1f},{high:.1f})",
                "n": int(mask.sum()),
                "mean_confidence": round(bin_confidence, 4),
                "agreement": round(bin_accuracy, 4),
            }
        )

    selective = []
    for threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept = confidence >= threshold
        selective.append(
            {
                "threshold": threshold,
                "coverage": round(float(kept.mean()), 4),
                "agreement_on_kept": round(float(correct[kept].mean()), 4) if kept.any() else None,
            }
        )
    return {
        "expected_calibration_error": round(float(ece), 4),
        "reliability_curve": curve,
        "selective_prediction": selective,
        "note": "Fitted on DEV only. The test pool was never read.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    started = time.time()
    print(HEADER_WARNING + "\n")
    print(f"Frozen taxonomy {TAXONOMY.version} ({TAXONOMY.frozen_hash[:16]}...)")

    train_texts = load_split("train")
    dev_texts = load_split("dev")
    print(f"train={len(train_texts):,}  dev={len(dev_texts):,}  (test pool NOT read)")

    x_train, y_train, train_stats = build_weak_training_set(train_texts)
    x_dev, y_dev, dev_stats = build_weak_training_set(dev_texts)
    print(f"\nweak coverage: train {train_stats['coverage']:.1%}, dev {dev_stats['coverage']:.1%}")
    print(f"dev evaluable examples: {len(x_dev):,} of {len(dev_texts):,}")

    results: dict[str, dict] = {}
    models: list = [
        ("baseline_a_majority", MajorityBaseline()),
        ("baseline_b_tfidf", TfidfLogisticClassifier(seed=SEED)),
    ]
    if not args.skip_embeddings:
        models.append(("candidate_embeddings", EmbeddingClassifier(seed=SEED)))

    for key, model in models:
        print(f"\n--- {key} ({model.name}) ---")
        fit_start = time.time()
        model.fit(x_train, y_train)
        fit_seconds = time.time() - fit_start

        predict_start = time.time()
        predicted = [p.intent for p in model.predict_batch(x_dev)]
        predict_seconds = time.time() - predict_start

        metrics = intent_metrics(y_dev, predicted)
        entry = {
            "model": model.name,
            "version": model.version,
            "intent": metrics,
            "fit_seconds": round(fit_seconds, 2),
            "predict_seconds_for_dev": round(predict_seconds, 2),
            "predict_ms_per_message": round(predict_seconds / max(len(x_dev), 1) * 1000, 3),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_dev)
            correct = np.array(
                [p == t for p, t in zip(predicted, y_dev)]
            )
            entry["calibration"] = calibration_report(probabilities, correct)

        results[key] = entry
        print(f"  agreement (NOT accuracy): {metrics['accuracy']:.4f}  "
              f"macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  fit {fit_seconds:.1f}s   predict {entry['predict_ms_per_message']:.3f} ms/msg")

    # --- attribute detectors -------------------------------------------------------------
    print("\n--- attribute detectors (rule-based) ---")
    dev_labels, _ = label_batch(dev_texts)
    security_truth = [label.security_sensitive for label in dev_labels]
    context_truth = [label.context_sufficient for label in dev_labels]
    security_predicted = [SecurityDetector().detect(t).value for t in dev_texts]
    context_predicted = [ContextDetector().detect(t).value for t in dev_texts]

    attributes = {
        "security_sensitive": binary_metrics(security_truth, security_predicted),
        "context_sufficient": binary_metrics(context_truth, context_predicted),
        "caveat": (
            "The detector and the weak labeller share their patterns, so these are IDENTITY "
            "checks confirming the detector reproduces the labelling function - not evidence "
            "of detection quality. Real precision and recall require human labels. Reported "
            "so the tautology is visible rather than mistaken for validation."
        ),
    }
    print(f"  security positive rate on dev: {attributes['security_sensitive']['positive_rate']:.4f}")
    print(f"  context-insufficient on dev:   "
          f"{1 - attributes['context_sufficient']['positive_rate']:.4f}")

    artifact = {
        "WARNING": HEADER_WARNING,
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/evaluate_dev.py",
            "git_sha": _git_sha(),
            "split_evaluated": "dev",
            "test_pool_read": False,
            "seed": SEED,
            "taxonomy_version": TAXONOMY.version,
            "taxonomy_hash": TAXONOMY.frozen_hash,
            "label_source": "WEAK SUPERVISION",
            "label_warning": WEAK_PROVENANCE,
            "train_messages": len(train_texts),
            "dev_messages": len(dev_texts),
            "dev_evaluable": len(x_dev),
            "train_weak_stats": train_stats,
            "dev_weak_stats": dev_stats,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "models": results,
        "attributes": attributes,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "classifier_dev_results.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

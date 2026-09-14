"""Metrics for the golden-set evaluation.

Two conventions run through every function, because both silently flatter a system when
broken:

**Undefined is ``None``.** Precision with no predicted positives, a rate whose denominator is
empty, an F1 whose parts are undefined: none of these has a value. Reporting 0 understates,
reporting 1 overstates, and either one ends up as a headline.

**The false auto-handle rate is conditioned on examples that should have escalated.** It is
the safety number. Dividing by all traffic would dilute a dangerous router into a safe-looking
one, so it is never computed that way here.

Bootstrap intervals resample examples with replacement and skip resamples where the metric is
undefined, rather than counting them as zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _check(gold: Sequence, pred: Sequence) -> None:
    if len(gold) != len(pred):
        raise ValueError(f"gold and pred differ in length: {len(gold)} vs {len(pred)}")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def accuracy(gold: Sequence, pred: Sequence) -> float | None:
    _check(gold, pred)
    return _ratio(sum(g == p for g, p in zip(gold, pred)), len(gold))


def lenient_accuracy(
    gold: Sequence[str], alternatives: Sequence[str | None], pred: Sequence[str]
) -> float | None:
    """Credit a prediction matching either the primary label or the recorded alternative."""
    _check(gold, pred)
    hits = sum(p == g or (a is not None and p == a) for g, a, p in zip(gold, alternatives, pred))
    return _ratio(hits, len(gold))


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def intent_report(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """Accuracy, per-class precision/recall/F1, and macro-F1 over the labels present in gold.

    Macro-F1 averages over gold classes only. A class the annotator never used has no recall
    to measure, and including it would let a system be scored on an empty category.
    """
    _check(gold, pred)
    per_class = {}
    for label in sorted(set(gold)):
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        predicted = sum(p == label for p in pred)
        actual = sum(g == label for g in gold)
        precision = _ratio(tp, predicted)
        recall = _ratio(tp, actual)
        per_class[label] = {
            "support": actual,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    f1s = [c["f1"] if c["f1"] is not None else 0.0 for c in per_class.values()]
    return {
        "accuracy": accuracy(gold, pred),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per_class,
    }


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    return intent_report(gold, pred)["macro_f1"] if gold else None


def binary_metrics(gold: Sequence[bool], pred: Sequence[bool]) -> dict:
    """Confusion counts plus precision, recall and F1 for the positive class."""
    _check(gold, pred)
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def recall_of(gold: Sequence[bool], pred: Sequence[bool]) -> float | None:
    return binary_metrics(gold, pred)["recall"]


def routing_metrics(gold_escalate: Sequence[bool], pred_escalate: Sequence[bool]) -> dict:
    """Routing quality against the human's independent should_escalate judgement."""
    _check(gold_escalate, pred_escalate)
    should = [p for g, p in zip(gold_escalate, pred_escalate) if g]
    could_automate = [p for g, p in zip(gold_escalate, pred_escalate) if not g]
    unsafe = sum(1 for p in should if not p)
    base = binary_metrics(gold_escalate, pred_escalate)
    return {
        "escalation_precision": base["precision"],
        "escalation_recall": base["recall"],
        "false_auto_handle_rate": _ratio(unsafe, len(should)),
        "false_escalation_rate": _ratio(sum(1 for p in could_automate if p), len(could_automate)),
        "auto_handle_rate": _ratio(sum(1 for p in pred_escalate if not p), len(pred_escalate)),
        "unsafe_auto_handles": unsafe,
        "gold_should_escalate": len(should),
    }


def false_auto_handle_rate(gold: Sequence[bool], pred: Sequence[bool]) -> float | None:
    return routing_metrics(gold, pred)["false_auto_handle_rate"]


def expected_cost(gold_escalate: Sequence[bool], pred_escalate: Sequence[bool], ratio: float) -> float | None:
    """Mean cost per message: an unsafe auto-handle costs ``ratio``, any escalation costs 1.

    ``ratio`` is C_bad / C_human, unknowable from tweets, so it is swept rather than asserted.
    """
    _check(gold_escalate, pred_escalate)
    costs = [
        1.0 if p else (float(ratio) if g else 0.0)
        for g, p in zip(gold_escalate, pred_escalate)
    ]
    return float(np.mean(costs)) if costs else None


def mean_of(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def bootstrap_ci(
    metric: Callable,
    *arrays: Sequence,
    n_resamples: int = 2000,
    seed: int = 20260911,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval, resampling examples with replacement.

    Resamples on which the metric is undefined are skipped. If none are defined the interval
    is ``None`` - an interval invented from no data would be the fabrication this avoids.
    """
    size = len(arrays[0])
    if size == 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, size, size)
        value = metric(*[[array[i] for i in index] for array in arrays])
        if value is not None:
            values.append(value)
    if not values:
        return None
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(low), 4), round(float(high), 4)


def _nan_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float] | None:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(low), 4), round(float(high), 4)


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.maximum(denominator, 1), np.nan)


def risk_coverage_curve(
    gold_escalate: Sequence[bool],
    system_auto: Sequence[bool],
    scores: Sequence[float | None],
    intent_correct: Sequence[bool],
    thresholds: Sequence[float],
    ratios: Sequence[float] = (2, 4, 8, 12, 20),
    n_boot: int = 1000,
    seed: int = 20260911,
) -> list[dict]:
    """The automation/safety tradeoff across confidence thresholds. Selects nothing.

    At threshold ``t`` an example is auto-handled only if the system auto-handled it AND its
    score is at least ``t``. At ``t <= 0`` the score is ignored, so the first row reproduces
    the system's own operating point exactly. A missing score fails closed above zero.

    Definitions, all per the human ``should_escalate`` label:

    * ``auto_handle_rate`` - share of all messages auto-handled; ``escalation_rate`` = 1 - it
    * ``coverage`` - share of automatable messages (gold says no escalation) auto-handled
    * ``false_auto_handle_rate`` - share of should-escalate messages auto-handled
    * ``unsafe_auto_handle_rate`` - should-escalate-but-auto-handled, over ALL messages
    * ``selective_risk`` - share of auto-handled messages that should have escalated
    * ``intent_accuracy_on_auto_handled`` - intent accuracy on the messages actually answered
    * ``cost_at_ratio`` - mean cost per message: unsafe auto-handle = ratio, escalation = 1

    No row is marked best. Choosing an operating point on the gold set would convert this
    measurement into tuning; the curve exists so the point can be chosen from the cost of an
    unsafe response, on data that is not the evaluation set.
    """
    gold = np.asarray(gold_escalate, dtype=bool)
    base = np.asarray(system_auto, dtype=bool)
    correct = np.asarray(intent_correct, dtype=bool)
    score = np.asarray([np.nan if s is None else float(s) for s in scores], dtype=float)
    if not (len(gold) == len(base) == len(score) == len(correct)):
        raise ValueError("gold, system_auto, scores and intent_correct differ in length")
    n = len(gold)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, (n_boot, n)) if n else np.zeros((0, 0), dtype=int)

    rows = []
    for t in thresholds:
        passes = np.ones(n, dtype=bool) if t <= 0 else np.nan_to_num(score, nan=-np.inf) >= t
        auto = base & passes
        unsafe = auto & gold
        n_auto, n_gold, n_safe = int(auto.sum()), int(gold.sum()), int((~gold).sum())

        g_b, a_b = gold[index], auto[index]
        fahr_b = _safe_div((a_b & g_b).sum(1).astype(float), g_b.sum(1).astype(float))
        cov_b = _safe_div((a_b & ~g_b).sum(1).astype(float), (~g_b).sum(1).astype(float))

        rows.append({
            "threshold": round(float(t), 4),
            "n": n,
            "auto_handled": n_auto,
            "unsafe_auto_handles": int(unsafe.sum()),
            "auto_handle_rate": n_auto / n if n else None,
            "escalation_rate": 1 - n_auto / n if n else None,
            "coverage": _ratio(int((auto & ~gold).sum()), n_safe),
            "false_auto_handle_rate": _ratio(int(unsafe.sum()), n_gold),
            "unsafe_auto_handle_rate": _ratio(int(unsafe.sum()), n),
            "selective_risk": _ratio(int(unsafe.sum()), n_auto),
            "intent_accuracy_on_auto_handled": _ratio(int((auto & correct).sum()), n_auto),
            "cost_at_ratio": {
                str(r): float(np.mean(np.where(auto, np.where(gold, float(r), 0.0), 1.0)))
                if n else None
                for r in ratios
            },
            "false_auto_handle_rate_ci95": _nan_ci(fahr_b),
            "coverage_ci95": _nan_ci(cov_b),
        })
    return rows

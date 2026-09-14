"""Metric functions for the golden-set evaluation.

Every number in the final report comes through these functions, so they are tested against
hand-computed values rather than against their own output. Two conventions are pinned because
getting them wrong silently flatters a system:

* **An undefined metric is ``None``, never 0 or 1.** Precision with no predicted positives has
  no value; reporting 0 understates, reporting 1 overstates, and either becomes a headline.
* **The false auto-handle rate is conditioned on examples that should have escalated.** It is
  the safety number, and diluting it by all traffic would make an unsafe router look safe.
"""

from __future__ import annotations

import pytest

from hiver_support.evaluation.metrics import (
    accuracy,
    binary_metrics,
    bootstrap_ci,
    expected_cost,
    intent_report,
    lenient_accuracy,
    routing_metrics,
)


class TestAccuracyAndIntent:
    def test_accuracy_counts_exact_matches(self):
        assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "x"]) == 0.5

    def test_accuracy_of_nothing_is_undefined(self):
        assert accuracy([], []) is None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            accuracy(["a"], ["a", "b"])

    def test_macro_f1_averages_per_class_f1_over_gold_labels(self):
        gold = ["a", "a", "b", "b"]
        pred = ["a", "b", "b", "b"]
        report = intent_report(gold, pred)
        # a: P=1/1, R=1/2, F1=2/3 ; b: P=2/3, R=2/2, F1=0.8
        assert report["per_class"]["a"]["f1"] == pytest.approx(2 / 3)
        assert report["per_class"]["b"]["f1"] == pytest.approx(0.8)
        assert report["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
        assert report["accuracy"] == 0.75

    def test_a_never_predicted_class_has_undefined_precision_not_zero(self):
        report = intent_report(["a", "b"], ["a", "a"])
        assert report["per_class"]["b"]["precision"] is None
        assert report["per_class"]["b"]["recall"] == 0.0

    def test_lenient_accuracy_credits_the_recorded_alternative(self):
        gold = ["a", "b"]
        alternatives = [None, "c"]
        assert lenient_accuracy(gold, alternatives, ["a", "c"]) == 1.0
        assert lenient_accuracy(gold, alternatives, ["x", "c"]) == 0.5


class TestBinaryMetrics:
    def test_counts_and_rates(self):
        m = binary_metrics([True, True, False, False], [True, False, True, False])
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
        assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5

    def test_no_predicted_positives_leaves_precision_undefined(self):
        m = binary_metrics([True, False], [False, False])
        assert m["precision"] is None
        assert m["recall"] == 0.0
        assert m["f1"] is None

    def test_no_gold_positives_leaves_recall_undefined(self):
        m = binary_metrics([False, False], [True, False])
        assert m["recall"] is None


class TestRouting:
    def test_false_auto_handle_rate_is_over_examples_that_should_escalate(self):
        gold = [True, True, True, True, False, False]
        pred = [True, False, True, True, False, False]
        m = routing_metrics(gold, pred)
        assert m["false_auto_handle_rate"] == pytest.approx(1 / 4)
        assert m["unsafe_auto_handles"] == 1

    def test_false_escalation_rate_is_over_examples_that_could_be_automated(self):
        m = routing_metrics([False, False, True], [True, False, True])
        assert m["false_escalation_rate"] == pytest.approx(1 / 2)

    def test_always_escalate_has_zero_unsafe_auto_handles(self):
        # The strong safety baseline must be presented as strong, not strawmanned.
        m = routing_metrics([True, False, False, True], [True, True, True, True])
        assert m["unsafe_auto_handles"] == 0
        assert m["false_auto_handle_rate"] == 0.0
        assert m["auto_handle_rate"] == 0.0

    def test_rates_are_undefined_when_their_denominator_is_empty(self):
        m = routing_metrics([False, False], [False, True])
        assert m["false_auto_handle_rate"] is None


class TestExpectedCost:
    def test_a_bad_auto_handle_costs_the_ratio_and_an_escalation_costs_one(self):
        gold = [True, False, False]
        pred = [False, True, False]
        # item 1: unsafe auto -> 4 ; item 2: escalate -> 1 ; item 3: correct auto -> 0
        assert expected_cost(gold, pred, ratio=4) == pytest.approx(5 / 3)

    def test_always_escalate_costs_exactly_one_per_message(self):
        assert expected_cost([True, False], [True, True], ratio=20) == 1.0


class TestBootstrap:
    def test_it_is_deterministic_for_a_fixed_seed(self):
        gold = ["a", "b", "a", "b", "a"] * 8
        pred = ["a", "b", "b", "b", "a"] * 8
        first = bootstrap_ci(accuracy, gold, pred, seed=7)
        assert first == bootstrap_ci(accuracy, gold, pred, seed=7)

    def test_the_interval_brackets_the_point_estimate(self):
        gold = ["a", "b", "a", "b", "a"] * 8
        pred = ["a", "b", "b", "b", "a"] * 8
        low, high = bootstrap_ci(accuracy, gold, pred, seed=7)
        assert low <= accuracy(gold, pred) <= high

    def test_a_perfect_system_has_a_degenerate_interval(self):
        assert bootstrap_ci(accuracy, ["a", "b"] * 10, ["a", "b"] * 10, seed=1) == (1.0, 1.0)

    def test_resamples_where_the_metric_is_undefined_are_skipped(self):
        # A metric that is undefined on every resample yields no interval, not a fake one.
        assert bootstrap_ci(lambda g, p: None, [1, 2], [1, 2], seed=1) is None


class TestRiskCoverageCurve:
    """The routing tradeoff, measured across thresholds rather than tuned to one.

    The curve raises a confidence bar on top of each system's own decisions: an example is
    auto-handled at threshold t only if the system auto-handled it AND its score clears t.
    Nothing here selects an operating point - choosing one on the gold set would turn the
    evaluation into tuning.
    """

    GOLD = [True, True, False, False]
    AUTO = [True, True, True, False]
    SCORES = [0.9, 0.3, 0.8, 0.5]
    CORRECT = [True, False, True, True]

    def _curve(self, thresholds=(0.0, 0.5, 1.0), **kw):
        from hiver_support.evaluation.metrics import risk_coverage_curve

        return risk_coverage_curve(
            self.GOLD, self.AUTO, self.SCORES, self.CORRECT, thresholds, ratios=(4,), n_boot=50, **kw
        )

    def test_threshold_zero_is_the_systems_own_operating_point(self):
        row = self._curve()[0]
        assert row["auto_handle_rate"] == 0.75
        assert row["unsafe_auto_handles"] == 2
        assert row["false_auto_handle_rate"] == 1.0
        assert row["coverage"] == 0.5
        assert row["unsafe_auto_handle_rate"] == 0.5
        assert row["selective_risk"] == pytest.approx(2 / 3)
        assert row["cost_at_ratio"]["4"] == pytest.approx(2.25)

    def test_raising_the_threshold_trades_coverage_for_safety(self):
        row = self._curve()[1]
        assert row["auto_handle_rate"] == 0.5
        assert row["false_auto_handle_rate"] == 0.5
        assert row["coverage"] == 0.5
        assert row["selective_risk"] == 0.5
        assert row["intent_accuracy_on_auto_handled"] == 1.0
        assert row["cost_at_ratio"]["4"] == pytest.approx(1.5)

    def test_a_threshold_nothing_clears_is_always_escalate(self):
        row = self._curve()[2]
        assert row["auto_handle_rate"] == 0.0 and row["escalation_rate"] == 1.0
        assert row["false_auto_handle_rate"] == 0.0 and row["coverage"] == 0.0
        assert row["selective_risk"] is None
        assert row["intent_accuracy_on_auto_handled"] is None
        assert row["cost_at_ratio"]["4"] == 1.0

    def test_a_missing_score_fails_closed_above_zero_but_not_at_zero(self):
        from hiver_support.evaluation.metrics import risk_coverage_curve

        rows = risk_coverage_curve([False], [True], [None], [True], (0.0, 0.1), ratios=(4,), n_boot=10)
        assert rows[0]["auto_handle_rate"] == 1.0
        assert rows[1]["auto_handle_rate"] == 0.0

    def test_automation_never_increases_as_the_threshold_rises(self):
        import random

        from hiver_support.evaluation.metrics import risk_coverage_curve

        rng = random.Random(3)
        n = 60
        gold = [rng.random() < 0.4 for _ in range(n)]
        auto = [rng.random() < 0.7 for _ in range(n)]
        scores = [rng.random() for _ in range(n)]
        rows = risk_coverage_curve(
            gold, auto, scores, [True] * n, [i / 20 for i in range(21)], ratios=(8,), n_boot=20
        )
        rates = [r["auto_handle_rate"] for r in rows]
        assert rates == sorted(rates, reverse=True)
        for r in rows:
            assert r["auto_handle_rate"] + r["escalation_rate"] == pytest.approx(1.0)

    def test_no_operating_point_is_selected(self):
        for row in self._curve():
            assert not {"best", "optimal", "recommended", "selected"} & set(row)

    def test_intervals_are_deterministic_and_ordered(self):
        first = self._curve(seed=5)
        assert first == self._curve(seed=5)
        low, high = first[0]["false_auto_handle_rate_ci95"]
        assert low <= high

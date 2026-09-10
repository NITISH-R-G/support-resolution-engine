"""Operational-handling profiles — the evidence base for merge/split decisions.

A taxonomy label earns its existence by changing what support *does*, not by having distinct
vocabulary (`SPEC.md` §5). The brand's own historical replies are the record of what support
did, so comparing reply behaviour between two candidate groups is the empirical test of
whether a distinction is operational or merely lexical.

Two guards matter and are tested here:

* **Rates must never be read as exact.** Small groups produce noisy rates, so every comparison
  carries a bootstrap confidence interval and a significance flag. A difference whose CI spans
  zero must not be reported as a difference.
* **Determinism.** The same inputs and seed must produce the same interval, or a merge decision
  becomes unreproducible.

Note on scope: replies inform taxonomy *design* here, from the train split. They must never be
used to assign labels during annotation — that is forbidden in `docs/ANNOTATION_GUIDE.md`,
because the reply is downstream of the intent and using it would contaminate the gold labels.
"""

from __future__ import annotations

import pytest

from hiver_support.analysis.handling import (
    HandlingProfile,
    compare_handling,
    handling_profile,
)

DEFLECTIONS = [
    "Please DM us and we can help you there.",
    "Send us a DM with more details so we can look into this.",
    "We've received your DM and will respond there shortly.",
    "Please contact us here: [URL] so our team can connect.",
] * 10  # >= MIN_GROUP_FOR_VERDICT (30); smaller groups correctly yield no verdict

INSTRUCTIONS = [
    "Go to Settings > General > Software Update and tap Download and Install.",
    "Try restarting the device by holding the side button, then check again.",
    "Open Settings, tap Wi-Fi, then select your network and enter the password.",
    "Please reset network settings: Settings > General > Reset > Reset Network Settings.",
] * 10


class TestProfileBasics:
    def test_profile_reports_group_size(self):
        assert handling_profile("x", DEFLECTIONS).n == len(DEFLECTIONS)

    def test_all_rates_are_fractions(self):
        profile = handling_profile("x", DEFLECTIONS + INSTRUCTIONS)
        for field in (
            "deflection_rate",
            "actionable_rate",
            "substantive_rate",
            "asks_question_rate",
        ):
            assert 0.0 <= getattr(profile, field) <= 1.0

    def test_deflection_group_scores_high_on_deflection(self):
        assert handling_profile("x", DEFLECTIONS).deflection_rate > 0.8

    def test_instruction_group_scores_high_on_actionable(self):
        assert handling_profile("x", INSTRUCTIONS).actionable_rate > 0.8

    def test_instruction_group_is_not_mostly_deflection(self):
        assert handling_profile("x", INSTRUCTIONS).deflection_rate < 0.2

    def test_empty_group_is_reported_not_crashed(self):
        profile = handling_profile("x", [])
        assert profile.n == 0
        assert profile.deflection_rate == 0.0

    def test_profile_is_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            handling_profile("x", DEFLECTIONS).n = 99  # type: ignore[misc]

    def test_profile_is_deterministic(self):
        assert handling_profile("x", DEFLECTIONS) == handling_profile("x", DEFLECTIONS)


class TestComparison:
    def test_identical_groups_show_no_significant_difference(self):
        """The most important guard: same behaviour must never look like a difference."""
        results = compare_handling(DEFLECTIONS, list(DEFLECTIONS), seed=1, iterations=300)
        for difference in results:
            assert difference.significant is False, f"{difference.metric} falsely significant"

    def test_clearly_different_groups_show_a_significant_difference(self):
        results = compare_handling(DEFLECTIONS, INSTRUCTIONS, seed=1, iterations=300)
        by_metric = {d.metric: d for d in results}
        assert by_metric["deflection_rate"].significant is True
        assert by_metric["actionable_rate"].significant is True

    def test_confidence_interval_brackets_the_point_difference(self):
        for difference in compare_handling(DEFLECTIONS, INSTRUCTIONS, seed=1, iterations=300):
            assert difference.ci_low <= difference.difference <= difference.ci_high

    def test_comparison_is_reproducible_under_the_same_seed(self):
        first = compare_handling(DEFLECTIONS, INSTRUCTIONS, seed=7, iterations=300)
        second = compare_handling(DEFLECTIONS, INSTRUCTIONS, seed=7, iterations=300)
        assert [d.ci_low for d in first] == [d.ci_low for d in second]
        assert [d.ci_high for d in first] == [d.ci_high for d in second]

    def test_reversing_arguments_flips_the_sign_of_the_difference(self):
        forward = {d.metric: d for d in compare_handling(DEFLECTIONS, INSTRUCTIONS, seed=3)}
        reverse = {d.metric: d for d in compare_handling(INSTRUCTIONS, DEFLECTIONS, seed=3)}
        for metric, difference in forward.items():
            assert difference.difference == pytest.approx(-reverse[metric].difference, abs=1e-9)

    def test_tiny_groups_are_not_declared_significant(self):
        """With too little data, an apparent gap must yield no verdict at all.

        Bootstrap degenerates here: resampling two identical values reproduces them every
        time, giving a CI of [1, 1] and manufacturing certainty from nothing.
        """
        results = compare_handling(DEFLECTIONS[:2], INSTRUCTIONS[:2], seed=1, iterations=300)
        assert all(d.significant is False for d in results)

    def test_groups_just_below_the_verdict_threshold_yield_no_verdict(self):
        from hiver_support.analysis.handling import MIN_GROUP_FOR_VERDICT

        n = MIN_GROUP_FOR_VERDICT - 1
        results = compare_handling(DEFLECTIONS[:n], INSTRUCTIONS[:n], seed=1, iterations=300)
        assert all(d.significant is False for d in results)

    def test_empty_group_comparison_is_never_significant(self):
        results = compare_handling([], INSTRUCTIONS, seed=1, iterations=300)
        assert all(d.significant is False for d in results)

    def test_every_reported_metric_is_named(self):
        metrics = {d.metric for d in compare_handling(DEFLECTIONS, INSTRUCTIONS, seed=1)}
        assert {"deflection_rate", "actionable_rate", "substantive_rate"} <= metrics


class TestHonestyOfReporting:
    def test_profile_exposes_n_so_rates_are_never_quoted_alone(self):
        assert "n" in HandlingProfile.__dataclass_fields__

    def test_difference_exposes_an_interval_not_just_a_point(self):
        from hiver_support.analysis.handling import ProfileDifference

        fields = ProfileDifference.__dataclass_fields__
        assert "ci_low" in fields and "ci_high" in fields

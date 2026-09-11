"""Evidence relevance: a separate question from groundedness, and an unvalidated one.

Milestone 5 produced a reply that was faithful to its evidence and useless: the customer said
their problem began with iOS 11.0.2 and the agent advised updating to 11.0.2. Grounding passed
correctly — nothing was invented. The evidence was simply wrong for the question.

These tests pin two things:

* the two questions stay **separate**, and this module never claims evidence *is* relevant;
* the checks generalise beyond the 11.0.2 case that exposed the problem, which is why every
  version test below uses a different product and a different version scheme.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.agent.relevance import (
    RelevanceSignal,
    assess_relevance,
)
from hiver_support.agent.retrieval import EvidenceCase

WHEN = datetime(2017, 11, 1, tzinfo=timezone.utc)


def evidence(*resolutions: str) -> tuple[EvidenceCase, ...]:
    return tuple(
        EvidenceCase(
            case_id=f"case{i}",
            customer_text="something is wrong",
            resolution_text=text,
            created_at=WHEN - timedelta(days=i + 1),
        )
        for i, text in enumerate(resolutions)
    )


class TestTheContractIsHonestAboutWhatItKnows:
    def test_it_never_claims_evidence_is_relevant(self):
        report = assess_relevance("my wifi drops", evidence("Try resetting network settings."))
        assert not hasattr(report, "relevant")
        assert report.contradicted is False
        assert RelevanceSignal.UNKNOWN in report.signals

    def test_absence_of_contradiction_is_reported_as_unknown_not_as_relevant(self):
        report = assess_relevance("my screen flickers", evidence("Try restarting."))
        assert report.signals == (RelevanceSignal.UNKNOWN,)

    def test_the_report_declares_itself_unvalidated(self):
        report = assess_relevance("my wifi drops", evidence("Try restarting."))
        assert report.validated is False
        assert "NOT validated" in report.to_dict()["note"]

    def test_signals_are_a_closed_set(self):
        assert {s.value for s in RelevanceSignal} == {
            "ok",
            "version_contradiction",
            "already_attempted",
            "unknown",
        }

    def test_no_evidence_is_unknown_rather_than_contradicted(self):
        report = assess_relevance("my wifi drops", ())
        assert report.contradicted is False
        assert RelevanceSignal.UNKNOWN in report.signals

    def test_it_rejects_a_non_string_message(self):
        with pytest.raises(TypeError):
            assess_relevance(None, evidence("Try restarting."))


class TestVersionContradiction:
    """Written against the general shape, not the iOS 11.0.2 case that exposed it."""

    def test_the_original_measured_failure_is_caught(self):
        report = assess_relevance(
            "the autocorrect bug started with iOS 11.0.2, when is it fixed",
            evidence("We have released 11.0.2. Make sure you back up and get updated."),
        )
        assert RelevanceSignal.VERSION_CONTRADICTION in report.signals
        assert report.contradicted is True

    @pytest.mark.parametrize(
        "message,resolution",
        [
            ("everything broke since 4.2.1", "Please update to 4.2.1 for a fix."),
            ("my laptop has been slow after 2.14", "You should upgrade to version 2.14."),
            ("crashing ever since the update to 9.0.3", "Install 9.0.3 and let us know."),
        ],
    )
    def test_other_products_and_version_schemes_are_caught_too(self, message, resolution):
        assert RelevanceSignal.VERSION_CONTRADICTION in assess_relevance(
            message, evidence(resolution)
        ).signals

    def test_a_different_recommended_version_is_not_a_contradiction(self):
        # Advising a customer forward off a broken release is exactly right.
        report = assess_relevance(
            "everything broke since 11.0.2",
            evidence("We have released 11.0.3, please update to 11.0.3."),
        )
        assert RelevanceSignal.VERSION_CONTRADICTION not in report.signals

    def test_merely_mentioning_a_version_is_not_blaming_it(self):
        report = assess_relevance(
            "how do I get iOS 11.0.2 on my phone",
            evidence("You can update to 11.0.2 from Settings."),
        )
        assert report.contradicted is False


class TestAlreadyAttempted:
    def test_evidence_offering_only_what_the_customer_already_tried_is_contradicted(self):
        report = assess_relevance(
            "I already restarted the phone and it still drops wifi",
            evidence("Please try restarting your device."),
        )
        assert RelevanceSignal.ALREADY_ATTEMPTED in report.signals

    def test_inflected_reports_are_recognised(self):
        report = assess_relevance(
            "I have tried resetting the network settings twice now",
            evidence("Reset network settings in Settings."),
        )
        assert RelevanceSignal.ALREADY_ATTEMPTED in report.signals

    def test_evidence_offering_something_new_is_not_contradicted(self):
        # The customer restarted; the evidence suggests something else as well, so it still
        # has something to offer.
        report = assess_relevance(
            "I already restarted and it still drops",
            evidence("Try restarting, and if that fails reinstall the app."),
        )
        assert RelevanceSignal.ALREADY_ATTEMPTED not in report.signals

    def test_a_customer_who_tried_nothing_triggers_nothing(self):
        report = assess_relevance(
            "my wifi keeps dropping", evidence("Please try restarting your device.")
        )
        assert report.contradicted is False


class TestRelevanceIsSeparateFromGroundedness:
    def test_a_contradicted_reply_can_still_be_perfectly_grounded(self):
        from hiver_support.agent.grounding import validate_grounding

        cases = evidence("We have released 11.0.2, please update to 11.0.2.")
        reply = "We have released 11.0.2, please update to 11.0.2."
        assert validate_grounding(reply, cases).passed is True
        assert assess_relevance("broken since 11.0.2", cases).contradicted is True

    def test_the_two_modules_share_no_decision(self):
        import inspect

        from hiver_support.agent import relevance

        source = inspect.getsource(relevance)
        assert "validate_grounding" not in source

    def test_evidence_naming_the_broken_version_and_a_fix_recommends_only_the_fix(self):
        # "the bug came in 11.0.2, update to 11.0.3" names both versions and means one.
        # Explicit targets take precedence, so this must not read as a contradiction.
        report = assess_relevance(
            "everything broke since 11.0.2",
            evidence("The issue was introduced in 11.0.2. Please update to 11.0.3."),
        )
        assert RelevanceSignal.VERSION_CONTRADICTION not in report.signals

    def test_a_named_version_with_a_bare_update_instruction_counts_as_recommended(self):
        # The measured shape: target in one sentence, instruction in the next.
        report = assess_relevance(
            "broken since 3.1.4",
            evidence("We have released 3.1.4. Make sure you back up and get updated."),
        )
        assert RelevanceSignal.VERSION_CONTRADICTION in report.signals

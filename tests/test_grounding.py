"""Grounding validation — the check that stops the agent asserting things it cannot support.

This is the most safety-critical deterministic component in the system. The agent has **no
ability to act**: it cannot issue refunds, cancel orders, reset accounts or dispatch repairs.
Any reply claiming it has done so is false by construction, regardless of how plausible the
sentence looks, so those claims are fatal rather than merely suspicious.

The validator runs *before* an LLM is consulted and can fail a reply on its own. That ordering
matters: a deterministic check that never needs a model is cheaper, faster and more auditable
than asking a model whether another model hallucinated.

Two failure directions are tested. Missing a fabrication lets the agent lie to a customer.
Rejecting a well-grounded reply pushes safe traffic to humans and makes the system useless, so
paraphrase of retrieved evidence must pass.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hiver_support.agent.grounding import GroundingReport, ViolationType, validate_grounding
from hiver_support.agent.retrieval import EvidenceCase

WHEN = datetime(2017, 1, 1, tzinfo=timezone.utc)

EVIDENCE = (
    EvidenceCase(
        case_id="1",
        customer_text="my battery drains fast",
        resolution_text=(
            "Please go to Settings > Battery > Battery Health and check Maximum Capacity. "
            "If it is below 80% the battery may need servicing."
        ),
        created_at=WHEN,
    ),
    EvidenceCase(
        case_id="2",
        customer_text="wifi keeps dropping",
        resolution_text=(
            "Try resetting network settings: Settings > General > Reset > Reset Network Settings."
        ),
        created_at=WHEN,
    ),
)


class TestGroundedRepliesPass:
    @pytest.mark.parametrize(
        "reply",
        [
            "Please go to Settings > Battery > Battery Health and check Maximum Capacity.",
            "You can check Maximum Capacity under Settings > Battery > Battery Health.",
            "Try resetting network settings via Settings > General > Reset.",
            "If Maximum Capacity is below 80% the battery may need servicing.",
        ],
    )
    def test_paraphrase_of_evidence_passes(self, reply):
        assert validate_grounding(reply, EVIDENCE).passed is True

    def test_a_passing_report_lists_no_violations(self):
        report = validate_grounding("Check Battery Health in Settings.", EVIDENCE)
        assert report.violations == ()


class TestFabricatedActionsAreFatal:
    """The agent cannot act, so any claim that it has is false by construction."""

    @pytest.mark.parametrize(
        "reply",
        [
            "I've issued a refund to your account.",
            "We have reset your password for you.",
            "I've cancelled your order.",
            "Your replacement has been dispatched.",
            "I have escalated your case to our engineers.",
            "We've applied a credit to your account.",
        ],
    )
    def test_claiming_an_action_fails(self, reply):
        report = validate_grounding(reply, EVIDENCE)
        assert report.passed is False
        assert ViolationType.FABRICATED_ACTION in {v.type for v in report.violations}

    def test_action_claim_fails_even_when_evidence_is_strong(self):
        assert validate_grounding(
            "Go to Settings > Battery > Battery Health. I've also refunded you.", EVIDENCE
        ).passed is False


class TestInventedSpecifics:
    def test_a_number_absent_from_evidence_fails(self):
        report = validate_grounding("Your battery will last 47 hours.", EVIDENCE)
        assert report.passed is False
        assert ViolationType.UNSUPPORTED_SPECIFIC in {v.type for v in report.violations}

    def test_a_number_present_in_evidence_passes(self):
        assert validate_grounding("If it is below 80% it may need servicing.", EVIDENCE).passed

    def test_an_invented_timeline_fails(self):
        assert validate_grounding("This will be fixed within 3 business days.", EVIDENCE).passed is False

    def test_an_invented_money_amount_fails(self):
        assert validate_grounding("You will be refunded $49.99.", EVIDENCE).passed is False


class TestInventedPolicy:
    @pytest.mark.parametrize(
        "reply",
        [
            "Our policy guarantees a replacement within 30 days.",
            "Apple always replaces batteries free of charge.",
            "You are entitled to a full refund under our terms.",
        ],
    )
    def test_policy_assertions_absent_from_evidence_fail(self, reply):
        assert validate_grounding(reply, EVIDENCE).passed is False


class TestEmptyEvidence:
    def test_any_substantive_reply_fails_without_evidence(self):
        """With nothing retrieved there is nothing to ground in, so nothing may be asserted."""
        report = validate_grounding("Go to Settings > Battery.", ())
        assert report.passed is False
        assert ViolationType.NO_EVIDENCE in {v.type for v in report.violations}

    def test_empty_reply_with_empty_evidence_still_fails(self):
        assert validate_grounding("", ()).passed is False


class TestReportContract:
    def test_report_is_immutable(self):
        report = validate_grounding("Check Battery Health.", EVIDENCE)
        with pytest.raises((AttributeError, TypeError)):
            report.passed = False  # type: ignore[misc]

    def test_violations_name_the_offending_text(self):
        report = validate_grounding("I've issued a refund.", EVIDENCE)
        assert any(v.detail for v in report.violations)

    def test_report_is_serialisable(self):
        import json

        json.dumps(validate_grounding("I've issued a refund.", EVIDENCE).to_dict())

    def test_is_deterministic(self):
        first = validate_grounding("Your battery lasts 47 hours.", EVIDENCE)
        second = validate_grounding("Your battery lasts 47 hours.", EVIDENCE)
        assert first == second


class TestEdgeCases:
    def test_non_string_reply_raises(self):
        with pytest.raises(TypeError):
            validate_grounding(None, EVIDENCE)

    def test_whitespace_reply_fails(self):
        assert validate_grounding("   ", EVIDENCE).passed is False

    def test_very_long_reply_does_not_crash(self):
        assert isinstance(validate_grounding("Settings. " * 2000, EVIDENCE), GroundingReport)

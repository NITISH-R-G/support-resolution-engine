"""Post-generation policy validation, independent of whatever wrote the reply.

Grounding asks *"is this derived from the evidence?"*. Policy asks *"is this allowed at all?"*
The two are genuinely different, and Milestone 6 measured why: gpt-oss-120b produced

    "We'd love to help with the update issues you're seeing. Please DM us with your iPhone
     model and iOS version so we can dive in."

That reply is **grounded** — it invents nothing — and it is **automated deflection**, the one
outcome the whole retrieval corpus was filtered to prevent. The corpus filter stops us
*retrieving* a deflection; it cannot stop a model *composing* one. Only a check on the output
catches that.

Every check here runs on text the generator cannot influence, and no model is asked to judge
its own reply.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.agent.policy import (
    PolicyViolationType,
    validate_policy,
)
from hiver_support.agent.retrieval import EvidenceCase

WHEN = datetime(2017, 11, 1, tzinfo=timezone.utc)


def evidence(*resolutions: str) -> tuple[EvidenceCase, ...]:
    texts = resolutions or (
        "Please restart your device and check Settings > General > About for the version.",
    )
    return tuple(
        EvidenceCase(
            case_id=f"case{i}",
            customer_text="my device is misbehaving",
            resolution_text=text,
            created_at=WHEN - timedelta(days=i + 1),
        )
        for i, text in enumerate(texts)
    )


def kinds(report) -> set[str]:
    return {v.violation.value for v in report.violations}


class TestTheContract:
    def test_a_clean_grounded_reply_passes(self):
        report = validate_policy(
            "Please restart your device and check Settings > General > About.", evidence()
        )
        assert report.passed is True
        assert report.violations == ()

    def test_violation_types_are_a_closed_set(self):
        assert {v.value for v in PolicyViolationType} == {
            "generated_deflection",
            "unsupported_refund",
            "unsupported_account_action",
            "action_already_taken",
            "unsupported_url",
            "unsupported_promise",
            "security_advice",
            "unsupported_instruction",
        }

    def test_a_report_serialises_with_its_reasons(self):
        payload = validate_policy("I have issued your refund.", evidence()).to_dict()
        assert payload["passed"] is False
        assert payload["violations"][0]["violation"]
        assert payload["violations"][0]["detail"]

    def test_an_empty_reply_is_not_a_policy_matter(self):
        # Emptiness is the generator declining; the agent escalates on that path already.
        assert validate_policy("", evidence()).passed is True

    def test_it_rejects_a_non_string_reply(self):
        with pytest.raises(TypeError):
            validate_policy(None, evidence())

    def test_the_validator_takes_no_model_and_no_generator(self):
        # Independence is structural: there is nothing to pass a model into.
        import inspect

        params = set(inspect.signature(validate_policy).parameters)
        assert not params & {"model", "provider", "generator", "llm"}


class TestGeneratedDeflection:
    """The Milestone 6 failure, in the words the model actually used."""

    def test_the_exact_measured_failure_is_caught(self):
        reply = (
            "We'd love to help with the update issues you're seeing. Please DM us with your "
            "iPhone model and iOS version so we can dive in."
        )
        assert PolicyViolationType.GENERATED_DEFLECTION.value in kinds(
            validate_policy(reply, evidence())
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "Please send us a DM with your details.",
            "Could you direct message us so we can look into it?",
            "Reach out to us here and we will help.",
            "Send us a message with your serial number.",
            "Contact us directly and we can investigate.",
        ],
    )
    def test_deflection_phrasings_are_caught(self, reply):
        assert PolicyViolationType.GENERATED_DEFLECTION.value in kinds(
            validate_policy(reply, evidence())
        )

    def test_a_deflection_is_caught_even_when_the_evidence_deflected_too(self):
        # Grounding would pass this: the reply faithfully reproduces the evidence. Automating
        # a deflection is still the thing the corpus filter exists to prevent.
        deflecting = evidence("Please DM us and we will take a look.")
        assert validate_policy("Please DM us and we will take a look.", deflecting).passed is False

    def test_an_ordinary_invitation_to_reply_is_not_a_deflection(self):
        assert validate_policy(
            "Please restart the device and let us know how it goes.", evidence()
        ).passed is True


class TestUnsupportedActions:
    @pytest.mark.parametrize(
        "reply",
        [
            "I have issued a refund to your account.",
            "We've refunded the 79 dollars for you.",
            "Your refund has been processed.",
        ],
    )
    def test_refund_claims_are_fatal(self, reply):
        report = validate_policy(reply, evidence())
        assert report.passed is False
        assert report.fatal is True

    @pytest.mark.parametrize(
        "reply",
        [
            "We have reset your password for you.",
            "I've unlocked your account.",
            "Your subscription has been cancelled as requested.",
            "We have already dispatched a replacement device.",
        ],
    )
    def test_account_and_order_action_claims_are_fatal(self, reply):
        assert validate_policy(reply, evidence()).passed is False

    def test_advising_an_action_is_allowed(self):
        # The agent may advise; it may not act. That distinction is the whole rule.
        supported = evidence("You can reset your password at Settings > Password & Security.")
        assert validate_policy(
            "You can reset your password in Settings > Password & Security.", supported
        ).passed is True

    def test_a_completed_action_claim_is_caught_even_when_evidence_mentions_it(self):
        supported = evidence("You can request a refund from your purchase history.")
        assert validate_policy("We have refunded you.", supported).passed is False


class TestUnsupportedUrlsAndPromises:
    def test_a_url_absent_from_the_evidence_is_a_violation(self):
        report = validate_policy(
            "See https://support.apple.com/fix-it for the steps.", evidence()
        )
        assert PolicyViolationType.UNSUPPORTED_URL.value in kinds(report)

    def test_a_placeholder_url_is_allowed_when_the_evidence_carried_one(self):
        # The pipeline masks real links to [URL]; reusing the placeholder invents nothing.
        supported = evidence("Take a look at this article: [URL]")
        assert validate_policy("Take a look at this article: [URL]", supported).passed is True

    def test_a_placeholder_url_with_no_link_in_evidence_is_a_violation(self):
        report = validate_policy("Take a look at this article: [URL]", evidence())
        assert PolicyViolationType.UNSUPPORTED_URL.value in kinds(report)

    @pytest.mark.parametrize(
        "reply",
        [
            "You will receive a replacement within 24 hours.",
            "This is guaranteed to fix the problem.",
            "Our engineers will call you back today.",
            "We promise this will be resolved by tomorrow.",
        ],
    )
    def test_operational_promises_are_violations(self, reply):
        assert PolicyViolationType.UNSUPPORTED_PROMISE.value in kinds(
            validate_policy(reply, evidence())
        )


class TestSecuritySensitiveAdvice:
    def test_any_reply_at_all_is_a_violation_when_the_message_was_security_sensitive(self):
        # The deterministic gate should have escalated before generation. If a reply exists,
        # something upstream failed, and the reply must not go out regardless of its content.
        report = validate_policy(
            "Please restart your device.", evidence(), security_sensitive=True
        )
        assert PolicyViolationType.SECURITY_ADVICE.value in kinds(report)
        assert report.fatal is True

    def test_the_same_reply_is_fine_when_the_message_was_not_security_sensitive(self):
        assert validate_policy(
            "Please restart your device.", evidence(), security_sensitive=False
        ).passed is True

    def test_security_advice_is_flagged_when_evidence_does_not_support_it(self):
        report = validate_policy(
            "Change your Apple ID password immediately and enable two-factor authentication.",
            evidence(),
        )
        assert PolicyViolationType.SECURITY_ADVICE.value in kinds(report)


class TestUnsupportedInstructions:
    def test_a_navigation_path_absent_from_the_evidence_is_flagged(self):
        report = validate_policy(
            "Go to Settings > Privacy > Analytics and turn it off.", evidence()
        )
        assert PolicyViolationType.UNSUPPORTED_INSTRUCTION.value in kinds(report)

    def test_a_navigation_path_present_in_the_evidence_is_allowed(self):
        supported = evidence("Check Settings > General > About for your version.")
        assert validate_policy(
            "Check Settings > General > About for your version.", supported
        ).passed is True

    def test_instructions_are_not_flagged_when_there_is_no_evidence_to_compare_against(self):
        # With nothing retrieved the agent escalates earlier; flagging here would double-count
        # a failure the evidence gate already owns.
        assert validate_policy("Go to Settings > Privacy.", ()).passed is True


class TestFatality:
    def test_every_defined_violation_is_fatal_for_now(self):
        # Nothing is advisory yet. The flag exists so a future non-fatal advisory can be added
        # without every caller having to re-derive severity.
        report = validate_policy("I have refunded you. Please DM us.", evidence())
        assert report.fatal is True
        assert len(report.violations) >= 2

    def test_a_passing_report_is_not_fatal(self):
        assert validate_policy("Please restart your device.", evidence()).fatal is False

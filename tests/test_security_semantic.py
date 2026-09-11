"""Semantic security detection, tested by category rather than by phrase.

Milestone 6 measured the lexical detector failing on a real model run: *"someone is logged
into my Apple ID from another country and I think they changed my recovery email"* was
AUTO_HANDLED, because it describes a takeover without using the word "hacked". The gate the
whole fail-closed design rests on was a keyword list.

**Nothing here tests the probe sentence.** Fitting a detector to the sentence that exposed it
would produce a detector that passes one test and fails the next paraphrase — the benchmark
gaming this project audits other submissions for. Every case below is an independent way of
*describing* the same situation, and the detector has to generalise or fail visibly.

The hard requirement is discrimination, not sensitivity. Flagging everything would pass every
positive case here and make the agent useless, so the benign cases carry equal weight — in
particular "I forgot my password", which is an ordinary account-access request that shares
most of its vocabulary with a credential compromise.
"""

from __future__ import annotations

import pytest

from hiver_support.classifier.attributes import SecurityDetector
from hiver_support.classifier.security import (
    CompositeSecurityDetector,
    SecurityAssessment,
    SecurityReason,
    SemanticSecurityDetector,
)

# --------------------------------------------------------------------------- category corpora
# Written as independent descriptions of each situation. None is a probe sentence.

ACCOUNT_TAKEOVER = [
    "somebody else has control of my apple id and I cannot get back in",
    "a stranger is using my account and I never gave them permission",
    "my account is being used by someone who is not me right now",
    "there is another person signed into my id and changing things",
]
SUSPICIOUS_LOGIN = [
    "I got an alert about a sign in from a country I have never visited",
    "there is a login on my account from a device I do not recognise",
    "my account shows activity from a place I have never been to",
    "an unfamiliar computer signed into my id last night",
]
UNAUTHORIZED_CHANGE = [
    "my recovery email was changed and it was not me who changed it",
    "someone reset my password without my permission",
    "my security questions were altered by somebody else",
    "the phone number on my account was swapped without my knowledge",
]
CREDENTIAL_COMPROMISE = [
    "I think my password has been exposed in a breach",
    "somebody found out my password and I need help",
    "my login details have been leaked somewhere",
]
DEVICE_LOST_OR_STOLEN = [
    "my iphone was taken from my bag yesterday",
    "someone stole my ipad and my data is on it",
    "my phone was snatched and I am worried about what is on it",
]
FRAUDULENT_ACTIVITY = [
    "there are purchases on my account that I never made",
    "I am seeing charges for things I did not buy",
    "somebody is spending money through my account",
]
PHISHING_SCAM = [
    "I got a message pretending to be from you asking for my details",
    "someone is trying to trick me into handing over my login",
    "is this email asking for my password actually from apple",
]

SECURITY_CATEGORIES = {
    "account_takeover": ACCOUNT_TAKEOVER,
    "suspicious_login": SUSPICIOUS_LOGIN,
    "unauthorized_change": UNAUTHORIZED_CHANGE,
    "credential_compromise": CREDENTIAL_COMPROMISE,
    "device_lost_or_stolen": DEVICE_LOST_OR_STOLEN,
    "fraudulent_activity": FRAUDULENT_ACTIVITY,
    "phishing_scam": PHISHING_SCAM,
}

# Ordinary support traffic. Flagging these would make the agent useless, so they are the
# other half of the requirement.
BENIGN = [
    "I forgot my password and need to reset it",
    "how do I change my password to something stronger",
    "my wifi keeps dropping every ten minutes at home",
    "the battery on my iphone drains in about two hours",
    "how do I update to the latest ios version",
    "my screen is cracked, how much is a repair",
    "I want a refund for my icloud storage subscription",
    "the speaker stopped working after the update",
    "how do I transfer my photos to a new phone",
    "I cannot remember my passcode to unlock the device",
]

ALL_SECURITY = [text for texts in SECURITY_CATEGORIES.values() for text in texts]


@pytest.fixture(scope="module")
def semantic() -> SemanticSecurityDetector:
    return SemanticSecurityDetector()


@pytest.fixture(scope="module")
def composite() -> CompositeSecurityDetector:
    return CompositeSecurityDetector()


class TestTheContract:
    def test_reasons_are_a_closed_set(self):
        assert {r.value for r in SecurityReason} == {
            "none",
            "account_takeover",
            "suspicious_login",
            "unauthorized_change",
            "credential_compromise",
            "device_lost_or_stolen",
            "fraudulent_activity",
            "phishing_scam",
            "uncertain",
        }

    def test_an_assessment_carries_a_verdict_a_confidence_and_a_reason(self, semantic):
        result = semantic.assess("somebody else is inside my account")
        assert isinstance(result, SecurityAssessment)
        assert isinstance(result.security_sensitive, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reason, SecurityReason)

    def test_a_safe_verdict_reports_the_none_reason(self, semantic):
        assert semantic.assess("how do I update the software on my phone").reason is SecurityReason.NONE

    def test_it_rejects_a_non_string(self, semantic):
        with pytest.raises(TypeError):
            semantic.assess(None)

    def test_it_never_sees_or_accepts_an_intent(self):
        # Security is orthogonal to intent. Measured on train data: 306 of 340
        # security-sensitive messages (90%) were not account messages, so a detector that
        # consulted the intent would miss most of the traffic that matters.
        import inspect

        signature = inspect.signature(SemanticSecurityDetector.assess)
        assert "intent" not in signature.parameters
        source = inspect.getsource(SemanticSecurityDetector)
        assert "intent" not in source.lower()


class TestFailClosed:
    def test_an_empty_message_is_not_flagged(self, semantic):
        # Nothing to describe a threat with; the context gate handles emptiness instead.
        assert semantic.assess("").security_sensitive is False

    def test_an_encoder_failure_fails_closed_rather_than_passing_the_message(self):
        class BrokenEncoder:
            def encode(self, *_args, **_kwargs):
                raise RuntimeError("model unavailable")

        detector = SemanticSecurityDetector(encoder=BrokenEncoder())
        result = detector.assess("somebody else has control of my account")
        assert result.security_sensitive is True
        assert result.reason is SecurityReason.UNCERTAIN

    def test_an_ambiguous_score_fails_closed(self, semantic):
        # The band between "clearly safe" and "clearly a threat" escalates. A false flag
        # costs one human review; a miss can mean auto-answering a live compromise.
        assert semantic.thresholds.uncertain_floor < semantic.thresholds.sensitive_floor

    def test_the_uncertain_band_is_flagged_as_sensitive(self, semantic):
        midband = (
            semantic.thresholds.uncertain_floor + semantic.thresholds.sensitive_floor
        ) / 2
        sensitive, reason = semantic.verdict_for_score(midband, margin=0.02)
        assert sensitive is True
        assert reason is SecurityReason.UNCERTAIN

    def test_a_message_no_closer_to_a_threat_than_to_ordinary_traffic_is_not_flagged(
        self, semantic
    ):
        # Zero margin means the message resembles a routine request exactly as much as a
        # threat, which is not a signal. Escalating on it would flag every mention of a
        # password and the gate would be switched off within a week.
        midband = (
            semantic.thresholds.uncertain_floor + semantic.thresholds.sensitive_floor
        ) / 2
        assert semantic.verdict_for_score(midband, margin=0.0)[0] is False
        assert semantic.verdict_for_score(midband, margin=-0.1)[0] is False


class TestTheCompositeIsAUnionAndNeverWeakens:
    def test_the_lexical_detector_is_retained(self, composite):
        assert isinstance(composite.lexical, SecurityDetector)

    def test_anything_the_lexical_detector_catches_is_still_caught(self, composite):
        lexical = SecurityDetector()
        for text in ("my account was hacked", "unauthorized access to my apple id",
                     "my card was used fraudulently", "someone else is using my apple id"):
            assert lexical.detect(text).value is True, "fixture no longer exercises the rule"
            assert composite.detect(text).value is True

    def test_the_semantic_path_can_add_a_detection_the_rule_misses(self, composite):
        # The Milestone 6 failure class, in different words.
        text = "a person in another country is signed into my id and swapped my recovery email"
        assert SecurityDetector().detect(text).value is False
        assert composite.detect(text).value is True

    def test_the_composite_can_only_add_detections_never_remove_them(self, composite):
        lexical = SecurityDetector()
        for text in ALL_SECURITY + BENIGN:
            if lexical.detect(text).value:
                assert composite.detect(text).value is True

    def test_evidence_names_which_path_fired(self, composite):
        evidence = composite.detect("my account was hacked").evidence
        assert any(e.startswith("lexical:") for e in evidence)

    def test_the_composite_result_is_shaped_like_the_rule_it_replaces(self, composite):
        # The pipeline consumes .value and .evidence, so the composite drops in unchanged.
        result = composite.detect("how do I update the software on my phone")
        assert hasattr(result, "value") and hasattr(result, "evidence")


class TestSemanticCategoryRecall:
    """Measured per category. A category that fails is reported, not tuned away."""

    @pytest.mark.parametrize(
        "category,texts",
        list(SECURITY_CATEGORIES.items()),
        ids=list(SECURITY_CATEGORIES),
    )
    def test_each_security_category_is_detected(self, composite, category, texts):
        missed = [t for t in texts if not composite.detect(t).value]
        assert missed == [], f"{category}: undetected -> {missed}"

    def test_overall_recall_on_security_descriptions(self, composite):
        caught = sum(1 for t in ALL_SECURITY if composite.detect(t).value)
        assert caught == len(ALL_SECURITY), (
            f"caught {caught}/{len(ALL_SECURITY)} security descriptions"
        )


class TestBenignTrafficIsNotSwamped:
    """Flagging everything would pass every positive test and make the agent useless."""

    @pytest.mark.parametrize("text", BENIGN, ids=lambda t: t[:40])
    def test_ordinary_support_traffic_is_not_flagged(self, composite, text):
        assert composite.detect(text).value is False, f"false positive on {text!r}"

    def test_a_forgotten_password_is_not_a_compromise(self, composite):
        # The discrimination that matters most: it shares almost all of its vocabulary with
        # credential compromise but needs no human.
        assert composite.detect("I forgot my password and need to reset it").value is False

    def test_false_positive_rate_on_benign_traffic_is_zero(self, composite):
        flagged = [t for t in BENIGN if composite.detect(t).value]
        assert flagged == [], f"benign messages flagged: {flagged}"

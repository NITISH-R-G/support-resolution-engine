"""The two attribute detectors and the composed classifier.

`security_sensitive` and `context_sufficient` are **independent** of the intent, as frozen
taxonomy v0.3.0 requires. Inferring safety from the intent is the specific failure the freeze
was designed to prevent: 306 of 340 security-sensitive messages in the train split (90%) were
not account messages, so an intent-derived flag would miss nine tenths of them.

Both detectors are deterministic rules rather than trained models, and that is a deliberate
engineering decision recorded in `docs/CLASSIFIER.md`. No human labels exist, so the only
training signal available *is* the rule; a model trained to imitate it could only lose recall
on the safety path while adding the appearance of learning. On a path where a miss means
auto-answering a real account takeover, a rule with known behaviour beats a model with unknown
recall.

The composition tests are the safety-critical ones: **no intent confidence, however high, may
override an attribute.**
"""

from __future__ import annotations

import pytest

from hiver_support.classifier.attributes import ContextDetector, SecurityDetector
from hiver_support.classifier.models import MajorityBaseline
from hiver_support.classifier.pipeline import IntentClassifier
from hiver_support.taxonomy import TAXONOMY

SECURITY_ACROSS_INTENTS = [
    ("account", "i think my apple id has been hacked"),
    ("account", "someone else is signed into my account"),
    ("billing", "there is an unauthorised charge on my card"),
    ("billing", "someone made a fraudulent purchase with my apple pay"),
    ("device", "my iphone was stolen yesterday"),
    ("apps", "there is a scam app on the app store pretending to be my bank"),
    ("phishing", "is this email from apple or is it a phishing attempt"),
    ("phishing", "i got a fake apple text asking for my password"),
]

BENIGN = [
    "my battery drains so fast since the update",
    "how do i turn off notifications",
    "wifi keeps disconnecting at home",
    "i was charged twice for icloud storage",
    "i forgot my apple id password",
]


class TestSecurityDetector:
    @pytest.mark.parametrize(("topic", "text"), SECURITY_ACROSS_INTENTS)
    def test_detects_security_across_every_topic(self, topic, text):
        assert SecurityDetector().detect(text).value is True, topic

    @pytest.mark.parametrize("text", BENIGN)
    def test_does_not_fire_on_benign_messages(self, text):
        assert SecurityDetector().detect(text).value is False

    def test_a_plain_password_reset_is_not_security_sensitive(self):
        """The distinction the freeze rests on: access failure is not compromise."""
        assert SecurityDetector().detect("i forgot my password").value is False

    def test_reports_the_matched_evidence(self):
        result = SecurityDetector().detect("i think my account was hacked")
        assert result.evidence

    def test_is_independent_of_any_intent_input(self):
        """The detector's signature takes text only; intent cannot leak in."""
        import inspect

        parameters = inspect.signature(SecurityDetector().detect).parameters
        assert "intent" not in parameters

    def test_is_deterministic(self):
        detector = SecurityDetector()
        assert detector.detect("my account was hacked") == detector.detect("my account was hacked")

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            SecurityDetector().detect(None)


class TestContextDetector:
    @pytest.mark.parametrize(
        "text", ["Ok", "thanks", "11.0.3", "iPhone 7", "yes", "help", "?", ""]
    )
    def test_bare_messages_are_insufficient(self, text):
        assert ContextDetector().detect(text).value is False

    @pytest.mark.parametrize(
        "text",
        [
            "my battery drains so fast since yesterday",
            "how do i back up my photos to icloud",
        ],
    )
    def test_substantive_messages_are_sufficient(self, text):
        assert ContextDetector().detect(text).value is True

    def test_reports_why_it_judged_insufficient(self):
        assert ContextDetector().detect("Ok").evidence

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            ContextDetector().detect(None)


class TestComposedClassifier:
    @staticmethod
    def _classifier() -> IntentClassifier:
        intent_model = MajorityBaseline().fit(
            ["my battery drains", "battery dies fast"], ["battery_charging", "battery_charging"]
        )
        return IntentClassifier(intent_model=intent_model)

    def test_returns_a_contract_valid_prediction(self):
        prediction = self._classifier().predict("my battery drains so fast today")
        assert prediction.intent in {i.name for i in TAXONOMY.intents}

    def test_attributes_are_populated_from_the_detectors(self):
        prediction = self._classifier().predict("i think my account was hacked badly")
        assert prediction.security_sensitive is True

    def test_security_forces_escalation_despite_a_safe_confident_intent(self):
        """The safety-critical invariant: confidence never overrides the attribute."""
        prediction = self._classifier().predict("my battery was hacked by someone else")
        assert prediction.security_sensitive is True
        assert prediction.confidence == pytest.approx(1.0)
        assert prediction.must_escalate is True

    def test_insufficient_context_forces_escalation(self):
        prediction = self._classifier().predict("Ok")
        assert prediction.context_sufficient is False
        assert prediction.must_escalate is True

    def test_insufficient_context_does_not_assert_a_confident_intent(self):
        """A content-free message must not inherit the majority label at full confidence."""
        prediction = self._classifier().predict("iPhone 7")
        assert prediction.context_sufficient is False
        assert prediction.abstained is True

    def test_benign_message_does_not_escalate(self):
        assert self._classifier().predict("my battery drains so fast").must_escalate is False

    def test_every_intent_escalates_when_security_is_flagged(self):
        classifier = self._classifier()
        for intent in TAXONOMY.intents:
            assert TAXONOMY.must_escalate(intent.name, security_sensitive=True) is True

    def test_batch_matches_single_prediction(self):
        classifier = self._classifier()
        texts = ["my battery drains so fast", "i think i was hacked badly today"]
        assert [p.intent for p in classifier.predict_batch(texts)] == [
            classifier.predict(t).intent for t in texts
        ]

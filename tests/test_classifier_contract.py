"""The classifier output contract.

Everything downstream — routing, retrieval, generation, the harness — consumes this object,
so it must be impossible to construct an invalid one. The contract fails **closed**: an
out-of-taxonomy label, a confidence outside [0,1] or a non-boolean attribute raises rather
than being coerced to something plausible. A classifier that silently emits a wrong-but-valid
label is far more dangerous than one that crashes.

The taxonomy hash is carried in every prediction. Taxonomy v0.3.0 is frozen, and a prediction
made against a different taxonomy is not comparable to one made against this one; binding the
hash makes that drift detectable instead of silent.
"""

from __future__ import annotations

import math

import pytest

from hiver_support.classifier.contract import (
    ContractError,
    Prediction,
    PredictionSource,
)
from hiver_support.taxonomy import TAXONOMY

VALID = dict(
    intent="battery_charging",
    confidence=0.82,
    security_sensitive=False,
    context_sufficient=True,
    model_name="tfidf_logreg",
    model_version="1.0.0",
    taxonomy_version=TAXONOMY.version,
    taxonomy_hash=TAXONOMY.frozen_hash,
    source=PredictionSource.MODEL,
    evidence=("battery", "draining"),
)


def make(**overrides) -> Prediction:
    return Prediction(**{**VALID, **overrides})


class TestValidPrediction:
    def test_constructs_with_valid_fields(self):
        assert make().intent == "battery_charging"

    def test_is_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            make().intent = "connectivity"  # type: ignore[misc]

    def test_carries_model_and_taxonomy_metadata(self):
        prediction = make()
        assert prediction.model_name and prediction.model_version
        assert prediction.taxonomy_hash == TAXONOMY.frozen_hash

    def test_evidence_is_available_for_inspection(self):
        assert make().evidence == ("battery", "draining")


class TestIntentValidity:
    def test_rejects_an_intent_outside_the_frozen_taxonomy(self):
        with pytest.raises(ContractError, match="intent"):
            make(intent="software_update_issue")

    def test_rejects_an_empty_intent(self):
        with pytest.raises(ContractError):
            make(intent="")

    @pytest.mark.parametrize("name", [i.name for i in TAXONOMY.intents])
    def test_accepts_every_frozen_intent(self, name):
        assert make(intent=name).intent == name

    def test_there_is_no_default_intent_to_fall_back_to(self):
        """A missing intent must fail, never silently become other_unclear."""
        with pytest.raises(TypeError):
            Prediction(**{k: v for k, v in VALID.items() if k != "intent"})


class TestConfidenceValidity:
    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0, -5.0])
    def test_rejects_confidence_outside_the_unit_interval(self, value):
        with pytest.raises(ContractError, match="confidence"):
            make(confidence=value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_boundary_and_interior_values(self, value):
        assert make(confidence=value).confidence == value

    def test_rejects_nan(self):
        with pytest.raises(ContractError):
            make(confidence=math.nan)

    def test_rejects_infinity(self):
        with pytest.raises(ContractError):
            make(confidence=math.inf)

    def test_rejects_non_numeric_confidence(self):
        with pytest.raises(ContractError):
            make(confidence="high")  # type: ignore[arg-type]


class TestAttributeValidity:
    @pytest.mark.parametrize("field", ["security_sensitive", "context_sufficient"])
    def test_rejects_non_boolean_attributes(self, field):
        for bad in ("true", 1, 0, None):
            with pytest.raises(ContractError, match=field):
                make(**{field: bad})

    @pytest.mark.parametrize("field", ["security_sensitive", "context_sufficient"])
    def test_accepts_real_booleans(self, field):
        for good in (True, False):
            assert getattr(make(**{field: good}), field) is good


class TestTaxonomyBinding:
    def test_rejects_a_mismatched_taxonomy_hash(self):
        """Predictions made against a different taxonomy are not comparable."""
        with pytest.raises(ContractError, match="taxonomy"):
            make(taxonomy_hash="0" * 64)

    def test_rejects_a_mismatched_taxonomy_version(self):
        with pytest.raises(ContractError, match="taxonomy"):
            make(taxonomy_version="0.1.0")


class TestEscalationWiring:
    """Attributes dominate the intent. Confidence may never override safety."""

    def test_security_sensitive_forces_escalation_at_any_confidence(self):
        for confidence in (0.0, 0.5, 0.99, 1.0):
            prediction = make(security_sensitive=True, confidence=confidence)
            assert prediction.must_escalate is True

    def test_security_sensitive_forces_escalation_for_every_intent(self):
        for intent in TAXONOMY.intents:
            assert make(intent=intent.name, security_sensitive=True).must_escalate is True

    def test_insufficient_context_forces_escalation_at_any_confidence(self):
        for confidence in (0.0, 0.99, 1.0):
            assert make(context_sufficient=False, confidence=confidence).must_escalate is True

    def test_escalation_sensitive_intent_escalates_even_when_clean(self):
        assert make(intent="billing_and_subscription").must_escalate is True

    def test_safe_intent_with_clean_attributes_does_not_force_escalation(self):
        assert make(intent="battery_charging").must_escalate is False


class TestAbstention:
    def test_abstention_is_recorded_explicitly(self):
        prediction = make(source=PredictionSource.ABSTAINED, intent="other_unclear")
        assert prediction.abstained is True

    def test_a_model_prediction_is_not_an_abstention(self):
        assert make().abstained is False

    def test_abstention_must_escalate(self):
        prediction = make(
            source=PredictionSource.ABSTAINED, intent="other_unclear", confidence=0.1
        )
        assert prediction.must_escalate is True


class TestSerialisation:
    def test_round_trips_through_dict(self):
        prediction = make()
        assert Prediction.from_dict(prediction.to_dict()) == prediction

    def test_serialises_to_json(self):
        import json

        json.dumps(make().to_dict())

    def test_serialised_form_carries_provenance(self):
        payload = make().to_dict()
        for key in ("model_name", "model_version", "taxonomy_version", "taxonomy_hash"):
            assert payload[key]

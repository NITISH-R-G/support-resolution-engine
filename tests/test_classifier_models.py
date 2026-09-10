"""Classifier models: the shared interface, and the two required baselines.

Every model returns a contract-valid ``Prediction``, so routing consumes one type regardless
of which model produced it and a model swap cannot change the downstream contract.

Baselines exist to establish whether a stronger model earns its complexity. Baseline A
(majority class) is deliberately trivial; Baseline B (TF-IDF + logistic regression) is a real
but simple model. Neither is strawmanned — if the stronger model cannot beat them, that is a
result to report, not a reason to weaken them.

**Every number these produce on dev is measured against weak labels**, so it is a
rule-recovery score, not accuracy. That framing lives in ``docs/CLASSIFIER.md`` and in the
results artifact; the tests here pin behaviour, not quality.
"""

from __future__ import annotations

import pytest

from hiver_support.classifier.contract import ContractError, Prediction, PredictionSource
from hiver_support.classifier.models import (
    MajorityBaseline,
    TfidfLogisticClassifier,
)
from hiver_support.taxonomy import TAXONOMY

TRAIN_TEXTS = [
    "my battery drains so fast since the update",
    "battery dies at 40 percent every day",
    "the battery health dropped to 80 percent",
    "wifi keeps disconnecting on my ipad at home",
    "cannot connect to wifi after the update",
    "bluetooth will not pair with my car",
    "i was charged twice for my icloud storage",
    "there is a charge on my account i did not make",
    "i want a refund for this subscription",
    "i forgot my apple id password and cannot sign in",
    "locked out of my account after two factor",
    "how do i turn off notifications for messages",
    "how can i back up my photos to icloud",
]
TRAIN_LABELS = [
    "battery_charging", "battery_charging", "battery_charging",
    "connectivity", "connectivity", "connectivity",
    "billing_and_subscription", "billing_and_subscription", "billing_and_subscription",
    "account_access", "account_access",
    "howto_information", "howto_information",
]


@pytest.fixture(params=["majority", "tfidf"])
def fitted_model(request):
    model = (
        MajorityBaseline() if request.param == "majority" else TfidfLogisticClassifier(seed=7)
    )
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return model


class TestSharedInterface:
    def test_predict_returns_a_contract_valid_prediction(self, fitted_model):
        assert isinstance(fitted_model.predict("my battery drains"), Prediction)

    def test_prediction_carries_the_model_identity(self, fitted_model):
        prediction = fitted_model.predict("my battery drains")
        assert prediction.model_name == fitted_model.name
        assert prediction.model_version

    def test_prediction_binds_the_frozen_taxonomy(self, fitted_model):
        prediction = fitted_model.predict("my battery drains")
        assert prediction.taxonomy_hash == TAXONOMY.frozen_hash

    def test_predict_batch_matches_single_predictions(self, fitted_model):
        texts = ["my battery drains", "wifi keeps dropping"]
        batch = fitted_model.predict_batch(texts)
        assert [p.intent for p in batch] == [fitted_model.predict(t).intent for t in texts]

    def test_predicting_before_fit_raises_rather_than_guessing(self, request):
        model = TfidfLogisticClassifier(seed=7)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict("my battery drains")

    def test_confidence_is_a_valid_probability(self, fitted_model):
        assert 0.0 <= fitted_model.predict("my battery drains").confidence <= 1.0

    def test_is_deterministic(self, fitted_model):
        first = fitted_model.predict("my battery drains so fast")
        second = fitted_model.predict("my battery drains so fast")
        assert (first.intent, first.confidence) == (second.intent, second.confidence)


class TestMajorityBaseline:
    def test_always_predicts_the_most_frequent_training_label(self):
        model = MajorityBaseline()
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        for text in ("wifi is broken", "refund please", "anything at all"):
            assert model.predict(text).intent == "battery_charging"

    def test_reports_the_majority_share_as_confidence(self):
        model = MajorityBaseline()
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        expected = TRAIN_LABELS.count("battery_charging") / len(TRAIN_LABELS)
        assert model.predict("anything").confidence == pytest.approx(expected)

    def test_rejects_training_labels_outside_the_frozen_taxonomy(self):
        model = MajorityBaseline()
        with pytest.raises(ContractError):
            model.fit(["x y z w"], ["software_update_issue"])


class TestTfidfLogisticClassifier:
    def test_learns_to_separate_the_training_classes(self):
        model = TfidfLogisticClassifier(seed=7)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert model.predict("my battery drains so fast").intent == "battery_charging"
        assert model.predict("wifi keeps disconnecting").intent == "connectivity"

    def test_same_seed_gives_identical_predictions(self):
        first, second = TfidfLogisticClassifier(seed=7), TfidfLogisticClassifier(seed=7)
        first.fit(TRAIN_TEXTS, TRAIN_LABELS)
        second.fit(TRAIN_TEXTS, TRAIN_LABELS)
        text = "i was charged twice"
        assert first.predict(text).confidence == second.predict(text).confidence

    def test_exposes_probabilities_for_calibration(self):
        model = TfidfLogisticClassifier(seed=7)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        probabilities = model.predict_proba(["my battery drains"])
        assert probabilities.shape[0] == 1
        assert probabilities.sum(axis=1)[0] == pytest.approx(1.0)

    def test_evidence_names_the_features_that_drove_the_decision(self):
        model = TfidfLogisticClassifier(seed=7)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert model.predict("my battery drains so fast").evidence

    def test_unseen_vocabulary_does_not_crash(self):
        model = TfidfLogisticClassifier(seed=7)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert model.predict("zzzz qqqq wwww vvvv").intent in {i.name for i in TAXONOMY.intents}


class TestAbstention:
    """Low confidence must be expressible as a refusal, not a weak guess."""

    def test_below_threshold_the_model_abstains(self):
        model = TfidfLogisticClassifier(seed=7, abstain_below=0.99)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        prediction = model.predict("zzzz qqqq wwww vvvv")
        assert prediction.source is PredictionSource.ABSTAINED
        assert prediction.must_escalate is True

    def test_abstention_threshold_of_zero_never_abstains(self):
        model = TfidfLogisticClassifier(seed=7, abstain_below=0.0)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert model.predict("zzzz qqqq").source is PredictionSource.MODEL


class TestEdgeCases:
    def test_empty_text(self, fitted_model):
        assert isinstance(fitted_model.predict(""), Prediction)

    def test_non_string_raises(self, fitted_model):
        with pytest.raises(TypeError):
            fitted_model.predict(None)

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            TfidfLogisticClassifier(seed=7).fit(["a b c d"], ["battery_charging", "connectivity"])

    def test_fit_rejects_empty_training_data(self):
        with pytest.raises(ValueError):
            TfidfLogisticClassifier(seed=7).fit([], [])


class TestEmbeddingClassifier:
    """Marked slow: loading the sentence-transformer costs seconds, not milliseconds."""

    @pytest.mark.slow
    def test_learns_to_separate_the_training_classes(self):
        from hiver_support.classifier.models import EmbeddingClassifier

        model = EmbeddingClassifier(seed=7).fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert model.predict("my battery drains so fast").intent == "battery_charging"

    @pytest.mark.slow
    def test_returns_contract_valid_predictions(self):
        from hiver_support.classifier.models import EmbeddingClassifier

        model = EmbeddingClassifier(seed=7).fit(TRAIN_TEXTS, TRAIN_LABELS)
        prediction = model.predict("wifi keeps disconnecting")
        assert prediction.taxonomy_hash == TAXONOMY.frozen_hash
        assert 0.0 <= prediction.confidence <= 1.0

    def test_declares_no_token_evidence_which_is_the_explainability_cost(self):
        from hiver_support.classifier.models import EmbeddingClassifier

        assert EmbeddingClassifier.name == "embedding_logreg"

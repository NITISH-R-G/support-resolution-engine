"""Composes an intent model with the two attribute detectors into one classifier.

The composition enforces the safety property that the frozen taxonomy exists to guarantee:
**attributes dominate the intent, and confidence never overrides them.** A confidently
predicted `battery_charging` on a message reporting a stolen device still escalates.

Kept deliberately thin so retrieval, generation and routing can consume `Prediction` without
coupling to any particular intent model — swapping TF-IDF for embeddings changes nothing
downstream.
"""

from __future__ import annotations

from hiver_support.classifier.attributes import ContextDetector, SecurityDetector
from hiver_support.classifier.contract import Prediction, PredictionSource
from hiver_support.classifier.models import IntentModel
from hiver_support.taxonomy import TAXONOMY


class IntentClassifier:
    """The classification subsystem: intent plus both attributes."""

    name = "intent_classifier"
    version = "1.0.0"

    def __init__(
        self,
        intent_model: IntentModel,
        security_detector: SecurityDetector | None = None,
        context_detector: ContextDetector | None = None,
    ) -> None:
        self.intent_model = intent_model
        self.security = security_detector or SecurityDetector()
        self.context = context_detector or ContextDetector()

    def predict(self, text: str) -> Prediction:
        if not isinstance(text, str):
            raise TypeError(f"predict expects str, got {type(text).__name__}")

        security = self.security.detect(text)
        context = self.context.detect(text)

        # Without enough context to identify a request, the intent model's output is not
        # evidence about this message — it reflects the training prior. Abstain rather than
        # inherit a confident label from a content-free message.
        if not context.value:
            return Prediction(
                intent="other_unclear",
                confidence=0.0,
                security_sensitive=security.value,
                context_sufficient=False,
                model_name=self.name,
                model_version=self.version,
                taxonomy_version=TAXONOMY.version,
                taxonomy_hash=TAXONOMY.frozen_hash,
                source=PredictionSource.ABSTAINED,
                evidence=context.evidence + security.evidence,
            )

        base = self.intent_model.predict(text)
        return Prediction(
            intent=base.intent,
            confidence=base.confidence,
            security_sensitive=security.value,
            context_sufficient=True,
            model_name=self.name,
            model_version=self.version,
            taxonomy_version=TAXONOMY.version,
            taxonomy_hash=TAXONOMY.frozen_hash,
            source=base.source,
            evidence=base.evidence + security.evidence,
        )

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]

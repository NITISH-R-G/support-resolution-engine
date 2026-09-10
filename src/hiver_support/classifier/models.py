"""Intent models: a shared interface plus the two required baselines.

Every model returns a contract-valid ``Prediction``, so routing consumes one type whichever
model produced it, and swapping models cannot change the downstream contract.

Baselines are here to establish whether a stronger model earns its complexity, so they are
built to be genuinely competitive rather than strawmanned. If TF-IDF matches an embedding
model, that is the result — a simpler, faster, fully explainable model is then the better
engineering choice, and `docs/CLASSIFIER.md` says so.

**Every dev number these produce is measured against weak labels** (``weak_labels.py``), which
makes it a rule-recovery score rather than accuracy. Nothing in this module should be read as
evidence of real-world classification quality.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from hiver_support.classifier.contract import (
    ContractError,
    Prediction,
    PredictionSource,
)
from hiver_support.taxonomy import TAXONOMY


class IntentModel:
    """Shared interface. Subclasses implement ``fit`` and ``_predict_one``."""

    name: str = "base"
    version: str = "0.0.0"

    def __init__(self, abstain_below: float = 0.0) -> None:
        self.abstain_below = abstain_below
        self._fitted = False

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _check_training_data(texts: list[str], labels: list[str]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels differ in length: {len(texts)} vs {len(labels)}")
        if not texts:
            raise ValueError("cannot fit on empty training data")
        valid = {i.name for i in TAXONOMY.intents}
        unknown = sorted(set(labels) - valid)
        if unknown:
            raise ContractError(
                f"training labels outside frozen taxonomy {TAXONOMY.version}: {unknown}"
            )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fit before predict")

    def _make(
        self,
        intent: str,
        confidence: float,
        *,
        evidence: tuple[str, ...] = (),
        security_sensitive: bool = False,
        context_sufficient: bool = True,
    ) -> Prediction:
        """Wrap a raw decision in the contract, applying the abstention threshold.

        Abstention is recorded as its own source rather than as a low-confidence guess: routing
        must treat "I do not know" differently from "probably this".
        """
        abstaining = confidence < self.abstain_below
        return Prediction(
            intent="other_unclear" if abstaining else intent,
            confidence=float(confidence),
            security_sensitive=security_sensitive,
            context_sufficient=context_sufficient,
            model_name=self.name,
            model_version=self.version,
            taxonomy_version=TAXONOMY.version,
            taxonomy_hash=TAXONOMY.frozen_hash,
            source=PredictionSource.ABSTAINED if abstaining else PredictionSource.MODEL,
            evidence=evidence,
        )

    # ------------------------------------------------------------------ api

    def fit(self, texts: list[str], labels: list[str]) -> IntentModel:  # pragma: no cover
        raise NotImplementedError

    def predict(self, text: str) -> Prediction:
        if not isinstance(text, str):
            raise TypeError(f"predict expects str, got {type(text).__name__}")
        self._require_fitted()
        return self._predict_one(text)

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]

    def _predict_one(self, text: str) -> Prediction:  # pragma: no cover
        raise NotImplementedError


class MajorityBaseline(IntentModel):
    """Baseline A: always predict the most frequent training label.

    Trivial by design. Its role is to expose how much of any headline number is explained by
    class imbalance alone — on a skewed distribution a majority predictor can look deceptively
    respectable on accuracy while having macro-F1 near zero.
    """

    name = "majority"
    version = "1.0.0"

    def fit(self, texts: list[str], labels: list[str]) -> MajorityBaseline:
        self._check_training_data(texts, labels)
        counts = Counter(labels)
        # Ties break alphabetically so the baseline is reproducible across runs.
        self._label = min(counts, key=lambda name: (-counts[name], name))
        self._share = counts[self._label] / len(labels)
        self._fitted = True
        return self

    def _predict_one(self, text: str) -> Prediction:
        return self._make(self._label, self._share, evidence=("majority-class",))


class TfidfLogisticClassifier(IntentModel):
    """Baseline B: TF-IDF features with multinomial logistic regression.

    Chosen over a naive-Bayes or SVM baseline because it produces calibratable probabilities
    and per-feature coefficients, which give both the confidence needed for selective
    prediction and a deterministic explanation of each decision.

    ``class_weight="balanced"`` is deliberate: weak labels are heavily skewed, and without it
    the rare intents that matter most for routing are never predicted at all.
    """

    name = "tfidf_logreg"
    version = "1.0.0"

    def __init__(
        self,
        seed: int = 20260910,
        abstain_below: float = 0.0,
        max_features: int = 50_000,
        min_df: int = 1,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
    ) -> None:
        super().__init__(abstain_below=abstain_below)
        self.seed = seed
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.C = C

    def fit(self, texts: list[str], labels: list[str]) -> TfidfLogisticClassifier:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self._check_training_data(texts, labels)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        features = self._vectorizer.fit_transform(texts)
        self._model = LogisticRegression(
            max_iter=2_000,
            random_state=self.seed,
            class_weight="balanced",
            C=self.C,
        )
        self._model.fit(features, labels)
        self._vocabulary = np.array(self._vectorizer.get_feature_names_out())
        self._fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Class probabilities, exposed so calibration can be fitted on dev."""
        self._require_fitted()
        return self._model.predict_proba(self._vectorizer.transform(texts))

    @property
    def classes(self) -> np.ndarray:
        self._require_fitted()
        return self._model.classes_

    def _evidence_for(self, text: str, label: str) -> tuple[str, ...]:
        """The highest-weighted features present in this message for the chosen class.

        A deterministic explanation, not a saliency approximation: these are the exact model
        coefficients acting on the exact features that fired.
        """
        vector = self._vectorizer.transform([text])
        present = vector.nonzero()[1]
        if present.size == 0:
            return ()
        class_index = int(np.where(self._model.classes_ == label)[0][0])
        weights = self._model.coef_[class_index][present]
        order = np.argsort(-weights)[:5]
        return tuple(self._vocabulary[present[i]] for i in order if weights[i] > 0)

    def _predict_one(self, text: str) -> Prediction:
        probabilities = self.predict_proba([text])[0]
        best = int(np.argmax(probabilities))
        label = str(self._model.classes_[best])
        return self._make(
            label,
            float(probabilities[best]),
            evidence=self._evidence_for(text, label),
        )


class EmbeddingClassifier(IntentModel):
    """Candidate stronger model: local sentence embeddings + logistic regression.

    Runs entirely locally (no API, no cost) and is included so the choice between a lexical
    and a semantic model rests on evidence rather than assumption.

    **A structural caveat governs how its dev score may be read.** The training labels are
    produced by lexical labelling functions, so agreement with those labels rewards
    reproducing a regex. A semantic model that correctly generalises to a paraphrase the regex
    misses is *penalised* for it. TF-IDF therefore has an unfair advantage on this metric, and
    the comparison cannot settle which model classifies intent better — only which better
    imitates the labeller. See `docs/CLASSIFIER.md`.
    """

    name = "embedding_logreg"
    version = "1.0.0"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, seed: int = 20260910, abstain_below: float = 0.0) -> None:
        super().__init__(abstain_below=abstain_below)
        self.seed = seed
        self._encoder = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)
        return np.asarray(
            self._encoder.encode(
                texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
            ),
            dtype=np.float32,
        )

    def fit(self, texts: list[str], labels: list[str]) -> EmbeddingClassifier:
        from sklearn.linear_model import LogisticRegression

        self._check_training_data(texts, labels)
        self._model = LogisticRegression(
            max_iter=2_000, random_state=self.seed, class_weight="balanced"
        )
        self._model.fit(self._encode(texts), labels)
        self._fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self._require_fitted()
        return self._model.predict_proba(self._encode(texts))

    @property
    def classes(self) -> np.ndarray:
        self._require_fitted()
        return self._model.classes_

    def _predict_one(self, text: str) -> Prediction:
        probabilities = self.predict_proba([text])[0]
        best = int(np.argmax(probabilities))
        # No token-level evidence: the decision is made in embedding space, which is exactly
        # the explainability cost weighed in the model-selection decision.
        return self._make(
            str(self._model.classes_[best]),
            float(probabilities[best]),
            evidence=("embedding-similarity",),
        )

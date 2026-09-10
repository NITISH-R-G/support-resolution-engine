"""The typed contract every classifier must satisfy.

Routing, retrieval, generation and the evaluation harness all consume ``Prediction``, so an
invalid one must be impossible to construct. Validation happens in ``__post_init__`` and
**raises**: a wrong-but-plausible label that flows silently into a routing decision is far
more dangerous than a crash at the boundary.

Two design choices are load-bearing:

**Fail closed, never default.** There is no fallback intent. Omitting the field is a
``TypeError`` and an out-of-taxonomy value is a ``ContractError``. Quietly substituting
``other_unclear`` would turn a model defect into a plausible-looking prediction.

**Bind the taxonomy.** Every prediction carries the taxonomy version and content hash it was
made against. Taxonomy v0.3.0 is frozen; a prediction made against different label definitions
is not comparable to one made against these, and binding the hash makes that drift loud
instead of invisible.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

from hiver_support.taxonomy import TAXONOMY


class ContractError(ValueError):
    """Raised when a prediction violates the contract. Never caught inside the pipeline."""


class PredictionSource(str, Enum):
    """How the prediction was produced.

    ``ABSTAINED`` is distinct from a low-confidence model output: it records that the system
    declined to commit, which routing must treat as escalation rather than as a weak guess.
    """

    MODEL = "model"
    RULE = "rule"
    ABSTAINED = "abstained"


def _validate_probability(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number in [0, 1], got {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ContractError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise ContractError(f"{name} must lie in [0, 1], got {number}")


def _validate_bool(name: str, value: object) -> None:
    # `isinstance(1, int)` is True for bools, so the check is deliberately strict: accepting
    # 1/0 here would let a probability leak into a safety flag.
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__} {value!r}")


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classifier output for one customer message.

    The three predicted quantities are independent axes, matching frozen taxonomy v0.3.0:
    an intent, plus the ``security_sensitive`` and ``context_sufficient`` attributes.
    """

    intent: str
    confidence: float
    security_sensitive: bool
    context_sufficient: bool
    model_name: str
    model_version: str
    taxonomy_version: str
    taxonomy_hash: str
    source: PredictionSource = PredictionSource.MODEL
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.intent not in {i.name for i in TAXONOMY.intents}:
            raise ContractError(
                f"intent {self.intent!r} is not in frozen taxonomy "
                f"{TAXONOMY.version}; valid: {sorted(i.name for i in TAXONOMY.intents)}"
            )
        _validate_probability("confidence", self.confidence)
        _validate_bool("security_sensitive", self.security_sensitive)
        _validate_bool("context_sufficient", self.context_sufficient)

        if self.taxonomy_version != TAXONOMY.version:
            raise ContractError(
                f"taxonomy_version {self.taxonomy_version!r} does not match the frozen "
                f"taxonomy {TAXONOMY.version!r}"
            )
        if self.taxonomy_hash != TAXONOMY.frozen_hash:
            raise ContractError(
                "taxonomy_hash does not match the frozen taxonomy; this prediction was made "
                "against different label definitions and is not comparable"
            )
        if not self.model_name or not self.model_version:
            raise ContractError("model_name and model_version are required for provenance")

    @property
    def abstained(self) -> bool:
        return self.source is PredictionSource.ABSTAINED

    @property
    def must_escalate(self) -> bool:
        """Whether routing must escalate, per SPEC section 8.

        Attributes and abstention dominate the intent, and confidence never overrides them.
        A confidently-predicted safe intent on a message the customer flagged as a suspected
        compromise must still escalate — that is the entire reason safety is an orthogonal
        attribute rather than a label.
        """
        if self.abstained:
            return True
        return TAXONOMY.must_escalate(
            self.intent,
            security_sensitive=self.security_sensitive,
            context_sufficient=self.context_sufficient,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["evidence"] = list(self.evidence)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> Prediction:
        return cls(
            **{
                **payload,
                "source": PredictionSource(payload.get("source", "model")),
                "evidence": tuple(payload.get("evidence", ())),
            }
        )

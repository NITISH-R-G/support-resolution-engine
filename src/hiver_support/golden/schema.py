"""Records for the golden evaluation set, and the guards that keep gold labels human.

Two structural guarantees live here, and both are enforced by construction rather than by
discipline:

**A candidate cannot carry the answer or a label.** ``GoldenCandidate`` has no field for the
brand's reply and none for an intent. The annotation tool therefore *cannot* show the answer,
and no script can backfill a label, because there is nowhere to put one. The brand's reply is
downstream of the intent — an annotator who reads "let's take this to DM" and infers
"escalation-worthy" has made the gold label a function of the behaviour being scored.

**An annotation cannot be fabricated.** ``GoldenAnnotation`` refuses any provenance but
``HUMAN_LABELED`` and refuses machine-sounding annotator ids. This is not paranoia about our
own intentions: one audited public repository shipped keyword rules in a file named
``human_review_pass.py``, and the only defence that survives a deadline is one that raises.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum

from hiver_support.taxonomy import TAXONOMY

ANNOTATION_VERSION = "1.0.0"

# Substrings that would let a candidate record carry the brand's reply, checked against the
# dataclass fields themselves so the guarantee cannot rot as fields are added.
_ANSWER_BEARING = ("reply", "support", "resolution", "answer")
_LABEL_BEARING = ("intent", "escalate", "gold")

# An annotator id is the only thing standing between a human label and a generated one, so
# ids that would let generated labels pass as human are refused outright.
_RESERVED_ANNOTATOR_IDS = frozenset(
    {
        "llm", "gpt", "gpt-4", "gpt4", "claude", "anthropic", "openai", "openrouter",
        "model", "auto", "automatic", "script", "bot", "ai", "agent", "system",
        "weak_labels", "weak", "classifier", "pseudo", "synthetic", "none", "unknown",
    }
)


class GoldenSetError(ValueError):
    """Raised when a golden record would violate the protocol. Never caught in the pipeline."""


class LabelProvenance(str, Enum):
    """Where a label came from. Exactly one applies to every labelled quantity we report.

    The distinction is the deliverable. A number computed against weak labels is a
    rule-recovery score; the same number against human labels is accuracy. Conflating them is
    the failure this project exists to avoid.
    """

    UNLABELED = "UNLABELED"
    WEAKLY_LABELED = "WEAKLY_LABELED"
    MODEL_GENERATED = "MODEL_GENERATED"
    HUMAN_LABELED = "HUMAN_LABELED"

    @property
    def is_gold(self) -> bool:
        return self is LabelProvenance.HUMAN_LABELED


class ExpectedResolutionKind(str, Enum):
    """What *kind* of thing would resolve this message.

    This is the evidence expectation the eventual reply evaluation needs: a message whose
    resolution requires a human action can never be answered well from historical text, so
    scoring a grounded reply against it would measure the wrong thing.
    """

    SELF_SERVE_STEPS = "self_serve_steps"
    INFORMATION = "information"
    HUMAN_ACTION_REQUIRED = "human_action_required"
    NO_RESOLUTION_POSSIBLE = "no_resolution_possible"
    UNCLEAR = "unclear"


class LabelConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoldenSetError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_bool(name: str, value: object) -> None:
    # `isinstance(1, int)` is True for bools, so the check is deliberately strict: a 1 in a
    # safety flag is how a probability silently becomes a routing decision.
    if not isinstance(value, bool):
        raise GoldenSetError(f"{name} must be a bool, got {type(value).__name__} {value!r}")


def _require_intent(name: str, value: str) -> str:
    if value not in TAXONOMY.names:
        raise GoldenSetError(
            f"{name} {value!r} is not in frozen taxonomy {TAXONOMY.version}; "
            f"valid: {sorted(TAXONOMY.names)}"
        )
    return value


def _coerce_enum(name: str, value: object, enum_cls: type[Enum]) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid = sorted(m.value for m in enum_cls)
        raise GoldenSetError(f"{name} {value!r} is not one of {valid}") from exc


class ReviewAction(str, Enum):
    """How a human arrived at a label. NOT a provenance class.

    Provenance answers "where did this label come from" and stays four-valued. This answers
    "how was it reviewed", which is what makes anchoring bias measurable: the accept rate on
    suggested examples compared against the blind ones.
    """

    ENTERED = "entered"      # blind - no suggestion was shown
    ACCEPTED = "accepted"    # a suggestion was shown and adopted unchanged
    CORRECTED = "corrected"  # a suggestion was shown and changed


@dataclass(frozen=True, slots=True)
class FlagRecord:
    """A human marking an example as needing deeper review. Leaves it UNRESOLVED.

    Distinct from skipping, which leaves no trace. A flag says a person looked and could not
    decide, so the example must not quietly vanish from the queue or slip into the freeze as
    though it were done.
    """

    pair_id: str
    annotator_id: str
    reason: str
    pass_number: int = 1
    timestamp_utc: str = field(default_factory=_utc_now)
    record_type: str = "flag"

    def __post_init__(self) -> None:
        _require_text("pair_id", self.pair_id)
        annotator = _require_text("annotator_id", self.annotator_id)
        if annotator.strip().lower() in _RESERVED_ANNOTATOR_IDS:
            raise GoldenSetError(f"annotator_id {annotator!r} is reserved")
        _require_text("reason", self.reason)
        if self.record_type != "flag":
            raise GoldenSetError("record_type of a FlagRecord is always 'flag'")

    def to_dict(self) -> dict:
        return {
            "record_type": "flag",
            "pair_id": self.pair_id,
            "annotator_id": self.annotator_id,
            "reason": self.reason,
            "pass_number": self.pass_number,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FlagRecord:
        data = dict(payload)
        data.pop("record_type", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ContextTurn:
    """A conversation turn that *preceded* the message being annotated.

    Preceding turns are shown because the agent would have had them. The turn that *answers*
    the message is not a preceding turn and never appears here.
    """

    author_role: str
    text: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.author_role not in ("customer", "brand"):
            raise GoldenSetError(
                f"author_role must be 'customer' or 'brand', got {self.author_role!r}"
            )

    def to_dict(self) -> dict:
        return {
            "author_role": self.author_role,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ContextTurn:
        return cls(
            author_role=payload["author_role"],
            text=payload["text"],
            created_at=datetime.fromisoformat(payload["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class GoldenCandidate:
    """One example awaiting human annotation. Carries no label and no answer.

    ``label_status`` exists solely so a serialised candidate states out loud that it is
    unlabelled. It cannot hold any other value.
    """

    pair_id: str
    conversation_id: str
    customer_tweet_id: str
    customer_message: str
    created_at: datetime
    context: tuple[ContextTurn, ...] = ()
    label_status: LabelProvenance = LabelProvenance.UNLABELED

    def __post_init__(self) -> None:
        _require_text("pair_id", self.pair_id)
        _require_text("conversation_id", self.conversation_id)
        _require_text("customer_tweet_id", self.customer_tweet_id)
        _require_text("customer_message", self.customer_message)
        if self.label_status is not LabelProvenance.UNLABELED:
            raise GoldenSetError(
                "a GoldenCandidate is always UNLABELED; labels live in the separate "
                "annotations file and may only be written by scripts/annotate_golden.py"
            )

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "conversation_id": self.conversation_id,
            "customer_tweet_id": self.customer_tweet_id,
            "customer_message": self.customer_message,
            "created_at": self.created_at.isoformat(),
            "context": [turn.to_dict() for turn in self.context],
            "label_status": self.label_status.value,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> GoldenCandidate:
        return cls(
            pair_id=payload["pair_id"],
            conversation_id=payload["conversation_id"],
            customer_tweet_id=payload["customer_tweet_id"],
            customer_message=payload["customer_message"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            context=tuple(ContextTurn.from_dict(t) for t in payload.get("context", ())),
        )


def assert_candidate_schema_is_blind() -> None:
    """Prove the candidate record cannot carry the answer or a label.

    Called by the sampler before writing anything, so adding a field like ``support_text`` for
    convenience fails the build rather than quietly contaminating every future annotation.
    """
    names = {f.name for f in fields(GoldenCandidate)}
    leaky = sorted(n for n in names if any(word in n for word in _ANSWER_BEARING))
    if leaky:
        raise GoldenSetError(
            f"GoldenCandidate has answer-bearing fields {leaky}; the brand's reply must never "
            f"reach an annotator (docs/GOLDEN_SET.md section 4)"
        )
    labelled = sorted(
        n for n in names if any(word in n for word in _LABEL_BEARING) and n != "label_status"
    )
    if labelled:
        raise GoldenSetError(
            f"GoldenCandidate has label-bearing fields {labelled}; a candidate with a label "
            f"field is a candidate a script can backfill"
        )



@dataclass(frozen=True, slots=True)
class GoldenAnnotation:
    """One human labelling action. The only record type this project accepts as gold.

    ``should_escalate`` is annotated independently of intent and the two attributes on
    purpose. The deterministic policy already *derives* escalation from them, so a derived
    gold label would score the policy against itself and could not fail. Disagreement between
    this field and the derived policy is a finding, not a labelling error.
    """

    pair_id: str
    annotator_id: str
    intent: str
    security_sensitive: bool
    context_sufficient: bool
    should_escalate: bool
    expected_resolution_kind: ExpectedResolutionKind
    is_ambiguous: bool = False
    alternative_intent: str | None = None
    label_confidence: LabelConfidence = LabelConfidence.HIGH
    reference_resolution: str = ""
    notes: str = ""
    pass_number: int = 1
    seconds_spent: float = 0.0
    # How this label was reached, and what was on screen when it was. Recorded so the
    # accept rate on suggested examples can be compared against the blind ones - which is
    # the only way anchoring bias is measurable.
    review_action: ReviewAction = ReviewAction.ENTERED
    model_suggestion: dict | None = None
    corrected_fields: tuple[str, ...] = ()
    timestamp_utc: str = field(default_factory=_utc_now)
    annotation_version: str = ANNOTATION_VERSION
    taxonomy_version: str = TAXONOMY.version
    taxonomy_hash: str = TAXONOMY.frozen_hash
    provenance: LabelProvenance = LabelProvenance.HUMAN_LABELED

    def __post_init__(self) -> None:
        if self.provenance is not LabelProvenance.HUMAN_LABELED:
            raise GoldenSetError(
                "a GoldenAnnotation is by definition HUMAN_LABELED; weak, model and "
                "unlabelled records belong in their own files and are never gold"
            )
        _require_text("pair_id", self.pair_id)
        annotator = _require_text("annotator_id", self.annotator_id)
        if annotator.strip().lower() in _RESERVED_ANNOTATOR_IDS:
            raise GoldenSetError(
                f"annotator_id {annotator!r} is reserved; gold labels must be attributed to a "
                f"named human. If a model produced this label it is not gold."
            )

        object.__setattr__(self, "intent", _require_intent("intent", self.intent))
        for name in ("security_sensitive", "context_sufficient", "should_escalate", "is_ambiguous"):
            _require_bool(name, getattr(self, name))

        if self.alternative_intent is not None:
            if not self.is_ambiguous:
                raise GoldenSetError(
                    "alternative_intent requires is_ambiguous=True; a runner-up label on an "
                    "unambiguous example is a contradiction"
                )
            _require_intent("alternative_intent", self.alternative_intent)
            if self.alternative_intent == self.intent:
                raise GoldenSetError("alternative_intent must differ from the primary intent")

        object.__setattr__(
            self,
            "expected_resolution_kind",
            _coerce_enum(
                "expected_resolution_kind", self.expected_resolution_kind, ExpectedResolutionKind
            ),
        )
        object.__setattr__(
            self,
            "label_confidence",
            _coerce_enum("label_confidence", self.label_confidence, LabelConfidence),
        )

        if not isinstance(self.pass_number, int) or self.pass_number < 1:
            raise GoldenSetError(f"pass_number must be a positive int, got {self.pass_number!r}")

        object.__setattr__(
            self, "review_action", _coerce_enum("review_action", self.review_action, ReviewAction)
        )
        if self.review_action is not ReviewAction.ENTERED and not self.model_suggestion:
            # Claiming a suggestion was accepted when none was shown would fabricate the
            # anchoring record that makes bias measurable.
            raise GoldenSetError(
                f"review_action {self.review_action.value!r} requires the model_suggestion "
                f"that was shown; only ENTERED means no suggestion existed"
            )
        if self.review_action is ReviewAction.CORRECTED and not self.corrected_fields:
            raise GoldenSetError(
                "a CORRECTED annotation must name at least one changed field in "
                "corrected_fields, otherwise it is an acceptance"
            )
        if self.taxonomy_version != TAXONOMY.version or self.taxonomy_hash != TAXONOMY.frozen_hash:
            raise GoldenSetError(
                "annotation was made against different taxonomy definitions and is not "
                f"comparable to frozen {TAXONOMY.version} ({TAXONOMY.frozen_hash[:12]}...)"
            )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["expected_resolution_kind"] = self.expected_resolution_kind.value
        payload["label_confidence"] = self.label_confidence.value
        payload["provenance"] = self.provenance.value
        payload["review_action"] = self.review_action.value
        payload["corrected_fields"] = list(self.corrected_fields)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> GoldenAnnotation:
        data = dict(payload)
        data.pop("provenance", None)
        data.pop("record_type", None)
        if "corrected_fields" in data:
            data["corrected_fields"] = tuple(data["corrected_fields"] or ())
        return cls(**data)


RETRACTION_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class Retraction:
    """A record withdrawing an earlier annotation, without destroying it.

    Forced by a real incident: pass 1 was started before the annotation guide had been read,
    and one example was labelled carelessly and saved. The log is append-only and the CLI
    skips examples already annotated in a pass, so the only options were to delete the line
    and lose the audit trail, overwrite it and leave an invalid label indistinguishable from a
    considered one, or abandon the pass.

    A retraction is the supported alternative. It is **another append-only record**: the
    original annotation stays exactly as written, this states who withdrew it and why, and
    everything downstream treats the example as unannotated again.

    It is **not** a fifth provenance class. A retraction describes an annotation; it is not a
    kind of label, and it carries no label fields at all — there is deliberately no back door
    for writing a label through this type.

    Bounded in time: it invalidates records written at or before its own timestamp, so
    re-annotating the same example afterwards works normally.
    """

    pair_id: str
    annotator_id: str
    reason: str
    pass_number: int = 1
    timestamp_utc: str = field(default_factory=_utc_now)
    retraction_version: str = RETRACTION_VERSION
    record_type: str = "retraction"

    def __post_init__(self) -> None:
        _require_text("pair_id", self.pair_id)
        annotator = _require_text("annotator_id", self.annotator_id)
        if annotator.strip().lower() in _RESERVED_ANNOTATOR_IDS:
            raise GoldenSetError(
                f"annotator_id {annotator!r} is reserved; a retraction is a human act and "
                f"must be attributed to a named person"
            )
        # A retraction with no reason is a deletion with extra steps, and the audit trail is
        # the entire justification for the mechanism.
        _require_text("reason", self.reason)
        if not isinstance(self.pass_number, int) or self.pass_number < 1:
            raise GoldenSetError(f"pass_number must be a positive int, got {self.pass_number!r}")
            raise GoldenSetError("record_type of a Retraction is always 'retraction'")

    def to_dict(self) -> dict:
        return {
            "record_type": "retraction",
            "pair_id": self.pair_id,
            "annotator_id": self.annotator_id,
            "reason": self.reason,
            "pass_number": self.pass_number,
            "timestamp_utc": self.timestamp_utc,
            "retraction_version": self.retraction_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Retraction:
        data = dict(payload)
        data.pop("record_type", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class AnnotatedExample:
    """A candidate joined to its human annotation, if one exists yet."""

    candidate: GoldenCandidate
    annotation: GoldenAnnotation | None = None

    @property
    def is_labelled(self) -> bool:
        return self.annotation is not None

    @property
    def provenance(self) -> LabelProvenance:
        return (
            LabelProvenance.HUMAN_LABELED if self.annotation else LabelProvenance.UNLABELED
        )

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "annotation": self.annotation.to_dict() if self.annotation else None,
            "provenance": self.provenance.value,
        }

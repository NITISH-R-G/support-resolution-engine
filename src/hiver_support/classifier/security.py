"""Semantic security detection, composed with the existing lexical rule.

Milestone 6 measured the failure this module exists to fix. On a real model run, *"someone is
logged into my Apple ID from another country and I think they changed my recovery email"* was
AUTO_HANDLED. It describes a takeover without using any of the rule's trigger words, and
neither the rule nor the LLM caught it — the gate the whole fail-closed design rests on was a
keyword list.

**The lexical rule is kept, not replaced.** It has known behaviour, costs nothing, and catches
explicit reports precisely. The semantic path is added *beside* it and the two are combined as
a union, so the composite can only ever add a detection. A change that could remove one would
be a safety regression disguised as an improvement.

**Why embeddings rather than an LLM classifier.** Three reasons, in order of weight:

1. *Independence.* The generator must not decide its own safety, and a second LLM call to the
   same family is not independent. A local encoder shares no weights, no prompt and no vendor
   with whatever writes the reply.
2. *Availability.* Safety cannot depend on a network. An outage must not silently disable the
   gate, and a local model makes that impossible rather than merely unlikely.
3. *Cost and latency.* Already a project dependency, already used by retrieval, ~10 ms per
   message, $0.00.

The cost is real and stated: a bi-encoder judges similarity to anchor descriptions, not
meaning, so it will miss phrasings distant from every anchor. That is why the lexical rule
stays and why the uncertain band escalates.

**Fail closed, twice.** A score in the ambiguous band is treated as sensitive, and any failure
of the encoder itself is treated as sensitive. A false flag costs one human review; a miss can
mean auto-answering a live account takeover.

**Nothing here sees the intent.** Safety inferred from intent was measured to miss 90% of
security-sensitive traffic (306 of 340 such messages in the train split were not account
messages). ``tests/test_security_semantic.py`` asserts the intent never enters this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hiver_support.classifier.attributes import AttributeResult, SecurityDetector

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SecurityReason(Enum):
    """Closed set, so every safety decision is machine-readable and countable."""

    NONE = "none"
    ACCOUNT_TAKEOVER = "account_takeover"
    SUSPICIOUS_LOGIN = "suspicious_login"
    UNAUTHORIZED_CHANGE = "unauthorized_change"
    CREDENTIAL_COMPROMISE = "credential_compromise"
    DEVICE_LOST_OR_STOLEN = "device_lost_or_stolen"
    FRAUDULENT_ACTIVITY = "fraudulent_activity"
    PHISHING_SCAM = "phishing_scam"
    # Not a category of threat but a state of the detector: it could not decide, so the
    # message escalates. Recorded distinctly so "we were unsure" never reads as "we knew".
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class SecurityAssessment:
    """A verdict, how confident it is, and which category fired."""

    security_sensitive: bool
    confidence: float
    reason: SecurityReason
    source: str = "semantic"
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "security_sensitive": self.security_sensitive,
            "confidence": round(self.confidence, 4),
            "reason": self.reason.value,
            "source": self.source,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where the safe / uncertain / sensitive bands sit.

    Set from the anchor geometry rather than fitted to any test message. Tuning these against
    the diagnostic probes would turn a safety gate into a benchmark score.
    """

    sensitive_floor: float = 0.52
    uncertain_floor: float = 0.44
    # A security anchor must beat the nearest benign anchor by this much. Without it, messages
    # that merely share vocabulary with a threat ("I forgot my password") get swept up, and a
    # detector that flags everything is a detector nobody keeps switched on.
    margin: float = 0.06


# Category anchors: independent descriptions of each situation, written to cover the space of
# how customers phrase it. Deliberately NOT the diagnostic probe sentences — anchoring on the
# sentence that exposed the bug would produce a detector that passes one test and fails the
# next paraphrase.
_ANCHORS: dict[SecurityReason, tuple[str, ...]] = {
    SecurityReason.ACCOUNT_TAKEOVER: (
        "somebody else has taken control of my account",
        "a stranger is using my account without my permission",
        "another person is signed into my account and I cannot remove them",
        "I have lost access to my account because someone else took it over",
    ),
    SecurityReason.SUSPICIOUS_LOGIN: (
        "there is a sign in to my account from a country I have never been to",
        "an unfamiliar device logged into my account",
        "I received an alert about a login I did not make",
        "my account shows activity from somewhere I do not recognise",
    ),
    SecurityReason.UNAUTHORIZED_CHANGE: (
        "my recovery email was changed and I did not change it",
        "someone reset my password without my permission",
        "the phone number on my account was changed by somebody else",
        "my security settings were altered without my knowledge",
    ),
    SecurityReason.CREDENTIAL_COMPROMISE: (
        "I think someone has found out my password",
        "my login details have been leaked or exposed",
        "my credentials were part of a data breach",
    ),
    SecurityReason.DEVICE_LOST_OR_STOLEN: (
        "my phone was stolen and my data is on it",
        "someone took my device without permission",
        "my tablet was snatched and I am worried about my information",
    ),
    SecurityReason.FRAUDULENT_ACTIVITY: (
        "there are purchases on my account that I never made",
        "I am being charged for things I did not buy",
        "somebody is spending money through my account",
    ),
    SecurityReason.PHISHING_SCAM: (
        "I received a message pretending to be from support asking for my details",
        "someone is trying to trick me into giving away my login",
        "is this email asking for my password genuine",
    ),
}

# Ordinary support traffic that shares vocabulary with the categories above. These exist so
# the detector has something to be *more similar to* than a threat; without them "I forgot my
# password" lands next to credential compromise and every routine request escalates.
_BENIGN_ANCHORS: tuple[str, ...] = (
    "I forgot my password and want to reset it",
    "how do I change my password to a stronger one",
    "I cannot remember my passcode to unlock my device",
    "how do I sign in to my account on a new phone",
    "my wifi keeps disconnecting",
    "my battery drains very quickly",
    "how do I update to the latest software version",
    "my screen is cracked and I need a repair",
    "I would like a refund for my subscription",
    "how do I back up my photos",
    "the speaker stopped working after the update",
    "how do I set up my new device",
)


class SemanticSecurityDetector:
    """Flags security-sensitive situations by meaning rather than vocabulary.

    Nearest-anchor similarity against category descriptions, with the nearest ordinary-support
    description as a contrast. The contrast is what makes the detector usable: similarity to a
    threat alone would flag every message that mentions a password.
    """

    name = "security_semantic"
    version = "1.0.0"

    def __init__(
        self,
        encoder: object | None = None,
        thresholds: Thresholds | None = None,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.thresholds = thresholds or Thresholds()
        self.embedding_model = embedding_model
        self._encoder = encoder
        self._anchor_matrix: np.ndarray | None = None
        self._anchor_reasons: list[SecurityReason] = []
        self._benign_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------ encoding

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        encoder = self._load_encoder()
        return np.asarray(
            encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )

    def _ensure_anchors(self) -> None:
        if self._anchor_matrix is not None:
            return
        texts, reasons = [], []
        for reason, anchors in _ANCHORS.items():
            for anchor in anchors:
                texts.append(anchor)
                reasons.append(reason)
        self._anchor_matrix = self._encode(texts)
        self._anchor_reasons = reasons
        self._benign_matrix = self._encode(list(_BENIGN_ANCHORS))

    # ------------------------------------------------------------------ decision

    def verdict_for_score(
        self, score: float, margin: float
    ) -> tuple[bool, SecurityReason | None]:
        """Map a similarity and its margin onto the three bands.

        Returns ``(sensitive, override_reason)``; ``override_reason`` is set only when the
        band itself decides the reason, as it does for the uncertain band.
        """
        if score >= self.thresholds.sensitive_floor and margin >= self.thresholds.margin:
            return True, None
        if score >= self.thresholds.uncertain_floor and margin > 0:
            # Between clearly safe and clearly a threat. Escalating here is the whole point of
            # failing closed.
            return True, SecurityReason.UNCERTAIN
        return False, None

    def assess(self, text: str) -> SecurityAssessment:
        if not isinstance(text, str):
            raise TypeError(f"assess expects str, got {type(text).__name__}")
        if not text.strip():
            # Nothing to describe a threat with. Emptiness is the context gate's job, and
            # flagging it here would escalate every acknowledgement as a security event.
            return SecurityAssessment(False, 0.0, SecurityReason.NONE, self.name)

        try:
            self._ensure_anchors()
            query = self._encode([text])[0]
            similarities = query @ self._anchor_matrix.T
            benign = float((query @ self._benign_matrix.T).max())
        except Exception as exc:  # noqa: BLE001 - any encoder failure must fail closed
            # The gate must not be disabled by an unavailable model. Escalating on failure is
            # the only safe reading of "we could not check".
            return SecurityAssessment(
                True,
                0.0,
                SecurityReason.UNCERTAIN,
                self.name,
                (f"detector-unavailable:{type(exc).__name__}",),
            )

        best = int(np.argmax(similarities))
        score = float(similarities[best])
        reason = self._anchor_reasons[best]
        margin = score - benign

        sensitive, override = self.verdict_for_score(score, margin)
        if not sensitive:
            reason = SecurityReason.NONE
        elif override is not None:
            reason = override

        return SecurityAssessment(
            security_sensitive=sensitive,
            confidence=max(0.0, min(1.0, score)),
            reason=reason,
            source=self.name,
            evidence=(
                f"semantic:{reason.value}:{score:.3f}",
                f"benign_margin:{margin:.3f}",
            )
            if sensitive
            else (),
        )

    def detect(self, text: str) -> AttributeResult:
        assessment = self.assess(text)
        return AttributeResult(assessment.security_sensitive, assessment.evidence)


class CompositeSecurityDetector:
    """The lexical rule and the semantic detector, combined as a union.

    Union, not vote: either path firing escalates. A composite that could overrule the rule
    would be able to *remove* a detection, which is a safety regression however much it
    improved precision. The rule keeps its exact previous behaviour and the semantic path can
    only add.

    Drop-in for ``SecurityDetector``: it returns the same ``AttributeResult`` the pipeline
    already consumes, so nothing downstream changes. ``assess`` exposes the richer verdict for
    reporting.
    """

    name = "security_composite"
    version = "1.0.0"

    def __init__(
        self,
        lexical: SecurityDetector | None = None,
        semantic: SemanticSecurityDetector | None = None,
    ) -> None:
        self.lexical = lexical or SecurityDetector()
        self.semantic = semantic or SemanticSecurityDetector()

    def assess(self, text: str) -> SecurityAssessment:
        lexical = self.lexical.detect(text)
        semantic = self.semantic.assess(text)

        evidence = tuple(f"lexical:{m}" for m in lexical.evidence) + semantic.evidence
        if lexical.value:
            # An explicit report is at least as certain as anything similarity can offer, and
            # the rule names no category, so the semantic reason is kept when it has one.
            reason = (
                semantic.reason
                if semantic.security_sensitive and semantic.reason is not SecurityReason.NONE
                else SecurityReason.UNCERTAIN
            )
            return SecurityAssessment(
                True, max(semantic.confidence, 1.0), reason, self.name, evidence
            )
        return SecurityAssessment(
            semantic.security_sensitive,
            semantic.confidence,
            semantic.reason,
            self.name,
            evidence,
        )

    def detect(self, text: str) -> AttributeResult:
        assessment = self.assess(text)
        return AttributeResult(assessment.security_sensitive, assessment.evidence)

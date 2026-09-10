"""The support resolution agent: classify, retrieve, generate, validate, decide.

```
customer message
  -> normalise + PII mask
  -> classify           intent, security_sensitive, context_sufficient
  -> risk gate          security / context / policy intent  -> ESCALATE
  -> retrieve           historical resolutions, temporally filtered
  -> evidence gate      empty or thin evidence               -> ESCALATE
  -> generate           grounded in retrieved evidence only
  -> grounding gate     deterministic validation             -> ESCALATE
  -> AUTO_HANDLE + reply + evidence ids
```

**The whole design is fail-closed.** Every gate can only send work to a human; none can
promote an uncertain case to automation. An agent that auto-handles when unsure is worse than
no agent, because the failure is invisible until a customer is harmed.

Ordering is deliberate. The cheap deterministic gates run first, so a security-flagged message
never reaches retrieval and never costs a model call. Grounding validation runs last and
independently of the generator, because a generator that judged its own output would be
marking its own homework.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from hiver_support.agent.generation import EvidenceTemplateGenerator, ReplyGenerator
from hiver_support.agent.grounding import validate_grounding
from hiver_support.agent.retrieval import HybridRetriever
from hiver_support.classifier.pipeline import IntentClassifier
from hiver_support.data.normalise import normalise_text
from hiver_support.data.pii import mask_pii
from hiver_support.taxonomy import TAXONOMY

BRAND = "AppleSupport"
DEFAULT_MIN_RETRIEVAL_CONFIDENCE = 0.35
DEFAULT_TOP_K = 4


class EscalationReason(str, Enum):
    """Closed set, so every escalation is machine-readable and countable in evaluation."""

    NONE = "none"
    SECURITY_SENSITIVE = "security_sensitive"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    POLICY_INTENT = "policy_intent"
    NO_EVIDENCE = "no_evidence"
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    UNGROUNDED = "ungrounded"
    EMPTY_DRAFT = "empty_draft"


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """The agent's full output for one customer message."""

    message: str
    intent: str
    intent_confidence: float
    security_sensitive: bool
    context_sufficient: bool
    action: str
    reason: EscalationReason
    reason_detail: str
    reply: str | None
    evidence_ids: tuple[str, ...]
    retrieval_confidence: float
    grounding: dict
    taxonomy_version: str
    taxonomy_hash: str

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "security_sensitive": self.security_sensitive,
            "context_sufficient": self.context_sufficient,
            "action": self.action,
            "reason": self.reason.value,
            "reason_detail": self.reason_detail,
            "reply": self.reply,
            "evidence_ids": list(self.evidence_ids),
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "grounding": self.grounding,
            "taxonomy_version": self.taxonomy_version,
            "taxonomy_hash": self.taxonomy_hash,
        }


class ReplyAgent:
    """Orchestrates the classifier, retriever, generator and grounding validator."""

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: HybridRetriever,
        generator: ReplyGenerator | None = None,
        min_retrieval_confidence: float = DEFAULT_MIN_RETRIEVAL_CONFIDENCE,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator or EvidenceTemplateGenerator()
        self.min_retrieval_confidence = min_retrieval_confidence
        self.top_k = top_k

    def _escalate(
        self,
        message: str,
        prediction,
        reason: EscalationReason,
        detail: str,
        evidence_ids: tuple[str, ...] = (),
        retrieval_confidence: float = 0.0,
        grounding: dict | None = None,
    ) -> AgentDecision:
        return AgentDecision(
            message=message,
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            security_sensitive=prediction.security_sensitive,
            context_sufficient=prediction.context_sufficient,
            action="ESCALATE",
            reason=reason,
            reason_detail=detail,
            reply=None,
            evidence_ids=evidence_ids,
            retrieval_confidence=retrieval_confidence,
            grounding=grounding or {"passed": False, "violations": []},
            taxonomy_version=TAXONOMY.version,
            taxonomy_hash=TAXONOMY.frozen_hash,
        )

    def handle(
        self,
        message: str,
        exclude_case_ids: set[str] | None = None,
        before: datetime | None = None,
    ) -> AgentDecision:
        """Process one customer message end to end.

        Args:
            exclude_case_ids: evidence to withhold, so a message cannot retrieve itself.
            before: retrieve only resolutions written before this instant.
        """
        if not isinstance(message, str):
            raise TypeError(f"handle expects str, got {type(message).__name__}")

        # PII is masked before anything else so nothing sensitive can reach a provider, and
        # normalisation happens once so every stage sees identical text.
        cleaned = mask_pii(normalise_text(message, brand=BRAND)).text
        prediction = self.classifier.predict(cleaned)

        # --- deterministic gates, cheapest and most safety-critical first ------------------
        if prediction.security_sensitive:
            return self._escalate(
                cleaned, prediction, EscalationReason.SECURITY_SENSITIVE,
                "customer expressed a security concern; never auto-handled",
            )

        if not prediction.context_sufficient:
            return self._escalate(
                cleaned, prediction, EscalationReason.INSUFFICIENT_CONTEXT,
                "the specific request cannot be determined from this message",
            )

        if TAXONOMY.get(prediction.intent).escalation_sensitive:
            return self._escalate(
                cleaned, prediction, EscalationReason.POLICY_INTENT,
                f"intent {prediction.intent!r} is escalation-sensitive by policy",
            )

        # --- evidence ---------------------------------------------------------------------
        retrieved = self.retriever.retrieve(
            cleaned, top_k=self.top_k, exclude_ids=exclude_case_ids, before=before
        )
        evidence_ids = tuple(c.case_id for c in retrieved.cases)

        if retrieved.is_empty:
            return self._escalate(
                cleaned, prediction, EscalationReason.NO_EVIDENCE,
                "no historical resolution was retrieved to ground a reply in",
            )

        if retrieved.confidence < self.min_retrieval_confidence:
            return self._escalate(
                cleaned, prediction, EscalationReason.LOW_RETRIEVAL_CONFIDENCE,
                f"retrieval confidence {retrieved.confidence:.3f} below "
                f"{self.min_retrieval_confidence:.3f}",
                evidence_ids, retrieved.confidence,
            )

        # --- generation and independent validation ----------------------------------------
        draft = self.generator.generate(cleaned, retrieved.cases, prediction.intent)
        if not draft.text.strip():
            return self._escalate(
                cleaned, prediction, EscalationReason.EMPTY_DRAFT,
                "generator declined to draft a reply from the available evidence",
                evidence_ids, retrieved.confidence,
            )

        report = validate_grounding(draft.text, retrieved.cases)
        if not report.passed:
            return self._escalate(
                cleaned, prediction, EscalationReason.UNGROUNDED,
                "; ".join(v.detail for v in report.violations),
                evidence_ids, retrieved.confidence, report.to_dict(),
            )

        return AgentDecision(
            message=cleaned,
            intent=prediction.intent,
            intent_confidence=prediction.confidence,
            security_sensitive=False,
            context_sufficient=True,
            action="AUTO_HANDLE",
            reason=EscalationReason.NONE,
            reason_detail="all gates passed: evidence retrieved and reply grounded",
            reply=draft.text,
            evidence_ids=draft.cited_case_ids or evidence_ids,
            retrieval_confidence=retrieved.confidence,
            grounding=report.to_dict(),
            taxonomy_version=TAXONOMY.version,
            taxonomy_hash=TAXONOMY.frozen_hash,
        )

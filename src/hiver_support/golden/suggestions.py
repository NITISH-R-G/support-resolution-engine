"""Provisional model suggestions for assisted annotation. Never gold.

SPEC section 9.2 calls for exactly this: *"160 of 200 receive a pre-annotator suggestion; 40
are blind."* The blind 40 are not a leftover — comparing the human's agreement with
suggestions against their blind labels is the only way anchoring bias becomes measurable.

The one risk in assisted annotation is that a suggestion quietly becomes a label. Four things
prevent it:

* a suggestion is a **different type** from an annotation and cannot be appended to the
  annotation log;
* it carries ``MODEL_GENERATED`` provenance, which ``is_gold`` refuses;
* it lives in a **different file**;
* it becomes gold only through an explicit human action that is itself recorded.

**The pre-annotator must not share a model family with the system under test.**
``leakage.assert_independent_models`` raises otherwise: gold labels produced by the model
being scored make the reported accuracy a measure of the annotator's edit rate, which is the
failure proved numerically in ``docs/PUBLIC_REPO_COMPARISON.md`` section 2.2.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hiver_support.golden.schema import (
    ExpectedResolutionKind,
    GoldenSetError,
    LabelProvenance,
    _require_bool,
    _require_intent,
)
from hiver_support.taxonomy import TAXONOMY

SUGGESTION_PROMPT_VERSION = "preannotate-v1"

# Below this, the model is guessing and the human decides from scratch.
MIN_CONFIDENT = 0.75

# SPEC 9.2: 160 of 200 receive a suggestion, 40 are blind. Seeded, so the blind subset is
# fixed before any suggestion exists and cannot drift afterwards.
BLIND_COUNT = 40
BLIND_SEED = 20260911


def blind_pair_ids(candidates: Sequence) -> set[str]:
    """The examples that must never see a suggestion.

    Lives here rather than in the generation script so the annotation CLI can compute it
    without importing anything from the agent - the CLI's blindness is asserted against
    its import list, and one convenience import would break it.
    """
    ordered = sorted(c.pair_id for c in candidates)
    rng = np.random.default_rng(BLIND_SEED)
    picks = rng.choice(len(ordered), size=min(BLIND_COUNT, len(ordered)), replace=False)
    return {ordered[int(i)] for i in picks}


@dataclass(frozen=True, slots=True)
class ModelSuggestion:
    """One provisional label. Provenance is MODEL_GENERATED and cannot be anything else."""

    pair_id: str
    intent: str
    security_sensitive: bool
    context_sufficient: bool
    should_escalate: bool
    expected_resolution_kind: str
    confidence: float
    rationale: str = ""
    model: str = ""
    provider: str = ""
    prompt_version: str = SUGGESTION_PROMPT_VERSION

    def __post_init__(self) -> None:
        _require_intent("intent", self.intent)
        for name in ("security_sensitive", "context_sufficient", "should_escalate"):
            _require_bool(name, getattr(self, name))
        if self.expected_resolution_kind not in {k.value for k in ExpectedResolutionKind}:
            raise GoldenSetError(
                f"expected_resolution_kind {self.expected_resolution_kind!r} is not valid"
            )
        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            raise GoldenSetError(f"confidence must lie in [0, 1], got {self.confidence!r}")

    @property
    def provenance(self) -> LabelProvenance:
        return LabelProvenance.MODEL_GENERATED

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "intent": self.intent,
            "security_sensitive": self.security_sensitive,
            "context_sufficient": self.context_sufficient,
            "should_escalate": self.should_escalate,
            "expected_resolution_kind": self.expected_resolution_kind,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model": self.model,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "provenance": LabelProvenance.MODEL_GENERATED.value,
            "_warning": "PROVISIONAL MODEL OUTPUT - never gold until a human accepts it",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ModelSuggestion:
        data = {k: v for k, v in payload.items() if not k.startswith("_")}
        data.pop("provenance", None)
        return cls(**data)


def needs_mandatory_review(suggestion: ModelSuggestion | None) -> tuple[bool, tuple[str, ...]]:
    """Whether one-key acceptance is forbidden for this example, and why.

    Fails closed: no suggestion means the human works from scratch. The conditions below are
    the ones where a wrong label is either safety-relevant or systematically likely, so
    accepting them with a single keystroke would be how a model's opinion becomes gold.
    """
    if suggestion is None:
        return True, ("no model suggestion available",)

    reasons: list[str] = []
    if suggestion.confidence < MIN_CONFIDENT:
        reasons.append(f"model confidence {suggestion.confidence:.2f} below {MIN_CONFIDENT}")
    if suggestion.security_sensitive:
        reasons.append("security-sensitive: a wrong label here is the costliest possible")
    if not suggestion.context_sufficient:
        reasons.append("context judged insufficient; these are systematically contested")
    if TAXONOMY.get(suggestion.intent).escalation_sensitive:
        reasons.append(f"intent {suggestion.intent!r} is escalation-sensitive by policy")
    if suggestion.should_escalate:
        reasons.append("escalation decision: routing gold must not be adopted unexamined")
    return bool(reasons), tuple(reasons)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_PROMPT = """You are pre-annotating customer support messages sent to {brand} on Twitter, so
that a human annotator can review your work quickly. You are NOT deciding the final label.

TAXONOMY (choose exactly one intent):
{taxonomy}

TWO INDEPENDENT ATTRIBUTES - a message can have ANY intent and still be either of these:
- security_sensitive: the customer expresses concern about compromise, unauthorised access,
  fraud, theft, or a scam. Flag on their SUSPICION, never on confirmation.
- context_sufficient: false when the specific request cannot be determined at all.

{context}CUSTOMER MESSAGE:
{message}

Answer with ONE JSON object and nothing else:
{{
  "intent": "one of the intent names above, exactly",
  "security_sensitive": true or false,
  "context_sufficient": true or false,
  "should_escalate": true or false,
  "expected_resolution_kind": "self_serve_steps | information | human_action_required | no_resolution_possible | unclear",
  "confidence": a number between 0 and 1,
  "rationale": "one short sentence, under 20 words"
}}

Be honest with confidence. A low number sends the example to careful human review, which is
the correct outcome when the message is genuinely unclear."""


def build_suggestion_prompt(candidate, brand: str = "AppleSupport") -> str:
    """Render the pre-annotation prompt for one candidate."""
    taxonomy = "\n".join(
        f"- {intent.name}: {intent.definition}" for intent in TAXONOMY.intents
    )
    context = ""
    if candidate.context:
        turns = "\n".join(f"  {t.author_role}: {t.text}" for t in candidate.context)
        context = f"EARLIER IN THIS CONVERSATION:\n{turns}\n\n"
    return _PROMPT.format(
        brand=brand, taxonomy=taxonomy, context=context, message=candidate.customer_message
    )


def parse_suggestion(pair_id: str, text: str, model: str, provider: str) -> ModelSuggestion:
    """Parse a model response into a suggestion. Raises rather than repairing.

    A repaired suggestion is one whose provenance is partly ours, and a malformed one simply
    means the human annotates that example from scratch — which is the safe outcome.
    """
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"pre-annotator did not return JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GoldenSetError("pre-annotator returned a non-object")
    return ModelSuggestion(
        pair_id=pair_id,
        intent=str(payload.get("intent", "")),
        security_sensitive=bool(payload.get("security_sensitive", False)),
        context_sufficient=bool(payload.get("context_sufficient", True)),
        should_escalate=bool(payload.get("should_escalate", False)),
        expected_resolution_kind=str(payload.get("expected_resolution_kind", "unclear")),
        confidence=float(payload.get("confidence", 0.0)),
        rationale=str(payload.get("rationale", ""))[:200],
        model=model,
        provider=provider,
    )


def write_suggestions(path: Path, suggestions) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(s.to_dict(), ensure_ascii=False) for s in suggestions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_suggestions(path: Path) -> dict[str, ModelSuggestion]:
    """Suggestions keyed by pair_id. An absent file simply means none were generated."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, ModelSuggestion] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            suggestion = ModelSuggestion.from_dict(json.loads(line))
        except (GoldenSetError, TypeError, ValueError):
            # An invalid suggestion is simply not offered; the human annotates that example
            # from scratch. Failing closed here costs one careful review, not a wrong label.
            continue
        out[suggestion.pair_id] = suggestion
    return out

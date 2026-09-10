"""Independent detectors for the two orthogonal attributes of frozen taxonomy v0.3.0.

Both are **deterministic rules, not trained models**, and that is deliberate. No human labels
exist yet, so the only available training signal is the rule itself; a model trained to
imitate it could only lose recall while adding the appearance of learning. On the safety path
— where a miss means auto-answering a real account takeover — a rule with known behaviour
beats a model with unknown recall. `docs/CLASSIFIER.md` records this decision and the
conditions under which it should be revisited.

Neither detector ever sees the intent. Safety inferred from intent was measured to miss 90% of
security-sensitive traffic (306 of 340 such messages in the train split were not account
messages), which is precisely why the taxonomy carries safety as an orthogonal attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hiver_support.classifier.weak_labels import (
    MIN_CONTENT_WORDS,
    _ACKNOWLEDGEMENT_RE,
    _DEVICE_ONLY_RE,
    _SECURITY_RE,
    _VERSION_ONLY_RE,
)


@dataclass(frozen=True, slots=True)
class AttributeResult:
    """A detector verdict plus the evidence for it, so decisions stay inspectable."""

    value: bool
    evidence: tuple[str, ...] = ()


class SecurityDetector:
    """Flags messages where the customer expresses a security concern.

    Fires on **suspicion expressed by the customer**, never on confirmation: whether an account
    was truly compromised is not observable from a tweet, so waiting for certainty would mean
    never flagging anything. Recall is preferred over precision here, because a false flag
    costs one unnecessary escalation while a miss can mean auto-answering a live compromise.
    """

    name = "security_rule"
    version = "1.0.0"

    def detect(self, text: str) -> AttributeResult:
        if not isinstance(text, str):
            raise TypeError(f"detect expects str, got {type(text).__name__}")
        matches = tuple(sorted({m.group(0).lower() for m in _SECURITY_RE.finditer(text)}))
        return AttributeResult(value=bool(matches), evidence=matches)


class ContextDetector:
    """Judges whether the specific request can be determined from this message.

    Insufficient context is a *failure to determine the request*, not a kind of request. Bare
    acknowledgements, isolated version strings and lone device names name a topic without
    asking anything, and roughly 8% of the train split is exactly that.
    """

    name = "context_rule"
    version = "1.0.0"

    def detect(self, text: str) -> AttributeResult:
        if not isinstance(text, str):
            raise TypeError(f"detect expects str, got {type(text).__name__}")

        stripped = text.strip()
        if not stripped:
            return AttributeResult(False, ("empty",))
        if _ACKNOWLEDGEMENT_RE.match(stripped):
            return AttributeResult(False, ("bare-acknowledgement",))
        if _VERSION_ONLY_RE.match(stripped):
            return AttributeResult(False, ("version-string-only",))
        if _DEVICE_ONLY_RE.match(stripped):
            return AttributeResult(False, ("device-name-only",))

        words = re.findall(r"[A-Za-z']+", stripped)
        if len(words) < MIN_CONTENT_WORDS:
            return AttributeResult(False, (f"fewer-than-{MIN_CONTENT_WORDS}-words",))
        return AttributeResult(True, ())

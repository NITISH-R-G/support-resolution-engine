"""Weak supervision for classifier training labels.

**Nothing here is a human label.** No hand-labelled data exists: the golden set is drawn from
the held-out test pool and has not been built. Training labels are produced by labelling
functions, and every output carries ``is_weak=True`` so the distinction cannot be lost
downstream. Three of the seven public repositories audited in
``docs/PUBLIC_REPO_COMPARISON.md`` presented machine-generated labels as human ones; the
flags, the provenance string and the batch warning exist so that cannot happen here by
accident.

Two circularity warnings, both structural:

1. The frozen taxonomy was derived from the train split, so labelling functions written
   against that taxonomy and applied to that split are doubly self-referential.
2. **A model trained on weak labels and evaluated against weak labels measures agreement with
   these heuristics, not intent accuracy.** Any dev number obtained that way is a
   rule-recovery score and must be reported as one.

Design notes:

* **Abstention is the default.** A labelling function that fires on everything carries no
  information, so patterns are deliberately high-precision and low-recall, and coverage is
  reported rather than maximised.
* **Priority order resolves conflicts deterministically**, mirroring the frozen taxonomy's
  tie-breaks: more specific and more escalation-sensitive labels win. Conflicts are recorded,
  never silently discarded.
* **Context is checked first.** A message with too little content to determine a request gets
  no intent at all, so keyword luck cannot manufacture a confident label from "iPhone 7".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import Counter

WEAK_PROVENANCE = (
    "WEAK SUPERVISION: produced by deterministic labelling functions, NOT human annotation. "
    "Not ground truth. Metrics computed against these labels measure agreement with the "
    "heuristics, not intent accuracy."
)

# Priority order mirrors the frozen taxonomy's tie-breaks: escalation-sensitive and more
# specific labels win, so conflict resolution matches routing precedence rather than accident.
_INTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "repair_order_replacement",
        r"\b(repair\w*|replacement|replace my|warranty|applecare|genius bar|appointment|"
        r"my order|deliver\w*|shipp\w*|cracked screen|water damage|send it (in|back)|"
        r"trade.?in|refurb\w*)\b",
    ),
    (
        "billing_and_subscription",
        r"\b(charg\w*|billing|billed|refund\w*|invoice|receipt|subscription|subscribed|"
        r"renew\w*|payment|paid for|apple ?pay|gift card|itunes credit|double.?charg\w*)\b",
    ),
    (
        "account_access",
        r"\b(apple ?id|password|passcode|sign ?in|signin|log ?in|login|locked out|"
        r"two.?factor|2fa|verification code|recovery key|security question|account (locked|"
        r"disabled))\b",
    ),
    (
        "battery_charging",
        r"\b(battery|charging|charger|charge\b|drain\w*|overheat\w*|dies (fast|quick)|"
        r"percent\w*|power\w* (off|down))\b",
    ),
    (
        "connectivity",
        r"\b(wi-?fi|wifi|bluetooth|cellular|lte|4g|5g|no service|won'?t connect|not connecting|"
        r"disconnect\w*|hotspot|airdrop|network|signal)\b",
    ),
    (
        "apps_and_services",
        r"\b(app ?store|itunes|apple music|imessage|facetime|siri|safari|icloud|apple tv|"
        r"podcast\w*|apple pay|find my|garageband|keynote|numbers app|pages app)\b",
    ),
    (
        "howto_information",
        r"\b(how (do|can|would|to)|is there a way|where (do|can) i|what does|can i |"
        r"how come|any way to)\b",
    ),
    (
        "complaint_feedback",
        r"\b(worst|terrible|awful|disgrace\w*|useless|ridiculous|unacceptable|complain\w*|"
        r"feedback|suggestion|please add|fix your|so disappointed|never buying)\b",
    ),
    (
        "device_malfunction",
        r"\b(not working|doesn'?t work|isn'?t working|broken|freez\w*|crash\w*|stuck|"
        r"glitch\w*|unresponsive|won'?t turn on|black screen|error|bug\b|lagg\w*|slow\w*)\b",
    ),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _INTENT_PATTERNS
)

# Security cuts across every intent by design: 306 of 340 security-sensitive messages in the
# train split fell outside the account topic, so this must never be inferred from the intent.
_SECURITY_RE = re.compile(
    r"\b(hack\w*|compromis\w*|unauthoriz\w*|unauthoris\w*|"
    r"someone (else )?(has|is using|accessed|got into|logged into|is signed into|"
    r"signed into|is in my)|took over my account|"
    r"phish\w*|scam\w*|fake (email|apple|message|text)|impersonat\w*|pretending to be|"
    r"stolen|stole my|fraud\w*|identity theft|"
    r"is this (email|text|message) (from you|legit|real))\b",
    re.IGNORECASE,
)

# Content-free messages: bare acknowledgements, isolated version strings, device names alone.
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\W*(ok(ay)?|yes|no|thanks?( you)?|ta|cheers|done|sure|k|yep|nope|help|please|"
    r"hello|hi|hey)\W*$",
    re.IGNORECASE,
)
_VERSION_ONLY_RE = re.compile(r"^\W*(ios\s*)?\d+(\.\d+)*\W*$", re.IGNORECASE)
_DEVICE_ONLY_RE = re.compile(
    r"^\W*(iphone|ipad|ipod|macbook|imac|watch|airpods)\s*\w{0,3}\W*$", re.IGNORECASE
)

MIN_CONTENT_WORDS = 4


@dataclass(frozen=True, slots=True)
class WeakLabel:
    """One weakly-supervised label. ``is_weak`` is always True and cannot be unset."""

    intent: str | None
    security_sensitive: bool
    context_sufficient: bool
    votes: tuple[str, ...] = ()
    is_weak: bool = True

    @property
    def abstained(self) -> bool:
        return self.intent is None

    @property
    def has_conflict(self) -> bool:
        return len(self.votes) > 1

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "security_sensitive": self.security_sensitive,
            "context_sufficient": self.context_sufficient,
            "votes": list(self.votes),
            "is_weak": True,
            "provenance": WEAK_PROVENANCE,
        }


@dataclass(frozen=True, slots=True)
class WeakLabelStats:
    """Health of a weak-labelling run. Low coverage or high conflict invalidates conclusions."""

    total: int
    covered: int
    abstained: int
    conflicts: int
    intent_counts: dict[str, int] = field(default_factory=dict)
    security_count: int = 0
    context_insufficient_count: int = 0

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0

    @property
    def conflict_rate(self) -> float:
        return self.conflicts / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "covered": self.covered,
            "abstained": self.abstained,
            "coverage": round(self.coverage, 4),
            "conflicts": self.conflicts,
            "conflict_rate": round(self.conflict_rate, 4),
            "intent_counts": dict(sorted(self.intent_counts.items())),
            "security_count": self.security_count,
            "context_insufficient_count": self.context_insufficient_count,
            "warning": WEAK_PROVENANCE,
        }


def _has_sufficient_context(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _ACKNOWLEDGEMENT_RE.match(stripped):
        return False
    if _VERSION_ONLY_RE.match(stripped):
        return False
    if _DEVICE_ONLY_RE.match(stripped):
        return False
    return len(re.findall(r"[A-Za-z']+", stripped)) >= MIN_CONTENT_WORDS


def weak_label(text: str) -> WeakLabel:
    """Apply the labelling functions to one message.

    Raises:
        TypeError: if ``text`` is not a string, so a ``None`` fails here rather than becoming
            the string "None" and matching nothing.
    """
    if not isinstance(text, str):
        raise TypeError(f"weak_label expects str, got {type(text).__name__}")

    security = bool(_SECURITY_RE.search(text))
    sufficient = _has_sufficient_context(text)

    # Context is decided first: without enough information to identify a request, keyword luck
    # must not manufacture an intent. "iPhone 7" mentions a device but asks nothing.
    if not sufficient:
        return WeakLabel(
            intent=None,
            security_sensitive=security,
            context_sufficient=False,
            votes=(),
        )

    votes = tuple(name for name, pattern in _COMPILED if pattern.search(text))
    # Priority order is the declaration order of _INTENT_PATTERNS, so the first vote wins.
    intent = votes[0] if votes else None

    return WeakLabel(
        intent=intent,
        security_sensitive=security,
        context_sufficient=True,
        votes=votes,
    )


def label_batch(texts: list[str]) -> tuple[list[WeakLabel], WeakLabelStats]:
    """Label a batch and report its health alongside the labels."""
    labels = [weak_label(t) for t in texts]
    counts = Counter(label.intent for label in labels if label.intent)
    return labels, WeakLabelStats(
        total=len(labels),
        covered=sum(1 for label in labels if label.intent is not None),
        abstained=sum(1 for label in labels if label.abstained),
        conflicts=sum(1 for label in labels if label.has_conflict),
        intent_counts=dict(counts),
        security_count=sum(1 for label in labels if label.security_sensitive),
        context_insufficient_count=sum(1 for label in labels if not label.context_sufficient),
    )

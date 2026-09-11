"""Evidence *relevance*, which is not evidence *groundedness*.

Milestone 5 produced a reply that was faithful to its evidence and useless to the customer:

    Customer: "the problem started with iOS 11.0.2"
    Agent:    "We have released 11.0.2. Make sure you have a backup and get your iPhone
               updated."

Grounding passed, correctly — every claim came from a retrieved resolution. The evidence was
simply the wrong evidence. Those are different failures and they need different names:

* **Groundedness** — did the reply derive its claims from the supplied evidence?
  ``agent/grounding.py``.
* **Relevance** — was that evidence appropriate for *this* customer's problem? Here.

**This module is NOT validated.** Whether retrieved evidence genuinely fits a customer's
problem is a judgement, and no human labels exist yet. What is implemented is the contract
plus two deterministic contradiction checks — cases where the evidence is *demonstrably* at
odds with what the customer said, not cases where it merely seems weak. Everything else
returns ``UNKNOWN``, which is honest rather than useless: it says the question was asked and
not answered.

Deliberately **not** tuned to the iOS 11.0.2 example that exposed the problem. Both checks are
written against the general shape — a version the customer names as the source of their
trouble, and a remedy they have already tried — so the 11.0.2 case is one instance rather than
the target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from hiver_support.agent.retrieval import EvidenceCase


class RelevanceSignal(Enum):
    """Closed set. ``UNKNOWN`` is a real answer: the question was asked and not settled."""

    OK = "ok"
    VERSION_CONTRADICTION = "version_contradiction"
    ALREADY_ATTEMPTED = "already_attempted"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RelevanceReport:
    """What the deterministic checks could establish about evidence fit.

    ``contradicted`` is the only load-bearing field. ``relevant`` is deliberately absent:
    asserting evidence *is* relevant would be a claim this module cannot support without
    human labels.
    """

    contradicted: bool
    signals: tuple[RelevanceSignal, ...] = ()
    details: tuple[str, ...] = ()
    # Stated in the artifact so no reader mistakes a deterministic check for a measurement.
    validated: bool = False

    def to_dict(self) -> dict:
        return {
            "contradicted": self.contradicted,
            "signals": [s.value for s in self.signals],
            "details": list(self.details),
            "validated": self.validated,
            "note": (
                "Deterministic contradiction checks only. Evidence relevance is NOT validated "
                "and cannot be until human labels exist; absence of contradiction is not "
                "evidence of relevance."
            ),
        }


# A version number in any product's scheme: 11, 11.0, 11.0.2, 2.14.3.
_VERSION_RE = re.compile(r"\b\d+(?:\.\d+){1,3}\b")

# The customer naming a version as the ORIGIN of the problem, rather than merely mentioning
# it. "since 11.0.2" and "started with 11.0.2" are complaints; "how do I get 11.0.2" is not.
_VERSION_BLAMED_RE = re.compile(
    r"(?:since|after|with|on|following|started\s+(?:with|after|on)|ever\s+since)\s+"
    r"(?:the\s+)?(?:ios\s*|version\s*|update\s*(?:to\s*)?)?v?(\d+(?:\.\d+){1,3})",
    re.IGNORECASE,
)

# Evidence telling the customer to move TO a version.
_UPDATE_TO_RE = re.compile(
    r"(?:update|upgrade|install|move|get)\s+(?:\w+\s+){0,3}?(?:to\s+)?"
    r"(?:ios\s*|version\s*)?v?(\d+(?:\.\d+){1,3})",
    re.IGNORECASE,
)

# Remedies a customer commonly reports having already tried, and the evidence commonly
# recommends. Stems, so inflections match.
_REMEDIES = {
    "restart": r"restart\w*|reboot\w*|power\s*cycl\w*|turn\w*\s+(?:it\s+)?off\s+and\s+on",
    "reset network settings": r"reset\w*\s+(?:the\s+)?network\s+settings",
    "reset": r"reset\w*",
    "update": r"updat\w*|upgrad\w*",
    "reinstall": r"reinstall\w*|delete\w*\s+and\s+reinstall\w*",
    "backup": r"back\s*up|backup",
}
# "I already restarted", "I have tried resetting", "I did update".
_ALREADY_RE = re.compile(
    r"\b(?:already|have\s+(?:already\s+)?tried|tried|did|done|keep\s+\w+ing)\b", re.IGNORECASE
)


def _versions_blamed(message: str) -> set[str]:
    return {m.group(1) for m in _VERSION_BLAMED_RE.finditer(message)}


# An instruction to update, with no version attached to the verb. Real support replies say
# "We have released 11.0.2. Make sure you back up and get updated", which names the target in
# one sentence and the instruction in the next.
_UPDATE_IMPERATIVE_RE = re.compile(
    r"\b(?:get\s+updated|update\s+your|please\s+updat\w*|make\s+sure\s+you(?:\s+\w+){0,3}"
    r"\s+updat\w*|upgrad\w*|install\s+it)",
    re.IGNORECASE,
)


def _versions_recommended(text: str) -> set[str]:
    """Versions the evidence is steering the customer towards.

    Explicit targets win. Only when the evidence names no explicit target does a named
    version plus a bare update instruction count, because evidence that says "the bug came in
    11.0.2, update to 11.0.3" names both and means only one.
    """
    explicit = {m.group(1) for m in _UPDATE_TO_RE.finditer(text)}
    if explicit:
        return explicit
    if _UPDATE_IMPERATIVE_RE.search(text):
        return set(_VERSION_RE.findall(text))
    return set()


def _remedies_already_tried(message: str) -> set[str]:
    if not _ALREADY_RE.search(message):
        return set()
    tried = set()
    for name, pattern in _REMEDIES.items():
        if re.search(pattern, message, re.IGNORECASE):
            tried.add(name)
    return tried


def _remedies_offered(text: str) -> set[str]:
    return {
        name for name, pattern in _REMEDIES.items()
        if re.search(pattern, text, re.IGNORECASE)
    }


def assess_relevance(
    message: str, evidence: tuple[EvidenceCase, ...]
) -> RelevanceReport:
    """Check whether the retrieved evidence contradicts what the customer said.

    Returns ``contradicted=True`` only when the evidence is demonstrably at odds with the
    message — never merely because it looks weak. Weak-but-not-contradictory evidence is
    reported as ``UNKNOWN``, because calling it irrelevant would be a judgement this module
    cannot make.
    """
    if not isinstance(message, str):
        raise TypeError(f"assess_relevance expects str, got {type(message).__name__}")
    if not evidence:
        return RelevanceReport(False, (RelevanceSignal.UNKNOWN,), ("no evidence supplied",))

    signals: list[RelevanceSignal] = []
    details: list[str] = []
    resolutions = " ".join(c.resolution_text for c in evidence)

    # 1. The customer blames a version and the evidence recommends moving to that same
    #    version. Advising someone onto the release that broke their device is not a weak
    #    answer, it is a contradictory one.
    blamed = _versions_blamed(message)
    recommended = _versions_recommended(resolutions)
    overlap = blamed & recommended
    if overlap:
        signals.append(RelevanceSignal.VERSION_CONTRADICTION)
        details.append(
            f"customer reports the problem beginning with {sorted(overlap)}, and the evidence "
            f"recommends updating to the same version"
        )

    # 2. The customer says they have already done the only thing the evidence suggests.
    tried = _remedies_already_tried(message)
    offered = _remedies_offered(resolutions)
    if tried and offered and offered <= tried:
        signals.append(RelevanceSignal.ALREADY_ATTEMPTED)
        details.append(
            f"the evidence offers only remedies the customer already reports trying: "
            f"{sorted(offered)}"
        )

    if not signals:
        # No contradiction found. Deliberately not "relevant": this check cannot establish fit.
        return RelevanceReport(False, (RelevanceSignal.UNKNOWN,), ())
    return RelevanceReport(True, tuple(signals), tuple(details))

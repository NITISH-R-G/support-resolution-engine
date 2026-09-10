"""Deterministic grounding validation.

The safety-critical check: does this reply assert anything the retrieved evidence does not
support? It runs **before** any model is consulted and can fail a reply on its own, because a
deterministic check that needs no model is cheaper, faster and far more auditable than asking
one model whether another hallucinated.

The strongest rule here follows from a fact about this system rather than from heuristics:
**the agent cannot act.** It has no ability to issue a refund, cancel an order, reset an
account or dispatch a replacement. Any sentence claiming it has done so is false by
construction, no matter how plausible it reads or how good the retrieved evidence is. Those
claims are therefore fatal rather than merely suspicious.

Both failure directions matter. Missing a fabrication lets the agent lie to a customer.
Rejecting a faithful paraphrase pushes safe traffic to humans and makes the system pointless,
so the specific-value checks compare against evidence content rather than banning specifics
outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from hiver_support.agent.retrieval import EvidenceCase


class ViolationType(str, Enum):
    FABRICATED_ACTION = "fabricated_action"
    UNSUPPORTED_SPECIFIC = "unsupported_specific"
    INVENTED_POLICY = "invented_policy"
    NO_EVIDENCE = "no_evidence"
    EMPTY_REPLY = "empty_reply"


@dataclass(frozen=True, slots=True)
class Violation:
    type: ViolationType
    detail: str


@dataclass(frozen=True, slots=True)
class GroundingReport:
    passed: bool
    violations: tuple[Violation, ...] = ()

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": [{"type": v.type.value, "detail": v.detail} for v in self.violations],
        }


# First-person claims that an action has been performed. The agent performs none.
_ACTION_RE = re.compile(
    r"\b("
    # An optional adverb is allowed between auxiliary and verb: "I've *also* refunded you",
    # "we have *now* reset it". Without it the fatal check is trivially evaded by one word.
    r"(i|we)\s*(have|'ve|has)\s+(?:just|also|now|already|gone ahead and\s*)?\s*"
    r"(issued|refunded|reset|cancelled|canceled|applied|credited|dispatched|sent|shipped|"
    r"escalated|updated|removed|deleted|activated|deactivated|replaced|processed|arranged)"
    r"|(i|we)\s+(?:just|also|now|already)?\s*"
    r"(issued|refunded|reset|cancelled|canceled|escalated|replaced|processed)"
    r"|your\s+(refund|replacement|order|account|password|repair|case)\s+"
    r"(has been|is being|was)\s+"
    r"(issued|processed|reset|cancelled|canceled|dispatched|sent|shipped|updated|replaced)"
    r")\b",
    re.IGNORECASE,
)

# Assertions about entitlement or company policy. These require a source; the agent has none
# beyond retrieved evidence, so they must appear there.
_POLICY_RE = re.compile(
    r"\b(our policy|company policy|under our terms|you are entitled|guarantee[sd]?|"
    r"always (replaces?|refunds?|covers?)|never (charges?|refuses?)|"
    r"free of charge|at no cost|covered under warranty)\b",
    re.IGNORECASE,
)

# Specific values a reply might invent: bare numbers, money, percentages, durations.
_SPECIFIC_RE = re.compile(
    r"(\$\s?\d[\d,]*(?:\.\d+)?|£\s?\d[\d,]*(?:\.\d+)?|\b\d+\s?%|"
    r"\b\d+\s*(?:hours?|days?|weeks?|months?|business days?)\b|\b\d[\d,]*(?:\.\d+)?\b)",
    re.IGNORECASE,
)

MIN_SUBSTANTIVE_CHARS = 10


def _evidence_text(evidence: tuple[EvidenceCase, ...]) -> str:
    return " ".join(case.resolution_text for case in evidence).lower()


def _digits(text: str) -> set[str]:
    """Bare digit runs, so '80%' in evidence supports 'below 80' in a reply."""
    return set(re.findall(r"\d[\d,]*(?:\.\d+)?", text))


def validate_grounding(reply: str, evidence: tuple[EvidenceCase, ...]) -> GroundingReport:
    """Check a generated reply against the evidence it claims to rest on.

    Raises:
        TypeError: if ``reply`` is not a string.
    """
    if not isinstance(reply, str):
        raise TypeError(f"validate_grounding expects str, got {type(reply).__name__}")

    violations: list[Violation] = []
    stripped = reply.strip()

    if not stripped:
        violations.append(Violation(ViolationType.EMPTY_REPLY, "reply is empty"))

    # No evidence means nothing may be asserted. Checked before the content rules so the
    # failure reason is the real one rather than an incidental specific-value mismatch.
    if not evidence and len(stripped) >= MIN_SUBSTANTIVE_CHARS:
        violations.append(
            Violation(
                ViolationType.NO_EVIDENCE,
                "no retrieved evidence, so no substantive claim can be grounded",
            )
        )

    corpus = _evidence_text(evidence)

    action = _ACTION_RE.search(stripped)
    if action:
        violations.append(
            Violation(
                ViolationType.FABRICATED_ACTION,
                f"claims an action the agent cannot perform: {action.group(0)!r}",
            )
        )

    policy = _POLICY_RE.search(stripped)
    if policy and policy.group(0).lower() not in corpus:
        violations.append(
            Violation(
                ViolationType.INVENTED_POLICY,
                f"asserts a policy absent from evidence: {policy.group(0)!r}",
            )
        )

    if evidence:
        supported = _digits(corpus)
        for match in _SPECIFIC_RE.finditer(stripped):
            found = _digits(match.group(0))
            if found and not found <= supported:
                violations.append(
                    Violation(
                        ViolationType.UNSUPPORTED_SPECIFIC,
                        f"specific value not present in evidence: {match.group(0)!r}",
                    )
                )
                break  # one is enough to fail; listing every number adds noise, not signal

    return GroundingReport(passed=not violations, violations=tuple(violations))

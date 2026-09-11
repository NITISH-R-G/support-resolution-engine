"""Post-generation policy validation, independent of whatever wrote the reply.

Grounding and policy answer different questions, and Milestone 6 measured why both are needed.
On a real run, gpt-oss-120b produced:

    "We'd love to help with the update issues you're seeing. Please DM us with your iPhone
     model and iOS version so we can dive in."

That reply invents nothing, so grounding passes it. It is also **automated deflection** — the
single outcome the entire retrieval corpus was filtered to prevent, and the thing that makes
an automated support agent worse than useless. The corpus filter stops the agent *retrieving*
a deflection; only a check on the output stops it *composing* one.

So: grounding asks *"is this derived from the evidence?"*, policy asks *"is this allowed at
all?"*. A reply can pass either and fail the other.

Two properties are structural rather than conventional:

**No model is consulted.** There is no parameter to pass one into. A generator asked to judge
its own reply produces a more persuasive version of the same reply.

**A fatal violation escalates.** Nothing here rewrites or repairs a reply. Repairing would
make the output partly ours while still attributing it to the model, and would hide the very
failure the smoke test exists to surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from hiver_support.agent.retrieval import _DEFLECTION_TAIL_RE, EvidenceCase


class PolicyViolationType(Enum):
    """Closed set, so every refusal is machine-readable and countable in evaluation."""

    GENERATED_DEFLECTION = "generated_deflection"
    UNSUPPORTED_REFUND = "unsupported_refund"
    UNSUPPORTED_ACCOUNT_ACTION = "unsupported_account_action"
    ACTION_ALREADY_TAKEN = "action_already_taken"
    UNSUPPORTED_URL = "unsupported_url"
    UNSUPPORTED_PROMISE = "unsupported_promise"
    SECURITY_ADVICE = "security_advice"
    UNSUPPORTED_INSTRUCTION = "unsupported_instruction"


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    violation: PolicyViolationType
    detail: str
    fatal: bool = True

    def to_dict(self) -> dict:
        return {
            "violation": self.violation.value,
            "detail": self.detail,
            "fatal": self.fatal,
        }


@dataclass(frozen=True, slots=True)
class PolicyReport:
    violations: tuple[PolicyViolation, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def fatal(self) -> bool:
        return any(v.fatal for v in self.violations)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "fatal": self.fatal,
            "violations": [v.to_dict() for v in self.violations],
        }


# --------------------------------------------------------------------------- patterns

# A first-person claim that an action has already been carried out. The agent cannot act, so
# any of these is false by construction however plausible the sentence reads.
_DONE = r"(?:i|we)\s*(?:'ve|'ll|\s+have|\s+has)?\s*(?:just\s+|already\s+|now\s+)?"
_REFUND_DONE_RE = re.compile(
    rf"\b(?:{_DONE}(?:issued|processed|sent|given|arranged|applied)\s+"
    rf"(?:you\s+)?(?:a\s+|the\s+|your\s+)?refund"
    rf"|{_DONE}refunded"
    rf"|(?:your|the)\s+refund\s+(?:has\s+been|is)\s+(?:issued|processed|sent|complete))",
    re.IGNORECASE,
)
_ACCOUNT_ACTION_DONE_RE = re.compile(
    rf"\b(?:{_DONE}(?:reset|unlocked|restored|cancelled|canceled|closed|deleted|"
    rf"dispatched|shipped|replaced|upgraded|activated|deactivated)\s+"
    rf"(?:your|the|a)\s+\w+"
    rf"|(?:your|the)\s+(?:account|subscription|order|password|replacement|device)\s+"
    rf"(?:has\s+been|was)\s+(?:reset|unlocked|cancelled|canceled|closed|dispatched|"
    rf"shipped|replaced|restored|activated))",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+|\[URL\]", re.IGNORECASE)

_PROMISE_RE = re.compile(
    r"\b(?:you\s+will\s+receive|we\s+will\s+(?:call|contact|send|ship|replace|refund)"
    r"|guarantee(?:d|s)?|we\s+promise|will\s+be\s+(?:resolved|fixed|delivered)\s+by"
    r"|within\s+\d+\s*(?:hour|day|week|business\s+day)s?"
    r"|our\s+(?:engineers?|team)\s+will\s+\w+)",
    re.IGNORECASE,
)

# Advice that only a human should give on a compromised account. Generating it from retrieved
# tweets is how an agent talks a customer through the wrong recovery path.
_SECURITY_ADVICE_RE = re.compile(
    r"\b(?:change\s+(?:your\s+)?password|reset\s+(?:your\s+)?password"
    r"|two[-\s]?factor|2fa|recovery\s+(?:email|key|contact)"
    r"|secure\s+your\s+account|sign\s+out\s+(?:of\s+)?all\s+devices"
    r"|revoke\s+access|report\s+(?:it\s+)?to\s+the\s+police)",
    re.IGNORECASE,
)

# "Settings > General > About" style navigation. Specific enough to be checkable against the
# evidence, unlike free prose.
_NAV_PATH_RE = re.compile(r"\b[A-Z][\w&]*(?:\s*>\s*[\w&][\w&\s]*){1,4}")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def validate_policy(
    reply: str,
    evidence: tuple[EvidenceCase, ...],
    *,
    security_sensitive: bool = False,
) -> PolicyReport:
    """Check a drafted reply against what the agent is permitted to say.

    Args:
        reply: the drafted text, from any generator.
        evidence: the retrieved cases, used only to check whether a specific claim was
            *available* to make. Policy still forbids some things the evidence supports.
        security_sensitive: whether the incoming message was flagged. If it was, a reply
            should not exist at all, and its existence is itself the violation.

    Returns:
        A ``PolicyReport``. A fatal violation means ESCALATE; nothing is ever repaired.
    """
    if not isinstance(reply, str):
        raise TypeError(f"validate_policy expects str, got {type(reply).__name__}")

    violations: list[PolicyViolation] = []
    if not reply.strip():
        # The generator declined. That is the evidence gate's business, not policy's.
        return PolicyReport()

    evidence_text = _normalise(" ".join(c.resolution_text for c in evidence))

    # 1. Deflection. Checked with the same pattern that filters the corpus, so the input
    #    filter and the output filter cannot drift apart.
    if _DEFLECTION_TAIL_RE.search(reply):
        violations.append(
            PolicyViolation(
                PolicyViolationType.GENERATED_DEFLECTION,
                "reply redirects the customer to another channel; automating a deflection "
                "defeats the purpose of the agent, and is not excused by the evidence "
                "deflecting too",
            )
        )

    # 2 and 3. Claimed actions. The agent cannot act, so these are false by construction and
    #    remain violations even when the evidence discusses the same action.
    if _REFUND_DONE_RE.search(reply):
        violations.append(
            PolicyViolation(
                PolicyViolationType.UNSUPPORTED_REFUND,
                "reply claims a refund was issued; the agent cannot move money",
            )
        )
    if _ACCOUNT_ACTION_DONE_RE.search(reply):
        violations.append(
            PolicyViolation(
                PolicyViolationType.UNSUPPORTED_ACCOUNT_ACTION,
                "reply claims an account or order action was carried out; the agent can "
                "only advise",
            )
        )

    # 5. Links. A link the evidence never carried is an invented destination, and a customer
    #    following it is the worst kind of fabrication.
    if _URL_RE.search(reply) and not _URL_RE.search(evidence_text):
        violations.append(
            PolicyViolation(
                PolicyViolationType.UNSUPPORTED_URL,
                "reply offers a link that no retrieved resolution contained",
            )
        )

    # 6. Operational promises. Nothing in a tweet corpus can commit the company to a timeline.
    promise = _PROMISE_RE.search(reply)
    if promise:
        violations.append(
            PolicyViolation(
                PolicyViolationType.UNSUPPORTED_PROMISE,
                f"reply makes an operational promise ({promise.group(0)!r}) that no evidence "
                f"can support",
            )
        )

    # 7. Security. If the message was flagged, the reply should not exist; if it was not,
    #    unsupported recovery advice is still a human's call.
    if security_sensitive:
        violations.append(
            PolicyViolation(
                PolicyViolationType.SECURITY_ADVICE,
                "a reply was drafted for a security-sensitive message; it should have "
                "escalated before generation, so an upstream gate failed",
            )
        )
    else:
        advice = _SECURITY_ADVICE_RE.search(reply)
        if advice and not _SECURITY_ADVICE_RE.search(evidence_text):
            violations.append(
                PolicyViolation(
                    PolicyViolationType.SECURITY_ADVICE,
                    f"reply gives account-security advice ({advice.group(0)!r}) that no "
                    f"retrieved resolution supports",
                )
            )

    # 8. Instructions. Only checked when there is evidence to check against: with nothing
    #    retrieved the agent escalates earlier, and flagging here would double-count a failure
    #    the evidence gate already owns.
    if evidence:
        for path in _NAV_PATH_RE.findall(reply):
            if _normalise(path) not in evidence_text:
                violations.append(
                    PolicyViolation(
                        PolicyViolationType.UNSUPPORTED_INSTRUCTION,
                        f"reply gives a navigation path ({path.strip()!r}) that no retrieved "
                        f"resolution contains",
                    )
                )
                break

    return PolicyReport(tuple(violations))

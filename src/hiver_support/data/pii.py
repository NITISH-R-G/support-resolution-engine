"""Mask personally identifying information before text leaves the machine.

This runs at the boundary to any third-party LLM API. The corpus is only partly
anonymised: author ids are numeric, but customers paste emails, phone numbers, order
references and card digits into the message body constantly.

Patterns are deliberately conservative. Over-masking is not the safe direction it looks
like -- "iPhone 7", "$9.99" and "2 weeks" are the ordinary signal a support classifier
runs on, and a masker that eats them degrades the system while looking cautious. Each
pattern below is anchored tightly enough to leave that text alone, and
``TestDoesNotOverMask`` pins the behaviour.

Order of application matters: a full card number would otherwise be caught by the phone
pattern and mislabelled, so the more specific patterns run first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class PIIKind(str, Enum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CARD = "CARD"
    ORDER_ID = "ORDER_ID"
    TRACKING = "TRACKING"


@dataclass(frozen=True, slots=True)
class PIIReport:
    """Masked text plus what was found, so masking is auditable rather than invisible."""

    text: str
    kinds_found: frozenset[PIIKind]

    @property
    def had_pii(self) -> bool:
        return bool(self.kinds_found)


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Amazon-style 3-7-7, and hash-prefixed references that contain at least one digit so
# ordinary uppercase hashtags ("#AMAZON") are not mistaken for order ids.
_ORDER_ID_RE = re.compile(r"\b\d{3}-\d{7}-\d{7}\b|#(?=[A-Z0-9]*\d)[A-Z0-9]{6,}\b")

# UPS (1Z + 16) and USPS-style (2 letters, 9 digits, 2 letters).
_TRACKING_RE = re.compile(r"\b1Z[0-9A-Z]{16}\b|\b[A-Z]{2}\d{9}[A-Z]{2}\b")

_CARD_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")

# A candidate phone-like run; the digit-count check below decides whether it really is one.
_PHONE_RE = re.compile(
    r"(?<![\d/.-])"
    r"(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)|\d{2,4})"
    r"(?:[\s.-]?\d{2,4}){2,4}"
    r"(?![\d/.-])"
)

_PHONE_MIN_DIGITS = 7
_PHONE_MAX_DIGITS = 15


def _mask_phone(match: re.Match[str]) -> str:
    """Accept a candidate only if its digit count is plausible for a real number.

    This is what keeps "iOS 11.0.1" and "Oct 31 2017" out of the phone bucket without
    hand-listing exceptions.
    """
    digits = sum(character.isdigit() for character in match.group())
    if _PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS:
        return "[PHONE]"
    return match.group()


def mask_pii(text: str) -> PIIReport:
    """Replace PII with typed placeholders and report which kinds were present.

    Idempotent: the placeholders it emits match none of its own patterns, so masking
    already-masked text is a no-op.

    Raises:
        TypeError: if ``text`` is not a string. A ``None`` slipping through to an API call
            and serialising as the string "None" is a silent data-quality bug, so the
            boundary fails loudly instead.
    """
    if not isinstance(text, str):
        raise TypeError(f"mask_pii expects str, got {type(text).__name__}")

    found: set[PIIKind] = set()

    def substitute(pattern: re.Pattern[str], replacement, kind: PIIKind, value: str) -> str:
        masked, count = pattern.subn(replacement, value)
        if count and masked != value:
            found.add(kind)
        return masked

    result = substitute(_EMAIL_RE, "[EMAIL]", PIIKind.EMAIL, text)
    result = substitute(_ORDER_ID_RE, "[ORDER_ID]", PIIKind.ORDER_ID, result)
    result = substitute(_TRACKING_RE, "[TRACKING]", PIIKind.TRACKING, result)
    result = substitute(_CARD_RE, "[CARD]", PIIKind.CARD, result)
    result = substitute(_PHONE_RE, _mask_phone, PIIKind.PHONE, result)

    return PIIReport(text=result, kinds_found=frozenset(found))

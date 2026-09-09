"""Normalise customer and support text for the pipeline.

The guiding rule is that normalisation removes *noise* and preserves *signal*. URLs, foreign
handles, HTML entities and encoding artefacts are noise: they vary arbitrarily between
otherwise identical messages and give the model nothing to learn from. Casing, punctuation,
emphasis, hashtags and emoji are signal -- "THIS IS UNACCEPTABLE 😤" and "this is
unacceptable" call for different escalation decisions, and lowercasing the corpus (the
reflexive default) destroys exactly the feature the routing policy needs.

Contrast ``leakage.normalise_for_comparison``, which is deliberately lossy because it only
ever compares two strings for equality and never feeds a model.
"""

from __future__ import annotations

import html
import re
import unicodedata

_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@(\w+)")
_ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿]")
_WHITESPACE_RE = re.compile(r"\s+")

URL_PLACEHOLDER = "[URL]"
USER_PLACEHOLDER = "[USER]"


def normalise_text(text: str, brand: str | None = None) -> str:
    """Clean text while preserving the features the downstream models rely on.

    Args:
        text: Raw message text.
        brand: The brand handle to preserve, without the ``@``. Every *other* mention is
            replaced, because a foreign handle is noise while the brand handle is the
            addressee and tells the model who is being spoken to.

    Returns:
        Normalised text. Idempotent: the placeholders emitted contain no ``@`` or URL
        pattern, so a second pass is a no-op.

    Raises:
        TypeError: if ``text`` is not a string, so a ``None`` fails loudly here rather than
            reaching a model serialised as the string "None".
    """
    if not isinstance(text, str):
        raise TypeError(f"normalise_text expects str, got {type(text).__name__}")

    # Entities first: an encoded "&amp;" must become "&" before anything else inspects it.
    result = html.unescape(text)

    # NFKC folds fullwidth forms and ligatures onto their ASCII equivalents. It does not
    # touch emoji, which have no compatibility decomposition.
    result = unicodedata.normalize("NFKC", result)
    result = _ZERO_WIDTH_RE.sub("", result)

    result = _URL_RE.sub(URL_PLACEHOLDER, result)

    if brand is None:
        result = _MENTION_RE.sub(USER_PLACEHOLDER, result)
    else:
        brand_lower = brand.lower()
        result = _MENTION_RE.sub(
            lambda match: match.group(0)
            if match.group(1).lower() == brand_lower
            else USER_PLACEHOLDER,
            result,
        )

    return _WHITESPACE_RE.sub(" ", result).strip()

"""Classify a brand reply as deflection, substantive resolution, or actionable guidance.

These rates decide which brand the project is built on, so the logic lives here with tests
against verbatim corpus examples rather than as regexes inside an analysis script.

The distinction that matters is **what the reply does**, not which phrases it contains. A
deflection's entire function is to move the conversation elsewhere ("please DM us", "contact
us here: [URL]"), leaving nothing a generated reply could be grounded in. A substantive reply
carries information about the customer's problem — even if it *also* invites the customer to
get in touch. That overlap is the hard case, and matching redirect phrases alone gets it
wrong in both directions:

* too narrow (the original defect): "Please contact us directly here: [URL]" scored as a
  resolution, because the lexicon only knew about "DM";
* too broad (the obvious over-correction): "have you had a chance to contact us? If the
  charges are pending, they could be authorisations..." discarded as a deflection, throwing
  away real grounding evidence.

So a redirect phrase only *makes* a reply a deflection when the reply carries no information
of its own — judged by whether it says anything beyond the redirect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Phrases whose job is to move the conversation to another channel.
_REDIRECT_RE = re.compile(
    r"\b("
    r"dm|d\.m\.|direct message|private message|pm us|"
    r"message us|send us a (dm|message|note)|shoot us a (dm|message)|"
    r"contact us|reach us|reach out to us|get in touch (with us )?(here|via|at)|"
    r"write (back )?to (the email|us)|request a callback|call us|email us|"
    r"visit for more info|use this link|click the link|follow (us )?and dm|"
    r"check your (dm|messages)|respond there|connect with you there|reply there"
    r")\b",
    re.IGNORECASE,
)

# Acknowledgements that a DM channel is already in play. Always deflection: they contain no
# resolution at all, whatever their length.
_DM_ACKNOWLEDGEMENT_RE = re.compile(
    r"\b(we('ve| have)? (received|got|see)n? your (dm|message)|"
    r"we('ve| have) (followed up|replied|responded) (via|in) dm|"
    r"check your (dm|messages)|expect a response there|respond there)\b",
    re.IGNORECASE,
)

# Markers of concrete instruction the customer can act on themselves.
#
# Verb stems take a \w* suffix rather than a closing \b. Support replies overwhelmingly use
# continuous and past forms — "restarting", "updating", "resetting" — and a stem-only pattern
# matched none of them, silently under-counting actionable guidance.
_ACTIONABLE_RE = re.compile(
    r"\b(go to|head to|tap\w*|click\w*|select\w*|choos\w*|open\w*|navigat\w*|settings|"
    r"restart\w*|reboot\w*|reinstall\w*|uninstall\w*|toggl\w*|enabl\w*|disabl\w*|"
    r"updat\w*|upgrad\w*|clear\w*( your)? cache|log ?out|sign ?out|reset\w*|hold down|"
    r"press\w*|swip\w*|check\w*|try|tried|"
    r"here's how|follow these steps|step 1|make sure (you|to)|you'll need to)\b",
    re.IGNORECASE,
)

# Markers that a reply states something about the problem: facts, conditions, explanations,
# or a specific question that advances diagnosis.
_INFORMATIVE_RE = re.compile(
    r"\b(is only|are only|is not|isn't|does not|doesn't|will not|won't|"
    r"because|since|due to|which means|this means|if (you|the|they)|when (you|the)|"
    r"available|supported|unavailable|currently|typically|usually|normally|"
    r"within \d|\d+ (hours?|days?|weeks?|business days?)|"
    r"error|version|model|account|order|refund|charge|"
    r"what (version|error|device|model)|which (version|device|model)|"
    r"can you tell us|could you (confirm|tell|let us know)|are you (getting|seeing|able)|"
    r"tell us more|please confirm"
    r")\b",
    re.IGNORECASE,
)

# Purely social replies. Present to keep them out of the resolution count.
_PLEASANTRY_RE = re.compile(
    r"^\W*(you'?re (very )?welcome|thanks?( you)? for (letting us know|the update)|"
    r"glad to (know|hear)|happy to hear|have a (great|good|nice)|no problem|anytime)\b",
    re.IGNORECASE,
)

_SIGNATURE_RE = re.compile(r"\^\w{1,4}\s*$")
_PLACEHOLDER_RE = re.compile(r"\[(URL|USER|EMAIL|PHONE|CARD|ORDER_ID|TRACKING)\]")

# Function words that are near-universal in English support replies. Absence across a whole
# reply is a strong signal of another language; the corpus contains German, Spanish and
# Portuguese replies that would otherwise be counted as resolutions.
_ENGLISH_MARKERS = frozenset(
    """the be to of and a in that have it for not on with as you do at this but his by from
    we they is are was were please your our can will would could should us thank sorry help
    know let more about here there when what which how if so""".split()
)

MIN_SUBSTANTIVE_CHARS = 60
MIN_ENGLISH_MARKER_RATIO = 0.12

# Navigation instructions are mostly nouns ("Settings > General > Reset > Reset Network
# Settings") and carry very few function words, so the ratio test above rejected them as
# non-English. Because `is_substantive` requires English, that discarded precisely the most
# actionable replies in the corpus — a bias in the one direction most damaging to any measure
# of resolution quality. These words are unambiguous English support vocabulary.
_ENGLISH_STRONG_MARKERS = re.compile(
    r"\b(settings|general|please|thanks?|sorry|reset|restart|tap|click|download|install|"
    r"iphone|ipad|battery|account|version|device|help|check|network|software|capacity|"
    r"maximum|health)\b",
    re.IGNORECASE,
)
MIN_STRONG_MARKERS = 2


@dataclass(frozen=True, slots=True)
class ReplyClassification:
    is_deflection: bool
    is_substantive: bool
    is_actionable: bool
    is_english: bool


def _strip_furniture(text: str) -> str:
    """Remove agent signatures and placeholders so length reflects actual content."""
    stripped = _SIGNATURE_RE.sub("", text)
    stripped = _PLACEHOLDER_RE.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def is_probably_english(text: str) -> bool:
    """Cheap, deterministic language check.

    A word-list ratio rather than a language-detection dependency: it needs to be fast enough
    for millions of replies, deterministic for reproducibility, and only accurate enough to
    keep obvious non-English out of the resolution counts.
    """
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 4:
        return True  # too short to judge; filtered on length elsewhere
    hits = sum(1 for word in words if word in _ENGLISH_MARKERS)
    if (hits / len(words)) >= MIN_ENGLISH_MARKER_RATIO:
        return True
    # Fall back to unambiguous English support vocabulary before rejecting. Two *distinct*
    # markers are required so a single loanword ("update", "iPhone") inside a Spanish or
    # German sentence cannot smuggle it through.
    return len(set(_ENGLISH_STRONG_MARKERS.findall(text.lower()))) >= MIN_STRONG_MARKERS


def classify_reply(text: str) -> ReplyClassification:
    """Classify a single brand reply.

    Raises:
        TypeError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"classify_reply expects str, got {type(text).__name__}")

    content = _strip_furniture(text)
    english = is_probably_english(content)

    if not content:
        return ReplyClassification(False, False, False, english)

    informative = bool(_INFORMATIVE_RE.search(content))
    actionable_marker = bool(_ACTIONABLE_RE.search(content))

    # A bare acknowledgement of a DM thread is always a deflection: there is no resolution in
    # it to weigh against the redirect.
    if _DM_ACKNOWLEDGEMENT_RE.search(content):
        deflection = True
    elif _REDIRECT_RE.search(content):
        # Redirect plus real content is a resolution that happens to invite contact; redirect
        # alone is a deflection.
        deflection = not (informative or actionable_marker)
    else:
        deflection = False

    pleasantry = bool(_PLEASANTRY_RE.match(content))

    substantive = (
        not deflection
        and not pleasantry
        and english
        and len(content) >= MIN_SUBSTANTIVE_CHARS
        and (informative or actionable_marker)
    )

    return ReplyClassification(
        is_deflection=deflection,
        is_substantive=substantive,
        is_actionable=substantive and actionable_marker,
        is_english=english,
    )

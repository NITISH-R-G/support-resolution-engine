"""Reply generation, grounded in retrieved historical resolutions.

Two generators share one interface so the agent runs today and improves later:

* ``EvidenceTemplateGenerator`` — deterministic, no model, no cost. It adapts the highest
  scoring retrieved resolution rather than composing new prose, so it is grounded by
  construction. This is what makes the whole agent runnable end-to-end **right now**, with no
  API key, and it doubles as the honest baseline any LLM must beat.
* ``LLMReplyGenerator`` — the same interface over a provider. Its prompt is versioned and its
  responses cached, so a prompt change misses the cache instead of silently reusing a reply
  written under different instructions.

Both are validated by the same deterministic grounding check afterwards. The template
generator is not exempt: quoting evidence badly can still produce an ungrounded reply, and a
generator that policed itself would be marking its own homework.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hiver_support.agent.llm import LLMProvider, NullProvider
from hiver_support.agent.retrieval import EvidenceCase

MAX_REPLY_CHARS = 480


@dataclass(frozen=True, slots=True)
class GeneratedReply:
    """A draft reply plus the evidence it was built from."""

    text: str
    cited_case_ids: tuple[str, ...]
    generator_name: str
    generator_version: str
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "cited_case_ids": list(self.cited_case_ids),
            "generator": self.generator_name,
            "generator_version": self.generator_version,
            "from_cache": self.from_cache,
        }


class ReplyGenerator:
    """Interface for anything that drafts a reply from evidence."""

    name: str = "base"
    version: str = "0.0.0"

    def generate(
        self, message: str, evidence: tuple[EvidenceCase, ...], intent: str
    ) -> GeneratedReply:
        raise NotImplementedError


_SIGNATURE_RE = re.compile(r"\s*\^\w{1,4}\s*$")
_PLACEHOLDER_RE = re.compile(r"\[(URL|USER|EMAIL|PHONE|CARD|ORDER_ID|TRACKING)\]")


class EvidenceTemplateGenerator(ReplyGenerator):
    """Adapts the best retrieved resolution into a reply. Deterministic, free, always available.

    Deliberately conservative: it reuses the retrieved resolution's own wording rather than
    paraphrasing. Paraphrase is where fabrication enters, and there is no model here to judge
    whether a rewrite preserved meaning. The cost is a stilted reply; the benefit is that
    grounding is structural rather than hoped for.
    """

    name = "evidence_template"
    version = "1.0.0"

    def generate(
        self, message: str, evidence: tuple[EvidenceCase, ...], intent: str
    ) -> GeneratedReply:
        if not evidence:
            # No evidence means no reply. The agent escalates; it does not improvise.
            return GeneratedReply("", (), self.name, self.version)

        best = evidence[0]
        resolution = _SIGNATURE_RE.sub("", best.resolution_text).strip()
        resolution = _PLACEHOLDER_RE.sub("", resolution).strip()
        resolution = re.sub(r"\s+", " ", resolution)

        text = f"Thanks for reaching out. {resolution}"
        if len(text) > MAX_REPLY_CHARS:
            text = text[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + "..."

        return GeneratedReply(
            text=text,
            cited_case_ids=(best.case_id,),
            generator_name=self.name,
            generator_version=self.version,
        )


PROMPT_VERSION = "v1"

_PROMPT = """You are drafting a reply for {brand} customer support on Twitter.

CUSTOMER MESSAGE:
{message}

CLASSIFIED INTENT: {intent}

HISTORICAL RESOLUTIONS FOR SIMILAR ISSUES (your ONLY permitted source of facts):
{evidence}

RULES — these are absolute:
1. Use ONLY information present in the historical resolutions above.
2. Never claim you have performed an action. You cannot issue refunds, reset accounts,
   cancel orders or dispatch replacements. You can only advise.
3. Never invent policies, timelines, prices, percentages or entitlements.
4. If the resolutions do not address the customer's problem, reply with exactly:
   INSUFFICIENT_EVIDENCE
5. Keep it under 280 characters, in a warm, plain support tone.

Reply with the draft text only, no preamble."""


class LLMReplyGenerator(ReplyGenerator):
    """Generates via an LLM provider, constrained to the retrieved evidence.

    The prompt forbids invented actions and policies, but that is a request, not a guarantee —
    the deterministic grounding validator still runs afterwards and can reject the result. The
    model is treated as untrusted, which is the only safe assumption.
    """

    name = "llm"
    version = "1.0.0"

    def __init__(self, provider: LLMProvider | None = None, brand: str = "AppleSupport") -> None:
        self.provider = provider or NullProvider()
        self.brand = brand
        self.name = f"llm:{self.provider.name}:{self.provider.model}"

    def build_prompt(
        self, message: str, evidence: tuple[EvidenceCase, ...], intent: str
    ) -> str:
        rendered = "\n\n".join(
            f"[case {case.case_id}]\nCustomer asked: {case.customer_text}\n"
            f"Support replied: {case.resolution_text}"
            for case in evidence
        ) or "(none retrieved)"
        return _PROMPT.format(
            brand=self.brand, message=message, intent=intent, evidence=rendered
        )

    def generate(
        self, message: str, evidence: tuple[EvidenceCase, ...], intent: str
    ) -> GeneratedReply:
        if not evidence:
            return GeneratedReply("", (), self.name, self.version)

        text = self.provider.complete(self.build_prompt(message, evidence, intent)).strip()
        # The model's own escape hatch. Treated as an empty draft so the agent escalates
        # through the same path as any other missing-evidence case.
        if text.upper().startswith("INSUFFICIENT_EVIDENCE"):
            text = ""
        return GeneratedReply(
            text=text,
            cited_case_ids=tuple(c.case_id for c in evidence),
            generator_name=self.name,
            generator_version=self.version,
            from_cache=getattr(self.provider, "hits", 0) > 0,
        )

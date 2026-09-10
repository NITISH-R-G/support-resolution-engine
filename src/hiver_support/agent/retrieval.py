"""Historical-resolution retrieval — the evidence layer the agent grounds replies in.

This is what makes the system a support *resolution* engine rather than a classifier. The
generator may only assert what these cases support, so what comes back here bounds the
truthfulness of everything downstream.

Three design decisions carry most of the weight:

**Only groundable cases are indexed.** A "please DM us" deflection contains no resolution, so
indexing one would let the agent retrieve three deflections and produce a confident reply
grounded in nothing. AppleSupport deflects about a third of the time, so this filter is the
difference between evidence and noise.

**Hybrid rather than either method alone.** BM25 and embeddings fail differently — BM25 misses
paraphrases ("tablet losing internet" vs "wifi disconnecting"), embeddings miss exact product
and error strings ("iOS 11.0.3", "Error 4013"). Support text carries both kinds of signal, so
scores are fused rather than chosen between.

**Temporal filtering is available and used by the agent.** A resolution written after the
question could not have informed a real reply; retrieving one inflates every downstream number
in a way production could never reproduce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from hiver_support.data.normalise import normalise_text
from hiver_support.data.reply_classify import classify_reply
from hiver_support.data.schema import SupportPair

BRAND = "AppleSupport"
DEFAULT_TOP_K = 5
DEFAULT_SEMANTIC_WEIGHT = 0.5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class EvidenceCase:
    """One historical customer/resolution pair the agent may ground a reply in."""

    case_id: str
    customer_text: str
    resolution_text: str
    created_at: datetime
    score: float = 0.0
    retrieval_method: str = "hybrid"

    def with_score(self, score: float, method: str) -> EvidenceCase:
        return EvidenceCase(
            case_id=self.case_id,
            customer_text=self.customer_text,
            resolution_text=self.resolution_text,
            created_at=self.created_at,
            score=score,
            retrieval_method=method,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Retrieved evidence plus the confidence that anything relevant was found.

    ``confidence`` is what lets the agent escalate on thin evidence instead of generating a
    reply from whatever happened to rank first.
    """

    query: str
    cases: tuple[EvidenceCase, ...]
    confidence: float
    method: str = "hybrid"

    @property
    def is_empty(self) -> bool:
        return not self.cases


def build_corpus(pairs: list[SupportPair]) -> tuple[EvidenceCase, ...]:
    """Keep only pairs whose reply actually resolves something.

    Deflections and pleasantries are dropped: they cannot ground a generated reply, and
    including them would let the agent report high retrieval confidence for evidence that
    says nothing.
    """
    cases = []
    for pair in pairs:
        resolution = normalise_text(pair.support_text, brand=BRAND)
        classified = classify_reply(resolution)
        # Actionable is necessary but not sufficient. A reply can contain real steps AND still
        # deflect ("Force restart... DM us if that fails"), and grounding in one produces an
        # auto-handled reply that tells the customer to DM — automated deflection, which
        # defeats the point. Found by inspecting a real run: 11 of 32 auto-handled replies.
        if not classified.is_actionable or classified.is_deflection:
            continue
        if _DEFLECTION_TAIL_RE.search(resolution):
            continue
        cases.append(
            EvidenceCase(
                case_id=pair.pair_id,
                customer_text=normalise_text(pair.customer_text, brand=BRAND),
                resolution_text=resolution,
                created_at=pair.customer_tweet.created_at,
            )
        )
    return tuple(cases)


# `classify_reply` treats a reply as non-deflecting when it carries real content alongside a
# redirect — correct for measuring resolution density, wrong for choosing groundable evidence.
# Any surviving redirect makes the case unusable as the basis of an auto-handled reply.
_DEFLECTION_TAIL_RE = re.compile(
    r"\b(dm us|dm me|send us a (dm|message|note)|direct message|check your dm|"
    r"contact us|reach out to us|message us|dm if|dm for|in dm|via dm)\b",
    re.IGNORECASE,
)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """BM25 plus dense-embedding retrieval over historical resolutions.

    Both scorers are min-max normalised before fusion so neither dominates by scale alone —
    BM25 is unbounded while cosine similarity sits in [-1, 1].
    """

    def __init__(
        self,
        corpus: tuple[EvidenceCase, ...],
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.corpus = corpus
        self.semantic_weight = semantic_weight
        self.embedding_model = embedding_model
        self._fitted = False
        self._encoder = None

    def fit(self) -> HybridRetriever:
        if not self.corpus:
            self._fitted = True
            return self

        from rank_bm25 import BM25Okapi

        # Indexing the customer side, not the resolution: the query is a customer message, so
        # matching question-to-question finds cases about the same problem. Matching a question
        # against answers rewards replies that merely share vocabulary with the question.
        self._documents = [c.customer_text for c in self.corpus]
        self._bm25 = BM25Okapi([_tokenise(d) for d in self._documents])

        if self.semantic_weight > 0:
            self._embeddings = self._encode(self._documents)
        self._fitted = True
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)
        return np.asarray(
            self._encoder.encode(
                texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        span = scores.max() - scores.min()
        if span <= 0:
            return np.zeros_like(scores)
        return (scores - scores.min()) / span

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_ids: set[str] | None = None,
        before: datetime | None = None,
    ) -> RetrievalResult:
        """Retrieve evidence for one customer message.

        Args:
            exclude_ids: case ids to drop, used to stop a message retrieving itself.
            before: keep only cases strictly earlier than this instant.
        """
        if not isinstance(query, str):
            raise TypeError(f"retrieve expects str, got {type(query).__name__}")
        if not self._fitted:
            raise RuntimeError("HybridRetriever must be fit before retrieve")
        if not self.corpus or not query.strip():
            return RetrievalResult(query=query, cases=(), confidence=0.0)

        eligible = np.array(
            [
                (exclude_ids is None or case.case_id not in exclude_ids)
                and (before is None or case.created_at < before)
                for case in self.corpus
            ]
        )
        if not eligible.any():
            return RetrievalResult(query=query, cases=(), confidence=0.0)

        lexical = self._normalise(
            np.asarray(self._bm25.get_scores(_tokenise(query)), dtype=float)
        )
        if self.semantic_weight > 0:
            semantic = self._normalise(self._encode([query])[0] @ self._embeddings.T)
            fused = (1 - self.semantic_weight) * lexical + self.semantic_weight * semantic
            if self.semantic_weight == 1.0:
                method = "semantic"
            elif self.semantic_weight == 0.0:
                method = "bm25"
            else:
                method = "hybrid"
        else:
            fused, method = lexical, "bm25"

        fused = np.where(eligible, fused, -np.inf)
        order = np.argsort(-fused)[:top_k]
        cases = tuple(
            self.corpus[i].with_score(float(fused[i]), method)
            for i in order
            if np.isfinite(fused[i])
        )

        # Confidence is the top fused score. Because fusion is min-max normalised per query,
        # the best in-corpus match tends toward 1.0, so this is a RELATIVE signal for
        # thresholding rather than a probability. docs/AGENT.md states that limitation.
        confidence = float(cases[0].score) if cases else 0.0
        return RetrievalResult(query=query, cases=cases, confidence=confidence, method=method)

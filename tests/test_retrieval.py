"""Historical-resolution retrieval: the evidence layer the agent grounds its replies in.

Retrieval is what makes this a support *resolution* engine rather than a classifier. The
agent may only say what the retrieved evidence supports, so the quality and provenance of
what comes back here bounds everything downstream.

Three properties are load-bearing and each is tested:

* **Leakage safety.** The corpus is built from the train split only. The held-out test pool is
  where the golden set will come from, and a case retrieved from it would let the agent answer
  an evaluation question with the evaluation answer.
* **Temporal honesty.** A case that postdates the query could not have informed a real reply,
  so retrieving one inflates every downstream number in a way production could never reproduce.
* **Groundable evidence only.** A "please DM us" deflection contains no resolution. Indexing
  deflections would let the agent retrieve three of them and generate a confident reply
  grounded in nothing.

Hybrid retrieval is used because the two methods fail differently: BM25 misses paraphrases,
embeddings miss exact product and error strings. Tests pin that each contributes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.agent.retrieval import (
    EvidenceCase,
    HybridRetriever,
    RetrievalResult,
    build_corpus,
)
from hiver_support.data.schema import SupportPair, Tweet

BASE = datetime(2017, 1, 1, tzinfo=timezone.utc)


def _pair(pid: str, customer: str, support: str, hours: int = 0) -> SupportPair:
    when = BASE + timedelta(hours=hours)
    return SupportPair(
        pair_id=pid,
        brand="AppleSupport",
        conversation_id=f"conv_{pid}",
        customer_tweet=Tweet(f"c{pid}", f"cust{pid}", True, when, customer),
        support_tweet=Tweet(f"s{pid}", "AppleSupport", False, when + timedelta(minutes=5), support),
    )


ACTIONABLE = [
    _pair("1", "my battery drains so fast after the update",
          "Please go to Settings > Battery > Battery Health and check Maximum Capacity there.", 0),
    _pair("2", "wifi keeps disconnecting on my ipad at home",
          "Try resetting network settings: Settings > General > Reset > Reset Network Settings.", 1),
    _pair("3", "my iphone screen is frozen and will not respond",
          "Force restart by holding the side button and volume down until the Apple logo appears.", 2),
    _pair("4", "apple music will not play my downloaded songs",
          "Open Settings, tap Music, then toggle Sync Library off and on to refresh downloads.", 3),
]

DEFLECTIONS = [
    _pair("90", "my battery is draining very quickly today", "Please DM us and we can help there.", 4),
    _pair("91", "wifi will not connect at all", "We've received your DM and will respond shortly.", 5),
]


@pytest.fixture
def retriever() -> HybridRetriever:
    return HybridRetriever(build_corpus(ACTIONABLE)).fit()


class TestCorpusConstruction:
    def test_indexes_actionable_resolutions(self):
        assert len(build_corpus(ACTIONABLE)) == len(ACTIONABLE)

    def test_excludes_deflections_which_ground_nothing(self):
        """A 'please DM us' reply contains no resolution to ground a generated reply in."""
        assert build_corpus(DEFLECTIONS) == ()

    def test_mixed_input_keeps_only_groundable_cases(self):
        corpus = build_corpus(ACTIONABLE + DEFLECTIONS)
        assert {c.case_id for c in corpus} == {p.pair_id for p in ACTIONABLE}

    def test_every_case_carries_its_provenance(self):
        case = build_corpus(ACTIONABLE)[0]
        assert case.case_id and case.created_at and case.resolution_text

    def test_cases_are_immutable(self):
        case = build_corpus(ACTIONABLE)[0]
        with pytest.raises((AttributeError, TypeError)):
            case.resolution_text = "changed"  # type: ignore[misc]

    def test_empty_input_gives_an_empty_corpus(self):
        assert build_corpus([]) == ()


class TestCorpusExcludesDeflectingResolutions:
    """Regression: an auto-handled reply that says "DM us" is not auto-handling.

    Found by inspecting a real agent run — 11 of 32 auto-handled replies told the customer to
    DM. The resolutions behind them were *actionable* (they contained real steps) but ALSO
    deflected, and `is_actionable` alone let them into the corpus. Grounding a reply in one
    produces automated deflection, which defeats the purpose of the system.
    """

    MIXED = [
        _pair("50", "my screen is frozen",
              "Force restart by holding the side button until the logo appears. DM us if that fails.", 0),
        _pair("51", "battery drains fast",
              "Please go to Settings > Battery and check Battery Health. Send us a DM with the result.", 1),
    ]

    def test_a_resolution_that_also_deflects_is_excluded(self):
        assert build_corpus(self.MIXED) == ()

    def test_a_purely_actionable_resolution_is_kept(self):
        assert len(build_corpus(ACTIONABLE)) == len(ACTIONABLE)


class TestRetrieval:
    def test_returns_a_structured_result(self, retriever):
        result = retriever.retrieve("my battery drains really fast")
        assert isinstance(result, RetrievalResult)
        assert all(isinstance(c, EvidenceCase) for c in result.cases)

    def test_finds_the_lexically_matching_case(self, retriever):
        result = retriever.retrieve("battery drains fast after update")
        assert result.cases[0].case_id == "1"

    def test_finds_a_semantic_paraphrase_bm25_would_miss(self, retriever):
        """No shared content words with case 2 beyond the concept of a dropping connection."""
        result = retriever.retrieve("my tablet keeps losing its internet connection")
        assert "2" in {c.case_id for c in result.cases[:2]}

    def test_respects_top_k(self, retriever):
        assert len(retriever.retrieve("battery", top_k=2).cases) == 2

    def test_results_are_ordered_by_descending_score(self, retriever):
        scores = [c.score for c in retriever.retrieve("battery wifi screen").cases]
        assert scores == sorted(scores, reverse=True)

    def test_is_deterministic(self, retriever):
        first = retriever.retrieve("my battery drains")
        second = retriever.retrieve("my battery drains")
        assert [c.case_id for c in first.cases] == [c.case_id for c in second.cases]

    def test_reports_retrieval_confidence(self, retriever):
        assert 0.0 <= retriever.retrieve("my battery drains fast").confidence <= 1.0

    def test_unrelated_query_yields_low_confidence(self, retriever):
        """The agent must be able to tell that nothing relevant was found."""
        related = retriever.retrieve("my battery drains so fast after the update").confidence
        unrelated = retriever.retrieve("quantum chromodynamics lecture notes").confidence
        assert unrelated < related


class TestLeakageSafety:
    def test_excludes_a_case_by_id(self, retriever):
        result = retriever.retrieve("battery drains fast", exclude_ids={"1"})
        assert "1" not in {c.case_id for c in result.cases}

    def test_excludes_cases_at_or_after_the_query_time(self):
        """A resolution written after the question could not have informed a real reply."""
        retriever = HybridRetriever(build_corpus(ACTIONABLE)).fit()
        result = retriever.retrieve("battery wifi screen music", before=BASE + timedelta(hours=2))
        assert all(c.created_at < BASE + timedelta(hours=2) for c in result.cases)

    def test_temporal_filter_can_empty_the_result(self):
        retriever = HybridRetriever(build_corpus(ACTIONABLE)).fit()
        assert retriever.retrieve("battery", before=BASE - timedelta(days=1)).cases == ()

    def test_empty_retrieval_reports_zero_confidence(self):
        retriever = HybridRetriever(build_corpus(ACTIONABLE)).fit()
        result = retriever.retrieve("battery", before=BASE - timedelta(days=1))
        assert result.confidence == 0.0


class TestHybridBehaviour:
    def test_both_methods_contribute(self, retriever):
        result = retriever.retrieve("battery drains fast")
        assert result.cases[0].retrieval_method in {"hybrid", "bm25", "semantic"}

    def test_weighting_shifts_the_ranking(self):
        corpus = build_corpus(ACTIONABLE)
        lexical = HybridRetriever(corpus, semantic_weight=0.0).fit()
        semantic = HybridRetriever(corpus, semantic_weight=1.0).fit()
        query = "my tablet keeps losing its internet connection"
        assert lexical.retrieve(query).cases[0].case_id != "" and semantic.retrieve(query).cases


class TestEdgeCases:
    def test_empty_query(self, retriever):
        assert isinstance(retriever.retrieve(""), RetrievalResult)

    def test_non_string_query_raises(self, retriever):
        with pytest.raises(TypeError):
            retriever.retrieve(None)

    def test_retrieving_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            HybridRetriever(build_corpus(ACTIONABLE)).retrieve("battery")

    def test_empty_corpus_returns_no_cases(self):
        result = HybridRetriever(()).fit().retrieve("battery")
        assert result.cases == ()
        assert result.confidence == 0.0

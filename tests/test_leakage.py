"""Leakage guards — every one must be observed to fail on a violating fixture.

A guard that has never been seen to raise is not a guard, it is a comment. So each check
here is tested in both directions: it stays silent on clean data and raises on a fixture
built to violate it.

These exist because the two clearest methodological failures found in the public field were
both leakage of this kind (docs/PUBLIC_REPO_COMPARISON.md §2.2, §2.3): a classifier graded
against labels it generated, and a retrieval metric scoring the reranker's own sort key.
Neither repository had a guard that could have caught it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.data.schema import SupportPair, Tweet
from hiver_support.leakage import (
    LeakageError,
    assert_independent_models,
    assert_no_id_overlap,
    assert_no_near_duplicates,
    assert_no_response_leakage,
    assert_no_text_duplicates,
    assert_temporal_split,
    model_family,
    run_all_checks,
)

BASE = datetime(2017, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


def _tweet(tweet_id: str, author: str, inbound: bool, text: str, offset_hours: int = 0) -> Tweet:
    return Tweet(
        tweet_id=tweet_id,
        author_id=author,
        inbound=inbound,
        created_at=BASE + timedelta(hours=offset_hours),
        text=text,
    )


def _pair(
    pid: str,
    customer_text: str,
    support_text: str = "We can help, please DM us.",
    *,
    offset_hours: int = 0,
    customer_id: str = "cust1",
    conversation_id: str | None = None,
) -> SupportPair:
    return SupportPair(
        pair_id=pid,
        brand="TestBrand",
        conversation_id=conversation_id or f"conv_{pid}",
        customer_tweet=_tweet(f"c{pid}", customer_id, True, customer_text, offset_hours),
        support_tweet=_tweet(f"s{pid}", "TestBrand", False, support_text, offset_hours),
    )


def _corpus() -> list[SupportPair]:
    return [
        _pair("1", "my package never arrived", offset_hours=0, customer_id="cust1"),
        _pair("2", "wrong item in the box", offset_hours=1, customer_id="cust2"),
        _pair("3", "how do I return this", offset_hours=2, customer_id="cust3"),
    ]


def _golden() -> list[SupportPair]:
    return [
        _pair("90", "charged twice for one order", offset_hours=100, customer_id="cust9"),
        _pair("91", "app crashes on checkout", offset_hours=101, customer_id="cust8"),
    ]


class TestIdOverlap:
    def test_silent_on_disjoint_splits(self):
        assert_no_id_overlap(_corpus(), _golden())

    def test_raises_on_shared_pair_id(self):
        shared = _pair("1", "my package never arrived", offset_hours=100)
        with pytest.raises(LeakageError, match="pair_id"):
            assert_no_id_overlap(_corpus(), [*_golden(), shared])

    def test_raises_on_shared_conversation_id(self):
        leaked = _pair("99", "different text entirely", offset_hours=100, conversation_id="conv_1")
        with pytest.raises(LeakageError, match="conversation_id"):
            assert_no_id_overlap(_corpus(), [leaked])

    def test_raises_on_shared_customer_id(self):
        """Same customer across splits leaks their phrasing and their history."""
        leaked = _pair("99", "brand new complaint", offset_hours=100, customer_id="cust1")
        with pytest.raises(LeakageError, match="customer_id"):
            assert_no_id_overlap(_corpus(), [leaked])

    def test_error_names_the_offending_id(self):
        leaked = _pair("99", "x", offset_hours=100, conversation_id="conv_1")
        with pytest.raises(LeakageError, match="conv_1"):
            assert_no_id_overlap(_corpus(), [leaked])


class TestTextDuplicates:
    def test_silent_on_distinct_text(self):
        assert_no_text_duplicates(_corpus(), _golden())

    def test_raises_on_exact_duplicate(self):
        leaked = _pair("99", "my package never arrived", offset_hours=100, customer_id="cust9")
        with pytest.raises(LeakageError, match="duplicate"):
            assert_no_text_duplicates(_corpus(), [leaked])

    def test_raises_on_duplicate_differing_only_by_case_and_punctuation(self):
        leaked = _pair("99", "MY PACKAGE NEVER ARRIVED!!!", offset_hours=100, customer_id="cust9")
        with pytest.raises(LeakageError):
            assert_no_text_duplicates(_corpus(), [leaked])

    def test_raises_on_duplicate_differing_only_by_mention_and_url(self):
        leaked = _pair(
            "99",
            "@TestBrand my package never arrived https://t.co/abc",
            offset_hours=100,
            customer_id="cust9",
        )
        with pytest.raises(LeakageError):
            assert_no_text_duplicates(_corpus(), [leaked])


class TestNearDuplicates:
    def test_silent_on_semantically_different_messages(self):
        assert_no_near_duplicates(_corpus(), _golden())

    def test_raises_on_near_duplicate_with_one_word_changed(self):
        leaked = _pair("99", "my parcel never arrived", offset_hours=100, customer_id="cust9")
        with pytest.raises(LeakageError, match="near-duplicate"):
            assert_no_near_duplicates(_corpus(), [leaked], threshold=0.7)

    def test_threshold_is_configurable(self):
        leaked = _pair("99", "my parcel never arrived", offset_hours=100, customer_id="cust9")
        assert_no_near_duplicates(_corpus(), [leaked], threshold=0.999)

    def test_handles_empty_inputs_without_crashing(self):
        assert_no_near_duplicates([], [])
        assert_no_near_duplicates(_corpus(), [])


class TestResponseLeakage:
    def test_silent_when_only_the_canned_reply_repeats(self):
        """Brands send identical canned replies constantly; that is not leakage."""
        golden = [
            _pair("90", "a totally different question", "We can help, please DM us.", offset_hours=100)
        ]
        assert_no_response_leakage(_corpus(), golden)

    def test_raises_when_the_same_question_and_answer_pair_appears_in_the_corpus(self):
        leaked = _pair(
            "99", "my package never arrived", "We can help, please DM us.", offset_hours=100
        )
        with pytest.raises(LeakageError, match="response"):
            assert_no_response_leakage(_corpus(), [leaked])


class TestTemporalSplit:
    def test_silent_when_corpus_entirely_precedes_golden(self):
        assert_temporal_split(_corpus(), _golden())

    def test_raises_when_a_corpus_item_postdates_a_golden_item(self):
        future = _pair("4", "from the future", offset_hours=200, customer_id="cust4")
        with pytest.raises(LeakageError, match="temporal"):
            assert_temporal_split([*_corpus(), future], _golden())

    def test_handles_empty_inputs(self):
        assert_temporal_split([], _golden())
        assert_temporal_split(_corpus(), [])


class TestModelIndependence:
    """The guard that would have caught the field's worst failure."""

    def test_silent_when_all_roles_use_different_families(self):
        assert_independent_models(
            generator="claude-sonnet-5",
            judge="gemini-2.5-pro",
            pre_annotator="qwen3-32b",
            system_under_test="claude-sonnet-5",
        )

    def test_raises_when_judge_shares_a_family_with_the_generator(self):
        with pytest.raises(LeakageError, match="judge"):
            assert_independent_models(
                generator="claude-sonnet-5",
                judge="claude-opus-5",
                pre_annotator="qwen3-32b",
                system_under_test="claude-sonnet-5",
            )

    def test_raises_when_pre_annotator_is_the_system_under_test(self):
        """This is exactly the shubham failure: gold prefilled by the model being graded."""
        with pytest.raises(LeakageError, match="pre-annotator"):
            assert_independent_models(
                generator="claude-sonnet-5",
                judge="gemini-2.5-pro",
                pre_annotator="claude-sonnet-5",
                system_under_test="claude-sonnet-5",
            )

    def test_raises_when_pre_annotator_merely_shares_a_family_with_system_under_test(self):
        with pytest.raises(LeakageError, match="pre-annotator"):
            assert_independent_models(
                generator="gpt-4o",
                judge="gemini-2.5-pro",
                pre_annotator="gpt-4o-mini",
                system_under_test="gpt-4o",
            )

    @pytest.mark.parametrize(
        ("model", "family"),
        [
            ("claude-sonnet-5", "anthropic"),
            ("gpt-4o-mini", "openai"),
            ("gemini-2.5-pro", "google"),
            ("qwen3-32b", "qwen"),
            ("llama-3.3-70b", "meta"),
            ("mistral-large", "mistral"),
        ],
    )
    def test_model_family_detection(self, model, family):
        assert model_family(model) == family

    def test_unknown_model_family_is_distinct_not_silently_equal(self):
        """Two unknown models must not be treated as the same family by accident."""
        assert model_family("some-new-model") != model_family("another-new-model")


class TestRunAllChecks:
    def test_silent_on_clean_splits(self):
        run_all_checks(_corpus(), _golden())

    def test_raises_and_aggregates_every_violation(self):
        leaked = _pair("1", "my package never arrived", offset_hours=-5, customer_id="cust1")
        with pytest.raises(LeakageError) as excinfo:
            run_all_checks(_corpus(), [leaked])

        message = str(excinfo.value)
        assert "pair_id" in message
        assert "duplicate" in message

    def test_reports_all_failures_not_merely_the_first(self):
        leaked = _pair("1", "my package never arrived", offset_hours=-5, customer_id="cust1")
        with pytest.raises(LeakageError) as excinfo:
            run_all_checks(_corpus(), [leaked])
        assert str(excinfo.value).count("- ") >= 2

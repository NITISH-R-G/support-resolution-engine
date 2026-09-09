"""Milestone 1 — conversation reconstruction from raw TWCS rows.

The Customer Support on Twitter corpus is noisy in specific, documented ways, and every
quirk exercised here was observed in the real file rather than imagined:

  * ``response_tweet_id`` holds a comma-separated list ("2,3") when a reply fans out.
  * ``in_response_to_tweet_id`` is absent for conversation roots and, in a subsample, for
    replies whose parent was not sampled (orphans).
  * ids arrive as ints from pandas but as strings after a CSV round-trip.
  * ``created_at`` uses Twitter's format: "Tue Oct 31 22:10:47 +0000 2017".
  * malformed rows can point at each other, forming a cycle.

These tests define the contract before the implementation exists.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from hiver_support.data.schema import Tweet
from hiver_support.data.threads import (
    extract_support_pairs,
    identify_brand,
    parse_tweets,
    reconstruct_conversations,
)

TWCS_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def _rows(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=TWCS_COLUMNS)


def _simple_thread() -> pd.DataFrame:
    """Customer asks, brand replies, customer thanks. The canonical happy path."""
    return _rows(
        (1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "@sprintcare my data is not working", "2", None),
        (2, "sprintcare", False, "Tue Oct 31 22:11:45 +0000 2017", "Can you DM us your account details?", "3", 1),
        (3, "115712", True, "Tue Oct 31 22:12:10 +0000 2017", "just sent it, thanks", None, 2),
    )


# ---------------------------------------------------------------- parse_tweets


class TestParseTweets:
    def test_parses_minimal_row_into_typed_tweet(self):
        tweets = parse_tweets(_simple_thread())

        assert len(tweets) == 3
        first = tweets[0]
        assert isinstance(first, Tweet)
        assert first.tweet_id == "1"
        assert first.author_id == "115712"
        assert first.inbound is True
        assert first.text == "@sprintcare my data is not working"

    def test_parses_twitter_datetime_format_as_utc(self):
        tweet = parse_tweets(_simple_thread())[0]
        assert tweet.created_at == datetime(2017, 10, 31, 22, 10, 47, tzinfo=timezone.utc)

    def test_splits_comma_separated_response_ids(self):
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "help", "2,3", None),
            (2, "sprintcare", False, "Tue Oct 31 22:11:45 +0000 2017", "hi", None, 1),
            (3, "sprintcare", False, "Tue Oct 31 22:11:50 +0000 2017", "also this", None, 1),
        )
        assert parse_tweets(df)[0].response_tweet_ids == ("2", "3")

    def test_absent_in_response_to_becomes_none(self):
        assert parse_tweets(_simple_thread())[0].in_response_to is None

    def test_ids_are_normalised_to_strings_regardless_of_input_type(self):
        """Guards the int-vs-str mismatch that silently breaks reply linkage."""
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "help", "2", None),
            ("2", "sprintcare", False, "Tue Oct 31 22:11:45 +0000 2017", "hi", None, "1"),
        )
        tweets = parse_tweets(df)
        assert {t.tweet_id for t in tweets} == {"1", "2"}
        assert tweets[1].in_response_to == "1"

    def test_empty_frame_yields_no_tweets(self):
        assert parse_tweets(_rows()) == []

    def test_rows_with_missing_text_are_dropped(self):
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", None, None, None),
            (2, "115712", True, "Tue Oct 31 22:10:48 +0000 2017", "real text", None, None),
        )
        assert [t.tweet_id for t in parse_tweets(df)] == ["2"]


# ------------------------------------------------------- reconstruct_conversations


class TestReconstructConversations:
    def test_links_replies_into_one_conversation(self):
        convs = reconstruct_conversations(parse_tweets(_simple_thread()))
        assert len(convs) == 1
        assert [t.tweet_id for t in convs[0].tweets] == ["1", "2", "3"]

    def test_every_tweet_belongs_to_exactly_one_conversation(self):
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "@sprintcare my data is not working", "2", None),
            (2, "sprintcare", False, "Tue Oct 31 22:11:45 +0000 2017", "Can you DM us your account details?", "3", 1),
            (3, "115712", True, "Tue Oct 31 22:12:10 +0000 2017", "just sent it, thanks", None, 2),
            (9, "115999", True, "Tue Oct 31 23:00:00 +0000 2017", "separate issue", None, None),
        )
        tweets = parse_tweets(df)
        convs = reconstruct_conversations(tweets)

        seen = [t.tweet_id for c in convs for t in c.tweets]
        assert sorted(seen) == sorted(t.tweet_id for t in tweets)
        assert len(seen) == len(set(seen)), "a tweet appeared in more than one conversation"

    def test_tweets_are_chronologically_ordered_even_when_rows_are_shuffled(self):
        shuffled = _simple_thread().iloc[::-1].reset_index(drop=True)
        convs = reconstruct_conversations(parse_tweets(shuffled))
        times = [t.created_at for t in convs[0].tweets]
        assert times == sorted(times)

    def test_conversation_id_is_stable_regardless_of_row_order(self):
        forward = reconstruct_conversations(parse_tweets(_simple_thread()))[0]
        backward = reconstruct_conversations(
            parse_tweets(_simple_thread().iloc[::-1].reset_index(drop=True))
        )[0]
        assert forward.conversation_id == backward.conversation_id

    def test_cyclic_reply_links_terminate(self):
        """Two tweets each claiming to reply to the other must not hang or recurse forever."""
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "a", "2", 2),
            (2, "sprintcare", False, "Tue Oct 31 22:11:45 +0000 2017", "b", "1", 1),
        )
        convs = reconstruct_conversations(parse_tweets(df))
        assert len(convs) == 1
        assert len(convs[0].tweets) == 2

    def test_orphan_reply_becomes_its_own_conversation(self):
        """In a subsample the parent is often absent; that must not drop the tweet."""
        df = _rows(
            (5, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "orphan reply", None, 999),
        )
        convs = reconstruct_conversations(parse_tweets(df))
        assert len(convs) == 1
        assert [t.tweet_id for t in convs[0].tweets] == ["5"]

    def test_no_tweets_yields_no_conversations(self):
        assert reconstruct_conversations([]) == []


# ---------------------------------------------------------------- identify_brand


class TestIdentifyBrand:
    def test_brand_is_the_outbound_author(self):
        conv = reconstruct_conversations(parse_tweets(_simple_thread()))[0]
        assert identify_brand(conv) == "sprintcare"

    def test_conversation_without_outbound_author_has_no_brand(self):
        df = _rows((1, "115712", True, "Tue Oct 31 22:10:47 +0000 2017", "nobody replied", None, None))
        conv = reconstruct_conversations(parse_tweets(df))[0]
        assert identify_brand(conv) is None

    def test_multi_brand_conversation_resolves_to_the_most_frequent_responder(self):
        """Customers routinely @ two brands; the thread belongs to whoever actually worked it."""
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:00 +0000 2017", "@AppleSupport @VerizonSupport help", "2", None),
            (2, "AppleSupport", False, "Tue Oct 31 22:11:00 +0000 2017", "we can help", "3", 1),
            (3, "115712", True, "Tue Oct 31 22:12:00 +0000 2017", "thanks", "4", 2),
            (4, "AppleSupport", False, "Tue Oct 31 22:13:00 +0000 2017", "anytime", None, 3),
            (5, "VerizonSupport", False, "Tue Oct 31 22:14:00 +0000 2017", "us too", None, 1),
        )
        conv = reconstruct_conversations(parse_tweets(df))[0]
        assert identify_brand(conv) == "AppleSupport"


# ------------------------------------------------------------ extract_support_pairs


class TestExtractSupportPairs:
    def test_pairs_customer_message_with_the_brand_reply_that_follows(self):
        conv = reconstruct_conversations(parse_tweets(_simple_thread()))[0]
        pairs = extract_support_pairs(conv, brand="sprintcare")

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.customer_tweet.tweet_id == "1"
        assert pair.support_tweet.tweet_id == "2"
        assert pair.brand == "sprintcare"

    def test_pair_carries_preceding_turns_as_context(self):
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:00 +0000 2017", "first question", "2", None),
            (2, "sprintcare", False, "Tue Oct 31 22:11:00 +0000 2017", "first answer", "3", 1),
            (3, "115712", True, "Tue Oct 31 22:12:00 +0000 2017", "follow up question", "4", 2),
            (4, "sprintcare", False, "Tue Oct 31 22:13:00 +0000 2017", "second answer", None, 3),
        )
        conv = reconstruct_conversations(parse_tweets(df))[0]
        pairs = extract_support_pairs(conv, brand="sprintcare")

        assert len(pairs) == 2
        assert pairs[0].context == ()
        assert [t.tweet_id for t in pairs[1].context] == ["1", "2"]

    def test_customer_message_without_a_reply_produces_no_pair(self):
        df = _rows((1, "115712", True, "Tue Oct 31 22:10:00 +0000 2017", "ignored question", None, None))
        conv = reconstruct_conversations(parse_tweets(df))[0]
        assert extract_support_pairs(conv, brand="sprintcare") == []

    def test_replies_from_a_different_brand_are_not_paired(self):
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:00 +0000 2017", "help me", "2", None),
            (2, "VerizonSupport", False, "Tue Oct 31 22:11:00 +0000 2017", "not our problem", None, 1),
        )
        conv = reconstruct_conversations(parse_tweets(df))[0]
        assert extract_support_pairs(conv, brand="sprintcare") == []

    def test_consecutive_brand_replies_pair_only_the_first(self):
        """Brands often send two tweets in a row; the pair must not duplicate the customer turn."""
        df = _rows(
            (1, "115712", True, "Tue Oct 31 22:10:00 +0000 2017", "help me", "2", None),
            (2, "sprintcare", False, "Tue Oct 31 22:11:00 +0000 2017", "part one", "3", 1),
            (3, "sprintcare", False, "Tue Oct 31 22:11:30 +0000 2017", "part two", None, 2),
        )
        conv = reconstruct_conversations(parse_tweets(df))[0]
        pairs = extract_support_pairs(conv, brand="sprintcare")
        assert len(pairs) == 1
        assert pairs[0].support_tweet.tweet_id == "2"

    def test_pair_id_is_deterministic(self):
        conv = reconstruct_conversations(parse_tweets(_simple_thread()))[0]
        a = extract_support_pairs(conv, brand="sprintcare")[0]
        b = extract_support_pairs(conv, brand="sprintcare")[0]
        assert a.pair_id == b.pair_id

"""Typed models for the TWCS corpus.

All models are frozen: reconstruction and evaluation both depend on these objects being
hashable and non-mutating, and an accidental in-place edit during retrieval is exactly the
kind of bug that quietly corrupts an evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Tweet:
    """A single tweet from the corpus.

    ``tweet_id``, ``in_response_to`` and ``response_tweet_ids`` are always strings, because
    the raw file mixes integer and string representations of the same id and comparing
    across the two silently breaks reply linkage.
    """

    tweet_id: str
    author_id: str
    inbound: bool
    created_at: datetime
    text: str
    response_tweet_ids: tuple[str, ...] = ()
    in_response_to: str | None = None


@dataclass(frozen=True, slots=True)
class Conversation:
    """A reply-linked thread, ordered chronologically."""

    conversation_id: str
    tweets: tuple[Tweet, ...]

    @property
    def start(self) -> datetime:
        return self.tweets[0].created_at

    @property
    def length(self) -> int:
        return len(self.tweets)


@dataclass(frozen=True, slots=True)
class SupportPair:
    """A customer message and the brand reply that answered it.

    ``context`` holds the turns that preceded the customer message in the same conversation,
    so a later turn can be evaluated with the history the agent would really have seen.
    """

    pair_id: str
    brand: str
    conversation_id: str
    customer_tweet: Tweet
    support_tweet: Tweet
    context: tuple[Tweet, ...] = field(default=())

    @property
    def customer_text(self) -> str:
        return self.customer_tweet.text

    @property
    def support_text(self) -> str:
        return self.support_tweet.text

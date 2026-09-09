"""Reconstruct conversation threads and customer/support pairs from raw TWCS rows.

The corpus stores reply structure in two partially redundant columns
(``in_response_to_tweet_id`` and ``response_tweet_id``), either of which can be missing on
any given row. Grouping uses both directions and a union-find, which is inherently immune
to the reply cycles that a few malformed rows contain -- a recursive parent-walk is not.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from hiver_support.data.schema import Conversation, SupportPair, Tweet

TWITTER_DATETIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _normalise_id(value: object) -> str | None:
    """Coerce an id to a canonical string, or ``None`` if absent.

    pandas reads an integer column containing blanks as floats, turning tweet id 1 into
    ``1.0``. Left alone, ``"1.0" != "1"`` breaks every reply link in the file.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        # A string that round-tripped through a float column ("1.0") normalises too.
        try:
            as_float = float(text)
        except ValueError:
            return text
        return str(int(as_float)) if as_float.is_integer() else text
    return str(value)


def _split_response_ids(value: object) -> tuple[str, ...]:
    """``"2,3"`` -> ``("2", "3")``; blanks -> ``()``."""
    raw = _normalise_id(value) if not isinstance(value, str) else value
    if raw is None:
        return ()
    parts = (_normalise_id(part) for part in str(raw).split(","))
    return tuple(part for part in parts if part is not None)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), TWITTER_DATETIME_FORMAT)
    except ValueError:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(parsed) else parsed.to_pydatetime()


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_tweets(frame: pd.DataFrame) -> list[Tweet]:
    """Turn raw TWCS rows into ``Tweet`` objects, dropping rows that cannot be used.

    A row is unusable when it has no id, no text, or no parseable timestamp; each of those
    appears in the real file and none can be repaired by guessing.
    """
    tweets: list[Tweet] = []
    for row in frame.to_dict("records"):
        tweet_id = _normalise_id(row.get("tweet_id"))
        if tweet_id is None:
            continue

        text = row.get("text")
        if _is_blank(text) or not str(text).strip():
            continue

        created_at = _parse_datetime(row.get("created_at"))
        if created_at is None:
            continue

        author_id = _normalise_id(row.get("author_id"))
        if author_id is None:
            continue

        tweets.append(
            Tweet(
                tweet_id=tweet_id,
                author_id=author_id,
                inbound=_parse_bool(row.get("inbound")),
                created_at=created_at,
                text=str(text),
                response_tweet_ids=_split_response_ids(row.get("response_tweet_id")),
                in_response_to=_normalise_id(row.get("in_response_to_tweet_id")),
            )
        )
    return tweets


class _UnionFind:
    """Minimal union-find. Cycle-safe by construction, which the raw data requires."""

    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _conversation_id(tweet_ids: list[str]) -> str:
    """Deterministic id derived from membership, so row order cannot change it."""
    digest = hashlib.sha1(",".join(sorted(tweet_ids)).encode()).hexdigest()
    return f"conv_{digest[:16]}"


def reconstruct_conversations(tweets: list[Tweet]) -> list[Conversation]:
    """Group tweets into reply-linked conversations, ordered chronologically.

    Links pointing outside the supplied set (common when working on a subsample) are
    ignored rather than dropped, so an orphaned reply becomes a single-tweet conversation
    instead of vanishing from the corpus.
    """
    if not tweets:
        return []

    by_id = {tweet.tweet_id: tweet for tweet in tweets}
    union_find = _UnionFind(list(by_id))

    for tweet in tweets:
        if tweet.in_response_to is not None and tweet.in_response_to in by_id:
            union_find.union(tweet.tweet_id, tweet.in_response_to)
        for child_id in tweet.response_tweet_ids:
            if child_id in by_id:
                union_find.union(tweet.tweet_id, child_id)

    groups: dict[str, list[Tweet]] = {}
    for tweet in tweets:
        groups.setdefault(union_find.find(tweet.tweet_id), []).append(tweet)

    conversations = [
        Conversation(
            conversation_id=_conversation_id([t.tweet_id for t in members]),
            # Secondary sort on id keeps ordering stable when timestamps tie.
            tweets=tuple(sorted(members, key=lambda t: (t.created_at, t.tweet_id))),
        )
        for members in groups.values()
    ]
    return sorted(conversations, key=lambda c: (c.start, c.conversation_id))


def identify_brand(conversation: Conversation) -> str | None:
    """The brand that worked the thread, or ``None`` if nobody replied.

    Customers routinely tag several brands, so the busiest responder wins; ties break on
    who replied first, then alphabetically, to keep the result deterministic.
    """
    responders = [t.author_id for t in conversation.tweets if not t.inbound]
    if not responders:
        return None

    counts = Counter(responders)
    first_seen = {author: index for index, author in enumerate(reversed(responders))}
    return min(counts, key=lambda author: (-counts[author], -first_seen[author], author))


def extract_support_pairs(conversation: Conversation, brand: str) -> list[SupportPair]:
    """Pair each brand reply with the customer message immediately preceding it.

    Anchoring on the reply rather than the question is what makes consecutive brand tweets
    ("part one", "part two") yield one pair instead of duplicating the customer's turn.
    """
    pairs: list[SupportPair] = []
    tweets = conversation.tweets

    for index, tweet in enumerate(tweets):
        if index == 0 or tweet.inbound or tweet.author_id != brand:
            continue

        previous = tweets[index - 1]
        if not previous.inbound:
            continue

        pairs.append(
            SupportPair(
                pair_id=f"{previous.tweet_id}__{tweet.tweet_id}",
                brand=brand,
                conversation_id=conversation.conversation_id,
                customer_tweet=previous,
                support_tweet=tweet,
                context=tuple(tweets[: index - 1]),
            )
        )
    return pairs

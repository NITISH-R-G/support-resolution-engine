"""Deterministic temporal splitting into train / dev / test-pool.

Three constraints pull against each other:

1. **Chronology.** The retrieval corpus may not contain a resolution written after the
   question it answers. A random split quietly violates this and inflates every downstream
   number, which is why it is not offered here as an option.
2. **Conversation integrity.** A thread must land whole on one side, or the same incident
   appears in both the corpus and the evaluation set.
3. **Customer integrity.** A customer must not appear on both sides; the same person phrases
   complaints the same way and often raises the same issue twice.

Constraints 2 and 3 are satisfied by *discarding* the offending pairs rather than by bending
the boundary, and the discards are counted in the manifest instead of being silently
absorbed. Where a customer straddles a boundary, the earlier pairs are dropped: the
evaluation pool is the scarce resource and training data yields to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from hiver_support.data.schema import SupportPair

MIN_PAIRS = 10


class SplitError(ValueError):
    """Raised when a split cannot be produced honestly."""


@dataclass(frozen=True, slots=True)
class Splits:
    """Split assignment plus the manifest needed to reproduce and audit it."""

    train: tuple[SupportPair, ...]
    dev: tuple[SupportPair, ...]
    test_pool: tuple[SupportPair, ...]
    dropped: tuple[SupportPair, ...]
    manifest: dict


def _conversation_time(pairs: Sequence[SupportPair]) -> datetime:
    """A conversation is placed by its earliest turn, so it cannot be split by its own span."""
    return min(p.customer_tweet.created_at for p in pairs)


def temporal_split(
    pairs: Sequence[SupportPair],
    *,
    train_fraction: float = 0.7,
    dev_fraction: float = 0.1,
) -> Splits:
    """Split pairs chronologically, preserving conversation and customer integrity.

    Args:
        pairs: All customer/support pairs for one brand.
        train_fraction: Share of conversations for the retrieval corpus and training.
        dev_fraction: Share for threshold fitting and calibration. Everything after these
            two is the test pool, from which the golden set is later sampled.

    Returns:
        ``Splits`` whose ``train``/``dev``/``test_pool`` satisfy every guard in
        ``hiver_support.leakage``, plus the pairs discarded to achieve that.

    Raises:
        SplitError: on empty input, too few pairs to split meaningfully, or invalid
            fractions. Returning three near-empty splits would be worse than failing, since
            the resulting metrics would look computable but mean nothing.
    """
    if not pairs:
        raise SplitError("cannot split an empty set of pairs")
    if len(pairs) < MIN_PAIRS:
        raise SplitError(f"too few pairs to split meaningfully: {len(pairs)} < {MIN_PAIRS}")
    if not (0 < train_fraction < 1) or not (0 < dev_fraction < 1):
        raise SplitError("train_fraction and dev_fraction must each lie in (0, 1)")
    if train_fraction + dev_fraction >= 1:
        raise SplitError(
            f"train_fraction + dev_fraction must leave room for a test pool "
            f"({train_fraction} + {dev_fraction} >= 1)"
        )

    # --- group into conversations, ordered by their earliest turn ------------------------
    conversations: dict[str, list[SupportPair]] = {}
    for pair in pairs:
        conversations.setdefault(pair.conversation_id, []).append(pair)

    # Secondary sort on the id keeps the order total, so the result cannot depend on the
    # order the pairs happened to arrive in.
    ordered = sorted(
        conversations.items(), key=lambda item: (_conversation_time(item[1]), item[0])
    )

    train_end = int(len(ordered) * train_fraction)
    dev_end = int(len(ordered) * (train_fraction + dev_fraction))

    assignment: dict[str, str] = {}
    for index, (conversation_id, _) in enumerate(ordered):
        if index < train_end:
            assignment[conversation_id] = "train"
        elif index < dev_end:
            assignment[conversation_id] = "dev"
        else:
            assignment[conversation_id] = "test_pool"

    # --- drop conversations that straddle a boundary in time -----------------------------
    # Conversations are ordered by their *earliest* turn, but a thread has duration: one
    # starting just before the cut can still end well after it, leaving a later resolution
    # in the retrieval corpus than the question it is supposed to precede. An index cut
    # alone therefore does not deliver the strict ordering assert_temporal_split demands.
    # Boundaries are computed once, before any drop, so the decision cannot depend on the
    # order drops are applied in.
    def _earliest_start(split_name: str) -> datetime | None:
        starts = [
            _conversation_time(members)
            for conversation_id, members in ordered
            if assignment[conversation_id] == split_name
        ]
        return min(starts) if starts else None

    dev_start = _earliest_start("dev")
    test_start = _earliest_start("test_pool")
    next_boundary = {"train": dev_start or test_start, "dev": test_start, "test_pool": None}

    dropped: list[SupportPair] = []
    straddlers = 0
    surviving: list[tuple[str, list[SupportPair]]] = []

    for conversation_id, members in ordered:
        split_name = assignment[conversation_id]
        boundary = next_boundary[split_name]
        latest_turn = max(p.customer_tweet.created_at for p in members)
        if boundary is not None and latest_turn >= boundary:
            dropped.extend(members)
            straddlers += 1
            continue
        surviving.append((conversation_id, members))

    # --- resolve customers spanning a boundary ------------------------------------------
    # Later splits win, so the evaluation pool keeps its data and training absorbs the loss.
    rank = {"train": 0, "dev": 1, "test_pool": 2}
    customer_home: dict[str, str] = {}
    for conversation_id, members in surviving:
        split_name = assignment[conversation_id]
        for pair in members:
            customer = pair.customer_tweet.author_id
            current = customer_home.get(customer)
            if current is None or rank[split_name] > rank[current]:
                customer_home[customer] = split_name

    buckets: dict[str, list[SupportPair]] = {"train": [], "dev": [], "test_pool": []}
    customer_conflicts = 0

    for conversation_id, members in surviving:
        split_name = assignment[conversation_id]
        for pair in members:
            if customer_home[pair.customer_tweet.author_id] == split_name:
                buckets[split_name].append(pair)
            else:
                dropped.append(pair)
                customer_conflicts += 1

    for name in buckets:
        buckets[name].sort(key=lambda p: (p.customer_tweet.created_at, p.pair_id))
    dropped.sort(key=lambda p: (p.customer_tweet.created_at, p.pair_id))

    def _boundary(name: str) -> str | None:
        bucket = buckets[name]
        if not bucket:
            return None
        return max(p.customer_tweet.created_at for p in bucket).isoformat()

    # A split emptied by boundary drops must fail loudly. Everything downstream -- threshold
    # calibration on dev, metrics on the test pool -- would otherwise run to completion
    # against nothing and report numbers that look valid. Found by inspecting a manifest,
    # not by a test, which is why the check exists at all.
    empty = [name for name, bucket in buckets.items() if not bucket]
    if empty:
        raise SplitError(
            f"split produced an empty {', '.join(empty)} split: "
            f"{straddlers} conversation(s) crossed a boundary and were dropped. "
            f"Threads are long relative to their cadence; widen the fractions, use more "
            f"data, or split on a coarser time unit."
        )

    manifest = {
        "strategy": "temporal, grouped by conversation and customer",
        "train_fraction": train_fraction,
        "dev_fraction": dev_fraction,
        "counts": {name: len(bucket) for name, bucket in buckets.items()},
        "conversations": {
            name: len({p.conversation_id for p in bucket}) for name, bucket in buckets.items()
        },
        "customers": {
            name: len({p.customer_tweet.author_id for p in bucket})
            for name, bucket in buckets.items()
        },
        "dropped": len(dropped),
        # Surfaced explicitly because a high drop rate is a signal about the data, not a
        # detail: it means threads are long relative to the split cadence, and the retained
        # sample may no longer represent the corpus.
        "drop_rate": round(len(dropped) / len(pairs), 4),
        "dropped_breakdown": {
            "straddling_conversations": straddlers,
            "pairs_from_straddling_conversations": len(dropped) - customer_conflicts,
            "customer_conflicts": customer_conflicts,
        },
        "dropped_reason": (
            "conversation crossed a split boundary in time, or customer appeared in a later "
            "split (earlier pairs discarded so the evaluation pool keeps its data)"
        ),
        "boundaries": {
            "train_end": _boundary("train"),
            "dev_end": _boundary("dev"),
            "test_pool_end": _boundary("test_pool"),
        },
    }

    return Splits(
        train=tuple(buckets["train"]),
        dev=tuple(buckets["dev"]),
        test_pool=tuple(buckets["test_pool"]),
        dropped=tuple(dropped),
        manifest=manifest,
    )

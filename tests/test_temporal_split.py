"""Deterministic temporal splitting.

The split has to satisfy three constraints that pull against each other: it must be
chronological (a retrieval corpus may not contain resolutions written after the question it
answers), conversations may not straddle a boundary, and a customer may not appear on both
sides. The last two force some data to be discarded, and the tests below pin *which* side
loses it.

The decisive test is ``test_split_output_passes_every_leakage_guard``: the splitter and the
guards are written against the same contract, so if they ever disagree the suite fails.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.data.schema import SupportPair, Tweet
from hiver_support.data.split import SplitError, temporal_split
from hiver_support.leakage import run_all_checks

BASE = datetime(2017, 1, 1, tzinfo=timezone.utc)


# Distinct vocabulary per pair. Templated text like "issue number {i}" makes "…11" and
# "…111" genuine char-level near-duplicates, which trips the near-duplicate guard on the
# fixture rather than on anything the splitter did.
_SUBJECTS = ["parcel", "invoice", "headphones", "router", "subscription", "kettle", "laptop"]
_PROBLEMS = ["never arrived", "was damaged", "stopped working", "charged twice", "went missing"]
_EXTRAS = ["since Tuesday", "after the update", "for the third time", "despite two calls", ""]


def _pair(index: int, *, customer: str, conversation: str, hours: int) -> SupportPair:
    when = BASE + timedelta(hours=hours)
    subject = _SUBJECTS[index % len(_SUBJECTS)]
    problem = _PROBLEMS[(index // len(_SUBJECTS)) % len(_PROBLEMS)]
    extra = _EXTRAS[(index // (len(_SUBJECTS) * len(_PROBLEMS))) % len(_EXTRAS)]
    return SupportPair(
        pair_id=f"p{index}",
        brand="Br",
        conversation_id=conversation,
        customer_tweet=Tweet(
            f"c{index}", customer, True, when, f"my {subject} {problem} {extra} ref {index}".strip()
        ),
        support_tweet=Tweet(
            f"s{index}", "Br", False, when + timedelta(minutes=5), f"resolution for {subject}"
        ),
    )


def _corpus(n: int = 100) -> list[SupportPair]:
    """One conversation and one customer per pair, spread evenly through time."""
    return [_pair(i, customer=f"cust{i}", conversation=f"conv{i}", hours=i) for i in range(n)]


class TestChronology:
    def test_train_precedes_dev_precedes_test(self):
        splits = temporal_split(_corpus())

        latest_train = max(p.customer_tweet.created_at for p in splits.train)
        earliest_dev = min(p.customer_tweet.created_at for p in splits.dev)
        latest_dev = max(p.customer_tweet.created_at for p in splits.dev)
        earliest_test = min(p.customer_tweet.created_at for p in splits.test_pool)

        assert latest_train < earliest_dev
        assert latest_dev < earliest_test

    def test_respects_requested_proportions_approximately(self):
        splits = temporal_split(_corpus(100), train_fraction=0.7, dev_fraction=0.1)
        assert len(splits.train) == pytest.approx(70, abs=3)
        assert len(splits.dev) == pytest.approx(10, abs=3)
        assert len(splits.test_pool) == pytest.approx(20, abs=3)

    def test_every_pair_is_assigned_or_explicitly_dropped(self):
        pairs = _corpus()
        splits = temporal_split(pairs)
        accounted = len(splits.train) + len(splits.dev) + len(splits.test_pool) + len(splits.dropped)
        assert accounted == len(pairs)


class TestGroupingConstraints:
    def test_a_conversation_never_straddles_a_boundary(self):
        """A thread spanning the cut must land whole on one side."""
        pairs = _corpus(60)
        # One conversation with turns either side of the 70% cut.
        pairs += [
            _pair(900, customer="spanner", conversation="convX", hours=5),
            _pair(901, customer="spanner", conversation="convX", hours=55),
        ]
        splits = temporal_split(pairs)

        placement = {}
        for name in ("train", "dev", "test_pool"):
            for pair in getattr(splits, name):
                placement.setdefault(pair.conversation_id, set()).add(name)
        assert all(len(names) == 1 for names in placement.values())

    def test_a_customer_never_appears_on_both_sides(self):
        pairs = _corpus(60)
        pairs += [
            _pair(902, customer="repeat", conversation="convA", hours=2),
            _pair(903, customer="repeat", conversation="convB", hours=58),
        ]
        splits = temporal_split(pairs)

        seen: dict[str, set[str]] = {}
        for name in ("train", "dev", "test_pool"):
            for pair in getattr(splits, name):
                seen.setdefault(pair.customer_tweet.author_id, set()).add(name)
        assert all(len(names) == 1 for names in seen.values())

    def test_a_spanning_customer_is_dropped_from_the_earlier_split_not_the_later(self):
        """The evaluation pool is the scarce resource, so training data yields to it."""
        pairs = _corpus(60)
        early = _pair(904, customer="repeat", conversation="convA", hours=2)
        late = _pair(905, customer="repeat", conversation="convB", hours=58)
        splits = temporal_split([*pairs, early, late])

        assert late.pair_id in {p.pair_id for p in splits.test_pool}
        assert early.pair_id in {p.pair_id for p in splits.dropped}


class TestLeakageContract:
    def test_split_output_passes_every_leakage_guard(self):
        splits = temporal_split(_corpus(120))
        run_all_checks(list(splits.train), list(splits.test_pool))

    def test_guards_also_pass_between_train_and_dev(self):
        splits = temporal_split(_corpus(120))
        run_all_checks(list(splits.train), list(splits.dev))


class TestDeterminism:
    def test_identical_input_yields_identical_splits(self):
        first = temporal_split(_corpus())
        second = temporal_split(_corpus())
        assert [p.pair_id for p in first.train] == [p.pair_id for p in second.train]
        assert [p.pair_id for p in first.test_pool] == [p.pair_id for p in second.test_pool]

    def test_input_order_does_not_change_the_result(self):
        pairs = _corpus()
        forward = temporal_split(pairs)
        backward = temporal_split(list(reversed(pairs)))
        assert [p.pair_id for p in forward.test_pool] == [p.pair_id for p in backward.test_pool]


class TestManifest:
    def test_manifest_records_counts_and_drops(self):
        splits = temporal_split(_corpus())
        manifest = splits.manifest
        assert manifest["counts"]["train"] == len(splits.train)
        assert manifest["counts"]["test_pool"] == len(splits.test_pool)
        assert manifest["dropped"] == len(splits.dropped)

    def test_manifest_records_boundaries_for_reproducibility(self):
        manifest = temporal_split(_corpus()).manifest
        assert "boundaries" in manifest
        assert manifest["boundaries"]["train_end"] < manifest["boundaries"]["dev_end"]

    def test_manifest_is_json_serialisable(self):
        import json

        json.dumps(temporal_split(_corpus()).manifest)


class TestEdgeCases:
    def test_empty_input_raises_rather_than_returning_empty_splits(self):
        with pytest.raises(SplitError):
            temporal_split([])

    def test_too_few_pairs_to_split_raises(self):
        with pytest.raises(SplitError, match="too few"):
            temporal_split(_corpus(3))

    def test_invalid_fractions_raise(self):
        with pytest.raises(SplitError):
            temporal_split(_corpus(), train_fraction=0.9, dev_fraction=0.2)


class TestOverlappingConversations:
    """Conversations are placed by their earliest turn, but they have duration.

    A long-running thread that starts before the cut can still end after it, so a
    conversation-index boundary does not by itself guarantee the strict temporal ordering
    that ``assert_temporal_split`` demands. Single-turn fixtures hide this entirely; real
    multi-turn threads do not.
    """

    @staticmethod
    def _long_threads() -> list[SupportPair]:
        pairs: list[SupportPair] = []
        index = 0
        for conversation in range(40):
            start = conversation * 4
            # Each thread runs 6 hours on a 4-hour cadence, so neighbours genuinely overlap.
            for turn in range(3):
                pairs.append(
                    _pair(
                        index,
                        customer=f"cust{conversation}",
                        conversation=f"conv{conversation}",
                        hours=start + turn * 3,
                    )
                )
                index += 1
        return pairs

    def test_overlapping_threads_still_satisfy_the_temporal_guard(self):
        splits = temporal_split(self._long_threads())
        run_all_checks(list(splits.train), list(splits.test_pool))

    def test_overlapping_threads_still_satisfy_guards_against_dev(self):
        splits = temporal_split(self._long_threads())
        run_all_checks(list(splits.train), list(splits.dev))


class TestDegenerateSplits:
    """A split that empties out must fail loudly rather than produce uncalibratable output.

    Discovered by manual inspection, not by a test: threads long enough relative to their
    cadence can strand an entire split at zero pairs, and every downstream metric would then
    be computed against nothing while still looking like a successful run.
    """

    @staticmethod
    def _very_long_threads() -> list[SupportPair]:
        pairs: list[SupportPair] = []
        index = 0
        for conversation in range(40):
            for turn in range(3):
                pairs.append(
                    _pair(
                        index,
                        customer=f"cust{conversation}",
                        conversation=f"conv{conversation}",
                        hours=conversation * 4 + turn * 15,
                    )
                )
                index += 1
        return pairs

    def test_a_split_emptied_by_boundary_drops_raises(self):
        with pytest.raises(SplitError, match="empty"):
            temporal_split(self._very_long_threads())

    def test_the_error_names_the_empty_split(self):
        with pytest.raises(SplitError, match="dev"):
            temporal_split(self._very_long_threads())


class TestDropRate:
    def test_manifest_reports_drop_rate(self):
        splits = temporal_split(_corpus(120))
        assert splits.manifest["drop_rate"] == pytest.approx(0.0)

    def test_drop_rate_is_reported_when_data_is_discarded(self):
        splits = temporal_split(TestOverlappingConversations._long_threads())
        assert 0.0 < splits.manifest["drop_rate"] < 1.0

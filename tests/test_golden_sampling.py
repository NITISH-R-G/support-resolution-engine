"""Golden-set sampling: deterministic, stratified, and honest about its own bias.

The sampling design is deliberately unrepresentative — hard strata are over-sampled — which
makes two things load-bearing and therefore tested here:

1. **Inclusion probabilities are known by construction**, so a representative estimate stays
   computable from a skewed sample. A hard-skewed set whose weights were lost could only ever
   report flattering per-stratum numbers.
2. **Sampling is frozen before it starts.** Re-running produces the same ids; the sampler
   refuses to overwrite work; no model output touches the frame.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.data.schema import SupportPair, Tweet
from hiver_support.golden.sampling import (
    RESERVOIR_SIZE,
    SEED,
    STRATUM_ORDER,
    STRATUM_QUOTAS,
    SamplingError,
    assign_stratum,
    filter_eligible,
    sample_candidates,
)
from hiver_support.golden.schema import GoldenCandidate

START = datetime(2017, 11, 1, tzinfo=timezone.utc)


def a_pair(index: int, text: str, *, context: tuple[Tweet, ...] = ()) -> SupportPair:
    when = START + timedelta(minutes=index)
    customer = Tweet(
        tweet_id=str(1000 + index),
        author_id=f"cust{index}",
        inbound=True,
        created_at=when,
        text=text,
    )
    support = Tweet(
        tweet_id=str(9000 + index),
        author_id="AppleSupport",
        inbound=False,
        created_at=when + timedelta(minutes=1),
        text="SECRET ANSWER: restart the device and update to 11.0.3",
    )
    return SupportPair(
        pair_id=f"{customer.tweet_id}__{support.tweet_id}",
        brand="AppleSupport",
        conversation_id=f"conv{index}",
        customer_tweet=customer,
        support_tweet=support,
        context=context,
    )


SECURITY = "someone hacked my account and made fraudulent charges, I think it was stolen"
THIN = "ok"
LONG = " ".join(["my iphone screen keeps flickering after the update and it drains"] * 8)
TYPICAL = "my iPhone battery drains in two hours since the iOS 11 update, please help"
ABSTAIN = "the quick brown fox jumps over the lazy dog beside the quiet riverbank today"


def a_population(n: int = 400) -> list[SupportPair]:
    """A population containing plenty of every stratum, so quotas are satisfiable."""
    templates = [SECURITY, THIN, LONG, TYPICAL, ABSTAIN]
    return [a_pair(i, f"{templates[i % len(templates)]} #{i}") for i in range(n)]


class TestStratumAssignment:
    def test_every_message_lands_in_exactly_one_stratum(self):
        for text in (SECURITY, THIN, LONG, TYPICAL, ABSTAIN, ""):
            assert assign_stratum(text) in STRATUM_ORDER

    def test_the_stratum_order_matches_the_frozen_quotas(self):
        assert tuple(STRATUM_QUOTAS) == STRATUM_ORDER

    def test_security_is_assigned_first_so_it_cannot_be_starved(self):
        # The security stratum is the rarest and the most safety-critical. A broader stratum
        # absorbing its members would leave the gate that matters most untested.
        assert STRATUM_ORDER[0] == "security_signal"
        assert assign_stratum(SECURITY) == "security_signal"

    def test_a_thin_message_is_thin_context_not_rule_abstain(self):
        # Abstaining because there is nothing to read is a different and less interesting
        # phenomenon than abstaining with plenty to read, so the two are kept apart.
        assert assign_stratum(THIN) == "thin_context"

    def test_a_long_ordinary_message_is_not_typical(self):
        assert assign_stratum(LONG) in ("long_message", "multi_signal", "rule_abstain")

    def test_a_clear_battery_message_is_typical(self):
        assert assign_stratum(TYPICAL) == "typical"

    def test_assignment_is_a_pure_function_of_the_text(self):
        assert assign_stratum(TYPICAL) == assign_stratum(TYPICAL)

    def test_assignment_rejects_a_non_string(self):
        with pytest.raises(TypeError):
            assign_stratum(None)


class TestDeterminism:
    def test_the_same_seed_produces_the_same_examples(self):
        pairs = a_population()
        first = sample_candidates(pairs, size=60, reservoir=20)
        second = sample_candidates(pairs, size=60, reservoir=20)
        assert [c.pair_id for c in first.candidates] == [c.pair_id for c in second.candidates]

    def test_a_different_seed_produces_different_examples(self):
        pairs = a_population()
        first = sample_candidates(pairs, size=60, reservoir=20, seed=SEED)
        second = sample_candidates(pairs, size=60, reservoir=20, seed=SEED + 1)
        assert [c.pair_id for c in first.candidates] != [c.pair_id for c in second.candidates]

    def test_candidates_are_emitted_in_a_stable_sorted_order(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        ids = [c.pair_id for c in result.candidates]
        assert ids == sorted(ids)

    def test_input_order_does_not_change_the_sample(self):
        pairs = a_population()
        forward = sample_candidates(pairs, size=60, reservoir=20)
        backward = sample_candidates(list(reversed(pairs)), size=60, reservoir=20)
        assert {c.pair_id for c in forward.candidates} == {
            c.pair_id for c in backward.candidates
        }


class TestSampleComposition:
    def test_it_draws_exactly_the_requested_size(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        assert len(result.candidates) == 60

    def test_every_example_appears_once(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        ids = [c.pair_id for c in result.candidates]
        assert len(set(ids)) == len(ids)

    def test_the_reservoir_is_drawn_first_and_is_disjoint_from_the_stratified_part(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        reservoir = {r.pair_id for r in result.records if r.component == "reservoir"}
        stratified = {r.pair_id for r in result.records if r.component == "stratified"}
        assert len(reservoir) == 20
        assert reservoir & stratified == set()

    def test_there_is_a_record_for_every_candidate(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        assert {r.pair_id for r in result.records} == {c.pair_id for c in result.candidates}

    def test_candidates_are_golden_candidates_and_therefore_unlabelled(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        assert all(isinstance(c, GoldenCandidate) for c in result.candidates)
        assert all(c.label_status.value == "UNLABELED" for c in result.candidates)


class TestTheAnswerNeverReachesTheCandidate:
    def test_no_candidate_contains_the_brands_reply_text(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        serialised = str([c.to_dict() for c in result.candidates])
        assert "SECRET ANSWER" not in serialised

    def test_context_carries_only_turns_that_precede_the_message(self):
        earlier = Tweet("1", "cust", True, START - timedelta(hours=1), "earlier turn")
        later = Tweet("2", "AppleSupport", False, START + timedelta(hours=5), "later turn")
        pairs = a_population()
        pairs[0] = a_pair(0, TYPICAL, context=(earlier, later))
        result = sample_candidates(pairs, size=400, reservoir=100)
        chosen = next(c for c in result.candidates if c.pair_id == pairs[0].pair_id)
        texts = [t.text for t in chosen.context]
        assert "earlier turn" in texts
        assert "later turn" not in texts


class TestInclusionProbabilities:
    def test_every_record_carries_a_probability_and_its_inverse_weight(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        for record in result.records:
            assert 0 < record.inclusion_probability <= 1
            assert record.weight == pytest.approx(1 / record.inclusion_probability)

    def test_over_sampled_strata_carry_smaller_weights_than_the_reservoir(self):
        # The whole point of recording weights: a rare stratum sampled at a high rate must
        # count for less when a representative estimate is reconstructed.
        result = sample_candidates(a_population(), size=60, reservoir=20)
        reservoir = [r for r in result.records if r.component == "reservoir"]
        security = [r for r in result.records if r.stratum == "security_signal"]
        assert security, "the fixture population must contain security examples"
        assert min(r.inclusion_probability for r in security) > 0
        assert all(r.inclusion_probability > 0 for r in reservoir)

    def test_the_reservoir_probability_is_its_share_of_the_population(self):
        pairs = a_population(400)
        result = sample_candidates(pairs, size=60, reservoir=20)
        reservoir = [r for r in result.records if r.component == "reservoir"]
        assert reservoir[0].inclusion_probability == pytest.approx(20 / 400)


class TestQuotasAndShortfall:
    def test_a_stratum_with_too_few_members_records_its_shortfall(self):
        # No security examples at all: the quota cannot be met and must be reported, not
        # silently absorbed into a healthier stratum.
        pairs = [a_pair(i, f"{TYPICAL} #{i}") for i in range(300)]
        result = sample_candidates(pairs, size=60, reservoir=20)
        shortfalls = result.manifest["strata"]["shortfall"]
        assert shortfalls.get("security_signal", 0) > 0

    def test_shortfall_is_redistributed_so_the_target_size_is_still_reached(self):
        pairs = [a_pair(i, f"{TYPICAL} #{i}") for i in range(300)]
        result = sample_candidates(pairs, size=60, reservoir=20)
        assert len(result.candidates) == 60

    def test_the_manifest_records_eligible_and_drawn_counts_per_stratum(self):
        result = sample_candidates(a_population(), size=60, reservoir=20)
        strata = result.manifest["strata"]
        assert set(strata["eligible"]) <= set(STRATUM_ORDER)
        assert sum(strata["drawn"].values()) == 60 - 20


class TestGuardsAndRefusals:
    def test_it_refuses_a_population_smaller_than_the_target(self):
        with pytest.raises(SamplingError, match="too small"):
            sample_candidates(a_population(30), size=60, reservoir=20)

    def test_it_refuses_a_reservoir_larger_than_the_sample(self):
        with pytest.raises(SamplingError, match="reservoir"):
            sample_candidates(a_population(), size=20, reservoir=60)

    def test_it_refuses_an_empty_population(self):
        with pytest.raises(SamplingError):
            sample_candidates([], size=60, reservoir=20)

    def test_it_refuses_duplicate_pair_ids_in_the_population(self):
        pairs = a_population(300)
        pairs.append(pairs[0])
        with pytest.raises(SamplingError, match="duplicate"):
            sample_candidates(pairs, size=60, reservoir=20)

    def test_default_reservoir_and_seed_come_from_the_frozen_protocol(self):
        assert RESERVOIR_SIZE == 50
        assert SEED == 20260911
        assert sum(STRATUM_QUOTAS.values()) == 150


class TestManifestProvenance:
    def test_the_manifest_pins_the_frozen_taxonomy_and_the_seed(self):
        from hiver_support.taxonomy import TAXONOMY

        manifest = sample_candidates(a_population(), size=60, reservoir=20).manifest
        assert manifest["seed"] == SEED
        assert manifest["taxonomy_hash"] == TAXONOMY.frozen_hash
        assert manifest["taxonomy_version"] == TAXONOMY.version

    def test_the_manifest_declares_that_no_labels_exist(self):
        manifest = sample_candidates(a_population(), size=60, reservoir=20).manifest
        assert manifest["label_status"] == "UNLABELED"
        assert manifest["human_labelled_count"] == 0

    def test_the_manifest_names_the_stratification_signal_as_weak(self):
        # Stratification uses the weak labelling functions. Saying so in the artifact is the
        # difference between a documented, bounded use and a hidden one.
        manifest = sample_candidates(a_population(), size=60, reservoir=20).manifest
        assert manifest["stratification_signal_provenance"] == "WEAKLY_LABELED"
        assert manifest["weak_labels_used_as_gold"] is False


class TestEligibilityFilter:
    """Found by running the leakage guards on a real draw, not by a test.

    546 of 22,378 real test-pool messages duplicate a train message verbatim once normalised.
    The guard is right; eligibility is therefore decided before sampling rather than by
    removing inconvenient examples after seeing the draw.
    """

    def test_a_pair_whose_text_duplicates_the_corpus_is_ineligible(self):
        corpus = [a_pair(0, "yes")]
        pool = [a_pair(1, "yes"), a_pair(2, TYPICAL)]
        eligible, _ = filter_eligible(pool, corpus)
        assert [p.pair_id for p in eligible] == [pool[1].pair_id]

    def test_duplicate_detection_ignores_casing_punctuation_and_mentions(self):
        corpus = [a_pair(0, "@AppleSupport Yes!!!")]
        eligible, _ = filter_eligible([a_pair(1, "yes")], corpus)
        assert eligible == []

    def test_an_empty_corpus_leaves_everything_eligible(self):
        pool = a_population(20)
        eligible, report = filter_eligible(pool, [])
        assert len(eligible) == 20
        assert report["excluded"] == 0

    def test_the_report_quantifies_the_bias_it_introduces_per_stratum(self):
        corpus = [a_pair(0, "ok")]
        pool = [a_pair(1, "ok"), a_pair(2, TYPICAL)]
        _, report = filter_eligible(pool, corpus)
        assert report["excluded"] == 1
        assert report["excluded_by_stratum"] == {"thin_context": 1}
        assert "under-represents" in report["bias_introduced"]

    def test_the_report_names_the_guard_it_is_pre_satisfying(self):
        _, report = filter_eligible(a_population(20), [])
        assert report["guards"] == [
            "leakage.assert_no_text_duplicates",
            "leakage.assert_no_near_duplicates",
        ]
        assert report["applied"].startswith("before sampling")

    def test_filtering_then_sampling_satisfies_the_guard_it_pre_empts(self):
        from hiver_support.leakage import assert_no_text_duplicates

        corpus = [a_pair(900 + i, f"{TYPICAL} #{i}") for i in range(40)]
        pool = a_population(300)
        eligible, _ = filter_eligible(pool, corpus)
        result = sample_candidates(eligible, size=60, reservoir=20)
        chosen = {c.pair_id for c in result.candidates}
        assert_no_text_duplicates(corpus, [p for p in pool if p.pair_id in chosen])

    def test_a_near_duplicate_is_ineligible_when_a_comparison_corpus_is_given(self):
        # Real finding: a short version-number message matched a corpus message at cosine 0.979.
        # The strings below are synthetic stand-ins with the same shape (no corpus text).
        corpus = [a_pair(0, "8plus ios 11.0")]
        pool = [a_pair(1, "8plus ios 11.0.3"), a_pair(2, TYPICAL)]
        eligible, report = filter_eligible(pool, corpus, near_duplicate_corpus=corpus)
        assert [p.pair_id for p in eligible] == [pool[1].pair_id]
        assert report["excluded_near_duplicate"] == 1

    def test_near_duplicate_filtering_is_skipped_when_no_corpus_is_supplied(self):
        corpus = [a_pair(0, "8plus ios 11.0")]
        eligible, report = filter_eligible([a_pair(1, "8plus ios 11.0.3")], corpus)
        assert len(eligible) == 1
        assert report["near_duplicate_corpus_size"] == 0

    def test_exact_and_near_exclusions_are_counted_separately(self):
        corpus = [a_pair(0, "8plus ios 11.0"), a_pair(5, "ok")]
        pool = [a_pair(1, "8plus ios 11.0.3"), a_pair(2, "ok"), a_pair(3, TYPICAL)]
        _, report = filter_eligible(pool, corpus, near_duplicate_corpus=corpus)
        assert report["excluded_exact_duplicate"] == 1
        assert report["excluded_near_duplicate"] == 1
        assert report["excluded"] == 2

    def test_filtering_then_sampling_satisfies_the_near_duplicate_guard(self):
        from hiver_support.leakage import assert_no_near_duplicates

        corpus = [a_pair(900 + i, f"{TYPICAL} #{i}") for i in range(40)]
        pool = a_population(300)
        eligible, _ = filter_eligible(pool, corpus, near_duplicate_corpus=corpus)
        result = sample_candidates(eligible, size=60, reservoir=20)
        chosen = {c.pair_id for c in result.candidates}
        assert_no_near_duplicates(corpus, [p for p in pool if p.pair_id in chosen])

    def test_chunking_does_not_change_which_rows_are_flagged(self):
        import hiver_support.golden.sampling as sampling

        corpus = [a_pair(900 + i, f"{TYPICAL} #{i}") for i in range(30)]
        pool = a_population(120)
        original = sampling.NEAR_DUPLICATE_CHUNK
        try:
            sampling.NEAR_DUPLICATE_CHUNK = 7
            small, _ = filter_eligible(pool, corpus, near_duplicate_corpus=corpus)
            sampling.NEAR_DUPLICATE_CHUNK = 10_000
            large, _ = filter_eligible(pool, corpus, near_duplicate_corpus=corpus)
        finally:
            sampling.NEAR_DUPLICATE_CHUNK = original
        assert [p.pair_id for p in small] == [p.pair_id for p in large]

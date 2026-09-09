"""Validation against the real TWCS corpus.

**This is the only test module that reads real data.** Everything else in `tests/` runs on
synthetic in-memory fixtures (`docs/DATA_PROVENANCE.md`). The separation is deliberate and
enforced: `test_data_provenance.py` allows exactly this file to touch the corpus, so the
boundary between "passes on fixtures" and "validated on real data" stays explicit rather
than eroding.

Every test here is skipped when the corpus is absent, so the suite still runs for a reviewer
who has not downloaded 493 MB. A skip is reported as a skip — never as a pass.

These tests exist because fixtures encode assumptions about the schema, and assumptions are
not findings. They pin the quirks actually observed in the file: float-coerced ids, fan-out
reply lists, and orphaned replies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hiver_support.data.threads import (
    extract_support_pairs,
    identify_brand,
    parse_tweets,
    reconstruct_conversations,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
PROFILES = ROOT / "reports" / "brand_profiles.json"
DECISION = ROOT / "reports" / "brand_decision.json"

SAMPLE_ROWS = 50_000

pytestmark = pytest.mark.skipif(
    not CORPUS.exists(), reason="real corpus absent; run scripts/fetch_data.py"
)

EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    return pd.read_csv(
        CORPUS, nrows=SAMPLE_ROWS, dtype=str, keep_default_na=False, na_values=[""]
    )


@pytest.fixture(scope="module")
def conversations(raw_sample):
    return reconstruct_conversations(parse_tweets(raw_sample))


class TestSchema:
    def test_columns_match_what_the_pipeline_expects(self, raw_sample):
        assert list(raw_sample.columns) == EXPECTED_COLUMNS

    def test_corpus_is_large_enough_to_be_the_real_file(self):
        assert CORPUS.stat().st_size > 400 * 1024**2


class TestQuirksTheFixturesModelled:
    """Confirms the fixtures were modelled on quirks that genuinely occur."""

    def test_fan_out_reply_lists_exist(self, raw_sample):
        fan_out = raw_sample.response_tweet_id.fillna("").str.contains(",")
        assert fan_out.sum() > 0

    def test_orphan_replies_exist(self, raw_sample):
        assert raw_sample.in_response_to_tweet_id.isna().sum() > 0

    def test_ids_are_float_coerced_when_pandas_infers_dtypes(self):
        """The defect behind DECISION_LOG.md D3, confirmed in the real file."""
        inferred = pd.read_csv(CORPUS, nrows=5)
        assert inferred.response_tweet_id.dtype.kind == "f"


class TestReconstructionOnRealRows:
    def test_no_rows_are_silently_lost(self, raw_sample):
        assert len(parse_tweets(raw_sample)) == len(raw_sample)

    def test_every_tweet_belongs_to_exactly_one_conversation(self, raw_sample, conversations):
        seen = [t.tweet_id for c in conversations for t in c.tweets]
        assert len(seen) == len(set(seen))
        assert len(seen) == len(parse_tweets(raw_sample))

    def test_conversations_are_chronological(self, conversations):
        for conversation in conversations[:5_000]:
            times = [t.created_at for t in conversation.tweets]
            assert times == sorted(times)

    def test_multi_turn_threads_are_reconstructed(self, conversations):
        """A pipeline that only ever produced singletons would pass weaker assertions."""
        assert max(len(c.tweets) for c in conversations) > 10
        assert sum(1 for c in conversations if len(c.tweets) > 2) > 1_000

    def test_brands_are_identified(self, conversations):
        brands = {identify_brand(c) for c in conversations}
        brands.discard(None)
        assert len(brands) > 10

    def test_pairs_have_an_inbound_question_and_an_outbound_reply(self, conversations):
        checked = 0
        for conversation in conversations:
            brand = identify_brand(conversation)
            if brand is None:
                continue
            for pair in extract_support_pairs(conversation, brand):
                assert pair.customer_tweet.inbound is True
                assert pair.support_tweet.inbound is False
                assert pair.support_tweet.author_id == brand
                assert pair.customer_tweet.created_at <= pair.support_tweet.created_at
                checked += 1
                if checked >= 2_000:
                    return
        assert checked > 0


@pytest.mark.skipif(not PROFILES.exists(), reason="brand profiles not yet generated")
class TestBrandAnalysisArtifacts:
    @staticmethod
    def _load(path: Path) -> dict:
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def test_profiles_carry_provenance(self):
        provenance = self._load(PROFILES)["provenance"]
        assert provenance["corpus_modified"] is False
        assert provenance["corpus_sha256_first_64mb"]
        assert provenance["random_seed"]

    def test_every_profile_has_the_thirteen_feature_families(self):
        required = {
            "pair_count",
            "distinct_intents_at_3pct",
            "intent_entropy_normalised",
            "rare_intent_pairs_estimated",
            "substantive_resolution_rate",
            "dm_deflection_rate",
            "actionable_resolution_rate",
            "median_thread_length",
            "exact_duplicate_rate",
            "near_duplicate_rate",
            "distinct_months",
            "escalation_sensitive_rate",
            "retrieval_nn_similarity_median",
            "usable_grounding_evidence_pairs",
        }
        for profile in self._load(PROFILES)["profiles"]:
            assert required <= set(profile), f"{profile['brand']} missing features"

    def test_rates_are_within_bounds(self):
        rate_fields = [k for k in ("dm_deflection_rate", "substantive_resolution_rate",
                                   "actionable_resolution_rate", "exact_duplicate_rate",
                                   "non_english_reply_rate", "escalation_sensitive_rate")]
        for profile in self._load(PROFILES)["profiles"]:
            for field in rate_fields:
                assert 0.0 <= profile[field] <= 1.0, f"{profile['brand']}.{field}"


@pytest.mark.skipif(not DECISION.exists(), reason="brand decision not yet made")
class TestBrandDecisionIntegrity:
    @staticmethod
    def _decision() -> dict:
        import json

        return json.loads(DECISION.read_text(encoding="utf-8"))

    def test_decision_declares_no_model_performance_was_used(self):
        """The pre-registration guarantee, asserted rather than merely claimed in prose."""
        assert self._decision()["model_performance_used"] is False

    def test_selected_brand_is_the_top_ranked_survivor(self):
        decision = self._decision()
        assert decision["ranking"][0]["brand"] == decision["selected_brand"]

    def test_rubric_weights_are_equal(self):
        weights = self._decision()["rubric_weights"]
        assert len(set(weights.values())) == 1, "unequal weights invite post-hoc tuning"
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_ranking_is_sorted_by_score(self):
        scores = [entry["rubric_score"] for entry in self._decision()["ranking"]]
        assert scores == sorted(scores, reverse=True)

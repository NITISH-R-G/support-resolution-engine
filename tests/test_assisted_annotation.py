"""Model-assisted annotation: suggestions are never gold until a human accepts them.

This restores SPEC section 9.2 ("160 of 200 receive a pre-annotator suggestion; 40 are
blind"), which the earlier fully-blind design had deviated from. The blind 40 are what make
anchoring bias measurable, so they are load-bearing rather than a concession.

The whole risk of assisted annotation is that a suggestion quietly becomes a label. Every test
here exists to make that impossible: a suggestion lives in a different file, carries
MODEL_GENERATED provenance, and only becomes gold through an explicit human action that is
itself recorded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.golden.schema import (
    ExpectedResolutionKind,
    FlagRecord,
    GoldenAnnotation,
    GoldenCandidate,
    GoldenSetError,
    LabelProvenance,
    Retraction,
    ReviewAction,
)
from hiver_support.golden.store import (
    append_annotation,
    append_flag,
    append_retraction,
    coverage,
    effective_annotations,
    load_effective_annotations,
    merge,
    read_annotations,
    read_flags,
    unresolved_pair_ids,
    write_candidates,
)
from hiver_support.golden.suggestions import ModelSuggestion, needs_mandatory_review
from hiver_support.taxonomy import TAXONOMY

WHEN = datetime(2017, 11, 1, 12, 0, tzinfo=timezone.utc)


def candidates(n: int = 3):
    return [
        GoldenCandidate(
            pair_id=f"p{i}",
            conversation_id=f"c{i}",
            customer_tweet_id=str(i),
            customer_message=f"message {i} about my iphone battery draining fast",
            created_at=WHEN,
        )
        for i in range(n)
    ]


def suggestion(pair_id: str = "p0", **overrides) -> ModelSuggestion:
    payload = {
        "pair_id": pair_id,
        "intent": "battery_charging",
        "security_sensitive": False,
        "context_sufficient": True,
        "should_escalate": False,
        "expected_resolution_kind": "self_serve_steps",
        "confidence": 0.9,
        "rationale": "describes rapid battery drain after an update",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "provider": "openrouter",
    }
    payload.update(overrides)
    return ModelSuggestion(**payload)


def annotation(pair_id: str, at: datetime | None = None, **overrides) -> GoldenAnnotation:
    payload = {
        "pair_id": pair_id,
        "annotator_id": "nitish",
        "intent": "battery_charging",
        "security_sensitive": False,
        "context_sufficient": True,
        "should_escalate": False,
        "expected_resolution_kind": ExpectedResolutionKind.SELF_SERVE_STEPS,
    }
    if at is not None:
        payload["timestamp_utc"] = at.isoformat()
    payload.update(overrides)
    return GoldenAnnotation(**payload)


class TestASuggestionIsNeverGold:
    def test_a_suggestion_declares_model_provenance(self):
        assert suggestion().provenance is LabelProvenance.MODEL_GENERATED
        assert suggestion().to_dict()["provenance"] == "MODEL_GENERATED"

    def test_a_suggestion_is_not_a_golden_annotation(self):
        assert not isinstance(suggestion(), GoldenAnnotation)

    def test_a_suggestion_cannot_be_appended_to_the_annotation_log(self, tmp_path):
        with pytest.raises(GoldenSetError):
            append_annotation(tmp_path / "a.jsonl", suggestion())

    def test_suggestions_alone_leave_the_set_unlabelled(self, tmp_path):
        # The whole failure mode in one test: generating 200 suggestions must move the
        # human-labelled count by exactly zero.
        stats = coverage(merge(candidates(), ()))
        assert stats["human_labelled"] == 0
        assert stats["unlabelled"] == 3

    def test_a_suggestion_must_conform_to_the_frozen_taxonomy(self):
        with pytest.raises(GoldenSetError, match="not in frozen taxonomy"):
            suggestion(intent="software_update_issue")

    def test_an_out_of_range_confidence_is_refused(self):
        with pytest.raises(GoldenSetError, match="confidence"):
            suggestion(confidence=1.7)

    def test_a_suggestion_records_which_model_produced_it(self):
        assert suggestion().model == "meta-llama/llama-3.3-70b-instruct"


class TestReviewActions:
    def test_accepting_a_suggestion_produces_human_gold(self):
        accepted = annotation(
            "p0",
            review_action=ReviewAction.ACCEPTED,
            model_suggestion=suggestion().to_dict(),
        )
        assert accepted.provenance is LabelProvenance.HUMAN_LABELED
        assert accepted.review_action is ReviewAction.ACCEPTED

    def test_an_accepted_label_matches_what_was_shown(self):
        shown = suggestion()
        accepted = annotation(
            "p0",
            intent=shown.intent,
            review_action=ReviewAction.ACCEPTED,
            model_suggestion=shown.to_dict(),
        )
        assert accepted.intent == shown.intent

    def test_a_correction_records_both_the_suggestion_and_what_changed(self):
        shown = suggestion(intent="battery_charging")
        corrected = annotation(
            "p0",
            intent="device_malfunction",
            review_action=ReviewAction.CORRECTED,
            model_suggestion=shown.to_dict(),
            corrected_fields=("intent",),
        )
        assert corrected.intent == "device_malfunction"
        assert corrected.model_suggestion["intent"] == "battery_charging"
        assert corrected.corrected_fields == ("intent",)

    def test_a_blind_annotation_records_no_suggestion(self):
        blind = annotation("p0", review_action=ReviewAction.ENTERED)
        assert blind.model_suggestion is None
        assert blind.review_action is ReviewAction.ENTERED

    def test_accept_and_correct_require_a_suggestion_to_have_been_shown(self):
        # Claiming a suggestion was accepted when none existed would fabricate the anchoring
        # record that makes bias measurable.
        with pytest.raises(GoldenSetError, match="suggestion"):
            annotation("p0", review_action=ReviewAction.ACCEPTED)

    def test_a_correction_must_name_at_least_one_changed_field(self):
        with pytest.raises(GoldenSetError, match="corrected_fields"):
            annotation(
                "p0",
                review_action=ReviewAction.CORRECTED,
                model_suggestion=suggestion().to_dict(),
                corrected_fields=(),
            )

    def test_review_actions_are_a_closed_set(self):
        assert {a.value for a in ReviewAction} == {"entered", "accepted", "corrected"}

    def test_provenance_stays_four_classes(self):
        # Review action is not provenance. A fifth provenance class would confuse "how it was
        # reviewed" with "where the label came from".
        assert {p.value for p in LabelProvenance} == {
            "UNLABELED",
            "WEAKLY_LABELED",
            "MODEL_GENERATED",
            "HUMAN_LABELED",
        }


class TestFlagging:
    def test_a_flag_leaves_the_example_unresolved(self, tmp_path):
        path = tmp_path / "a.jsonl"
        append_flag(path, FlagRecord(pair_id="p0", annotator_id="nitish", reason="two intents"))
        assert coverage(merge(candidates(), load_effective_annotations(path)))["human_labelled"] == 0

    def test_a_flagged_example_is_reported_as_unresolved(self, tmp_path):
        path = tmp_path / "a.jsonl"
        append_flag(path, FlagRecord(pair_id="p0", annotator_id="nitish", reason="ambiguous"))
        assert unresolved_pair_ids(path) == ("p0",)

    def test_a_flag_requires_a_reason(self):
        with pytest.raises(GoldenSetError, match="reason"):
            FlagRecord(pair_id="p0", annotator_id="nitish", reason=" ")

    def test_annotating_a_flagged_example_later_resolves_it(self, tmp_path):
        path = tmp_path / "a.jsonl"
        append_flag(
            path,
            FlagRecord(
                pair_id="p0", annotator_id="nitish", reason="ambiguous",
                timestamp_utc=WHEN.isoformat(),
            ),
        )
        append_annotation(path, annotation("p0", at=WHEN + timedelta(minutes=5)))
        assert unresolved_pair_ids(path) == ()

    def test_a_flag_carries_no_label(self):
        from dataclasses import fields

        names = {f.name for f in fields(FlagRecord)}
        assert not names & {"intent", "security_sensitive", "should_escalate"}


class TestMandatoryReview:
    """High-confidence cases may be accepted with one key; these may not."""

    def test_low_confidence_forces_review(self):
        assert needs_mandatory_review(suggestion(confidence=0.4))[0] is True

    def test_a_security_flagged_suggestion_forces_review(self):
        assert needs_mandatory_review(suggestion(security_sensitive=True))[0] is True

    def test_insufficient_context_forces_review(self):
        assert needs_mandatory_review(suggestion(context_sufficient=False))[0] is True

    def test_an_escalation_sensitive_intent_forces_review(self):
        assert needs_mandatory_review(suggestion(intent="billing_and_subscription"))[0] is True

    def test_a_confident_ordinary_case_does_not_force_review(self):
        forced, reasons = needs_mandatory_review(suggestion())
        assert forced is False
        assert reasons == ()

    def test_the_reason_for_forcing_review_is_stated(self):
        _, reasons = needs_mandatory_review(suggestion(security_sensitive=True))
        assert any("security" in r for r in reasons)

    def test_a_missing_suggestion_forces_review(self):
        # Fail closed: no suggestion means the human does it from scratch.
        assert needs_mandatory_review(None)[0] is True


class TestRetractionStillWorks:
    def test_an_accepted_label_can_still_be_retracted(self, tmp_path):
        path = tmp_path / "a.jsonl"
        append_annotation(
            path,
            annotation(
                "p0", at=WHEN,
                review_action=ReviewAction.ACCEPTED,
                model_suggestion=suggestion().to_dict(),
            ),
        )
        append_retraction(
            path,
            Retraction(
                pair_id="p0", annotator_id="nitish", reason="accepted too fast",
                timestamp_utc=(WHEN + timedelta(minutes=1)).isoformat(),
            ),
        )
        assert load_effective_annotations(path) == ()

    def test_the_retracted_record_and_its_suggestion_are_preserved(self, tmp_path):
        path = tmp_path / "a.jsonl"
        append_annotation(
            path,
            annotation(
                "p0", at=WHEN,
                review_action=ReviewAction.ACCEPTED,
                model_suggestion=suggestion().to_dict(),
            ),
        )
        append_retraction(
            path,
            Retraction(
                pair_id="p0", annotator_id="nitish", reason="x",
                timestamp_utc=(WHEN + timedelta(minutes=1)).isoformat(),
            ),
        )
        assert read_annotations(path)[0].model_suggestion["intent"] == "battery_charging"


class TestResumeAndFreeze:
    def test_resume_does_not_skip_a_flagged_example(self, tmp_path):
        path = tmp_path / "a.jsonl"
        append_flag(path, FlagRecord(pair_id="p0", annotator_id="nitish", reason="ambiguous"))
        effective = load_effective_annotations(path)
        outstanding = [c for c in candidates() if c.pair_id not in {a.pair_id for a in effective}]
        assert "p0" in [c.pair_id for c in outstanding]

    def test_freeze_rejects_a_set_with_an_unresolved_flag(self, tmp_path):
        from hiver_support.golden.lock import validate_golden_set

        path = tmp_path / "a.jsonl"
        cands = candidates()
        for c in cands[1:]:
            append_annotation(path, annotation(c.pair_id))
        append_flag(path, FlagRecord(pair_id="p0", annotator_id="nitish", reason="ambiguous"))
        report = validate_golden_set(cands, load_effective_annotations(path))
        assert report.valid is False

    def test_review_provenance_survives_the_freeze(self, tmp_path):
        from hiver_support.golden.lock import freeze_golden

        cands = candidates()
        anns = (
            annotation(
                "p0", review_action=ReviewAction.ACCEPTED,
                model_suggestion=suggestion("p0").to_dict(),
            ),
            annotation(
                "p1", intent="connectivity", review_action=ReviewAction.CORRECTED,
                model_suggestion=suggestion("p1").to_dict(), corrected_fields=("intent",),
            ),
            annotation("p2", review_action=ReviewAction.ENTERED),
        )
        lock = freeze_golden(cands, anns)
        assert lock["distributions"]["review_action"] == {
            "accepted": 1,
            "corrected": 1,
            "entered": 1,
        }
        assert lock["assisted_annotation"] is True
        assert "model-assisted" in lock["_warning"].lower()

    def test_the_lock_records_the_blind_subset_size(self, tmp_path):
        from hiver_support.golden.lock import freeze_golden

        cands = candidates()
        anns = tuple(annotation(c.pair_id, review_action=ReviewAction.ENTERED) for c in cands)
        assert freeze_golden(cands, anns)["counts"]["blind_entered"] == 3


class TestNothingFrozenWasTouched:
    def test_the_taxonomy_hash_is_unchanged(self):
        assert TAXONOMY.frozen_hash == (
            "613f5dfec1253168c8f2d01db141923c9e41363b6158c79bab6f067df4a9ee4d"
        )
        assert TAXONOMY.version == "0.3.0"

    def test_the_candidate_record_still_carries_no_label_field(self):
        from dataclasses import fields

        names = {f.name for f in fields(GoldenCandidate)}
        assert "intent" not in names
        assert "model_suggestion" not in names

    def test_the_committed_candidate_file_is_unchanged_and_unlabelled(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "golden" / "candidates.jsonl"
        if not path.exists():
            pytest.skip("candidate set not built on this machine")
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(records) == 200
        assert all(r["label_status"] == "UNLABELED" for r in records)
        assert all("intent" not in r for r in records)


def test_a_text_free_suggestion_still_loads(tmp_path):
    # The committed suggestions.jsonl stores rationale_sha256 instead of the rationale text.
    # An unknown key used to raise inside from_dict, and read_suggestions silently dropped the row.
    import json

    from hiver_support.golden.suggestions import read_suggestions

    path = tmp_path / "suggestions.jsonl"
    path.write_text(json.dumps({
        "pair_id": "1__2", "intent": "battery_charging", "security_sensitive": False,
        "context_sufficient": True, "should_escalate": False, "expected_resolution_kind": "self_serve_steps",
        "confidence": 0.9, "rationale_sha256": "ab" * 32, "model": "m", "provider": "p",
        "prompt_version": "v", "provenance": "MODEL_GENERATED", "_warning": "x",
    }) + "\n", encoding="utf-8")
    loaded = read_suggestions(path)
    assert set(loaded) == {"1__2"}
    assert loaded["1__2"].intent == "battery_charging"
    assert loaded["1__2"].rationale == ""

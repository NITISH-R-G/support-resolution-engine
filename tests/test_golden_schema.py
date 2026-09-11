"""The golden-set record contract.

These tests exist to make one class of failure impossible rather than merely discouraged:
a label that was not typed by a human ending up in the gold column. The project's whole
value proposition is evaluation integrity, and one audited public repo shipped keyword rules
in a file named ``human_review_pass.py`` — so "we would never do that" is not a control.

Two structural guarantees are asserted here:

1. A candidate record physically cannot carry the brand's reply or a label. There is no field
   to put one in, so the annotation tool cannot leak the answer and no script can backfill.
2. An annotation cannot be marked as anything but ``HUMAN_LABELED``, and cannot be attributed
   to a machine-sounding annotator.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

import pytest

from hiver_support.golden.schema import (
    ANNOTATION_VERSION,
    ContextTurn,
    ExpectedResolutionKind,
    GoldenAnnotation,
    GoldenCandidate,
    GoldenSetError,
    LabelConfidence,
    LabelProvenance,
    assert_candidate_schema_is_blind,
)
from hiver_support.taxonomy import TAXONOMY

WHEN = datetime(2017, 11, 1, 12, 0, tzinfo=timezone.utc)


def a_candidate(**overrides) -> GoldenCandidate:
    payload = {
        "pair_id": "100__101",
        "conversation_id": "c1",
        "customer_tweet_id": "100",
        "customer_message": "my iPhone battery drains in two hours since the update",
        "created_at": WHEN,
    }
    payload.update(overrides)
    return GoldenCandidate(**payload)


def an_annotation(**overrides) -> GoldenAnnotation:
    payload = {
        "pair_id": "100__101",
        "annotator_id": "nitish",
        "intent": "battery_charging",
        "security_sensitive": False,
        "context_sufficient": True,
        "should_escalate": False,
        "expected_resolution_kind": ExpectedResolutionKind.SELF_SERVE_STEPS,
    }
    payload.update(overrides)
    return GoldenAnnotation(**payload)


class TestProvenanceClasses:
    def test_exactly_four_provenance_classes_exist(self):
        assert {p.value for p in LabelProvenance} == {
            "UNLABELED",
            "WEAKLY_LABELED",
            "MODEL_GENERATED",
            "HUMAN_LABELED",
        }

    def test_only_human_labelled_is_usable_as_gold(self):
        assert LabelProvenance.HUMAN_LABELED.is_gold
        for other in (
            LabelProvenance.UNLABELED,
            LabelProvenance.WEAKLY_LABELED,
            LabelProvenance.MODEL_GENERATED,
        ):
            assert not other.is_gold


class TestCandidateIsStructurallyBlind:
    def test_candidate_has_no_field_that_could_hold_the_brands_reply(self):
        names = {f.name for f in fields(GoldenCandidate)}
        leaky = {n for n in names if any(w in n for w in ("reply", "support", "resolution"))}
        assert leaky == set(), f"candidate record can carry the answer: {leaky}"

    def test_candidate_has_no_label_field_to_backfill(self):
        names = {f.name for f in fields(GoldenCandidate)}
        assert "intent" not in names
        assert "should_escalate" not in names
        assert "gold" not in names

    def test_serialised_candidate_exposes_no_answer_and_no_label(self):
        payload = a_candidate().to_dict()
        for key in payload:
            assert not any(w in key for w in ("reply", "support", "resolution", "intent"))

    def test_the_blindness_assertion_is_callable_and_passes(self):
        assert_candidate_schema_is_blind()

    def test_candidate_is_always_unlabeled(self):
        assert a_candidate().label_status is LabelProvenance.UNLABELED
        assert a_candidate().to_dict()["label_status"] == "UNLABELED"

    def test_candidate_cannot_be_constructed_claiming_to_be_labelled(self):
        with pytest.raises(GoldenSetError, match="UNLABELED"):
            a_candidate(label_status=LabelProvenance.HUMAN_LABELED)

    def test_candidate_requires_a_pair_id_and_a_message(self):
        with pytest.raises(GoldenSetError):
            a_candidate(pair_id="")
        with pytest.raises(GoldenSetError):
            a_candidate(customer_message="   ")

    def test_context_turns_round_trip(self):
        turn = ContextTurn(author_role="customer", text="hello", created_at=WHEN)
        candidate = a_candidate(context=(turn,))
        assert candidate.to_dict()["context"] == [
            {"author_role": "customer", "text": "hello", "created_at": WHEN.isoformat()}
        ]

    def test_context_author_role_is_restricted(self):
        with pytest.raises(GoldenSetError, match="author_role"):
            ContextTurn(author_role="agent", text="hi", created_at=WHEN)

    def test_candidate_round_trips_through_a_dict(self):
        candidate = a_candidate(context=(ContextTurn("brand", "hi", WHEN),))
        assert GoldenCandidate.from_dict(candidate.to_dict()) == candidate


class TestAnnotationCannotBeFabricated:
    def test_provenance_is_human_labelled(self):
        assert an_annotation().provenance is LabelProvenance.HUMAN_LABELED

    @pytest.mark.parametrize(
        "provenance",
        [
            LabelProvenance.UNLABELED,
            LabelProvenance.WEAKLY_LABELED,
            LabelProvenance.MODEL_GENERATED,
        ],
    )
    def test_an_annotation_cannot_declare_a_non_human_provenance(self, provenance):
        with pytest.raises(GoldenSetError, match="HUMAN_LABELED"):
            an_annotation(provenance=provenance)

    def test_an_empty_annotator_id_is_refused(self):
        with pytest.raises(GoldenSetError, match="annotator_id"):
            an_annotation(annotator_id="  ")

    @pytest.mark.parametrize(
        "annotator", ["llm", "GPT-4", "claude", "model", "auto", "script", "weak_labels"]
    )
    def test_machine_sounding_annotator_ids_are_refused(self, annotator):
        with pytest.raises(GoldenSetError, match="annotator_id"):
            an_annotation(annotator_id=annotator)


class TestAnnotationBindsTheFrozenTaxonomy:
    def test_intent_must_be_in_the_frozen_taxonomy(self):
        with pytest.raises(GoldenSetError, match="not in frozen taxonomy"):
            an_annotation(intent="software_update_issue")

    def test_every_frozen_intent_is_accepted(self):
        for intent in TAXONOMY.names:
            assert an_annotation(intent=intent).intent == intent

    def test_taxonomy_hash_is_recorded_and_must_match(self):
        assert an_annotation().taxonomy_hash == TAXONOMY.frozen_hash
        with pytest.raises(GoldenSetError, match="taxonomy"):
            an_annotation(taxonomy_hash="0" * 64)

    def test_annotation_version_is_recorded(self):
        assert an_annotation().annotation_version == ANNOTATION_VERSION


class TestAnnotationValidation:
    @pytest.mark.parametrize(
        "field", ["security_sensitive", "context_sufficient", "should_escalate", "is_ambiguous"]
    )
    def test_flags_must_be_real_bools_not_truthy_values(self, field):
        with pytest.raises(GoldenSetError, match=field):
            an_annotation(**{field: 1})

    def test_alternative_intent_requires_the_ambiguity_flag(self):
        with pytest.raises(GoldenSetError, match="is_ambiguous"):
            an_annotation(alternative_intent="device_malfunction")

    def test_alternative_intent_must_differ_from_the_primary(self):
        with pytest.raises(GoldenSetError, match="differ"):
            an_annotation(is_ambiguous=True, alternative_intent="battery_charging")

    def test_alternative_intent_must_be_in_the_taxonomy(self):
        with pytest.raises(GoldenSetError, match="not in frozen taxonomy"):
            an_annotation(is_ambiguous=True, alternative_intent="nonsense")

    def test_ambiguity_without_a_runner_up_is_allowed(self):
        # The annotator may know a case is ambiguous without a clean second reading. Forcing
        # one would manufacture the false precision the mechanism exists to avoid.
        assert an_annotation(is_ambiguous=True).alternative_intent is None

    def test_expected_resolution_kind_is_a_closed_set(self):
        assert {k.value for k in ExpectedResolutionKind} == {
            "self_serve_steps",
            "information",
            "human_action_required",
            "no_resolution_possible",
            "unclear",
        }

    def test_expected_resolution_kind_rejects_an_invented_value(self):
        with pytest.raises(GoldenSetError, match="expected_resolution_kind"):
            an_annotation(expected_resolution_kind="refund_it")

    def test_label_confidence_is_a_closed_set(self):
        assert {c.value for c in LabelConfidence} == {"high", "medium", "low"}

    def test_pass_number_must_be_positive(self):
        with pytest.raises(GoldenSetError, match="pass_number"):
            an_annotation(pass_number=0)

    def test_timestamp_is_recorded_automatically_when_not_supplied(self):
        assert an_annotation().timestamp_utc.endswith("+00:00")

    def test_annotation_round_trips_through_a_dict(self):
        annotation = an_annotation(
            is_ambiguous=True,
            alternative_intent="device_malfunction",
            label_confidence=LabelConfidence.LOW,
            notes="could be either",
        )
        assert GoldenAnnotation.from_dict(annotation.to_dict()) == annotation

    def test_serialised_annotation_states_its_provenance_explicitly(self):
        assert an_annotation().to_dict()["provenance"] == "HUMAN_LABELED"

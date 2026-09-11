"""Retracting an annotation without deleting it.

A real incident forced this. Pass 1 was started before the annotation guide had been read, and
one example (`101236__101237`) was labelled carelessly and saved. The log is append-only, the
CLI skips examples already annotated in that pass, and nothing in the tooling could mark a
record as invalid — so there were only bad options: delete the line and lose the audit trail,
quietly overwrite it and leave an invalid label indistinguishable from a considered one, or
abandon pass 1 entirely.

A retraction is the supported fourth option. It is **another append-only record**, not an
edit: the original annotation stays exactly as written, a retraction states who invalidated it
and why, and everything downstream treats the example as unannotated again.

Two properties carry the weight:

* **A retraction never destroys.** The retracted label remains readable in the log forever, so
  "this was labelled badly and withdrawn" is a fact anyone can check rather than a claim.
* **A retraction is bounded in time.** It invalidates records written *at or before* it, so
  re-annotating the same example afterwards works normally and is not swept up.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hiver_support.golden.schema import (
    ExpectedResolutionKind,
    GoldenAnnotation,
    GoldenCandidate,
    GoldenSetError,
    LabelProvenance,
    Retraction,
)
from hiver_support.golden.store import (
    append_annotation,
    append_retraction,
    coverage,
    effective_annotations,
    latest_by_pair,
    load_gold,
    merge,
    read_annotations,
    read_retractions,
    write_candidates,
)

WHEN = datetime(2017, 11, 1, 12, 0, tzinfo=timezone.utc)


def candidates(n: int = 3) -> list[GoldenCandidate]:
    return [
        GoldenCandidate(
            pair_id=f"p{i}",
            conversation_id=f"c{i}",
            customer_tweet_id=str(i),
            customer_message=f"message {i} about my iphone battery",
            created_at=WHEN,
        )
        for i in range(n)
    ]


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


def retraction(pair_id: str, at: datetime | None = None, **overrides) -> Retraction:
    payload = {
        "pair_id": pair_id,
        "annotator_id": "nitish",
        "reason": "labelled before reading the annotation guide; answers were not considered",
    }
    if at is not None:
        payload["timestamp_utc"] = at.isoformat()
    payload.update(overrides)
    return Retraction(**payload)


class TestTheRetractionRecord:
    def test_it_requires_a_reason(self):
        # A retraction with no reason is a deletion with extra steps. The audit trail is the
        # whole point of the mechanism.
        with pytest.raises(GoldenSetError, match="reason"):
            retraction("p0", reason="   ")

    def test_it_requires_a_named_annotator(self):
        with pytest.raises(GoldenSetError, match="annotator_id"):
            retraction("p0", annotator_id="")

    def test_a_machine_sounding_annotator_is_refused(self):
        with pytest.raises(GoldenSetError, match="annotator_id"):
            retraction("p0", annotator_id="llm")

    def test_it_requires_a_pair_id(self):
        with pytest.raises(GoldenSetError, match="pair_id"):
            retraction("")

    def test_it_records_when_it_was_made(self):
        assert retraction("p0").timestamp_utc.endswith("+00:00")

    def test_it_is_tagged_as_a_retraction_not_an_annotation(self):
        payload = retraction("p0").to_dict()
        assert payload["record_type"] == "retraction"
        assert "intent" not in payload

    def test_it_carries_no_label_of_any_kind(self):
        # A retraction must not be a back door for writing a label.
        from dataclasses import fields

        names = {f.name for f in fields(Retraction)}
        assert not names & {"intent", "security_sensitive", "should_escalate"}

    def test_it_round_trips(self):
        original = retraction("p0")
        assert Retraction.from_dict(original.to_dict()) == original


class TestTheLogKeepsBothRecordKinds:
    def test_a_retraction_is_appended_not_substituted(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        append_retraction(path, retraction("p0"))
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["intent"] == "battery_charging"

    def test_the_retracted_label_is_still_readable_afterwards(self, tmp_path):
        # "This was labelled badly and withdrawn" must stay a checkable fact.
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", intent="connectivity"))
        append_retraction(path, retraction("p0"))
        assert read_annotations(path)[0].intent == "connectivity"

    def test_reading_annotations_ignores_retraction_records(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        append_retraction(path, retraction("p0"))
        assert all(isinstance(a, GoldenAnnotation) for a in read_annotations(path))

    def test_reading_retractions_returns_them(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        append_retraction(path, retraction("p0"))
        assert [r.pair_id for r in read_retractions(path)] == ["p0"]

    def test_a_log_with_no_retractions_reads_cleanly(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        assert read_retractions(path) == ()

    def test_only_a_retraction_object_may_be_appended_as_one(self, tmp_path):
        with pytest.raises(GoldenSetError):
            append_retraction(tmp_path / "a.jsonl", annotation("p0"))


class TestRetractedLabelsNeverBecomeGold:
    def test_a_retracted_annotation_is_dropped_from_the_effective_set(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", at=WHEN))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1)))
        assert effective_annotations(read_annotations(path), read_retractions(path)) == ()

    def test_the_example_reads_as_unlabelled_again(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", at=WHEN))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1)))
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        stats = coverage(merge(candidates(), effective))
        assert stats["human_labelled"] == 0
        assert stats["unlabelled"] == 3

    def test_other_examples_are_untouched(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", at=WHEN))
        append_annotation(path, annotation("p1", at=WHEN))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1)))
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        assert [a.pair_id for a in effective] == ["p1"]

    def test_load_gold_refuses_a_set_whose_only_label_was_retracted(self, tmp_path):
        cpath, apath = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
        write_candidates(cpath, candidates())
        append_annotation(apath, annotation("p0", at=WHEN))
        append_retraction(apath, retraction("p0", at=WHEN + timedelta(minutes=1)))
        with pytest.raises(GoldenSetError, match="0 of 3"):
            load_gold(cpath, apath)

    def test_a_retracted_label_cannot_reach_the_freeze(self, tmp_path):
        from hiver_support.golden.lock import validate_golden_set

        path = tmp_path / "annotations.jsonl"
        for c in candidates():
            append_annotation(path, annotation(c.pair_id, at=WHEN))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1)))
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        report = validate_golden_set(candidates(), effective)
        assert report.valid is False
        assert any("p0" in problem for problem in report.problems)


class TestRetractionIsBoundedInTime:
    def test_a_later_clean_annotation_supersedes_the_retraction(self, tmp_path):
        # The point of the whole exercise: the example can be annotated properly afterwards.
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", at=WHEN, intent="connectivity"))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1)))
        append_annotation(
            path, annotation("p0", at=WHEN + timedelta(minutes=2), intent="battery_charging")
        )
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        assert len(effective) == 1
        assert effective[0].intent == "battery_charging"

    def test_a_retraction_does_not_reach_forward_indefinitely(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_retraction(path, retraction("p0", at=WHEN))
        append_annotation(path, annotation("p0", at=WHEN + timedelta(hours=1)))
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        assert len(effective) == 1

    def test_a_retraction_only_affects_its_own_pass(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", at=WHEN, pass_number=1))
        append_annotation(path, annotation("p0", at=WHEN, pass_number=2))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1), pass_number=1))
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        assert [a.pass_number for a in effective] == [2]

    def test_retracting_an_example_that_was_never_annotated_is_harmless(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p1", at=WHEN))
        append_retraction(path, retraction("p0", at=WHEN + timedelta(minutes=1)))
        effective = effective_annotations(read_annotations(path), read_retractions(path))
        assert [a.pair_id for a in effective] == ["p1"]


class TestProvenanceOfARetraction:
    def test_a_retraction_is_not_itself_a_label(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_retraction(path, retraction("p0"))
        assert read_annotations(path) == ()
        assert coverage(merge(candidates(), ()))["human_labelled"] == 0

    def test_no_provenance_class_is_invented_for_retractions(self):
        # Four provenance classes, not five. A retraction describes an annotation; it is not a
        # kind of label.
        assert {p.value for p in LabelProvenance} == {
            "UNLABELED",
            "WEAKLY_LABELED",
            "MODEL_GENERATED",
            "HUMAN_LABELED",
        }

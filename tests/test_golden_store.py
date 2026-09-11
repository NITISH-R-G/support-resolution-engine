"""Persistence for the golden set, and the refusal that makes it trustworthy.

The single most important behaviour in this module is a *failure*: ``load_gold`` raises rather
than returning a partially-labelled set. An evaluation harness that quietly scores 37 of 200
examples reports a number that looks like a result and is not one, and the missing 163 are
invisible in the output.

The second is separation: candidates and annotations live in different files. Nothing that
writes candidates can write a label, and the annotation log is append-only so a re-labelling
pass cannot erase what it disagreed with.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hiver_support.golden.schema import (
    ExpectedResolutionKind,
    GoldenAnnotation,
    GoldenCandidate,
    GoldenSetError,
    LabelProvenance,
)
from hiver_support.golden.store import (
    append_annotation,
    coverage,
    latest_by_pair,
    load_gold,
    merge,
    read_annotations,
    read_candidates,
    write_candidates,
)

WHEN = datetime(2017, 11, 1, 12, 0, tzinfo=timezone.utc)


def candidates(n: int = 3) -> list[GoldenCandidate]:
    return [
        GoldenCandidate(
            pair_id=f"p{i}",
            conversation_id=f"c{i}",
            customer_tweet_id=str(i),
            customer_message=f"message number {i} about my iphone battery",
            created_at=WHEN,
        )
        for i in range(n)
    ]


def annotation(pair_id: str, **overrides) -> GoldenAnnotation:
    payload = {
        "pair_id": pair_id,
        "annotator_id": "nitish",
        "intent": "battery_charging",
        "security_sensitive": False,
        "context_sufficient": True,
        "should_escalate": False,
        "expected_resolution_kind": ExpectedResolutionKind.SELF_SERVE_STEPS,
    }
    payload.update(overrides)
    return GoldenAnnotation(**payload)


class TestCandidateFile:
    def test_candidates_round_trip_through_jsonl(self, tmp_path):
        path = tmp_path / "candidates.jsonl"
        write_candidates(path, candidates())
        assert read_candidates(path) == tuple(candidates())

    def test_writing_returns_a_content_hash_for_pinning(self, tmp_path):
        path = tmp_path / "candidates.jsonl"
        digest = write_candidates(path, candidates())
        assert len(digest) == 64

    def test_it_refuses_to_overwrite_an_existing_candidate_file(self, tmp_path):
        # Re-sampling after annotation has begun would silently discard human work.
        path = tmp_path / "candidates.jsonl"
        write_candidates(path, candidates())
        with pytest.raises(GoldenSetError, match="exists"):
            write_candidates(path, candidates())

    def test_overwriting_is_possible_only_by_asking_for_it_explicitly(self, tmp_path):
        path = tmp_path / "candidates.jsonl"
        write_candidates(path, candidates())
        write_candidates(path, candidates(2), overwrite=True)
        assert len(read_candidates(path)) == 2

    def test_the_written_file_contains_no_label_and_no_reply(self, tmp_path):
        path = tmp_path / "candidates.jsonl"
        write_candidates(path, candidates())
        raw = path.read_text(encoding="utf-8")
        assert '"intent"' not in raw
        assert '"support_text"' not in raw
        assert raw.count('"UNLABELED"') == 3

    def test_reading_a_missing_candidate_file_raises(self, tmp_path):
        with pytest.raises(GoldenSetError, match="not found"):
            read_candidates(tmp_path / "absent.jsonl")

    def test_duplicate_candidates_are_refused_on_write(self, tmp_path):
        duplicated = candidates() + candidates(1)
        with pytest.raises(GoldenSetError, match="duplicate"):
            write_candidates(tmp_path / "candidates.jsonl", duplicated)


class TestAnnotationLog:
    def test_annotations_round_trip(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        assert read_annotations(path)[0].pair_id == "p0"

    def test_the_log_is_append_only(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", notes="first"))
        append_annotation(path, annotation("p0", notes="second", pass_number=2))
        stored = read_annotations(path)
        assert [a.notes for a in stored] == ["first", "second"]

    def test_reading_an_absent_log_returns_nothing_rather_than_raising(self, tmp_path):
        # An unstarted annotation effort is a legitimate state; a missing candidate file is not.
        assert read_annotations(tmp_path / "absent.jsonl") == ()

    def test_a_later_pass_supersedes_an_earlier_one_without_deleting_it(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", intent="battery_charging"))
        append_annotation(path, annotation("p0", intent="device_malfunction", pass_number=2))
        latest = latest_by_pair(read_annotations(path))
        assert latest["p0"].intent == "device_malfunction"
        assert len(read_annotations(path)) == 2

    def test_a_specific_pass_can_be_isolated_for_agreement_analysis(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0", intent="battery_charging"))
        append_annotation(path, annotation("p0", intent="device_malfunction", pass_number=2))
        first = latest_by_pair(read_annotations(path), pass_number=1)
        assert first["p0"].intent == "battery_charging"

    def test_a_corrupt_line_names_itself_rather_than_being_skipped(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        with pytest.raises(GoldenSetError, match="line 2"):
            read_annotations(path)

    def test_an_annotation_claiming_non_human_provenance_is_rejected_on_read(self, tmp_path):
        path = tmp_path / "annotations.jsonl"
        payload = annotation("p0").to_dict()
        payload["provenance"] = "MODEL_GENERATED"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(GoldenSetError, match="HUMAN_LABELED"):
            read_annotations(path)


class TestMergeAndCoverage:
    def test_unannotated_candidates_merge_to_unlabelled_examples(self, tmp_path):
        examples = merge(candidates(), ())
        assert all(not e.is_labelled for e in examples)
        assert all(e.provenance is LabelProvenance.UNLABELED for e in examples)

    def test_merge_attaches_the_latest_annotation(self):
        examples = merge(candidates(), (annotation("p1"),))
        labelled = {e.candidate.pair_id: e.is_labelled for e in examples}
        assert labelled == {"p0": False, "p1": True, "p2": False}

    def test_merge_rejects_an_annotation_for_an_unknown_candidate(self):
        with pytest.raises(GoldenSetError, match="unknown"):
            merge(candidates(), (annotation("not-a-candidate"),))

    def test_coverage_counts_labelled_and_unlabelled_separately(self):
        stats = coverage(merge(candidates(), (annotation("p1"),)))
        assert stats["total"] == 3
        assert stats["human_labelled"] == 1
        assert stats["unlabelled"] == 2
        assert stats["complete"] is False

    def test_coverage_never_reports_weak_or_model_labels_as_progress(self):
        stats = coverage(merge(candidates(), (annotation("p1"),)))
        assert stats["weakly_labelled"] == 0
        assert stats["model_generated"] == 0


class TestTheHarnessRefusesToScoreUnlabelledData:
    def test_load_gold_raises_when_any_example_is_unlabelled(self, tmp_path):
        cpath, apath = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
        write_candidates(cpath, candidates())
        append_annotation(apath, annotation("p0"))
        with pytest.raises(GoldenSetError, match="1 of 3"):
            load_gold(cpath, apath)

    def test_the_error_names_the_unlabelled_examples(self, tmp_path):
        cpath, apath = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
        write_candidates(cpath, candidates())
        append_annotation(apath, annotation("p0"))
        with pytest.raises(GoldenSetError, match="p1"):
            load_gold(cpath, apath)

    def test_load_gold_succeeds_once_every_example_is_human_labelled(self, tmp_path):
        cpath, apath = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
        write_candidates(cpath, candidates())
        for pair_id in ("p0", "p1", "p2"):
            append_annotation(apath, annotation(pair_id))
        gold = load_gold(cpath, apath)
        assert len(gold) == 3
        assert all(e.provenance is LabelProvenance.HUMAN_LABELED for e in gold)

    def test_a_partial_set_can_be_inspected_only_by_asking_for_it_explicitly(self, tmp_path):
        cpath, apath = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
        write_candidates(cpath, candidates())
        append_annotation(apath, annotation("p0"))
        partial = load_gold(cpath, apath, require_complete=False)
        assert sum(e.is_labelled for e in partial) == 1

    def test_load_gold_with_no_annotations_at_all_still_raises(self, tmp_path):
        cpath, apath = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
        write_candidates(cpath, candidates())
        with pytest.raises(GoldenSetError, match="0 of 3"):
            load_gold(cpath, apath)

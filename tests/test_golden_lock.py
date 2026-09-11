"""Validation and freezing of the golden set, and the wall that keeps it evaluation-only.

A golden set is only worth what its provenance is worth. Three things are enforced here:

* **It cannot be frozen early.** A partially annotated set is not gold, and freezing one would
  stamp a hash onto something that is still changing.
* **Every label is human.** The freeze re-checks provenance rather than trusting the file it
  is reading, because a hash over contaminated labels certifies contamination.
* **Training code cannot reach it.** Asserted against the import graph, not by convention —
  the same structural control used for annotation blindness.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hiver_support.golden.lock import (
    GoldenLockError,
    freeze_golden,
    validate_golden_set,
    verify_lock,
)
from hiver_support.golden.schema import (
    ExpectedResolutionKind,
    GoldenAnnotation,
    GoldenCandidate,
)

WHEN = datetime(2017, 11, 1, 12, 0, tzinfo=timezone.utc)


def candidates(n: int = 3) -> tuple[GoldenCandidate, ...]:
    return tuple(
        GoldenCandidate(
            pair_id=f"p{i}",
            conversation_id=f"c{i}",
            customer_tweet_id=str(i),
            customer_message=f"message {i} about my iphone battery draining",
            created_at=WHEN,
        )
        for i in range(n)
    )


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


def complete(n: int = 3):
    cands = candidates(n)
    return cands, tuple(annotation(c.pair_id) for c in cands)


class TestValidationRefusesAnIncompleteSet:
    def test_a_set_with_missing_labels_is_invalid(self):
        cands = candidates(3)
        report = validate_golden_set(cands, (annotation("p0"),))
        assert report.valid is False
        assert any("unlabelled" in p.lower() for p in report.problems)

    def test_the_problem_names_the_missing_examples(self):
        report = validate_golden_set(candidates(3), (annotation("p0"),))
        assert any("p1" in p for p in report.problems)

    def test_a_fully_annotated_set_is_valid(self):
        assert validate_golden_set(*complete()).valid is True

    def test_an_empty_set_is_invalid(self):
        assert validate_golden_set((), ()).valid is False


class TestValidationChecksEveryRequiredProperty:
    def test_an_annotation_for_an_unknown_candidate_is_caught(self):
        cands, anns = complete()
        report = validate_golden_set(cands, anns + (annotation("not-a-candidate"),))
        assert report.valid is False
        assert any("unknown" in p.lower() for p in report.problems)

    def test_every_required_field_is_present_on_every_annotation(self):
        report = validate_golden_set(*complete())
        assert report.counts["human_labelled"] == 3
        assert report.counts["required_fields_missing"] == 0

    def test_taxonomy_binding_is_re_checked_at_freeze_time(self):
        # The dataclass validates on construction, but the freeze must not trust the file it
        # is reading: a hand-edited annotations.jsonl is exactly what this catches.
        report = validate_golden_set(*complete())
        assert report.counts["taxonomy_mismatches"] == 0

    def test_a_hand_edited_out_of_taxonomy_label_is_rejected(self, tmp_path):
        from hiver_support.golden.store import append_annotation, read_annotations

        path = tmp_path / "annotations.jsonl"
        append_annotation(path, annotation("p0"))
        payload = json.loads(path.read_text(encoding="utf-8").strip())
        payload["intent"] = "software_update_issue"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(Exception):
            read_annotations(path)

    def test_a_non_human_provenance_makes_the_set_invalid(self):
        cands, anns = complete()

        class Fake:
            pair_id = "p0"
            provenance = type("P", (), {"is_gold": False, "value": "MODEL_GENERATED"})()

        report = validate_golden_set(cands, tuple(anns) + (Fake(),))
        assert report.valid is False
        assert any("provenance" in p.lower() for p in report.problems)

    def test_duplicate_candidate_ids_are_caught(self):
        cands = candidates(2)
        report = validate_golden_set(cands + (cands[0],), tuple(annotation(c.pair_id) for c in cands))
        assert report.valid is False
        assert any("duplicate" in p.lower() for p in report.problems)


class TestFreezing:
    def test_an_incomplete_set_cannot_be_frozen(self):
        with pytest.raises(GoldenLockError, match="not valid"):
            freeze_golden(candidates(3), (annotation("p0"),))

    def test_freezing_produces_a_stable_content_hash(self):
        cands, anns = complete()
        first = freeze_golden(cands, anns)
        second = freeze_golden(cands, anns)
        assert first["content_sha256"] == second["content_sha256"]
        assert len(first["content_sha256"]) == 64

    def test_the_hash_covers_the_labels_not_only_the_messages(self):
        cands, anns = complete()
        baseline = freeze_golden(cands, anns)["content_sha256"]
        changed = list(anns)
        changed[0] = annotation("p0", intent="device_malfunction")
        assert freeze_golden(cands, tuple(changed))["content_sha256"] != baseline

    def test_the_hash_is_independent_of_annotation_order(self):
        cands, anns = complete()
        assert (
            freeze_golden(cands, tuple(reversed(anns)))["content_sha256"]
            == freeze_golden(cands, anns)["content_sha256"]
        )

    def test_the_lock_declares_the_set_evaluation_only(self):
        lock = freeze_golden(*complete())
        assert lock["usage"] == "EVALUATION_ONLY"
        assert lock["may_be_used_for_training"] is False

    def test_the_lock_records_distributions_and_provenance(self):
        lock = freeze_golden(*complete())
        assert lock["counts"]["human_labelled"] == 3
        assert lock["distributions"]["intent"]["battery_charging"] == 3
        assert lock["taxonomy_hash"]
        assert lock["annotators"] == ["nitish"]

    def test_verify_lock_accepts_the_set_it_was_made_from(self):
        cands, anns = complete()
        assert verify_lock(freeze_golden(cands, anns), cands, anns) is True

    def test_verify_lock_rejects_a_changed_label(self):
        cands, anns = complete()
        lock = freeze_golden(cands, anns)
        tampered = list(anns)
        tampered[0] = annotation("p0", intent="connectivity")
        with pytest.raises(GoldenLockError, match="hash"):
            verify_lock(lock, cands, tuple(tampered))


class TestTrainingCodeCannotReachTheGoldenSet:
    """Structural, like the annotation-blindness guard. Convention would not survive a deadline."""

    @staticmethod
    def _imports(path):
        import ast

        modules = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        return modules

    def test_no_classifier_module_imports_the_golden_package(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "hiver_support"
        offenders = [
            str(p.relative_to(root))
            for p in list((root / "classifier").glob("*.py")) + list((root / "data").glob("*.py"))
            if any(m.startswith("hiver_support.golden") for m in self._imports(p))
        ]
        assert offenders == [], (
            f"training-path modules import the golden set: {offenders}. The golden set is "
            f"evaluation-only; anything that fits a model must never be able to read it."
        )

    def test_no_training_script_imports_the_golden_package(self):
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        offenders = [
            p.name
            for p in scripts.glob("train_*.py")
            if any(m.startswith("hiver_support.golden") for m in self._imports(p))
        ]
        assert offenders == []

    def test_the_agent_runtime_does_not_import_the_golden_package(self):
        from pathlib import Path

        agent = Path(__file__).resolve().parents[1] / "src" / "hiver_support" / "agent"
        offenders = [
            p.name
            for p in agent.glob("*.py")
            if any(m.startswith("hiver_support.golden") for m in self._imports(p))
        ]
        assert offenders == []

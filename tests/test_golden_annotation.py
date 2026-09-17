"""The annotation tool's blindness, enforced against its source rather than against memory.

Pass 1 must be blind (``docs/GOLDEN_SET.md`` section 7). "Blind" is easy to state and easy to
erode: one helpful line showing the weak label to speed annotation up, and every gold label
downstream is anchored to the rules the system is being scored against.

So blindness is asserted here as a property of the file — what it imports, what it reads, what
it prints. A convenience import fails the build.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hiver_support.golden.schema import GoldenAnnotation, GoldenSetError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "annotate_golden.py"
BUILDER = ROOT / "scripts" / "build_golden_candidates.py"

# Modules whose output would anchor an annotator. The classifier and the agent are the system
# under test; the weak labeller produced the training labels and the sampling strata.
FORBIDDEN_MODULES = (
    "hiver_support.classifier",
    "hiver_support.agent",
    "hiver_support.classifier.weak_labels",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class TestTheAnnotationToolIsStructurallyBlind:
    def test_it_imports_no_classifier_retriever_or_generator(self):
        imported = _imported_modules(SCRIPT)
        leaks = sorted(m for m in imported if m.startswith(FORBIDDEN_MODULES))
        assert leaks == [], (
            f"scripts/annotate_golden.py imports {leaks}. Pass 1 is blind: a suggestion, a "
            f"confidence or a retrieved case would anchor every gold label to the system "
            f"being scored (docs/GOLDEN_SET.md section 7)."
        )

    def test_it_never_reads_the_sampling_frame(self):
        # The frame holds the stratum and the weak label. Both are hints.
        source = SCRIPT.read_text(encoding="utf-8")
        assert "sampling_frame" not in source

    def test_it_reads_only_the_candidate_and_annotation_files(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "local_candidates" in source
        assert "annotations.jsonl" in source
        assert "twcs.csv" not in source

    def test_it_requires_a_named_annotator(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "--annotator is required" in source

    def test_it_writes_through_the_append_only_store(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "append_annotation" in source
        # Anything that opens the annotation log directly could rewrite history, and
        # disagreement between passes is the measurement.
        assert "ANNOTATIONS.write_text" not in source

    def test_the_builder_may_use_weak_labels_because_it_never_shows_them(self):
        # The sampler legitimately stratifies on weak labels; it writes them to a separate
        # file the annotator never opens. The distinction is the whole control.
        assert "sampling" in " ".join(_imported_modules(BUILDER))


class TestBulkFillingIsRefused:
    def test_a_piped_stdin_is_refused_rather_than_silently_accepting_defaults(self, monkeypatch):
        import importlib.util

        spec = importlib.util.spec_from_file_location("annotate_golden", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def closed_stdin(_prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", closed_stdin)
        with pytest.raises(GoldenSetError, match="interactive"):
            module._ask("intent: ")


class TestAnnotationsCarryTheirOwnProvenance:
    def test_a_saved_annotation_names_its_human_and_its_pass(self):
        annotation = GoldenAnnotation(
            pair_id="p1",
            annotator_id="nitish",
            intent="battery_charging",
            security_sensitive=False,
            context_sufficient=True,
            should_escalate=False,
            expected_resolution_kind="self_serve_steps",
            pass_number=2,
        )
        payload = annotation.to_dict()
        assert payload["annotator_id"] == "nitish"
        assert payload["pass_number"] == 2
        assert payload["provenance"] == "HUMAN_LABELED"
        assert payload["annotation_version"]


class TestTheInteractiveFlowActuallyProducesAValidAnnotation:
    """Drives the real prompt sequence with scripted keystrokes.

    Without this the tool could only be verified by a human typing 200 labels, and a
    tool that crashes on example 3 would waste the most expensive resource in the project.
    Scripted keystrokes here are test input, not labels: nothing is written to
    ``data/golden/annotations.jsonl``.
    """

    @staticmethod
    def _module():
        import importlib.util

        spec = importlib.util.spec_from_file_location("annotate_golden", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _candidate():
        from datetime import datetime, timezone

        from hiver_support.golden.schema import GoldenCandidate

        return GoldenCandidate(
            pair_id="p1",
            conversation_id="c1",
            customer_tweet_id="1",
            customer_message="my battery drains in two hours since the update",
            created_at=datetime(2017, 11, 1, tzinfo=timezone.utc),
        )

    def _run(self, monkeypatch, keystrokes):
        module = self._module()
        answers = iter(keystrokes)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
        return module._annotate_one(self._candidate(), "nitish", 1)

    def test_a_straightforward_pass_produces_a_human_labelled_annotation(self, monkeypatch):
        annotation = self._run(
            monkeypatch,
            ["battery_charging", "n", "y", "n", "self_serve_steps", "n", "high", "", ""],
        )
        assert annotation.intent == "battery_charging"
        assert annotation.should_escalate is False
        assert annotation.provenance.is_gold
        assert annotation.seconds_spent >= 0

    def test_intents_can_be_chosen_by_number(self, monkeypatch):
        from hiver_support.taxonomy import TAXONOMY

        annotation = self._run(
            monkeypatch, ["1", "n", "y", "n", "1", "n", "1", "", ""]
        )
        assert annotation.intent == TAXONOMY.names[0]

    def test_an_ambiguous_case_records_its_runner_up(self, monkeypatch):
        annotation = self._run(
            monkeypatch,
            [
                "battery_charging", "n", "y", "n", "self_serve_steps",
                "y", "device_malfunction", "low", "explain the battery health screen", "torn",
            ],
        )
        assert annotation.is_ambiguous is True
        assert annotation.alternative_intent == "device_malfunction"
        assert annotation.label_confidence.value == "low"
        assert annotation.reference_resolution == "explain the battery health screen"

    def test_skipping_writes_nothing(self, monkeypatch):
        assert self._run(monkeypatch, ["s"]) is None

    def test_an_invalid_intent_is_re_prompted_rather_than_coerced(self, monkeypatch):
        annotation = self._run(
            monkeypatch,
            ["software_update_issue", "battery_charging", "n", "y", "n",
             "self_serve_steps", "n", "high", "", ""],
        )
        assert annotation.intent == "battery_charging"

    def test_a_security_flag_is_recorded_independently_of_the_intent(self, monkeypatch):
        annotation = self._run(
            monkeypatch,
            ["battery_charging", "y", "y", "y", "human_action_required", "n", "high", "", ""],
        )
        assert annotation.security_sensitive is True
        assert annotation.intent == "battery_charging"

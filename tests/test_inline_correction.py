"""Queue filtering, the forced-review gate, and the blind condition.

This file originally also covered a developer shorthand for corrections (``i5``, ``sec``,
``ctx``). That shorthand was removed when the annotation screens were rewritten in plain
English for the person actually annotating: an accidental ``e`` would have saved a label with
escalation silently flipped. Its guarantees - changed fields recorded, unchanged fields kept,
an edit that changes nothing recorded as an acceptance, invalid input re-asked rather than
guessed - are now pinned against the plain-English flow in
``tests/test_readable_annotation_ui.py``.

Two things are deliberately *not* relaxed, and are pinned here:

* **Forced review stays forced.** The gated examples are the low-confidence, security-flagged,
  context-insufficient and escalation-sensitive ones - exactly where a wrong label costs most.
* **Blind stays blind.** The 40 blind examples exist so anchoring bias is measurable. No queue
  mode may hand one a suggestion, so that is asserted against the data rather than trusted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hiver_support.golden.store import read_candidates
from hiver_support.golden.suggestions import ModelSuggestion, blind_pair_ids, read_suggestions
from hiver_support.taxonomy import TAXONOMY

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "annotate_golden.py"


def cli():
    spec = importlib.util.spec_from_file_location("annotate_golden", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suggestion(**overrides) -> ModelSuggestion:
    payload = {
        "pair_id": "p0",
        "intent": "battery_charging",
        "security_sensitive": False,
        "context_sufficient": True,
        "should_escalate": False,
        "expected_resolution_kind": "self_serve_steps",
        "confidence": 0.9,
        "model": "meta-llama/llama-3.3-70b-instruct",
        "provider": "openrouter",
    }
    payload.update(overrides)
    return ModelSuggestion(**payload)


def a_candidate():
    from datetime import datetime, timezone

    from hiver_support.golden.schema import GoldenCandidate

    return GoldenCandidate(
        pair_id="p0",
        conversation_id="c0",
        customer_tweet_id="0",
        customer_message="my battery dies fast",
        created_at=datetime(2017, 11, 1, tzinfo=timezone.utc),
    )



def _local_v1():
    from hiver_support.golden import paths

    path = paths.local_candidates("v1")
    if not path.exists():
        pytest.skip("golden text not materialised (python scripts/materialize_text.py --golden)")
    return path


class TestForcedReviewIsNotRelaxed:
    def test_inline_editing_does_not_bypass_the_forced_gate(self):
        from hiver_support.golden.suggestions import needs_mandatory_review

        for gated in (
            suggestion(confidence=0.3),
            suggestion(security_sensitive=True),
            suggestion(context_sufficient=False),
            suggestion(intent="billing_and_subscription"),
            suggestion(should_escalate=True),
        ):
            assert needs_mandatory_review(gated)[0] is True

    def test_a_stray_shorthand_keystroke_saves_nothing(self, monkeypatch):
        # The removed shorthand saved immediately. A non-developer pressing "e" by accident
        # must get the question again, not a stored label with escalation flipped.
        answers = iter(["e", "sec", "i5", "s"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
        assert cli()._review_one(a_candidate(), suggestion(), "nitish", 1) is None

class TestQueueFiltering:
    def test_the_three_groups_are_supported(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for group in ("assisted", "blind", "all"):
            assert group in source

    def test_filtering_selects_the_right_candidates(self):
        module = cli()
        candidates = read_candidates(_local_v1())
        blind = blind_pair_ids(candidates)
        assisted_queue = module._filter_group(candidates, "assisted", blind)
        blind_queue = module._filter_group(candidates, "blind", blind)
        all_queue = module._filter_group(candidates, "all", blind)
        assert len(assisted_queue) + len(blind_queue) == len(all_queue) == len(candidates)
        assert {c.pair_id for c in blind_queue} == blind
        assert not {c.pair_id for c in assisted_queue} & blind

    def test_filtering_does_not_change_the_split_itself(self):
        candidates = read_candidates(_local_v1())
        before = blind_pair_ids(candidates)
        cli()._filter_group(candidates, "assisted", before)
        assert blind_pair_ids(candidates) == before
        assert len(before) == 40


class TestBlindCandidatesNeverSeeASuggestion:
    """Asserted against the real data, in every queue mode."""

    def test_no_blind_candidate_has_a_stored_suggestion(self):
        candidates = read_candidates(_local_v1())
        suggestions = read_suggestions(ROOT / "data" / "golden" / "suggestions.jsonl")
        leaked = sorted(blind_pair_ids(candidates) & set(suggestions))
        assert leaked == [], f"blind candidates carry suggestions: {leaked}"

    @pytest.mark.parametrize("group", ["assisted", "blind", "all"])
    def test_no_queue_mode_hands_a_blind_candidate_a_suggestion(self, group):
        module = cli()
        candidates = read_candidates(_local_v1())
        blind = blind_pair_ids(candidates)
        suggestions = read_suggestions(ROOT / "data" / "golden" / "suggestions.jsonl")
        for candidate in module._filter_group(candidates, group, blind):
            if candidate.pair_id in blind:
                assert suggestions.get(candidate.pair_id) is None

    def test_the_split_is_exactly_forty_sixty_as_the_protocol_fixes_it(self):
        candidates = read_candidates(_local_v1())
        assert len(blind_pair_ids(candidates)) == 40
        assert len(candidates) == 200

    def test_the_blind_set_is_deterministic(self):
        candidates = read_candidates(_local_v1())
        assert blind_pair_ids(candidates) == blind_pair_ids(candidates)


class TestProgressDisplay:
    def test_it_reports_remaining_and_forced_remaining(self):
        line = cli()._progress_line(done=12, total=160, forced_left=52)
        assert "12" in line and "160" in line and "52" in line

    def test_it_survives_a_zero_total(self):
        assert cli()._progress_line(done=0, total=0, forced_left=0)


class TestTheToolStaysBlindOfTheSystemUnderTest:
    def test_it_still_imports_no_classifier_retriever_or_generator(self):
        import ast

        modules = set()
        for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        leaks = sorted(
            m for m in modules
            if m.startswith(("hiver_support.classifier", "hiver_support.agent"))
        )
        assert leaks == [], f"annotation tool imports {leaks}"

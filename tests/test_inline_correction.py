"""Inline corrections, queue filtering, and the blind condition under both.

The 64 forced-review examples were costing six keystrokes each — `C`, then Enter through five
fields — even when a single field was wrong. This adds a one-line edit syntax for that case.

Two things are deliberately *not* relaxed, and are pinned here:

* **Forced review stays forced.** The gated examples are the low-confidence, security-flagged,
  context-insufficient and escalation-sensitive ones — exactly where a wrong label costs most.
  A faster correction path must not become a faster acceptance path.
* **Blind stays blind.** The 40 blind examples exist so anchoring bias is measurable. No queue
  mode may hand one a suggestion, so that is asserted against the data rather than trusted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hiver_support.golden.schema import ReviewAction
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


NAMES = list(TAXONOMY.names)
# A position that is NOT the default suggestion's intent, so an edit genuinely changes it.
OTHER_INDEX = next(i for i, n in enumerate(NAMES, start=1) if n != "battery_charging")
OTHER_NAME = NAMES[OTHER_INDEX - 1]


class TestInlineEditParsing:
    def test_a_numeric_intent_edit_selects_by_position(self):
        values, changed = cli()._parse_inline_edit(f"i{OTHER_INDEX}", suggestion(), NAMES)
        assert values["intent"] == OTHER_NAME
        assert changed == ("intent",)

    def test_a_named_intent_edit_is_accepted(self):
        values, changed = cli()._parse_inline_edit("i=connectivity", suggestion(), NAMES)
        assert values["intent"] == "connectivity"
        assert changed == ("intent",)

    @pytest.mark.parametrize(
        "token,field",
        [
            ("e", "should_escalate"),
            ("sec", "security_sensitive"),
            ("ctx", "context_sufficient"),
        ],
    )
    def test_a_flag_token_toggles_that_field(self, token, field):
        base = suggestion()
        values, changed = cli()._parse_inline_edit(token, base, NAMES)
        assert values[field] is not getattr(base, field)
        assert changed == (field,)

    def test_tokens_combine(self):
        values, changed = cli()._parse_inline_edit(f"i{OTHER_INDEX} e", suggestion(), NAMES)
        assert values["intent"] == OTHER_NAME
        assert values["should_escalate"] is True
        assert set(changed) == {"intent", "should_escalate"}

    def test_unchanged_fields_keep_the_suggested_value(self):
        base = suggestion()
        values, _ = cli()._parse_inline_edit("e", base, NAMES)
        assert values["intent"] == base.intent
        assert values["expected_resolution_kind"] == base.expected_resolution_kind

    def test_editing_to_the_same_intent_reports_no_change(self):
        # Which makes it an acceptance, not a correction.
        index = NAMES.index("battery_charging") + 1
        _, changed = cli()._parse_inline_edit(f"i{index}", suggestion(), NAMES)
        assert changed == ()

    @pytest.mark.parametrize("action", ["a", "c", "f", "s", "", "?"])
    def test_the_existing_actions_are_not_inline_edits(self, action):
        assert cli()._parse_inline_edit(action, suggestion(), NAMES) is None

    @pytest.mark.parametrize("bad", ["i99", "i=nonsense", "i0", "zzz", "i"])
    def test_a_malformed_edit_is_rejected_rather_than_guessed(self, bad):
        with pytest.raises(ValueError):
            cli()._parse_inline_edit(bad, suggestion(), NAMES)

    def test_case_and_spacing_are_tolerated(self):
        values, changed = cli()._parse_inline_edit(f"  I{OTHER_INDEX}   E  ", suggestion(), NAMES)
        assert values["intent"] == OTHER_NAME
        assert changed == ("intent", "should_escalate") or set(changed) == {
            "intent",
            "should_escalate",
        }


class TestInlineEditsPreserveProvenance:
    def test_a_changed_field_is_recorded_for_the_audit_trail(self):
        _, changed = cli()._parse_inline_edit(f"i{OTHER_INDEX}", suggestion(), NAMES)
        assert "intent" in changed

    def test_the_review_actions_remain_the_only_two_outcomes(self):
        # An inline edit is just a faster route to CORRECTED or ACCEPTED. It introduces no
        # third state, and in particular no way to record a label with no human action.
        assert {a.value for a in ReviewAction} == {"entered", "accepted", "corrected"}

    def test_an_inline_edit_never_alters_the_resolution_kind_silently(self):
        base = suggestion()
        values, changed = cli()._parse_inline_edit("e", base, NAMES)
        assert values["expected_resolution_kind"] == base.expected_resolution_kind
        assert "expected_resolution_kind" not in changed


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

    def test_the_prompt_still_omits_accept_when_review_is_forced(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "[C]orrect" in source
        assert "one-key accept is disabled" in source


class TestQueueFiltering:
    def test_the_three_groups_are_supported(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for group in ("assisted", "blind", "all"):
            assert group in source

    def test_filtering_selects_the_right_candidates(self):
        module = cli()
        candidates = read_candidates(ROOT / "data" / "golden" / "candidates.jsonl")
        blind = blind_pair_ids(candidates)
        assisted_queue = module._filter_group(candidates, "assisted", blind)
        blind_queue = module._filter_group(candidates, "blind", blind)
        all_queue = module._filter_group(candidates, "all", blind)
        assert len(assisted_queue) + len(blind_queue) == len(all_queue) == len(candidates)
        assert {c.pair_id for c in blind_queue} == blind
        assert not {c.pair_id for c in assisted_queue} & blind

    def test_filtering_does_not_change_the_split_itself(self):
        candidates = read_candidates(ROOT / "data" / "golden" / "candidates.jsonl")
        before = blind_pair_ids(candidates)
        cli()._filter_group(candidates, "assisted", before)
        assert blind_pair_ids(candidates) == before
        assert len(before) == 40


class TestBlindCandidatesNeverSeeASuggestion:
    """Asserted against the real data, in every queue mode."""

    def test_no_blind_candidate_has_a_stored_suggestion(self):
        candidates = read_candidates(ROOT / "data" / "golden" / "candidates.jsonl")
        suggestions = read_suggestions(ROOT / "data" / "golden" / "suggestions.jsonl")
        leaked = sorted(blind_pair_ids(candidates) & set(suggestions))
        assert leaked == [], f"blind candidates carry suggestions: {leaked}"

    @pytest.mark.parametrize("group", ["assisted", "blind", "all"])
    def test_no_queue_mode_hands_a_blind_candidate_a_suggestion(self, group):
        module = cli()
        candidates = read_candidates(ROOT / "data" / "golden" / "candidates.jsonl")
        blind = blind_pair_ids(candidates)
        suggestions = read_suggestions(ROOT / "data" / "golden" / "suggestions.jsonl")
        for candidate in module._filter_group(candidates, group, blind):
            if candidate.pair_id in blind:
                assert suggestions.get(candidate.pair_id) is None

    def test_the_split_is_exactly_forty_sixty_as_the_protocol_fixes_it(self):
        candidates = read_candidates(ROOT / "data" / "golden" / "candidates.jsonl")
        assert len(blind_pair_ids(candidates)) == 40
        assert len(candidates) == 200

    def test_the_blind_set_is_deterministic(self):
        candidates = read_candidates(ROOT / "data" / "golden" / "candidates.jsonl")
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

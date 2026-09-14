"""The annotation screens, written for the person doing the annotating rather than the developer.

The previous assisted screen showed internal names (``battery_charging``, ``True``) and asked
for developer shorthand (``i5``, ``sec``, ``ctx``). The annotator is not a developer, and a
screen they have to decode is how a correct intention turns into a wrong gold label.

Nothing about what gets *recorded* changes. These tests pin both halves: the screens speak
plain English, and the saved values are still the frozen taxonomy names, real bools and the
resolution enum, with ACCEPTED/CORRECTED, the model suggestion shown and the changed fields
recorded exactly as before.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hiver_support.golden.schema import (
    ExpectedResolutionKind,
    FlagRecord,
    GoldenAnnotation,
    GoldenCandidate,
    ReviewAction,
)
from hiver_support.golden.suggestions import ModelSuggestion
from hiver_support.taxonomy import TAXONOMY

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "annotate_golden.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("annotate_golden_ui", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate() -> GoldenCandidate:
    return GoldenCandidate(
        pair_id="p1",
        conversation_id="c1",
        customer_tweet_id="1",
        customer_message="my battery dies in two hours since the update",
        created_at=datetime(2017, 11, 1, tzinfo=timezone.utc),
    )


def suggestion(**overrides) -> ModelSuggestion:
    payload = {
        "pair_id": "p1",
        "intent": "battery_charging",
        "security_sensitive": False,
        "context_sufficient": True,
        "should_escalate": False,
        "expected_resolution_kind": "self_serve_steps",
        "confidence": 0.9,
        "rationale": "battery drain after an update",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "provider": "openrouter",
    }
    payload.update(overrides)
    return ModelSuggestion(**payload)


def drive(monkeypatch, keystrokes):
    """Feed keystrokes. Asking for more than supplied raises StopIteration, which fails the test."""
    answers = iter(keystrokes)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))


def menu_number(name: str) -> str:
    return str(list(TAXONOMY.names).index(name) + 1)


def resolution_number(value: str) -> str:
    return str([k.value for k in ExpectedResolutionKind].index(value) + 1)


INTERNAL_NAMES = (
    "battery_charging",
    "security_sensitive",
    "context_sufficient",
    "should_escalate",
    "expected_resolution_kind",
    "self_serve_steps",
)


class TestHumanReadableLabels:
    def test_every_frozen_intent_has_a_label_and_nothing_is_invented(self, cli):
        assert set(cli.INTENT_LABELS) == set(TAXONOMY.names)

    def test_every_resolution_kind_has_a_label_and_nothing_is_invented(self, cli):
        assert set(cli.RESOLUTION_LABELS) == {k.value for k in ExpectedResolutionKind}

    def test_labels_are_words_not_identifiers(self, cli):
        for label in list(cli.INTENT_LABELS.values()) + list(cli.RESOLUTION_LABELS.values()):
            assert "_" not in label and label[0].isupper()

    def test_the_labels_match_the_annotator_facing_wording(self, cli):
        assert cli.INTENT_LABELS["battery_charging"] == "Battery & charging"
        assert cli.INTENT_LABELS["howto_information"] == "How-to / information"
        assert cli.RESOLUTION_LABELS["self_serve_steps"] == "Self-serve steps"


class TestTheSuggestionScreen:
    def test_it_shows_the_suggestion_in_plain_english(self, cli, capsys, monkeypatch):
        drive(monkeypatch, ["s"])
        cli._review_one(candidate(), suggestion(), "nitish", 1)
        out = capsys.readouterr().out
        for line in (
            "MODEL'S SUGGESTION",
            "Intent: Battery & charging",
            "Security sensitive: No",
            "Context sufficient: Yes",
            "Should escalate: No",
            "Expected resolution: Self-serve steps",
        ):
            assert line in out

    def test_it_asks_one_plain_question(self, cli, capsys, monkeypatch):
        drive(monkeypatch, ["s"])
        cli._review_one(candidate(), suggestion(), "nitish", 1)
        out = capsys.readouterr().out
        for line in (
            "Is this suggestion correct?",
            "[Y] Yes",
            "[N] No, I want to correct it",
            "[S] Skip",
            "[F] Flag",
        ):
            assert line in out

    def test_it_exposes_no_internal_names_or_shorthand(self, cli, capsys, monkeypatch):
        drive(monkeypatch, ["s"])
        cli._review_one(candidate(), suggestion(), "nitish", 1)
        out = capsys.readouterr().out
        for internal in INTERNAL_NAMES + ("True", "False", "i=", " ctx", " sec ", "i5"):
            assert internal not in out, f"screen shows developer text {internal!r}"

    def test_it_still_says_the_suggestion_is_not_a_label(self, cli, capsys, monkeypatch):
        drive(monkeypatch, ["s"])
        cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert "not a label" in capsys.readouterr().out.lower()


class TestYesAcceptsWithOneKey:
    def test_a_single_y_accepts_an_ordinary_suggestion(self, cli, monkeypatch):
        drive(monkeypatch, ["y"])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert isinstance(result, GoldenAnnotation)
        assert result.review_action is ReviewAction.ACCEPTED
        assert result.corrected_fields == ()

    def test_an_accepted_label_stores_the_structured_values_and_the_suggestion(
        self, cli, monkeypatch
    ):
        shown = suggestion()
        drive(monkeypatch, ["y"])
        result = cli._review_one(candidate(), shown, "nitish", 1)
        assert result.intent == "battery_charging"
        assert result.security_sensitive is False
        assert result.expected_resolution_kind is ExpectedResolutionKind.SELF_SERVE_STEPS
        assert result.model_suggestion == shown.to_dict()

    def test_only_the_human_yes_makes_it_human_labelled(self, cli, monkeypatch):
        drive(monkeypatch, ["y"])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert result.provenance.is_gold
        assert suggestion().provenance.value == "MODEL_GENERATED"


class TestCarefulLookSuggestionsNeedAConfirmation:
    """The forced-review gate survives the new UI as one plain-English confirmation."""

    def test_y_alone_does_not_accept_a_security_suggestion(self, cli, monkeypatch):
        drive(monkeypatch, ["y"])
        with pytest.raises(StopIteration):
            cli._review_one(candidate(), suggestion(security_sensitive=True), "nitish", 1)

    def test_y_then_y_accepts_it(self, cli, monkeypatch):
        drive(monkeypatch, ["y", "y"])
        result = cli._review_one(candidate(), suggestion(security_sensitive=True), "nitish", 1)
        assert result.review_action is ReviewAction.ACCEPTED

    def test_declining_the_confirmation_goes_back_to_the_question(self, cli, monkeypatch):
        drive(monkeypatch, ["y", "n", "s"])
        assert cli._review_one(candidate(), suggestion(confidence=0.3), "nitish", 1) is None

    def test_the_reason_is_given_in_plain_words(self, cli, capsys, monkeypatch):
        drive(monkeypatch, ["s"])
        cli._review_one(candidate(), suggestion(security_sensitive=True), "nitish", 1)
        out = capsys.readouterr().out.lower()
        assert "careful" in out and "security" in out
        assert "escalation-sensitive by policy" not in out
        assert "0.75" not in out


class TestNoWalksThroughReadableQuestions:
    def test_the_questions_are_the_plain_english_ones(self, cli, capsys, monkeypatch):
        drive(monkeypatch, ["n", "", "", "", "", ""])
        cli._review_one(candidate(), suggestion(), "nitish", 1)
        out = capsys.readouterr().out
        for question in (
            "What is the correct intent?",
            "Is this security-sensitive?",
            "Is there enough context to understand the customer's issue?",
            "Should this be escalated to a human?",
            "What kind of resolution is expected?",
        ):
            assert question in out
        assert "Device malfunction" in out and "Self-serve steps" in out
        for internal in INTERNAL_NAMES:
            assert internal not in out

    def test_choosing_a_different_intent_saves_the_internal_name(self, cli, monkeypatch):
        drive(monkeypatch, ["n", menu_number("device_malfunction"), "", "", "", ""])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert result.intent == "device_malfunction"
        assert result.review_action is ReviewAction.CORRECTED
        assert result.corrected_fields == ("intent",)
        assert result.model_suggestion["intent"] == "battery_charging"

    def test_pressing_enter_everywhere_keeps_the_suggestion_and_records_acceptance(
        self, cli, monkeypatch
    ):
        drive(monkeypatch, ["n", "", "", "", "", ""])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert result.review_action is ReviewAction.ACCEPTED
        assert result.corrected_fields == ()

    def test_a_yes_no_answer_is_saved_as_a_real_bool(self, cli, monkeypatch):
        drive(monkeypatch, ["n", "", "", "", "y", ""])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert result.should_escalate is True
        assert result.corrected_fields == ("should_escalate",)

    def test_a_resolution_choice_is_saved_as_the_enum(self, cli, monkeypatch):
        drive(monkeypatch, ["n", "", "", "", "", resolution_number("human_action_required")])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert result.expected_resolution_kind is ExpectedResolutionKind.HUMAN_ACTION_REQUIRED
        assert result.corrected_fields == ("expected_resolution_kind",)

    def test_several_corrections_are_all_recorded(self, cli, monkeypatch):
        drive(monkeypatch, ["n", menu_number("account_access"), "y", "", "y", ""])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert set(result.corrected_fields) == {"intent", "security_sensitive", "should_escalate"}

    def test_an_invalid_choice_asks_again_instead_of_guessing(self, cli, monkeypatch):
        drive(monkeypatch, ["n", "99", menu_number("connectivity"), "", "", "", ""])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert result.intent == "connectivity"


class TestSkipAndFlag:
    def test_skip_records_nothing(self, cli, monkeypatch):
        drive(monkeypatch, ["s"])
        assert cli._review_one(candidate(), suggestion(), "nitish", 1) is None

    def test_flag_records_a_flag_with_the_reason(self, cli, monkeypatch):
        drive(monkeypatch, ["f", "two problems in one message"])
        result = cli._review_one(candidate(), suggestion(), "nitish", 1)
        assert isinstance(result, FlagRecord)
        assert result.reason == "two problems in one message"


class TestBlindExamplesStayBlind:
    BLIND_KEYS = ["battery_charging", "n", "y", "n", "self_serve_steps", "n", "high", "", ""]

    def test_a_blind_example_shows_no_suggestion(self, cli, capsys, monkeypatch):
        drive(monkeypatch, self.BLIND_KEYS)
        result = cli._review_one(candidate(), None, "nitish", 1)
        out = capsys.readouterr().out
        assert "MODEL'S SUGGESTION" not in out
        assert result.review_action is ReviewAction.ENTERED
        assert result.model_suggestion is None

    def test_the_blind_screens_are_readable_too(self, cli, capsys, monkeypatch):
        drive(monkeypatch, self.BLIND_KEYS)
        cli._review_one(candidate(), None, "nitish", 1)
        out = capsys.readouterr().out
        assert "Battery & charging" in out
        assert "Is this security-sensitive?" in out
        for internal in INTERNAL_NAMES:
            assert internal not in out

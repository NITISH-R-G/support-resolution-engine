"""Weak supervision for training labels.

**These are NOT human labels and must never be reported as ground truth.** No hand-labelled
data exists yet: the golden set is drawn from the held-out test pool and has not been built.
Training labels therefore come from labelling functions, and the code is built so that fact
cannot be lost downstream — every weak label carries ``is_weak=True`` and the aggregate
carries its own health statistics.

Two circularity warnings are structural, not incidental, and the tests pin the honesty
machinery that keeps them visible:

1. The frozen taxonomy was itself derived from the train split, so labelling functions written
   against that taxonomy and applied to that split are doubly self-referential.
2. A model trained on weak labels and *evaluated* against weak labels measures agreement with
   the heuristics, **not** intent accuracy. Any dev metric computed this way is a
   rule-recovery score. `docs/CLASSIFIER.md` states this and the results artifact repeats it.

Abstention matters as much as voting. A labelling function that fires on everything provides
no signal, so coverage and conflict are first-class outputs rather than diagnostics.
"""

from __future__ import annotations

import pytest

from hiver_support.classifier.weak_labels import (
    WeakLabel,
    WeakLabelStats,
    label_batch,
    weak_label,
)
from hiver_support.taxonomy import TAXONOMY


class TestWeakLabelIsNeverMistakenForGold:
    def test_every_weak_label_is_marked_weak(self):
        assert weak_label("my battery drains so fast").is_weak is True

    def test_weak_flag_cannot_be_turned_off(self):
        label = weak_label("my battery drains so fast")
        with pytest.raises((AttributeError, TypeError)):
            label.is_weak = False  # type: ignore[misc]

    def test_serialised_form_announces_weak_supervision(self):
        payload = weak_label("my battery drains").to_dict()
        assert payload["is_weak"] is True
        assert "weak" in payload["provenance"].lower()


class TestIntentVoting:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("my battery drains so fast since yesterday", "battery_charging"),
            ("wifi keeps disconnecting on my ipad", "connectivity"),
            ("i was charged twice for my icloud storage subscription", "billing_and_subscription"),
            ("i forgot my apple id password and cannot sign in", "account_access"),
            ("how do i turn off notifications for messages", "howto_information"),
            ("my genius bar appointment for the screen repair was cancelled", "repair_order_replacement"),
            ("apple music keeps crashing every time i open the app", "apps_and_services"),
        ],
    )
    def test_clear_cases_receive_the_expected_intent(self, text, expected):
        assert weak_label(text).intent == expected

    def test_intent_is_always_a_frozen_taxonomy_label_or_none(self):
        for text in ("battery dead", "asdkjhasd", "", "how do i"):
            label = weak_label(text)
            assert label.intent is None or label.intent in {i.name for i in TAXONOMY.intents}

    def test_abstains_rather_than_guessing_when_nothing_matches(self):
        label = weak_label("the weather is nice today and i like trains")
        assert label.intent is None
        assert label.abstained is True

    def test_votes_are_recorded_for_inspection(self):
        label = weak_label("my battery drains and wifi keeps dropping")
        assert len(label.votes) >= 2

    def test_conflicts_are_recorded_not_hidden(self):
        label = weak_label("my battery drains and wifi keeps dropping")
        assert label.has_conflict is True

    def test_conflict_resolution_is_deterministic(self):
        text = "my battery drains and wifi keeps dropping"
        assert weak_label(text).intent == weak_label(text).intent


class TestSecuritySensitive:
    @pytest.mark.parametrize(
        "text",
        [
            "i think my apple id has been hacked",
            "someone is using my account to buy things",
            "is this email from you or is it a scam",
            "my iphone was stolen last night",
            "there is a fraudulent charge on my account",
            "i received a phishing sms that claims to be from apple",
        ],
    )
    def test_detects_security_across_intents(self, text):
        assert weak_label(text).security_sensitive is True

    def test_security_is_not_inferred_from_intent_alone(self):
        """A plain account message is not security-sensitive."""
        assert weak_label("i forgot my apple id password").security_sensitive is False

    def test_security_detected_on_a_billing_message(self):
        label = weak_label("there is an unauthorised charge on my card")
        assert label.security_sensitive is True

    def test_security_detected_on_a_device_message(self):
        label = weak_label("my iphone was stolen and i need to lock it")
        assert label.security_sensitive is True

    def test_ordinary_message_is_not_security_sensitive(self):
        assert weak_label("my battery drains fast").security_sensitive is False


class TestContextSufficiency:
    @pytest.mark.parametrize(
        "text", ["Ok", "thanks", "11.0.3", "iPhone 7", "yes", "?", "help"]
    )
    def test_bare_messages_are_context_insufficient(self, text):
        assert weak_label(text).context_sufficient is False

    @pytest.mark.parametrize(
        "text",
        [
            "my battery drains so fast since the update yesterday",
            "how do i turn off notifications for imessage on my ipad",
        ],
    )
    def test_substantive_messages_are_context_sufficient(self, text):
        assert weak_label(text).context_sufficient is True

    def test_context_insufficient_messages_do_not_get_a_confident_intent(self):
        """Insufficient context must not be resolved into a label by keyword luck."""
        label = weak_label("iPhone 7")
        assert label.context_sufficient is False
        assert label.intent is None


class TestBatchStatistics:
    """Coverage and conflict are outputs, not diagnostics: a labeller that fires on
    everything, or agrees with itself always, provides no signal."""

    @staticmethod
    def _texts() -> list[str]:
        return [
            "my battery drains so fast",
            "wifi keeps disconnecting",
            "i got charged twice for icloud storage this month",
            "the weather is nice today",
            "Ok",
            "i think my account was hacked",
        ]

    def test_batch_returns_one_label_per_input(self):
        labels, _ = label_batch(self._texts())
        assert len(labels) == len(self._texts())

    def test_stats_report_coverage(self):
        _, stats = label_batch(self._texts())
        assert isinstance(stats, WeakLabelStats)
        assert 0.0 <= stats.coverage <= 1.0

    def test_stats_report_abstention(self):
        _, stats = label_batch(self._texts())
        assert stats.abstained >= 1

    def test_stats_report_conflict_rate(self):
        _, stats = label_batch(self._texts())
        assert 0.0 <= stats.conflict_rate <= 1.0

    def test_stats_report_per_intent_counts(self):
        _, stats = label_batch(self._texts())
        assert set(stats.intent_counts) <= {i.name for i in TAXONOMY.intents}

    def test_stats_are_serialisable_with_the_weak_warning(self):
        import json

        _, stats = label_batch(self._texts())
        payload = json.loads(json.dumps(stats.to_dict()))
        assert "weak" in payload["warning"].lower()
        assert "not" in payload["warning"].lower()


class TestDeterminism:
    def test_same_input_gives_same_label(self):
        assert weak_label("my battery drains") == weak_label("my battery drains")

    def test_batch_is_order_independent_per_item(self):
        texts = ["my battery drains", "wifi keeps dropping"]
        forward, _ = label_batch(texts)
        backward, _ = label_batch(list(reversed(texts)))
        assert forward[0] == backward[1]


class TestEdgeCases:
    def test_empty_string(self):
        label = weak_label("")
        assert label.intent is None
        assert label.context_sufficient is False

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            weak_label(None)  # type: ignore[arg-type]

    def test_very_long_input_does_not_crash(self):
        assert weak_label("battery " * 5000).intent == "battery_charging"

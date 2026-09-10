"""Classification of brand replies into deflection / substantive / actionable.

These three rates drive brand selection, so they get a tested module rather than regexes
buried in an analysis script.

**Every example below is a verbatim brand reply observed in the real TWCS corpus**, captured
during manual inspection of `scripts/analyse_brands.py` output. That inspection is what
exposed the defect these tests now pin: an earlier lexicon matched only explicit "DM us"
phrasings, so "Please contact us directly here: [URL]" and "please request a callback here:
[URL]" were being counted as substantive resolutions. The result under-counted deflection
and over-counted resolution density — on the exact feature the brand-selection rubric
depends on.

The hard case is a reply that redirects *and* informs. "If the charges are pending, they
could be authorisations" mentions contacting support but carries real information, and must
not be discarded as a deflection.
"""

from __future__ import annotations

import pytest

from hiver_support.data.reply_classify import classify_reply, is_probably_english

# --- verbatim from the corpus -----------------------------------------------------------

PURE_DEFLECTIONS = [
    "Got it. Please DM us and we will continue troubleshooting from there. [URL]",
    "We've received your DM and you can expect a response there shortly.",
    "Hello! Thanks for your patience. Please check your DM for our reply.",
    "We see your DM and will connect with you there. Keep an eye out for our message!",
    "We received your DM and We'll respond there shortly. Talk to you soon!",
    "We've followed up via DM. Please check your messages for an update.",
    "So sorry about that! Send us a DM with your email address so we can connect right away.",
    "We apologize for the trouble! Please contact us via [URL] so our team can connect.",
    "Sorry to hear about that! Can you please contact us via [URL] so our team can connect.",
    # The ones the original lexicon missed:
    "[USER] Please contact us directly here: [URL] and we can look into this further. ^PK",
    "We apologize for the trouble! Send us a note here, [URL] and our team will be in touch.",
    "Here to help! Send us a note via [URL] and we'll be in touch.",
    "Happy to assist! Just visit for more info: [URL]",
    "[USER] We appreciate you reaching out to us. Please use this link to provide us feedback:"
    " [URL]",
]

SUBSTANTIVE_REPLIES = [
    "[USER] Thanks for those details. Picture in Picture on iOS is only supported on some iPad"
    " models: [URL]",
    "[USER] We're happy to help. Please confirm which macOS version is installed on the Mac right"
    " now so we can get started.",
    "[USER] Hi, I'm sorry to hear this. Without giving any personal information, can you tell"
    " us what problem you're having?^PJ",
    "[USER] We'd like to help! Tell us more please about what you're experiencing. Are you"
    " getting any error messages?",
]

# Redirects *and* informs. Discarding these would throw away real grounding evidence.
#
# Both entries were initially labelled pure deflections during manual inspection. The
# classifier disagreed, and on review the classifier was right: each answers the customer's
# question before inviting further contact. "The Echo Show is supported" *is* the answer, and
# a rate that discards it would understate genuine resolution density. The labels were
# corrected rather than the code — the failing test was measuring a hasty judgement, not a
# defect.
MIXED_BUT_INFORMATIVE = [
    "[USER] Hi Sam, did you get a chance to contact us? If the charge is still pending it may"
    " be a temporary authorisation, which is not taken from your account and is released"
    " within a few days.",
    "[USER] That speaker model is supported, please reach us for live troubleshooting whenever"
    " it suits you: [URL] ^AB",
]

# Channel routing: tells the customer where support exists, but resolves no product problem.
# Neither a deflection to DM nor a resolution — it must not inflate the resolution rate.
CHANNEL_ROUTING = [
    "[USER] Thanks for reaching out. We offer support via Twitter in English. Get help in"
    " Spanish here: [URL] or join: [URL]",
]


class TestChannelRouting:
    @pytest.mark.parametrize("reply", CHANNEL_ROUTING)
    def test_routing_reply_is_not_counted_as_a_resolution(self, reply):
        assert classify_reply(reply).is_substantive is False


class TestPureDeflections:
    @pytest.mark.parametrize("reply", PURE_DEFLECTIONS)
    def test_is_deflection(self, reply):
        assert classify_reply(reply).is_deflection is True

    @pytest.mark.parametrize("reply", PURE_DEFLECTIONS)
    def test_is_not_substantive(self, reply):
        """A redirect carries nothing to ground a generated reply in."""
        assert classify_reply(reply).is_substantive is False


class TestSubstantiveReplies:
    @pytest.mark.parametrize("reply", SUBSTANTIVE_REPLIES)
    def test_is_not_deflection(self, reply):
        assert classify_reply(reply).is_deflection is False

    @pytest.mark.parametrize("reply", SUBSTANTIVE_REPLIES)
    def test_is_substantive(self, reply):
        assert classify_reply(reply).is_substantive is True


class TestMixedRepliesAreNotDiscarded:
    @pytest.mark.parametrize("reply", MIXED_BUT_INFORMATIVE)
    def test_informative_reply_survives_a_redirect_phrase(self, reply):
        result = classify_reply(reply)
        assert result.is_deflection is False
        assert result.is_substantive is True


class TestPleasantries:
    """Short social replies are neither deflection nor resolution."""

    @pytest.mark.parametrize(
        "reply",
        [
            "[USER] Glad to know you were able to place an order. Keep us posted for further"
            " updates :). ^SB",
            "You're very welcome! Have a great day!",
            "Thanks for letting us know!",
        ],
    )
    def test_pleasantry_is_not_substantive(self, reply):
        assert classify_reply(reply).is_substantive is False


class TestActionable:
    def test_instructional_reply_is_actionable(self):
        reply = "Please go to Settings > General > Software Update and tap Download and Install."
        result = classify_reply(reply)
        assert result.is_substantive is True
        assert result.is_actionable is True

    def test_substantive_but_non_instructional_reply_is_not_actionable(self):
        reply = (
            "Thanks for that detail. Picture in Picture on iOS works only on some iPad"
            " models, which is why it does not appear on your device."
        )
        assert classify_reply(reply).is_actionable is False

    def test_deflection_is_never_actionable(self):
        assert classify_reply("Please DM us and we can help you there.").is_actionable is False


class TestLanguageDetection:
    """Non-English replies are real and must be measured, not silently counted as resolutions."""

    @pytest.mark.parametrize(
        "reply",
        [
            "Massgeblich ist immer das Datum in Ihrer Bestellung. Es haengt von mehreren"
            " Faktoren ab.",
            "Que alegria que chegaram! Qual deles voce vai ler primeiro?",
        ],
    )
    def test_detects_non_english(self, reply):
        assert is_probably_english(reply) is False

    @pytest.mark.parametrize(
        "reply",
        [
            "Please go to Settings and tap Software Update to install the latest version.",
            "Thanks for the detail. Picture in Picture works only on some iPad models.",
        ],
    )
    def test_detects_english(self, reply):
        assert is_probably_english(reply) is True

    def test_non_english_reply_is_not_counted_as_substantive(self):
        reply = (
            "Massgeblich ist immer das Datum in Ihrer Bestellung. Es haengt von mehreren"
            " Faktoren ab. Viele Gruesse ^AB"
        )
        assert classify_reply(reply).is_substantive is False


class TestEdgeCases:
    def test_empty_reply(self):
        result = classify_reply("")
        assert result.is_deflection is False
        assert result.is_substantive is False

    def test_url_only_reply_is_not_substantive(self):
        assert classify_reply("[URL]").is_substantive is False

    def test_signature_only_reply_is_not_substantive(self):
        assert classify_reply("^AN").is_substantive is False

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            classify_reply(None)  # type: ignore[arg-type]


class TestInflectedActionVerbs:
    """Regression: `\brestart\b` does not match "restarting".

    Found while building operational-handling profiles for the taxonomy adjudication. Support
    replies overwhelmingly use continuous and past forms ("restarting", "updating", "resetting"),
    so a stem-only pattern silently under-counts actionable guidance.
    """

    @pytest.mark.parametrize(
        "reply",
        [
            "Try restarting the device by holding the side button, then check again.",
            "We suggest updating to the latest version and checking whether it persists.",
            "Resetting your network settings usually clears this up for most people.",
            "Have you tried reinstalling the application from the App Store recently?",
        ],
    )
    def test_inflected_instruction_is_actionable(self, reply):
        assert classify_reply(reply).is_actionable is True


class TestTerseInstructionsSurviveTheLanguageGate:
    """Regression: instructional replies use few function words and were failing as non-English.

    `is_probably_english` required a 12% function-word ratio. Terse navigation instructions
    ("Settings > General > Reset > Reset Network Settings") are mostly nouns, so the gate
    rejected them — and because `is_substantive` requires English, the most actionable replies
    in the corpus were the most likely to be discarded. That is a bias in the exact direction
    that would distort any measure of resolution quality.
    """

    @pytest.mark.parametrize(
        "reply",
        [
            "Please reset network settings: Settings > General > Reset > Reset Network Settings.",
            "Go to Settings > Battery > Battery Health and check Maximum Capacity there.",
            "Open Settings, tap General, tap Software Update, then tap Download and Install.",
        ],
    )
    def test_navigation_instructions_are_english(self, reply):
        assert is_probably_english(reply) is True

    @pytest.mark.parametrize(
        "reply",
        [
            "Please reset network settings: Settings > General > Reset > Reset Network Settings.",
            "Go to Settings > Battery > Battery Health and check Maximum Capacity there.",
        ],
    )
    def test_navigation_instructions_are_actionable(self, reply):
        assert classify_reply(reply).is_actionable is True

    def test_the_language_gate_still_rejects_genuine_non_english(self):
        """The fix must not be a blanket loosening that lets other languages through."""
        for reply in [
            "Massgeblich ist immer das Datum in Ihrer Bestellung. Es haengt von mehreren"
            " Faktoren ab.",
            "Que alegria que chegaram! Qual deles voce vai ler primeiro?",
            "Por favor, actualiza tu dispositivo a la ultima version del sistema operativo.",
        ]:
            assert is_probably_english(reply) is False

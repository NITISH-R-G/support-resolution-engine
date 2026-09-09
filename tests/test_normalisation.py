"""Pipeline text normalisation.

Distinct from the aggressive normalisation in ``leakage.py``: that one is lossy on purpose
because it only compares strings. This one feeds the classifier and the generator, so it
removes noise (URLs, foreign handles, HTML entities, encoding artefacts) while preserving
everything that carries meaning -- casing, punctuation, emphasis and emoji all signal the
sentiment that escalation decisions depend on.
"""

from __future__ import annotations

import pytest

from hiver_support.data.normalise import normalise_text


class TestURLs:
    @pytest.mark.parametrize(
        "raw",
        [
            "see https://t.co/abc123 for details",
            "see http://example.com/path?q=1 for details",
            "see www.example.com for details",
        ],
    )
    def test_urls_become_a_placeholder(self, raw):
        assert "[URL]" in normalise_text(raw)

    def test_url_placeholder_replaces_the_whole_url(self):
        assert normalise_text("go to https://t.co/abc123 now") == "go to [URL] now"

    def test_multiple_urls_each_replaced(self):
        assert normalise_text("a https://t.co/x b https://t.co/y") == "a [URL] b [URL]"


class TestMentions:
    def test_foreign_mention_becomes_placeholder(self):
        assert normalise_text("@randomuser is annoyed") == "[USER] is annoyed"

    def test_brand_handle_is_preserved(self):
        assert normalise_text("@AmazonHelp help me", brand="AmazonHelp") == "@AmazonHelp help me"

    def test_brand_handle_match_is_case_insensitive(self):
        assert normalise_text("@amazonhelp help", brand="AmazonHelp") == "@amazonhelp help"

    def test_other_brands_are_not_preserved(self):
        result = normalise_text("@AmazonHelp @AppleSupport help", brand="AmazonHelp")
        assert result == "@AmazonHelp [USER] help"

    def test_without_a_brand_all_mentions_are_masked(self):
        assert normalise_text("@AmazonHelp help") == "[USER] help"


class TestPreservesMeaning:
    """These carry the sentiment signal escalation depends on; normalising them away is a bug."""

    def test_casing_is_preserved(self):
        assert normalise_text("THIS IS UNACCEPTABLE") == "THIS IS UNACCEPTABLE"

    def test_punctuation_and_emphasis_are_preserved(self):
        assert normalise_text("what?!?! seriously...") == "what?!?! seriously..."

    def test_emoji_are_preserved(self):
        assert normalise_text("broken again 😤🔥") == "broken again 😤🔥"

    def test_currency_and_versions_are_preserved(self):
        assert normalise_text("$9.99 on iOS 11.0.1") == "$9.99 on iOS 11.0.1"

    def test_hashtags_are_preserved(self):
        assert normalise_text("#neveragain awful") == "#neveragain awful"


class TestEncodingArtefacts:
    def test_html_entities_are_decoded(self):
        """TWCS contains raw HTML entities; left alone they reach the model as literals."""
        assert normalise_text("Tom &amp; Jerry") == "Tom & Jerry"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("a &lt; b", "a < b"), ("a &gt; b", "a > b"), ("say &quot;hi&quot;", 'say "hi"')],
    )
    def test_common_entities(self, raw, expected):
        assert normalise_text(raw) == expected

    def test_unicode_is_nfkc_normalised(self):
        assert normalise_text("ﬁle") == "file"

    def test_fullwidth_characters_are_folded(self):
        assert normalise_text("ＨＥＬＬＯ") == "HELLO"

    def test_zero_width_characters_are_removed(self):
        assert normalise_text("bro​ken") == "broken"


class TestWhitespace:
    def test_runs_of_whitespace_collapse(self):
        assert normalise_text("too    many     spaces") == "too many spaces"

    def test_newlines_and_tabs_collapse_to_spaces(self):
        assert normalise_text("line one\n\nline two\ttabbed") == "line one line two tabbed"

    def test_leading_and_trailing_whitespace_is_stripped(self):
        assert normalise_text("   padded   ") == "padded"


class TestInvariants:
    def test_is_idempotent(self):
        raw = "@user check https://t.co/x   Tom &amp; Jerry 😤"
        once = normalise_text(raw)
        assert normalise_text(once) == once

    def test_placeholders_survive_a_second_pass(self):
        assert normalise_text("[URL] and [USER]") == "[URL] and [USER]"

    def test_empty_string(self):
        assert normalise_text("") == ""

    def test_whitespace_only_becomes_empty(self):
        assert normalise_text("   \n  ") == ""

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            normalise_text(None)  # type: ignore[arg-type]

    def test_a_message_that_is_only_noise_normalises_to_something_short(self):
        """Retrieval must be able to detect content-free messages rather than index them."""
        assert normalise_text("@brand https://t.co/x") == "[USER] [URL]"

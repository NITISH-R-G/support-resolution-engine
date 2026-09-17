"""PII masking — the boundary that customer text must not cross unmasked.

Two failure directions matter and both are tested. Under-masking leaks a real customer's
email or card digits to a third-party API. Over-masking is subtler and just as damaging to
the system: "iPhone 7", "$9.99" and "2 weeks" are ordinary support signal, and a masker
that eats them destroys the very features the classifier depends on.
"""

from __future__ import annotations

import pytest

from hiver_support.data.pii import PIIKind, mask_pii


def masked(text: str) -> str:
    return mask_pii(text).text


class TestEmailMasking:
    def test_masks_a_plain_email(self):
        assert masked("contact me at jane.doe@example.com please") == (
            "contact me at [EMAIL] please"
        )

    def test_masks_email_with_plus_addressing_and_subdomain(self):
        assert masked("a+tag@mail.co.uk") == "[EMAIL]"

    def test_reports_the_email_kind(self):
        assert PIIKind.EMAIL in mask_pii("x@y.com").kinds_found


class TestPhoneMasking:
    @pytest.mark.parametrize(
        "raw",
        [
            "555-123-4567",
            "(555) 123-4567",
            "+1 555 123 4567",
            "+44 20 7946 0958",
            "5551234567",
        ],
    )
    def test_masks_common_phone_formats(self, raw):
        assert "[PHONE]" in masked(f"call me on {raw} today")

    def test_does_not_mask_a_short_number(self):
        assert masked("I waited 20 minutes") == "I waited 20 minutes"

    @pytest.mark.parametrize(
        "raw",
        [
            "1-800-555-0134",
            "1 800 555 0134",
            "1.800.555.0134",
            "1-555-123-4567",
        ],
    )
    def test_masks_a_north_american_number_with_a_bare_country_code(self, raw):
        # Release audit finding: a leading "1-" without "+" defeated the pattern, so the whole
        # number reached the LLM prompt unmasked. One golden-set message contained one.
        text = masked(f"call {raw} and ask")
        assert "[PHONE]" in text
        assert "555" not in text

    @pytest.mark.parametrize(
        "text",
        [
            "step 1 - 10 minutes later it died",
            "1 of 2 devices, 3 weeks old",
            "updated to 11.1 - 11.2 broke it",
            "2 phones 1 800 number? no idea",
        ],
    )
    def test_a_bare_leading_one_is_not_masked_without_a_full_number(self, text):
        assert masked(text) == text


class TestOrderAndTrackingMasking:
    def test_masks_amazon_style_order_id(self):
        assert "[ORDER_ID]" in masked("order 123-4567890-1234567 never arrived")

    def test_masks_long_tracking_number(self):
        assert "[TRACKING]" in masked("tracking 1Z999AA10123456784 shows nothing")

    def test_masks_hash_prefixed_order_reference(self):
        assert "[ORDER_ID]" in masked("my order #A1B2C3D4E5 is late")


class TestCardMasking:
    def test_masks_a_full_card_number(self):
        assert "[CARD]" in masked("charged 4111 1111 1111 1111 twice")

    def test_masks_a_hyphenated_card_number(self):
        assert "[CARD]" in masked("card 4111-1111-1111-1111")

    def test_card_is_masked_before_phone_so_it_is_not_mislabelled(self):
        report = mask_pii("4111111111111111")
        assert PIIKind.CARD in report.kinds_found
        assert PIIKind.PHONE not in report.kinds_found


class TestDoesNotOverMask:
    """Over-masking silently degrades the classifier; these are the real false positives."""

    @pytest.mark.parametrize(
        "text",
        [
            "my iPhone 7 is broken",
            "charged me $9.99 for nothing",
            "this is my 3rd order in 2 weeks",
            "waiting since Oct 31 2017",
            "iOS 11.0.1 broke my wifi",
            "5 stars would not recommend",
        ],
    )
    def test_ordinary_support_text_is_untouched(self, text):
        assert masked(text) == text

    def test_brand_handle_survives(self):
        assert "@AmazonHelp" in masked("@AmazonHelp where is my stuff")


class TestReportAndInvariants:
    def test_reports_nothing_for_clean_text(self):
        report = mask_pii("where is my order")
        assert report.kinds_found == frozenset()
        assert report.text == "where is my order"

    def test_masking_is_idempotent(self):
        once = masked("mail me at a@b.com or call 555-123-4567")
        assert masked(once) == once

    def test_placeholders_are_not_themselves_masked(self):
        assert masked("[EMAIL] [PHONE] [CARD]") == "[EMAIL] [PHONE] [CARD]"

    def test_handles_empty_and_whitespace(self):
        assert masked("") == ""
        assert masked("   ") == "   "

    def test_handles_unicode_and_emoji_without_corruption(self):
        text = "my phone is broken 😤 très mauvais"
        assert masked(text) == text

    def test_multiple_kinds_in_one_message(self):
        report = mask_pii("email a@b.com, phone 555-123-4567, card 4111-1111-1111-1111")
        assert report.kinds_found == {PIIKind.EMAIL, PIIKind.PHONE, PIIKind.CARD}

    def test_none_input_raises_rather_than_silently_passing(self):
        """A None reaching the API boundary must fail loudly, not serialise as 'None'."""
        with pytest.raises((TypeError, AttributeError)):
            mask_pii(None)  # type: ignore[arg-type]


class TestFrozenV1Masker:
    """mask_pii_v1 must keep reproducing golden v1 and the evaluated inputs exactly."""

    def test_v1_leaves_the_bare_country_code_form_unmasked(self):
        from hiver_support.data.pii import mask_pii_v1

        assert mask_pii_v1("call 1-800-555-0134 now").text == "call 1-800-555-0134 now"

    def test_v1_and_v2_agree_everywhere_else(self):
        from hiver_support.data.pii import mask_pii_v1

        for text in ("call me on 555-123-4567", "mail a@b.com", "card 4111 1111 1111 1111",
                     "order 123-4567890-1234567", "my iPhone 7 is broken", "iOS 11.0.1"):
            assert mask_pii_v1(text) == mask_pii(text)

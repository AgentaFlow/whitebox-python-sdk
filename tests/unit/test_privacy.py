"""
Tests for WhiteBoxXAI SDK privacy/PII detection.

Regression coverage for three regex bugs (PR6): the email pattern's TLD
character class contained a literal ``|``, the credit-card pattern had no
Luhn validation (so any 16-digit run matched), and the IPv4 pattern had no
octet-range validation (so 999.999.999.999 matched).
"""

from whiteboxxai.privacy import PIIDetector, mask_pii


class TestEmailDetection:
    def test_detects_standard_email(self):
        detections = PIIDetector().detect("Contact us at support@example.com please")
        assert any(d["type"] == "email" and d["value"] == "support@example.com" for d in detections)

    def test_tld_character_class_has_no_literal_pipe_bug(self):
        """The old pattern's [A-Z|a-z] class matched a literal '|' as a
        valid TLD character -- e.g. "user@example.co|m" would wrongly be
        accepted as a full match up to and including the '|'."""
        detector = PIIDetector()
        detections = detector.detect("weird@example.co|m")
        for d in detections:
            if d["type"] == "email":
                assert "|" not in d["value"]

    def test_multi_letter_tld_still_matches(self):
        detections = PIIDetector().detect("reach me at person@company.info now")
        assert any(d["type"] == "email" and d["value"] == "person@company.info" for d in detections)


class TestCreditCardDetection:
    def test_valid_luhn_card_is_detected(self):
        # 4111 1111 1111 1111 is the standard Visa test number (Luhn-valid).
        detections = PIIDetector().detect("Card on file: 4111 1111 1111 1111")
        assert any(d["type"] == "credit_card" for d in detections)

    def test_luhn_invalid_digit_run_is_not_detected_as_a_card(self):
        """A 16-digit run that merely fits the grouping pattern (e.g. a
        reference/account number) but fails the Luhn checksum must not be
        reported as a credit card."""
        # 1234 5678 9012 3456 fails Luhn.
        detections = PIIDetector().detect("Reference number: 1234 5678 9012 3456")
        assert not any(d["type"] == "credit_card" for d in detections)

    def test_luhn_invalid_card_is_not_masked(self):
        text = "Reference number: 1234 5678 9012 3456"
        assert mask_pii(text) == text


class TestIPv4Detection:
    def test_valid_ipv4_is_detected(self):
        detections = PIIDetector().detect("Client connected from 192.168.1.42 today")
        assert any(d["type"] == "ip_address" and d["value"] == "192.168.1.42" for d in detections)

    def test_out_of_range_octets_are_not_detected(self):
        """999.999.999.999 fits the bare \\d{1,3} grouping but every octet
        is out of the valid 0-255 range."""
        detections = PIIDetector().detect("Bogus address: 999.999.999.999 logged")
        assert not any(d["type"] == "ip_address" for d in detections)

    def test_boundary_octet_255_is_valid(self):
        detections = PIIDetector().detect("Broadcast: 255.255.255.255")
        assert any(
            d["type"] == "ip_address" and d["value"] == "255.255.255.255" for d in detections
        )

    def test_octet_256_is_invalid(self):
        detections = PIIDetector().detect("Bad octet: 256.1.1.1 here")
        assert not any(d["type"] == "ip_address" for d in detections)

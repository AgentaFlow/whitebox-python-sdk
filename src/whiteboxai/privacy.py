"""
Privacy Utilities

PII detection and data masking utilities.
"""

import re
from typing import Any, Dict, List, Optional, Pattern


class PIIDetector:
    """
    Detect personally identifiable information (PII) in data.

    Supports:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social security numbers
    - IP addresses
    """

    def __init__(self):
        """Initialize PII detector with regex patterns."""
        self.patterns: Dict[str, Pattern] = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of detected PII items with type and location
        """
        detections = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                detections.append(
                    {
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return detections

    def mask(
        self,
        text: str,
        mask_char: str = "*",
        preserve_length: bool = True,
    ) -> str:
        """
        Mask PII in text.

        Args:
            text: Text to mask
            mask_char: Character to use for masking
            preserve_length: Whether to preserve original length

        Returns:
            Masked text
        """
        detections = self.detect(text)

        # Sort by position (reverse order to avoid index shifting)
        detections.sort(key=lambda x: x["start"], reverse=True)

        # Mask each detection
        for detection in detections:
            start = detection["start"]
            end = detection["end"]
            original = detection["value"]

            if preserve_length:
                masked = mask_char * len(original)
            else:
                masked = f"[{detection['type'].upper()}]"

            text = text[:start] + masked + text[end:]

        return text


class DataMasker:
    """
    Mask sensitive data in structured formats (dict, list).
    """

    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        """
        Initialize data masker.

        Args:
            pii_detector: PII detector instance (creates new if None)
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.sensitive_keys = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
        }

    def mask(
        self,
        data: Any,
        mask_pii: bool = True,
        mask_sensitive_keys: bool = True,
    ) -> Any:
        """
        Mask sensitive data.

        Args:
            data: Data to mask (dict, list, or primitive)
            mask_pii: Whether to mask PII in strings
            mask_sensitive_keys: Whether to mask values of sensitive keys

        Returns:
            Masked data
        """
        if isinstance(data, dict):
            return self._mask_dict(data, mask_pii, mask_sensitive_keys)
        elif isinstance(data, list):
            return self._mask_list(data, mask_pii, mask_sensitive_keys)
        elif isinstance(data, str):
            if mask_pii:
                return self.pii_detector.mask(data)
            return data
        else:
            return data

    def _mask_dict(
        self,
        data: dict,
        mask_pii: bool,
        mask_sensitive_keys: bool,
    ) -> dict:
        """Mask sensitive data in dictionary."""
        masked = {}
        for key, value in data.items():
            # Check if key is sensitive
            if mask_sensitive_keys and key.lower() in self.sensitive_keys:
                masked[key] = "***MASKED***"
            else:
                masked[key] = self.mask(value, mask_pii, mask_sensitive_keys)
        return masked

    def _mask_list(
        self,
        data: list,
        mask_pii: bool,
        mask_sensitive_keys: bool,
    ) -> list:
        """Mask sensitive data in list."""
        return [self.mask(item, mask_pii, mask_sensitive_keys) for item in data]


# Global instances
_pii_detector = PIIDetector()
_data_masker = DataMasker(_pii_detector)


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Detect PII in text using global detector.

    Args:
        text: Text to scan

    Returns:
        List of detected PII items
    """
    return _pii_detector.detect(text)


def mask_pii(text: str, mask_char: str = "*") -> str:
    """
    Mask PII in text using global detector.

    Args:
        text: Text to mask
        mask_char: Character to use for masking

    Returns:
        Masked text
    """
    return _pii_detector.mask(text, mask_char=mask_char)


def mask_data(data: Any, mask_pii: bool = True, mask_sensitive_keys: bool = True) -> Any:
    """
    Mask sensitive data using global masker.

    Args:
        data: Data to mask
        mask_pii: Whether to mask PII
        mask_sensitive_keys: Whether to mask sensitive keys

    Returns:
        Masked data
    """
    return _data_masker.mask(data, mask_pii, mask_sensitive_keys)

"""
SDK Configuration
"""

import os
from typing import Any, Optional


class Config:
    """
    SDK configuration class.

    Manages configuration settings for the WhiteBoxAI SDK including API keys,
    URLs, timeouts, and feature flags.

    Args:
        api_key: WhiteBoxAI API key
        base_url: Base URL for WhiteBoxAI API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        **kwargs: Additional configuration options
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize configuration."""
        # API key (from parameter or environment)
        self.api_key = api_key or os.getenv("EXPLAINAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide via 'api_key' parameter or "
                "EXPLAINAI_API_KEY environment variable."
            )

        # Base URL
        self.base_url = base_url or os.getenv("EXPLAINAI_BASE_URL") or "https://api.whiteboxai.io"

        # Request settings
        self.timeout = timeout
        self.max_retries = max_retries

        # Feature flags
        self.enable_caching = kwargs.get("enable_caching", True)
        self.enable_offline_mode = kwargs.get("enable_offline_mode", False)
        self.enable_privacy_filters = kwargs.get("enable_privacy_filters", True)
        self.enable_sampling = kwargs.get("enable_sampling", True)

        # Sampling settings
        self.sampling_rate = kwargs.get("sampling_rate", 1.0)  # 100% by default

        # Cache settings
        self.cache_ttl = kwargs.get("cache_ttl", 3600)  # 1 hour default
        self.cache_max_size = kwargs.get("cache_max_size", 1000)

        # Privacy settings
        self.detect_pii = kwargs.get("detect_pii", True)
        self.mask_pii = kwargs.get("mask_pii", True)

        # Performance settings
        self.batch_size = kwargs.get("batch_size", 100)
        self.async_enabled = kwargs.get("async_enabled", True)

        # SDK metadata
        self.sdk_version = "0.2.0"

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enable_caching": self.enable_caching,
            "enable_offline_mode": self.enable_offline_mode,
            "enable_privacy_filters": self.enable_privacy_filters,
            "enable_sampling": self.enable_sampling,
            "sampling_rate": self.sampling_rate,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "detect_pii": self.detect_pii,
            "mask_pii": self.mask_pii,
            "batch_size": self.batch_size,
            "async_enabled": self.async_enabled,
            "sdk_version": self.sdk_version,
        }

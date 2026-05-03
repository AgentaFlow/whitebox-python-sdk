"""
WhiteBoxAI Client

Main client class for interacting with the WhiteBoxAI API.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx
from whiteboxai.config import Config
from whiteboxai.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)
from whiteboxai.resources import (
    AgentWorkflowsResource,
    AlertsResource,
    DriftResource,
    ExplanationsResource,
    ModelsResource,
    PredictionsResource,
)
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class WhiteBoxAI:
    """
    Main client for WhiteBoxAI SDK.

    Provides access to all WhiteBoxAI API resources including models, predictions,
    explanations, drift detection, and alerting.

    Args:
        api_key: WhiteBoxAI API key
        base_url: Base URL for WhiteBoxAI API (default: https://api.whiteboxai.io)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retry attempts (default: 3)
        enable_offline: Enable offline mode (queues operations when API unavailable)
        offline_dir: Directory for offline queue storage (default: ./whiteboxai_offline)
        offline_max_queue_size: Maximum operations to queue (default: 10000, 0 = unlimited)
        offline_auto_sync: Enable automatic syncing (default: True)
        offline_sync_interval: Seconds between sync attempts (default: 60)
        **kwargs: Additional configuration options

    Example:
        >>> client = WhiteBoxAI(api_key="your_api_key")
        >>> model = client.models.register(name="fraud_detection", model_type="classification")
        >>> client.predictions.log(model_id=model.id, inputs=features, outputs=prediction)

        With offline mode:
        >>> client = WhiteBoxAI(
        ...     api_key="your_api_key",
        ...     enable_offline=True,
        ...     offline_dir="./offline_queue"
        ... )
        >>> # Operations queued when offline
        >>> client.predictions.log(...)  # Queued if API unavailable
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.whiteboxai.io",
        timeout: int = 30,
        max_retries: int = 3,
        enable_offline: bool = False,
        offline_dir: str = "./whiteboxai_offline",
        offline_max_queue_size: int = 10000,
        offline_auto_sync: bool = True,
        offline_sync_interval: int = 60,
        **kwargs: Any,
    ):
        """Initialize WhiteBoxAI client."""
        self.config = Config(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Initialize HTTP clients
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

        # Initialize offline mode
        self._offline_manager = None
        if enable_offline:
            from whiteboxai.offline import OfflineManager
            self._offline_manager = OfflineManager(
                offline_dir=offline_dir,
                max_queue_size=offline_max_queue_size,
                auto_sync=offline_auto_sync,
                sync_interval=offline_sync_interval
            )
            self._offline_manager.set_client(self)
            logger.info("Offline mode enabled")

        # Initialize resource managers
        self.models = ModelsResource(self)
        self.predictions = PredictionsResource(self)
        self.explanations = ExplanationsResource(self)
        self.drift = DriftResource(self)
        self.alerts = AlertsResource(self)
        self.agent_workflows = AgentWorkflowsResource(self)

    @property
    def sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.config.base_url,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
        return self._async_client

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"whiteboxai-python-sdk/{self.config.sdk_version}",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make synchronous HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            **kwargs: Additional request options

        Returns:
            Response data as dictionary

        Raises:
            APIError: For general API errors
            AuthenticationError: For authentication failures
            RateLimitError: For rate limit errors
            ValidationError: For validation errors
        """
        url = urljoin(self.config.base_url, endpoint)

        try:
            response = self.sync_client.request(
                method=method, url=url, json=data, params=params, **kwargs
            )
            return self._handle_response(response)
        except httpx.HTTPError as e:
            raise APIError(f"Request failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def arequest(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make asynchronous HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            **kwargs: Additional request options

        Returns:
            Response data as dictionary

        Raises:
            APIError: For general API errors
            AuthenticationError: For authentication failures
            RateLimitError: For rate limit errors
            ValidationError: For validation errors
        """
        url = urljoin(self.config.base_url, endpoint)

        try:
            response = await self.async_client.request(
                method=method, url=url, json=data, params=params, **kwargs
            )
            return self._handle_response(response)
        except httpx.HTTPError as e:
            raise APIError(f"Request failed: {str(e)}")

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Handle HTTP response and raise appropriate exceptions.

        Args:
            response: HTTP response

        Returns:
            Response data as dictionary

        Raises:
            AuthenticationError: For 401 status codes
            RateLimitError: For 429 status codes
            ValidationError: For 422 status codes
            APIError: For other error status codes
        """
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key or authentication failed")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded. Please retry later.")
        elif response.status_code == 422:
            try:
                error_data = response.json()
                raise ValidationError(f"Validation error: {error_data}")
            except Exception:
                raise ValidationError("Validation error occurred")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("detail", response.text)
            except Exception:
                message = response.text
            raise APIError(f"API error ({response.status_code}): {message}")

        try:
            return response.json()
        except Exception:
            return {"status": "success"}

    def is_offline_enabled(self) -> bool:
        """Check if offline mode is enabled."""
        return self._offline_manager is not None

    def sync_offline_queue(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Manually sync offline queue with API.

        Args:
            batch_size: Number of operations to sync per batch

        Returns:
            Dictionary with sync statistics

        Raises:
            ValueError: If offline mode is not enabled
        """
        if not self._offline_manager:
            raise ValueError("Offline mode is not enabled")

        return self._offline_manager.sync(batch_size=batch_size)

    def get_offline_status(self) -> Dict[str, Any]:
        """
        Get offline mode status.

        Returns:
            Status dictionary with queue stats

        Raises:
            ValueError: If offline mode is not enabled
        """
        if not self._offline_manager:
            raise ValueError("Offline mode is not enabled")

        return self._offline_manager.get_status()

    def cleanup_offline_queue(self, older_than_days: int = 7):
        """
        Clean up old completed operations from offline queue.

        Args:
            older_than_days: Remove operations older than this many days

        Raises:
            ValueError: If offline mode is not enabled
        """
        if not self._offline_manager:
            raise ValueError("Offline mode is not enabled")

        self._offline_manager.cleanup(older_than_days=older_than_days)

    def close(self) -> None:
        """Close HTTP clients and cleanup resources."""
        # Stop offline sync if enabled
        if self._offline_manager:
            self._offline_manager.stop_auto_sync()
            logger.info("Offline mode stopped")

        if self._sync_client:
            self._sync_client.close()
        if self._async_client:
            asyncio.run(self._async_client.aclose())

    async def aclose(self) -> None:
        """Asynchronously close HTTP clients and cleanup resources."""
        # Stop offline sync if enabled
        if self._offline_manager:
            self._offline_manager.stop_auto_sync()
            logger.info("Offline mode stopped")

        if self._async_client:
            await self._async_client.aclose()
        if self._sync_client:
            self._sync_client.close()

    def __enter__(self) -> "WhiteBoxAI":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> "WhiteBoxAI":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

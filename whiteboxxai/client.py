"""
WhiteBoxXAI Client

Main client class for interacting with the WhiteBoxXAI API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urljoin

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from whiteboxxai.config import Config
from whiteboxxai.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from whiteboxxai.resources import (
    AgentWorkflowsResource,
    AlertIntelligenceResource,
    AlertManagementResource,
    AlertsResource,
    ComplianceResource,
    DriftResource,
    ExplanationsResource,
    FairnessResource,
    GovernanceResource,
    LLMResource,
    LLMXAIResource,
    MetricsResource,
    ModelsResource,
    PredictionsResource,
    PromptsResource,
    RAGResource,
    ReportTypesResource,
    RiskRegisterResource,
    SafetyResource,
)
from whiteboxxai.runs import with_run_header

logger = logging.getLogger(__name__)


#: A 429 that asks for a longer wait than this is raised, not slept on. Short
#: waits are throttling worth riding out; a long one is a lockout or an exhausted
#: quota, and sleeping through it (twice, at up to 15 minutes each) would hold an
#: interactive call -- a governance vote, say -- for far longer than any client
#: would wait. The caller gets the RateLimitError, with ``retry_after`` set.
MAX_AUTO_RETRY_WAIT_SECONDS = 60


def _should_retry(exc: BaseException) -> bool:
    """Which failures ``request``/``arequest`` retry automatically."""
    if isinstance(exc, RateLimitError):
        return exc.retry_after is None or exc.retry_after <= MAX_AUTO_RETRY_WAIT_SECONDS
    return isinstance(exc, (APIConnectionError, ServerError))


def _retry_wait(retry_state: Any) -> float:
    """Wait strategy for the request retry decorators.

    Honors RateLimitError.retry_after when the server told us exactly how
    long to wait (a 429 response's Retry-After header); otherwise falls
    back to the same exponential backoff used before this was split out.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitError) and exc.retry_after:
        return float(exc.retry_after)
    return wait_exponential(multiplier=1, min=2, max=10)(retry_state)


class WhiteBoxXAI:
    """
    Main client for WhiteBoxXAI SDK.

    Provides access to all WhiteBoxXAI API resources including models, predictions,
    explanations, drift detection, fairness/bias audits, alerting, LLM usage
    monitoring, RAG pipeline monitoring, and content safety analysis.

    Args:
        api_key: WhiteBoxXAI API key (default: the WHITEBOX_API_KEY environment
            variable, or WHITEBOXXAI_API_KEY, the name SDK 1.2.0 read)
        base_url: Base URL for WhiteBoxXAI API (default: the WHITEBOX_BASE_URL
            environment variable, else https://api.whiteboxxai.com)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retry attempts (default: 3)
        enable_offline: Enable offline mode (queues operations when API unavailable)
        offline_dir: Directory for offline queue storage (default: ./WHITEBOX_offline,
            or ./whiteboxxai_offline, the SDK 1.2.0 default, if only that exists)
        offline_max_queue_size: Maximum operations to queue (default: 10000, 0 = unlimited)
        offline_auto_sync: Enable automatic syncing (default: True)
        offline_sync_interval: Seconds between sync attempts (default: 60)
        **kwargs: Additional configuration options

    Example:
        >>> client = WhiteBoxXAI(api_key="your_api_key")
        >>> model = client.models.register(
        ...     name="fraud_detection",
        ...     version="1.0.0",
        ...     model_type="random_forest",
        ...     features=["amount", "merchant_category"],
        ...     baseline_metrics={"accuracy": 0.94},
        ... )
        >>> client.predictions.log(model_id=model["id"], inputs=features, outputs=prediction)

        With offline mode:
        >>> client = WhiteBoxXAI(
        ...     api_key="your_api_key",
        ...     enable_offline=True,
        ...     offline_dir="./offline_queue"
        ... )
        >>> # Operations queued when offline
        >>> client.predictions.log(...)  # Queued if API unavailable
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        enable_offline: bool = False,
        offline_dir: str | None = None,
        offline_max_queue_size: int = 10000,
        offline_auto_sync: bool = True,
        offline_sync_interval: int = 60,
        **kwargs: Any,
    ):
        """Initialize WhiteBoxXAI client."""
        self.config = Config(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Initialize HTTP clients
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

        # Initialize offline mode
        self._offline_manager = None
        if enable_offline:
            from whiteboxxai.offline import OfflineManager

            self._offline_manager = OfflineManager(
                offline_dir=offline_dir,
                max_queue_size=offline_max_queue_size,
                auto_sync=offline_auto_sync,
                sync_interval=offline_sync_interval,
            )
            self._offline_manager.set_client(self)
            logger.info("Offline mode enabled")

        # Initialize resource managers
        self.models = ModelsResource(self)
        self.predictions = PredictionsResource(self)
        self.explanations = ExplanationsResource(self)
        self.drift = DriftResource(self)
        self.fairness = FairnessResource(self)
        self.alerts = AlertsResource(self)
        self.reports = ReportTypesResource(self)
        self.alert_intelligence = AlertIntelligenceResource(self)
        self.alert_management = AlertManagementResource(self)
        self.risk_register = RiskRegisterResource(self)
        self.governance = GovernanceResource(self)
        self.compliance = ComplianceResource(self)
        self.llm = LLMResource(self)
        self.rag = RAGResource(self)
        self.safety = SafetyResource(self)
        self.llm_xai = LLMXAIResource(self)
        self.agent_workflows = AgentWorkflowsResource(self)
        self.metrics = MetricsResource(self)
        self.prompts = PromptsResource(self)

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

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"whiteboxxai-python-sdk/{self.config.sdk_version}",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=_retry_wait,
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make synchronous HTTP request with retry logic.

        Only retries on transient failures (connection errors, 5xx, 429) --
        a 4xx like AuthenticationError/ValidationError/a plain 404 will
        never succeed on replay, so those raise immediately instead of
        burning 3 attempts' worth of backoff on a request that can't work.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            **kwargs: Additional request options

        Returns:
            Response data as dictionary

        Raises:
            APIConnectionError: If the request never reached the server
            ServerError: For 5xx responses
            APIError: For other general API errors
            AuthenticationError: For authentication failures
            RateLimitError: For rate limit errors
            ValidationError: For validation errors
        """
        url = urljoin(self.config.base_url, endpoint)

        try:
            response = self.sync_client.request(
                method=method,
                url=url,
                json=data,
                params=params,
                **with_run_header(kwargs),
            )
            return self._handle_response(response)
        except httpx.TransportError as e:
            raise APIConnectionError(f"Connection failed: {str(e)}") from e
        except httpx.HTTPError as e:
            raise APIError(f"Request failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=_retry_wait,
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async def arequest(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make asynchronous HTTP request with retry logic.

        Only retries on transient failures -- see request()'s docstring.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            **kwargs: Additional request options

        Returns:
            Response data as dictionary

        Raises:
            APIConnectionError: If the request never reached the server
            ServerError: For 5xx responses
            APIError: For other general API errors
            AuthenticationError: For authentication failures
            RateLimitError: For rate limit errors
            ValidationError: For validation errors
        """
        url = urljoin(self.config.base_url, endpoint)

        try:
            response = await self.async_client.request(
                method=method,
                url=url,
                json=data,
                params=params,
                **with_run_header(kwargs),
            )
            return self._handle_response(response)
        except httpx.TransportError as e:
            raise APIConnectionError(f"Connection failed: {str(e)}") from e
        except httpx.HTTPError as e:
            raise APIError(f"Request failed: {str(e)}") from e

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
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
            ServerError: For 5xx status codes
            APIError: For other error status codes
        """
        if response.status_code >= 500:
            try:
                error_data = response.json()
                message = error_data.get("detail", response.text)
            except Exception:
                error_data = None
                message = response.text
            raise ServerError(
                f"Server error ({response.status_code}): {message}",
                status_code=response.status_code,
                response=error_data,
                request_id=response.headers.get("X-Request-ID"),
            )
        elif response.status_code == 401:
            raise AuthenticationError(
                "Invalid API key or authentication failed",
                status_code=response.status_code,
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            # Not every 429 is a plan quota: a locked-out two-factor attempt is
            # one too, and the backend's own words are the only way to tell them
            # apart.
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = None
            raise RateLimitError(
                "Rate limit exceeded. Please retry later."
                + (f" ({detail})" if isinstance(detail, str) and detail else ""),
                retry_after=(
                    int(retry_after)
                    if isinstance(retry_after, str) and retry_after.isdigit()
                    else None
                ),
            )
        elif response.status_code == 422:
            try:
                error_data = response.json()
                raise ValidationError(f"Validation error: {error_data}", fields=error_data)
            except ValidationError:
                raise
            except Exception:
                raise ValidationError("Validation error occurred")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("detail", response.text)
            except Exception:
                error_data = None
                message = response.text
            raise APIError(
                f"API error ({response.status_code}): {message}",
                status_code=response.status_code,
                response=error_data,
                request_id=response.headers.get("X-Request-ID"),
            )

        try:
            return response.json()
        except Exception:
            return {"status": "success"}

    # ------------------------------------------------------------------
    # Offline-queue replay targets.
    #
    # These mirror the sync write methods on ModelsResource/PredictionsResource/
    # LLMResource (resources.py) that catch APIConnectionError and
    # enqueue instead of raising when offline mode is enabled. Each one
    # re-issues the exact request/params those methods already built --
    # none of them re-derive any business logic (e.g. git-context
    # auto-detection) -- so OfflineManager.sync() (offline.py) can replay
    # a queued operation with `self._client._api_*(**data)` using whatever
    # `data` the original call enqueued.
    # ------------------------------------------------------------------

    def _api_predict(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/predictions/log", data=data)

    def _api_log_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/predictions/log/batch", data=data)

    def _api_register_model(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/models/", data=data)

    def _api_llm_log(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/llm/logs", data=data)

    def _api_update_baseline(
        self,
        model_id: str,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.request(
            "PATCH",
            f"/api/v1/models/{model_id}/baseline",
            data=data,
            params=params or {},
        )

    def is_offline_enabled(self) -> bool:
        """Check if offline mode is enabled."""
        return self._offline_manager is not None

    def sync_offline_queue(self, batch_size: int = 100) -> dict[str, int]:
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

    def get_offline_status(self) -> dict[str, Any]:
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
        """Close HTTP clients and cleanup resources.

        Raises:
            RuntimeError: If called from inside a running event loop --
                `asyncio.run()` can't be nested, and doing so used to crash
                with an opaque asyncio error instead of this clear message.
                Use `await client.aclose()` in async code instead.
        """
        # Stop offline sync if enabled
        if self._offline_manager:
            self._offline_manager.stop_auto_sync()
            logger.info("Offline mode stopped")

        if self._sync_client:
            self._sync_client.close()
        if self._async_client:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop in this thread -- safe to spin one up
                # just to close the async client.
                asyncio.run(self._async_client.aclose())
            else:
                raise RuntimeError(
                    "close() cannot close the async HTTP client from inside "
                    "a running event loop. Use `await client.aclose()` "
                    "instead when running inside async code."
                )

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

    def __enter__(self) -> "WhiteBoxXAI":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> "WhiteBoxXAI":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

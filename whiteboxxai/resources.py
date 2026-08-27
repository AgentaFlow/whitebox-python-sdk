"""
API Resources

Resource classes for interacting with different WhiteBoxXAI API endpoints.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from whiteboxxai.exceptions import APIConnectionError

if TYPE_CHECKING:
    from whiteboxxai.client import WhiteBoxXAI

logger = logging.getLogger(__name__)


class BaseResource:
    """Base class for API resources."""

    def __init__(self, client: "WhiteBoxXAI"):
        """Initialize resource with client."""
        self.client = client

    def _request_or_queue(
        self,
        method: str,
        endpoint: str,
        operation_type: str,
        queue_data: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST/PATCH, falling back to the offline queue on connection failure.

        Only the four sync write methods that mutate server state (model
        register/update_baseline, prediction log/log_batch) use this --
        offline mode's queue is inherently synchronous (SQLite-backed), so
        the async (`a*`) variants are unaffected and simply raise as
        before. `queue_data` must match the kwargs the corresponding
        `WhiteBoxXAI._api_*` replay method (client.py) expects.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            operation_type: whiteboxxai.offline.OperationType value name
                (e.g. "REGISTER_MODEL") for this call
            queue_data: Exact kwargs to replay via `_api_*(**queue_data)`
            data: Request body data
            params: Query parameters

        Returns:
            The normal response, or `{"status": "queued", "operation_id": ...}`
            when offline mode is enabled and the request couldn't reach
            the server.

        Raises:
            APIConnectionError: If the request couldn't reach the server
                and offline mode is not enabled
        """
        try:
            return self.client.request(method, endpoint, data=data, params=params)
        except APIConnectionError:
            if not self.client.is_offline_enabled():
                raise

            from whiteboxxai.offline import OperationType

            op_id = self.client._offline_manager.queue.enqueue(
                OperationType[operation_type], queue_data
            )
            logger.info(
                "API unreachable; queued %s as operation %s for offline sync",
                operation_type,
                op_id,
            )
            return {"status": "queued", "operation_id": op_id}


class ModelsResource(BaseResource):
    """Models API resource (backed by backend/api/v1/models.py)."""

    def _build_register_data(
        self,
        name: str,
        model_type: str,
        framework: Optional[str],
        version: Optional[str],
        auto_detect_git: bool,
        require_clean_git: bool,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = {
            "name": name,
            "model_type": model_type,
            "framework": framework,
            "version": version,
            **kwargs,
        }

        if auto_detect_git:
            from whiteboxxai.git_utils import detect_git_context, validate_git_context

            git_context = detect_git_context()
            if git_context:
                if validate_git_context(git_context, require_clean=require_clean_git):
                    git_data = git_context.to_dict()
                    for key, value in git_data.items():
                        if key not in data and value is not None:
                            data[key] = value
                    logger.info(f"Auto-detected Git context: {git_context}")
                else:
                    logger.warning("Git context validation failed")
            else:
                logger.warning(
                    "auto_detect_git=True but no Git repository found. "
                    "Proceeding without Git metadata."
                )

        return data

    def register(
        self,
        name: str,
        model_type: str,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        auto_detect_git: bool = False,
        require_clean_git: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Register a new model.

        Args:
            name: Model name
            model_type: Model type (classification, regression, etc.)
            framework: ML framework (sklearn, pytorch, tensorflow, etc.)
            version: Model version
            auto_detect_git: Automatically detect Git repository context (commit, branch, etc.)
            require_clean_git: Require clean Git working directory (no uncommitted changes)
            **kwargs: Additional model metadata (features, baseline_metrics, tags, etc.)

        Returns:
            Registered model data

        Example:
            >>> client.models.register(
            ...     name="fraud_detector",
            ...     model_type="classification",
            ...     framework="sklearn",
            ...     features=["amount", "merchant_category"],
            ...     baseline_metrics={"accuracy": 0.94},
            ... )
        """
        data = self._build_register_data(
            name,
            model_type,
            framework,
            version,
            auto_detect_git,
            require_clean_git,
            kwargs,
        )
        return self._request_or_queue(
            "POST",
            "/api/v1/models/",
            operation_type="REGISTER_MODEL",
            queue_data={"data": data},
            data=data,
        )

    async def aregister(
        self,
        name: str,
        model_type: str,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        auto_detect_git: bool = False,
        require_clean_git: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of register()."""
        data = self._build_register_data(
            name,
            model_type,
            framework,
            version,
            auto_detect_git,
            require_clean_git,
            kwargs,
        )
        return await self.client.arequest("POST", "/api/v1/models/", data=data)

    def get(self, model_id: str) -> Dict[str, Any]:
        """Get model by ID."""
        return self.client.request("GET", f"/api/v1/models/{model_id}")

    async def aget(self, model_id: str) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/models/{model_id}")

    def list(
        self,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None,
        model_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        tags: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List registered models with optional filters."""
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if status_filter is not None:
            params["status_filter"] = status_filter
        if model_type is not None:
            params["model_type"] = model_type
        if owner_id is not None:
            params["owner_id"] = owner_id
        if tags is not None:
            params["tags"] = tags
        if search is not None:
            params["search"] = search
        return self.client.request("GET", "/api/v1/models/", params=params)

    async def alist(
        self,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None,
        model_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        tags: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of list()."""
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if status_filter is not None:
            params["status_filter"] = status_filter
        if model_type is not None:
            params["model_type"] = model_type
        if owner_id is not None:
            params["owner_id"] = owner_id
        if tags is not None:
            params["tags"] = tags
        if search is not None:
            params["search"] = search
        return await self.client.arequest("GET", "/api/v1/models/", params=params)

    def update(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Update model metadata (partial update)."""
        return self.client.request("PATCH", f"/api/v1/models/{model_id}", data=kwargs)

    async def aupdate(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Async version of update()."""
        return await self.client.arequest("PATCH", f"/api/v1/models/{model_id}", data=kwargs)

    def update_status(self, model_id: str, new_status: str) -> Dict[str, Any]:
        """Update model status (ACTIVE, INACTIVE, DEPRECATED, ARCHIVED)."""
        return self.client.request(
            "PATCH",
            f"/api/v1/models/{model_id}/status",
            params={"new_status": new_status},
        )

    async def aupdate_status(self, model_id: str, new_status: str) -> Dict[str, Any]:
        """Async version of update_status()."""
        return await self.client.arequest(
            "PATCH",
            f"/api/v1/models/{model_id}/status",
            params={"new_status": new_status},
        )

    def get_versions(
        self, model_name: str, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all versions of a model by name."""
        return self.client.request(
            "GET",
            f"/api/v1/models/{model_name}/versions",
            params={"skip": skip, "limit": limit},
        )

    async def aget_versions(
        self, model_name: str, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Async version of get_versions()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/models/{model_name}/versions",
            params={"skip": skip, "limit": limit},
        )

    def get_latest(self, model_name: str) -> Dict[str, Any]:
        """Get the latest version of a model by name."""
        return self.client.request("GET", f"/api/v1/models/{model_name}/latest")

    async def aget_latest(self, model_name: str) -> Dict[str, Any]:
        """Async version of get_latest()."""
        return await self.client.arequest("GET", f"/api/v1/models/{model_name}/latest")

    def update_baseline(
        self,
        model_id: str,
        baseline_metrics: Dict[str, Any],
        baseline_data_hash: Optional[str] = None,
        baseline_data_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update model baseline metrics/profile (e.g. after retraining)."""
        params: Dict[str, Any] = {}
        if baseline_data_hash is not None:
            params["baseline_data_hash"] = baseline_data_hash
        if baseline_data_count is not None:
            params["baseline_data_count"] = baseline_data_count
        return self._request_or_queue(
            "PATCH",
            f"/api/v1/models/{model_id}/baseline",
            operation_type="UPDATE_BASELINE",
            queue_data={
                "model_id": model_id,
                "data": baseline_metrics,
                "params": params,
            },
            data=baseline_metrics,
            params=params,
        )

    async def aupdate_baseline(
        self,
        model_id: str,
        baseline_metrics: Dict[str, Any],
        baseline_data_hash: Optional[str] = None,
        baseline_data_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async version of update_baseline()."""
        params: Dict[str, Any] = {}
        if baseline_data_hash is not None:
            params["baseline_data_hash"] = baseline_data_hash
        if baseline_data_count is not None:
            params["baseline_data_count"] = baseline_data_count
        return await self.client.arequest(
            "PATCH",
            f"/api/v1/models/{model_id}/baseline",
            data=baseline_metrics,
            params=params,
        )

    def archive(self, model_id: str) -> Dict[str, Any]:
        """Archive a model (soft delete)."""
        return self.client.request("POST", f"/api/v1/models/{model_id}/archive")

    async def aarchive(self, model_id: str) -> Dict[str, Any]:
        """Async version of archive()."""
        return await self.client.arequest("POST", f"/api/v1/models/{model_id}/archive")

    def restore(self, model_id: str) -> Dict[str, Any]:
        """Restore an archived model to active status."""
        return self.client.request("POST", f"/api/v1/models/{model_id}/restore")

    async def arestore(self, model_id: str) -> Dict[str, Any]:
        """Async version of restore()."""
        return await self.client.arequest("POST", f"/api/v1/models/{model_id}/restore")

    def delete(self, model_id: str) -> Dict[str, Any]:
        """Permanently delete a model (hard delete, cannot be undone)."""
        return self.client.request("DELETE", f"/api/v1/models/{model_id}")

    async def adelete(self, model_id: str) -> Dict[str, Any]:
        """Async version of delete()."""
        return await self.client.arequest("DELETE", f"/api/v1/models/{model_id}")


class PredictionsResource(BaseResource):
    """Predictions API resource (backed by backend/api/v1/predictions.py)."""

    def log(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        prediction_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a single prediction.

        Args:
            model_id: Model ID
            input_data: Input features/data
            output_data: Model prediction/output
            prediction_id: Optional external prediction ID
            latency_ms: Prediction latency in milliseconds
            metadata: Additional prediction metadata

        Returns:
            Logged prediction data
        """
        data: Dict[str, Any] = {
            "model_id": model_id,
            "input_data": input_data,
            "output_data": output_data,
        }
        if prediction_id is not None:
            data["prediction_id"] = prediction_id
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        if metadata is not None:
            data["metadata"] = metadata
        return self._request_or_queue(
            "POST",
            "/api/v1/predictions/log",
            operation_type="PREDICT",
            queue_data={"data": data},
            data=data,
        )

    async def alog(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        prediction_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of log()."""
        data: Dict[str, Any] = {
            "model_id": model_id,
            "input_data": input_data,
            "output_data": output_data,
        }
        if prediction_id is not None:
            data["prediction_id"] = prediction_id
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        if metadata is not None:
            data["metadata"] = metadata
        return await self.client.arequest("POST", "/api/v1/predictions/log", data=data)

    def log_batch(
        self,
        model_id: str,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Log multiple predictions in batch (max 1000 per call).

        Args:
            model_id: Model ID
            predictions: List of dicts, each with input_data/output_data
                (and optionally prediction_id/latency_ms/metadata)

        Returns:
            Batch logging summary: {total, logged, failed, errors}
        """
        data = {"model_id": model_id, "predictions": predictions}
        return self._request_or_queue(
            "POST",
            "/api/v1/predictions/log/batch",
            operation_type="LOG_BATCH",
            queue_data={"data": data},
            data=data,
        )

    async def alog_batch(
        self,
        model_id: str,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Async version of log_batch()."""
        data = {"model_id": model_id, "predictions": predictions}
        return await self.client.arequest("POST", "/api/v1/predictions/log/batch", data=data)

    def get(self, prediction_id: str) -> Dict[str, Any]:
        """Get a prediction by its ID."""
        return self.client.request("GET", f"/api/v1/predictions/{prediction_id}")

    async def aget(self, prediction_id: str) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/predictions/{prediction_id}")

    def query(
        self,
        model_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query predictions with filters.

        Args:
            model_id: Model ID (required by the backend)
            start_time: ISO-8601 timestamp, filter predictions after this time
            end_time: ISO-8601 timestamp, filter predictions before this time
            limit: Maximum results to return
            offset: Number of results to skip
        """
        data: Dict[str, Any] = {"model_id": model_id, "limit": limit, "offset": offset}
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        return self.client.request("POST", "/api/v1/predictions/query", data=data)

    async def aquery(
        self,
        model_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Async version of query()."""
        data: Dict[str, Any] = {"model_id": model_id, "limit": limit, "offset": offset}
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        return await self.client.arequest("POST", "/api/v1/predictions/query", data=data)

    def get_stats(
        self,
        model_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get prediction statistics for a model."""
        params: Dict[str, Any] = {}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return self.client.request(
            "GET", f"/api/v1/predictions/models/{model_id}/stats", params=params
        )

    async def aget_stats(
        self,
        model_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of get_stats()."""
        params: Dict[str, Any] = {}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return await self.client.arequest(
            "GET", f"/api/v1/predictions/models/{model_id}/stats", params=params
        )

    def get_recent(self, model_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent predictions for a model (max 100)."""
        return self.client.request(
            "GET",
            f"/api/v1/predictions/models/{model_id}/recent",
            params={"limit": limit},
        )

    async def aget_recent(self, model_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Async version of get_recent()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/predictions/models/{model_id}/recent",
            params={"limit": limit},
        )


class ExplanationsResource(BaseResource):
    """Explanations API resource."""

    def generate(
        self,
        prediction_id: int,
        method: str = "shap",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.

        Args:
            prediction_id: Prediction ID
            method: Explanation method (shap, lime)
            **kwargs: Method-specific parameters

        Returns:
            Explanation data
        """
        data = {"prediction_id": prediction_id, "method": method, **kwargs}
        return self.client.request("POST", "/api/v1/explanations/generate", data=data)

    async def agenerate(
        self,
        prediction_id: int,
        method: str = "shap",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of generate()."""
        data = {"prediction_id": prediction_id, "method": method, **kwargs}
        return await self.client.arequest("POST", "/api/v1/explanations/generate", data=data)

    def generate_async(
        self,
        model_id: str,
        instance: Dict[str, Any],
        method: Optional[str] = None,
        prediction_id: Optional[str] = None,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> Dict[str, Any]:
        """
        Start asynchronous explanation generation (non-blocking).

        Creates a PENDING explanation record immediately and dispatches a
        real Celery task to compute SHAP/LIME output in the background.
        Poll get()/aget() with the returned explanation_id until its status
        is no longer "pending".

        Args:
            model_id: ID of the model to explain
            instance: Feature values for the instance to explain
            method: "shap" or "lime" (uses the model's configured default if omitted)
            prediction_id: Optional associated prediction ID
            num_features: Number of top features to return (1-100)
            num_samples: Number of samples the explainer draws (100-50000)

        Returns:
            {"explanation_id": ..., "status": "pending", "message": ...}
        """
        data = {
            "model_id": model_id,
            "instance": instance,
            "method": method,
            "prediction_id": prediction_id,
            "num_features": num_features,
            "num_samples": num_samples,
        }
        return self.client.request("POST", "/api/v1/explanations/generate/async", data=data)

    async def agenerate_async(
        self,
        model_id: str,
        instance: Dict[str, Any],
        method: Optional[str] = None,
        prediction_id: Optional[str] = None,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> Dict[str, Any]:
        """Async version of generate_async()."""
        data = {
            "model_id": model_id,
            "instance": instance,
            "method": method,
            "prediction_id": prediction_id,
            "num_features": num_features,
            "num_samples": num_samples,
        }
        return await self.client.arequest("POST", "/api/v1/explanations/generate/async", data=data)

    def get(self, explanation_id: str) -> Dict[str, Any]:
        """Get explanation by ID."""
        return self.client.request("GET", f"/api/v1/explanations/{explanation_id}")

    async def aget(self, explanation_id: str) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/explanations/{explanation_id}")


class DriftResource(BaseResource):
    """Drift detection API resource (backed by backend/api/v1/drift.py)."""

    def detect(
        self,
        model_id: str,
        window_size: int = 1000,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect data drift for a model (ephemeral, not persisted).

        Args:
            model_id: Model ID
            window_size: Number of recent predictions to analyze (100-10000)
            feature_names: Optional subset of features to analyze

        Returns:
            Drift detection results
        """
        params: Dict[str, Any] = {"window_size": window_size}
        if feature_names is not None:
            params["feature_names"] = feature_names
        return self.client.request("POST", f"/api/v1/drift/detect/{model_id}", params=params)

    async def adetect(
        self,
        model_id: str,
        window_size: int = 1000,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async version of detect()."""
        params: Dict[str, Any] = {"window_size": window_size}
        if feature_names is not None:
            params["feature_names"] = feature_names
        return await self.client.arequest("POST", f"/api/v1/drift/detect/{model_id}", params=params)

    def create_report(self, model_id: str, window_size: int = 1000) -> Dict[str, Any]:
        """Run drift analysis and persist the result as a drift report."""
        return self.client.request(
            "POST",
            "/api/v1/drift/reports",
            params={"model_id": model_id, "window_size": window_size},
        )

    async def acreate_report(self, model_id: str, window_size: int = 1000) -> Dict[str, Any]:
        """Async version of create_report()."""
        return await self.client.arequest(
            "POST",
            "/api/v1/drift/reports",
            params={"model_id": model_id, "window_size": window_size},
        )

    def get_reports(self, model_id: str, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """List drift reports for a model, most recent first."""
        return self.client.request(
            "GET",
            f"/api/v1/drift/reports/{model_id}",
            params={"limit": limit, "skip": skip},
        )

    async def aget_reports(
        self, model_id: str, limit: int = 10, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Async version of get_reports()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/drift/reports/{model_id}",
            params={"limit": limit, "skip": skip},
        )

    def get_report(self, model_id: str, report_id: str) -> Dict[str, Any]:
        """Get a specific drift report with per-feature statistics."""
        return self.client.request("GET", f"/api/v1/drift/reports/{model_id}/{report_id}")

    async def aget_report(self, model_id: str, report_id: str) -> Dict[str, Any]:
        """Async version of get_report()."""
        return await self.client.arequest("GET", f"/api/v1/drift/reports/{model_id}/{report_id}")

    def get_trend(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """Get drift trend over time for a model."""
        return self.client.request("GET", f"/api/v1/drift/trend/{model_id}", params={"days": days})

    async def aget_trend(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """Async version of get_trend()."""
        return await self.client.arequest(
            "GET", f"/api/v1/drift/trend/{model_id}", params={"days": days}
        )


class FairnessResource(BaseResource):
    """Bias/fairness auditing API resource (backed by backend/api/v1/fairness.py)."""

    def audit(
        self,
        model_id: str,
        sensitive_attributes: List[str],
        y_true: List[int],
        y_pred: List[int],
        group_data: Dict[str, List[Any]],
        y_prob: Optional[List[float]] = None,
        fairness_thresholds: Optional[Dict[str, float]] = None,
        metrics_to_compute: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a bias audit on model predictions.

        Note: unlike other endpoints, this is not a data-source query — the
        caller must already have the true labels, predicted labels, and
        per-attribute group assignments in hand (e.g. pulled via
        PredictionsResource.query() and joined with demographic data).

        Args:
            model_id: Model ID (UUID string)
            sensitive_attributes: Protected attributes to analyze (e.g. ["gender", "race"])
            y_true: Ground-truth labels
            y_pred: Predicted labels
            group_data: Per-attribute group assignments, e.g. {"gender": ["M", "F", ...]}
            y_prob: Optional predicted probabilities
            fairness_thresholds: Optional custom thresholds per fairness metric type
            metrics_to_compute: Optional subset of fairness metrics to compute (all if None)

        Returns:
            Complete bias audit results with metrics, group comparisons, and recommendations
        """
        data: Dict[str, Any] = {
            "model_id": model_id,
            "sensitive_attributes": sensitive_attributes,
            "y_true": y_true,
            "y_pred": y_pred,
            "group_data": group_data,
        }
        if y_prob is not None:
            data["y_prob"] = y_prob
        if fairness_thresholds is not None:
            data["fairness_thresholds"] = fairness_thresholds
        if metrics_to_compute is not None:
            data["metrics_to_compute"] = metrics_to_compute
        return self.client.request("POST", "/api/v1/fairness/audit", data=data)

    async def aaudit(
        self,
        model_id: str,
        sensitive_attributes: List[str],
        y_true: List[int],
        y_pred: List[int],
        group_data: Dict[str, List[Any]],
        y_prob: Optional[List[float]] = None,
        fairness_thresholds: Optional[Dict[str, float]] = None,
        metrics_to_compute: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async version of audit()."""
        data: Dict[str, Any] = {
            "model_id": model_id,
            "sensitive_attributes": sensitive_attributes,
            "y_true": y_true,
            "y_pred": y_pred,
            "group_data": group_data,
        }
        if y_prob is not None:
            data["y_prob"] = y_prob
        if fairness_thresholds is not None:
            data["fairness_thresholds"] = fairness_thresholds
        if metrics_to_compute is not None:
            data["metrics_to_compute"] = metrics_to_compute
        return await self.client.arequest("POST", "/api/v1/fairness/audit", data=data)

    def get_audit(self, audit_id: str) -> Dict[str, Any]:
        """Get bias audit results by ID."""
        return self.client.request("GET", f"/api/v1/fairness/audits/{audit_id}")

    async def aget_audit(self, audit_id: str) -> Dict[str, Any]:
        """Async version of get_audit()."""
        return await self.client.arequest("GET", f"/api/v1/fairness/audits/{audit_id}")

    def list_audits(
        self,
        model_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List bias audits, optionally filtered by model."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if model_id is not None:
            params["model_id"] = model_id
        return self.client.request("GET", "/api/v1/fairness/audits", params=params)

    async def alist_audits(
        self,
        model_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Async version of list_audits()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if model_id is not None:
            params["model_id"] = model_id
        return await self.client.arequest("GET", "/api/v1/fairness/audits", params=params)

    def get_bias_history(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Get historical bias metrics/trend for a model."""
        return self.client.request(
            "GET",
            f"/api/v1/fairness/models/{model_id}/bias-history",
            params={"days": days},
        )

    async def aget_bias_history(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Async version of get_bias_history()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/fairness/models/{model_id}/bias-history",
            params={"days": days},
        )

    def get_metric_history(self, model_id: str, metric_type: str, days: int = 30) -> Dict[str, Any]:
        """Get history for a specific fairness metric type."""
        return self.client.request(
            "GET",
            f"/api/v1/fairness/models/{model_id}/metrics/{metric_type}/history",
            params={"days": days},
        )

    async def aget_metric_history(
        self, model_id: str, metric_type: str, days: int = 30
    ) -> Dict[str, Any]:
        """Async version of get_metric_history()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/fairness/models/{model_id}/metrics/{metric_type}/history",
            params={"days": days},
        )

    def get_latest_audit(self, model_id: str) -> Dict[str, Any]:
        """Get the most recent bias audit for a model."""
        return self.client.request("GET", f"/api/v1/fairness/models/{model_id}/latest-audit")

    async def aget_latest_audit(self, model_id: str) -> Dict[str, Any]:
        """Async version of get_latest_audit()."""
        return await self.client.arequest("GET", f"/api/v1/fairness/models/{model_id}/latest-audit")


class AlertsResource(BaseResource):
    """Alerts API resource (backed by backend/api/v1/alerts.py).

    create()/list() previously targeted /api/v1/alerts instead of the
    router's real /api/v1/alerts/rules path (a leftover from when the
    backend router was unregistered and this resource was never exercised
    against a live server -- see backend/api/v1/__init__.py's history note).
    Both are now fixed alongside the rest of the rule/instance CRUD surface
    (issue #141).
    """

    def create(
        self,
        name: str,
        alert_type: str,
        severity: str,
        conditions: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        logic_operator: str = "AND",
        notification_channels: Optional[List[Dict[str, Any]]] = None,
        throttle_minutes: int = 60,
        cooldown_minutes: int = 0,
        is_active: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create an alert rule.

        Args:
            name: Alert rule name
            alert_type: Alert type (e.g. "drift", "performance", "fairness")
            severity: Severity level (e.g. "low", "medium", "high", "critical")
            conditions: List of dicts, each {metric_name, operator
                (gt/gte/lt/lte/eq/ne), threshold, window_minutes?, aggregation?},
                combined by logic_operator
            model_id: Model to monitor (organization-wide if omitted)
            description: Rule description
            logic_operator: "AND" or "OR" -- how conditions combine
            notification_channels: List of notification channel configs
            throttle_minutes: Minimum minutes between repeated alerts
            cooldown_minutes: Cool-down period after a trigger
            is_active: Whether the rule is active on creation
            **kwargs: Additional AlertRuleCreate fields (metric_name,
                threshold_value, anomaly_config, schedule_config)

        Returns:
            Created alert rule
        """
        data = _alert_rule_payload(
            name=name,
            alert_type=alert_type,
            severity=severity,
            conditions=conditions,
            model_id=model_id,
            description=description,
            logic_operator=logic_operator,
            notification_channels=notification_channels,
            throttle_minutes=throttle_minutes,
            cooldown_minutes=cooldown_minutes,
            is_active=is_active,
            **kwargs,
        )
        return self.client.request("POST", "/api/v1/alerts/rules", data=data)

    async def acreate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of create()."""
        data = _alert_rule_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/alerts/rules", data=data)

    def list(
        self,
        model_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List alert rules for the caller's organization."""
        params = _drop_none(
            model_id=model_id,
            alert_type=alert_type,
            is_active=is_active,
            skip=skip,
            limit=limit,
        )
        return self.client.request("GET", "/api/v1/alerts/rules", params=params)

    async def alist(
        self,
        model_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of list()."""
        params = _drop_none(
            model_id=model_id,
            alert_type=alert_type,
            is_active=is_active,
            skip=skip,
            limit=limit,
        )
        return await self.client.arequest("GET", "/api/v1/alerts/rules", params=params)

    def get_rule(self, rule_id: str) -> Dict[str, Any]:
        """Get a specific alert rule by ID."""
        return self.client.request("GET", f"/api/v1/alerts/rules/{rule_id}")

    async def aget_rule(self, rule_id: str) -> Dict[str, Any]:
        """Async version of get_rule()."""
        return await self.client.arequest("GET", f"/api/v1/alerts/rules/{rule_id}")

    def update_rule(self, rule_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Update an alert rule. Any AlertRuleCreate field may be passed as
        a kwarg; only supplied fields are changed."""
        return self.client.request("PATCH", f"/api/v1/alerts/rules/{rule_id}", data=kwargs)

    async def aupdate_rule(self, rule_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Async version of update_rule()."""
        return await self.client.arequest("PATCH", f"/api/v1/alerts/rules/{rule_id}", data=kwargs)

    def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        """Delete an alert rule."""
        return self.client.request("DELETE", f"/api/v1/alerts/rules/{rule_id}")

    async def adelete_rule(self, rule_id: str) -> Dict[str, Any]:
        """Async version of delete_rule()."""
        return await self.client.arequest("DELETE", f"/api/v1/alerts/rules/{rule_id}")

    def evaluate_rule(
        self,
        rule_id: str,
        metric_values: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        historical_values: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Test whether an alert rule would trigger given metric values,
        without creating a real alert instance. Useful for validating a rule
        before activation."""
        data = _alert_evaluation_context_payload(
            metric_values=metric_values,
            model_id=model_id,
            historical_values=historical_values,
            metadata=metadata,
        )
        return self.client.request("POST", f"/api/v1/alerts/rules/{rule_id}/evaluate", data=data)

    async def aevaluate_rule(
        self,
        rule_id: str,
        metric_values: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        historical_values: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of evaluate_rule()."""
        data = _alert_evaluation_context_payload(
            metric_values=metric_values,
            model_id=model_id,
            historical_values=historical_values,
            metadata=metadata,
        )
        return await self.client.arequest(
            "POST", f"/api/v1/alerts/rules/{rule_id}/evaluate", data=data
        )

    def list_instances(
        self,
        rule_id: Optional[str] = None,
        model_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        hours: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List triggered alert instances with optional filters."""
        params = _drop_none(
            rule_id=rule_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            status=status,
            hours=hours,
            skip=skip,
            limit=limit,
        )
        return self.client.request("GET", "/api/v1/alerts/instances", params=params)

    async def alist_instances(
        self,
        rule_id: Optional[str] = None,
        model_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        hours: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of list_instances()."""
        params = _drop_none(
            rule_id=rule_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            status=status,
            hours=hours,
            skip=skip,
            limit=limit,
        )
        return await self.client.arequest("GET", "/api/v1/alerts/instances", params=params)

    def get_instance(self, alert_id: str) -> Dict[str, Any]:
        """Get a specific triggered alert instance by ID."""
        return self.client.request("GET", f"/api/v1/alerts/instances/{alert_id}")

    async def aget_instance(self, alert_id: str) -> Dict[str, Any]:
        """Async version of get_instance()."""
        return await self.client.arequest("GET", f"/api/v1/alerts/instances/{alert_id}")

    def acknowledge(
        self, alert_id: str, user_id: str, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Acknowledge an active alert instance."""
        data = _drop_none(user_id=user_id, notes=notes)
        return self.client.request(
            "POST", f"/api/v1/alerts/instances/{alert_id}/acknowledge", data=data
        )

    async def aacknowledge(
        self, alert_id: str, user_id: str, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of acknowledge()."""
        data = _drop_none(user_id=user_id, notes=notes)
        return await self.client.arequest(
            "POST", f"/api/v1/alerts/instances/{alert_id}/acknowledge", data=data
        )

    def resolve(
        self, alert_id: str, user_id: str, resolution_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolve an alert instance."""
        data = _drop_none(user_id=user_id, resolution_notes=resolution_notes)
        return self.client.request(
            "POST", f"/api/v1/alerts/instances/{alert_id}/resolve", data=data
        )

    async def aresolve(
        self, alert_id: str, user_id: str, resolution_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of resolve()."""
        data = _drop_none(user_id=user_id, resolution_notes=resolution_notes)
        return await self.client.arequest(
            "POST", f"/api/v1/alerts/instances/{alert_id}/resolve", data=data
        )

    def snooze(self, alert_id: str, snooze_minutes: int) -> Dict[str, Any]:
        """Snooze an active alert instance for `snooze_minutes` (max 1 week)."""
        return self.client.request(
            "POST",
            f"/api/v1/alerts/instances/{alert_id}/snooze",
            data={"snooze_minutes": snooze_minutes},
        )

    async def asnooze(self, alert_id: str, snooze_minutes: int) -> Dict[str, Any]:
        """Async version of snooze()."""
        return await self.client.arequest(
            "POST",
            f"/api/v1/alerts/instances/{alert_id}/snooze",
            data={"snooze_minutes": snooze_minutes},
        )

    def statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the organization (counts, severity
        distribution, MTTA/MTTR) over the last `hours`."""
        return self.client.request("GET", "/api/v1/alerts/statistics", params={"hours": hours})

    async def astatistics(self, hours: int = 24) -> Dict[str, Any]:
        """Async version of statistics()."""
        return await self.client.arequest(
            "GET", "/api/v1/alerts/statistics", params={"hours": hours}
        )


def _alert_rule_payload(
    name: str,
    alert_type: str,
    severity: str,
    conditions: List[Dict[str, Any]],
    model_id: Optional[str] = None,
    description: Optional[str] = None,
    logic_operator: str = "AND",
    notification_channels: Optional[List[Dict[str, Any]]] = None,
    throttle_minutes: int = 60,
    cooldown_minutes: int = 0,
    is_active: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Shared request-body builder for AlertsResource.create/acreate."""
    data: Dict[str, Any] = {
        "name": name,
        "alert_type": alert_type,
        "severity": severity,
        "conditions": conditions,
        "logic_operator": logic_operator,
        "notification_channels": notification_channels or [],
        "throttle_minutes": throttle_minutes,
        "cooldown_minutes": cooldown_minutes,
        "is_active": is_active,
    }
    data.update(_drop_none(model_id=model_id, description=description))
    data.update(kwargs)
    return data


def _alert_evaluation_context_payload(
    metric_values: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    historical_values: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for AlertsResource.evaluate_rule/aevaluate_rule."""
    return {
        "metric_values": metric_values or {},
        "model_id": model_id,
        "historical_values": historical_values,
        "metadata": metadata or {},
    }


class RiskRegisterResource(BaseResource):
    """AI Risk Register API resource (backed by backend/api/v1/risk_register.py).

    Read-only: the risk register is normally populated via the dashboard or
    auto-drafted from bias/drift/governance-review events server-side, not
    written to directly by SDK users.
    """

    def list(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """List AI risk register entries, optionally filtered.

        Args:
            status: One of identified, assessed, mitigation_planned, mitigated, closed
            severity: One of low, medium, high, critical
            category: Risk category (org-specific taxonomy, e.g. bias, drift, security)
            model_id: Restrict to risks linked to one registered model
            limit: Page size
            skip: Page offset

        Returns:
            {"items": [...], "total": int, "skip": int, "limit": int} --
            each item includes `iso42001_clause_refs`, the ISO 42001 clause
            IDs this risk maps to.
        """
        params: Dict[str, Any] = {"limit": limit, "skip": skip}
        for key, value in (
            ("status", status),
            ("severity", severity),
            ("category", category),
            ("model_id", model_id),
        ):
            if value is not None:
                params[key] = value
        return self.client.request("GET", "/api/v1/risk-register", params=params)

    async def alist(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """Async version of list()."""
        params: Dict[str, Any] = {"limit": limit, "skip": skip}
        for key, value in (
            ("status", status),
            ("severity", severity),
            ("category", category),
            ("model_id", model_id),
        ):
            if value is not None:
                params[key] = value
        return await self.client.arequest("GET", "/api/v1/risk-register", params=params)

    def get(self, entry_id: str) -> Dict[str, Any]:
        """Get a single risk register entry by ID."""
        return self.client.request("GET", f"/api/v1/risk-register/{entry_id}")

    async def aget(self, entry_id: str) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/risk-register/{entry_id}")

    def portfolio(self) -> Dict[str, Any]:
        """Get the likelihood x impact heat map and top open risks."""
        return self.client.request("GET", "/api/v1/risk-register/portfolio")

    async def aportfolio(self) -> Dict[str, Any]:
        """Async version of portfolio()."""
        return await self.client.arequest("GET", "/api/v1/risk-register/portfolio")


class GovernanceResource(BaseResource):
    """Governance review board API resource (backed by backend/api/v1/review_boards.py).

    Read-only surface for now: querying board/request/RACI state. Creating
    review requests, casting votes, etc. is an intentionally human/dashboard
    workflow, not exposed here.
    """

    def list_boards(self) -> List[Dict[str, Any]]:
        """List governance review boards for the organization."""
        return self.client.request("GET", "/api/v1/governance/review-boards")

    async def alist_boards(self) -> List[Dict[str, Any]]:
        """Async version of list_boards()."""
        return await self.client.arequest("GET", "/api/v1/governance/review-boards")

    def list_review_requests(
        self, board_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List governance review requests, optionally filtered."""
        params: Dict[str, Any] = {}
        if board_id is not None:
            params["board_id"] = board_id
        if status is not None:
            params["status"] = status
        return self.client.request(
            "GET", "/api/v1/governance/review-boards/requests", params=params
        )

    async def alist_review_requests(
        self, board_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Async version of list_review_requests()."""
        params: Dict[str, Any] = {}
        if board_id is not None:
            params["board_id"] = board_id
        if status is not None:
            params["status"] = status
        return await self.client.arequest(
            "GET", "/api/v1/governance/review-boards/requests", params=params
        )

    def raci_grid(self, board_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the RACI grid: review requests and their Responsible /
        Accountable / Consulted / Informed assignments, optionally scoped to
        one board.
        """
        params = {"board_id": board_id} if board_id is not None else {}
        return self.client.request(
            "GET", "/api/v1/governance/review-boards/raci-grid", params=params
        )

    async def araci_grid(self, board_id: Optional[str] = None) -> Dict[str, Any]:
        """Async version of raci_grid()."""
        params = {"board_id": board_id} if board_id is not None else {}
        return await self.client.arequest(
            "GET", "/api/v1/governance/review-boards/raci-grid", params=params
        )


class LLMResource(BaseResource):
    """LLM monitoring API resource (backed by backend/api/v1/llm_monitoring.py)."""

    def log_call(
        self,
        provider: str,
        model_name: str,
        prompt: str,
        latency_ms: float,
        model_id: Optional[str] = None,
        completion: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Log a single LLM call with real token/cost/latency data.

        Args:
            provider: LLM provider name (openai, anthropic, etc.)
            model_name: Model name (gpt-4o, claude-sonnet-5, etc.)
            prompt: The prompt sent to the LLM
            latency_ms: Request duration in milliseconds
            model_id: Associated WhiteBoxXAI model ID
            completion: The completion returned by the LLM
            prompt_tokens: Number of tokens in the prompt, if known
            completion_tokens: Number of tokens in the completion, if known
            finish_reason: Why the LLM stopped (stop, length, etc.)
            response_metadata: Additional response metadata
            messages: Full message history for chat models
            temperature: Temperature parameter used
            max_tokens: Max tokens parameter used
            status: Request status (success, failure, timeout, rate_limited)
            error_message: Error message if the request failed
            user_id: User who made the request
            session_id: Session identifier
            request_id: Request tracking ID
            environment: Environment (dev, staging, prod)

        Returns:
            {id, message} for the created log entry
        """
        data = _llm_log_payload(
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            latency_ms=latency_ms,
            model_id=model_id,
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            response_metadata=response_metadata,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            status=status,
            error_message=error_message,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            environment=environment,
        )
        return self.client.request("POST", "/api/v1/llm/logs", data=data)

    async def alog_call(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of log_call()."""
        data = _llm_log_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm/logs", data=data)

    def log_calls_batch(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Log multiple LLM calls in one request (max 1000).

        Args:
            logs: List of dicts, each shaped like log_call()'s kwargs
                (provider, model_name, prompt, latency_ms required per entry)
        """
        return self.client.request("POST", "/api/v1/llm/logs/batch", data={"logs": logs})

    async def alog_calls_batch(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Async version of log_calls_batch()."""
        return await self.client.arequest("POST", "/api/v1/llm/logs/batch", data={"logs": logs})

    def get_stats(
        self,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        environment: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated LLM usage statistics (tokens, cost, latency, by_model)."""
        params = _drop_none(
            model_id=model_id,
            provider=provider,
            environment=environment,
            start_time=start_time,
            end_time=end_time,
        )
        return self.client.request("GET", "/api/v1/llm/stats", params=params)

    async def aget_stats(
        self,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        environment: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of get_stats()."""
        params = _drop_none(
            model_id=model_id,
            provider=provider,
            environment=environment,
            start_time=start_time,
            end_time=end_time,
        )
        return await self.client.arequest("GET", "/api/v1/llm/stats", params=params)

    def get_recent(
        self,
        limit: int = 100,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent LLM logs (requires a model_id owned by the caller's org)."""
        params = _drop_none(limit=limit, model_id=model_id, provider=provider)
        return self.client.request("GET", "/api/v1/llm/logs/recent", params=params)

    async def aget_recent(
        self,
        limit: int = 100,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of get_recent()."""
        params = _drop_none(limit=limit, model_id=model_id, provider=provider)
        return await self.client.arequest("GET", "/api/v1/llm/logs/recent", params=params)

    def get_log(self, log_id: str) -> Dict[str, Any]:
        """Get a single LLM log by ID."""
        return self.client.request("GET", f"/api/v1/llm/logs/{log_id}")

    async def aget_log(self, log_id: str) -> Dict[str, Any]:
        """Async version of get_log()."""
        return await self.client.arequest("GET", f"/api/v1/llm/logs/{log_id}")

    def query_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        environment: Optional[str] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_latency_ms: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Query LLM logs with filters and pagination. Returns {logs, total, skip, limit}."""
        data = _llm_query_logs_payload(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            status=status,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
            min_cost=min_cost,
            max_cost=max_cost,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            skip=skip,
            limit=limit,
        )
        return self.client.request("POST", "/api/v1/llm/logs/query", data=data)

    async def aquery_logs(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of query_logs()."""
        data = _llm_query_logs_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm/logs/query", data=data)

    def session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all LLM logs for a specific session."""
        return self.client.request("GET", f"/api/v1/llm/logs/session/{session_id}")

    async def asession_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Async version of session_logs()."""
        return await self.client.arequest("GET", f"/api/v1/llm/logs/session/{session_id}")

    def cost_breakdown(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        include_user: bool = False,
        include_environment: bool = False,
    ) -> Dict[str, Any]:
        """Get detailed cost breakdown (platform-wide aggregate)."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            include_user=include_user,
            include_environment=include_environment,
        )
        return self.client.request("GET", "/api/v1/llm/costs/breakdown", params=params)

    async def acost_breakdown(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        include_user: bool = False,
        include_environment: bool = False,
    ) -> Dict[str, Any]:
        """Async version of cost_breakdown()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            include_user=include_user,
            include_environment=include_environment,
        )
        return await self.client.arequest("GET", "/api/v1/llm/costs/breakdown", params=params)

    def performance(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get LLM latency/throughput performance metrics."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            provider=provider,
        )
        return self.client.request("GET", "/api/v1/llm/performance", params=params)

    async def aperformance(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of performance()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            provider=provider,
        )
        return await self.client.arequest("GET", "/api/v1/llm/performance", params=params)

    def trends_tokens(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get token usage trend over time (granularity: hour/day/week)."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
        )
        return self.client.request("GET", "/api/v1/llm/trends/tokens", params=params)

    async def atrends_tokens(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of trends_tokens()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
        )
        return await self.client.arequest("GET", "/api/v1/llm/trends/tokens", params=params)

    def trends_costs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get cost trend over time (platform-wide; granularity: hour/day/week)."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            provider=provider,
        )
        return self.client.request("GET", "/api/v1/llm/trends/costs", params=params)

    async def atrends_costs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of trends_costs()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            provider=provider,
        )
        return await self.client.arequest("GET", "/api/v1/llm/trends/costs", params=params)

    def usage_stats(
        self,
        start_time: str,
        end_time: str,
        granularity: str = "day",
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get detailed, grouped usage statistics (granularity: hour/day/week/month)."""
        data = _llm_usage_stats_payload(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            provider=provider,
            model_name=model_name,
            model_id=model_id,
            user_id=user_id,
            environment=environment,
        )
        return self.client.request("POST", "/api/v1/llm/usage-stats", data=data)

    async def ausage_stats(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Async version of usage_stats()."""
        data = _llm_usage_stats_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm/usage-stats", data=data)

    def cost_threshold_alert(
        self, period_minutes: int = 60, threshold: float = 100.0
    ) -> Dict[str, Any]:
        """Check whether costs over the recent period exceed a threshold."""
        params = {"period_minutes": period_minutes, "threshold": threshold}
        return self.client.request("GET", "/api/v1/llm/alerts/cost-threshold", params=params)

    async def acost_threshold_alert(
        self, period_minutes: int = 60, threshold: float = 100.0
    ) -> Dict[str, Any]:
        """Async version of cost_threshold_alert()."""
        params = {"period_minutes": period_minutes, "threshold": threshold}
        return await self.client.arequest("GET", "/api/v1/llm/alerts/cost-threshold", params=params)

    def latency_threshold_alert(
        self, period_minutes: int = 60, threshold_ms: float = 5000.0
    ) -> Dict[str, Any]:
        """Check whether average latency over the recent period exceeds a threshold."""
        params = {"period_minutes": period_minutes, "threshold_ms": threshold_ms}
        return self.client.request("GET", "/api/v1/llm/alerts/latency-threshold", params=params)

    async def alatency_threshold_alert(
        self, period_minutes: int = 60, threshold_ms: float = 5000.0
    ) -> Dict[str, Any]:
        """Async version of latency_threshold_alert()."""
        params = {"period_minutes": period_minutes, "threshold_ms": threshold_ms}
        return await self.client.arequest(
            "GET", "/api/v1/llm/alerts/latency-threshold", params=params
        )

    def error_rate_alert(
        self, period_minutes: int = 60, threshold_percent: float = 10.0
    ) -> Dict[str, Any]:
        """Check whether the error rate over the recent period exceeds a threshold."""
        params = {
            "period_minutes": period_minutes,
            "threshold_percent": threshold_percent,
        }
        return self.client.request("GET", "/api/v1/llm/alerts/error-rate", params=params)

    async def aerror_rate_alert(
        self, period_minutes: int = 60, threshold_percent: float = 10.0
    ) -> Dict[str, Any]:
        """Async version of error_rate_alert()."""
        params = {
            "period_minutes": period_minutes,
            "threshold_percent": threshold_percent,
        }
        return await self.client.arequest("GET", "/api/v1/llm/alerts/error-rate", params=params)

    def cleanup_logs(self, days: int = 30) -> Dict[str, Any]:
        """Delete LLM logs older than `days`. Admin-only on the backend."""
        params = {"days": days}
        return self.client.request("DELETE", "/api/v1/llm/logs/cleanup", params=params)

    async def acleanup_logs(self, days: int = 30) -> Dict[str, Any]:
        """Async version of cleanup_logs()."""
        params = {"days": days}
        return await self.client.arequest("DELETE", "/api/v1/llm/logs/cleanup", params=params)


def _llm_query_logs_payload(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    environment: Optional[str] = None,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    min_latency_ms: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
    skip: int = 0,
    limit: int = 100,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMResource.query_logs/aquery_logs."""
    data: Dict[str, Any] = {"skip": skip, "limit": limit}
    data.update(
        _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            status=status,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
            min_cost=min_cost,
            max_cost=max_cost,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
        )
    )
    return data


def _llm_usage_stats_payload(
    start_time: str,
    end_time: str,
    granularity: str = "day",
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMResource.usage_stats/ausage_stats."""
    data: Dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "granularity": granularity,
    }
    data.update(
        _drop_none(
            provider=provider,
            model_name=model_name,
            model_id=model_id,
            user_id=user_id,
            environment=environment,
        )
    )
    return data


def _llm_log_payload(
    provider: str,
    model_name: str,
    prompt: str,
    latency_ms: float,
    model_id: Optional[str] = None,
    completion: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
    response_metadata: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMResource.log_call/alog_call."""
    data: Dict[str, Any] = {
        "provider": provider,
        "model_name": model_name,
        "prompt": prompt,
        "latency_ms": latency_ms,
    }
    data.update(
        _drop_none(
            model_id=model_id,
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            response_metadata=response_metadata,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            status=status,
            error_message=error_message,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            environment=environment,
        )
    )
    return data


def _drop_none(**kwargs: Any) -> Dict[str, Any]:
    """Filter out None-valued kwargs, so optional fields aren't sent at all
    (letting the backend's own schema defaults apply) rather than sent as
    explicit nulls."""
    return {k: v for k, v in kwargs.items() if v is not None}


class RAGResource(BaseResource):
    """RAG pipeline monitoring API resource (backed by backend/api/v1/rag.py)."""

    def log_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        retrieval_method: str,
        query_type: str = "semantic",
        retrieval_params: Optional[Dict[str, Any]] = None,
        retrieval_latency_ms: Optional[float] = None,
        answer: Optional[str] = None,
        answer_latency_ms: Optional[float] = None,
        ground_truth_ids: Optional[List[str]] = None,
        llm_log_id: Optional[str] = None,
        model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        environment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a retrieval operation, with real quality metrics computed
        server-side when ground_truth_ids is supplied.

        Args:
            query: Query text
            results: List of dicts, each with document_id/rank/score (and
                optionally document_content/document_metadata/score_type/
                is_ground_truth)
            top_k: Number of results retrieved
            retrieval_method: Retrieval method used (e.g. "vector", "hybrid")
            query_type: Type of query (default "semantic")
            retrieval_params: Additional retrieval parameters
            retrieval_latency_ms: Retrieval latency in milliseconds
            answer: Generated answer, if a generation step was included
            answer_latency_ms: Answer generation latency in milliseconds
            ground_truth_ids: Relevant document IDs, to compute precision/
                recall/MRR/NDCG at log time
            llm_log_id: Associated LLM log ID
            model_id: Associated WhiteBoxXAI model ID
            user_id: User who made the request
            session_id: Session identifier
            environment: Environment (dev, staging, prod)
            metadata: Additional metadata

        Returns:
            The created retrieval log, including computed metrics
        """
        data = _rag_retrieval_payload(
            query=query,
            results=results,
            top_k=top_k,
            retrieval_method=retrieval_method,
            query_type=query_type,
            retrieval_params=retrieval_params,
            retrieval_latency_ms=retrieval_latency_ms,
            answer=answer,
            answer_latency_ms=answer_latency_ms,
            ground_truth_ids=ground_truth_ids,
            llm_log_id=llm_log_id,
            model_id=model_id,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
            metadata=metadata,
        )
        return self.client.request("POST", "/api/v1/rag/retrievals", data=data)

    async def alog_retrieval(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of log_retrieval()."""
        data = _rag_retrieval_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/rag/retrievals", data=data)

    def create_evaluation(
        self,
        name: str,
        queries: List[str],
        description: Optional[str] = None,
        evaluation_type: str = "manual",
        model_id: Optional[str] = None,
        system_version: Optional[str] = None,
        ground_truth: Optional[Dict[str, List[str]]] = None,
        environment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a RAG evaluation, aggregating already-logged retrievals for
        each query in `queries` (optionally recomputing precision/recall/
        MRR/NDCG against `ground_truth`). Queries with no matching logged
        retrieval are reported in the response's
        detailed_results.queries_without_data, not silently scored as zero.

        Args:
            name: Evaluation name
            queries: Test queries to evaluate
            description: Evaluation description
            evaluation_type: Type of evaluation (default "manual")
            model_id: Associated WhiteBoxXAI model ID
            system_version: System version under evaluation
            ground_truth: Relevant document IDs per query, for metric
                recomputation
            environment: Environment (dev, staging, prod)
            metadata: Additional metadata

        Returns:
            The created evaluation, including aggregated metrics
        """
        data = _rag_evaluation_payload(
            name=name,
            queries=queries,
            description=description,
            evaluation_type=evaluation_type,
            model_id=model_id,
            system_version=system_version,
            ground_truth=ground_truth,
            environment=environment,
            metadata=metadata,
        )
        return self.client.request("POST", "/api/v1/rag/evaluations", data=data)

    async def acreate_evaluation(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of create_evaluation()."""
        data = _rag_evaluation_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/rag/evaluations", data=data)

    def get_stats(
        self,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated RAG statistics (precision/recall/MRR/NDCG/relevance/latency)."""
        params = _drop_none(
            model_id=model_id,
            environment=environment,
            start_time=start_time,
            end_time=end_time,
        )
        return self.client.request("GET", "/api/v1/rag/stats", params=params)

    async def aget_stats(
        self,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of get_stats()."""
        params = _drop_none(
            model_id=model_id,
            environment=environment,
            start_time=start_time,
            end_time=end_time,
        )
        return await self.client.arequest("GET", "/api/v1/rag/stats", params=params)

    def list_evaluations(
        self,
        model_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List RAG evaluations with optional filters."""
        params = _drop_none(
            model_id=model_id,
            evaluation_type=evaluation_type,
            environment=environment,
            limit=limit,
            offset=offset,
        )
        return self.client.request("GET", "/api/v1/rag/evaluations", params=params)

    async def alist_evaluations(
        self,
        model_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Async version of list_evaluations()."""
        params = _drop_none(
            model_id=model_id,
            evaluation_type=evaluation_type,
            environment=environment,
            limit=limit,
            offset=offset,
        )
        return await self.client.arequest("GET", "/api/v1/rag/evaluations", params=params)

    def get_retrieval(self, log_id: str) -> Dict[str, Any]:
        """Get a specific retrieval log by ID."""
        return self.client.request("GET", f"/api/v1/rag/retrievals/{log_id}")

    async def aget_retrieval(self, log_id: str) -> Dict[str, Any]:
        """Async version of get_retrieval()."""
        return await self.client.arequest("GET", f"/api/v1/rag/retrievals/{log_id}")

    def list_retrievals(
        self,
        model_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retrieval_method: Optional[str] = None,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query retrieval logs. model_id is required (owned by the caller's
        organization) until organization-wide RAG log scoping ships."""
        params = {"model_id": model_id, "limit": limit, "offset": offset}
        params.update(
            _drop_none(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                session_id=session_id,
                retrieval_method=retrieval_method,
                environment=environment,
            )
        )
        return self.client.request("GET", "/api/v1/rag/retrievals", params=params)

    async def alist_retrievals(
        self,
        model_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retrieval_method: Optional[str] = None,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Async version of list_retrievals()."""
        params = {"model_id": model_id, "limit": limit, "offset": offset}
        params.update(
            _drop_none(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                session_id=session_id,
                retrieval_method=retrieval_method,
                environment=environment,
            )
        )
        return await self.client.arequest("GET", "/api/v1/rag/retrievals", params=params)

    def trends(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get RAG quality metrics trend over time (granularity: hour/day/week)."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
            environment=environment,
        )
        return self.client.request("GET", "/api/v1/rag/trends", params=params)

    async def atrends(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of trends()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
            environment=environment,
        )
        return await self.client.arequest("GET", "/api/v1/rag/trends", params=params)

    def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get a specific RAG evaluation by ID."""
        return self.client.request("GET", f"/api/v1/rag/evaluations/{evaluation_id}")

    async def aget_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Async version of get_evaluation()."""
        return await self.client.arequest("GET", f"/api/v1/rag/evaluations/{evaluation_id}")

    def metrics_precision(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Optional[float]]:
        """Get Precision@K metrics for the given K values (default [1, 3, 5, 10])."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            environment=environment,
            k_values=k_values,
        )
        return self.client.request("GET", "/api/v1/rag/metrics/precision", params=params)

    async def ametrics_precision(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Optional[float]]:
        """Async version of metrics_precision()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            environment=environment,
            k_values=k_values,
        )
        return await self.client.arequest("GET", "/api/v1/rag/metrics/precision", params=params)

    def metrics_relevance(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get average relevance metrics (context/answer/faithfulness)."""
        params = _drop_none(start_time=start_time, end_time=end_time, model_id=model_id)
        return self.client.request("GET", "/api/v1/rag/metrics/relevance", params=params)

    async def ametrics_relevance(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Async version of metrics_relevance()."""
        params = _drop_none(start_time=start_time, end_time=end_time, model_id=model_id)
        return await self.client.arequest("GET", "/api/v1/rag/metrics/relevance", params=params)


def _rag_retrieval_payload(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int,
    retrieval_method: str,
    query_type: str = "semantic",
    retrieval_params: Optional[Dict[str, Any]] = None,
    retrieval_latency_ms: Optional[float] = None,
    answer: Optional[str] = None,
    answer_latency_ms: Optional[float] = None,
    ground_truth_ids: Optional[List[str]] = None,
    llm_log_id: Optional[str] = None,
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    environment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for RAGResource.log_retrieval/alog_retrieval."""
    data: Dict[str, Any] = {
        "query": query,
        "results": results,
        "top_k": top_k,
        "retrieval_method": retrieval_method,
        "query_type": query_type,
    }
    data.update(
        _drop_none(
            retrieval_params=retrieval_params,
            retrieval_latency_ms=retrieval_latency_ms,
            answer=answer,
            answer_latency_ms=answer_latency_ms,
            ground_truth_ids=ground_truth_ids,
            llm_log_id=llm_log_id,
            model_id=model_id,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
            metadata=metadata,
        )
    )
    return data


def _rag_evaluation_payload(
    name: str,
    queries: List[str],
    description: Optional[str] = None,
    evaluation_type: str = "manual",
    model_id: Optional[str] = None,
    system_version: Optional[str] = None,
    ground_truth: Optional[Dict[str, List[str]]] = None,
    environment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for RAGResource.create_evaluation/acreate_evaluation."""
    data: Dict[str, Any] = {
        "name": name,
        "queries": queries,
        "evaluation_type": evaluation_type,
    }
    data.update(
        _drop_none(
            description=description,
            model_id=model_id,
            system_version=system_version,
            ground_truth=ground_truth,
            environment=environment,
            metadata=metadata,
        )
    )
    return data


class SafetyResource(BaseResource):
    """Content safety API resource (backed by backend/api/v1/safety.py)."""

    def analyze(
        self,
        content: str,
        content_type: str = "text",
        check_toxicity: bool = True,
        check_pii: bool = True,
        check_harmful: bool = True,
        language: str = "en",
        llm_log_id: Optional[str] = None,
        model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze content for toxicity, PII, and harmful content.

        Args:
            content: Content to analyze
            content_type: Type of content (prompt, completion, message)
            check_toxicity: Whether to check for toxic content
            check_pii: Whether to check for PII
            check_harmful: Whether to check for harmful content
            language: Content language
            llm_log_id: Associated LLM log ID
            model_id: Associated WhiteBoxXAI model ID
            user_id: User ID
            session_id: Session ID
            environment: Environment (dev, staging, prod)

        Returns:
            Full safety analysis: toxicity_scores, pii_results,
            harmful_content, safety_status, flagged_categories, etc.
        """
        data = _safety_analyze_payload(
            content=content,
            content_type=content_type,
            check_toxicity=check_toxicity,
            check_pii=check_pii,
            check_harmful=check_harmful,
            language=language,
            llm_log_id=llm_log_id,
            model_id=model_id,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
        )
        return self.client.request("POST", "/api/v1/safety/analyze", data=data)

    async def aanalyze(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of analyze()."""
        data = _safety_analyze_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/safety/analyze", data=data)

    def analyze_batch(
        self,
        contents: List[str],
        content_type: str = "text",
        check_toxicity: bool = True,
        check_pii: bool = True,
        check_harmful: bool = True,
        language: str = "en",
        model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze multiple content items in one request (max 100)."""
        data = _safety_analyze_batch_payload(
            contents=contents,
            content_type=content_type,
            check_toxicity=check_toxicity,
            check_pii=check_pii,
            check_harmful=check_harmful,
            language=language,
            model_id=model_id,
            user_id=user_id,
            environment=environment,
        )
        return self.client.request("POST", "/api/v1/safety/analyze/batch", data=data)

    async def aanalyze_batch(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Async version of analyze_batch()."""
        data = _safety_analyze_batch_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/safety/analyze/batch", data=data)

    def get_scores(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        safety_status: Optional[str] = None,
        contains_pii: Optional[bool] = None,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query safety analysis scores with filters."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            safety_status=safety_status,
            contains_pii=contains_pii,
            environment=environment,
            limit=limit,
            offset=offset,
        )
        return self.client.request("GET", "/api/v1/safety/scores", params=params)

    async def aget_scores(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        safety_status: Optional[str] = None,
        contains_pii: Optional[bool] = None,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Async version of get_scores()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            safety_status=safety_status,
            contains_pii=contains_pii,
            environment=environment,
            limit=limit,
            offset=offset,
        )
        return await self.client.arequest("GET", "/api/v1/safety/scores", params=params)

    def get_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated safety statistics (per-category toxicity averages,
        flagged categories, PII types)."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            environment=environment,
        )
        return self.client.request("GET", "/api/v1/safety/stats", params=params)

    async def aget_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of get_stats()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            environment=environment,
        )
        return await self.client.arequest("GET", "/api/v1/safety/stats", params=params)

    def get_score(self, score_id: str) -> Dict[str, Any]:
        """Get a specific safety score by ID."""
        return self.client.request("GET", f"/api/v1/safety/scores/{score_id}")

    async def aget_score(self, score_id: str) -> Dict[str, Any]:
        """Async version of get_score()."""
        return await self.client.arequest("GET", f"/api/v1/safety/scores/{score_id}")

    def trends(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get content-safety trend over time (per-category toxicity averages,
        PII detections; granularity: hour/day/week)."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
            environment=environment,
        )
        return self.client.request("GET", "/api/v1/safety/trends", params=params)

    async def atrends(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of trends()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
            environment=environment,
        )
        return await self.client.arequest("GET", "/api/v1/safety/trends", params=params)

    def create_threshold(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """Create a safety threshold configuration.

        Args:
            name: Threshold configuration name
            **kwargs: Any other SafetyThresholdCreate field, e.g. model_id,
                toxicity_threshold, severe_toxicity_threshold,
                identity_attack_threshold, insult_threshold,
                profanity_threshold, threat_threshold,
                sexually_explicit_threshold, block_pii, allowed_pii_types,
                max_pii_count, action_on_violation ("warning"/"block"/"alert"),
                send_alerts, alert_channels
        """
        data = {"name": name, **kwargs}
        return self.client.request("POST", "/api/v1/safety/thresholds", data=data)

    async def acreate_threshold(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """Async version of create_threshold()."""
        data = {"name": name, **kwargs}
        return await self.client.arequest("POST", "/api/v1/safety/thresholds", data=data)

    def list_thresholds(
        self, model_id: Optional[str] = None, is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """List safety thresholds for the caller's organization."""
        params = _drop_none(model_id=model_id, is_active=is_active)
        return self.client.request("GET", "/api/v1/safety/thresholds", params=params)

    async def alist_thresholds(
        self, model_id: Optional[str] = None, is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Async version of list_thresholds()."""
        params = _drop_none(model_id=model_id, is_active=is_active)
        return await self.client.arequest("GET", "/api/v1/safety/thresholds", params=params)

    def get_threshold(self, threshold_id: str) -> Dict[str, Any]:
        """Get a specific safety threshold by ID."""
        return self.client.request("GET", f"/api/v1/safety/thresholds/{threshold_id}")

    async def aget_threshold(self, threshold_id: str) -> Dict[str, Any]:
        """Async version of get_threshold()."""
        return await self.client.arequest("GET", f"/api/v1/safety/thresholds/{threshold_id}")

    def update_threshold(self, threshold_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Update a safety threshold configuration.

        Args:
            threshold_id: Threshold to update
            **kwargs: Any SafetyThresholdUpdate field to change, e.g. name,
                description, is_active, toxicity_threshold, block_pii,
                action_on_violation, send_alerts (see create_threshold() for
                the full field list)
        """
        return self.client.request(
            "PATCH", f"/api/v1/safety/thresholds/{threshold_id}", data=kwargs
        )

    async def aupdate_threshold(self, threshold_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Async version of update_threshold()."""
        return await self.client.arequest(
            "PATCH", f"/api/v1/safety/thresholds/{threshold_id}", data=kwargs
        )

    def delete_threshold(self, threshold_id: str) -> Dict[str, Any]:
        """Delete a safety threshold configuration."""
        return self.client.request("DELETE", f"/api/v1/safety/thresholds/{threshold_id}")

    async def adelete_threshold(self, threshold_id: str) -> Dict[str, Any]:
        """Async version of delete_threshold()."""
        return await self.client.arequest("DELETE", f"/api/v1/safety/thresholds/{threshold_id}")


def _safety_analyze_payload(
    content: str,
    content_type: str = "text",
    check_toxicity: bool = True,
    check_pii: bool = True,
    check_harmful: bool = True,
    language: str = "en",
    llm_log_id: Optional[str] = None,
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for SafetyResource.analyze/aanalyze."""
    data: Dict[str, Any] = {
        "content": content,
        "content_type": content_type,
        "check_toxicity": check_toxicity,
        "check_pii": check_pii,
        "check_harmful": check_harmful,
        "language": language,
    }
    data.update(
        _drop_none(
            llm_log_id=llm_log_id,
            model_id=model_id,
            user_id=user_id,
            session_id=session_id,
            environment=environment,
        )
    )
    return data


def _safety_analyze_batch_payload(
    contents: List[str],
    content_type: str = "text",
    check_toxicity: bool = True,
    check_pii: bool = True,
    check_harmful: bool = True,
    language: str = "en",
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for SafetyResource.analyze_batch/aanalyze_batch."""
    data: Dict[str, Any] = {
        "contents": contents,
        "content_type": content_type,
        "check_toxicity": check_toxicity,
        "check_pii": check_pii,
        "check_harmful": check_harmful,
        "language": language,
    }
    data.update(_drop_none(model_id=model_id, user_id=user_id, environment=environment))
    return data


class LLMXAIResource(BaseResource):
    """LLM explainability API resource (backed by backend/api/v1/llm_xai.py):
    attention/token-importance/sensitivity/counterfactual analysis and
    prompt debugging for LLM calls."""

    def attention(
        self,
        prompt: str,
        completion: Optional[str] = None,
        model_id: Optional[str] = None,
        layers: Optional[List[int]] = None,
        aggregate_heads: bool = True,
        attention_weights: Optional[List[Any]] = None,
        top_k: int = 10,
        llm_log_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze transformer attention patterns for a prompt/completion."""
        data = _llm_xai_attention_payload(
            prompt=prompt,
            completion=completion,
            model_id=model_id,
            layers=layers,
            aggregate_heads=aggregate_heads,
            attention_weights=attention_weights,
            top_k=top_k,
            llm_log_id=llm_log_id,
        )
        return self.client.request("POST", "/api/v1/llm-xai/attention", data=data)

    async def aattention(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of attention()."""
        data = _llm_xai_attention_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/attention", data=data)

    def token_importance(
        self,
        prompt: str,
        completion: str,
        model_id: Optional[str] = None,
        method: str = "perturbation",
        top_k: int = 10,
        llm_log_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate token-level importance scores for a prompt/completion pair."""
        data = _llm_xai_token_importance_payload(
            prompt=prompt,
            completion=completion,
            model_id=model_id,
            method=method,
            top_k=top_k,
            llm_log_id=llm_log_id,
        )
        return self.client.request("POST", "/api/v1/llm-xai/token-importance", data=data)

    async def atoken_importance(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of token_importance()."""
        data = _llm_xai_token_importance_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/token-importance", data=data)

    def prompt_sensitivity(
        self,
        prompt: str,
        baseline_output: str,
        model_id: Optional[str] = None,
        num_perturbations: int = 10,
        perturbation_type: str = "mixed",
        llm_log_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test prompt robustness by perturbing it and measuring output stability."""
        data = _llm_xai_sensitivity_payload(
            prompt=prompt,
            baseline_output=baseline_output,
            model_id=model_id,
            num_perturbations=num_perturbations,
            perturbation_type=perturbation_type,
            llm_log_id=llm_log_id,
        )
        return self.client.request("POST", "/api/v1/llm-xai/sensitivity", data=data)

    async def aprompt_sensitivity(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of prompt_sensitivity()."""
        data = _llm_xai_sensitivity_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/sensitivity", data=data)

    def counterfactuals(
        self,
        prompt: str,
        original_output: str,
        target_change: str = "any",
        model_id: Optional[str] = None,
        max_edits: int = 3,
        num_candidates: int = 10,
        search_strategy: str = "greedy",
        llm_log_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate minimal prompt edits that change the model's output."""
        data = _llm_xai_counterfactuals_payload(
            prompt=prompt,
            original_output=original_output,
            target_change=target_change,
            model_id=model_id,
            max_edits=max_edits,
            num_candidates=num_candidates,
            search_strategy=search_strategy,
            llm_log_id=llm_log_id,
        )
        return self.client.request("POST", "/api/v1/llm-xai/counterfactuals", data=data)

    async def acounterfactuals(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of counterfactuals()."""
        data = _llm_xai_counterfactuals_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/counterfactuals", data=data)

    def debug_prompt(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        check_clarity: bool = True,
        check_specificity: bool = True,
        check_completeness: bool = True,
        check_consistency: bool = True,
        suggest_improvements: bool = True,
        llm_log_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess prompt quality and get improvement recommendations."""
        data = _llm_xai_debug_prompt_payload(
            prompt=prompt,
            model_id=model_id,
            check_clarity=check_clarity,
            check_specificity=check_specificity,
            check_completeness=check_completeness,
            check_consistency=check_consistency,
            suggest_improvements=suggest_improvements,
            llm_log_id=llm_log_id,
        )
        return self.client.request("POST", "/api/v1/llm-xai/debug-prompt", data=data)

    async def adebug_prompt(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of debug_prompt()."""
        data = _llm_xai_debug_prompt_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/debug-prompt", data=data)

    def get_explanation(self, explanation_id: str) -> Dict[str, Any]:
        """Get a specific attention/token-importance/sensitivity/counterfactual
        explanation by ID."""
        return self.client.request("GET", f"/api/v1/llm-xai/explanations/{explanation_id}")

    async def aget_explanation(self, explanation_id: str) -> Dict[str, Any]:
        """Async version of get_explanation()."""
        return await self.client.arequest("GET", f"/api/v1/llm-xai/explanations/{explanation_id}")

    def get_prompt_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Get a specific prompt debug analysis by ID."""
        return self.client.request("GET", f"/api/v1/llm-xai/prompt-analyses/{analysis_id}")

    async def aget_prompt_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Async version of get_prompt_analysis()."""
        return await self.client.arequest("GET", f"/api/v1/llm-xai/prompt-analyses/{analysis_id}")

    def list_explanations(
        self,
        llm_log_id: Optional[str] = None,
        explanation_type: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List explanations, optionally filtered by LLM log, type, or model."""
        params = _drop_none(
            llm_log_id=llm_log_id,
            explanation_type=explanation_type,
            model_id=model_id,
            limit=limit,
        )
        return self.client.request("GET", "/api/v1/llm-xai/explanations", params=params)

    async def alist_explanations(
        self,
        llm_log_id: Optional[str] = None,
        explanation_type: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Async version of list_explanations()."""
        params = _drop_none(
            llm_log_id=llm_log_id,
            explanation_type=explanation_type,
            model_id=model_id,
            limit=limit,
        )
        return await self.client.arequest("GET", "/api/v1/llm-xai/explanations", params=params)

    def list_prompt_analyses(
        self,
        prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List prompt analyses, optionally filtered by similar prompt text,
        model, or minimum quality score."""
        params = _drop_none(
            prompt=prompt,
            model_id=model_id,
            min_quality_score=min_quality_score,
            limit=limit,
        )
        return self.client.request("GET", "/api/v1/llm-xai/prompt-analyses", params=params)

    async def alist_prompt_analyses(
        self,
        prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Async version of list_prompt_analyses()."""
        params = _drop_none(
            prompt=prompt,
            model_id=model_id,
            min_quality_score=min_quality_score,
            limit=limit,
        )
        return await self.client.arequest("GET", "/api/v1/llm-xai/prompt-analyses", params=params)

    def stats(self, model_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Get aggregated LLM-XAI usage statistics."""
        params = _drop_none(model_id=model_id, days=days)
        return self.client.request("GET", "/api/v1/llm-xai/stats", params=params)

    async def astats(self, model_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Async version of stats()."""
        params = _drop_none(model_id=model_id, days=days)
        return await self.client.arequest("GET", "/api/v1/llm-xai/stats", params=params)

    def visualize_attention(self, explanation_id: str) -> Dict[str, Any]:
        """Get attention-heatmap visualization data for an explanation."""
        return self.client.request("GET", f"/api/v1/llm-xai/attention/visualize/{explanation_id}")

    async def avisualize_attention(self, explanation_id: str) -> Dict[str, Any]:
        """Async version of visualize_attention()."""
        return await self.client.arequest(
            "GET", f"/api/v1/llm-xai/attention/visualize/{explanation_id}"
        )

    def visualize_token_importance(self, explanation_id: str, top_k: int = 20) -> Dict[str, Any]:
        """Get token-importance bar-chart visualization data for an explanation."""
        params = _drop_none(top_k=top_k)
        return self.client.request(
            "GET",
            f"/api/v1/llm-xai/token-importance/visualize/{explanation_id}",
            params=params,
        )

    async def avisualize_token_importance(
        self, explanation_id: str, top_k: int = 20
    ) -> Dict[str, Any]:
        """Async version of visualize_token_importance()."""
        params = _drop_none(top_k=top_k)
        return await self.client.arequest(
            "GET",
            f"/api/v1/llm-xai/token-importance/visualize/{explanation_id}",
            params=params,
        )

    def batch_analyze(
        self,
        llm_log_id: str,
        include_attention: bool = True,
        include_token_importance: bool = True,
        include_sensitivity: bool = False,
        include_prompt_debug: bool = True,
    ) -> Dict[str, Any]:
        """Run multiple explanation analyses on a single LLM log in one call."""
        params = {
            "llm_log_id": llm_log_id,
            "include_attention": include_attention,
            "include_token_importance": include_token_importance,
            "include_sensitivity": include_sensitivity,
            "include_prompt_debug": include_prompt_debug,
        }
        return self.client.request("POST", "/api/v1/llm-xai/batch-analyze", params=params)

    async def abatch_analyze(
        self,
        llm_log_id: str,
        include_attention: bool = True,
        include_token_importance: bool = True,
        include_sensitivity: bool = False,
        include_prompt_debug: bool = True,
    ) -> Dict[str, Any]:
        """Async version of batch_analyze()."""
        params = {
            "llm_log_id": llm_log_id,
            "include_attention": include_attention,
            "include_token_importance": include_token_importance,
            "include_sensitivity": include_sensitivity,
            "include_prompt_debug": include_prompt_debug,
        }
        return await self.client.arequest("POST", "/api/v1/llm-xai/batch-analyze", params=params)


def _llm_xai_attention_payload(
    prompt: str,
    completion: Optional[str] = None,
    model_id: Optional[str] = None,
    layers: Optional[List[int]] = None,
    aggregate_heads: bool = True,
    attention_weights: Optional[List[Any]] = None,
    top_k: int = 10,
    llm_log_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.attention/aattention."""
    data: Dict[str, Any] = {
        "prompt": prompt,
        "aggregate_heads": aggregate_heads,
        "top_k": top_k,
    }
    data.update(
        _drop_none(
            completion=completion,
            model_id=model_id,
            layers=layers,
            attention_weights=attention_weights,
            llm_log_id=llm_log_id,
        )
    )
    return data


def _llm_xai_token_importance_payload(
    prompt: str,
    completion: str,
    model_id: Optional[str] = None,
    method: str = "perturbation",
    top_k: int = 10,
    llm_log_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.token_importance/atoken_importance."""
    data: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "method": method,
        "top_k": top_k,
    }
    data.update(_drop_none(model_id=model_id, llm_log_id=llm_log_id))
    return data


def _llm_xai_sensitivity_payload(
    prompt: str,
    baseline_output: str,
    model_id: Optional[str] = None,
    num_perturbations: int = 10,
    perturbation_type: str = "mixed",
    llm_log_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.prompt_sensitivity/aprompt_sensitivity."""
    data: Dict[str, Any] = {
        "prompt": prompt,
        "baseline_output": baseline_output,
        "num_perturbations": num_perturbations,
        "perturbation_type": perturbation_type,
    }
    data.update(_drop_none(model_id=model_id, llm_log_id=llm_log_id))
    return data


def _llm_xai_counterfactuals_payload(
    prompt: str,
    original_output: str,
    target_change: str = "any",
    model_id: Optional[str] = None,
    max_edits: int = 3,
    num_candidates: int = 10,
    search_strategy: str = "greedy",
    llm_log_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.counterfactuals/acounterfactuals."""
    data: Dict[str, Any] = {
        "prompt": prompt,
        "original_output": original_output,
        "target_change": target_change,
        "max_edits": max_edits,
        "num_candidates": num_candidates,
        "search_strategy": search_strategy,
    }
    data.update(_drop_none(model_id=model_id, llm_log_id=llm_log_id))
    return data


def _llm_xai_debug_prompt_payload(
    prompt: str,
    model_id: Optional[str] = None,
    check_clarity: bool = True,
    check_specificity: bool = True,
    check_completeness: bool = True,
    check_consistency: bool = True,
    suggest_improvements: bool = True,
    llm_log_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.debug_prompt/adebug_prompt."""
    data: Dict[str, Any] = {
        "prompt": prompt,
        "check_clarity": check_clarity,
        "check_specificity": check_specificity,
        "check_completeness": check_completeness,
        "check_consistency": check_consistency,
        "suggest_improvements": suggest_improvements,
    }
    data.update(_drop_none(model_id=model_id, llm_log_id=llm_log_id))
    return data


# The backend's cost_breakdown/bottlenecks/timeline endpoints block
# server-side up to 30s via Celery `.get(timeout=30)`
# (backend/services/agent_workflow_service.py:356-371) -- these three tools
# need a client timeout strictly greater than that ceiling. 40s (not just
# 31s) leaves headroom for network latency on top of the worker's own budget.
_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S = 40


class AgentWorkflowsResource(BaseResource):
    """Multi-agent workflow tracking API resource (backed by
    backend/api/v1/agent_workflows.py): CrewAI/LangChain/AutoGen/n8n workflow,
    agent, execution, interaction, and task tracking, plus cost/bottleneck/
    timeline analytics."""

    def create_and_start(
        self,
        name: str,
        framework: str,
        meta_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new multi-agent workflow.

        framework: "crewai", "langchain", "autogen", "n8n", or "custom".
        Returns the created workflow (including its id) for subsequent
        agent/task registration.
        """
        data = _drop_none(name=name, framework=framework, meta_data=meta_data, tags=tags)
        return self.client.request("POST", "/api/v1/workflows/multi-agent/start", data=data)

    async def acreate_and_start(
        self,
        name: str,
        framework: str,
        meta_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async version of create_and_start()."""
        data = _drop_none(name=name, framework=framework, meta_data=meta_data, tags=tags)
        return await self.client.arequest("POST", "/api/v1/workflows/multi-agent/start", data=data)

    def start(self, workflow_id: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mark a workflow as running. Call after registering agents/tasks."""
        data = _drop_none(inputs=inputs)
        return self.client.request(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/start", data=data
        )

    async def astart(
        self, workflow_id: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async version of start()."""
        data = _drop_none(inputs=inputs)
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/start", data=data
        )

    def complete(
        self,
        workflow_id: str,
        status: str,
        outputs: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        trigger_analytics: bool = True,
    ) -> Dict[str, Any]:
        """Mark a workflow completed/failed/cancelled and optionally trigger
        async analytics (metrics, cost, bottlenecks, timeline)."""
        data = {"status": status}
        data.update(_drop_none(outputs=outputs, error_message=error_message))
        return self.client.request(
            "POST",
            f"/api/v1/workflows/multi-agent/{workflow_id}/complete",
            data=data,
            params={"trigger_analytics": trigger_analytics},
        )

    async def acomplete(
        self,
        workflow_id: str,
        status: str,
        outputs: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        trigger_analytics: bool = True,
    ) -> Dict[str, Any]:
        """Async version of complete()."""
        data = {"status": status}
        data.update(_drop_none(outputs=outputs, error_message=error_message))
        return await self.client.arequest(
            "POST",
            f"/api/v1/workflows/multi-agent/{workflow_id}/complete",
            data=data,
            params={"trigger_analytics": trigger_analytics},
        )

    def list(
        self,
        status: Optional[str] = None,
        framework: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List workflows for the organization. Returns {workflows, total, skip, limit}."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(status=status, framework=framework))
        return self.client.request("GET", "/api/v1/workflows/multi-agent", params=params)

    async def alist(
        self,
        status: Optional[str] = None,
        framework: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Async version of list()."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(status=status, framework=framework))
        return await self.client.arequest("GET", "/api/v1/workflows/multi-agent", params=params)

    def get(self, workflow_id: str) -> Dict[str, Any]:
        """Get a workflow by ID."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}")

    async def aget(self, workflow_id: str) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/workflows/multi-agent/{workflow_id}")

    def delete(self, workflow_id: str) -> Dict[str, Any]:
        """Delete a workflow and all related agents/executions/tasks/interactions."""
        return self.client.request("DELETE", f"/api/v1/workflows/multi-agent/{workflow_id}")

    async def adelete(self, workflow_id: str) -> Dict[str, Any]:
        """Async version of delete()."""
        return await self.client.arequest("DELETE", f"/api/v1/workflows/multi-agent/{workflow_id}")

    def register_agent(
        self,
        workflow_id: str,
        name: str,
        role: Optional[str] = None,
        agent_type: Optional[str] = None,
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
        tools: Optional[List[str]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register an agent in a workflow. Call once per agent before
        starting execution."""
        data = _agent_register_payload(
            name=name,
            role=role,
            agent_type=agent_type,
            model_name=model_name,
            llm_provider=llm_provider,
            llm_config=llm_config,
            system_prompt=system_prompt,
            goal=goal,
            backstory=backstory,
            tools=tools,
            capabilities=capabilities,
            meta_data=meta_data,
        )
        return self.client.request(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/agents", data=data
        )

    async def aregister_agent(self, workflow_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of register_agent()."""
        data = _agent_register_payload(*args, **kwargs)
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/agents", data=data
        )

    def list_agents(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all agents registered in a workflow."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}/agents")

    async def alist_agents(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Async version of list_agents()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/agents"
        )

    def create_execution(
        self,
        workflow_id: str,
        agent_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        parent_execution_id: Optional[str] = None,
        execution_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Log that an agent started executing. Set parent_execution_id for
        hierarchical/delegated executions."""
        data = {"agent_id": agent_id}
        data.update(
            _drop_none(
                inputs=inputs,
                parent_execution_id=parent_execution_id,
                execution_order=execution_order,
            )
        )
        return self.client.request(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/executions", data=data
        )

    async def acreate_execution(
        self,
        workflow_id: str,
        agent_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        parent_execution_id: Optional[str] = None,
        execution_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async version of create_execution()."""
        data = {"agent_id": agent_id}
        data.update(
            _drop_none(
                inputs=inputs,
                parent_execution_id=parent_execution_id,
                execution_order=execution_order,
            )
        )
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/executions", data=data
        )

    def log_interaction(
        self,
        workflow_id: str,
        interaction_type: str,
        from_agent_id: Optional[str] = None,
        to_agent_id: Optional[str] = None,
        from_execution_id: Optional[str] = None,
        to_execution_id: Optional[str] = None,
        message: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log agent-to-agent communication (delegation/handoff/query/feedback/broadcast)."""
        data = {"interaction_type": interaction_type}
        data.update(
            _drop_none(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                from_execution_id=from_execution_id,
                to_execution_id=to_execution_id,
                message=message,
                meta_data=meta_data,
            )
        )
        return self.client.request(
            "POST",
            f"/api/v1/workflows/multi-agent/{workflow_id}/interactions",
            data=data,
        )

    async def alog_interaction(
        self,
        workflow_id: str,
        interaction_type: str,
        from_agent_id: Optional[str] = None,
        to_agent_id: Optional[str] = None,
        from_execution_id: Optional[str] = None,
        to_execution_id: Optional[str] = None,
        message: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of log_interaction()."""
        data = {"interaction_type": interaction_type}
        data.update(
            _drop_none(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                from_execution_id=from_execution_id,
                to_execution_id=to_execution_id,
                message=message,
                meta_data=meta_data,
            )
        )
        return await self.client.arequest(
            "POST",
            f"/api/v1/workflows/multi-agent/{workflow_id}/interactions",
            data=data,
        )

    def list_interactions(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all agent-to-agent interactions in a workflow, chronologically."""
        return self.client.request(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/interactions"
        )

    async def alist_interactions(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Async version of list_interactions()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/interactions"
        )

    def create_task(
        self,
        workflow_id: str,
        task_name: str,
        description: Optional[str] = None,
        task_type: Optional[str] = None,
        expected_output: Optional[str] = None,
        agent_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        priority: int = 0,
        input_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a workflow task. task_type: research/write/analyze/delegate/review."""
        data = _agent_task_create_payload(
            task_name=task_name,
            description=description,
            task_type=task_type,
            expected_output=expected_output,
            agent_id=agent_id,
            parent_task_id=parent_task_id,
            priority=priority,
            input_data=input_data,
            context=context,
        )
        return self.client.request(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks", data=data
        )

    async def acreate_task(self, workflow_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of create_task()."""
        data = _agent_task_create_payload(*args, **kwargs)
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks", data=data
        )

    def update_task_status(
        self,
        task_id: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a task's status. status: pending/running/completed/failed/skipped."""
        data = {"status": status}
        data.update(_drop_none(output_data=output_data, error_message=error_message))
        return self.client.request(
            "PATCH", f"/api/v1/workflows/multi-agent/tasks/{task_id}", data=data
        )

    async def aupdate_task_status(
        self,
        task_id: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of update_task_status()."""
        data = {"status": status}
        data.update(_drop_none(output_data=output_data, error_message=error_message))
        return await self.client.arequest(
            "PATCH", f"/api/v1/workflows/multi-agent/tasks/{task_id}", data=data
        )

    def list_tasks(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all tasks in a workflow, with status, assignments, and results."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks")

    async def alist_tasks(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Async version of list_tasks()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks"
        )

    def analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow analytics (tokens, costs, execution counts, durations).
        Calculated asynchronously after workflow completion."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}/analytics")

    async def aanalytics(self, workflow_id: str) -> Dict[str, Any]:
        """Async version of analytics()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/analytics"
        )

    def cost_breakdown(self, workflow_id: str) -> Dict[str, Any]:
        """Get per-agent cost breakdown (tokens, cost, execution count, avg
        duration). Computed via a Celery task the backend blocks on for up
        to 30s server-side, so this call uses an extended client timeout."""
        return self.client.request(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/cost-breakdown",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    async def acost_breakdown(self, workflow_id: str) -> Dict[str, Any]:
        """Async version of cost_breakdown()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/cost-breakdown",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    def bottlenecks(self, workflow_id: str) -> Dict[str, Any]:
        """Identify workflow bottlenecks: slowest agents, slowest tasks,
        failing agents. Computed via a Celery task the backend blocks on for
        up to 30s server-side, so this call uses an extended client timeout."""
        return self.client.request(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/bottlenecks",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    async def abottlenecks(self, workflow_id: str) -> Dict[str, Any]:
        """Async version of bottlenecks()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/bottlenecks",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    def timeline(self, workflow_id: str) -> Dict[str, Any]:
        """Get a chronological timeline of workflow events (start/end, agent
        executions, task updates, interactions). Computed via a Celery task
        the backend blocks on for up to 30s server-side, so this call uses
        an extended client timeout."""
        return self.client.request(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/timeline",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    async def atimeline(self, workflow_id: str) -> Dict[str, Any]:
        """Async version of timeline()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/timeline",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )


def _agent_register_payload(
    name: str,
    role: Optional[str] = None,
    agent_type: Optional[str] = None,
    model_name: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    goal: Optional[str] = None,
    backstory: Optional[str] = None,
    tools: Optional[List[str]] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    meta_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for AgentWorkflowsResource.register_agent/aregister_agent."""
    data: Dict[str, Any] = {"name": name}
    data.update(
        _drop_none(
            role=role,
            agent_type=agent_type,
            model_name=model_name,
            llm_provider=llm_provider,
            llm_config=llm_config,
            system_prompt=system_prompt,
            goal=goal,
            backstory=backstory,
            tools=tools,
            capabilities=capabilities,
            meta_data=meta_data,
        )
    )
    return data


def _agent_task_create_payload(
    task_name: str,
    description: Optional[str] = None,
    task_type: Optional[str] = None,
    expected_output: Optional[str] = None,
    agent_id: Optional[str] = None,
    parent_task_id: Optional[str] = None,
    priority: int = 0,
    input_data: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for AgentWorkflowsResource.create_task/acreate_task."""
    data: Dict[str, Any] = {"task_name": task_name, "priority": priority}
    data.update(
        _drop_none(
            description=description,
            task_type=task_type,
            expected_output=expected_output,
            agent_id=agent_id,
            parent_task_id=parent_task_id,
            input_data=input_data,
            context=context,
        )
    )
    return data


class MetricsResource(BaseResource):
    """Model performance metrics API resource (backed by backend/api/v1/metrics.py)."""

    def create(
        self,
        model_id: str,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        metric_metadata: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new metric record."""
        data = _metric_create_payload(
            model_id=model_id,
            metric_type=metric_type,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_metadata=metric_metadata,
            sample_size=sample_size,
        )
        return self.client.request("POST", "/api/v1/metrics/", data=data)

    async def acreate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async version of create()."""
        data = _metric_create_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/metrics/", data=data)

    def create_batch(self, model_id: str, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create multiple metrics at once (max 100). Each dict in `metrics`
        needs metric_type/metric_name/metric_value."""
        data = {"model_id": model_id, "metrics": metrics}
        return self.client.request("POST", "/api/v1/metrics/batch", data=data)

    async def acreate_batch(
        self, model_id: str, metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Async version of create_batch()."""
        data = {"model_id": model_id, "metrics": metrics}
        return await self.client.arequest("POST", "/api/v1/metrics/batch", data=data)

    def calculate_classification(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        y_pred_proba: Optional[List[float]] = None,
        average: str = "binary",
        pos_label: int = 1,
    ) -> Dict[str, float]:
        """Calculate classification metrics (accuracy/precision/recall/F1/etc.)
        from predictions, without persisting anything."""
        params = {"average": average, "pos_label": pos_label}
        data = _drop_none(y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
        return self.client.request(
            "POST", "/api/v1/metrics/calculate/classification", data=data, params=params
        )

    async def acalculate_classification(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        y_pred_proba: Optional[List[float]] = None,
        average: str = "binary",
        pos_label: int = 1,
    ) -> Dict[str, float]:
        """Async version of calculate_classification()."""
        params = {"average": average, "pos_label": pos_label}
        data = _drop_none(y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
        return await self.client.arequest(
            "POST", "/api/v1/metrics/calculate/classification", data=data, params=params
        )

    def calculate_regression(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Calculate regression metrics (MAE/MSE/RMSE/R2/etc.) from
        predictions, without persisting anything."""
        data = {"y_true": y_true, "y_pred": y_pred}
        return self.client.request("POST", "/api/v1/metrics/calculate/regression", data=data)

    async def acalculate_regression(
        self, y_true: List[float], y_pred: List[float]
    ) -> Dict[str, float]:
        """Async version of calculate_regression()."""
        data = {"y_true": y_true, "y_pred": y_pred}
        return await self.client.arequest("POST", "/api/v1/metrics/calculate/regression", data=data)

    def get_model_metrics(
        self,
        model_id: str,
        metric_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get metrics recorded for a model, with optional type/date filters."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return self.client.request("GET", f"/api/v1/metrics/{model_id}", params=params)

    async def aget_model_metrics(
        self,
        model_id: str,
        metric_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of get_model_metrics()."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return await self.client.arequest("GET", f"/api/v1/metrics/{model_id}", params=params)

    def latest(self, model_id: str, metric_type: str) -> Dict[str, Any]:
        """Get the most recent metric of a specific type for a model."""
        return self.client.request("GET", f"/api/v1/metrics/{model_id}/latest/{metric_type}")

    async def alatest(self, model_id: str, metric_type: str) -> Dict[str, Any]:
        """Async version of latest()."""
        return await self.client.arequest("GET", f"/api/v1/metrics/{model_id}/latest/{metric_type}")

    def timeseries(
        self, model_id: str, metric_type: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """Get time series data for a metric over a required date range."""
        params = {"start_date": start_date, "end_date": end_date}
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/timeseries/{metric_type}", params=params
        )

    async def atimeseries(
        self, model_id: str, metric_type: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """Async version of timeseries()."""
        params = {"start_date": start_date, "end_date": end_date}
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/timeseries/{metric_type}", params=params
        )

    def aggregate(
        self,
        model_id: str,
        period: str,
        metric_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_recompute: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get metrics aggregated over `period` (e.g. "daily", "weekly").
        Defaults to the last 30 days if no date range is given."""
        params = {"period": period, "force_recompute": force_recompute}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return self.client.request("GET", f"/api/v1/metrics/{model_id}/aggregate", params=params)

    async def aaggregate(
        self,
        model_id: str,
        period: str,
        metric_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_recompute: bool = False,
    ) -> List[Dict[str, Any]]:
        """Async version of aggregate()."""
        params = {"period": period, "force_recompute": force_recompute}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/aggregate", params=params
        )

    def trend(self, model_id: str, metric_type: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Get trend statistics for a metric over `lookback_days`."""
        params = {"lookback_days": lookback_days}
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/trend/{metric_type}", params=params
        )

    async def atrend(
        self, model_id: str, metric_type: str, lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Async version of trend()."""
        params = {"lookback_days": lookback_days}
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/trend/{metric_type}", params=params
        )

    def rolling(self, model_id: str, metric_type: str, window_days: int = 7) -> Dict[str, Any]:
        """Get rolling-window statistics for a metric over `window_days`."""
        params = {"window_days": window_days}
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/rolling/{metric_type}", params=params
        )

    async def arolling(
        self, model_id: str, metric_type: str, window_days: int = 7
    ) -> Dict[str, Any]:
        """Async version of rolling()."""
        params = {"window_days": window_days}
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/rolling/{metric_type}", params=params
        )

    def summary(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Get a summary of all metrics for a model over the last `days`."""
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/summary", params={"days": days}
        )

    async def asummary(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Async version of summary()."""
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/summary", params={"days": days}
        )

    def delete(self, model_id: str) -> Dict[str, Any]:
        """Delete all metrics for a model."""
        return self.client.request("DELETE", f"/api/v1/metrics/{model_id}")

    async def adelete(self, model_id: str) -> Dict[str, Any]:
        """Async version of delete()."""
        return await self.client.arequest("DELETE", f"/api/v1/metrics/{model_id}")


def _metric_create_payload(
    model_id: str,
    metric_type: str,
    metric_name: str,
    metric_value: float,
    metric_metadata: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Shared request-body builder for MetricsResource.create/acreate."""
    data: Dict[str, Any] = {
        "model_id": model_id,
        "metric_type": metric_type,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }
    data.update(_drop_none(metric_metadata=metric_metadata, sample_size=sample_size))
    return data

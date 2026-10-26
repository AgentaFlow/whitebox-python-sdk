"""
API Resources

Resource classes for interacting with different WhiteBoxXAI API endpoints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from whiteboxxai.exceptions import APIConnectionError

if TYPE_CHECKING:
    from whiteboxxai.client import WhiteBoxXAI

logger = logging.getLogger(__name__)


class BaseResource:
    """Base class for API resources."""

    def __init__(self, client: WhiteBoxXAI):
        """Initialize resource with client."""
        self.client = client

    def _request_or_queue(
        self,
        method: str,
        endpoint: str,
        operation_type: str,
        queue_data: dict[str, Any],
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST/PATCH, falling back to the offline queue on connection failure.

        Only the sync write methods that mutate server state (model
        register/update_baseline, prediction log/log_batch, LLM log_call) use this --
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
        framework: str | None,
        version: str | None,
        auto_detect_git: bool,
        require_clean_git: bool,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        data = {
            "name": name,
            "model_type": model_type,
            # The backend's ModelCreate schema field is model_framework, not
            # framework (backend/schemas/model.py) -- Pydantic silently
            # drops unrecognized extra fields by default, so sending
            # "framework" here used to mean a caller's framework=... was
            # accepted but never actually persisted (issue #227, found via
            # a real end-to-end integration test). The public parameter
            # name here stays `framework` for backward compatibility; only
            # the wire key changes.
            "model_framework": framework,
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
        framework: str | None = None,
        version: str | None = None,
        auto_detect_git: bool = False,
        require_clean_git: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Register a new model.

        Args:
            name: Model name
            model_type: One of xgboost, random_forest, neural_network,
                linear_model, llm, custom
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
            ...     version="1.0.0",
            ...     model_type="random_forest",
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
        framework: str | None = None,
        version: str | None = None,
        auto_detect_git: bool = False,
        require_clean_git: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

    def get(self, model_id: str) -> dict[str, Any]:
        """Get model by ID."""
        return self.client.request("GET", f"/api/v1/models/{model_id}")

    async def aget(self, model_id: str) -> dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/models/{model_id}")

    def list(
        self,
        skip: int = 0,
        limit: int = 100,
        status_filter: str | None = None,
        model_type: str | None = None,
        owner_id: str | None = None,
        tags: str | None = None,
        search: str | None = None,
    ) -> List[dict[str, Any]]:
        """List registered models with optional filters."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
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
        status_filter: str | None = None,
        model_type: str | None = None,
        owner_id: str | None = None,
        tags: str | None = None,
        search: str | None = None,
    ) -> List[dict[str, Any]]:
        """Async version of list()."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
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

    def update(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update model metadata (partial update)."""
        return self.client.request("PATCH", f"/api/v1/models/{model_id}", data=kwargs)

    async def aupdate(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        """Async version of update()."""
        return await self.client.arequest("PATCH", f"/api/v1/models/{model_id}", data=kwargs)

    def update_status(self, model_id: str, new_status: str) -> dict[str, Any]:
        """Update model status (ACTIVE, INACTIVE, DEPRECATED, ARCHIVED)."""
        return self.client.request(
            "PATCH",
            f"/api/v1/models/{model_id}/status",
            params={"new_status": new_status},
        )

    async def aupdate_status(self, model_id: str, new_status: str) -> dict[str, Any]:
        """Async version of update_status()."""
        return await self.client.arequest(
            "PATCH",
            f"/api/v1/models/{model_id}/status",
            params={"new_status": new_status},
        )

    def get_versions(
        self, model_name: str, skip: int = 0, limit: int = 100
    ) -> List[dict[str, Any]]:
        """Get all versions of a model by name."""
        return self.client.request(
            "GET",
            f"/api/v1/models/{model_name}/versions",
            params={"skip": skip, "limit": limit},
        )

    async def aget_versions(
        self, model_name: str, skip: int = 0, limit: int = 100
    ) -> List[dict[str, Any]]:
        """Async version of get_versions()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/models/{model_name}/versions",
            params={"skip": skip, "limit": limit},
        )

    def get_latest(self, model_name: str) -> dict[str, Any]:
        """Get the latest version of a model by name."""
        return self.client.request("GET", f"/api/v1/models/{model_name}/latest")

    async def aget_latest(self, model_name: str) -> dict[str, Any]:
        """Async version of get_latest()."""
        return await self.client.arequest("GET", f"/api/v1/models/{model_name}/latest")

    def update_baseline(
        self,
        model_id: str,
        baseline_metrics: dict[str, Any],
        baseline_data_hash: str | None = None,
        baseline_data_count: int | None = None,
    ) -> dict[str, Any]:
        """Update model baseline metrics/profile (e.g. after retraining)."""
        params: dict[str, Any] = {}
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
        baseline_metrics: dict[str, Any],
        baseline_data_hash: str | None = None,
        baseline_data_count: int | None = None,
    ) -> dict[str, Any]:
        """Async version of update_baseline()."""
        params: dict[str, Any] = {}
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

    def archive(self, model_id: str) -> dict[str, Any]:
        """Archive a model (soft delete)."""
        return self.client.request("POST", f"/api/v1/models/{model_id}/archive")

    async def aarchive(self, model_id: str) -> dict[str, Any]:
        """Async version of archive()."""
        return await self.client.arequest("POST", f"/api/v1/models/{model_id}/archive")

    def restore(self, model_id: str) -> dict[str, Any]:
        """Restore an archived model to active status."""
        return self.client.request("POST", f"/api/v1/models/{model_id}/restore")

    async def arestore(self, model_id: str) -> dict[str, Any]:
        """Async version of restore()."""
        return await self.client.arequest("POST", f"/api/v1/models/{model_id}/restore")

    def delete(self, model_id: str) -> dict[str, Any]:
        """Permanently delete a model (hard delete, cannot be undone)."""
        return self.client.request("DELETE", f"/api/v1/models/{model_id}")

    async def adelete(self, model_id: str) -> dict[str, Any]:
        """Async version of delete()."""
        return await self.client.arequest("DELETE", f"/api/v1/models/{model_id}")

    def compare_versions(self, version1_id: str, version2_id: str) -> dict[str, Any]:
        """Compare two Git-backed model versions (`POST /api/v1/github/diff/compare`)."""
        return self.client.request(
            "POST",
            "/api/v1/github/diff/compare",
            data={"version1_id": version1_id, "version2_id": version2_id},
        )

    async def acompare_versions(self, version1_id: str, version2_id: str) -> dict[str, Any]:
        """Async version of compare_versions()."""
        return await self.client.arequest(
            "POST",
            "/api/v1/github/diff/compare",
            data={"version1_id": version1_id, "version2_id": version2_id},
        )

    def version_changelog(self, version1_id: str, version2_id: str) -> dict[str, Any]:
        """Generate a markdown changelog between two model versions (`GET /api/v1/github/diff/changelog`)."""
        return self.client.request(
            "GET",
            "/api/v1/github/diff/changelog",
            params={"version1_id": version1_id, "version2_id": version2_id},
        )

    async def aversion_changelog(self, version1_id: str, version2_id: str) -> dict[str, Any]:
        """Async version of version_changelog()."""
        return await self.client.arequest(
            "GET",
            "/api/v1/github/diff/changelog",
            params={"version1_id": version1_id, "version2_id": version2_id},
        )

    def list_git_versions(self, model_id: str, limit: int = 50) -> List[dict[str, Any]]:
        """List Git-backed version sources for a model (`GET /api/v1/github/diff/model/{model_id}/versions`)."""
        return self.client.request(
            "GET",
            f"/api/v1/github/diff/model/{model_id}/versions",
            params={"limit": limit},
        )

    async def alist_git_versions(self, model_id: str, limit: int = 50) -> List[dict[str, Any]]:
        """Async version of list_git_versions()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/github/diff/model/{model_id}/versions",
            params={"limit": limit},
        )

    def architecture_changes_timeline(self, model_id: str) -> dict[str, Any]:
        """Get the architecture changes timeline for a model (`GET /api/v1/github/diff/architecture-changes/{model_id}`)."""
        return self.client.request("GET", f"/api/v1/github/diff/architecture-changes/{model_id}")

    async def aarchitecture_changes_timeline(self, model_id: str) -> dict[str, Any]:
        """Async version of architecture_changes_timeline()."""
        return await self.client.arequest(
            "GET", f"/api/v1/github/diff/architecture-changes/{model_id}"
        )


class PredictionsResource(BaseResource):
    """Predictions API resource (backed by backend/api/v1/predictions.py)."""

    def log(
        self,
        model_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        prediction_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        data: dict[str, Any] = {
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
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        prediction_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of log()."""
        data: dict[str, Any] = {
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
        predictions: List[dict[str, Any]],
    ) -> dict[str, Any]:
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
        predictions: List[dict[str, Any]],
    ) -> dict[str, Any]:
        """Async version of log_batch()."""
        data = {"model_id": model_id, "predictions": predictions}
        return await self.client.arequest("POST", "/api/v1/predictions/log/batch", data=data)

    def get(self, prediction_id: str) -> dict[str, Any]:
        """Get a prediction by its ID."""
        return self.client.request("GET", f"/api/v1/predictions/{prediction_id}")

    async def aget(self, prediction_id: str) -> dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/predictions/{prediction_id}")

    def query(
        self,
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """
        Query predictions with filters.

        Args:
            model_id: Model ID (required by the backend)
            start_time: ISO-8601 timestamp, filter predictions after this time
            end_time: ISO-8601 timestamp, filter predictions before this time
            limit: Maximum results to return
            offset: Number of results to skip
        """
        data: dict[str, Any] = {"model_id": model_id, "limit": limit, "offset": offset}
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        return self.client.request("POST", "/api/v1/predictions/query", data=data)

    async def aquery(
        self,
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Async version of query()."""
        data: dict[str, Any] = {"model_id": model_id, "limit": limit, "offset": offset}
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        return await self.client.arequest("POST", "/api/v1/predictions/query", data=data)

    def get_stats(
        self,
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Get prediction statistics for a model."""
        params: dict[str, Any] = {}
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
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Async version of get_stats()."""
        params: dict[str, Any] = {}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return await self.client.arequest(
            "GET", f"/api/v1/predictions/models/{model_id}/stats", params=params
        )

    def get_recent(self, model_id: str, limit: int = 10) -> List[dict[str, Any]]:
        """Get the most recent predictions for a model (max 100)."""
        return self.client.request(
            "GET",
            f"/api/v1/predictions/models/{model_id}/recent",
            params={"limit": limit},
        )

    async def aget_recent(self, model_id: str, limit: int = 10) -> List[dict[str, Any]]:
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
        model_id: str,
        instance: dict[str, Any],
        method: str | None = None,
        prediction_id: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        use_cache: bool = True,
        total_cfs: int = 3,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """
        Generate a SHAP/LIME explanation synchronously (blocks until computed).

        Args:
            model_id: ID of the model to explain
            instance: Feature values for the instance to explain
            method: "shap" or "lime" (uses the model's configured default if omitted)
            prediction_id: Optional associated prediction ID
            num_features: Number of top features to return (1-100)
            num_samples: Number of samples the explainer draws (100-50000)
            use_cache: Reuse a cached explanation for an identical instance if one exists

        Returns:
            The computed explanation (feature_weights, base_value, score, ...)
        """
        data = {
            "model_id": model_id,
            "instance": instance,
            "method": method,
            "prediction_id": prediction_id,
            "num_features": num_features,
            "num_samples": num_samples,
            "use_cache": use_cache,
            "total_cfs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
        }
        return self.client.request("POST", "/api/v1/explanations/generate", data=data)

    async def agenerate(
        self,
        model_id: str,
        instance: dict[str, Any],
        method: str | None = None,
        prediction_id: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        use_cache: bool = True,
        total_cfs: int = 3,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Async version of generate()."""
        data = {
            "model_id": model_id,
            "instance": instance,
            "method": method,
            "prediction_id": prediction_id,
            "num_features": num_features,
            "num_samples": num_samples,
            "use_cache": use_cache,
            "total_cfs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
        }
        return await self.client.arequest("POST", "/api/v1/explanations/generate", data=data)

    def generate_bulk(
        self,
        model_id: str,
        instances: List[dict[str, Any]],
        method: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        parallel: bool = True,
        total_cfs: int = 3,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """
        Start bulk explanation generation for multiple instances (non-blocking).

        Dispatches one Celery task per instance (max 100 instances per call).
        Poll get()/aget() with each returned explanation_id.

        Args:
            model_id: ID of the model to explain
            instances: Feature values for each instance to explain (max 100)
            method: "shap" or "lime" (uses the model's configured default if omitted)
            num_features: Number of top features to return per explanation (1-100)
            num_samples: Number of samples the explainer draws (100-50000)
            parallel: Retained for backward compatibility; not read by the backend

        Returns:
            {"explanation_ids": [...], "total": ..., "message": ...}
        """
        data = {
            "model_id": model_id,
            "instances": instances,
            "method": method,
            "num_features": num_features,
            "num_samples": num_samples,
            "parallel": parallel,
            "total_cfs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
        }
        return self.client.request("POST", "/api/v1/explanations/generate/bulk", data=data)

    async def agenerate_bulk(
        self,
        model_id: str,
        instances: List[dict[str, Any]],
        method: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        parallel: bool = True,
        total_cfs: int = 3,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Async version of generate_bulk()."""
        data = {
            "model_id": model_id,
            "instances": instances,
            "method": method,
            "num_features": num_features,
            "num_samples": num_samples,
            "parallel": parallel,
            "total_cfs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
        }
        return await self.client.arequest("POST", "/api/v1/explanations/generate/bulk", data=data)

    def list_by_model(
        self,
        model_id: str,
        method: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List explanations for a model, optionally filtered by method/status."""
        params = _drop_none(method=method, status=status, limit=limit, offset=offset)
        return self.client.request("GET", f"/api/v1/explanations/model/{model_id}", params=params)

    async def alist_by_model(
        self,
        model_id: str,
        method: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Async version of list_by_model()."""
        params = _drop_none(method=method, status=status, limit=limit, offset=offset)
        return await self.client.arequest(
            "GET", f"/api/v1/explanations/model/{model_id}", params=params
        )

    def get_by_prediction(self, prediction_id: str) -> dict[str, Any]:
        """Get the explanation associated with a prediction, if one exists."""
        return self.client.request("GET", f"/api/v1/explanations/prediction/{prediction_id}")

    async def aget_by_prediction(self, prediction_id: str) -> dict[str, Any]:
        """Async version of get_by_prediction()."""
        return await self.client.arequest("GET", f"/api/v1/explanations/prediction/{prediction_id}")

    def get_stats(self, model_id: str) -> dict[str, Any]:
        """Get aggregated explanation statistics for a model."""
        return self.client.request("GET", f"/api/v1/explanations/model/{model_id}/stats")

    async def aget_stats(self, model_id: str) -> dict[str, Any]:
        """Async version of get_stats()."""
        return await self.client.arequest("GET", f"/api/v1/explanations/model/{model_id}/stats")

    def set_config(
        self,
        model_id: str,
        enabled: bool = True,
        default_method: str = "shap_kernel",
        auto_explain: bool = False,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        lime_config: dict[str, Any] | None = None,
        shap_config: dict[str, Any] | None = None,
        visualization_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or update a model's explanation configuration (upsert)."""
        data = {
            "model_id": model_id,
            "enabled": enabled,
            "default_method": default_method,
            "auto_explain": auto_explain,
            "cache_enabled": cache_enabled,
            "cache_ttl_hours": cache_ttl_hours,
            "lime_config": lime_config,
            "shap_config": shap_config,
            "visualization_config": visualization_config,
        }
        return self.client.request("POST", "/api/v1/explanations/config", data=data)

    async def aset_config(
        self,
        model_id: str,
        enabled: bool = True,
        default_method: str = "shap_kernel",
        auto_explain: bool = False,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        lime_config: dict[str, Any] | None = None,
        shap_config: dict[str, Any] | None = None,
        visualization_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of set_config()."""
        data = {
            "model_id": model_id,
            "enabled": enabled,
            "default_method": default_method,
            "auto_explain": auto_explain,
            "cache_enabled": cache_enabled,
            "cache_ttl_hours": cache_ttl_hours,
            "lime_config": lime_config,
            "shap_config": shap_config,
            "visualization_config": visualization_config,
        }
        return await self.client.arequest("POST", "/api/v1/explanations/config", data=data)

    def get_config(self, model_id: str) -> dict[str, Any]:
        """Get a model's explanation configuration."""
        return self.client.request("GET", f"/api/v1/explanations/config/{model_id}")

    async def aget_config(self, model_id: str) -> dict[str, Any]:
        """Async version of get_config()."""
        return await self.client.arequest("GET", f"/api/v1/explanations/config/{model_id}")

    def delete_config(self, model_id: str) -> dict[str, Any]:
        """Delete a model's explanation configuration."""
        return self.client.request("DELETE", f"/api/v1/explanations/config/{model_id}")

    async def adelete_config(self, model_id: str) -> dict[str, Any]:
        """Async version of delete_config()."""
        return await self.client.arequest("DELETE", f"/api/v1/explanations/config/{model_id}")

    def compare(
        self,
        explanation_ids: List[str],
        comparison_type: str = "feature_importance",
    ) -> dict[str, Any]:
        """
        Compare 2-20 explanations against each other.

        Args:
            explanation_ids: IDs of the explanations to compare (2-20)
            comparison_type: "feature_importance", "instance_similarity", or
                "method_agreement"
        """
        data = {"explanation_ids": explanation_ids, "comparison_type": comparison_type}
        return self.client.request("POST", "/api/v1/explanations/compare", data=data)

    async def acompare(
        self,
        explanation_ids: List[str],
        comparison_type: str = "feature_importance",
    ) -> dict[str, Any]:
        """Async version of compare()."""
        data = {"explanation_ids": explanation_ids, "comparison_type": comparison_type}
        return await self.client.arequest("POST", "/api/v1/explanations/compare", data=data)

    def visualize(
        self,
        explanation_id: str,
        plot_type: str,
        max_features: int = 10,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get visualization data for a single explanation.

        Args:
            explanation_id: The explanation to visualize
            plot_type: "waterfall", "force", or "feature_importance"
            max_features: Maximum number of features to include (1-50)
            config: Optional plot-specific configuration
        """
        data = {
            "explanation_id": explanation_id,
            "plot_type": plot_type,
            "max_features": max_features,
            "config": config,
        }
        return self.client.request("POST", "/api/v1/explanations/visualizations/single", data=data)

    async def avisualize(
        self,
        explanation_id: str,
        plot_type: str,
        max_features: int = 10,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of visualize()."""
        data = {
            "explanation_id": explanation_id,
            "plot_type": plot_type,
            "max_features": max_features,
            "config": config,
        }
        return await self.client.arequest(
            "POST", "/api/v1/explanations/visualizations/single", data=data
        )

    def visualize_multi(
        self,
        explanation_ids: List[str],
        plot_type: str,
        feature_index: int | None = None,
        interaction_index: int | None = None,
        max_features: int = 20,
        highlight_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Get visualization data spanning multiple explanations.

        Args:
            explanation_ids: The explanations to visualize together (1-100)
            plot_type: "summary", "decision", or "dependence"
            feature_index: Feature to focus on (for dependence plots)
            interaction_index: Feature to color by (for dependence plots)
            max_features: Maximum number of features to include (1-50)
            highlight_index: Index of an explanation to highlight (for decision plots)
        """
        data = {
            "explanation_ids": explanation_ids,
            "plot_type": plot_type,
            "feature_index": feature_index,
            "interaction_index": interaction_index,
            "max_features": max_features,
            "highlight_index": highlight_index,
        }
        return self.client.request("POST", "/api/v1/explanations/visualizations/multi", data=data)

    async def avisualize_multi(
        self,
        explanation_ids: List[str],
        plot_type: str,
        feature_index: int | None = None,
        interaction_index: int | None = None,
        max_features: int = 20,
        highlight_index: int | None = None,
    ) -> dict[str, Any]:
        """Async version of visualize_multi()."""
        data = {
            "explanation_ids": explanation_ids,
            "plot_type": plot_type,
            "feature_index": feature_index,
            "interaction_index": interaction_index,
            "max_features": max_features,
            "highlight_index": highlight_index,
        }
        return await self.client.arequest(
            "POST", "/api/v1/explanations/visualizations/multi", data=data
        )

    def generate_async(
        self,
        model_id: str,
        instance: dict[str, Any],
        method: str | None = None,
        prediction_id: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        total_cfs: int = 3,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
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
            "total_cfs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
        }
        return self.client.request("POST", "/api/v1/explanations/generate/async", data=data)

    async def agenerate_async(
        self,
        model_id: str,
        instance: dict[str, Any],
        method: str | None = None,
        prediction_id: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        total_cfs: int = 3,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Async version of generate_async()."""
        data = {
            "model_id": model_id,
            "instance": instance,
            "method": method,
            "prediction_id": prediction_id,
            "num_features": num_features,
            "num_samples": num_samples,
            "total_cfs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
        }
        return await self.client.arequest("POST", "/api/v1/explanations/generate/async", data=data)

    def get(self, explanation_id: str) -> dict[str, Any]:
        """Get explanation by ID."""
        return self.client.request("GET", f"/api/v1/explanations/{explanation_id}")

    async def aget(self, explanation_id: str) -> dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/explanations/{explanation_id}")


class DriftResource(BaseResource):
    """Drift detection API resource (backed by backend/api/v1/drift.py)."""

    def detect(
        self,
        model_id: str,
        window_size: int = 1000,
        feature_names: List[str] | None = None,
    ) -> dict[str, Any]:
        """
        Detect data drift for a model (ephemeral, not persisted).

        Args:
            model_id: Model ID
            window_size: Number of recent predictions to analyze (100-10000)
            feature_names: Optional subset of features to analyze

        Returns:
            Drift detection results
        """
        params: dict[str, Any] = {"window_size": window_size}
        if feature_names is not None:
            params["feature_names"] = feature_names
        return self.client.request("POST", f"/api/v1/drift/detect/{model_id}", params=params)

    async def adetect(
        self,
        model_id: str,
        window_size: int = 1000,
        feature_names: List[str] | None = None,
    ) -> dict[str, Any]:
        """Async version of detect()."""
        params: dict[str, Any] = {"window_size": window_size}
        if feature_names is not None:
            params["feature_names"] = feature_names
        return await self.client.arequest("POST", f"/api/v1/drift/detect/{model_id}", params=params)

    def create_report(self, model_id: str, window_size: int = 1000) -> dict[str, Any]:
        """Run drift analysis and persist the result as a drift report."""
        return self.client.request(
            "POST",
            "/api/v1/drift/reports",
            params={"model_id": model_id, "window_size": window_size},
        )

    async def acreate_report(self, model_id: str, window_size: int = 1000) -> dict[str, Any]:
        """Async version of create_report()."""
        return await self.client.arequest(
            "POST",
            "/api/v1/drift/reports",
            params={"model_id": model_id, "window_size": window_size},
        )

    def get_reports(self, model_id: str, limit: int = 10, skip: int = 0) -> List[dict[str, Any]]:
        """List drift reports for a model, most recent first."""
        return self.client.request(
            "GET",
            f"/api/v1/drift/reports/{model_id}",
            params={"limit": limit, "skip": skip},
        )

    async def aget_reports(
        self, model_id: str, limit: int = 10, skip: int = 0
    ) -> List[dict[str, Any]]:
        """Async version of get_reports()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/drift/reports/{model_id}",
            params={"limit": limit, "skip": skip},
        )

    def get_report(self, model_id: str, report_id: str) -> dict[str, Any]:
        """Get a specific drift report with per-feature statistics."""
        return self.client.request("GET", f"/api/v1/drift/reports/{model_id}/{report_id}")

    async def aget_report(self, model_id: str, report_id: str) -> dict[str, Any]:
        """Async version of get_report()."""
        return await self.client.arequest("GET", f"/api/v1/drift/reports/{model_id}/{report_id}")

    def get_trend(self, model_id: str, days: int = 7) -> dict[str, Any]:
        """Get drift trend over time for a model."""
        return self.client.request("GET", f"/api/v1/drift/trend/{model_id}", params={"days": days})

    async def aget_trend(self, model_id: str, days: int = 7) -> dict[str, Any]:
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
        group_data: dict[str, list[Any]],
        y_prob: List[float] | None = None,
        fairness_thresholds: dict[str, float] | None = None,
        metrics_to_compute: List[str] | None = None,
    ) -> dict[str, Any]:
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
        data: dict[str, Any] = {
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
        group_data: dict[str, list[Any]],
        y_prob: List[float] | None = None,
        fairness_thresholds: dict[str, float] | None = None,
        metrics_to_compute: List[str] | None = None,
    ) -> dict[str, Any]:
        """Async version of audit()."""
        data: dict[str, Any] = {
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

    def get_audit(self, audit_id: str) -> dict[str, Any]:
        """Get bias audit results by ID."""
        return self.client.request("GET", f"/api/v1/fairness/audits/{audit_id}")

    async def aget_audit(self, audit_id: str) -> dict[str, Any]:
        """Async version of get_audit()."""
        return await self.client.arequest("GET", f"/api/v1/fairness/audits/{audit_id}")

    def list_audits(
        self,
        model_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """List bias audits, optionally filtered by model."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if model_id is not None:
            params["model_id"] = model_id
        return self.client.request("GET", "/api/v1/fairness/audits", params=params)

    async def alist_audits(
        self,
        model_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Async version of list_audits()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if model_id is not None:
            params["model_id"] = model_id
        return await self.client.arequest("GET", "/api/v1/fairness/audits", params=params)

    def get_bias_history(self, model_id: str, days: int = 30) -> dict[str, Any]:
        """Get historical bias metrics/trend for a model."""
        return self.client.request(
            "GET",
            f"/api/v1/fairness/models/{model_id}/bias-history",
            params={"days": days},
        )

    async def aget_bias_history(self, model_id: str, days: int = 30) -> dict[str, Any]:
        """Async version of get_bias_history()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/fairness/models/{model_id}/bias-history",
            params={"days": days},
        )

    def get_metric_history(self, model_id: str, metric_type: str, days: int = 30) -> dict[str, Any]:
        """Get history for a specific fairness metric type."""
        return self.client.request(
            "GET",
            f"/api/v1/fairness/models/{model_id}/metrics/{metric_type}/history",
            params={"days": days},
        )

    async def aget_metric_history(
        self, model_id: str, metric_type: str, days: int = 30
    ) -> dict[str, Any]:
        """Async version of get_metric_history()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/fairness/models/{model_id}/metrics/{metric_type}/history",
            params={"days": days},
        )

    def get_latest_audit(self, model_id: str) -> dict[str, Any]:
        """Get the most recent bias audit for a model."""
        return self.client.request("GET", f"/api/v1/fairness/models/{model_id}/latest-audit")

    async def aget_latest_audit(self, model_id: str) -> dict[str, Any]:
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
        conditions: List[dict[str, Any]],
        model_id: str | None = None,
        description: str | None = None,
        logic_operator: str = "AND",
        notification_channels: List[dict[str, Any]] | None = None,
        throttle_minutes: int = 60,
        cooldown_minutes: int = 0,
        is_active: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

    async def acreate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of create()."""
        data = _alert_rule_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/alerts/rules", data=data)

    def list(
        self,
        model_id: str | None = None,
        alert_type: str | None = None,
        is_active: bool | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
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
        model_id: str | None = None,
        alert_type: str | None = None,
        is_active: bool | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
        """Async version of list()."""
        params = _drop_none(
            model_id=model_id,
            alert_type=alert_type,
            is_active=is_active,
            skip=skip,
            limit=limit,
        )
        return await self.client.arequest("GET", "/api/v1/alerts/rules", params=params)

    def get_rule(self, rule_id: str) -> dict[str, Any]:
        """Get a specific alert rule by ID."""
        return self.client.request("GET", f"/api/v1/alerts/rules/{rule_id}")

    async def aget_rule(self, rule_id: str) -> dict[str, Any]:
        """Async version of get_rule()."""
        return await self.client.arequest("GET", f"/api/v1/alerts/rules/{rule_id}")

    def update_rule(self, rule_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update an alert rule. Any AlertRuleCreate field may be passed as
        a kwarg; only supplied fields are changed."""
        return self.client.request("PATCH", f"/api/v1/alerts/rules/{rule_id}", data=kwargs)

    async def aupdate_rule(self, rule_id: str, **kwargs: Any) -> dict[str, Any]:
        """Async version of update_rule()."""
        return await self.client.arequest("PATCH", f"/api/v1/alerts/rules/{rule_id}", data=kwargs)

    def delete_rule(self, rule_id: str) -> dict[str, Any]:
        """Delete an alert rule."""
        return self.client.request("DELETE", f"/api/v1/alerts/rules/{rule_id}")

    async def adelete_rule(self, rule_id: str) -> dict[str, Any]:
        """Async version of delete_rule()."""
        return await self.client.arequest("DELETE", f"/api/v1/alerts/rules/{rule_id}")

    def evaluate_rule(
        self,
        rule_id: str,
        metric_values: dict[str, Any] | None = None,
        model_id: str | None = None,
        historical_values: List[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        metric_values: dict[str, Any] | None = None,
        model_id: str | None = None,
        historical_values: List[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        rule_id: str | None = None,
        model_id: str | None = None,
        alert_type: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        hours: int | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
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
        rule_id: str | None = None,
        model_id: str | None = None,
        alert_type: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        hours: int | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
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

    def get_instance(self, alert_id: str) -> dict[str, Any]:
        """Get a specific triggered alert instance by ID."""
        return self.client.request("GET", f"/api/v1/alerts/instances/{alert_id}")

    async def aget_instance(self, alert_id: str) -> dict[str, Any]:
        """Async version of get_instance()."""
        return await self.client.arequest("GET", f"/api/v1/alerts/instances/{alert_id}")

    def acknowledge(self, alert_id: str, user_id: str, notes: str | None = None) -> dict[str, Any]:
        """Acknowledge an active alert instance."""
        data = _drop_none(user_id=user_id, notes=notes)
        return self.client.request(
            "POST", f"/api/v1/alerts/instances/{alert_id}/acknowledge", data=data
        )

    async def aacknowledge(
        self, alert_id: str, user_id: str, notes: str | None = None
    ) -> dict[str, Any]:
        """Async version of acknowledge()."""
        data = _drop_none(user_id=user_id, notes=notes)
        return await self.client.arequest(
            "POST", f"/api/v1/alerts/instances/{alert_id}/acknowledge", data=data
        )

    def resolve(
        self, alert_id: str, user_id: str, resolution_notes: str | None = None
    ) -> dict[str, Any]:
        """Resolve an alert instance."""
        data = _drop_none(user_id=user_id, resolution_notes=resolution_notes)
        return self.client.request(
            "POST", f"/api/v1/alerts/instances/{alert_id}/resolve", data=data
        )

    async def aresolve(
        self, alert_id: str, user_id: str, resolution_notes: str | None = None
    ) -> dict[str, Any]:
        """Async version of resolve()."""
        data = _drop_none(user_id=user_id, resolution_notes=resolution_notes)
        return await self.client.arequest(
            "POST", f"/api/v1/alerts/instances/{alert_id}/resolve", data=data
        )

    def snooze(self, alert_id: str, snooze_minutes: int) -> dict[str, Any]:
        """Snooze an active alert instance for `snooze_minutes` (max 1 week)."""
        return self.client.request(
            "POST",
            f"/api/v1/alerts/instances/{alert_id}/snooze",
            data={"snooze_minutes": snooze_minutes},
        )

    async def asnooze(self, alert_id: str, snooze_minutes: int) -> dict[str, Any]:
        """Async version of snooze()."""
        return await self.client.arequest(
            "POST",
            f"/api/v1/alerts/instances/{alert_id}/snooze",
            data={"snooze_minutes": snooze_minutes},
        )

    def statistics(self, hours: int = 24) -> dict[str, Any]:
        """Get alert statistics for the organization (counts, severity
        distribution, MTTA/MTTR) over the last `hours`."""
        return self.client.request("GET", "/api/v1/alerts/statistics", params={"hours": hours})

    async def astatistics(self, hours: int = 24) -> dict[str, Any]:
        """Async version of statistics()."""
        return await self.client.arequest(
            "GET", "/api/v1/alerts/statistics", params={"hours": hours}
        )


def _alert_rule_payload(
    name: str,
    alert_type: str,
    severity: str,
    conditions: List[dict[str, Any]],
    model_id: str | None = None,
    description: str | None = None,
    logic_operator: str = "AND",
    notification_channels: List[dict[str, Any]] | None = None,
    throttle_minutes: int = 60,
    cooldown_minutes: int = 0,
    is_active: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Shared request-body builder for AlertsResource.create/acreate."""
    data: dict[str, Any] = {
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
    metric_values: dict[str, Any] | None = None,
    model_id: str | None = None,
    historical_values: List[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for AlertsResource.evaluate_rule/aevaluate_rule."""
    return {
        "metric_values": metric_values or {},
        "model_id": model_id,
        "historical_values": historical_values,
        "metadata": metadata or {},
    }


class AlertIntelligenceResource(BaseResource):
    """Alert Intelligence API resource (backed by
    backend/api/v1/alert_intelligence.py; issue #224).

    Unlike an earlier revision of this resource (issue #295), every
    endpoint here derives organization scoping from the caller's own
    authenticated session -- there is no organization_id argument to pass.

    predict_trends()/detect_seasonality()/forecast_volume() compute real
    statistics (numpy-based linear trend + hour-of-day seasonality) from
    the organization's actual alert history, not a full forecasting model
    (no ARIMA/Prophet) -- see each method's docstring for what's actually
    computed, including the honest "insufficient_data" fallback used when
    too little history exists.
    """

    # ==================== Alert Correlation ====================

    def detect_correlations(
        self, alert_id: str, look_back_hours: int = 24, min_score: float = 0.5
    ) -> dict[str, Any]:
        """Detect correlations for an alert (temporal, model, feature, and
        pattern-based relationships to other alerts)."""
        params = {"look_back_hours": look_back_hours, "min_score": min_score}
        return self.client.request(
            "POST",
            f"/api/v1/alert-intelligence/alerts/{alert_id}/correlations",
            params=params,
        )

    async def adetect_correlations(
        self, alert_id: str, look_back_hours: int = 24, min_score: float = 0.5
    ) -> dict[str, Any]:
        """Async version of detect_correlations()."""
        params = {"look_back_hours": look_back_hours, "min_score": min_score}
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-intelligence/alerts/{alert_id}/correlations",
            params=params,
        )

    def get_correlation_clusters(
        self,
        min_cluster_size: int = 3,
        min_score: float = 0.7,
        hours_back: int = 24,
    ) -> dict[str, Any]:
        """Get clusters of highly correlated alerts -- useful for spotting
        widespread issues or cascading failures."""
        params = {
            "min_cluster_size": min_cluster_size,
            "min_score": min_score,
            "hours_back": hours_back,
        }
        return self.client.request(
            "GET", "/api/v1/alert-intelligence/correlations/clusters", params=params
        )

    async def aget_correlation_clusters(
        self,
        min_cluster_size: int = 3,
        min_score: float = 0.7,
        hours_back: int = 24,
    ) -> dict[str, Any]:
        """Async version of get_correlation_clusters()."""
        params = {
            "min_cluster_size": min_cluster_size,
            "min_score": min_score,
            "hours_back": hours_back,
        }
        return await self.client.arequest(
            "GET", "/api/v1/alert-intelligence/correlations/clusters", params=params
        )

    # ==================== Root Cause Analysis ====================

    def analyze_root_cause(self, alert_id: str, analyze_history: bool = True) -> dict[str, Any]:
        """Analyze potential root causes (data drift, model degradation,
        feature anomalies, system issues) for an alert."""
        return self.client.request(
            "POST",
            f"/api/v1/alert-intelligence/alerts/{alert_id}/root-cause",
            params={"analyze_history": analyze_history},
        )

    async def aanalyze_root_cause(
        self, alert_id: str, analyze_history: bool = True
    ) -> dict[str, Any]:
        """Async version of analyze_root_cause()."""
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-intelligence/alerts/{alert_id}/root-cause",
            params={"analyze_history": analyze_history},
        )

    def verify_root_cause(
        self, root_cause_id: int, verified: bool, notes: str | None = None
    ) -> dict[str, Any]:
        """Mark a previously computed root cause as verified or rejected."""
        data = {"root_cause_id": root_cause_id, "verified": verified, "notes": notes}
        return self.client.request(
            "POST",
            f"/api/v1/alert-intelligence/root-causes/{root_cause_id}/verify",
            data=data,
        )

    async def averify_root_cause(
        self, root_cause_id: int, verified: bool, notes: str | None = None
    ) -> dict[str, Any]:
        """Async version of verify_root_cause()."""
        data = {"root_cause_id": root_cause_id, "verified": verified, "notes": notes}
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-intelligence/root-causes/{root_cause_id}/verify",
            data=data,
        )

    def get_similar_incidents(self, cause_type: str, limit: int = 10) -> List[dict[str, Any]]:
        """Find past alerts sharing the same root-cause type -- useful for
        learning from historical resolutions."""
        params = {"cause_type": cause_type, "limit": limit}
        return self.client.request(
            "GET", "/api/v1/alert-intelligence/root-causes/similar", params=params
        )

    async def aget_similar_incidents(
        self, cause_type: str, limit: int = 10
    ) -> List[dict[str, Any]]:
        """Async version of get_similar_incidents()."""
        params = {"cause_type": cause_type, "limit": limit}
        return await self.client.arequest(
            "GET", "/api/v1/alert-intelligence/root-causes/similar", params=params
        )

    # ==================== Alert Priority Ranking ====================

    def calculate_priority(
        self,
        alert_id: str,
        business_impact_weight: float = 0.3,
        urgency_weight: float = 0.3,
        criticality_weight: float = 0.25,
        history_weight: float = 0.15,
    ) -> dict[str, Any]:
        """Compute a weighted priority score (0-100) and level
        (critical/high/medium/low) for an alert."""
        params = {
            "business_impact_weight": business_impact_weight,
            "urgency_weight": urgency_weight,
            "criticality_weight": criticality_weight,
            "history_weight": history_weight,
        }
        return self.client.request(
            "POST",
            f"/api/v1/alert-intelligence/alerts/{alert_id}/priority",
            params=params,
        )

    async def acalculate_priority(
        self,
        alert_id: str,
        business_impact_weight: float = 0.3,
        urgency_weight: float = 0.3,
        criticality_weight: float = 0.25,
        history_weight: float = 0.15,
    ) -> dict[str, Any]:
        """Async version of calculate_priority()."""
        params = {
            "business_impact_weight": business_impact_weight,
            "urgency_weight": urgency_weight,
            "criticality_weight": criticality_weight,
            "history_weight": history_weight,
        }
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-intelligence/alerts/{alert_id}/priority",
            params=params,
        )

    def adjust_priority(
        self,
        alert_id: str,
        new_priority_score: float,
        new_priority_level: str,
        adjustment_reason: str,
    ) -> dict[str, Any]:
        """Manually override a calculated priority. `adjustment_reason` is
        required justification."""
        data = {
            "alert_id": alert_id,
            "new_priority_score": new_priority_score,
            "new_priority_level": new_priority_level,
            "adjustment_reason": adjustment_reason,
        }
        return self.client.request(
            "PATCH", f"/api/v1/alert-intelligence/alerts/{alert_id}/priority", data=data
        )

    async def aadjust_priority(
        self,
        alert_id: str,
        new_priority_score: float,
        new_priority_level: str,
        adjustment_reason: str,
    ) -> dict[str, Any]:
        """Async version of adjust_priority()."""
        data = {
            "alert_id": alert_id,
            "new_priority_score": new_priority_score,
            "new_priority_level": new_priority_level,
            "adjustment_reason": adjustment_reason,
        }
        return await self.client.arequest(
            "PATCH", f"/api/v1/alert-intelligence/alerts/{alert_id}/priority", data=data
        )

    def get_priority_distribution(self, hours_back: int = 24) -> dict[str, Any]:
        """Get a breakdown of alerts by priority level."""
        params = {"hours_back": hours_back}
        return self.client.request(
            "GET", "/api/v1/alert-intelligence/priority/distribution", params=params
        )

    async def aget_priority_distribution(self, hours_back: int = 24) -> dict[str, Any]:
        """Async version of get_priority_distribution()."""
        params = {"hours_back": hours_back}
        return await self.client.arequest(
            "GET", "/api/v1/alert-intelligence/priority/distribution", params=params
        )

    def get_priority_queue(self, limit: int = 20) -> List[dict[str, Any]]:
        """Get the prioritized alert queue for triage and assignment."""
        params = {"limit": limit}
        return self.client.request(
            "GET", "/api/v1/alert-intelligence/priority/queue", params=params
        )

    async def aget_priority_queue(self, limit: int = 20) -> List[dict[str, Any]]:
        """Async version of get_priority_queue()."""
        params = {"limit": limit}
        return await self.client.arequest(
            "GET", "/api/v1/alert-intelligence/priority/queue", params=params
        )

    # ==================== Pattern Detection ====================

    def detect_patterns(self, hours_back: int = 168) -> dict[str, Any]:
        """Detect recurring temporal/frequency/correlation/anomaly patterns
        in alerts."""
        params = {"hours_back": hours_back}
        return self.client.request("GET", "/api/v1/alert-intelligence/patterns", params=params)

    async def adetect_patterns(self, hours_back: int = 168) -> dict[str, Any]:
        """Async version of detect_patterns()."""
        params = {"hours_back": hours_back}
        return await self.client.arequest(
            "GET", "/api/v1/alert-intelligence/patterns", params=params
        )

    def get_pattern(self, pattern_id: int) -> dict[str, Any]:
        """Get a specific alert pattern's details."""
        return self.client.request("GET", f"/api/v1/alert-intelligence/patterns/{pattern_id}")

    async def aget_pattern(self, pattern_id: int) -> dict[str, Any]:
        """Async version of get_pattern()."""
        return await self.client.arequest(
            "GET", f"/api/v1/alert-intelligence/patterns/{pattern_id}"
        )

    def deactivate_pattern(self, pattern_id: int) -> dict[str, Any]:
        """Mark a pattern inactive (resolved or no longer relevant)."""
        return self.client.request(
            "PATCH", f"/api/v1/alert-intelligence/patterns/{pattern_id}/deactivate"
        )

    async def adeactivate_pattern(self, pattern_id: int) -> dict[str, Any]:
        """Async version of deactivate_pattern()."""
        return await self.client.arequest(
            "PATCH", f"/api/v1/alert-intelligence/patterns/{pattern_id}/deactivate"
        )

    # ==================== Fatigue Reduction ====================

    def apply_fatigue_reduction(
        self,
        enable_auto_throttling: bool = True,
        enable_intelligent_grouping: bool = True,
        enable_auto_snooze: bool = True,
        similarity_threshold: float = 0.8,
        grouping_window_minutes: int = 60,
        auto_snooze_threshold: int = 10,
    ) -> dict[str, Any]:
        """Apply automated throttling/grouping/snoozing to reduce alert
        noise. Returns a summary of actions taken."""
        data = _fatigue_reduction_config_payload(
            enable_auto_throttling=enable_auto_throttling,
            enable_intelligent_grouping=enable_intelligent_grouping,
            enable_auto_snooze=enable_auto_snooze,
            similarity_threshold=similarity_threshold,
            grouping_window_minutes=grouping_window_minutes,
            auto_snooze_threshold=auto_snooze_threshold,
        )
        return self.client.request(
            "POST",
            "/api/v1/alert-intelligence/fatigue-reduction/apply",
            data=data,
        )

    async def aapply_fatigue_reduction(
        self,
        enable_auto_throttling: bool = True,
        enable_intelligent_grouping: bool = True,
        enable_auto_snooze: bool = True,
        similarity_threshold: float = 0.8,
        grouping_window_minutes: int = 60,
        auto_snooze_threshold: int = 10,
    ) -> dict[str, Any]:
        """Async version of apply_fatigue_reduction()."""
        data = _fatigue_reduction_config_payload(
            enable_auto_throttling=enable_auto_throttling,
            enable_intelligent_grouping=enable_intelligent_grouping,
            enable_auto_snooze=enable_auto_snooze,
            similarity_threshold=similarity_threshold,
            grouping_window_minutes=grouping_window_minutes,
            auto_snooze_threshold=auto_snooze_threshold,
        )
        return await self.client.arequest(
            "POST",
            "/api/v1/alert-intelligence/fatigue-reduction/apply",
            data=data,
        )

    def get_throttle_recommendations(self) -> List[dict[str, Any]]:
        """Get recommended alert-throttling rules to reduce fatigue without
        missing critical issues."""
        return self.client.request(
            "GET",
            "/api/v1/alert-intelligence/fatigue-reduction/recommendations",
        )

    async def aget_throttle_recommendations(self) -> List[dict[str, Any]]:
        """Async version of get_throttle_recommendations()."""
        return await self.client.arequest(
            "GET",
            "/api/v1/alert-intelligence/fatigue-reduction/recommendations",
        )

    # ==================== Enhanced Grouping ====================

    def ml_group_alerts(
        self,
        clustering_method: str = "kmeans",
        num_clusters: int | None = None,
    ) -> dict[str, Any]:
        """Cluster alerts via ML (kmeans/dbscan/hierarchical). Automatically
        determines the optimal number of clusters if not specified."""
        params = _drop_none(
            clustering_method=clustering_method,
            num_clusters=num_clusters,
        )
        return self.client.request("POST", "/api/v1/alert-intelligence/grouping/ml", params=params)

    async def aml_group_alerts(
        self,
        clustering_method: str = "kmeans",
        num_clusters: int | None = None,
    ) -> dict[str, Any]:
        """Async version of ml_group_alerts()."""
        params = _drop_none(
            clustering_method=clustering_method,
            num_clusters=num_clusters,
        )
        return await self.client.arequest(
            "POST", "/api/v1/alert-intelligence/grouping/ml", params=params
        )

    def temporal_group_alerts(
        self,
        time_window_minutes: int = 60,
        similarity_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Group alerts occurring in the same time window with similar
        characteristics."""
        params = {
            "time_window_minutes": time_window_minutes,
            "similarity_threshold": similarity_threshold,
        }
        return self.client.request(
            "POST", "/api/v1/alert-intelligence/grouping/temporal", params=params
        )

    async def atemporal_group_alerts(
        self,
        time_window_minutes: int = 60,
        similarity_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Async version of temporal_group_alerts()."""
        params = {
            "time_window_minutes": time_window_minutes,
            "similarity_threshold": similarity_threshold,
        }
        return await self.client.arequest(
            "POST", "/api/v1/alert-intelligence/grouping/temporal", params=params
        )

    # ==================== Enhanced Trend Analysis ====================

    def predict_trends(self, time_horizon: str = "next_day") -> dict[str, Any]:
        """Forecast alert volume/severity/type for `time_horizon`
        (next_hour/next_day/next_week).

        Uses a least-squares linear trend (numpy.polyfit) over the
        organization's real historical hourly alert counts, not a full
        time-series forecasting model (no ARIMA/Prophet). Returns
        `trend_direction="insufficient_data"` and zeroed values when fewer
        than 24 real hourly data points exist; `sample_size` in the response
        always discloses exactly how much real history backed the estimate.
        """
        params = {"time_horizon": time_horizon}
        return self.client.request(
            "POST", "/api/v1/alert-intelligence/trends/predict", params=params
        )

    async def apredict_trends(self, time_horizon: str = "next_day") -> dict[str, Any]:
        """Async version of predict_trends()."""
        params = {"time_horizon": time_horizon}
        return await self.client.arequest(
            "POST", "/api/v1/alert-intelligence/trends/predict", params=params
        )

    def detect_seasonality(self, days_back: int = 30) -> dict[str, Any]:
        """Detect hour-of-day seasonality in alert volume.

        Real detection from the organization's actual alert history (24
        hourly buckets summed across the requested window). Day-of-week/
        week-of-month cycles are not computed, so `pattern_type` is always
        `"hourly"` (or `"insufficient_data"` below a 24-data-point floor),
        never a fabricated `"daily"`/`"weekly"` label.
        """
        params = {"days_back": days_back}
        return self.client.request(
            "GET", "/api/v1/alert-intelligence/trends/seasonality", params=params
        )

    async def adetect_seasonality(self, days_back: int = 30) -> dict[str, Any]:
        """Async version of detect_seasonality()."""
        params = {"days_back": days_back}
        return await self.client.arequest(
            "GET", "/api/v1/alert-intelligence/trends/seasonality", params=params
        )

    def enhanced_trend_analysis(
        self, metric: str = "alert_count", hours_back: int = 168
    ) -> dict[str, Any]:
        """Combine real alert-volume time-series data with a real
        hour-of-day seasonality estimate, a numpy.polyfit trend prediction
        (with an R²-based confidence score), and real z-score anomaly
        detection over the same window -- all computed from the
        organization's actual alert history. Not a full forecasting model
        (no ARIMA/Prophet)."""
        params = {"metric": metric, "hours_back": hours_back}
        return self.client.request(
            "POST", "/api/v1/alert-intelligence/trends/enhanced", params=params
        )

    async def aenhanced_trend_analysis(
        self, metric: str = "alert_count", hours_back: int = 168
    ) -> dict[str, Any]:
        """Async version of enhanced_trend_analysis()."""
        params = {"metric": metric, "hours_back": hours_back}
        return await self.client.arequest(
            "POST", "/api/v1/alert-intelligence/trends/enhanced", params=params
        )

    def forecast_volume(self, forecast_hours: int = 24) -> dict[str, Any]:
        """Forecast future alert volume with confidence intervals.

        Uses a least-squares linear trend (numpy.polyfit) over real
        historical hourly alert counts, shaped by the organization's real
        hour-of-day seasonality. NOT ARIMA/Prophet. Forward-looking
        `anomaly_predictions` are intentionally always empty -- predicting
        future anomalies needs its own model, out of scope here; anomaly
        detection over *observed* data is available via
        enhanced_trend_analysis() instead. Falls back to an
        honestly-labeled zeroed forecast (`seasonal_pattern.type ==
        "insufficient_data"`) when fewer than 24 hourly data points exist.
        """
        params = {"forecast_hours": forecast_hours}
        return self.client.request(
            "POST", "/api/v1/alert-intelligence/trends/forecast", params=params
        )

    async def aforecast_volume(self, forecast_hours: int = 24) -> dict[str, Any]:
        """Async version of forecast_volume()."""
        params = {"forecast_hours": forecast_hours}
        return await self.client.arequest(
            "POST", "/api/v1/alert-intelligence/trends/forecast", params=params
        )


def _fatigue_reduction_config_payload(
    enable_auto_throttling: bool = True,
    enable_intelligent_grouping: bool = True,
    enable_auto_snooze: bool = True,
    similarity_threshold: float = 0.8,
    grouping_window_minutes: int = 60,
    auto_snooze_threshold: int = 10,
) -> dict[str, Any]:
    """Shared request-body builder for
    AlertIntelligenceResource.apply_fatigue_reduction/aapply_fatigue_reduction."""
    return {
        "enable_auto_throttling": enable_auto_throttling,
        "enable_intelligent_grouping": enable_intelligent_grouping,
        "enable_auto_snooze": enable_auto_snooze,
        "similarity_threshold": similarity_threshold,
        "grouping_window_minutes": grouping_window_minutes,
        "auto_snooze_threshold": auto_snooze_threshold,
    }


class AlertManagementResource(BaseResource):
    """Alert Management API resource (backed by
    backend/api/v1/alert_management.py; issue #224).

    Unlike AlertIntelligenceResource, every endpoint here derives
    organization scoping from the caller's own authenticated session --
    there is no organization_id argument to pass.

    acknowledge()/resolve()/snooze() hit a different URL
    (/alert-management/alerts/{id}/...) than AlertsResource's same-named
    methods (/alerts/instances/{id}/...) -- they are not duplicates, they
    are genuinely different backend endpoints with different history.

    Escalation-policy CRUD (create/list/get/update/delete_escalation_policy)
    and bulk_action() are SDK-only: they are not exposed as MCP tools
    (admin-console-shaped, not agent-shaped -- see issue #224).
    """

    # ==================== Alert History ====================

    def get_history(self, alert_id: str) -> dict[str, Any]:
        """Get the full action timeline (triggered/acknowledged/snoozed/
        resolved/escalated/commented) for an alert."""
        return self.client.request("GET", f"/api/v1/alert-management/alerts/{alert_id}/history")

    async def aget_history(self, alert_id: str) -> dict[str, Any]:
        """Async version of get_history()."""
        return await self.client.arequest(
            "GET", f"/api/v1/alert-management/alerts/{alert_id}/history"
        )

    # ==================== Alert Analytics ====================

    def get_analytics(
        self, start_time: str | None = None, end_time: str | None = None
    ) -> dict[str, Any]:
        """Get comprehensive alert analytics (counts, MTTA/MTTR,
        distributions, escalation metrics) for the caller's organization.
        `start_time`/`end_time` are ISO 8601 timestamps."""
        params = _drop_none(start_time=start_time, end_time=end_time)
        return self.client.request("GET", "/api/v1/alert-management/analytics", params=params)

    async def aget_analytics(
        self, start_time: str | None = None, end_time: str | None = None
    ) -> dict[str, Any]:
        """Async version of get_analytics()."""
        params = _drop_none(start_time=start_time, end_time=end_time)
        return await self.client.arequest(
            "GET", "/api/v1/alert-management/analytics", params=params
        )

    def get_trend(
        self,
        start: str,
        end: str,
        interval: str = "hour",
        group_by: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get alert volume as a time series between `start` and `end`
        (ISO 8601 timestamps), bucketed by `interval` (hour/day/week)."""
        data = _alert_time_series_query_payload(start, end, interval, group_by, filters)
        return self.client.request("POST", "/api/v1/alert-management/analytics/trend", data=data)

    async def aget_trend(
        self,
        start: str,
        end: str,
        interval: str = "hour",
        group_by: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of get_trend()."""
        data = _alert_time_series_query_payload(start, end, interval, group_by, filters)
        return await self.client.arequest(
            "POST", "/api/v1/alert-management/analytics/trend", data=data
        )

    def get_velocity(self) -> dict[str, Any]:
        """Get alert generation rate trends (hour-over-hour, day-over-day,
        rolling averages) for the caller's organization."""
        return self.client.request("GET", "/api/v1/alert-management/analytics/velocity")

    async def aget_velocity(self) -> dict[str, Any]:
        """Async version of get_velocity()."""
        return await self.client.arequest("GET", "/api/v1/alert-management/analytics/velocity")

    def get_grouping(self, group_by: str) -> dict[str, Any]:
        """Group active alerts by a dimension: "type", "severity", "model",
        or "rule" -- useful for spotting hotspots."""
        return self.client.request(
            "GET",
            "/api/v1/alert-management/analytics/grouping",
            params={"group_by": group_by},
        )

    async def aget_grouping(self, group_by: str) -> dict[str, Any]:
        """Async version of get_grouping()."""
        return await self.client.arequest(
            "GET",
            "/api/v1/alert-management/analytics/grouping",
            params={"group_by": group_by},
        )

    def detect_fatigue(self) -> dict[str, Any]:
        """Detect alert-fatigue indicators and get recommendations for
        reducing alert noise."""
        return self.client.request("GET", "/api/v1/alert-management/analytics/fatigue")

    async def adetect_fatigue(self) -> dict[str, Any]:
        """Async version of detect_fatigue()."""
        return await self.client.arequest("GET", "/api/v1/alert-management/analytics/fatigue")

    # ==================== Alert Lifecycle Management ====================

    def acknowledge(self, alert_id: str, notes: str | None = None) -> dict[str, Any]:
        """Acknowledge an alert. Distinct from AlertsResource.acknowledge --
        this hits /alert-management/alerts/{id}/acknowledge."""
        params = _drop_none(notes=notes)
        return self.client.request(
            "POST",
            f"/api/v1/alert-management/alerts/{alert_id}/acknowledge",
            params=params,
        )

    async def aacknowledge(self, alert_id: str, notes: str | None = None) -> dict[str, Any]:
        """Async version of acknowledge()."""
        params = _drop_none(notes=notes)
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-management/alerts/{alert_id}/acknowledge",
            params=params,
        )

    def resolve(self, alert_id: str, resolution_notes: str | None = None) -> dict[str, Any]:
        """Resolve an alert. Distinct from AlertsResource.resolve -- this
        hits /alert-management/alerts/{id}/resolve."""
        params = _drop_none(resolution_notes=resolution_notes)
        return self.client.request(
            "POST",
            f"/api/v1/alert-management/alerts/{alert_id}/resolve",
            params=params,
        )

    async def aresolve(self, alert_id: str, resolution_notes: str | None = None) -> dict[str, Any]:
        """Async version of resolve()."""
        params = _drop_none(resolution_notes=resolution_notes)
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-management/alerts/{alert_id}/resolve",
            params=params,
        )

    def snooze(self, alert_id: str, snooze_minutes: int) -> dict[str, Any]:
        """Snooze an alert for `snooze_minutes` (max 1 week). Distinct from
        AlertsResource.snooze -- this hits
        /alert-management/alerts/{id}/snooze."""
        return self.client.request(
            "POST",
            f"/api/v1/alert-management/alerts/{alert_id}/snooze",
            params={"snooze_minutes": snooze_minutes},
        )

    async def asnooze(self, alert_id: str, snooze_minutes: int) -> dict[str, Any]:
        """Async version of snooze()."""
        return await self.client.arequest(
            "POST",
            f"/api/v1/alert-management/alerts/{alert_id}/snooze",
            params={"snooze_minutes": snooze_minutes},
        )

    # ==================== Bulk Operations (SDK-only) ====================

    def bulk_action(
        self,
        alert_ids: List[str],
        action: str,
        user_id: str,
        notes: str | None = None,
        snooze_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Acknowledge or resolve up to 100 alerts at once.

        Only action="acknowledge" and action="resolve" are actually
        implemented by the backend -- despite the schema documenting
        "snooze"/"mute" as possible actions, the handler 400s on anything
        else.
        """
        data: dict[str, Any] = {
            "alert_ids": alert_ids,
            "action": action,
            "user_id": user_id,
        }
        data.update(_drop_none(notes=notes, snooze_minutes=snooze_minutes))
        return self.client.request("POST", "/api/v1/alert-management/alerts/bulk-action", data=data)

    async def abulk_action(
        self,
        alert_ids: List[str],
        action: str,
        user_id: str,
        notes: str | None = None,
        snooze_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Async version of bulk_action(). Same acknowledge/resolve-only
        caveat."""
        data: dict[str, Any] = {
            "alert_ids": alert_ids,
            "action": action,
            "user_id": user_id,
        }
        data.update(_drop_none(notes=notes, snooze_minutes=snooze_minutes))
        return await self.client.arequest(
            "POST", "/api/v1/alert-management/alerts/bulk-action", data=data
        )

    # ==================== Alert Comments ====================

    def create_comment(
        self, alert_id: str, comment: str, comment_type: str = "note"
    ) -> dict[str, Any]:
        """Add a comment to an alert (comment_type: note/action/resolution)."""
        data = {
            "alert_id": alert_id,
            "comment": comment,
            "comment_type": comment_type,
        }
        return self.client.request(
            "POST", f"/api/v1/alert-management/alerts/{alert_id}/comments", data=data
        )

    async def acreate_comment(
        self, alert_id: str, comment: str, comment_type: str = "note"
    ) -> dict[str, Any]:
        """Async version of create_comment()."""
        data = {
            "alert_id": alert_id,
            "comment": comment,
            "comment_type": comment_type,
        }
        return await self.client.arequest(
            "POST", f"/api/v1/alert-management/alerts/{alert_id}/comments", data=data
        )

    def get_comments(self, alert_id: str) -> List[dict[str, Any]]:
        """List all comments on an alert."""
        return self.client.request("GET", f"/api/v1/alert-management/alerts/{alert_id}/comments")

    async def aget_comments(self, alert_id: str) -> List[dict[str, Any]]:
        """Async version of get_comments()."""
        return await self.client.arequest(
            "GET", f"/api/v1/alert-management/alerts/{alert_id}/comments"
        )

    # ==================== Escalation Policy Management (SDK-only) ====================

    def create_escalation_policy(
        self,
        name: str,
        escalation_rules: List[dict[str, Any]],
        description: str | None = None,
        applies_to_severities: List[str] | None = None,
        applies_to_types: List[str] | None = None,
        applies_to_model_ids: List[str] | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        """Create an escalation policy for unacknowledged alerts.

        escalation_rules: list of {level (1-10), delay_minutes (>=1),
        notify_user_ids?, notify_channels?, message_template?}. The
        backend derives organization_id from the caller's session and
        overwrites whatever is sent, so this method doesn't expose it.
        """
        data = _escalation_policy_payload(
            name=name,
            escalation_rules=escalation_rules,
            description=description,
            applies_to_severities=applies_to_severities,
            applies_to_types=applies_to_types,
            applies_to_model_ids=applies_to_model_ids,
            is_active=is_active,
        )
        return self.client.request(
            "POST", "/api/v1/alert-management/escalation-policies", data=data
        )

    async def acreate_escalation_policy(
        self,
        name: str,
        escalation_rules: List[dict[str, Any]],
        description: str | None = None,
        applies_to_severities: List[str] | None = None,
        applies_to_types: List[str] | None = None,
        applies_to_model_ids: List[str] | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        """Async version of create_escalation_policy()."""
        data = _escalation_policy_payload(
            name=name,
            escalation_rules=escalation_rules,
            description=description,
            applies_to_severities=applies_to_severities,
            applies_to_types=applies_to_types,
            applies_to_model_ids=applies_to_model_ids,
            is_active=is_active,
        )
        return await self.client.arequest(
            "POST", "/api/v1/alert-management/escalation-policies", data=data
        )

    def list_escalation_policies(self, is_active: bool | None = None) -> List[dict[str, Any]]:
        """List escalation policies for the caller's organization."""
        params = _drop_none(is_active=is_active)
        return self.client.request(
            "GET", "/api/v1/alert-management/escalation-policies", params=params
        )

    async def alist_escalation_policies(
        self, is_active: bool | None = None
    ) -> List[dict[str, Any]]:
        """Async version of list_escalation_policies()."""
        params = _drop_none(is_active=is_active)
        return await self.client.arequest(
            "GET", "/api/v1/alert-management/escalation-policies", params=params
        )

    def get_escalation_policy(self, policy_id: str) -> dict[str, Any]:
        """Get a specific escalation policy."""
        return self.client.request(
            "GET", f"/api/v1/alert-management/escalation-policies/{policy_id}"
        )

    async def aget_escalation_policy(self, policy_id: str) -> dict[str, Any]:
        """Async version of get_escalation_policy()."""
        return await self.client.arequest(
            "GET", f"/api/v1/alert-management/escalation-policies/{policy_id}"
        )

    def update_escalation_policy(self, policy_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update an escalation policy. Any EscalationPolicyUpdate field may
        be passed as a kwarg; only supplied fields are changed."""
        return self.client.request(
            "PATCH",
            f"/api/v1/alert-management/escalation-policies/{policy_id}",
            data=kwargs,
        )

    async def aupdate_escalation_policy(self, policy_id: str, **kwargs: Any) -> dict[str, Any]:
        """Async version of update_escalation_policy()."""
        return await self.client.arequest(
            "PATCH",
            f"/api/v1/alert-management/escalation-policies/{policy_id}",
            data=kwargs,
        )

    def delete_escalation_policy(self, policy_id: str) -> dict[str, Any]:
        """Delete an escalation policy."""
        return self.client.request(
            "DELETE", f"/api/v1/alert-management/escalation-policies/{policy_id}"
        )

    async def adelete_escalation_policy(self, policy_id: str) -> dict[str, Any]:
        """Async version of delete_escalation_policy()."""
        return await self.client.arequest(
            "DELETE", f"/api/v1/alert-management/escalation-policies/{policy_id}"
        )

    # ==================== Maintenance ====================

    def process_pending_escalations(self) -> dict[str, Any]:
        """Manually trigger processing of pending escalations (normally a
        scheduled background task)."""
        return self.client.request(
            "POST", "/api/v1/alert-management/maintenance/process-escalations"
        )

    async def aprocess_pending_escalations(self) -> dict[str, Any]:
        """Async version of process_pending_escalations()."""
        return await self.client.arequest(
            "POST", "/api/v1/alert-management/maintenance/process-escalations"
        )

    def unsnooze_expired_alerts(self) -> dict[str, Any]:
        """Manually trigger reactivation of alerts whose snooze period has
        expired (normally a scheduled background task)."""
        return self.client.request("POST", "/api/v1/alert-management/maintenance/unsnooze-alerts")

    async def aunsnooze_expired_alerts(self) -> dict[str, Any]:
        """Async version of unsnooze_expired_alerts()."""
        return await self.client.arequest(
            "POST", "/api/v1/alert-management/maintenance/unsnooze-alerts"
        )


def _alert_time_series_query_payload(
    start: str,
    end: str,
    interval: str = "hour",
    group_by: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for
    AlertManagementResource.get_trend/aget_trend."""
    data: dict[str, Any] = {
        "time_range": {"start": start, "end": end},
        "interval": interval,
    }
    data.update(_drop_none(group_by=group_by, filters=filters))
    return data


def _escalation_policy_payload(
    name: str,
    escalation_rules: List[dict[str, Any]],
    description: str | None = None,
    applies_to_severities: List[str] | None = None,
    applies_to_types: List[str] | None = None,
    applies_to_model_ids: List[str] | None = None,
    is_active: bool = True,
) -> dict[str, Any]:
    """Shared request-body builder for
    AlertManagementResource.create_escalation_policy/acreate_escalation_policy.

    organization_id is required by the backend's Pydantic schema but is
    always overwritten server-side from the caller's session, so an empty
    placeholder is sent rather than exposing a meaningless parameter.
    """
    data: dict[str, Any] = {
        "organization_id": "",
        "name": name,
        "escalation_rules": escalation_rules,
        "is_active": is_active,
    }
    data.update(
        _drop_none(
            description=description,
            applies_to_severities=applies_to_severities,
            applies_to_types=applies_to_types,
            applies_to_model_ids=applies_to_model_ids,
        )
    )
    return data


class RiskRegisterResource(BaseResource):
    """AI Risk Register API resource (backed by backend/api/v1/risk_register.py).

    Read-only: the risk register is normally populated via the dashboard or
    auto-drafted from bias/drift/governance-review events server-side, not
    written to directly by SDK users.
    """

    def list(
        self,
        status: str | None = None,
        severity: str | None = None,
        category: str | None = None,
        model_id: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> dict[str, Any]:
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
        params: dict[str, Any] = {"limit": limit, "skip": skip}
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
        status: str | None = None,
        severity: str | None = None,
        category: str | None = None,
        model_id: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> dict[str, Any]:
        """Async version of list()."""
        params: dict[str, Any] = {"limit": limit, "skip": skip}
        for key, value in (
            ("status", status),
            ("severity", severity),
            ("category", category),
            ("model_id", model_id),
        ):
            if value is not None:
                params[key] = value
        return await self.client.arequest("GET", "/api/v1/risk-register", params=params)

    def get(self, entry_id: str) -> dict[str, Any]:
        """Get a single risk register entry by ID."""
        return self.client.request("GET", f"/api/v1/risk-register/{entry_id}")

    async def aget(self, entry_id: str) -> dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/risk-register/{entry_id}")

    def portfolio(self) -> dict[str, Any]:
        """Get the likelihood x impact heat map and top open risks."""
        return self.client.request("GET", "/api/v1/risk-register/portfolio")

    async def aportfolio(self) -> dict[str, Any]:
        """Async version of portfolio()."""
        return await self.client.arequest("GET", "/api/v1/risk-register/portfolio")


class GovernanceResource(BaseResource):
    """Governance review board API resource (backed by backend/api/v1/review_boards.py).

    Read-only surface for now: querying board/request/RACI state. Creating
    review requests, casting votes, etc. is an intentionally human/dashboard
    workflow, not exposed here.
    """

    def list_boards(self) -> List[dict[str, Any]]:
        """List governance review boards for the organization."""
        return self.client.request("GET", "/api/v1/governance/review-boards")

    async def alist_boards(self) -> List[dict[str, Any]]:
        """Async version of list_boards()."""
        return await self.client.arequest("GET", "/api/v1/governance/review-boards")

    def list_review_requests(
        self, board_id: str | None = None, status: str | None = None
    ) -> List[dict[str, Any]]:
        """List governance review requests, optionally filtered."""
        params: dict[str, Any] = {}
        if board_id is not None:
            params["board_id"] = board_id
        if status is not None:
            params["status"] = status
        return self.client.request(
            "GET", "/api/v1/governance/review-boards/requests", params=params
        )

    async def alist_review_requests(
        self, board_id: str | None = None, status: str | None = None
    ) -> List[dict[str, Any]]:
        """Async version of list_review_requests()."""
        params: dict[str, Any] = {}
        if board_id is not None:
            params["board_id"] = board_id
        if status is not None:
            params["status"] = status
        return await self.client.arequest(
            "GET", "/api/v1/governance/review-boards/requests", params=params
        )

    def get_review_request(self, request_id: str) -> dict[str, Any]:
        """Get a specific governance review request with full detail (board,
        submitter, comments, decisions, attachments, and vote_status).
        """
        return self.client.request("GET", f"/api/v1/governance/review-boards/requests/{request_id}")

    async def aget_review_request(self, request_id: str) -> dict[str, Any]:
        """Async version of get_review_request()."""
        return await self.client.arequest(
            "GET", f"/api/v1/governance/review-boards/requests/{request_id}"
        )

    def raci_grid(self, board_id: str | None = None) -> dict[str, Any]:
        """Get the RACI grid: review requests and their Responsible /
        Accountable / Consulted / Informed assignments, optionally scoped to
        one board.
        """
        params = {"board_id": board_id} if board_id is not None else {}
        return self.client.request(
            "GET", "/api/v1/governance/review-boards/raci-grid", params=params
        )

    async def araci_grid(self, board_id: str | None = None) -> dict[str, Any]:
        """Async version of raci_grid()."""
        params = {"board_id": board_id} if board_id is not None else {}
        return await self.client.arequest(
            "GET", "/api/v1/governance/review-boards/raci-grid", params=params
        )

    def submit_decision(
        self,
        request_id: str,
        decision: str,
        justification: str | None = None,
        conditions: str | None = None,
        *,
        evidence_snapshot_hash: str,
        totp_code: str | None = None,
    ) -> dict[str, Any]:
        """Submit a board member's decision on a governance review request.

        Requires a personal human session (JWT); requests authenticated with
        an organization API key (`wbx_live_...`) are rejected with 403 by
        `require_human_session`. The voter must also have two-factor
        authentication enabled and supply a current authenticator (or backup)
        code as `totp_code`; each code works once.

        `evidence_snapshot_hash` is the `evidence_snapshot_hash` of the request
        detail the voter was shown. The backend refuses the vote with 409 if the
        request has changed since.
        """
        data = _drop_none(
            decision=decision,
            justification=justification,
            conditions=conditions,
            evidence_snapshot_hash=evidence_snapshot_hash,
            totp_code=totp_code,
        )
        return self.client.request(
            "POST",
            f"/api/v1/governance/review-boards/requests/{request_id}/decisions",
            data=data,
        )

    async def asubmit_decision(
        self,
        request_id: str,
        decision: str,
        justification: str | None = None,
        conditions: str | None = None,
        *,
        evidence_snapshot_hash: str,
        totp_code: str | None = None,
    ) -> dict[str, Any]:
        """Async version of submit_decision()."""
        data = _drop_none(
            decision=decision,
            justification=justification,
            conditions=conditions,
            evidence_snapshot_hash=evidence_snapshot_hash,
            totp_code=totp_code,
        )
        return await self.client.arequest(
            "POST",
            f"/api/v1/governance/review-boards/requests/{request_id}/decisions",
            data=data,
        )


class LLMResource(BaseResource):
    """LLM monitoring API resource (backed by backend/api/v1/llm_monitoring.py)."""

    def log_call(
        self,
        provider: str,
        model_name: str,
        prompt: str,
        latency_ms: float,
        model_id: str | None = None,
        completion: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
        response_metadata: dict[str, Any] | None = None,
        messages: List[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        status: str | None = None,
        error_message: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        environment: str | None = None,
        prompt_version_id: str | None = None,
    ) -> dict[str, Any]:
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
            prompt_version_id: Governed prompt version ID (optional)

        Returns:
            {id, message} for the created log entry, or
            {"status": "queued", "operation_id": ...} when offline mode is
            enabled and the API couldn't be reached.
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
            prompt_version_id=prompt_version_id,
        )
        return self._request_or_queue(
            "POST",
            "/api/v1/llm/logs",
            operation_type="LLM_LOG",
            queue_data={"data": data},
            data=data,
        )

    async def alog_call(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of log_call()."""
        data = _llm_log_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm/logs", data=data)

    def log_calls_batch(self, logs: List[dict[str, Any]]) -> dict[str, Any]:
        """
        Log multiple LLM calls in one request (max 1000).

        Args:
            logs: List of dicts, each shaped like log_call()'s kwargs
                (provider, model_name, prompt, latency_ms required per entry)
        """
        return self.client.request("POST", "/api/v1/llm/logs/batch", data={"logs": logs})

    async def alog_calls_batch(self, logs: List[dict[str, Any]]) -> dict[str, Any]:
        """Async version of log_calls_batch()."""
        return await self.client.arequest("POST", "/api/v1/llm/logs/batch", data={"logs": logs})

    def get_stats(
        self,
        model_id: str | None = None,
        provider: str | None = None,
        environment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
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
        model_id: str | None = None,
        provider: str | None = None,
        environment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
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
        model_id: str | None = None,
        provider: str | None = None,
    ) -> List[dict[str, Any]]:
        """Get recent LLM logs (requires a model_id owned by the caller's org)."""
        params = _drop_none(limit=limit, model_id=model_id, provider=provider)
        return self.client.request("GET", "/api/v1/llm/logs/recent", params=params)

    async def aget_recent(
        self,
        limit: int = 100,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> List[dict[str, Any]]:
        """Async version of get_recent()."""
        params = _drop_none(limit=limit, model_id=model_id, provider=provider)
        return await self.client.arequest("GET", "/api/v1/llm/logs/recent", params=params)

    def get_log(self, log_id: str) -> dict[str, Any]:
        """Get a single LLM log by ID."""
        return self.client.request("GET", f"/api/v1/llm/logs/{log_id}")

    async def aget_log(self, log_id: str) -> dict[str, Any]:
        """Async version of get_log()."""
        return await self.client.arequest("GET", f"/api/v1/llm/logs/{log_id}")

    def query_logs(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
        min_cost: float | None = None,
        max_cost: float | None = None,
        min_latency_ms: float | None = None,
        max_latency_ms: float | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
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

    async def aquery_logs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of query_logs()."""
        data = _llm_query_logs_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm/logs/query", data=data)

    def session_logs(self, session_id: str) -> List[dict[str, Any]]:
        """Get all LLM logs for a specific session."""
        return self.client.request("GET", f"/api/v1/llm/logs/session/{session_id}")

    async def asession_logs(self, session_id: str) -> List[dict[str, Any]]:
        """Async version of session_logs()."""
        return await self.client.arequest("GET", f"/api/v1/llm/logs/session/{session_id}")

    def cost_breakdown(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        include_user: bool = False,
        include_environment: bool = False,
    ) -> dict[str, Any]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        include_user: bool = False,
        include_environment: bool = False,
    ) -> dict[str, Any]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        provider: str | None = None,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        provider: str | None = None,
    ) -> List[dict[str, Any]]:
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
        provider: str | None = None,
        model_name: str | None = None,
        model_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
    ) -> List[dict[str, Any]]:
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

    async def ausage_stats(self, *args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        """Async version of usage_stats()."""
        data = _llm_usage_stats_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm/usage-stats", data=data)

    def cost_threshold_alert(
        self, period_minutes: int = 60, threshold: float = 100.0
    ) -> dict[str, Any]:
        """Check whether costs over the recent period exceed a threshold."""
        params = {"period_minutes": period_minutes, "threshold": threshold}
        return self.client.request("GET", "/api/v1/llm/alerts/cost-threshold", params=params)

    async def acost_threshold_alert(
        self, period_minutes: int = 60, threshold: float = 100.0
    ) -> dict[str, Any]:
        """Async version of cost_threshold_alert()."""
        params = {"period_minutes": period_minutes, "threshold": threshold}
        return await self.client.arequest("GET", "/api/v1/llm/alerts/cost-threshold", params=params)

    def latency_threshold_alert(
        self, period_minutes: int = 60, threshold_ms: float = 5000.0
    ) -> dict[str, Any]:
        """Check whether average latency over the recent period exceeds a threshold."""
        params = {"period_minutes": period_minutes, "threshold_ms": threshold_ms}
        return self.client.request("GET", "/api/v1/llm/alerts/latency-threshold", params=params)

    async def alatency_threshold_alert(
        self, period_minutes: int = 60, threshold_ms: float = 5000.0
    ) -> dict[str, Any]:
        """Async version of latency_threshold_alert()."""
        params = {"period_minutes": period_minutes, "threshold_ms": threshold_ms}
        return await self.client.arequest(
            "GET", "/api/v1/llm/alerts/latency-threshold", params=params
        )

    def error_rate_alert(
        self, period_minutes: int = 60, threshold_percent: float = 10.0
    ) -> dict[str, Any]:
        """Check whether the error rate over the recent period exceeds a threshold."""
        params = {
            "period_minutes": period_minutes,
            "threshold_percent": threshold_percent,
        }
        return self.client.request("GET", "/api/v1/llm/alerts/error-rate", params=params)

    async def aerror_rate_alert(
        self, period_minutes: int = 60, threshold_percent: float = 10.0
    ) -> dict[str, Any]:
        """Async version of error_rate_alert()."""
        params = {
            "period_minutes": period_minutes,
            "threshold_percent": threshold_percent,
        }
        return await self.client.arequest("GET", "/api/v1/llm/alerts/error-rate", params=params)

    def cleanup_logs(self, days: int = 30) -> dict[str, Any]:
        """Delete LLM logs older than `days`. Admin-only on the backend."""
        params = {"days": days}
        return self.client.request("DELETE", "/api/v1/llm/logs/cleanup", params=params)

    async def acleanup_logs(self, days: int = 30) -> dict[str, Any]:
        """Async version of cleanup_logs()."""
        params = {"days": days}
        return await self.client.arequest("DELETE", "/api/v1/llm/logs/cleanup", params=params)


def _llm_query_logs_payload(
    start_time: str | None = None,
    end_time: str | None = None,
    model_id: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    status: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    environment: str | None = None,
    min_cost: float | None = None,
    max_cost: float | None = None,
    min_latency_ms: float | None = None,
    max_latency_ms: float | None = None,
    skip: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """Shared request-body builder for LLMResource.query_logs/aquery_logs."""
    data: dict[str, Any] = {"skip": skip, "limit": limit}
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
    provider: str | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
    user_id: str | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMResource.usage_stats/ausage_stats."""
    data: dict[str, Any] = {
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
    model_id: str | None = None,
    completion: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    finish_reason: str | None = None,
    response_metadata: dict[str, Any] | None = None,
    messages: List[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    status: str | None = None,
    error_message: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    environment: str | None = None,
    prompt_version_id: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMResource.log_call/alog_call."""
    data: dict[str, Any] = {
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
            prompt_version_id=prompt_version_id,
        )
    )
    return data


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    """Filter out None-valued kwargs, so optional fields aren't sent at all
    (letting the backend's own schema defaults apply) rather than sent as
    explicit nulls."""
    return {k: v for k, v in kwargs.items() if v is not None}


class RAGResource(BaseResource):
    """RAG pipeline monitoring API resource (backed by backend/api/v1/rag.py)."""

    def log_retrieval(
        self,
        query: str,
        results: List[dict[str, Any]],
        top_k: int,
        retrieval_method: str,
        query_type: str = "semantic",
        retrieval_params: dict[str, Any] | None = None,
        retrieval_latency_ms: float | None = None,
        answer: str | None = None,
        answer_latency_ms: float | None = None,
        ground_truth_ids: List[str] | None = None,
        llm_log_id: str | None = None,
        model_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    async def alog_retrieval(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of log_retrieval()."""
        data = _rag_retrieval_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/rag/retrievals", data=data)

    def create_evaluation(
        self,
        name: str,
        queries: List[str],
        description: str | None = None,
        evaluation_type: str = "manual",
        model_id: str | None = None,
        system_version: str | None = None,
        ground_truth: dict[str, list[str]] | None = None,
        environment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    async def acreate_evaluation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of create_evaluation()."""
        data = _rag_evaluation_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/rag/evaluations", data=data)

    def get_stats(
        self,
        model_id: str | None = None,
        environment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
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
        model_id: str | None = None,
        environment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
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
        model_id: str | None = None,
        evaluation_type: str | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
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
        model_id: str | None = None,
        evaluation_type: str | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Async version of list_evaluations()."""
        params = _drop_none(
            model_id=model_id,
            evaluation_type=evaluation_type,
            environment=environment,
            limit=limit,
            offset=offset,
        )
        return await self.client.arequest("GET", "/api/v1/rag/evaluations", params=params)

    def get_retrieval(self, log_id: str) -> dict[str, Any]:
        """Get a specific retrieval log by ID."""
        return self.client.request("GET", f"/api/v1/rag/retrievals/{log_id}")

    async def aget_retrieval(self, log_id: str) -> dict[str, Any]:
        """Async version of get_retrieval()."""
        return await self.client.arequest("GET", f"/api/v1/rag/retrievals/{log_id}")

    def list_retrievals(
        self,
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        retrieval_method: str | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        retrieval_method: str | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
        environment: str | None = None,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
        environment: str | None = None,
    ) -> List[dict[str, Any]]:
        """Async version of trends()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
            environment=environment,
        )
        return await self.client.arequest("GET", "/api/v1/rag/trends", params=params)

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        """Get a specific RAG evaluation by ID."""
        return self.client.request("GET", f"/api/v1/rag/evaluations/{evaluation_id}")

    async def aget_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        """Async version of get_evaluation()."""
        return await self.client.arequest("GET", f"/api/v1/rag/evaluations/{evaluation_id}")

    def metrics_precision(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        environment: str | None = None,
        k_values: List[int] | None = None,
    ) -> dict[str, float | None]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        environment: str | None = None,
        k_values: List[int] | None = None,
    ) -> dict[str, float | None]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, float]:
        """Get average relevance metrics (context/answer/faithfulness)."""
        params = _drop_none(start_time=start_time, end_time=end_time, model_id=model_id)
        return self.client.request("GET", "/api/v1/rag/metrics/relevance", params=params)

    async def ametrics_relevance(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, float]:
        """Async version of metrics_relevance()."""
        params = _drop_none(start_time=start_time, end_time=end_time, model_id=model_id)
        return await self.client.arequest("GET", "/api/v1/rag/metrics/relevance", params=params)


def _rag_retrieval_payload(
    query: str,
    results: List[dict[str, Any]],
    top_k: int,
    retrieval_method: str,
    query_type: str = "semantic",
    retrieval_params: dict[str, Any] | None = None,
    retrieval_latency_ms: float | None = None,
    answer: str | None = None,
    answer_latency_ms: float | None = None,
    ground_truth_ids: List[str] | None = None,
    llm_log_id: str | None = None,
    model_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    environment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for RAGResource.log_retrieval/alog_retrieval."""
    data: dict[str, Any] = {
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
    description: str | None = None,
    evaluation_type: str = "manual",
    model_id: str | None = None,
    system_version: str | None = None,
    ground_truth: dict[str, list[str]] | None = None,
    environment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for RAGResource.create_evaluation/acreate_evaluation."""
    data: dict[str, Any] = {
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
        llm_log_id: str | None = None,
        model_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
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

    async def aanalyze(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
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
        model_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
    ) -> List[dict[str, Any]]:
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

    async def aanalyze_batch(self, *args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        """Async version of analyze_batch()."""
        data = _safety_analyze_batch_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/safety/analyze/batch", data=data)

    def get_scores(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        safety_status: str | None = None,
        contains_pii: bool | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        safety_status: str | None = None,
        contains_pii: bool | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Async version of get_stats()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            environment=environment,
        )
        return await self.client.arequest("GET", "/api/v1/safety/stats", params=params)

    def get_score(self, score_id: str) -> dict[str, Any]:
        """Get a specific safety score by ID."""
        return self.client.request("GET", f"/api/v1/safety/scores/{score_id}")

    async def aget_score(self, score_id: str) -> dict[str, Any]:
        """Async version of get_score()."""
        return await self.client.arequest("GET", f"/api/v1/safety/scores/{score_id}")

    def trends(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
        environment: str | None = None,
    ) -> List[dict[str, Any]]:
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
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
        environment: str | None = None,
    ) -> List[dict[str, Any]]:
        """Async version of trends()."""
        params = _drop_none(
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            model_id=model_id,
            environment=environment,
        )
        return await self.client.arequest("GET", "/api/v1/safety/trends", params=params)

    def create_threshold(self, name: str, **kwargs: Any) -> dict[str, Any]:
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

    async def acreate_threshold(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Async version of create_threshold()."""
        data = {"name": name, **kwargs}
        return await self.client.arequest("POST", "/api/v1/safety/thresholds", data=data)

    def list_thresholds(
        self, model_id: str | None = None, is_active: bool | None = None
    ) -> List[dict[str, Any]]:
        """List safety thresholds for the caller's organization."""
        params = _drop_none(model_id=model_id, is_active=is_active)
        return self.client.request("GET", "/api/v1/safety/thresholds", params=params)

    async def alist_thresholds(
        self, model_id: str | None = None, is_active: bool | None = None
    ) -> List[dict[str, Any]]:
        """Async version of list_thresholds()."""
        params = _drop_none(model_id=model_id, is_active=is_active)
        return await self.client.arequest("GET", "/api/v1/safety/thresholds", params=params)

    def get_threshold(self, threshold_id: str) -> dict[str, Any]:
        """Get a specific safety threshold by ID."""
        return self.client.request("GET", f"/api/v1/safety/thresholds/{threshold_id}")

    async def aget_threshold(self, threshold_id: str) -> dict[str, Any]:
        """Async version of get_threshold()."""
        return await self.client.arequest("GET", f"/api/v1/safety/thresholds/{threshold_id}")

    def update_threshold(self, threshold_id: str, **kwargs: Any) -> dict[str, Any]:
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

    async def aupdate_threshold(self, threshold_id: str, **kwargs: Any) -> dict[str, Any]:
        """Async version of update_threshold()."""
        return await self.client.arequest(
            "PATCH", f"/api/v1/safety/thresholds/{threshold_id}", data=kwargs
        )

    def delete_threshold(self, threshold_id: str) -> dict[str, Any]:
        """Delete a safety threshold configuration."""
        return self.client.request("DELETE", f"/api/v1/safety/thresholds/{threshold_id}")

    async def adelete_threshold(self, threshold_id: str) -> dict[str, Any]:
        """Async version of delete_threshold()."""
        return await self.client.arequest("DELETE", f"/api/v1/safety/thresholds/{threshold_id}")


def _safety_analyze_payload(
    content: str,
    content_type: str = "text",
    check_toxicity: bool = True,
    check_pii: bool = True,
    check_harmful: bool = True,
    language: str = "en",
    llm_log_id: str | None = None,
    model_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for SafetyResource.analyze/aanalyze."""
    data: dict[str, Any] = {
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
    model_id: str | None = None,
    user_id: str | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for SafetyResource.analyze_batch/aanalyze_batch."""
    data: dict[str, Any] = {
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
        completion: str | None = None,
        model_id: str | None = None,
        layers: List[int] | None = None,
        aggregate_heads: bool = True,
        attention_weights: List[Any] | None = None,
        top_k: int = 10,
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
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

    async def aattention(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of attention()."""
        data = _llm_xai_attention_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/attention", data=data)

    def token_importance(
        self,
        prompt: str,
        completion: str,
        model_id: str | None = None,
        method: str = "perturbation",
        top_k: int = 10,
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
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

    async def atoken_importance(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of token_importance()."""
        data = _llm_xai_token_importance_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/token-importance", data=data)

    def prompt_sensitivity(
        self,
        prompt: str,
        baseline_output: str,
        model_id: str | None = None,
        num_perturbations: int = 10,
        perturbation_type: str = "mixed",
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
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

    async def aprompt_sensitivity(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of prompt_sensitivity()."""
        data = _llm_xai_sensitivity_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/sensitivity", data=data)

    def counterfactuals(
        self,
        prompt: str,
        original_output: str,
        target_change: str = "any",
        model_id: str | None = None,
        max_edits: int = 3,
        num_candidates: int = 10,
        search_strategy: str = "greedy",
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
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

    async def acounterfactuals(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of counterfactuals()."""
        data = _llm_xai_counterfactuals_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/counterfactuals", data=data)

    def debug_prompt(
        self,
        prompt: str,
        model_id: str | None = None,
        check_clarity: bool = True,
        check_specificity: bool = True,
        check_completeness: bool = True,
        check_consistency: bool = True,
        suggest_improvements: bool = True,
        llm_log_id: str | None = None,
        prompt_version_id: str | None = None,
    ) -> dict[str, Any]:
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
            prompt_version_id=prompt_version_id,
        )
        return self.client.request("POST", "/api/v1/llm-xai/debug-prompt", data=data)

    async def adebug_prompt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of debug_prompt()."""
        data = _llm_xai_debug_prompt_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/llm-xai/debug-prompt", data=data)

    def get_explanation(self, explanation_id: str) -> dict[str, Any]:
        """Get a specific attention/token-importance/sensitivity/counterfactual
        explanation by ID."""
        return self.client.request("GET", f"/api/v1/llm-xai/explanations/{explanation_id}")

    async def aget_explanation(self, explanation_id: str) -> dict[str, Any]:
        """Async version of get_explanation()."""
        return await self.client.arequest("GET", f"/api/v1/llm-xai/explanations/{explanation_id}")

    def get_prompt_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Get a specific prompt debug analysis by ID."""
        return self.client.request("GET", f"/api/v1/llm-xai/prompt-analyses/{analysis_id}")

    async def aget_prompt_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Async version of get_prompt_analysis()."""
        return await self.client.arequest("GET", f"/api/v1/llm-xai/prompt-analyses/{analysis_id}")

    def list_explanations(
        self,
        llm_log_id: str | None = None,
        explanation_type: str | None = None,
        model_id: str | None = None,
        limit: int = 50,
    ) -> List[dict[str, Any]]:
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
        llm_log_id: str | None = None,
        explanation_type: str | None = None,
        model_id: str | None = None,
        limit: int = 50,
    ) -> List[dict[str, Any]]:
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
        prompt: str | None = None,
        model_id: str | None = None,
        min_quality_score: float | None = None,
        limit: int = 50,
    ) -> List[dict[str, Any]]:
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
        prompt: str | None = None,
        model_id: str | None = None,
        min_quality_score: float | None = None,
        limit: int = 50,
    ) -> List[dict[str, Any]]:
        """Async version of list_prompt_analyses()."""
        params = _drop_none(
            prompt=prompt,
            model_id=model_id,
            min_quality_score=min_quality_score,
            limit=limit,
        )
        return await self.client.arequest("GET", "/api/v1/llm-xai/prompt-analyses", params=params)

    def stats(self, model_id: str | None = None, days: int = 7) -> dict[str, Any]:
        """Get aggregated LLM-XAI usage statistics."""
        params = _drop_none(model_id=model_id, days=days)
        return self.client.request("GET", "/api/v1/llm-xai/stats", params=params)

    async def astats(self, model_id: str | None = None, days: int = 7) -> dict[str, Any]:
        """Async version of stats()."""
        params = _drop_none(model_id=model_id, days=days)
        return await self.client.arequest("GET", "/api/v1/llm-xai/stats", params=params)

    def visualize_attention(self, explanation_id: str) -> dict[str, Any]:
        """Get attention-heatmap visualization data for an explanation."""
        return self.client.request("GET", f"/api/v1/llm-xai/attention/visualize/{explanation_id}")

    async def avisualize_attention(self, explanation_id: str) -> dict[str, Any]:
        """Async version of visualize_attention()."""
        return await self.client.arequest(
            "GET", f"/api/v1/llm-xai/attention/visualize/{explanation_id}"
        )

    def visualize_token_importance(self, explanation_id: str, top_k: int = 20) -> dict[str, Any]:
        """Get token-importance bar-chart visualization data for an explanation."""
        params = _drop_none(top_k=top_k)
        return self.client.request(
            "GET",
            f"/api/v1/llm-xai/token-importance/visualize/{explanation_id}",
            params=params,
        )

    async def avisualize_token_importance(
        self, explanation_id: str, top_k: int = 20
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    completion: str | None = None,
    model_id: str | None = None,
    layers: List[int] | None = None,
    aggregate_heads: bool = True,
    attention_weights: List[Any] | None = None,
    top_k: int = 10,
    llm_log_id: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.attention/aattention."""
    data: dict[str, Any] = {
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
    model_id: str | None = None,
    method: str = "perturbation",
    top_k: int = 10,
    llm_log_id: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.token_importance/atoken_importance."""
    data: dict[str, Any] = {
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
    model_id: str | None = None,
    num_perturbations: int = 10,
    perturbation_type: str = "mixed",
    llm_log_id: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.prompt_sensitivity/aprompt_sensitivity."""
    data: dict[str, Any] = {
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
    model_id: str | None = None,
    max_edits: int = 3,
    num_candidates: int = 10,
    search_strategy: str = "greedy",
    llm_log_id: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.counterfactuals/acounterfactuals."""
    data: dict[str, Any] = {
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
    model_id: str | None = None,
    check_clarity: bool = True,
    check_specificity: bool = True,
    check_completeness: bool = True,
    check_consistency: bool = True,
    suggest_improvements: bool = True,
    llm_log_id: str | None = None,
    prompt_version_id: str | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for LLMXAIResource.debug_prompt/adebug_prompt."""
    data: dict[str, Any] = {
        "prompt": prompt,
        "check_clarity": check_clarity,
        "check_specificity": check_specificity,
        "check_completeness": check_completeness,
        "check_consistency": check_consistency,
        "suggest_improvements": suggest_improvements,
    }
    data.update(
        _drop_none(
            model_id=model_id,
            llm_log_id=llm_log_id,
            prompt_version_id=prompt_version_id,
        )
    )
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
        meta_data: dict[str, Any] | None = None,
        tags: List[str] | None = None,
    ) -> dict[str, Any]:
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
        meta_data: dict[str, Any] | None = None,
        tags: List[str] | None = None,
    ) -> dict[str, Any]:
        """Async version of create_and_start()."""
        data = _drop_none(name=name, framework=framework, meta_data=meta_data, tags=tags)
        return await self.client.arequest("POST", "/api/v1/workflows/multi-agent/start", data=data)

    def start(self, workflow_id: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Mark a workflow as running. Call after registering agents/tasks."""
        data = _drop_none(inputs=inputs)
        return self.client.request(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/start", data=data
        )

    async def astart(
        self, workflow_id: str, inputs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Async version of start()."""
        data = _drop_none(inputs=inputs)
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/start", data=data
        )

    def complete(
        self,
        workflow_id: str,
        status: str,
        outputs: dict[str, Any] | None = None,
        error_message: str | None = None,
        trigger_analytics: bool = True,
    ) -> dict[str, Any]:
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
        outputs: dict[str, Any] | None = None,
        error_message: str | None = None,
        trigger_analytics: bool = True,
    ) -> dict[str, Any]:
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
        status: str | None = None,
        framework: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List workflows for the organization. Returns {workflows, total, skip, limit}."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(status=status, framework=framework))
        return self.client.request("GET", "/api/v1/workflows/multi-agent", params=params)

    async def alist(
        self,
        status: str | None = None,
        framework: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Async version of list()."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(status=status, framework=framework))
        return await self.client.arequest("GET", "/api/v1/workflows/multi-agent", params=params)

    def get(self, workflow_id: str) -> dict[str, Any]:
        """Get a workflow by ID."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}")

    async def aget(self, workflow_id: str) -> dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/workflows/multi-agent/{workflow_id}")

    def delete(self, workflow_id: str) -> dict[str, Any]:
        """Delete a workflow and all related agents/executions/tasks/interactions."""
        return self.client.request("DELETE", f"/api/v1/workflows/multi-agent/{workflow_id}")

    async def adelete(self, workflow_id: str) -> dict[str, Any]:
        """Async version of delete()."""
        return await self.client.arequest("DELETE", f"/api/v1/workflows/multi-agent/{workflow_id}")

    def register_agent(
        self,
        workflow_id: str,
        name: str,
        role: str | None = None,
        agent_type: str | None = None,
        model_name: str | None = None,
        llm_provider: str | None = None,
        llm_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        tools: List[str] | None = None,
        capabilities: dict[str, Any] | None = None,
        meta_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    async def aregister_agent(self, workflow_id: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of register_agent()."""
        data = _agent_register_payload(*args, **kwargs)
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/agents", data=data
        )

    def list_agents(self, workflow_id: str) -> List[dict[str, Any]]:
        """List all agents registered in a workflow."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}/agents")

    async def alist_agents(self, workflow_id: str) -> List[dict[str, Any]]:
        """Async version of list_agents()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/agents"
        )

    def create_execution(
        self,
        workflow_id: str,
        agent_id: str,
        inputs: dict[str, Any] | None = None,
        parent_execution_id: str | None = None,
        execution_order: int | None = None,
    ) -> dict[str, Any]:
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
        inputs: dict[str, Any] | None = None,
        parent_execution_id: str | None = None,
        execution_order: int | None = None,
    ) -> dict[str, Any]:
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
        from_agent_id: str | None = None,
        to_agent_id: str | None = None,
        from_execution_id: str | None = None,
        to_execution_id: str | None = None,
        message: str | None = None,
        meta_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        from_agent_id: str | None = None,
        to_agent_id: str | None = None,
        from_execution_id: str | None = None,
        to_execution_id: str | None = None,
        message: str | None = None,
        meta_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    def list_interactions(self, workflow_id: str) -> List[dict[str, Any]]:
        """List all agent-to-agent interactions in a workflow, chronologically."""
        return self.client.request(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/interactions"
        )

    async def alist_interactions(self, workflow_id: str) -> List[dict[str, Any]]:
        """Async version of list_interactions()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/interactions"
        )

    def create_task(
        self,
        workflow_id: str,
        task_name: str,
        description: str | None = None,
        task_type: str | None = None,
        expected_output: str | None = None,
        agent_id: str | None = None,
        parent_task_id: str | None = None,
        priority: int = 0,
        input_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    async def acreate_task(self, workflow_id: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of create_task()."""
        data = _agent_task_create_payload(*args, **kwargs)
        return await self.client.arequest(
            "POST", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks", data=data
        )

    def update_task_status(
        self,
        task_id: str,
        status: str,
        output_data: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
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
        output_data: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Async version of update_task_status()."""
        data = {"status": status}
        data.update(_drop_none(output_data=output_data, error_message=error_message))
        return await self.client.arequest(
            "PATCH", f"/api/v1/workflows/multi-agent/tasks/{task_id}", data=data
        )

    def list_tasks(self, workflow_id: str) -> List[dict[str, Any]]:
        """List all tasks in a workflow, with status, assignments, and results."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks")

    async def alist_tasks(self, workflow_id: str) -> List[dict[str, Any]]:
        """Async version of list_tasks()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/tasks"
        )

    def analytics(self, workflow_id: str) -> dict[str, Any]:
        """Get workflow analytics (tokens, costs, execution counts, durations).
        Calculated asynchronously after workflow completion."""
        return self.client.request("GET", f"/api/v1/workflows/multi-agent/{workflow_id}/analytics")

    async def aanalytics(self, workflow_id: str) -> dict[str, Any]:
        """Async version of analytics()."""
        return await self.client.arequest(
            "GET", f"/api/v1/workflows/multi-agent/{workflow_id}/analytics"
        )

    def cost_breakdown(self, workflow_id: str) -> dict[str, Any]:
        """Get per-agent cost breakdown (tokens, cost, execution count, avg
        duration). Computed via a Celery task the backend blocks on for up
        to 30s server-side, so this call uses an extended client timeout."""
        return self.client.request(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/cost-breakdown",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    async def acost_breakdown(self, workflow_id: str) -> dict[str, Any]:
        """Async version of cost_breakdown()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/cost-breakdown",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    def bottlenecks(self, workflow_id: str) -> dict[str, Any]:
        """Identify workflow bottlenecks: slowest agents, slowest tasks,
        failing agents. Computed via a Celery task the backend blocks on for
        up to 30s server-side, so this call uses an extended client timeout."""
        return self.client.request(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/bottlenecks",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    async def abottlenecks(self, workflow_id: str) -> dict[str, Any]:
        """Async version of bottlenecks()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/bottlenecks",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    def timeline(self, workflow_id: str) -> dict[str, Any]:
        """Get a chronological timeline of workflow events (start/end, agent
        executions, task updates, interactions). Computed via a Celery task
        the backend blocks on for up to 30s server-side, so this call uses
        an extended client timeout."""
        return self.client.request(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/timeline",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )

    async def atimeline(self, workflow_id: str) -> dict[str, Any]:
        """Async version of timeline()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/workflows/multi-agent/{workflow_id}/timeline",
            timeout=_AGENT_WORKFLOW_ANALYTICS_TIMEOUT_S,
        )


def _agent_register_payload(
    name: str,
    role: str | None = None,
    agent_type: str | None = None,
    model_name: str | None = None,
    llm_provider: str | None = None,
    llm_config: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    goal: str | None = None,
    backstory: str | None = None,
    tools: List[str] | None = None,
    capabilities: dict[str, Any] | None = None,
    meta_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for AgentWorkflowsResource.register_agent/aregister_agent."""
    data: dict[str, Any] = {"name": name}
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
    description: str | None = None,
    task_type: str | None = None,
    expected_output: str | None = None,
    agent_id: str | None = None,
    parent_task_id: str | None = None,
    priority: int = 0,
    input_data: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for AgentWorkflowsResource.create_task/acreate_task."""
    data: dict[str, Any] = {"task_name": task_name, "priority": priority}
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
        metric_metadata: dict[str, Any] | None = None,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
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

    async def acreate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async version of create()."""
        data = _metric_create_payload(*args, **kwargs)
        return await self.client.arequest("POST", "/api/v1/metrics/", data=data)

    def create_batch(self, model_id: str, metrics: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Create multiple metrics at once (max 100). Each dict in `metrics`
        needs metric_type/metric_name/metric_value."""
        data = {"model_id": model_id, "metrics": metrics}
        return self.client.request("POST", "/api/v1/metrics/batch", data=data)

    async def acreate_batch(
        self, model_id: str, metrics: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        """Async version of create_batch()."""
        data = {"model_id": model_id, "metrics": metrics}
        return await self.client.arequest("POST", "/api/v1/metrics/batch", data=data)

    def calculate_classification(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        y_pred_proba: List[float] | None = None,
        average: str = "binary",
        pos_label: int = 1,
    ) -> dict[str, float]:
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
        y_pred_proba: List[float] | None = None,
        average: str = "binary",
        pos_label: int = 1,
    ) -> dict[str, float]:
        """Async version of calculate_classification()."""
        params = {"average": average, "pos_label": pos_label}
        data = _drop_none(y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
        return await self.client.arequest(
            "POST", "/api/v1/metrics/calculate/classification", data=data, params=params
        )

    def calculate_regression(self, y_true: List[float], y_pred: List[float]) -> dict[str, float]:
        """Calculate regression metrics (MAE/MSE/RMSE/R2/etc.) from
        predictions, without persisting anything."""
        data = {"y_true": y_true, "y_pred": y_pred}
        return self.client.request("POST", "/api/v1/metrics/calculate/regression", data=data)

    async def acalculate_regression(
        self, y_true: List[float], y_pred: List[float]
    ) -> dict[str, float]:
        """Async version of calculate_regression()."""
        data = {"y_true": y_true, "y_pred": y_pred}
        return await self.client.arequest("POST", "/api/v1/metrics/calculate/regression", data=data)

    def get_model_metrics(
        self,
        model_id: str,
        metric_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
        """Get metrics recorded for a model, with optional type/date filters."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return self.client.request("GET", f"/api/v1/metrics/{model_id}", params=params)

    async def aget_model_metrics(
        self,
        model_id: str,
        metric_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
        """Async version of get_model_metrics()."""
        params = {"skip": skip, "limit": limit}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return await self.client.arequest("GET", f"/api/v1/metrics/{model_id}", params=params)

    def latest(self, model_id: str, metric_type: str) -> dict[str, Any]:
        """Get the most recent metric of a specific type for a model."""
        return self.client.request("GET", f"/api/v1/metrics/{model_id}/latest/{metric_type}")

    async def alatest(self, model_id: str, metric_type: str) -> dict[str, Any]:
        """Async version of latest()."""
        return await self.client.arequest("GET", f"/api/v1/metrics/{model_id}/latest/{metric_type}")

    def timeseries(
        self, model_id: str, metric_type: str, start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Get time series data for a metric over a required date range."""
        params = {"start_date": start_date, "end_date": end_date}
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/timeseries/{metric_type}", params=params
        )

    async def atimeseries(
        self, model_id: str, metric_type: str, start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Async version of timeseries()."""
        params = {"start_date": start_date, "end_date": end_date}
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/timeseries/{metric_type}", params=params
        )

    def aggregate(
        self,
        model_id: str,
        period: str,
        metric_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        force_recompute: bool = False,
    ) -> List[dict[str, Any]]:
        """Get metrics aggregated over `period` (e.g. "daily", "weekly").
        Defaults to the last 30 days if no date range is given."""
        params = {"period": period, "force_recompute": force_recompute}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return self.client.request("GET", f"/api/v1/metrics/{model_id}/aggregate", params=params)

    async def aaggregate(
        self,
        model_id: str,
        period: str,
        metric_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        force_recompute: bool = False,
    ) -> List[dict[str, Any]]:
        """Async version of aggregate()."""
        params = {"period": period, "force_recompute": force_recompute}
        params.update(_drop_none(metric_type=metric_type, start_date=start_date, end_date=end_date))
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/aggregate", params=params
        )

    def trend(self, model_id: str, metric_type: str, lookback_days: int = 30) -> dict[str, Any]:
        """Get trend statistics for a metric over `lookback_days`."""
        params = {"lookback_days": lookback_days}
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/trend/{metric_type}", params=params
        )

    async def atrend(
        self, model_id: str, metric_type: str, lookback_days: int = 30
    ) -> dict[str, Any]:
        """Async version of trend()."""
        params = {"lookback_days": lookback_days}
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/trend/{metric_type}", params=params
        )

    def rolling(self, model_id: str, metric_type: str, window_days: int = 7) -> dict[str, Any]:
        """Get rolling-window statistics for a metric over `window_days`."""
        params = {"window_days": window_days}
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/rolling/{metric_type}", params=params
        )

    async def arolling(
        self, model_id: str, metric_type: str, window_days: int = 7
    ) -> dict[str, Any]:
        """Async version of rolling()."""
        params = {"window_days": window_days}
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/rolling/{metric_type}", params=params
        )

    def summary(self, model_id: str, days: int = 30) -> dict[str, Any]:
        """Get a summary of all metrics for a model over the last `days`."""
        return self.client.request(
            "GET", f"/api/v1/metrics/{model_id}/summary", params={"days": days}
        )

    async def asummary(self, model_id: str, days: int = 30) -> dict[str, Any]:
        """Async version of summary()."""
        return await self.client.arequest(
            "GET", f"/api/v1/metrics/{model_id}/summary", params={"days": days}
        )

    def delete(self, model_id: str) -> dict[str, Any]:
        """Delete all metrics for a model."""
        return self.client.request("DELETE", f"/api/v1/metrics/{model_id}")

    async def adelete(self, model_id: str) -> dict[str, Any]:
        """Async version of delete()."""
        return await self.client.arequest("DELETE", f"/api/v1/metrics/{model_id}")


def _metric_create_payload(
    model_id: str,
    metric_type: str,
    metric_name: str,
    metric_value: float,
    metric_metadata: dict[str, Any] | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Shared request-body builder for MetricsResource.create/acreate."""
    data: dict[str, Any] = {
        "model_id": model_id,
        "metric_type": metric_type,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }
    data.update(_drop_none(metric_metadata=metric_metadata, sample_size=sample_size))
    return data


class ReportTypesResource(BaseResource):
    """Report Types API resource.

    Generates and lists compliance and governance reports via the
    Workitem 134 Phase 6 Report Engine.
    """

    def list(self) -> dict[str, Any]:
        """
        List all available report types for the organization.

        Returns:
            Dictionary containing available report types.
        """
        return self.client.request("GET", "/api/v1/report-types")

    async def alist(self) -> dict[str, Any]:
        """Async version of list()."""
        return await self.client.arequest("GET", "/api/v1/report-types")

    def generate(
        self,
        report_type_id: str,
        model_id: str | None = None,
        format: str = "pdf",
    ) -> dict[str, Any]:
        """
        Generate a compliance or governance report.

        Args:
            report_type_id: The ID of the report type (e.g., 'model_card', 'fairness_audit')
            model_id: The model ID for single-model reports (optional for org-wide reports)
            format: Output format ('pdf', 'html', or 'docx')

        Returns:
            Dictionary with generation result and download URL.
        """
        data = {"format": format}
        if model_id:
            data["model_id"] = model_id

        return self.client.request(
            "POST",
            f"/api/v1/report-types/{report_type_id}/generate",
            data=data,
        )

    async def agenerate(
        self,
        report_type_id: str,
        model_id: str | None = None,
        format: str = "pdf",
    ) -> dict[str, Any]:
        """Async version of generate()."""
        data = {"format": format}
        if model_id:
            data["model_id"] = model_id

        return await self.client.arequest(
            "POST",
            f"/api/v1/report-types/{report_type_id}/generate",
            data=data,
        )

    def preview(
        self,
        report_type_id: str,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Preview a generated report's content structure before rendering.

        Args:
            report_type_id: The ID of the report type
            model_id: The model ID (optional)

        Returns:
            Dictionary containing the preview document blocks.
        """
        data = {"report_type_id": report_type_id}
        if model_id:
            data["model_id"] = model_id

        return self.client.request(
            "POST",
            "/api/v1/export/templates/preview",
            data=data,
        )

    async def apreview(
        self,
        report_type_id: str,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of preview()."""
        data = {"report_type_id": report_type_id}
        if model_id:
            data["model_id"] = model_id

        return await self.client.arequest(
            "POST",
            "/api/v1/export/templates/preview",
            data=data,
        )


class PromptsResource(BaseResource):
    """Prompt lifecycle, deployment and drift governance API resource (backed by /api/v1/prompts)."""

    def list_templates(self) -> List[dict[str, Any]]:
        """List all prompt templates in the organization."""
        return self.client.request("GET", "/api/v1/prompts")

    async def alist_templates(self) -> List[dict[str, Any]]:
        """Async version of list_templates()."""
        return await self.client.arequest("GET", "/api/v1/prompts")

    def get_template(self, template_id: str) -> dict[str, Any]:
        """Get a specific prompt template by ID."""
        return self.client.request("GET", f"/api/v1/prompts/{template_id}")

    async def aget_template(self, template_id: str) -> dict[str, Any]:
        """Async version of get_template()."""
        return await self.client.arequest("GET", f"/api/v1/prompts/{template_id}")

    def create_template(
        self,
        name: str,
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new prompt template."""
        data: dict[str, Any] = {"name": name}
        if description is not None:
            data["description"] = description
        if tags is not None:
            data["tags"] = tags
        return self.client.request("POST", "/api/v1/prompts", data=data)

    async def acreate_template(
        self,
        name: str,
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of create_template()."""
        data: dict[str, Any] = {"name": name}
        if description is not None:
            data["description"] = description
        if tags is not None:
            data["tags"] = tags
        return await self.client.arequest("POST", "/api/v1/prompts", data=data)

    def list_versions(self, template_id: str) -> List[dict[str, Any]]:
        """List all versions of a prompt template."""
        return self.client.request("GET", f"/api/v1/prompts/{template_id}/versions")

    async def alist_versions(self, template_id: str) -> List[dict[str, Any]]:
        """Async version of list_versions()."""
        return await self.client.arequest("GET", f"/api/v1/prompts/{template_id}/versions")

    def get_version(self, template_id: str, version_id: str) -> dict[str, Any]:
        """Get a specific prompt version."""
        return self.client.request("GET", f"/api/v1/prompts/{template_id}/versions/{version_id}")

    async def aget_version(self, template_id: str, version_id: str) -> dict[str, Any]:
        """Async version of get_version()."""
        return await self.client.arequest(
            "GET", f"/api/v1/prompts/{template_id}/versions/{version_id}"
        )

    def create_version(
        self,
        template_id: str,
        user_message_template: str,
        system_message: str | None = None,
        model_name: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
        tool_schemas: dict[str, Any] | None = None,
        version_tag: str | None = None,
        commit_hash: str | None = None,
    ) -> dict[str, Any]:
        """Create a new version under a prompt template."""
        data: dict[str, Any] = {"user_message_template": user_message_template}
        data.update(
            _drop_none(
                system_message=system_message,
                model_name=model_name,
                model_id=model_id,
                temperature=temperature,
                tool_schemas=tool_schemas,
                version_tag=version_tag,
                commit_hash=commit_hash,
            )
        )
        return self.client.request("POST", f"/api/v1/prompts/{template_id}/versions", data=data)

    async def acreate_version(
        self,
        template_id: str,
        user_message_template: str,
        system_message: str | None = None,
        model_name: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
        tool_schemas: dict[str, Any] | None = None,
        version_tag: str | None = None,
        commit_hash: str | None = None,
    ) -> dict[str, Any]:
        """Async version of create_version()."""
        data: dict[str, Any] = {"user_message_template": user_message_template}
        data.update(
            _drop_none(
                system_message=system_message,
                model_name=model_name,
                model_id=model_id,
                temperature=temperature,
                tool_schemas=tool_schemas,
                version_tag=version_tag,
                commit_hash=commit_hash,
            )
        )
        return await self.client.arequest(
            "POST", f"/api/v1/prompts/{template_id}/versions", data=data
        )

    def advance_stage(
        self,
        template_id: str,
        version_id: str,
        target_stage: str,
        notes: str | None = None,
        verification_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Advance prompt version lifecycle stage (design -> verify -> deploy -> retired)."""
        data: dict[str, Any] = {"target_stage": target_stage}
        data.update(_drop_none(notes=notes, verification_run_id=verification_run_id))
        return self.client.request(
            "POST",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/advance-stage",
            data=data,
        )

    async def aadvance_stage(
        self,
        template_id: str,
        version_id: str,
        target_stage: str,
        notes: str | None = None,
        verification_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of advance_stage()."""
        data: dict[str, Any] = {"target_stage": target_stage}
        data.update(_drop_none(notes=notes, verification_run_id=verification_run_id))
        return await self.client.arequest(
            "POST",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/advance-stage",
            data=data,
        )

    def deploy(
        self,
        template_id: str,
        version_id: str,
        deployment_notes: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Deploy prompt version, enforcing single active deployment and verification gate."""
        data: dict[str, Any] = {"force": force}
        data.update(_drop_none(deployment_notes=deployment_notes))
        return self.client.request(
            "POST",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/deploy",
            data=data,
        )

    async def adeploy(
        self,
        template_id: str,
        version_id: str,
        deployment_notes: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Async version of deploy()."""
        data: dict[str, Any] = {"force": force}
        data.update(_drop_none(deployment_notes=deployment_notes))
        return await self.client.arequest(
            "POST",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/deploy",
            data=data,
        )

    def rollback(
        self,
        template_id: str,
        target_version_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Rollback prompt template to a previous version."""
        data: dict[str, Any] = {"reason": reason or "Emergency rollback"}
        data.update(_drop_none(target_version_id=target_version_id))
        return self.client.request(
            "POST",
            f"/api/v1/prompts/{template_id}/rollback",
            data=data,
        )

    async def arollback(
        self,
        template_id: str,
        target_version_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Async version of rollback()."""
        data: dict[str, Any] = {"reason": reason or "Emergency rollback"}
        data.update(_drop_none(target_version_id=target_version_id))
        return await self.client.arequest(
            "POST",
            f"/api/v1/prompts/{template_id}/rollback",
            data=data,
        )

    def compute_drift(
        self,
        template_id: str,
        version_id: str,
        baseline_dataset_id: str | None = None,
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Compute drift report for a prompt version against baseline dataset and recent logs."""
        data: dict[str, Any] = {"sample_size": sample_size}
        data.update(_drop_none(baseline_dataset_id=baseline_dataset_id))
        return self.client.request(
            "POST",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/drift-reports",
            data=data,
        )

    async def acompute_drift(
        self,
        template_id: str,
        version_id: str,
        baseline_dataset_id: str | None = None,
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Async version of compute_drift()."""
        data: dict[str, Any] = {"sample_size": sample_size}
        data.update(_drop_none(baseline_dataset_id=baseline_dataset_id))
        return await self.client.arequest(
            "POST",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/drift-reports",
            data=data,
        )

    def list_drift_reports(self, template_id: str, version_id: str) -> List[dict[str, Any]]:
        """List historical drift reports for a prompt version."""
        return self.client.request(
            "GET",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/drift-reports",
        )

    async def alist_drift_reports(self, template_id: str, version_id: str) -> List[dict[str, Any]]:
        """Async version of list_drift_reports()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/drift-reports",
        )

    def get_drift_report(self, template_id: str, version_id: str, report_id: str) -> dict[str, Any]:
        """Get a specific prompt drift report."""
        return self.client.request(
            "GET",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/drift-reports/{report_id}",
        )

    async def aget_drift_report(
        self, template_id: str, version_id: str, report_id: str
    ) -> dict[str, Any]:
        """Async version of get_drift_report()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/drift-reports/{report_id}",
        )

    def get_incidents(self, template_id: str, version_id: str) -> dict[str, Any]:
        """Get security incidents and alerts associated with a prompt version."""
        return self.client.request(
            "GET",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/incidents",
        )

    async def aget_incidents(self, template_id: str, version_id: str) -> dict[str, Any]:
        """Async version of get_incidents()."""
        return await self.client.arequest(
            "GET",
            f"/api/v1/prompts/{template_id}/versions/{version_id}/incidents",
        )


class ComplianceResource(BaseResource):
    """Compliance & ISO/IEC 42001 readiness API resource (backed by
    backend/api/v1/iso42001.py and backend/api/v1/control_catalog.py).
    """

    def iso42001_score(self) -> dict[str, Any]:
        """Get the organization's current ISO/IEC 42001 readiness score and
        clause-level breakdown (25 shipped controls across Clauses 4-10).
        """
        return self.client.request("GET", "/api/v1/compliance/iso42001/score")

    async def aiso42001_score(self) -> dict[str, Any]:
        """Async version of iso42001_score()."""
        return await self.client.arequest("GET", "/api/v1/compliance/iso42001/score")

    def iso42001_gaps(
        self,
        clause_filter: str | None = None,
        severity_filter: str | None = None,
    ) -> dict[str, Any]:
        """Get ISO/IEC 42001 compliance gaps, optionally filtered by clause or severity."""
        params = _drop_none(
            clause_filter=clause_filter,
            severity_filter=severity_filter,
        )
        return self.client.request("GET", "/api/v1/compliance/iso42001/gaps", params=params)

    async def aiso42001_gaps(
        self,
        clause_filter: str | None = None,
        severity_filter: str | None = None,
    ) -> dict[str, Any]:
        """Async version of iso42001_gaps()."""
        params = _drop_none(
            clause_filter=clause_filter,
            severity_filter=severity_filter,
        )
        return await self.client.arequest("GET", "/api/v1/compliance/iso42001/gaps", params=params)

    def list_frameworks(self) -> List[dict[str, Any]]:
        """List the platform's five in-scope compliance frameworks (ISO/IEC
        42001, GDPR, CCPA, EU AI Act, NIST AI RMF) and their adoption status.
        """
        return self.client.request("GET", "/api/v1/compliance/frameworks")

    async def alist_frameworks(self) -> List[dict[str, Any]]:
        """Async version of list_frameworks()."""
        return await self.client.arequest("GET", "/api/v1/compliance/frameworks")

    def get_framework_controls(self, framework: str) -> dict[str, Any]:
        """Get the control catalog and Statement of Applicability status for a framework."""
        return self.client.request("GET", f"/api/v1/compliance/frameworks/{framework}/controls")

    async def aget_framework_controls(self, framework: str) -> dict[str, Any]:
        """Async version of get_framework_controls()."""
        return await self.client.arequest(
            "GET", f"/api/v1/compliance/frameworks/{framework}/controls"
        )

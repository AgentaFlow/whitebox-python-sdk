"""
API Resources

Resource classes for interacting with different WhiteBoxXAI API endpoints.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from whiteboxxai.client import WhiteBoxXAI

logger = logging.getLogger(__name__)


class BaseResource:
    """Base class for API resources."""

    def __init__(self, client: "WhiteBoxXAI"):
        """Initialize resource with client."""
        self.client = client


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
        return self.client.request("POST", "/api/v1/models/", data=data)

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
        return self.client.request(
            "PATCH",
            f"/api/v1/models/{model_id}/baseline",
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
        return self.client.request("POST", "/api/v1/predictions/log", data=data)

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
        return self.client.request("POST", "/api/v1/predictions/log/batch", data=data)

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

    def get(self, explanation_id: int) -> Dict[str, Any]:
        """Get explanation by ID."""
        return self.client.request("GET", f"/api/v1/explanations/{explanation_id}")

    async def aget(self, explanation_id: int) -> Dict[str, Any]:
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
    """Alerts API resource.

    NOTE: as of this writing, backend/api/v1/alerts.py is not registered in
    backend/api/v1/__init__.py (SQLAlchemy model collisions with the
    canonical alert tables -- see the NOTE comment there), so every method
    on this resource currently calls a route that does not exist on a live
    WhiteBoxXAI backend. Left in place for interface stability; do not build
    new functionality on top of it until the backend router is wired in.
    """

    def create(
        self,
        name: str,
        alert_type: str,
        conditions: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create an alert rule.

        Args:
            name: Alert name
            alert_type: Alert type
            conditions: Alert conditions
            **kwargs: Additional alert configuration

        Returns:
            Created alert data
        """
        data = {
            "name": name,
            "alert_type": alert_type,
            "conditions": conditions,
            **kwargs,
        }
        return self.client.request("POST", "/api/v1/alerts", data=data)

    async def acreate(
        self,
        name: str,
        alert_type: str,
        conditions: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of create()."""
        data = {
            "name": name,
            "alert_type": alert_type,
            "conditions": conditions,
            **kwargs,
        }
        return await self.client.arequest("POST", "/api/v1/alerts", data=data)

    def list(self, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List alert rules."""
        params = {"model_id": model_id} if model_id else {}
        return self.client.request("GET", "/api/v1/alerts", params=params)

    async def alist(self, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Async version of list()."""
        params = {"model_id": model_id} if model_id else {}
        return await self.client.arequest("GET", "/api/v1/alerts", params=params)

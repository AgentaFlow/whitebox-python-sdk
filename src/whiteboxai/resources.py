"""
API Resources

Resource classes for interacting with different WhiteBoxAI API endpoints.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from explainai.client import WhiteBoxAI


class BaseResource:
    """Base class for API resources."""

    def __init__(self, client: "WhiteBoxAI"):
        """Initialize resource with client."""
        self.client = client


class ModelsResource(BaseResource):
    """Models API resource."""

    def register(
        self,
        name: str,
        model_type: str,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Register a new model.

        Args:
            name: Model name
            model_type: Model type (classification, regression, etc.)
            framework: ML framework (sklearn, pytorch, tensorflow, etc.)
            version: Model version
            **kwargs: Additional model metadata

        Returns:
            Registered model data
        """
        data = {
            "name": name,
            "model_type": model_type,
            "framework": framework,
            "version": version,
            **kwargs,
        }
        return self.client.request("POST", "/api/v1/models", data=data)

    async def aregister(
        self,
        name: str,
        model_type: str,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of register()."""
        data = {
            "name": name,
            "model_type": model_type,
            "framework": framework,
            "version": version,
            **kwargs,
        }
        return await self.client.arequest("POST", "/api/v1/models", data=data)

    def get(self, model_id: int) -> Dict[str, Any]:
        """Get model by ID."""
        return self.client.request("GET", f"/api/v1/models/{model_id}")

    async def aget(self, model_id: int) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest("GET", f"/api/v1/models/{model_id}")

    def list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List registered models."""
        params = {"limit": limit, "offset": offset}
        return self.client.request("GET", "/api/v1/models", params=params)

    async def alist(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Async version of list()."""
        params = {"limit": limit, "offset": offset}
        return await self.client.arequest("GET", "/api/v1/models", params=params)

    def update(self, model_id: int, **kwargs: Any) -> Dict[str, Any]:
        """Update model metadata."""
        return self.client.request("PUT", f"/api/v1/models/{model_id}", data=kwargs)

    async def aupdate(self, model_id: int, **kwargs: Any) -> Dict[str, Any]:
        """Async version of update()."""
        return await self.client.arequest(
            "PUT", f"/api/v1/models/{model_id}", data=kwargs
        )


class PredictionsResource(BaseResource):
    """Predictions API resource."""

    def log(
        self,
        model_id: int,
        inputs: Any,
        outputs: Any,
        explain: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a single prediction.

        Args:
            model_id: Model ID
            inputs: Input features/data
            outputs: Model prediction/output
            explain: Whether to generate explanation
            metadata: Additional prediction metadata

        Returns:
            Logged prediction data
        """
        data = {
            "model_id": model_id,
            "inputs": inputs,
            "outputs": outputs,
            "explain": explain,
            "metadata": metadata or {},
        }
        return self.client.request("POST", "/api/v1/predictions/log", data=data)

    async def alog(
        self,
        model_id: int,
        inputs: Any,
        outputs: Any,
        explain: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of log()."""
        data = {
            "model_id": model_id,
            "inputs": inputs,
            "outputs": outputs,
            "explain": explain,
            "metadata": metadata or {},
        }
        return await self.client.arequest("POST", "/api/v1/predictions/log", data=data)

    def log_batch(
        self,
        model_id: int,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Log multiple predictions in batch.

        Args:
            model_id: Model ID
            predictions: List of prediction dictionaries

        Returns:
            Batch logging result
        """
        data = {"model_id": model_id, "predictions": predictions}
        return self.client.request("POST", "/api/v1/predictions/log/batch", data=data)

    async def alog_batch(
        self,
        model_id: int,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Async version of log_batch()."""
        data = {"model_id": model_id, "predictions": predictions}
        return await self.client.arequest(
            "POST", "/api/v1/predictions/log/batch", data=data
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
        return await self.client.arequest(
            "POST", "/api/v1/explanations/generate", data=data
        )

    def get(self, explanation_id: int) -> Dict[str, Any]:
        """Get explanation by ID."""
        return self.client.request("GET", f"/api/v1/explanations/{explanation_id}")

    async def aget(self, explanation_id: int) -> Dict[str, Any]:
        """Async version of get()."""
        return await self.client.arequest(
            "GET", f"/api/v1/explanations/{explanation_id}"
        )


class DriftResource(BaseResource):
    """Drift detection API resource."""

    def detect(
        self,
        model_id: int,
        reference_data: Optional[Any] = None,
        current_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Detect drift for a model.

        Args:
            model_id: Model ID
            reference_data: Reference/baseline data
            current_data: Current data to compare
            **kwargs: Detection parameters

        Returns:
            Drift detection results
        """
        data = {
            "model_id": model_id,
            "reference_data": reference_data,
            "current_data": current_data,
            **kwargs,
        }
        return self.client.request("POST", "/api/v1/drift/detect", data=data)

    async def adetect(
        self,
        model_id: int,
        reference_data: Optional[Any] = None,
        current_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of detect()."""
        data = {
            "model_id": model_id,
            "reference_data": reference_data,
            "current_data": current_data,
            **kwargs,
        }
        return await self.client.arequest("POST", "/api/v1/drift/detect", data=data)

    def get_report(self, model_id: int, report_id: int) -> Dict[str, Any]:
        """Get drift report."""
        return self.client.request(
            "GET", f"/api/v1/drift/models/{model_id}/reports/{report_id}"
        )

    async def aget_report(self, model_id: int, report_id: int) -> Dict[str, Any]:
        """Async version of get_report()."""
        return await self.client.arequest(
            "GET", f"/api/v1/drift/models/{model_id}/reports/{report_id}"
        )


class AlertsResource(BaseResource):
    """Alerts API resource."""

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

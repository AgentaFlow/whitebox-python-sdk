"""
Model Monitoring

Simplified monitoring interface for ML models.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from whiteboxxai.client import WhiteBoxXAI


class ModelMonitor:
    """
    Simplified model monitoring interface.

    Example:
        ```python
        from whiteboxxai import WhiteBoxXAI, ModelMonitor

        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = ModelMonitor(client, model_id=123)

        # Log prediction
        monitor.log_prediction(
            inputs={"feature1": 1.0, "feature2": 2.0},
            output=0.85
        )

        # Log batch
        monitor.log_batch([
            {"inputs": {...}, "output": 0.85},
            {"inputs": {...}, "output": 0.92},
        ])
        ```
    """

    def __init__(
        self,
        client: "WhiteBoxXAI",
        model_id: Optional[int] = None,
        model_name: Optional[str] = None,
        auto_explain: bool = False,
        sampling_rate: float = 1.0,
        buffer_size: Optional[int] = None,
    ):
        """
        Initialize model monitor.

        Args:
            client: WhiteBoxXAI client instance
            model_id: Model ID (if already registered; can be set later via
                register_model())
            model_name: Model name (for registration)
            auto_explain: Automatically generate explanations
            sampling_rate: Prediction sampling rate (0.0-1.0)
            buffer_size: If set, log_prediction() buffers locally and sends
                predictions as a batch once the buffer reaches this size (or
                on flush()/context-manager exit), instead of sending
                immediately on every call.
        """
        self.client = client
        self.model_id = model_id
        self.model_name = model_name
        self.auto_explain = auto_explain
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._prediction_count = 0
        self._baseline_data: Optional[np.ndarray] = None

    def __enter__(self) -> "ModelMonitor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()

    def register_model(
        self,
        name: str,
        model_type: str,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Register a new model.

        Args:
            name: Model name
            model_type: Model type (classification, regression, etc.)
            framework: ML framework
            version: Model version
            **kwargs: Additional metadata

        Returns:
            Model ID
        """
        result = self.client.models.register(
            name=name,
            model_type=model_type,
            framework=framework,
            version=version,
            **kwargs,
        )
        self.model_id = result["id"]
        self.model_name = name
        return self.model_id

    async def aregister_model(
        self,
        name: str,
        model_type: str,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Async version of register_model()."""
        result = await self.client.models.aregister(
            name=name,
            model_type=model_type,
            framework=framework,
            version=version,
            **kwargs,
        )
        self.model_id = result["id"]
        self.model_name = name
        return self.model_id

    def log_prediction(
        self,
        inputs: Any,
        output: Any,
        explain: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Log a single prediction.

        Args:
            inputs: Input features/data
            output: Model prediction/output
            explain: Generate explanation (default: use auto_explain)
            metadata: Additional metadata

        Returns:
            Prediction data if sent immediately, None if buffered or skipped
        """
        if output is None:
            raise ValueError("output cannot be None.")

        if not self._should_sample():
            return None

        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        self._prediction_count += 1

        if self.buffer_size:
            self._buffer.append({"input_data": inputs, "output_data": output, "metadata": metadata})
            if len(self._buffer) >= self.buffer_size:
                self.flush()
            return None

        explain = explain if explain is not None else self.auto_explain

        result = self.client.predictions.log(
            model_id=self.model_id,
            input_data=inputs,
            output_data=output,
            metadata=metadata,
        )

        if explain and isinstance(result, dict) and result.get("id"):
            result["explanation"] = self.client.explanations.generate(prediction_id=result["id"])

        return result

    async def alog_prediction(
        self,
        inputs: Any,
        output: Any,
        explain: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of log_prediction()."""
        if output is None:
            raise ValueError("output cannot be None.")

        if not self._should_sample():
            return None

        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        self._prediction_count += 1

        if self.buffer_size:
            self._buffer.append({"input_data": inputs, "output_data": output, "metadata": metadata})
            if len(self._buffer) >= self.buffer_size:
                await self.aflush()
            return None

        explain = explain if explain is not None else self.auto_explain

        result = await self.client.predictions.alog(
            model_id=self.model_id,
            input_data=inputs,
            output_data=output,
            metadata=metadata,
        )

        if explain and isinstance(result, dict) and result.get("id"):
            result["explanation"] = await self.client.explanations.agenerate(
                prediction_id=result["id"]
            )

        return result

    @staticmethod
    def _normalize_batch_item(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Map the SDK's `inputs`/`output` shorthand keys to the
        `input_data`/`output_data` keys the predictions API expects."""
        if "inputs" in prediction or "output" in prediction:
            normalized = dict(prediction)
            if "inputs" in normalized:
                normalized["input_data"] = normalized.pop("inputs")
            if "output" in normalized:
                normalized["output_data"] = normalized.pop("output")
            return normalized
        return prediction

    def log_batch(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Log multiple predictions in batch.

        Args:
            predictions: List of prediction dictionaries (each with
                `inputs`/`output` or `input_data`/`output_data` keys)

        Returns:
            Batch logging result
        """
        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        # Apply sampling
        if self.sampling_rate < 1.0:
            predictions = self._sample_predictions(predictions)

        predictions = [self._normalize_batch_item(p) for p in predictions]

        return self.client.predictions.log_batch(
            model_id=self.model_id,
            predictions=predictions,
        )

    async def alog_batch(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Async version of log_batch()."""
        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        # Apply sampling
        if self.sampling_rate < 1.0:
            predictions = self._sample_predictions(predictions)

        predictions = [self._normalize_batch_item(p) for p in predictions]

        return await self.client.predictions.alog_batch(
            model_id=self.model_id,
            predictions=predictions,
        )

    def flush(self) -> Optional[Dict[str, Any]]:
        """Send any buffered predictions as a batch and clear the buffer."""
        if not self._buffer:
            return None
        predictions, self._buffer = self._buffer, []
        return self.client.predictions.log_batch(
            model_id=self.model_id,
            predictions=predictions,
        )

    async def aflush(self) -> Optional[Dict[str, Any]]:
        """Async version of flush()."""
        if not self._buffer:
            return None
        predictions, self._buffer = self._buffer, []
        return await self.client.predictions.alog_batch(
            model_id=self.model_id,
            predictions=predictions,
        )

    def get_prediction_count(self) -> int:
        """Return the number of predictions logged (sent or buffered) by
        this monitor instance so far."""
        return self._prediction_count

    def get_drift_reports(self, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """Get persisted drift reports for this model, most recent first."""
        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        return self.client.drift.get_reports(model_id=self.model_id, limit=limit, skip=skip)

    def create_alert_rule(
        self,
        metric: str,
        threshold: float,
        condition: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create an alert rule for this model.

        Args:
            metric: Metric to monitor (e.g. "accuracy")
            threshold: Threshold value that triggers the alert
            condition: Comparison condition (e.g. "below", "above")
            **kwargs: Additional alert configuration (name, etc.)

        Returns:
            Created alert rule data
        """
        name = kwargs.pop("name", f"{self.model_id}_{metric}_{condition}_{threshold}")
        return self.client.alerts.create(
            name=name,
            alert_type="threshold",
            conditions={"metric": metric, "threshold": threshold, "condition": condition},
            model_id=self.model_id,
            **kwargs,
        )

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get alert rules for this model."""
        return self.client.alerts.list(model_id=self.model_id)

    def set_baseline(self, data: np.ndarray) -> None:
        """
        Set baseline data for drift detection.

        Args:
            data: Baseline data array
        """
        self._baseline_data = data

    def detect_drift(
        self,
        current_data: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Detect drift for the model.

        Args:
            current_data: Current data to compare against baseline
            **kwargs: Additional detection parameters

        Returns:
            Drift detection results
        """
        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        return self.client.drift.detect(
            model_id=self.model_id,
            reference_data=self._baseline_data.tolist()
            if self._baseline_data is not None
            else None,
            current_data=current_data.tolist() if current_data is not None else None,
            **kwargs,
        )

    async def adetect_drift(
        self,
        current_data: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of detect_drift()."""
        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        return await self.client.drift.adetect(
            model_id=self.model_id,
            reference_data=self._baseline_data.tolist()
            if self._baseline_data is not None
            else None,
            current_data=current_data.tolist() if current_data is not None else None,
            **kwargs,
        )

    def _should_sample(self) -> bool:
        """Check if prediction should be sampled."""
        if self.sampling_rate >= 1.0:
            return True
        return np.random.random() < self.sampling_rate

    def _sample_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample predictions based on sampling rate."""
        n_samples = int(len(predictions) * self.sampling_rate)
        if n_samples == 0:
            n_samples = 1  # Always log at least one
        indices = np.random.choice(len(predictions), n_samples, replace=False)
        return [predictions[i] for i in indices]

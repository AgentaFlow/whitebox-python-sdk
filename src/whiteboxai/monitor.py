"""
Model Monitoring

Simplified monitoring interface for ML models.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from whiteboxai.client import WhiteBoxAI


class ModelMonitor:
    """
    Simplified model monitoring interface.

    Example:
        ```python
        from whiteboxai import WhiteBoxAI, ModelMonitor

        client = WhiteBoxAI(api_key="your-api-key")
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
        client: "WhiteBoxAI",
        model_id: Optional[int] = None,
        model_name: Optional[str] = None,
        auto_explain: bool = False,
        sampling_rate: float = 1.0,
    ):
        """
        Initialize model monitor.

        Args:
            client: WhiteBoxAI client instance
            model_id: Model ID (if already registered)
            model_name: Model name (for registration)
            auto_explain: Automatically generate explanations
            sampling_rate: Prediction sampling rate (0.0-1.0)
        """
        self.client = client
        self.model_id = model_id
        self.model_name = model_name
        self.auto_explain = auto_explain
        self.sampling_rate = sampling_rate
        self._baseline_data: Optional[np.ndarray] = None

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
            Prediction data if sampled, None if skipped
        """
        if not self._should_sample():
            return None

        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        explain = explain if explain is not None else self.auto_explain

        return self.client.predictions.log(
            model_id=self.model_id,
            inputs=inputs,
            outputs=output,
            explain=explain,
            metadata=metadata,
        )

    async def alog_prediction(
        self,
        inputs: Any,
        output: Any,
        explain: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of log_prediction()."""
        if not self._should_sample():
            return None

        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        explain = explain if explain is not None else self.auto_explain

        return await self.client.predictions.alog(
            model_id=self.model_id,
            inputs=inputs,
            outputs=output,
            explain=explain,
            metadata=metadata,
        )

    def log_batch(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Log multiple predictions in batch.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Batch logging result
        """
        if self.model_id is None:
            raise ValueError("Model not registered. Call register_model() first.")

        # Apply sampling
        if self.sampling_rate < 1.0:
            predictions = self._sample_predictions(predictions)

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

        return await self.client.predictions.alog_batch(
            model_id=self.model_id,
            predictions=predictions,
        )

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
            reference_data=(
                self._baseline_data.tolist() if self._baseline_data is not None else None
            ),
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
            reference_data=(
                self._baseline_data.tolist() if self._baseline_data is not None else None
            ),
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

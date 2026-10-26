"""
PyTorch Integration

Integration for monitoring PyTorch models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

    class _NNStub:
        """Stand-in for `torch.nn` so `nn.Module` stays resolvable (as a type
        hint and as a base class) when PyTorch isn't installed."""

        Module = object

    nn = _NNStub()


from whiteboxxai.integrations._registration import positional_feature_names, registration_fields
from whiteboxxai.monitor import ModelMonitor


class TorchMonitor(ModelMonitor):
    """
    Monitor for PyTorch models.

    Example:
        ```python
        import torch
        import torch.nn as nn
        from whiteboxxai import WhiteBoxXAI
        from whiteboxxai.integrations.pytorch import TorchMonitor

        # Define model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Setup monitoring
        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = TorchMonitor(client, model=model)
        monitor.register_from_model(
            version="1.0.0", baseline_metrics={"accuracy": 0.94}
        )

        # Wrap model for automatic monitoring
        monitored_model = monitor.wrap_model(model)

        # Predictions are automatically logged
        outputs = monitored_model(inputs)
        ```
    """

    def __init__(self, client, model: nn.Module | None = None, **kwargs):
        """Initialize PyTorch monitor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

        super().__init__(client, **kwargs)
        self.model = model

    def register_from_model(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        features: list[str] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> int:
        """
        Register model from PyTorch module.

        Args:
            name: Model name (default: class name)
            model_type: One of the API's model types (default
                ``neural_network``). A task name such as ``"classification"`` is
                kept as ``task`` metadata.
            version: Model version (default ``1.0.0``)
            features: Feature names (default: ``feature_0..feature_{n-1}``, with
                n read from the first layer's ``in_features``; required when the
                module has no such layer)
            baseline_metrics: Baseline metrics, e.g. ``{"accuracy": 0.94}``
            **kwargs: Additional metadata

        Returns:
            Model ID

        Raises:
            ValueError: No model, an invalid ``model_type``, or no feature names
                could be determined.
        """
        name, fields, metadata = self._prepare_registration(
            name, model_type, version, features, baseline_metrics, kwargs
        )
        return self.register_model(name=name, framework="pytorch", **fields, **metadata)

    async def aregister_from_model(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        features: list[str] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> int:
        """Async version of register_from_model()."""
        name, fields, metadata = self._prepare_registration(
            name, model_type, version, features, baseline_metrics, kwargs
        )
        return await self.aregister_model(name=name, framework="pytorch", **fields, **metadata)

    def _prepare_registration(
        self,
        name: str | None,
        model_type: str | None,
        version: str | None,
        features: list[str] | None,
        baseline_metrics: dict[str, Any] | None,
        extra_metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Resolve (name, API-required fields, metadata) for registration."""
        if self.model is None:
            raise ValueError("No model provided")

        if name is None:
            name = self.model.__class__.__name__

        metadata = self._extract_model_metadata()
        metadata.update(extra_metadata)

        input_width = next(
            (
                module.in_features
                for module in self.model.modules()
                if isinstance(getattr(module, "in_features", None), int)
            ),
            None,
        )
        fields = registration_fields(
            model_type=model_type,
            default_model_type="neural_network",
            version=version,
            features=features,
            inferred_features=(positional_feature_names(input_width) if input_width else None),
            baseline_metrics=baseline_metrics,
            metadata=metadata,
        )
        return name, fields, metadata

    def wrap_model(self, model: nn.Module) -> TorchWrapper:
        """
        Wrap PyTorch model for automatic monitoring.

        Args:
            model: PyTorch module

        Returns:
            Wrapped model
        """
        return TorchWrapper(model, self)

    def _extract_model_metadata(self) -> dict[str, Any]:
        """Extract metadata from PyTorch model."""
        metadata = {}

        if self.model is None:
            return metadata

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        metadata["total_parameters"] = total_params
        metadata["trainable_parameters"] = trainable_params

        # Get model architecture
        metadata["architecture"] = str(self.model)

        # Get PyTorch version
        metadata["pytorch_version"] = torch.__version__

        # Get device
        device = next(self.model.parameters()).device
        metadata["device"] = str(device)

        return metadata


class TorchWrapper(nn.Module):
    """
    Wrapper for PyTorch models with automatic monitoring.

    This wrapper intercepts forward calls and logs predictions to WhiteBoxXAI.
    """

    def __init__(self, model: nn.Module, monitor: TorchMonitor):
        """Initialize wrapper."""
        super().__init__()
        self.model = model
        self.monitor = monitor

    def forward(self, x: torch.Tensor, log: bool = True) -> torch.Tensor:
        """Forward pass with automatic logging."""
        # Call original forward
        output = self.model(x)

        # Log predictions if enabled
        if log:
            self._log_batch_predictions(x, output)

        return output

    def _log_batch_predictions(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        """Log batch of predictions."""
        # Convert to numpy/lists
        inputs_np = inputs.detach().cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()

        # Create predictions list
        predictions = []
        for i in range(len(inputs_np)):
            pred = {
                "inputs": inputs_np[i].tolist(),
                "output": outputs_np[i].tolist(),
            }
            predictions.append(pred)

        # Log batch
        try:
            self.monitor.log_batch(predictions)
        except Exception as e:
            # Don't fail prediction if logging fails
            print(f"Warning: Failed to log predictions: {e}")

    def __getattr__(self, name: str):
        """Delegate attributes to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def monitor_forward(monitor: TorchMonitor, input_extractor: Callable | None = None):
    """
    Decorator to monitor PyTorch forward passes.

    Args:
        monitor: TorchMonitor instance
        input_extractor: Function to extract inputs from args

    Example:
        ```python
        from whiteboxxai import WhiteBoxXAI
        from whiteboxxai.integrations.pytorch import TorchMonitor, monitor_forward

        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = TorchMonitor(client, model_id=123)

        class MyModel(nn.Module):
            @monitor_forward(monitor)
            def forward(self, x):
                return self.layers(x)
        ```
    """

    def decorator(forward_fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Call original forward
            output = forward_fn(self, *args, **kwargs)

            # Extract inputs
            if input_extractor:
                inputs = input_extractor(args, kwargs)
            else:
                inputs = args[0] if args else None

            # Log prediction
            if TORCH_AVAILABLE and inputs is not None and isinstance(inputs, torch.Tensor):
                inputs_np = inputs.detach().cpu().numpy()
                outputs_np = output.detach().cpu().numpy()

                try:
                    monitor.log_prediction(
                        inputs=inputs_np.tolist(),
                        output=outputs_np.tolist(),
                    )
                except Exception as e:
                    print(f"Warning: Failed to log prediction: {e}")

            return output

        return wrapper

    return decorator

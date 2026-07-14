"""
Monitoring Decorators

Decorators for automatic model and prediction monitoring.
"""

import functools
import inspect
import time
from typing import Any, Callable, Dict, Optional

from whiteboxxai.monitor import ModelMonitor


def monitor_model(
    monitor: ModelMonitor,
    input_keys: Optional[list] = None,
    output_key: Optional[str] = None,
    explain: bool = False,
):
    """
    Decorator to automatically monitor model predictions.

    Args:
        monitor: ModelMonitor instance
        input_keys: Keys to extract inputs from function args/kwargs
        output_key: Key to extract output from function return value
        explain: Generate explanations for predictions

    Example:
        ```python
        from whiteboxxai import WhiteBoxXAI, ModelMonitor, monitor_model

        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = ModelMonitor(client, model_id=123)

        @monitor_model(monitor, input_keys=["features"], explain=True)
        def predict(features):
            # Your prediction logic
            return model.predict(features)

        # Predictions are automatically logged
        result = predict(features=[1.0, 2.0, 3.0])
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Call original function
            result = func(*args, **kwargs)

            # Extract inputs
            inputs = _extract_inputs(func, args, kwargs, input_keys)

            # Extract output
            output = _extract_output(result, output_key)

            # Calculate inference time
            inference_time = time.time() - start_time

            # Log prediction
            metadata = {"inference_time_ms": inference_time * 1000}
            monitor.log_prediction(
                inputs=inputs,
                output=output,
                explain=explain,
                metadata=metadata,
            )

            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            # Call original function
            result = await func(*args, **kwargs)

            # Extract inputs
            inputs = _extract_inputs(func, args, kwargs, input_keys)

            # Extract output
            output = _extract_output(result, output_key)

            # Calculate inference time
            inference_time = time.time() - start_time

            # Log prediction
            metadata = {"inference_time_ms": inference_time * 1000}
            await monitor.alog_prediction(
                inputs=inputs,
                output=output,
                explain=explain,
                metadata=metadata,
            )

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def monitor_prediction(
    monitor: ModelMonitor,
    input_extractor: Optional[Callable] = None,
    output_extractor: Optional[Callable] = None,
    explain: bool = False,
    metadata_extractor: Optional[Callable] = None,
):
    """
    Decorator to monitor individual predictions with custom extractors.

    Args:
        monitor: ModelMonitor instance
        input_extractor: Function to extract inputs from args/kwargs
        output_extractor: Function to extract output from result
        explain: Generate explanations for predictions
        metadata_extractor: Function to extract additional metadata

    Example:
        ```python
        from whiteboxxai import WhiteBoxXAI, ModelMonitor, monitor_prediction

        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = ModelMonitor(client, model_id=123)

        def extract_inputs(args, kwargs):
            return {"features": kwargs.get("data")}

        def extract_output(result):
            return result["prediction"]

        @monitor_prediction(
            monitor,
            input_extractor=extract_inputs,
            output_extractor=extract_output,
            explain=True
        )
        def predict(data):
            # Your prediction logic
            return {"prediction": 0.85, "confidence": 0.92}

        result = predict(data=[1.0, 2.0, 3.0])
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Call original function
            result = func(*args, **kwargs)

            # Extract inputs
            if input_extractor:
                inputs = input_extractor(args, kwargs)
            else:
                inputs = _default_input_extractor(func, args, kwargs)

            # Extract output
            if output_extractor:
                output = output_extractor(result)
            else:
                output = result

            # Extract metadata
            metadata = {"inference_time_ms": (time.time() - start_time) * 1000}
            if metadata_extractor:
                custom_metadata = metadata_extractor(args, kwargs, result)
                metadata.update(custom_metadata)

            # Log prediction
            monitor.log_prediction(
                inputs=inputs,
                output=output,
                explain=explain,
                metadata=metadata,
            )

            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            # Call original function
            result = await func(*args, **kwargs)

            # Extract inputs
            if input_extractor:
                inputs = input_extractor(args, kwargs)
            else:
                inputs = _default_input_extractor(func, args, kwargs)

            # Extract output
            if output_extractor:
                output = output_extractor(result)
            else:
                output = result

            # Extract metadata
            metadata = {"inference_time_ms": (time.time() - start_time) * 1000}
            if metadata_extractor:
                custom_metadata = metadata_extractor(args, kwargs, result)
                metadata.update(custom_metadata)

            # Log prediction
            await monitor.alog_prediction(
                inputs=inputs,
                output=output,
                explain=explain,
                metadata=metadata,
            )

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def _extract_inputs(
    func: Callable,
    args: tuple,
    kwargs: dict,
    input_keys: Optional[list] = None,
) -> Any:
    """Extract inputs from function arguments."""
    if input_keys:
        # Extract specified keys
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        inputs = {}
        for key in input_keys:
            if key in bound_args.arguments:
                inputs[key] = bound_args.arguments[key]

        return inputs
    else:
        # Use all arguments
        return _default_input_extractor(func, args, kwargs)


def _default_input_extractor(
    func: Callable, args: tuple, kwargs: dict
) -> Dict[str, Any]:
    """Default input extractor."""
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)


def _extract_output(result: Any, output_key: Optional[str] = None) -> Any:
    """Extract output from function result."""
    if output_key and isinstance(result, dict):
        return result.get(output_key)
    return result


def monitor_performance(threshold_ms: Optional[float] = None):
    """
    Decorator to monitor function performance.

    Args:
        threshold_ms: Optional performance threshold in milliseconds

    Example:
        ```python
        from whiteboxxai.decorators import monitor_performance

        @monitor_performance(threshold_ms=1000)
        def slow_function():
            # Your code here
            pass
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

            if threshold_ms and elapsed_ms > threshold_ms:
                print(
                    f"Warning: {func.__name__} took {elapsed_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )

            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

            if threshold_ms and elapsed_ms > threshold_ms:
                print(
                    f"Warning: {func.__name__} took {elapsed_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )

            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator

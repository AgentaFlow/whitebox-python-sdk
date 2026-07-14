"""
SDK Utilities

General utility functions for the WhiteBoxAI SDK.
"""

import json
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


def serialize_data(data: Any) -> Any:
    """
    Serialize data for API transmission.

    Handles numpy arrays, pandas DataFrames, and other common data types.

    Args:
        data: Data to serialize

    Returns:
        JSON-serializable data
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    elif isinstance(data, pd.Series):
        return data.to_dict()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]
    else:
        return data


def validate_model_type(model_type: str) -> bool:
    """
    Validate model type.

    Args:
        model_type: Model type string

    Returns:
        True if valid, False otherwise
    """
    valid_types = {
        "classification",
        "regression",
        "clustering",
        "ranking",
        "recommendation",
        "anomaly_detection",
        "time_series",
        "nlp",
        "computer_vision",
        "other",
    }
    return model_type.lower() in valid_types


def extract_feature_names(data: Union[np.ndarray, pd.DataFrame, Dict]) -> List[str]:
    """
    Extract feature names from data.

    Args:
        data: Input data

    Returns:
        List of feature names
    """
    if isinstance(data, pd.DataFrame):
        return data.columns.tolist()
    elif isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, np.ndarray):
        # Generate default feature names
        n_features = data.shape[1] if len(data.shape) > 1 else 1
        return [f"feature_{i}" for i in range(n_features)]
    else:
        return []


def format_bytes(size: int) -> str:
    """
    Format byte size as human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def dict_to_query_params(params: Dict[str, Any]) -> str:
    """
    Convert dictionary to URL query parameters.

    Args:
        params: Parameter dictionary

    Returns:
        Query string
    """
    parts = []
    for key, value in params.items():
        if value is not None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    parts.append(f"{key}={item}")
            else:
                parts.append(f"{key}={value}")
    return "&".join(parts)


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely load JSON with fallback.

    Args:
        text: JSON string
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def is_numeric(value: Any) -> bool:
    """
    Check if value is numeric.

    Args:
        value: Value to check

    Returns:
        True if numeric, False otherwise
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def validate_features(features: Any) -> bool:
    """
    Validate features for model input.

    Args:
        features: Features to validate

    Returns:
        True if valid

    Raises:
        ValueError: If features are invalid
    """
    if features is None:
        raise ValueError("Features cannot be None")

    if isinstance(features, (list, tuple)):
        if len(features) == 0:
            raise ValueError("Features cannot be empty")
        return True
    elif isinstance(features, np.ndarray):
        if features.size == 0:
            raise ValueError("Features cannot be empty")
        return True
    elif isinstance(features, dict):
        if len(features) == 0:
            raise ValueError("Features cannot be empty")
        return True
    else:
        raise ValueError(f"Unsupported features type: {type(features)}")


def format_features(features: Any) -> Dict[str, Any]:
    """
    Format features for API transmission.

    Args:
        features: Features to format

    Returns:
        Formatted features dictionary
    """
    if isinstance(features, dict):
        return features
    elif isinstance(features, (list, tuple, np.ndarray)):
        return {"features": serialize_data(features)}
    else:
        return {"features": features}


def validate_prediction(prediction: Any) -> bool:
    """
    Validate model prediction.

    Args:
        prediction: Prediction to validate

    Returns:
        True if valid

    Raises:
        ValueError: If prediction is invalid
    """
    if prediction is None:
        raise ValueError("Prediction cannot be None")
    return True


def serialize_numpy(arr: np.ndarray) -> List:
    """
    Serialize numpy array to list.

    Args:
        arr: Numpy array

    Returns:
        List representation
    """
    return arr.tolist()


def deserialize_numpy(data: List) -> np.ndarray:
    """
    Deserialize list to numpy array.

    Args:
        data: List data

    Returns:
        Numpy array
    """
    return np.array(data)


def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """
    Compute basic metrics for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Basic accuracy for classification
    if y_true.dtype == y_pred.dtype and np.issubdtype(y_true.dtype, np.integer):
        metrics["accuracy"] = float(np.mean(y_true == y_pred))
    else:
        # MSE for regression
        metrics["mse"] = float(np.mean((y_true - y_pred) ** 2))
        metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

    return metrics


def batch_iterator(iterable: Any, batch_size: int):
    """
    Iterate over data in batches.

    Args:
        iterable: Data to iterate over
        batch_size: Size of each batch

    Yields:
        Batches of data
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

"""
Registration payload rules shared by the framework integrations.

The backend's ``ModelCreate`` schema (``backend/schemas/model.py``) requires
``name``, ``version``, ``model_type``, ``features`` and ``baseline_metrics``, and
``model_type`` is a strict enum (``backend/models/database.py::ModelType``), not
free text. Helpers that send ``model_type="classification"`` or omit the other
fields are rejected with a 422, so every ``register_from_model`` /
``register_application`` goes through :func:`registration_fields` to build a
payload the API accepts, or to fail fast with a message saying what to pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Mirrors backend.models.database.ModelType. Keep in sync with the backend.
VALID_MODEL_TYPES = (
    "xgboost",
    "random_forest",
    "neural_network",
    "linear_model",
    "llm",
    "custom",
)

# Earlier SDK releases took a *task* ("classification", "regression", ...) in the
# ``model_type`` argument. The API never accepted those, but callers wrote them,
# so they are still allowed there: the value is kept as ``metadata["task"]`` and
# the framework's default ModelType is used instead.
TASK_NAMES = frozenset(
    {"classification", "regression", "ranking", "clustering", "qa", "generation"}
)

# ``version`` is required by the API but a helper often has nothing better to
# offer than the conventional first version.
DEFAULT_VERSION = "1.0.0"


def positional_feature_names(count: int) -> list[str]:
    """Names for ``count`` unnamed columns (``feature_0``, ``feature_1``, ...)."""
    return [f"feature_{i}" for i in range(count)]


def registration_fields(
    *,
    model_type: str | None,
    default_model_type: str,
    version: str | None,
    features: Sequence[str] | None,
    inferred_features: Sequence[str] | None,
    baseline_metrics: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the ``model_type`` / ``version`` / ``features`` / ``baseline_metrics``
    keyword arguments for ``ModelMonitor.register_model``.

    Args:
        model_type: Caller's choice. A value in :data:`VALID_MODEL_TYPES` is used
            as-is; a task name from :data:`TASK_NAMES` is recorded as
            ``metadata["task"]`` and replaced by ``default_model_type``; ``None``
            means ``default_model_type``.
        default_model_type: The framework's ModelType when the caller gave none.
        version: Model version (default :data:`DEFAULT_VERSION`).
        features: Explicit feature names; wins over ``inferred_features``.
        inferred_features: Names read from the model/training data, if any.
        baseline_metrics: Baseline metrics known at registration time. Empty by
            default; the baseline can be set afterwards via ``set_baseline()``.
        metadata: Registration metadata. Updated in place with ``task`` when a
            task name was passed as ``model_type``.

    Raises:
        ValueError: ``model_type`` is neither a ModelType nor a task name, or no
            feature names were given and none could be inferred.
    """
    if model_type is None:
        resolved_type = default_model_type
    elif model_type in VALID_MODEL_TYPES:
        resolved_type = model_type
    elif model_type in TASK_NAMES:
        resolved_type = default_model_type
        metadata.setdefault("task", model_type)
    else:
        raise ValueError(
            f"Invalid model_type {model_type!r}. The API accepts one of: "
            f"{', '.join(VALID_MODEL_TYPES)}."
        )

    names = features if features is not None else inferred_features
    if not names:
        raise ValueError(
            "Could not determine the model's feature names. Pass "
            "features=[...] with the input column names."
        )

    return {
        "model_type": resolved_type,
        "version": version or DEFAULT_VERSION,
        "features": [str(name) for name in names],
        "baseline_metrics": dict(baseline_metrics) if baseline_metrics else {},
    }

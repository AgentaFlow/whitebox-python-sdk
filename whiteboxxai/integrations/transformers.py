"""
Hugging Face Transformers Integration

Integration for monitoring Hugging Face Transformers models.
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    import transformers
    from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer

    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError):
    # transformers' top-level API is a lazy module (_LazyModule); a tokenizers/transformers
    # version mismatch surfaces from its _get_module() as RuntimeError, not ImportError. Left
    # uncaught, that exception escapes this module and aborts the rest of integrations/__init__.py
    # mid-sequence, silently dropping every integration registered after this one (crewai,
    # boosting, langchain_agents).
    TRANSFORMERS_AVAILABLE = False
    PreTrainedModel = object
    PreTrainedTokenizer = object
    Pipeline = object


from whiteboxxai.integrations._registration import registration_fields
from whiteboxxai.monitor import ModelMonitor

# Pipeline tasks whose models produce text; registered as the API's ``llm`` type.
_GENERATION_TASKS = frozenset(
    {
        "text-generation",
        "text2text-generation",
        "translation",
        "summarization",
        "conversational",
    }
)


class TransformersMonitor(ModelMonitor):
    """
    Monitor for Hugging Face Transformers models.

    Supports:
    - Text classification
    - Named entity recognition (NER)
    - Question answering
    - Text generation
    - Translation
    - Summarization
    - Sentiment analysis

    Example:
        ```python
        from transformers import pipeline
        from whiteboxxai import WhiteBoxXAI
        from whiteboxxai.integrations.transformers import TransformersMonitor

        # Load model
        classifier = pipeline("sentiment-analysis")

        # Setup monitoring
        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = TransformersMonitor(
            client=client,
            pipeline=classifier,
            model_name="sentiment_classifier"
        )

        # Make predictions with automatic logging
        result = monitor.predict("I love this product!", log=True)
        ```
    """

    def __init__(
        self,
        client,
        pipeline: Pipeline | None = None,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        model_name: str | None = None,
        task: str | None = None,
        **kwargs,
    ):
        """
        Initialize Transformers monitor.

        Args:
            client: WhiteBoxXAI client instance
            pipeline: Hugging Face pipeline (recommended)
            model: PreTrainedModel (if not using pipeline)
            tokenizer: PreTrainedTokenizer (if not using pipeline)
            model_name: Name for the model
            task: Task type (text-classification, ner, qa, etc.)
            **kwargs: Additional arguments for ModelMonitor
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Install with: pip install transformers"
            )

        super().__init__(client, **kwargs)
        self.pipeline = pipeline
        self.model = model
        self.tokenizer = tokenizer
        self._model_name = model_name
        self._task = task
        self._registered = False

        # Auto-detect task from pipeline
        if pipeline is not None and task is None:
            self._task = getattr(pipeline, "task", None)

    def register_from_model(
        self,
        name: str | None = None,
        task: str | None = None,
        version: str | None = None,
        framework_version: str | None = None,
        model_type: str | None = None,
        features: list[str] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> int:
        """
        Register Transformers model with WhiteBoxXAI.

        Args:
            name: Model name (default: model class name or pipeline task)
            task: Task type (text-classification, ner, qa, etc.)
            version: Model version (default ``1.0.0``)
            framework_version: Transformers version
            model_type: One of the API's model types. Default: ``llm`` for
                generation tasks (text-generation, translation, summarization,
                ...), else ``neural_network``.
            features: Feature names (default: ``["text"]``)
            baseline_metrics: Baseline metrics, e.g. ``{"accuracy": 0.91}``
            **kwargs: Additional metadata

        Returns:
            Model ID

        Raises:
            ValueError: No model or pipeline, or an invalid ``model_type``.
        """
        if self.pipeline is None and self.model is None:
            raise ValueError("No model or pipeline provided")

        # Auto-detect model name
        if name is None:
            if self._model_name:
                name = self._model_name
            elif self.pipeline is not None:
                name = f"{self.pipeline.task}_pipeline"
            else:
                name = self.model.__class__.__name__

        # Use provided or detected task
        task = task or self._task or "text-classification"

        # Get Transformers version
        if framework_version is None:
            framework_version = transformers.__version__

        # Extract model metadata
        metadata = {
            "framework": "transformers",
            "framework_version": framework_version,
            "task": task,
            **kwargs,
        }

        # Get model information from pipeline or model
        if self.pipeline is not None:
            metadata["pipeline_type"] = self.pipeline.task
            if hasattr(self.pipeline, "model"):
                metadata["model_class"] = self.pipeline.model.__class__.__name__
                if hasattr(self.pipeline.model, "config"):
                    config = self.pipeline.model.config
                    metadata["model_type"] = getattr(config, "model_type", None)
                    metadata["num_parameters"] = getattr(config, "num_parameters", None)
                    metadata["vocab_size"] = getattr(config, "vocab_size", None)
        elif self.model is not None:
            metadata["model_class"] = self.model.__class__.__name__
            if hasattr(self.model, "config"):
                config = self.model.config
                metadata["model_type"] = getattr(config, "model_type", None)
                metadata["num_parameters"] = getattr(config, "num_parameters", None)

        # Map the task to the API's model type; the task itself stays in metadata
        default_model_type = "llm" if task.split("_")[0] in _GENERATION_TASKS else "neural_network"

        # Register model
        fields = registration_fields(
            model_type=model_type,
            default_model_type=default_model_type,
            version=version,
            features=features,
            inferred_features=["text"],
            baseline_metrics=baseline_metrics,
            metadata=metadata,
        )
        self.register_model(name=name, framework="transformers", metadata=metadata, **fields)

        self._registered = True
        return self.model_id

    def predict(
        self,
        inputs: str | list[str],
        log: bool = True,
        actuals: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict | list[dict]:
        """
        Make predictions and optionally log them.

        Args:
            inputs: Input text or list of texts
            log: Whether to log the prediction
            actuals: Actual values (if available)
            **kwargs: Additional arguments for the pipeline/model

        Returns:
            Predictions
        """
        if self.pipeline is None and self.model is None:
            raise ValueError("No model or pipeline provided")

        # Ensure model is registered
        if not self._registered:
            self.register_from_model()

        # Make prediction using pipeline
        if self.pipeline is not None:
            predictions = self.pipeline(inputs, **kwargs)
        else:
            # Use model + tokenizer
            if self.tokenizer is None:
                raise ValueError("Tokenizer required when using model directly")

            # Tokenize inputs
            encoded = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True, **kwargs
            )

            # Get predictions
            outputs = self.model(**encoded)
            predictions = outputs.logits.detach().cpu().numpy()

        # Log if requested
        if log:
            # Determine if batch or single
            is_batch = isinstance(inputs, list)

            if is_batch:
                # Batch prediction
                self.log_batch_transformers(
                    inputs=inputs,
                    predictions=predictions,
                    actuals=actuals,
                )
            else:
                # Single prediction
                self.log_prediction_transformers(
                    input_text=inputs,
                    prediction=(predictions if isinstance(predictions, dict) else predictions[0]),
                    actual=actuals,
                )

        return predictions

    def log_prediction_transformers(
        self,
        input_text: str,
        prediction: dict | Any,
        actual: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a single Transformers prediction.

        Args:
            input_text: Input text
            prediction: Prediction result (dict or value)
            actual: Actual value (if available)
            **kwargs: Additional metadata
        """
        # Extract prediction value based on task
        if isinstance(prediction, dict):
            pred_value = self._extract_prediction_value(prediction)
        else:
            pred_value = prediction

        # Log to WhiteBoxXAI
        self.log_prediction(
            inputs={"text": input_text},
            output=pred_value,
            metadata={
                "task": self._task,
                "raw_prediction": str(prediction),
                "actual": actual,
                **kwargs,
            },
        )

    def log_batch_transformers(
        self,
        inputs: list[str],
        predictions: list[dict | Any],
        actuals: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log batch of Transformers predictions.

        Args:
            inputs: List of input texts
            predictions: List of predictions
            actuals: List of actual values (if available)
            **kwargs: Additional metadata
        """
        # Prepare batch data
        batch_predictions = [
            self._extract_prediction_value(pred) if isinstance(pred, dict) else pred
            for pred in predictions
        ]

        batch = []
        for i, text in enumerate(inputs):
            item: dict[str, Any] = {
                "inputs": {"text": text},
                "output": batch_predictions[i],
            }
            if actuals is not None:
                item["actual"] = actuals[i]
            batch.append(item)

        # Log batch
        self.log_batch(batch)

    def _extract_prediction_value(self, prediction: dict) -> Any:
        """Extract the main prediction value from a pipeline output."""
        if isinstance(prediction, list) and len(prediction) > 0:
            prediction = prediction[0]

        if isinstance(prediction, dict):
            # Common keys in transformers outputs
            if "label" in prediction:
                return prediction["label"]
            elif "answer" in prediction:
                return prediction["answer"]
            elif "generated_text" in prediction:
                return prediction["generated_text"]
            elif "translation_text" in prediction:
                return prediction["translation_text"]
            elif "summary_text" in prediction:
                return prediction["summary_text"]
            elif "score" in prediction:
                return prediction["score"]

        return prediction

    def set_baseline(
        self,
        baseline_texts: list[str],
        baseline_labels: list[Any] | None = None,
    ) -> None:
        """
        Set baseline data for drift detection.

        Args:
            baseline_texts: Baseline input texts
            baseline_labels: Baseline labels (optional)
        """
        # Make baseline predictions if labels not provided
        if baseline_labels is None and self.pipeline is not None:
            baseline_predictions = self.pipeline(baseline_texts)
            baseline_labels = [
                self._extract_prediction_value(pred) for pred in baseline_predictions
            ]

        # Convert to format expected by parent class
        baseline_inputs = [{"text": text} for text in baseline_texts]

        # Set baseline through parent class
        super().set_baseline(baseline_inputs, baseline_labels)

    def log_generation_metrics(
        self,
        prompt: str,
        generated_text: str,
        num_tokens: int | None = None,
        generation_time: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log metrics for text generation tasks.

        Args:
            prompt: Input prompt
            generated_text: Generated text
            num_tokens: Number of tokens generated
            generation_time: Time taken to generate (seconds)
            **kwargs: Additional metrics
        """
        metrics = {
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens": num_tokens,
            "generation_time": generation_time,
            **kwargs,
        }

        self.log_custom_metric("text_generation", metrics)


class TransformersPipelineWrapper:
    """
    Wrapper for Hugging Face pipelines with automatic logging.

    Example:
        ```python
        from transformers import pipeline
        from whiteboxxai import WhiteBoxXAI
        from whiteboxxai.integrations.transformers import TransformersPipelineWrapper

        classifier = pipeline("sentiment-analysis")
        client = WhiteBoxXAI(api_key="your-api-key")

        # Wrap pipeline
        wrapped = TransformersPipelineWrapper(classifier, client)

        # Predictions are automatically logged
        result = wrapped("I love this!")
        ```
    """

    def __init__(
        self,
        pipeline: Pipeline,
        client,
        model_name: str | None = None,
        auto_register: bool = True,
    ):
        """
        Initialize pipeline wrapper.

        Args:
            pipeline: Hugging Face pipeline
            client: WhiteBoxXAI client
            model_name: Model name
            auto_register: Auto-register model on first prediction
        """
        self.pipeline = pipeline
        self.monitor = TransformersMonitor(
            client=client,
            pipeline=pipeline,
            model_name=model_name,
        )
        self._auto_register = auto_register

    def __call__(self, *args, **kwargs):
        """Make prediction with automatic logging."""
        # Auto-register if needed
        if self._auto_register and not self.monitor._registered:
            self.monitor.register_from_model()

        # Get inputs
        inputs = args[0] if args else kwargs.get("inputs")

        # Make prediction
        result = self.pipeline(*args, **kwargs)

        # Log prediction
        try:
            if isinstance(inputs, list):
                self.monitor.log_batch_transformers(
                    inputs=inputs,
                    predictions=result,
                )
            else:
                self.monitor.log_prediction_transformers(
                    input_text=inputs,
                    prediction=result,
                )
        except Exception as e:
            warnings.warn(f"Failed to log prediction: {e}", stacklevel=2)

        return result


def wrap_transformers_pipeline(
    pipeline: Pipeline,
    monitor: TransformersMonitor,
) -> Pipeline:
    """
    Wrap a Hugging Face pipeline to automatically log predictions.

    Args:
        pipeline: Hugging Face pipeline to wrap
        monitor: TransformersMonitor instance

    Returns:
        Wrapped pipeline that logs predictions

    Example:
        ```python
        from transformers import pipeline

        classifier = pipeline("sentiment-analysis")
        wrapped = wrap_transformers_pipeline(classifier, monitor)

        # Predictions automatically logged
        result = wrapped("Great product!")
        ```
    """
    # Python looks up special methods like __call__ on the type, not the
    # instance, so `pipeline.__call__ = logged_call` would never actually
    # change what `pipeline(...)` does. Wrap in a small object whose own
    # __call__ is a real class method instead.
    return _LoggedPipelineCall(pipeline, monitor)


class _LoggedPipelineCall:
    """Callable wrapper that logs each pipeline call before returning its result."""

    def __init__(self, pipeline: Pipeline, monitor: "TransformersMonitor"):
        self._pipeline = pipeline
        self._monitor = monitor

    def __call__(self, *args, **kwargs):
        result = self._pipeline(*args, **kwargs)

        try:
            inputs = args[0] if args else kwargs.get("inputs")

            if isinstance(inputs, list):
                self._monitor.log_batch_transformers(
                    inputs=inputs,
                    predictions=result,
                )
            else:
                self._monitor.log_prediction_transformers(
                    input_text=inputs,
                    prediction=result,
                )
        except Exception as e:
            warnings.warn(f"Failed to log predictions: {e}", stacklevel=2)

        return result

    def __getattr__(self, name):
        return getattr(self._pipeline, name)


__all__ = [
    "TransformersMonitor",
    "TransformersPipelineWrapper",
    "wrap_transformers_pipeline",
]

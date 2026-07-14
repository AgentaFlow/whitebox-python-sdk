# API Reference

Complete API documentation for WhiteBoxXAI SDK.

## WhiteBoxXAI Client

Main client for interacting with the WhiteBoxXAI API.

### Constructor

```python
WhiteBoxXAI(
    api_key: str = None,
    base_url: str = "https://api.whiteboxxai.com",
    timeout: int = 30,
    max_retries: int = 3,
    enable_offline: bool = False,
    offline_dir: str = "./whiteboxxai_offline",
    offline_max_queue_size: int = 10000,
    offline_auto_sync: bool = False,
    offline_sync_interval: int = 60,
    enable_caching: bool = False,
    cache_ttl: int = 3600,
    cache_max_size: int = 1000,
    enable_privacy_filters: bool = False,
    enable_sampling: bool = False,
    sampling_rate: float = 1.0
)
```

**Parameters:**
- `api_key` (str): API key for authentication. Can be set via WHITEBOXXAI_API_KEY env var
- `base_url` (str): Base URL for API endpoint
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum number of retry attempts
- `enable_offline` (bool): Enable offline mode with persistent queue
- `offline_dir` (str): Directory for offline queue storage
- `offline_max_queue_size` (int): Maximum queued operations (0 = unlimited)
- `offline_auto_sync` (bool): Enable automatic background sync
- `offline_sync_interval` (int): Sync interval in seconds
- `enable_caching` (bool): Enable local caching
- `cache_ttl` (int): Cache time-to-live in seconds
- `cache_max_size` (int): Maximum cache entries
- `enable_privacy_filters` (bool): Enable PII detection and masking
- `enable_sampling` (bool): Enable prediction sampling
- `sampling_rate` (float): Sampling rate (0.0-1.0)

### Resources

```python
client.models          # Models resource
client.predictions     # Predictions resource
client.explanations    # Explanations resource
client.drift           # Drift detection resource
client.alerts          # Alerts resource
```

### Methods

#### Offline Mode

```python
get_offline_status() -> dict
```
Get offline queue status.

**Returns:** Dictionary with queue_size, last_sync_time, failed_count

```python
sync_offline_queue() -> dict
```
Manually sync offline queue.

**Returns:** Dictionary with synced and failed counts

```python
cleanup_offline_queue(older_than_days: int = 7)
```
Remove old operations from queue.

## ModelMonitor

Simplified monitoring interface.

### Constructor

```python
ModelMonitor(
    client: WhiteBoxXAI,
    model_id: int = None,
    sampling_rate: float = 1.0,
    enable_explanations: bool = False
)
```

### Methods

```python
register_model(
    name: str,
    model_type: str,
    framework: str = None,
    version: str = None,
    **metadata
) -> int
```
Register a new model.

**Parameters:**
- `name` (str): Model name
- `model_type` (str): Type (classification, regression, etc.)
- `framework` (str): Framework name (sklearn, pytorch, etc.)
- `version` (str): Model version
- `**metadata`: Additional metadata

**Returns:** Model ID

```python
log_prediction(
    inputs: dict,
    output: dict,
    ground_truth: Any = None,
    explanation: dict = None,
    metadata: dict = None,
    priority: str = "NORMAL"
)
```
Log a single prediction.

**Parameters:**
- `inputs` (dict): Input features
- `output` (dict): Model output
- `ground_truth` (Any): Actual value (optional)
- `explanation` (dict): Model explanation (optional)
- `metadata` (dict): Additional metadata (optional)
- `priority` (str): Queue priority (CRITICAL, HIGH, NORMAL, LOW)

```python
log_batch(predictions: List[dict])
```
Log multiple predictions.

```python
set_baseline(data: np.ndarray, **kwargs)
```
Set baseline data for drift detection.

```python
detect_drift(data: np.ndarray) -> dict
```
Detect drift in current data.

**Returns:** Drift detection report

## Decorators

### @monitor_model

```python
@monitor_model(
    monitor: ModelMonitor,
    input_keys: List[str] = None,
    output_key: str = None,
    explain: bool = False
)
def predict(...):
    ...
```

Monitor all predictions from a function.

### @monitor_prediction

```python
@monitor_prediction(
    monitor: ModelMonitor,
    input_extractor: Callable = None,
    output_extractor: Callable = None
)
def predict(...):
    ...
```

Monitor predictions with custom extractors.

## Framework Integrations

### SklearnMonitor

```python
from whiteboxxai.integrations.sklearn import SklearnMonitor

monitor = SklearnMonitor(
    client: WhiteBoxXAI,
    model=None,
    model_name: str = None
)

monitor.register_from_model(model_type: str, **kwargs) -> int
monitor.wrap_model(model)
monitor.predict(model, X, y=None, log=True)
```

### TorchMonitor

```python
from whiteboxxai.integrations.pytorch import TorchMonitor

monitor = TorchMonitor(
    client: WhiteBoxXAI,
    model=None,
    model_name: str = None
)

monitor.register_from_model(model_type: str, **kwargs) -> int
monitor.wrap_model(model)
```

### KerasMonitor

```python
from whiteboxxai.integrations.tensorflow import KerasMonitor, WhiteBoxXAICallback

monitor = KerasMonitor(
    client: WhiteBoxXAI,
    model=None,
    model_name: str = None
)

monitor.register_from_model(model_type: str, **kwargs) -> int
callback = WhiteBoxXAICallback(monitor, log_frequency=1)
monitor.predict(X, log=True)
```

### TransformersMonitor

```python
from whiteboxxai.integrations.transformers import TransformersMonitor

monitor = TransformersMonitor(
    client: WhiteBoxXAI,
    pipeline=None,
    model_name: str = None
)

monitor.register_from_model(name: str, version: str = "1.0.0") -> int
monitor.predict(text, log=True)
```

### LangChainMonitor

```python
from whiteboxxai.integrations.langchain import LangChainMonitor

monitor = LangChainMonitor(
    client: WhiteBoxXAI,
    application_name: str,
    track_tokens: bool = True,
    track_cost: bool = True
)

monitor.register_application(name: str, version: str = "1.0.0") -> int
callback = monitor.create_callback_handler()
```

### XGBoostMonitor / LightGBMMonitor

```python
from whiteboxxai.integrations.boosting import XGBoostMonitor, LightGBMMonitor

monitor = XGBoostMonitor(
    client: WhiteBoxXAI,
    model_name: str,
    track_feature_importance: bool = True,
    importance_type: str = "gain"
)

monitor.register_from_model(model, X_train, y_train) -> int
monitor.predict(model, X, y=None, log=True)
```

## Exceptions

```python
from whiteboxxai.exceptions import (
    WhiteBoxXAIError,          # Base exception
    AuthenticationError,       # Authentication failed
    APIError,                 # API request error
    ValidationError,          # Validation error
    RateLimitError,          # Rate limit exceeded
)
```

## Privacy

```python
from whiteboxxai.privacy import mask_data

masked = mask_data(
    data: dict,
    patterns: List[str] = None  # Custom PII patterns
) -> dict
```

Mask sensitive data before sending to API.

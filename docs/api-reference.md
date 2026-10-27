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
    offline_dir: str = None,  # default: ./WHITEBOX_offline
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
- `api_key` (str): Scoped API key (generate one under API Keys in your WhiteBoxXAI settings). Can be set via the `WHITEBOX_API_KEY` env var; `WHITEBOXXAI_API_KEY`, the name 1.2.0 read, is still accepted
- `base_url` (str): Base URL for API endpoint. Can be set via `WHITEBOX_BASE_URL` (`WHITEBOXXAI_BASE_URL` is still accepted)
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum number of retry attempts
- `enable_offline` (bool): Enable offline mode with persistent queue
- `offline_dir` (str): Directory for offline queue storage. Default `./WHITEBOX_offline`, or `./whiteboxxai_offline` (the 1.2.0 default) if only that one exists
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
client.models           # Models resource, incl. Git-backed version diffing
client.predictions      # Predictions resource
client.explanations     # SHAP/LIME and DiCE counterfactual explanations
client.drift            # Drift detection resource
client.fairness         # Bias/fairness auditing resource
client.alerts           # Alert rules, instances and triage
client.risk_register    # ISO 42001 risk register resource
client.governance       # Review boards, review requests and two-factor voting
client.compliance       # ISO/IEC 42001 readiness, gaps and control catalog
client.reports          # Report types, generation and preview
client.prompts          # Prompt template registry, versioning and drift
client.llm              # LLM call logging and cost/usage analytics resource
client.rag              # RAG retrieval and evaluation resource
client.safety           # Content safety analysis resource
client.llm_xai          # LLM explainability resource
client.agent_workflows  # Multi-agent workflow tracking resource
client.metrics          # Model performance metrics resource
client.alert_intelligence  # ML-driven alert correlation and root-cause analysis
client.alert_management    # Alert escalation policies, routing and lifecycle
```

**Explanations:** they run on the model artifact stored on the platform, so they only work
for models registered through GitHub auto-registration with `storage_mode: full`. A model
registered through the SDK or the dashboard stores metadata only, and its explanation request
ends in a failed explanation.

#### Reports

```python
client.reports.list() -> dict
client.reports.generate(report_type_id: str, model_id: str = None, format: str = "pdf") -> dict
client.reports.preview(report_type_id: str, model_id: str = None) -> dict
```

`format` is `"pdf"`, `"html"` or `"docx"`. Every method has an async twin (`alist`, `agenerate`,
`apreview`).

#### Compliance

```python
client.compliance.iso42001_score() -> dict
client.compliance.iso42001_gaps(clause_filter=None, severity_filter=None) -> dict
client.compliance.list_frameworks() -> dict
client.compliance.get_framework_controls(framework: str) -> dict
```

`framework` is the `key` returned by `list_frameworks()` (for example `iso_42001`).

#### Prompts

`client.prompts` covers templates (`list_templates`, `get_template`, `create_template`), versions
(`list_versions`, `get_version`, `create_version`), the lifecycle (`advance_stage`, `deploy`,
`rollback`) and drift (`compute_drift`, `list_drift_reports`, `get_drift_report`, `get_incidents`),
each with an `a`-prefixed async twin.

#### Governance runs

```python
from whiteboxxai import governance_run

with governance_run():
    client.llm.log_call(...)
    client.predictions.log(...)
```

Records sent inside the block carry one `X-WhiteBox-Run-Id` and share Governance Events (up to 100
per event). Pass `governance_run("my-id")` to group records across processes. The LangChain
callback handler does this for you.

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
- `model_type` (str): One of `xgboost`, `random_forest`, `neural_network`, `linear_model`, `llm`, `custom`
- `framework` (str): Framework name (sklearn, pytorch, etc.)
- `version` (str): Model version (required by the API)
- `**metadata`: Additional metadata. `features` (list of input names) and `baseline_metrics`
  (dict) are required by the API

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
set_baseline(data: np.ndarray, labels: Optional[np.ndarray] = None)
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

monitor.register_from_model(
    name=None, model_type=None, version=None, features=None, baseline_metrics=None, **metadata
) -> int
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

monitor.register_from_model(
    name=None, model_type=None, version=None, features=None, baseline_metrics=None, **metadata
) -> int
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

monitor.register_from_model(
    name=None, model_type=None, version=None, features=None, baseline_metrics=None, **metadata
) -> int
monitor.register_saved_model(
    model_path, metadata=None, version=None, features=None, baseline_metrics=None
)
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

monitor.register_from_model(
    name=None, task=None, version=None, model_type=None, features=None, baseline_metrics=None
) -> int
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

monitor.register_application(
    name=None, version=None, description=None, features=None, baseline_metrics=None
) -> int
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

monitor.register_from_model(
    model, X_train=None, y_train=None, model_type=None, metadata=None,
    version=None, features=None, baseline_metrics=None
) -> int
monitor.predict(model, X, y=None, log=True)
```

### Registration rules for `register_from_model()`

The API requires `version`, `features`, `baseline_metrics` and a `model_type` from a fixed list
(`xgboost`, `random_forest`, `neural_network`, `linear_model`, `llm`, `custom`). The helpers pick
the `model_type` for the framework and default `version` to `1.0.0` and `baseline_metrics` to `{}`;
`features` come from the model's own feature names, else `feature_0..feature_{n-1}` from its input
width. When none can be determined a `ValueError` asks for `features=[...]`. A task name such as
`"classification"` is still accepted as `model_type` and is stored as `metadata["task"]`; any other
value raises `ValueError` listing the valid types.

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

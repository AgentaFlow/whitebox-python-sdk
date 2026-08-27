# WhiteBoxXAI Python SDK

Official Python SDK for integrating WhiteBoxXAI monitoring into your ML applications.

## Features

- 🚀 **Easy Integration** - Monitor models with just a few lines of code
- 📊 **Framework Support** - Native integrations for Scikit-learn, PyTorch, TensorFlow, XGBoost, and more
- 🎯 **Decorator-based Monitoring** - Zero-code-change monitoring with decorators
- ⚡ **Async/Sync Interfaces** - Support for both synchronous and asynchronous workflows
- 🔒 **Privacy-First** - Built-in PII detection and data masking
- 💾 **Local Caching** - TTL-based caching to reduce API calls
- 📈 **Drift Detection** - Automatic model and data drift monitoring
- 🎨 **Flexible Configuration** - Extensive configuration options and feature flags

## Installation

```bash
pip install whitebox-xai-sdk

# With specific framework support
pip install whitebox-xai-sdk[sklearn]
pip install whitebox-xai-sdk[pytorch]
pip install whitebox-xai-sdk[all]  # All integrations
```

## Quick Start

### Basic Usage

```python
from whiteboxxai import WhiteBoxXAI, ModelMonitor

# Initialize client
client = WhiteBoxXAI(api_key="your-api-key")

# Create monitor
monitor = ModelMonitor(client)

# Register model
model_id = monitor.register_model(
    name="fraud_detection",
    model_type="classification",
    framework="sklearn"
)

# Log predictions
monitor.log_prediction(
    inputs={"amount": 100.0, "merchant": "store_123"},
    output={"fraud_probability": 0.15, "prediction": "legitimate"}
)
```

### Scikit-learn Integration

```python
from sklearn.ensemble import RandomForestClassifier
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.sklearn import SklearnMonitor

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = SklearnMonitor(client, model=model)
monitor.register_from_model(model_type="classification")

# Wrap model for automatic monitoring
monitored_model = monitor.wrap_model(model)

# Predictions are automatically logged
predictions = monitored_model.predict(X_test)
```

### PyTorch Integration

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
monitor.register_from_model(model_type="classification")

# Wrap model
monitored_model = monitor.wrap_model(model)

# Predictions are automatically logged
with torch.no_grad():
    outputs = monitored_model(inputs)
```

### TensorFlow/Keras Integration

```python
from tensorflow import keras
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.tensorflow import KerasMonitor, WhiteBoxXAICallback

# Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = KerasMonitor(client, model=model, model_name="keras_model")
monitor.register_from_model(model_type="regression")

# Train with monitoring callback
callback = WhiteBoxXAICallback(monitor, log_frequency=1)
model.fit(X_train, y_train,
          validation_split=0.2,
          callbacks=[callback],
          epochs=50)

# Make predictions with automatic logging
predictions = monitor.predict(X_test, log=True)
```

### Hugging Face Transformers Integration

```python
from transformers import pipeline
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.transformers import TransformersMonitor, wrap_transformers_pipeline

# Load model
classifier = pipeline("sentiment-analysis")

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = TransformersMonitor(
    client=client,
    pipeline=classifier,
    model_name="sentiment_classifier"
)

# Register model
monitor.register_from_model(name="Sentiment Classifier", version="1.0.0")

# Make predictions with automatic logging
result = monitor.predict("I love this product!", log=True)

# Or wrap pipeline for auto-logging
wrapped = wrap_transformers_pipeline(classifier, monitor)
result = wrapped("Great service!")  # Automatically logged
```

### LangChain Integration

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.langchain import LangChainMonitor, wrap_langchain_chain

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = LangChainMonitor(
    client=client,
    application_name="qa_bot",
    track_tokens=True,
    track_cost=True
)

# Register application
monitor.register_application(name="Q&A Bot", version="1.0.0")

# Create chain
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(input_variables=["question"], template="Answer: {question}")
chain = LLMChain(llm=llm, prompt=prompt)

# Option 1: Use callback handler
callback = monitor.create_callback_handler()
result = chain.run(question="What is AI?", callbacks=[callback])

# Option 2: Wrap chain for auto-logging
wrapped_chain = wrap_langchain_chain(chain, monitor)
result = wrapped_chain.run(question="What is AI?")  # Automatically logged
```

For multi-agent LangChain/LangGraph workflows and CrewAI, see
`whiteboxxai.integrations.langchain_agents` (`MultiAgentCallbackHandler`,
`LangGraphMultiAgentMonitor`, `monitor_langchain_agent`) and
`whiteboxxai.integrations.crewai_monitor` (`CrewAIMonitor`, `monitor_crew`).

### XGBoost/LightGBM Monitoring

```python
import xgboost as xgb
import lightgbm as lgb
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.boosting import XGBoostMonitor, LightGBMMonitor, wrap_xgboost_model

client = WhiteBoxXAI(api_key="your-api-key")

# XGBoost monitoring
xgb_monitor = XGBoostMonitor(
    client=client,
    model_name="fraud_detector",
    track_feature_importance=True,
    importance_type="gain"  # or 'weight', 'cover', 'total_gain', 'total_cover'
)

# Train and register model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
xgb_monitor.register_from_model(model, X_train, y_train)

# Make predictions with monitoring
predictions = xgb_monitor.predict(model, X_test, y_test)

# Or wrap model for automatic logging
wrapped_model = wrap_xgboost_model(model, xgb_monitor)
predictions = wrapped_model.predict(X_test)  # Auto-logged

# LightGBM monitoring
lgb_monitor = LightGBMMonitor(
    client=client,
    model_name="churn_predictor",
    track_feature_importance=True,
    importance_type="gain"  # or 'split'
)

model = lgb.LGBMClassifier(n_estimators=100)
model.fit(X_train, y_train)
lgb_monitor.register_from_model(model, X_train, y_train)
predictions = lgb_monitor.predict(model, X_test, y_test)
```

### Decorator-based Monitoring

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

### Async Support

```python
import asyncio
from whiteboxxai import WhiteBoxXAI, ModelMonitor

async def main():
    async with WhiteBoxXAI(api_key="your-api-key") as client:
        monitor = ModelMonitor(client)

        # Register model
        model_id = await monitor.aregister_model(
            name="async_model",
            model_type="classification"
        )

        # Log prediction
        await monitor.alog_prediction(
            inputs={"feature1": 1.0},
            output={"prediction": 0.85}
        )

asyncio.run(main())
```

## Advanced Features

### Local Buffering & Batch Flushing

For high-throughput logging, buffer predictions locally and flush them as a
batch instead of sending every prediction immediately:

```python
from whiteboxxai import WhiteBoxXAI, ModelMonitor

client = WhiteBoxXAI(api_key="your-api-key")

# Predictions are buffered locally; a batch is sent once 100 accumulate,
# or when the buffer is flushed (explicitly, or on context-manager exit).
with ModelMonitor(client, model_id=123, buffer_size=100) as monitor:
    for features, output in predictions:
        monitor.log_prediction(inputs=features, output=output)
    # Any remaining buffered predictions are flushed automatically here.

print(monitor.get_prediction_count())
```

### Offline Mode

Enable robust operation with unreliable network connectivity. Operations are queued locally and synced automatically.

```python
from whiteboxxai import WhiteBoxXAI

# Enable offline mode with auto-sync
client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_offline=True,
    offline_dir="./whiteboxxai_offline",
    offline_auto_sync=True,
    offline_sync_interval=60  # Sync every 60 seconds
)

# Operations are automatically queued when API is unavailable
# Check queue status
status = client.get_offline_status()
print(f"Queued operations: {status['queue_size']}")

# Manually trigger sync
result = client.sync_offline_queue()
print(f"Synced: {result['synced']}, Failed: {result['failed']}")

# Cleanup old operations
client.cleanup_offline_queue(older_than_days=7)
```

**Key Features:**
- **Persistent Queue**: SQLite-based storage survives restarts
- **Auto-Sync**: Background synchronization every 60s (configurable)
- **Priority-Based**: CRITICAL > HIGH > NORMAL > LOW
- **Retry Logic**: Automatic retry with exponential backoff (max 3 attempts)
- **Thread-Safe**: Supports concurrent operations

**Configuration:**
```python
client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_offline=True,
    offline_dir="./offline_queue",        # Storage directory
    offline_max_queue_size=10000,         # Max operations (0 = unlimited)
    offline_auto_sync=True,               # Enable auto-sync
    offline_sync_interval=60,             # Sync interval (seconds)
)
```

### Privacy Filters

```python
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.privacy import mask_data

client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_privacy_filters=True
)

# Data is automatically masked before sending
data = {
    "email": "user@example.com",
    "phone": "555-123-4567",
    "amount": 100.0
}

# Mask sensitive data
masked = mask_data(data)
# {"email": "***MASKED***", "phone": "***MASKED***", "amount": 100.0}
```

### Local Caching

```python
client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_caching=True,
    cache_ttl=3600,
    cache_max_size=1000
)
```

### Sampling

```python
# Monitor 10% of predictions
monitor = ModelMonitor(
    client,
    model_id=123,
    sampling_rate=0.1
)
```

### Drift Detection

```python
import numpy as np

# Set baseline data
baseline = np.random.randn(1000, 10)
monitor.set_baseline(baseline)

# Detect drift
current_data = np.random.randn(100, 10)
drift_report = monitor.detect_drift(current_data)

# Retrieve previously persisted drift reports
reports = monitor.get_drift_reports(limit=10)
```

## Configuration

The SDK can be configured via constructor parameters or environment variables:

```python
from whiteboxxai import WhiteBoxXAI

client = WhiteBoxXAI(
    api_key="your-api-key",              # or WHITEBOXXAI_API_KEY env var
    base_url="https://api.whiteboxxai.com", # Custom API endpoint, or WHITEBOXXAI_BASE_URL env var
    timeout=30,                           # Request timeout (seconds)
    max_retries=3,                        # Retry attempts

    # Offline mode
    enable_offline=True,                  # Enable offline queueing
    offline_dir="./whiteboxxai_offline",  # Queue storage directory
    offline_max_queue_size=10000,         # Max queued operations
    offline_auto_sync=True,               # Auto-sync in background
    offline_sync_interval=60,             # Sync interval (seconds)

    # Other features
    enable_caching=True,                  # Enable local caching
    enable_privacy_filters=True,          # Enable PII masking
    enable_sampling=True,                 # Enable prediction sampling
    sampling_rate=1.0                     # Sample 100% of predictions
)
```

### Authentication

The `api_key` parameter (or `WHITEBOXXAI_API_KEY` env var) is sent as a
bearer token: `Authorization: Bearer <api_key>`. As of this SDK's current
release, the WhiteBoxXAI backend validates this token as a standard JWT
obtained via account login (`/api/v1/auth/login`), not a separate,
dedicated API-key entity — there is no self-service API key issuance/
management endpoint yet. In practice: log in through your WhiteBoxXAI
account (dashboard or the `/auth/login` endpoint) and use the returned
token as `api_key`. This also matches how the companion MCP server
authenticates (see below) and will be updated here once dedicated,
long-lived API keys ship on the backend.

## Using WhiteBoxXAI from Other Languages (MCP)

This SDK is the primary, first-class integration path and is Python-only.
For other languages, or for agentic clients (Claude Desktop, Claude Code,
LangChain, custom agent harnesses), use the companion
[Model Context Protocol](https://modelcontextprotocol.io/) server,
`whiteboxxai-mcp`, instead of calling the API directly:

```bash
pip install whiteboxxai-mcp

# Configure a service-account credential, then run the stdio server
export WHITEBOXXAI_MCP_API_BASE_URL="https://api.whiteboxxai.com"
export WHITEBOXXAI_MCP_EMAIL="mcp-service@yourorg.com"
export WHITEBOXXAI_MCP_PASSWORD="..."
whiteboxxai-mcp
```

Point any MCP-compatible client at this command (e.g. Claude Desktop's
`mcpServers` config, or Claude Code's `.mcp.json`). MCP is a language- and
client-agnostic protocol, so this is the recommended path for non-Python
integrations.

**Current limitations** (as of this SDK's release): only models,
predictions, drift, and bias/fairness tools are available; explanations,
LLM/RAG observability, safety, alerts, and multi-agent workflow tools are
tracked as follow-on milestones, and `whitebox_explanations_generate`
currently returns placeholder feature-importance values rather than real
SHAP/LIME output. Auth uses the same username/password (or short-lived
JWT) mechanism described above — there is no long-lived API-key/
service-account system yet.

See the `whiteboxxai-mcp` package's own README for full setup and its
tool reference.

## API Reference

### WhiteBoxXAI Client

Main client for API interaction.

**Methods:**
- `models` - Models resource
- `predictions` - Predictions resource
- `explanations` - Explanations resource
- `drift` - Drift detection resource
- `fairness` - Bias/fairness auditing resource
- `alerts` - Alerts resource
- `risk_register` - ISO 42001 risk register resource
- `governance` - Governance boards and review requests resource
- `llm` - LLM call logging and cost/usage analytics resource
- `rag` - RAG retrieval and evaluation resource
- `safety` - Content safety analysis resource
- `llm_xai` - LLM explainability (attention, token importance, counterfactuals) resource
- `agent_workflows` - Multi-agent workflow tracking resource
- `metrics` - Model performance metrics resource

### ModelMonitor

Simplified monitoring interface.

**Methods:**
- `register_model()` - Register a new model
- `log_prediction()` - Log a single prediction (or buffer it, if `buffer_size` is set)
- `log_batch()` - Log multiple predictions
- `flush()` - Send any buffered predictions immediately
- `get_prediction_count()` - Number of predictions logged by this monitor instance
- `set_baseline()` - Set baseline data for drift detection
- `detect_drift()` - Detect model drift
- `get_drift_reports()` - Retrieve previously persisted drift reports
- `create_alert_rule()` - Create a threshold-based alert rule for this model
- `get_active_alerts()` - List alert rules for this model
- Can be used as a context manager (`with ModelMonitor(...) as monitor:`); buffered predictions are flushed on exit.

### Decorators

- `@monitor_model` - Monitor all predictions from a function
- `@monitor_prediction` - Monitor individual predictions with custom extractors

### Framework Integrations

- `whiteboxxai.integrations.sklearn` - Scikit-learn integration
- `whiteboxxai.integrations.pytorch` - PyTorch integration
- `whiteboxxai.integrations.tensorflow` - TensorFlow/Keras integration
- `whiteboxxai.integrations.transformers` - Hugging Face Transformers integration
- `whiteboxxai.integrations.langchain` - LangChain chains/agents integration
- `whiteboxxai.integrations.langchain_agents` - LangChain/LangGraph multi-agent integration
- `whiteboxxai.integrations.crewai_monitor` - CrewAI multi-agent integration
- `whiteboxxai.integrations.boosting` - XGBoost/LightGBM integration

## Examples

See the `examples/` directory for more examples:

- `basic_monitoring.py` - Basic monitoring example
- `sklearn_integration.py` - Scikit-learn integration
- `pytorch_integration.py` - PyTorch integration
- `tensorflow_example.py` - TensorFlow/Keras integration
- `transformers_example.py` - Hugging Face Transformers integration
- `langchain_example.py` - LangChain integration
- `boosting_example.py` - XGBoost/LightGBM integration
- `decorator_monitoring.py` - Decorator-based monitoring
- `async_monitoring.py` - Async API usage
- `offline_mode_example.py` - Offline mode with queue management

## Support

- Documentation: https://docs.whiteboxxai.com
- Issues: https://github.com/AgentaFlow/whitebox-python-sdk/issues
- Email: support@whiteboxxai.com

## License

MIT License - see LICENSE file for details

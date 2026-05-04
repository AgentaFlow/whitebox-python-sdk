# WhiteBoxAI Python SDK

Official Python SDK for integrating WhiteBoxAI monitoring into your ML applications.

## Features

- 🚀 **Easy Integration** - Monitor models with just a few lines of code
- 📊 **Framework Support** - Native integrations for Scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, Hugging Face Transformers, and LangChain
- 🤖 **Multi-Agent Monitoring** - First-class support for CrewAI and LangGraph multi-agent workflows
- 🎯 **Decorator-based Monitoring** - Zero-code-change monitoring with decorators
- ⚡ **Async/Sync Interfaces** - Support for both synchronous and asynchronous workflows
- 🔒 **Privacy-First** - Built-in PII detection and data masking
- 💾 **Local Caching** - TTL-based caching to reduce API calls
- 📈 **Drift Detection** - Automatic model and data drift monitoring
- 🗂️ **Git Context** - Automatic capture of git branch, commit, and author metadata
- 🎨 **Flexible Configuration** - Extensive configuration options and feature flags

## Installation

```bash
pip install whiteboxai-sdk

# With specific framework support
pip install whiteboxai-sdk[sklearn]
pip install whiteboxai-sdk[pytorch]
pip install whiteboxai-sdk[all]  # All integrations
```

## Quick Start

### Basic Usage

```python
from whiteboxai import WhiteBoxAI, ModelMonitor

# Initialize client
client = WhiteBoxAI(api_key="your-api-key")

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
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.sklearn import SklearnMonitor

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Setup monitoring
client = WhiteBoxAI(api_key="your-api-key")
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
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.pytorch import TorchMonitor

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Setup monitoring
client = WhiteBoxAI(api_key="your-api-key")
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
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.tensorflow import KerasMonitor, WhiteBoxAICallback

# Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Setup monitoring
client = WhiteBoxAI(api_key="your-api-key")
monitor = KerasMonitor(client, model=model, model_name="keras_model")
monitor.register_from_model(model_type="regression")

# Train with monitoring callback
callback = WhiteBoxAICallback(monitor, log_frequency=1)
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
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.transformers import TransformersMonitor, wrap_transformers_pipeline

# Load model
classifier = pipeline("sentiment-analysis")

# Setup monitoring
client = WhiteBoxAI(api_key="your-api-key")
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
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.langchain import LangChainMonitor, wrap_langchain_chain

# Setup monitoring
client = WhiteBoxAI(api_key="your-api-key")
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

### Multi-Agent Monitoring (CrewAI / LangGraph)

```python
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.crewai_monitor import CrewAIMonitor

client = WhiteBoxAI(api_key="your-api-key")
monitor = CrewAIMonitor(client=client)

# Start workflow monitoring
workflow_id = monitor.start_monitoring(
    workflow_name="research_pipeline",
    description="Multi-agent research and writing workflow"
)

# Register and track agents
monitor.register_agent(researcher, role="Research Analyst")
monitor.register_agent(writer, role="Content Writer")

# Log task completions and agent interactions
monitor.log_task_completion(task, status="completed", output_data=result)
monitor.log_interaction(from_agent=researcher, to_agent=writer, interaction_type="handoff")

# Complete and retrieve analytics
summary = monitor.complete_monitoring(status="completed")
```

For LangGraph multi-agent workflows, use `LangGraphMultiAgentMonitor` and `MultiAgentCallbackHandler`
from `whiteboxai.integrations.langchain_agents`.

### XGBoost/LightGBM Monitoring

```python
import xgboost as xgb
import lightgbm as lgb
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.boosting import XGBoostMonitor, LightGBMMonitor, wrap_xgboost_model

client = WhiteBoxAI(api_key="your-api-key")

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
from whiteboxai import WhiteBoxAI, ModelMonitor, monitor_model

client = WhiteBoxAI(api_key="your-api-key")
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
from whiteboxai import WhiteBoxAI, ModelMonitor

async def main():
    async with WhiteBoxAI(api_key="your-api-key") as client:
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

### Git Context

Capture git metadata automatically alongside model monitoring:

```python
from whiteboxai import WhiteBoxAI, GitContext, detect_git_context

client = WhiteBoxAI(api_key="your-api-key")

# Auto-detect from current git repo
git_ctx = detect_git_context()
print(git_ctx.branch, git_ctx.commit_hash, git_ctx.author)

# Or build manually
git_ctx = GitContext(
    branch="main",
    commit_hash="abc1234",
    author="dev@example.com"
)
```

### Offline Mode

Enable robust operation with unreliable network connectivity. Operations are queued locally and synced automatically.

```python
from whiteboxai import WhiteBoxAI

# Enable offline mode with auto-sync
client = WhiteBoxAI(
    api_key="your-api-key",
    enable_offline=True,
    offline_dir="./whiteboxai_offline",
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
client = WhiteBoxAI(
    api_key="your-api-key",
    enable_offline=True,
    offline_dir="./offline_queue",        # Storage directory
    offline_max_queue_size=10000,         # Max operations (0 = unlimited)
    offline_auto_sync=True,               # Enable auto-sync
    offline_sync_interval=60,             # Sync interval (seconds)
)
```

See [Offline Mode Guide](docs/offline-mode.md) for complete documentation.

### Privacy Filters

```python
from whiteboxai import WhiteBoxAI
from whiteboxai.privacy import mask_data

client = WhiteBoxAI(
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
client = WhiteBoxAI(
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
```

## Configuration

The SDK can be configured via constructor parameters or environment variables:

```python
from whiteboxai import WhiteBoxAI

client = WhiteBoxAI(
    api_key="your-api-key",              # or EXPLAINAI_API_KEY env var
    base_url="https://api.whiteboxai.io", # or EXPLAINAI_BASE_URL env var
    timeout=30,                           # Request timeout (seconds)
    max_retries=3,                        # Retry attempts

    # Offline mode
    enable_offline=True,                  # Enable offline queueing
    offline_dir="./whiteboxai_offline",  # Queue storage directory
    offline_max_queue_size=10000,        # Max queued operations
    offline_auto_sync=True,              # Auto-sync in background
    offline_sync_interval=60,            # Sync interval (seconds)

    # Other features
    enable_caching=True,                  # Enable local caching
    enable_privacy_filters=True,          # Enable PII masking
    enable_sampling=True,                 # Enable prediction sampling
    sampling_rate=1.0                     # Sample 100% of predictions
)
```

## API Reference

### WhiteBoxAI Client

Main client for API interaction.

**Resources:**
- `models` - Register and manage models
- `predictions` - Log predictions and batches
- `explanations` - Generate and retrieve explanations
- `drift` - Drift detection and reports
- `alerts` - Alert management
- `agent_workflows` - Multi-agent workflow tracking

### ModelMonitor

Simplified monitoring interface.

**Methods:**
- `register_model()` - Register a new model
- `log_prediction()` - Log a single prediction
- `log_batch()` - Log multiple predictions
- `set_baseline()` - Set baseline data for drift detection
- `detect_drift()` - Detect model drift

### Decorators

- `@monitor_model` - Monitor all predictions from a function
- `@monitor_prediction` - Monitor individual predictions with custom extractors

### Framework Integrations

- `whiteboxai.integrations.sklearn` - Scikit-learn integration
- `whiteboxai.integrations.pytorch` - PyTorch integration
- `whiteboxai.integrations.tensorflow` - TensorFlow/Keras integration
- `whiteboxai.integrations.transformers` - Hugging Face Transformers integration
- `whiteboxai.integrations.langchain` - LangChain chains and agents
- `whiteboxai.integrations.langchain_agents` - LangGraph multi-agent monitoring
- `whiteboxai.integrations.crewai_monitor` - CrewAI workflow monitoring
- `whiteboxai.integrations.boosting` - XGBoost and LightGBM integration

## Examples

See the `examples/` directory for more examples:

- `basic_monitoring.py` - Basic monitoring example
- `async_monitoring.py` - Async API usage
- `decorator_monitoring.py` - Zero-code-change decorator usage
- `sklearn_integration.py` - Scikit-learn integration
- `pytorch_integration.py` - PyTorch integration
- `tensorflow_example.py` - TensorFlow/Keras integration
- `transformers_example.py` - Hugging Face Transformers integration
- `langchain_example.py` - LangChain chains and agents
- `boosting_example.py` - XGBoost/LightGBM integration
- `offline_mode_example.py` - Offline mode with queue management

## Support

- Documentation: https://whitebox.agentaflow.com
- Issues: https://github.com/AgentaFlow/whitebox-python-sdk/issues

```
whiteboxai-python-sdk/
├── src/
│   └── whiteboxai/
│       ├── __init__.py
│       ├── __version__.py
│       ├── client.py
│       ├── config.py
│       ├── monitor.py
│       ├── decorators.py
│       ├── privacy.py
│       ├── offline.py
│       ├── cache.py
│       ├── git_utils.py
│       ├── resources.py
│       ├── utils.py
│       ├── exceptions.py
│       ├── integrations/
│       │   ├── sklearn.py
│       │   ├── pytorch.py
│       │   ├── tensorflow.py
│       │   ├── transformers.py
│       │   ├── langchain.py
│       │   ├── langchain_agents.py
│       │   ├── crewai_monitor.py
│       │   └── boosting.py
│       └── models/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── examples/
│   ├── basic_monitoring.py
│   ├── async_monitoring.py
│   ├── decorator_monitoring.py
│   ├── sklearn_integration.py
│   ├── pytorch_integration.py
│   ├── tensorflow_example.py
│   ├── transformers_example.py
│   ├── langchain_example.py
│   ├── boosting_example.py
│   └── offline_mode_example.py
├── docs/
│   ├── getting-started.md
│   ├── integrations.md
│   ├── offline-mode.md
│   ├── api-reference.md
│   ├── SKLEARN_INTEGRATION.md
│   ├── PYTORCH_INTEGRATION.md
│   ├── TENSORFLOW_INTEGRATION.md
│   ├── HUGGINGFACE_INTEGRATION.md
│   ├── LANGCHAIN_INTEGRATION.md
│   └── PRODUCTION_DEPLOYMENT.md
├── pyproject.toml
├── setup.py
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .github/
    └── workflows/
        ├── test.yml
        └── publish.yml
```

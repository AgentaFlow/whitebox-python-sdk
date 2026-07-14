# Getting Started with WhiteBoxXAI SDK

## Installation

Install the WhiteBoxXAI SDK using pip:

```bash
pip install whitebox-xai-sdk
```

### Optional Dependencies

Install with specific framework support:

```bash
# Scikit-learn support
pip install whitebox-xai-sdk[sklearn]

# PyTorch support
pip install whitebox-xai-sdk[pytorch]

# TensorFlow support
pip install whitebox-xai-sdk[tensorflow]

# All integrations
pip install whitebox-xai-sdk[all]
```

## Quick Start

### 1. Initialize the Client

```python
from whiteboxxai import WhiteBoxXAI

client = WhiteBoxXAI(api_key="your-api-key")
```

You can also set the API key via environment variable:

```bash
export WHITEBOXXAI_API_KEY=your-api-key
```

### 2. Create a Monitor

```python
from whiteboxxai import ModelMonitor

monitor = ModelMonitor(client)
```

### 3. Register Your Model

```python
model_id = monitor.register_model(
    name="my_model",
    model_type="classification",
    framework="sklearn"
)
```

### 4. Log Predictions

```python
monitor.log_prediction(
    inputs={"feature1": 1.0, "feature2": 2.0},
    output={"prediction": 1, "probability": 0.85}
)
```

## Next Steps

- Check out the [API Reference](api-reference.md) for detailed documentation
- Learn about [Framework Integrations](integrations.md)
- Explore [Offline Mode](offline-mode.md) for robust operation
- Review [Examples](../examples/) for more use cases

## Configuration

Configure the client with various options:

```python
client = WhiteBoxXAI(
    api_key="your-api-key",
    base_url="https://api.whiteboxxai.com",
    timeout=30,
    max_retries=3,
    enable_caching=True,
    enable_privacy_filters=True,
    enable_offline=True
)
```

## Support

For help and support:
- Documentation: https://docs.whiteboxxai.com
- Issues: https://github.com/AgentaFlow/whitebox-python-sdk/issues

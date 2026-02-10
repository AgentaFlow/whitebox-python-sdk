# WhiteBoxAI Python SDK

Official Python SDK for integrating WhiteBoxAI monitoring into your ML applications.

## Features

- 🚀 **Easy Integration** - Monitor models with just a few lines of code
- 📊 **Framework Support** - Native integrations for Scikit-learn, PyTorch, TensorFlow, XGBoost, and more
- 🎯 **Decorator-based Monitoring** - Zero-code-change monitoring with decorators
- ⚡ **Async/Sync Interfaces** - Support for both synchronous and asynchronous workflows
- 🔒 **Privacy-First** - Built-in PII detection and data masking
- 💾 **Local Caching** - TTL-based caching to reduce API calls
- 📈 **Drift Detection** - Automatic model and data drift monitoring
- 🎨 **Flexible Configuration** - Extensive configuration options and feature flags
- 🔍 **Git Integration** - Automatic Git context detection for model versioning
- 🤖 **Multi-Agent Support** - Monitor CrewAI and LangChain multi-agent workflows

## Installation

```bash
pip install whiteboxai-sdk

# With specific framework support
pip install whiteboxai-sdk[sklearn]
pip install whiteboxai-sdk[pytorch]
pip install whiteboxai-sdk[langchain]
pip install whiteboxai-sdk[crewai]
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

### Git Integration

```python
from whiteboxai import WhiteBoxAI, detect_git_context

# Auto-detect Git context
git_context = detect_git_context()

# Initialize with Git context
client = WhiteBoxAI(api_key="your-api-key")
model_id = client.models.register(
    name="my_model",
    **git_context.to_dict()  # Include Git metadata
)
```

### CrewAI Multi-Agent Monitoring

```python
from whiteboxai.integrations import CrewAIMonitor
from crewai import Agent, Task, Crew

# Initialize monitor
monitor = CrewAIMonitor(api_key="your-api-key")

# Define your crew
crew = Crew(agents=[...], tasks=[...])

# Start monitoring
workflow_id = monitor.start_monitoring(
    crew=crew,
    workflow_name="Research Workflow"
)

# Execute crew
result = crew.kickoff()

# Complete monitoring
summary = monitor.complete_monitoring(outputs={"result": result})
```

### LangChain Multi-Agent Monitoring

```python
from whiteboxai.integrations import LangGraphMultiAgentMonitor

# Create monitor
monitor = LangGraphMultiAgentMonitor(
    client=client,
    workflow_name="Multi-Agent Research"
)

# Start monitoring
workflow_id = monitor.start_monitoring()

# Register agents
monitor.register_agent("supervisor", role="Coordinates agents")
monitor.register_agent("researcher", role="Gathers information")

# Execute with callbacks
result = agent_executor.run(
    callbacks=monitor.get_callbacks("researcher")
)

# Complete monitoring
summary = monitor.complete_monitoring(outputs={"result": result})
```

## Framework Integrations

### Scikit-learn

```python
from whiteboxai.integrations import SklearnMonitor
from sklearn.ensemble import RandomForestClassifier

# Wrap your model
monitor = SklearnMonitor(client=client, model_id=model_id)
model = RandomForestClassifier()
wrapped_model = monitor.wrap(model)

# Use as normal - monitoring happens automatically
wrapped_model.fit(X_train, y_train)
predictions = wrapped_model.predict(X_test)
```

### PyTorch

```python
from whiteboxai.integrations import TorchMonitor
import torch.nn as nn

# Monitor your model
monitor = TorchMonitor(client=client, model_id=model_id)
model = MyNeuralNetwork()
monitor.attach(model)

# Training is automatically monitored
for epoch in range(num_epochs):
    train(model, train_loader)
```

### TensorFlow/Keras

```python
from whiteboxai.integrations import KerasMonitor

# Add callback
monitor = KerasMonitor(client=client, model_id=model_id)
model.fit(
    X_train, y_train,
    callbacks=[monitor.get_callback()],
    epochs=10
)
```

### LangChain

```python
from whiteboxai.integrations import LangChainMonitor

# Monitor chain execution
monitor = LangChainMonitor(client=client)
callback = monitor.create_callback()

chain.run("question", callbacks=[callback])
```

## Documentation

- [Getting Started Guide](getting-started.md) - Detailed installation and setup
- [Integration Guides](integrations.md) - Framework-specific integration tutorials
- [Offline Mode](offline-mode.md) - Running without internet connectivity
- [Production Deployment](PRODUCTION_DEPLOYMENT.md) - Best practices for production
- [API Reference](api-reference.md) - Complete API documentation

## Support

- **Documentation**: [Full Documentation](https://github.com/AgentaFlow/whitebox-python-sdk)
- **Issues**: [GitHub Issues](https://github.com/AgentaFlow/whitebox-python-sdk/issues)
- **Community**: [Discussions](https://github.com/AgentaFlow/whitebox-python-sdk/discussions)

## License

MIT License - see [LICENSE](https://github.com/AgentaFlow/whitebox-python-sdk/blob/main/LICENSE) for details.

# Framework Integrations

WhiteBoxXAI SDK provides native integrations for popular ML frameworks.

## Scikit-learn

Monitor scikit-learn models with automatic feature extraction:

```python
from sklearn.ensemble import RandomForestClassifier
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.sklearn import SklearnMonitor

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = SklearnMonitor(client, model=model)
monitor.register_from_model(model_type="classification")

# Wrap model for automatic monitoring
monitored_model = monitor.wrap_model(model)
predictions = monitored_model.predict(X_test)
```

See [SKLEARN_INTEGRATION.md](SKLEARN_INTEGRATION.md) for detailed documentation.

## PyTorch

Monitor PyTorch models with tensor tracking:

```python
import torch.nn as nn
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.pytorch import TorchMonitor

# Define your model
model = nn.Sequential(...)

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = TorchMonitor(client, model=model)
monitor.register_from_model(model_type="classification")

# Wrap model
monitored_model = monitor.wrap_model(model)
```

See [PYTORCH_INTEGRATION.md](PYTORCH_INTEGRATION.md) for detailed documentation.

## TensorFlow/Keras

Monitor TensorFlow models with training callbacks:

```python
from tensorflow import keras
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.tensorflow import KerasMonitor, WhiteBoxXAICallback

# Build your model
model = keras.Sequential([...])

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = KerasMonitor(client, model=model, model_name="keras_model")
monitor.register_from_model(model_type="regression")

# Train with monitoring
callback = WhiteBoxXAICallback(monitor, log_frequency=1)
model.fit(X_train, y_train, callbacks=[callback], epochs=50)
```

See [TENSORFLOW_INTEGRATION.md](TENSORFLOW_INTEGRATION.md) for detailed documentation.

## Hugging Face Transformers

Monitor transformers pipelines and models:

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

# Make monitored predictions
result = monitor.predict("I love this!", log=True)
```

See [HUGGINGFACE_INTEGRATION.md](HUGGINGFACE_INTEGRATION.md) for detailed documentation.

## LangChain

Monitor LangChain applications and chains:

```python
from langchain.chains import LLMChain
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.langchain import LangChainMonitor

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = LangChainMonitor(
    client=client,
    application_name="qa_bot",
    track_tokens=True,
    track_cost=True
)

# Create callback for chains
callback = monitor.create_callback_handler()
result = chain.run(question="What is AI?", callbacks=[callback])
```

See [LANGCHAIN_INTEGRATION.md](LANGCHAIN_INTEGRATION.md) for detailed documentation.

## XGBoost & LightGBM

Monitor gradient boosting models:

```python
import xgboost as xgb
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.boosting import XGBoostMonitor

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Setup monitoring
client = WhiteBoxXAI(api_key="your-api-key")
monitor = XGBoostMonitor(
    client=client,
    model_name="fraud_detector",
    track_feature_importance=True
)

monitor.register_from_model(model, X_train, y_train)
predictions = monitor.predict(model, X_test, y_test)
```

## Custom Integrations

Create custom integrations by extending the base `ModelMonitor` class:

```python
from whiteboxxai import ModelMonitor

class CustomMonitor(ModelMonitor):
    def __init__(self, client, model, **kwargs):
        super().__init__(client, **kwargs)
        self.model = model
    
    def predict(self, inputs, log=True):
        output = self.model.predict(inputs)
        if log:
            self.log_prediction(inputs=inputs, output=output)
        return output
```

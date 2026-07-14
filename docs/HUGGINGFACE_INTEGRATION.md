# Hugging Face Transformers Integration Guide

This guide demonstrates how to use WhiteBoxXAI to monitor Hugging Face Transformers models for NLP tasks including text classification, named entity recognition, question answering, and text generation.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Tasks](#supported-tasks)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

Install WhiteBoxXAI SDK with Transformers support:

```bash
pip install whiteboxxai[transformers]
```

Or install dependencies separately:

```bash
pip install whiteboxxai transformers torch
```

## Quick Start

```python
from transformers import pipeline
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.transformers import TransformersMonitor

# Initialize client
client = WhiteBoxXAI(api_key="your-api-key")

# Load Hugging Face pipeline
classifier = pipeline("sentiment-analysis")

# Create monitor
monitor = TransformersMonitor(
    client=client,
    pipeline=classifier,
    model_name="sentiment_classifier"
)

# Register model
monitor.register_from_model(
    name="Sentiment Classifier",
    version="1.0.0"
)

# Make predictions with automatic logging
result = monitor.predict("I love this product!", log=True)
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Supported Tasks

WhiteBoxXAI supports all major Hugging Face Transformers tasks:

### Text Classification

```python
from transformers import pipeline
from whiteboxxai.integrations.transformers import TransformersMonitor

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
monitor = TransformersMonitor(client, pipeline=classifier)
monitor.register_from_model(name="Sentiment Classifier")

# Single prediction
result = monitor.predict("This is amazing!", log=True)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch predictions
texts = ["Great product!", "Terrible experience.", "It's okay."]
results = monitor.predict(texts, log=True)
```

### Named Entity Recognition (NER)

```python
# Load NER pipeline
ner = pipeline("ner", aggregation_strategy="simple")
monitor = TransformersMonitor(client, pipeline=ner)
monitor.register_from_model(name="NER Model", task="ner")

# Detect entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
entities = monitor.predict(text, log=True)
# [
#   {'entity_group': 'ORG', 'word': 'Apple Inc.', 'score': 0.999},
#   {'entity_group': 'PER', 'word': 'Steve Jobs', 'score': 0.998},
#   {'entity_group': 'LOC', 'word': 'Cupertino', 'score': 0.997}
# ]
```

### Question Answering

```python
# Load QA pipeline
qa = pipeline("question-answering")
monitor = TransformersMonitor(client, pipeline=qa)
monitor.register_from_model(name="QA Model", task="question-answering")

# Answer questions
context = "Paris is the capital of France."
question = "What is the capital of France?"

result = qa(question=question, context=context)
# {'answer': 'Paris', 'score': 0.996}

# Log manually
monitor.log_prediction_transformers(
    input_text=f"Q: {question}\nContext: {context}",
    prediction=result
)
```

### Text Generation

```python
# Load generation pipeline
generator = pipeline("text-generation", model="gpt2")
monitor = TransformersMonitor(client, pipeline=generator)
monitor.register_from_model(name="GPT-2 Generator", task="text-generation")

# Generate text
prompt = "Artificial intelligence is"
result = generator(prompt, max_length=50)
# [{'generated_text': 'Artificial intelligence is transforming...'}]

# Log generation metrics
import time
start = time.time()
result = generator(prompt, max_length=50)
generation_time = time.time() - start

monitor.log_generation_metrics(
    prompt=prompt,
    generated_text=result[0]['generated_text'],
    num_tokens=len(result[0]['generated_text'].split()),
    generation_time=generation_time
)
```

### Translation

```python
# Load translation pipeline
translator = pipeline("translation_en_to_fr")
monitor = TransformersMonitor(client, pipeline=translator)
monitor.register_from_model(name="EN-FR Translator", task="translation")

# Translate
result = monitor.predict("Hello, how are you?", log=True)
# [{'translation_text': 'Bonjour, comment allez-vous?'}]
```

### Summarization

```python
# Load summarization pipeline
summarizer = pipeline("summarization")
monitor = TransformersMonitor(client, pipeline=summarizer)
monitor.register_from_model(name="Summarizer", task="summarization")

# Summarize text
article = """
Long article text here...
"""
summary = monitor.predict(article, max_length=130, min_length=30, log=True)
# [{'summary_text': 'Brief summary...'}]
```

## Basic Usage

### Method 1: Using TransformersMonitor

```python
from whiteboxxai.integrations.transformers import TransformersMonitor

# Create monitor
monitor = TransformersMonitor(
    client=client,
    pipeline=classifier,
    model_name="my_model"
)

# Register model
model_id = monitor.register_from_model(
    name="My Classifier",
    version="1.0.0",
    description="Production sentiment classifier"
)

# Make predictions
result = monitor.predict("Input text", log=True)

# Batch predictions
results = monitor.predict(["Text 1", "Text 2"], log=True)
```

### Method 2: Using Pipeline Wrapper

```python
from whiteboxxai.integrations.transformers import wrap_transformers_pipeline

# Wrap pipeline for auto-logging
wrapped_classifier = wrap_transformers_pipeline(classifier, monitor)

# All predictions automatically logged
result = wrapped_classifier("Input text")
```

### Method 3: Using TransformersPipelineWrapper

```python
from whiteboxxai.integrations.transformers import TransformersPipelineWrapper

# Wrap pipeline (with auto-registration)
wrapper = TransformersPipelineWrapper(
    pipeline=classifier,
    client=client,
    model_name="my_classifier",
    auto_register=True
)

# Use like normal pipeline - predictions logged automatically
result = wrapper("Input text")
```

## Advanced Features

### Custom Model and Tokenizer

If not using a pipeline, you can monitor models directly:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create monitor
monitor = TransformersMonitor(
    client=client,
    model=model,
    tokenizer=tokenizer,
    task="text-classification"
)

# Register and use
monitor.register_from_model(name="Custom Classifier")
```

### Baseline Data for Drift Detection

Set baseline data to enable drift detection:

```python
# Prepare baseline texts
baseline_texts = [
    "Excellent product, highly recommend!",
    "Poor quality, very disappointed.",
    "Average, nothing special.",
    "Love it! Best purchase ever!",
    "Waste of money.",
]

# Set baseline
monitor.set_baseline(baseline_texts)

# Now drift will be detected automatically
```

### Monitoring Multiple Models

```python
# Monitor multiple models
classifiers = {
    "sentiment": pipeline("sentiment-analysis"),
    "emotion": pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion"),
    "toxicity": pipeline("text-classification", model="unitary/toxic-bert"),
}

monitors = {}
for name, pipe in classifiers.items():
    monitor = TransformersMonitor(client, pipeline=pipe, model_name=name)
    monitor.register_from_model(name=f"{name.title()} Classifier")
    monitors[name] = monitor

# Use different monitors
text = "I absolutely love this!"
for name, monitor in monitors.items():
    result = monitor.predict(text, log=True)
    print(f"{name}: {result}")
```

### Custom Metadata

Add custom metadata to predictions:

```python
result = monitor.predict(
    "Input text",
    log=True,
    user_id="user_123",
    session_id="sess_456",
    source="mobile_app"
)
```

### Generation Metrics

For text generation tasks, track detailed metrics:

```python
import time

prompt = "The future of AI is"
start_time = time.time()

result = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.8,
    top_p=0.9
)

generation_time = time.time() - start_time
generated_text = result[0]['generated_text']

monitor.log_generation_metrics(
    prompt=prompt,
    generated_text=generated_text,
    num_tokens=len(generated_text.split()),
    generation_time=generation_time,
    temperature=0.8,
    top_p=0.9,
    model_size="124M"  # GPT-2 small
)
```

### Multi-GPU Inference

```python
# Use device_map for multi-GPU
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="facebook/opt-1.3b",
    device_map="auto",  # Automatic multi-GPU
    torch_dtype="auto"
)

monitor = TransformersMonitor(client, pipeline=generator)
monitor.register_from_model(
    name="OPT-1.3B Generator",
    metadata={"device_map": "auto", "dtype": "auto"}
)

result = monitor.predict("The meaning of life is", log=True)
```

## Best Practices

### 1. Use Pipelines When Possible

Pipelines provide the easiest integration:

```python
# ✅ Recommended
classifier = pipeline("sentiment-analysis")
monitor = TransformersMonitor(client, pipeline=classifier)

# ⚠️ More complex
model = AutoModel.from_pretrained("...")
tokenizer = AutoTokenizer.from_pretrained("...")
monitor = TransformersMonitor(client, model=model, tokenizer=tokenizer)
```

### 2. Register Models Once

Register models during initialization, not on every prediction:

```python
# ✅ Good
monitor = TransformersMonitor(client, pipeline=classifier)
monitor.register_from_model(name="My Model")  # Once

for text in texts:
    monitor.predict(text, log=True)

# ❌ Bad
for text in texts:
    monitor = TransformersMonitor(client, pipeline=classifier)
    monitor.register_from_model(name="My Model")  # Every time!
    monitor.predict(text, log=True)
```

### 3. Use Batch Predictions

Process multiple inputs efficiently:

```python
# ✅ Efficient
texts = ["Text 1", "Text 2", "Text 3", ...]
results = monitor.predict(texts, log=True)

# ❌ Inefficient
for text in texts:
    result = monitor.predict(text, log=True)
```

### 4. Set Appropriate Sampling Rates

For high-volume production:

```python
monitor = TransformersMonitor(
    client=client,
    pipeline=classifier,
    sampling_rate=0.1  # Log 10% of predictions
)
```

### 5. Handle Errors Gracefully

```python
try:
    result = monitor.predict(text, log=True)
except Exception as e:
    print(f"Prediction failed: {e}")
    # Fallback logic
    result = fallback_prediction(text)
```

### 6. Monitor Model Performance

Regularly check drift and performance:

```python
# Set baseline
monitor.set_baseline(training_texts)

# Periodically check drift
drift_report = client.get_drift_report(monitor.model_id)
if drift_report['severity'] == 'high':
    alert_team("Model drift detected!")
```

## Troubleshooting

### Issue: "transformers is not installed"

**Solution**: Install transformers:
```bash
pip install transformers torch
```

### Issue: "No model or pipeline provided"

**Solution**: Provide either a pipeline or model+tokenizer:
```python
# Option 1: Pipeline
monitor = TransformersMonitor(client, pipeline=classifier)

# Option 2: Model + Tokenizer
monitor = TransformersMonitor(client, model=model, tokenizer=tokenizer)
```

### Issue: Slow predictions

**Solution**: Use batch processing and GPU:
```python
# Use GPU
classifier = pipeline("sentiment-analysis", device=0)  # GPU 0

# Batch process
results = monitor.predict(texts, log=True)  # List of texts
```

### Issue: High memory usage

**Solution**: Use smaller models or quantization:
```python
# Use distilled models
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

# Or quantization
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    load_in_8bit=True  # 8-bit quantization
)
```

### Issue: Rate limiting

**Solution**: Adjust sampling rate:
```python
monitor = TransformersMonitor(
    client=client,
    pipeline=classifier,
    sampling_rate=0.05  # Log 5% of predictions
)
```

## Examples

See complete examples in:
- `sdk/examples/transformers_example.py` - Comprehensive examples
- `sdk/examples/notebooks/` - Jupyter notebooks

## API Reference

### TransformersMonitor

Main class for monitoring Transformers models.

**Methods**:
- `register_from_model()` - Register model with WhiteBoxXAI
- `predict()` - Make predictions with logging
- `set_baseline()` - Set baseline data for drift detection
- `log_generation_metrics()` - Log text generation metrics

### TransformersPipelineWrapper

Wrapper class for automatic logging.

### wrap_transformers_pipeline()

Function to wrap existing pipelines.

## Support

For issues or questions:
- GitHub Issues: https://github.com/whiteboxxai/whiteboxxai
- Documentation: https://docs.whiteboxxai.com
- Email: support@whiteboxxai.com

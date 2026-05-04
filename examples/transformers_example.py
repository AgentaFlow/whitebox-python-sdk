"""
Hugging Face Transformers Integration Example

This example demonstrates how to use WhiteBoxAI to monitor Hugging Face Transformers models.
"""

import os

from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.transformers import TransformersMonitor, wrap_transformers_pipeline

# Optional: Set API key
os.environ["WHITEBOXAI_API_KEY"] = "your-api-key-here"


def example_sentiment_analysis():
    """Example using sentiment analysis pipeline."""
    from transformers import pipeline

    print("=" * 60)
    print("Sentiment Analysis Example")
    print("=" * 60)

    # Initialize WhiteBoxAI client
    client = WhiteBoxAI(api_key=os.getenv("WHITEBOXAI_API_KEY"))

    # Load sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")

    # Create monitor
    monitor = TransformersMonitor(
        client=client, pipeline=classifier, model_name="sentiment_classifier_v1"
    )

    # Register model
    model_id = monitor.register_from_model(
        name="DistilBERT Sentiment Classifier",
        version="1.0.0",
        description="Sentiment analysis using DistilBERT",
    )
    print(f"✓ Model registered with ID: {model_id}")

    # Test texts
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic experience!",
        "Worst purchase ever.",
    ]

    # Make predictions with automatic logging
    print("\nMaking predictions...")
    for text in test_texts:
        result = monitor.predict(text, log=True)
        label = result[0]["label"]
        score = result[0]["score"]
        print(f"  Text: '{text[:40]}...'")
        print(f"  Prediction: {label} (confidence: {score:.3f})")

    # Set baseline data for drift detection
    baseline_texts = [
        "Great product, highly recommend!",
        "Poor quality, disappointed.",
        "Average performance.",
    ]
    print("\n✓ Setting baseline data for drift detection...")
    monitor.set_baseline(baseline_texts)

    print("\n✓ Sentiment analysis monitoring complete!")


def example_ner():
    """Example using named entity recognition."""
    from transformers import pipeline

    print("\n" + "=" * 60)
    print("Named Entity Recognition Example")
    print("=" * 60)

    # Initialize WhiteBoxAI client
    client = WhiteBoxAI(api_key=os.getenv("WHITEBOXAI_API_KEY"))

    # Load NER pipeline
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")

    # Create monitor
    monitor = TransformersMonitor(client=client, pipeline=ner_pipeline, model_name="ner_model_v1")

    # Register model
    model_id = monitor.register_from_model(name="BERT NER Model", version="1.0.0", task="ner")
    print(f"✓ Model registered with ID: {model_id}")

    # Test text
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

    # Make prediction
    print(f"\nAnalyzing: '{text}'")
    result = monitor.predict(text, log=True)

    print("\nDetected entities:")
    for entity in result:
        print(f"  - {entity['word']}: {entity['entity_group']} (score: {entity['score']:.3f})")

    print("\n✓ NER monitoring complete!")


def example_text_generation():
    """Example using text generation pipeline."""
    import time

    from transformers import pipeline

    print("\n" + "=" * 60)
    print("Text Generation Example")
    print("=" * 60)

    # Initialize WhiteBoxAI client
    client = WhiteBoxAI(api_key=os.getenv("WHITEBOXAI_API_KEY"))

    # Load text generation pipeline (using smaller GPT-2 for demo)
    generator = pipeline("text-generation", model="gpt2")

    # Create monitor
    monitor = TransformersMonitor(client=client, pipeline=generator, model_name="gpt2_generator")

    # Register model
    model_id = monitor.register_from_model(
        name="GPT-2 Text Generator", version="1.0.0", task="text-generation"
    )
    print(f"✓ Model registered with ID: {model_id}")

    # Generate text
    prompt = "Artificial intelligence is"
    print(f"\nPrompt: '{prompt}'")

    start_time = time.time()
    result = generator(prompt, max_length=50, num_return_sequences=1)
    generation_time = time.time() - start_time

    generated_text = result[0]["generated_text"]
    print(f"\nGenerated: '{generated_text}'")

    # Log generation metrics
    monitor.log_generation_metrics(
        prompt=prompt,
        generated_text=generated_text,
        num_tokens=len(generated_text.split()),
        generation_time=generation_time,
    )

    print(f"\n✓ Generation metrics logged (time: {generation_time:.2f}s)")


def example_wrapped_pipeline():
    """Example using wrapped pipeline for automatic logging."""
    from transformers import pipeline

    print("\n" + "=" * 60)
    print("Wrapped Pipeline Example (Auto-logging)")
    print("=" * 60)

    # Initialize WhiteBoxAI client
    client = WhiteBoxAI(api_key=os.getenv("WHITEBOXAI_API_KEY"))

    # Load pipeline
    classifier = pipeline("sentiment-analysis")

    # Create monitor and wrap pipeline
    monitor = TransformersMonitor(
        client=client, pipeline=classifier, model_name="wrapped_classifier"
    )

    # Register model
    monitor.register_from_model(name="Wrapped Sentiment Classifier")

    # Wrap pipeline for automatic logging
    wrapped_classifier = wrap_transformers_pipeline(classifier, monitor)

    print("✓ Pipeline wrapped - predictions will be automatically logged")

    # Use wrapped pipeline - predictions logged automatically
    texts = [
        "This is wonderful!",
        "I'm not satisfied.",
        "It's acceptable.",
    ]

    print("\nMaking predictions (auto-logged)...")
    for text in texts:
        result = wrapped_classifier(text)
        print(f"  '{text}' -> {result[0]['label']}")

    print("\n✓ All predictions automatically logged!")


def example_batch_prediction():
    """Example using batch predictions."""
    from transformers import pipeline

    print("\n" + "=" * 60)
    print("Batch Prediction Example")
    print("=" * 60)

    # Initialize WhiteBoxAI client
    client = WhiteBoxAI(api_key=os.getenv("WHITEBOXAI_API_KEY"))

    # Load pipeline
    classifier = pipeline("sentiment-analysis")

    # Create monitor
    monitor = TransformersMonitor(client=client, pipeline=classifier, model_name="batch_classifier")

    # Register model
    monitor.register_from_model(name="Batch Sentiment Classifier")

    # Batch of texts
    batch_texts = [
        "Excellent service!",
        "Very disappointing.",
        "Average experience.",
        "Highly recommend!",
        "Not worth the money.",
        "Satisfactory product.",
    ]

    print(f"\nProcessing batch of {len(batch_texts)} texts...")

    # Make batch prediction with logging
    results = monitor.predict(batch_texts, log=True)

    print("\nResults:")
    for text, result in zip(batch_texts, results):
        print(f"  '{text}' -> {result['label']} ({result['score']:.3f})")

    print(f"\n✓ Batch of {len(batch_texts)} predictions logged!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("WhiteBoxAI - Hugging Face Transformers Integration Examples")
    print("=" * 60)

    try:
        # Run examples
        example_sentiment_analysis()
        example_ner()
        example_text_generation()
        example_wrapped_pipeline()
        example_batch_prediction()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

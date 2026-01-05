# TensorFlow/Keras Integration Guide

Complete guide for integrating WhiteBoxAI monitoring with TensorFlow and Keras models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Models](#supported-models)
3. [Keras Callbacks](#keras-callbacks)
4. [Training Monitoring](#training-monitoring)
5. [SavedModel Support](#savedmodel-support)
6. [TensorFlow Datasets](#tensorflow-datasets)
7. [Distributed Training](#distributed-training)
8. [TensorFlow Serving](#tensorflow-serving)
9. [Best Practices](#best-practices)
10. [Complete Examples](#complete-examples)

---

## Quick Start

```python
import tensorflow as tf
from tensorflow import keras
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.tensorflow import KerasMonitor

# Initialize WhiteBoxAI
client = WhiteBoxAI(api_key='your-api-key')

# Define Keras model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Create monitor
monitor = KerasMonitor(
    model=model,
    client=client,
    model_name="keras_regressor",
    model_type="regression"
)

# Compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Make predictions with automatic logging
X_test = tf.random.normal((10, 10))
predictions = monitor.predict(X_test)
```

---

## Supported Models

### Sequential Models

```python
# Simple Sequential model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

monitor = KerasMonitor(model, client, "sequential_classifier")
```

### Functional API

```python
# Functional API model
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
monitor = KerasMonitor(model, client, "cnn_classifier")
```

### Custom Models (Subclassing)

```python
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

model = CustomModel()
monitor = KerasMonitor(model, client, "custom_model")
```

---

## Keras Callbacks

### WhiteBoxAI Training Callback

```python
from whiteboxai.integrations.tensorflow import WhiteBoxAICallback

# Create callback
callback = WhiteBoxAICallback(
    monitor=monitor,
    log_frequency=1,  # Log every epoch
    log_weights=False,
    log_gradients=False,
    log_validation=True
)

# Train with callback
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[callback],
    verbose=1
)
```

### Custom Metrics Logging

```python
class CustomMetricsCallback(keras.callbacks.Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        # Log standard metrics
        self.monitor.log_epoch(
            epoch=epoch,
            train_loss=logs.get('loss'),
            val_loss=logs.get('val_loss'),
            train_accuracy=logs.get('accuracy'),
            val_accuracy=logs.get('val_accuracy')
        )

        # Calculate and log custom metrics
        learning_rate = float(keras.backend.get_value(self.model.optimizer.lr))
        self.monitor.log_custom_metric(
            'learning_rate',
            learning_rate,
            epoch=epoch
        )

# Use callback
custom_callback = CustomMetricsCallback(monitor)
model.fit(X_train, y_train, epochs=50, callbacks=[custom_callback])
```

### Prediction Logging Callback

```python
class PredictionCallback(keras.callbacks.Callback):
    def __init__(self, monitor, X_val, y_val, log_frequency=5):
        super().__init__()
        self.monitor = monitor
        self.X_val = X_val
        self.y_val = y_val
        self.log_frequency = log_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            # Make predictions
            predictions = self.model.predict(self.X_val, verbose=0)

            # Log to WhiteBoxAI
            self.monitor.log_batch(
                inputs=self.X_val,
                predictions=predictions,
                actuals=self.y_val,
                metadata={'epoch': epoch}
            )

# Use callback
pred_callback = PredictionCallback(monitor, X_val, y_val, log_frequency=5)
model.fit(X_train, y_train, epochs=50, callbacks=[pred_callback])
```

---

## Training Monitoring

### Complete Training Setup

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

# Create monitor with baseline
monitor = KerasMonitor(
    model=model,
    client=client,
    model_name="keras_binary_classifier",
    model_type="classification"
)
monitor.set_baseline(X_train, y_train)

# Callbacks
callbacks = [
    WhiteBoxAICallback(monitor, log_frequency=1),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Log final test results
predictions = model.predict(X_test)
monitor.log_batch(
    inputs=X_test,
    predictions=predictions,
    actuals=y_test,
    metadata={'phase': 'final_test'}
)
```

### Learning Rate Scheduling

```python
# Learning rate schedule callback
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 0.95 ** epoch
)

# Custom callback to log learning rate
class LRLogger(keras.callbacks.Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        self.monitor.log_custom_metric('learning_rate', lr, epoch=epoch)

# Train with LR scheduling
model.fit(
    X_train, y_train,
    epochs=50,
    callbacks=[
        WhiteBoxAICallback(monitor),
        lr_schedule,
        LRLogger(monitor)
    ]
)
```

---

## SavedModel Support

### Saving Models

```python
# Save model in SavedModel format
model.save('saved_model/my_model')

# Register saved model with WhiteBoxAI
monitor.register_saved_model(
    model_path='saved_model/my_model',
    metadata={
        'framework': 'tensorflow',
        'version': tf.__version__,
        'input_shape': (20,),
        'output_shape': (1,)
    }
)
```

### Loading SavedModels

```python
# Load SavedModel
loaded_model = keras.models.load_model('saved_model/my_model')

# Create monitor for loaded model
loaded_monitor = KerasMonitor(
    model=loaded_model,
    client=client,
    model_name="loaded_keras_model"
)

# Use for predictions
predictions = loaded_monitor.predict(X_test)
```

### H5 Format Support

```python
# Save in H5 format
model.save('model.h5')

# Load H5 model
h5_model = keras.models.load_model('model.h5')

# Monitor H5 model
h5_monitor = KerasMonitor(h5_model, client, "h5_model")
```

---

## TensorFlow Datasets

### tf.data Integration

```python
import tensorflow_datasets as tfds

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Preprocessing
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with monitoring
monitor = KerasMonitor(model, client, "mnist_classifier")
callback = WhiteBoxAICallback(monitor)

model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    callbacks=[callback]
)
```

### Custom Dataset Monitoring

```python
class MonitoredDataset:
    def __init__(self, dataset, monitor):
        self.dataset = dataset
        self.monitor = monitor
        self.batch_count = 0

    def __iter__(self):
        for batch in self.dataset:
            self.batch_count += 1

            # Log data quality metrics
            if self.batch_count % 100 == 0:
                X_batch, y_batch = batch
                self.monitor.log_data_quality(
                    batch_idx=self.batch_count,
                    stats={
                        'mean': float(tf.reduce_mean(X_batch)),
                        'std': float(tf.math.reduce_std(X_batch)),
                        'min': float(tf.reduce_min(X_batch)),
                        'max': float(tf.reduce_max(X_batch))
                    }
                )

            yield batch

# Wrap dataset
monitored_train = MonitoredDataset(ds_train, monitor)
```

---

## Distributed Training

### Multi-GPU Training

```python
# Create strategy
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Create model within strategy scope
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create monitor
    monitor = KerasMonitor(
        model=model,
        client=client,
        model_name="distributed_model"
    )

# Train (monitoring happens on chief worker)
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32 * strategy.num_replicas_in_sync,
    callbacks=[WhiteBoxAICallback(monitor)]
)
```

### TPU Training

```python
# Resolve TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except ValueError:
    strategy = tf.distribute.get_strategy()
    print("Running on CPU/GPU")

# Create model with TPU strategy
with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    monitor = KerasMonitor(model, client, "tpu_model")

# Train
model.fit(ds_train, epochs=10, callbacks=[WhiteBoxAICallback(monitor)])
```

---

## TensorFlow Serving

### Preparing Models for Serving

```python
# Save model with version
version = 1
export_path = f'serving/my_model/{version}'
model.save(export_path, save_format='tf')

# Register with WhiteBoxAI
monitor.register_serving_model(
    model_path=export_path,
    version=version,
    serving_config={
        'signature_name': 'serving_default',
        'input_name': 'input',
        'output_name': 'output'
    }
)
```

### Monitoring Served Predictions

```python
import requests
import json

def predict_with_serving(inputs):
    \"\"\"Make prediction using TensorFlow Serving\"\"\"
    # Prepare request
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": inputs.tolist()
    })

    # Call serving endpoint
    response = requests.post(
        'http://localhost:8501/v1/models/my_model:predict',
        data=data,
        headers={"content-type": "application/json"}
    )

    predictions = response.json()['predictions']

    # Log to WhiteBoxAI
    monitor.log_batch(
        inputs=inputs,
        predictions=predictions,
        metadata={'serving': True}
    )

    return predictions

# Use
predictions = predict_with_serving(X_test[:10])
```

---

## Best Practices

### 1. Model Checkpointing

```python
# Create checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'checkpoints/model_{epoch:02d}_{val_loss:.2f}.h5',
    save_best_only=True,
    monitor='val_loss'
)

# Custom callback to log checkpoints
class CheckpointLogger(keras.callbacks.Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') == min(self.model.history.history.get('val_loss', [float('inf')])):
            self.monitor.log_checkpoint(
                epoch=epoch,
                checkpoint_path=f'checkpoints/model_{epoch:02d}_{logs["val_loss"]:.2f}.h5',
                metrics=logs
            )

# Train with checkpointing
model.fit(
    X_train, y_train,
    epochs=50,
    callbacks=[
        checkpoint_cb,
        CheckpointLogger(monitor),
        WhiteBoxAICallback(monitor)
    ]
)
```

### 2. TensorBoard Integration

```python
# TensorBoard callback
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True
)

# Combine with WhiteBoxAI
model.fit(
    X_train, y_train,
    epochs=50,
    callbacks=[
        WhiteBoxAICallback(monitor),
        tensorboard_cb
    ]
)
```

### 3. Mixed Precision Training

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build model (outputs should still be float32)
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(10)(x)
outputs = keras.layers.Activation('softmax', dtype='float32')(x)

model = keras.Model(inputs, outputs)

# Compile with loss scaling
optimizer = keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with monitoring
monitor = KerasMonitor(model, client, "mixed_precision_model")
model.fit(X_train, y_train, callbacks=[WhiteBoxAICallback(monitor)])
```

### 4. Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Train with augmented data
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[WhiteBoxAICallback(monitor)]
)
```

---

## Complete Examples

### Example 1: Image Classification (CIFAR-10)

```python
import tensorflow as tf
from tensorflow import keras
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.tensorflow import KerasMonitor, WhiteBoxAICallback

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Initialize monitoring
client = WhiteBoxAI(api_key='your-api-key')
monitor = KerasMonitor(
    model=model,
    client=client,
    model_name="cifar10_cnn",
    model_type="classification",
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']
)

# Set baseline
monitor.set_baseline(X_train[:1000], y_train[:1000])

# Callbacks
callbacks = [
    WhiteBoxAICallback(monitor, log_frequency=1),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
]

# Train
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Log final predictions
predictions = model.predict(X_test)
monitor.log_batch(
    inputs=X_test,
    predictions=predictions,
    actuals=y_test,
    metadata={'phase': 'final_evaluation'}
)

# Save model
model.save('models/cifar10_cnn')
print(f"Model saved and registered with ID: {monitor.model_id}")
```

### Example 2: Time Series Forecasting

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Synthetic data (replace with real time series)
time_series = np.sin(np.arange(0, 1000) * 0.1) + np.random.normal(0, 0.1, 1000)

# Normalize
scaler = MinMaxScaler()
time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

# Create sequences
seq_length = 50
X, y = create_sequences(time_series_scaled, seq_length)

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# Build LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Monitor
monitor = KerasMonitor(
    model=model,
    client=client,
    model_name="lstm_forecaster",
    model_type="regression"
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[WhiteBoxAICallback(monitor)],
    verbose=1
)

# Predict and log
predictions = model.predict(X_test)
monitor.log_batch(
    inputs=X_test,
    predictions=predictions,
    actuals=y_test
)

print(f"LSTM model ID: {monitor.model_id}")
```

---

## Troubleshooting

### Issue: Callback not logging

**Solution**: Ensure callback is in callbacks list

```python
# Correct
model.fit(X, y, callbacks=[WhiteBoxAICallback(monitor)])

# Incorrect (callback not passed)
model.fit(X, y)
```

### Issue: Out of memory errors

**Solution**: Use gradient checkpointing or reduce batch size

```python
# Reduce batch size
model.fit(X_train, y_train, batch_size=16)  # Instead of 32

# Or use mixed precision
mixed_precision.set_global_policy('mixed_float16')
```

### Issue: Model not serializing

**Solution**: Use SavedModel format instead of H5

```python
# SavedModel (recommended)
model.save('model_dir')

# Not H5 for custom models
# model.save('model.h5')  # May fail with custom layers
```

---

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [WhiteBoxAI API Reference](https://docs.whiteboxai.com/api)
- [Best Practices](https://docs.whiteboxai.com/best-practices)

---

*Last Updated: December 29, 2024*

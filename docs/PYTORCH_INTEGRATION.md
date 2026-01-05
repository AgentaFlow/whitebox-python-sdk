# PyTorch Integration Guide

Complete guide for integrating WhiteBoxAI monitoring with PyTorch models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Models](#supported-models)
3. [Training Loop Integration](#training-loop-integration)
4. [TorchScript Models](#torchscript-models)
5. [DataLoader Integration](#dataloader-integration)
6. [GPU Monitoring](#gpu-monitoring)
7. [Distributed Training](#distributed-training)
8. [Best Practices](#best-practices)
9. [Complete Examples](#complete-examples)

---

## Quick Start

```python
import torch
import torch.nn as nn
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.pytorch import TorchMonitor

# Initialize WhiteBoxAI
client = WhiteBoxAI(api_key='your-api-key')

# Define your PyTorch model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create and register model
model = SimpleNN()
monitor = TorchMonitor(
    model=model,
    client=client,
    model_name="simple_neural_network",
    model_type="regression"
)

# Make predictions with automatic logging
X = torch.randn(1, 10)
prediction = monitor.predict(X)
```

---

## Supported Models

WhiteBoxAI supports all PyTorch model types:

### Sequential Models

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

monitor = TorchMonitor(model, client, "sequential_model")
```

### Custom Modules

```python
class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

model = CustomModel(10, 64, 1)
monitor = TorchMonitor(model, client, "custom_model")
```

### Classification Models

```python
class Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits

model = Classifier(num_classes=3)
monitor = TorchMonitor(
    model,
    client,
    "classifier",
    model_type="classification",
    classes=['Class A', 'Class B', 'Class C']
)
```

---

## Training Loop Integration

### Basic Training with Monitoring

```python
from whiteboxai.integrations.pytorch import TrainingMonitor

# Create training monitor
training_monitor = TrainingMonitor(
    model=model,
    client=client,
    model_name="training_model",
    log_frequency=100  # Log every 100 batches
)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log training metrics
        training_monitor.log_batch(
            epoch=epoch,
            batch_idx=batch_idx,
            loss=loss.item(),
            outputs=output.detach(),
            targets=target
        )

        epoch_loss += loss.item()

    # Log epoch metrics
    training_monitor.log_epoch(
        epoch=epoch,
        train_loss=epoch_loss / len(train_loader),
        learning_rate=optimizer.param_groups[0]['lr']
    )
```

### Validation Monitoring

```python
# Validation loop with monitoring
model.eval()
val_predictions = []
val_targets = []
val_loss = 0

with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()

        val_predictions.append(output)
        val_targets.append(target)

# Concatenate all predictions
val_predictions = torch.cat(val_predictions)
val_targets = torch.cat(val_targets)

# Log validation results
training_monitor.log_validation(
    epoch=epoch,
    val_loss=val_loss / len(val_loader),
    predictions=val_predictions,
    actuals=val_targets
)
```

### Early Stopping with Drift Detection

```python
from whiteboxai.monitoring import EarlyStopping

early_stopping = EarlyStopping(
    monitor=training_monitor,
    patience=5,
    min_delta=0.001
)

for epoch in range(num_epochs):
    # Train model...
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Check early stopping
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch}")
        break

    # Log metrics
    training_monitor.log_epoch(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss
    )
```

---

## TorchScript Models

### Tracing Models

```python
import torch.jit

# Create example input
example_input = torch.randn(1, 10)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save traced model
torch.jit.save(traced_model, "model_traced.pt")

# Register with WhiteBoxAI
monitor = TorchMonitor(
    model=traced_model,
    client=client,
    model_name="traced_model",
    is_traced=True
)
```

### Scripting Models

```python
# Script the model
scripted_model = torch.jit.script(model)

# Save scripted model
torch.jit.save(scripted_model, "model_scripted.pt")

# Register with WhiteBoxAI
monitor = TorchMonitor(
    model=scripted_model,
    client=client,
    model_name="scripted_model",
    is_scripted=True
)
```

### Loading TorchScript Models

```python
# Load TorchScript model
loaded_model = torch.jit.load("model_traced.pt")

# Create monitor for loaded model
monitor = TorchMonitor(
    model=loaded_model,
    client=client,
    model_name="loaded_traced_model"
)

# Make predictions
predictions = monitor.predict(X)
```

---

## DataLoader Integration

### Batch Prediction with DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset and dataloader
dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Predict and log batches
model.eval()
all_predictions = []
all_actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        # Make predictions
        predictions = model(batch_X)

        # Log batch
        monitor.log_batch(
            inputs=batch_X.numpy(),
            predictions=predictions.numpy(),
            actuals=batch_y.numpy()
        )

        all_predictions.append(predictions)
        all_actuals.append(batch_y)

# Concatenate results
all_predictions = torch.cat(all_predictions)
all_actuals = torch.cat(all_actuals)
```

### Custom DataLoader Monitoring

```python
class MonitoredDataLoader:
    def __init__(self, dataloader, monitor):
        self.dataloader = dataloader
        self.monitor = monitor

    def __iter__(self):
        for batch_idx, (data, target) in enumerate(self.dataloader):
            # Yield data
            yield data, target

            # Optional: Monitor data quality
            self.monitor.log_data_quality(
                batch_idx=batch_idx,
                data_stats={
                    'mean': data.mean().item(),
                    'std': data.std().item(),
                    'min': data.min().item(),
                    'max': data.max().item()
                }
            )

# Usage
monitored_loader = MonitoredDataLoader(train_loader, training_monitor)
for data, target in monitored_loader:
    # Training code...
    pass
```

---

## GPU Monitoring

### GPU Metrics Tracking

```python
from whiteboxai.monitoring import GPUMonitor

# Enable GPU monitoring
gpu_monitor = GPUMonitor()

# Training with GPU monitoring
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # Track GPU utilization
        gpu_metrics = gpu_monitor.get_metrics()

        # Train step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log with GPU metrics
        training_monitor.log_batch(
            loss=loss.item(),
            gpu_memory_used=gpu_metrics['memory_used'],
            gpu_utilization=gpu_metrics['utilization']
        )
```

### Memory Optimization Tracking

```python
def track_memory_usage():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }
    return {}

# Log memory usage
for epoch in range(num_epochs):
    # Training...

    memory_stats = track_memory_usage()
    training_monitor.log_epoch(
        epoch=epoch,
        **memory_stats
    )

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

---

## Distributed Training

### Data Parallel

```python
import torch.nn as nn

# Wrap model in DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model_name = "data_parallel_model"
else:
    model_name = "single_gpu_model"

# Create monitor
monitor = TorchMonitor(
    model=model.module if isinstance(model, nn.DataParallel) else model,
    client=client,
    model_name=model_name
)
```

### Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = SimpleNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Only rank 0 logs to WhiteBoxAI
    if rank == 0:
        monitor = TorchMonitor(
            model=ddp_model.module,
            client=client,
            model_name="ddp_model"
        )

    # Training loop
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Log only from rank 0
            if rank == 0:
                monitor.log_batch(
                    loss=loss.item(),
                    rank=rank
                )

    cleanup()

# Launch DDP
import torch.multiprocessing as mp

world_size = torch.cuda.device_count()
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

---

## Best Practices

### 1. Model Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'whiteboxai_model_id': monitor.model_id
    }
    torch.save(checkpoint, filepath)

    # Log checkpoint to WhiteBoxAI
    monitor.log_checkpoint(
        epoch=epoch,
        checkpoint_path=filepath
    )

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['whiteboxai_model_id']
```

### 2. Gradient Monitoring

```python
def log_gradients(model, monitor):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    monitor.log_metrics({
        'gradient_norm': total_norm
    })

    return total_norm

# Use in training loop
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Log gradients before clipping
        grad_norm = log_gradients(model, training_monitor)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
```

### 3. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Step scheduler
    scheduler.step(val_loss)

    # Log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    training_monitor.log_epoch(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=current_lr
    )
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Log metrics
        training_monitor.log_batch(
            loss=loss.item(),
            scale=scaler.get_scale()
        )
```

---

## Complete Examples

### Example 1: Binary Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from whiteboxai import WhiteBoxAI
from whiteboxai.integrations.pytorch import TorchMonitor

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# Define model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize model and optimizer
model = BinaryClassifier(input_dim=20)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Initialize WhiteBoxAI monitor
client = WhiteBoxAI(api_key='your-api-key')
monitor = TorchMonitor(
    model=model,
    client=client,
    model_name="pytorch_binary_classifier",
    model_type="classification",
    classes=['Negative', 'Positive']
)

# Set baseline
monitor.set_baseline(X_train_scaled, y_train)

# Training loop
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    # Mini-batch training
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Log epoch metrics
    avg_loss = epoch_loss / (len(X_train_t) // batch_size)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation with monitoring
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    test_predictions = (test_outputs > 0.5).float()

    # Log predictions
    for i in range(len(X_test_t)):
        monitor.log_prediction(
            inputs=X_test_scaled[i],
            prediction=test_predictions[i].item(),
            actual=y_test[i],
            probability=test_outputs[i].item()
        )

print(f"Model registered with ID: {monitor.model_id}")
```

### Example 2: Multi-class Classification with CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Initialize monitor
monitor = TorchMonitor(
    model=model,
    client=client,
    model_name="mnist_cnn",
    model_type="classification",
    classes=[str(i) for i in range(10)]
)

# Training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Test with monitoring
def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Log predictions (sample every 10th batch)
            if correct % 10 == 0:
                monitor.log_batch(
                    inputs=data.cpu().numpy(),
                    predictions=pred.cpu().numpy(),
                    actuals=target.cpu().numpy()
                )

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

# Run training
for epoch in range(1, 11):
    train(epoch)
    test()
```

### Example 3: Regression with Custom Loss

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_regression

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)

# Define model
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze()

# Custom loss (Huber loss)
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = pred - target
        is_small_error = torch.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(is_small_error, squared_loss, linear_loss).mean()

# Initialize
model = Regressor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = HuberLoss(delta=1.0)

# Monitor
monitor = TorchMonitor(
    model=model,
    client=client,
    model_name="pytorch_regressor_huber",
    model_type="regression"
)

# Set baseline
monitor.set_baseline(X_train_scaled, y_train_scaled)

# Training
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / (len(X_train_t) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Evaluate and log predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t)

    # Log all test predictions
    monitor.log_batch(
        inputs=X_test_scaled,
        predictions=test_predictions.numpy(),
        actuals=y_test_scaled
    )

print(f"Model ID: {monitor.model_id}")
```

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size or enable gradient checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        return checkpoint(self.heavy_computation, x)
```

### Issue: Slow logging

**Solution**: Enable async logging or increase batch size

```python
monitor = TorchMonitor(
    model=model,
    client=client,
    model_name="model",
    async_logging=True,
    batch_size=100
)
```

### Issue: Model not serializing

**Solution**: Use state dict instead of full model

```python
# Save only state dict
torch.save(model.state_dict(), 'model_weights.pth')

# Register with WhiteBoxAI
monitor.register_model(
    model_path='model_weights.pth',
    save_weights_only=True
)
```

---

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [WhiteBoxAI API Reference](https://docs.whiteboxai.com/api)
- [Best Practices Guide](https://docs.whiteboxai.com/best-practices)

---

*Last Updated: December 29, 2024*

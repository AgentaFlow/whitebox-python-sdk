"""
PyTorch Integration Example

This example demonstrates monitoring PyTorch models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.pytorch import TorchMonitor


class SimpleClassifier(nn.Module):
    """Simple feedforward classifier."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, epochs=5):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def main():
    # Generate sample data
    print("Generating sample data...")
    torch.manual_seed(42)
    n_samples = 1000
    input_dim = 20
    output_dim = 2

    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))

    # Create data loader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    print("Creating model...")
    model = SimpleClassifier(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)

    # Train model
    print("\nTraining model...")
    train_model(model, train_loader, epochs=5)

    # Setup monitoring
    print("\nSetting up WhiteBoxXAI monitoring...")
    client = WhiteBoxXAI(api_key="your-api-key")
    monitor = TorchMonitor(client, model=model)

    # Register model
    model_id = monitor.register_from_model(
        name="pytorch_classifier",
        model_type="classification",
        version="1.0.0",
    )
    print(f"Model registered with ID: {model_id}")

    # Wrap model for automatic monitoring
    print("\nWrapping model...")
    monitored_model = monitor.wrap_model(model)

    # Make predictions (automatically logged)
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(10, input_dim)
        outputs = monitored_model(test_input)
        predictions = torch.argmax(outputs, dim=1)
        print(f"Predictions: {predictions}")

    # Manual logging (without wrapper)
    print("\nManual prediction logging...")
    with torch.no_grad():
        test_input = torch.randn(5, input_dim)
        outputs = model(test_input)

        monitor.log_prediction(
            inputs=test_input[0].numpy().tolist(),
            output=outputs[0].numpy().tolist(),
            metadata={"batch_id": "manual_test"},
        )

    # Close client
    client.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()

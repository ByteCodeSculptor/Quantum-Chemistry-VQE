# src/qml/train_qml_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.qml.qml_model import HybridQMLModel
from src import config

def generate_synthetic_data(n_samples=100):
    """
    Generates synthetic data for training.
    In a real-world application, this data would come from accurate quantum
    chemistry simulations, thus imbuing the AI with a foundational
    understanding of physics[cite: 29].
    """
    # Example: learn the function y = sin(pi*x1) * cos(pi*x2)
    X = torch.rand((n_samples, 2)) * 2 - 1 # features in [-1, 1]
    y = torch.sin(torch.pi * X[:, 0]) * torch.cos(torch.pi * X[:, 1])
    y = y.unsqueeze(1)
    return X, y

if __name__ == "__main__":
    print("--- Phase 2.2: Training Quantum-Enhanced ML Model ---")

    # 1. Generate Data and create DataLoader
    X_train, y_train = generate_synthetic_data()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 2. Initialize Model, Loss, and Optimizer
    model = HybridQMLModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.QML_LR)

    print("Starting training...")
    # 3. Training Loop
    for epoch in range(config.QML_EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{config.QML_EPOCHS}], Loss: {avg_loss:.6f}")

    # 4. Save the trained model
    torch.save(model.state_dict(), config.QML_MODEL_PATH)
    print(f"\nTraining complete. Model saved to {config.QML_MODEL_PATH}")
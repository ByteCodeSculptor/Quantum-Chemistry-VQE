# src/qml/qml_model.py

import pennylane as qml
import torch
import torch.nn as nn
from src import config

# Define the quantum device
dev = qml.device("default.qubit", wires=config.QML_N_QUBITS)

# Define the quantum node (circuit)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encoding layer: Embed classical data into quantum state
    qml.AngleEmbedding(inputs, wires=range(config.QML_N_QUBITS))

    # --- UPGRADE 1: Use a more powerful, "Strongly Entangling" quantum circuit ---
    # This circuit is more expressive and better at avoiding "barren plateaus,"
    # a common problem where simple quantum models stop learning.
    qml.StronglyEntanglingLayers(weights, wires=range(config.QML_N_QUBITS))

    # Measurement: Return expectation value of Pauli-Z on the first qubit
    return qml.expval(qml.PauliZ(0))

class HybridQMLModel(nn.Module):
    """
    A hybrid model combining classical pre-processing layers with a
    quantum circuit layer. This demonstrates a practical hybrid workflow.
    """
    def __init__(self):
        super().__init__()
        # --- UPGRADE 2: Enhance the classical layers to give the model more power ---
        # A more powerful classical "brain" can find more complex patterns
        # in the data before passing it to the quantum circuit.
        self.classical_head = nn.Sequential(
            nn.Linear(2, 8),      # More neurons (from 4 to 8)
            nn.ReLU(),
            nn.Linear(8, 4)       # An extra layer to add depth
        )

        # --- UPGRADE 3: Update weight shapes for the new, more complex quantum layer ---
        # This correctly calculates the number of trainable parameters for the
        # StronglyEntanglingLayers.
        weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=config.QML_N_LAYERS, n_wires=config.QML_N_QUBITS)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classical output layer
        self.classical_tail = nn.Linear(1, 1) # Quantum output -> final prediction

    def forward(self, x):
        # 1. Classical pre-processing
        x = self.classical_head(x)
        # 2. Quantum processing
        x = self.quantum_layer(x)
        # 3. Reshape for classical output layer
        x = x.unsqueeze(1)
        # 4. Classical post-processing
        x = self.classical_tail(x)
        return x


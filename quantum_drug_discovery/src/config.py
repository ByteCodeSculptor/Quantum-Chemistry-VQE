# src/config.py

import os

# --- Project Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
QML_MODEL_PATH = os.path.join(RESULTS_DIR, "qml_model.pth")

# --- VQE Simulation Parameters ---
VQE_STEPS = 50
VQE_LEARNING_RATE = 0.1 #How aggressively the optimizer adjusts the circuit parameters.
VQE_BASIS_SET = "sto-3g" # A minimal basis set for speed
VQE_ACTIVE_ELECTRONS = 2
VQE_ACTIVE_ORBITALS = 2

# --- QML Parameters ---
QML_N_QUBITS = 4
# --- FIX: Increase the number of layers for a more powerful quantum circuit ---
QML_N_LAYERS = 6
QML_EPOCHS = 30
QML_LR = 0.05


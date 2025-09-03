# src/vqe/run_vqe_poc.py

import pennylane as qml
from pennylane import numpy as np
import argparse
import time
import os

from src.molecular_utils import get_hamiltonian
from src import config

def run_vqe(hamiltonian, n_qubits):
    """
    Performs the VQE optimization to find the ground state energy.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    # --- FIX 1: Generate excitations and HF state correctly ---
    # The AllSinglesDoubles operator requires the Hartree-Fock (HF) state and a list
    # of single and double excitations, which we generate here.
    electrons = config.VQE_ACTIVE_ELECTRONS
    hf_state = qml.qchem.hf_state(electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(electrons, n_qubits)

    # Define the quantum circuit (ansatz)
    def circuit(params, wires):
        # We pass the correctly generated hf_state, singles, and doubles to the operator.
        qml.AllSinglesDoubles(params, wires=wires,
                              hf_state=hf_state,
                              singles=singles,
                              doubles=doubles)

    # Define the cost function: the expectation value of the Hamiltonian
    @qml.qnode(dev)
    def cost_fn(params):
        circuit(params, wires=range(n_qubits))
        return qml.expval(hamiltonian)

    # Classical optimizer
    optimizer = qml.AdamOptimizer(stepsize=config.VQE_LEARNING_RATE)

    # --- FIX 2: Calculate number of parameters robustly ---
    # Instead of using a deprecated .shape() method, we calculate the number
    # of parameters directly from the length of the excitation lists.
    n_params = len(singles) + len(doubles)
    # The 'requires_grad=True' is good practice for PennyLane's optimizers.
    params = np.random.uniform(0, 2 * np.pi, n_params, requires_grad=True)
    
    energy_history = []
    print("Starting VQE optimization...")
    start_time = time.time()

    for step in range(config.VQE_STEPS):
        params, energy = optimizer.step_and_cost(cost_fn, params)
        energy_history.append(energy)
        if step % 5 == 0:
            print(f"Step {step:3d}: Energy = {energy:.8f} Ha")

    end_time = time.time()
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")
    print(f"Final ground state energy: {energy:.8f} Ha")

    return energy_history, energy


if __name__ == "__main__":
    # We can simplify the argument parser since we only run H2 now.
    parser = argparse.ArgumentParser(description="Run VQE for the built-in H2 molecule.")
    parser.add_argument("--molecule", type=str, default='h2', help="Molecule name (ignored, always uses H2).")
    args = parser.parse_args()

    print(f"--- Phase 1: VQE Proof of Concept for H2 ---")

    hamiltonian, n_qubits = get_hamiltonian(args.molecule)
    print(f"Qubits required for H2 simulation: {n_qubits}")

    history, final_energy = run_vqe(hamiltonian, n_qubits)
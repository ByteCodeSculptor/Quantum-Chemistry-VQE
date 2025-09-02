# src/screening/run_virtual_screening.py

import os
import time
from src.molecular_utils import get_hamiltonian
from src.vqe.run_vqe_poc import run_vqe
from src import config

def screen_molecules():
    """
    Iterates through all molecules in the data directory, calculates their
    ground state energy using VQE, and ranks them. This simulates a
    high-fidelity virtual screening workflow[cite: 54].
    """
    molecule_files = [f.replace('.xyz', '') for f in os.listdir(config.DATA_DIR) if f.endswith('.xyz')]
    results = {}

    print(f"Found {len(molecule_files)} molecules to screen: {molecule_files}\n")

    for molecule in molecule_files:
        print(f"--- Screening Molecule: {molecule.upper()} ---")
        try:
            start_time = time.time()
            hamiltonian, n_qubits = get_hamiltonian(molecule)
            print(f"Hamiltonian created. Qubits: {n_qubits}")

            _, final_energy = run_vqe(hamiltonian, n_qubits)
            results[molecule] = final_energy

            end_time = time.time()
            print(f"--- Finished screening {molecule.upper()} in {end_time - start_time:.2f}s ---\n")

        except Exception as e:
            print(f"Could not process {molecule}. Error: {e}\n")

    return results

def save_results(ranked_results):
    """Saves the ranked screening results to a log file."""
    output_path = os.path.join(config.RESULTS_DIR, "screening_results.log")
    with open(output_path, 'w') as f:
        f.write("--- Virtual Screening Results ---\n")
        f.write("Ranked by ground state energy (lower is more stable):\n\n")
        for i, (molecule, energy) in enumerate(ranked_results):
            f.write(f"{i+1}. {molecule.upper():<10} Energy: {energy:.8f} Ha\n")
    print(f"Screening results saved to {output_path}")


if __name__ == "__main__":
    print("--- Phase 2.1: High-Fidelity Virtual Screening ---")
    all_results = screen_molecules()

    if all_results:
        # Sort molecules by energy (lower is better)
        sorted_results = sorted(all_results.items(), key=lambda item: item[1])

        print("\n--- Final Ranking ---")
        print("Molecule      Energy (Ha)")
        print("---------------------------")
        for molecule, energy in sorted_results:
            print(f"{molecule.upper():<12} {energy:.8f}")

        save_results(sorted_results)
    else:
        print("No molecules were successfully screened.")
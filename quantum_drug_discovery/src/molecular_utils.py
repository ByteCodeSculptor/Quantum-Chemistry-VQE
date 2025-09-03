# src/molecular_utils.py

import pennylane as qml
from pennylane import numpy as np

def get_hamiltonian(molecule_name: str = "H2"):
    """
    Returns a pre-defined Hamiltonian for the H2 molecule.

    This function serves as a replacement for the original file-reading
    and chemistry calculation, allowing the project to run without the
    pyscf dependency. It will always return the Hamiltonian for H2,
    regardless of the input 'molecule_name'.
    """
    # These are the standard pre-calculated values for H2 at its
    # typical bond length (0.74 angstroms). PennyLane provides this
    # as a built-in demo dataset.
    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

    # useing a built-in dataset and does NOT require pyscf.
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
    )

    print(f"Note: Using built-in H2 Hamiltonian. Input '{molecule_name}' is ignored.")
    return hamiltonian, n_qubits


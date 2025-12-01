from shades.stabilizer import StabilizerState
from typing import List
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

class StabilizerSubspace:

    S: NDArray[np.complex128]
    H: NDArray[np.complex128]
    N: int

    def __init__(self, states: List[StabilizerState], pauli_hamiltonian: List[tuple[np.complex128, str]]):
        self.S = self.build_overlap_matrix(states)
        self.H = self.build_hamiltonian_matrix(states, pauli_hamiltonian)
        self.N = len(states)

    def build_overlap_matrix(states: List[StabilizerState]):
        
        N = len(states)
        S = np.empty((N, N), dtype=np.complex128)

        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states[i::]):
                S[i,j] = s_i.inner_product(s_j, phase=True)

        return S
    
    def build_hamiltonian_matrix(states: List[StabilizerState], pauli_hamiltonian: List[tuple[np.complex128, str]]):
        
        N = len(states)
        H = np.empty((N, N), dtype=np.complex128)

        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states[i::]):
                E = s_j.compute_eproj([s_i], pauli_hamiltonian)
                H[i, j] += E

        return H

    def optimize_coefficients(self, reg=1e-10):
        """Solve Hc = ESc with regularization."""
        S_reg = self.S + reg * np.eye(self.n_states)
        eigvals, eigvecs = eigh(self.H, S_reg)
        self.energy = eigvals[0].real
        self.coeffs = eigvecs[:, 0]
        return self.energy, self.coeffs


class BitstringSubspace:

    pass


if __name__ == "__main__":

    from qiskit_nature.second_q.operators import FermionicOp
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from stim import stim


    mapper = JordanWignerMapper()
    op3 = FermionicOp({
        "+_0 -_0": 1.0,
        "+_1 -_1": -1.0,
        "+_0 -_1": -0.5,
        "+_1 -_0": -0.5,
        "+_0 +_1 -_1 -_0": 0.25
    }, num_spin_orbitals=2)

    qubit_op = mapper.map(op3)

    hamil = [(coeff, label) for label, coeff in qubit_op.label_iter()]

    from shades.shadows import CliffordShadow, CliffordGroup

    ensemble = CliffordGroup(2)
    tab = ensemble.generate_sample()

    stab = StabilizerState.from_stim_tableau(tab)

    stab.apply_hamiltonian(hamil)

    CliffordShadow.sample_state()



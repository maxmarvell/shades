from shades.stabilizer.stabilizer_state import StabilizerState
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
import stim
from qiskit.quantum_info import Statevector
from shades.utils import Bitstring
from shades.shadows import ComputationalShadow, CliffordShadow

def compose_tableau_bitstring(T_U: stim.Tableau, b: Bitstring):
    stabs = b.to_stabilizers()
    T_b = stim.Tableau.from_stabilizers(stabs)
    return T_U * T_b

def stabilizer_from_stim_tableau(tableau: stim.Tableau):
    n = len(tableau)
    generator_matrix = np.zeros((n , n+1), dtype=np.int16)

    for i in range(n):
        pauli_string = tableau.z_output(i)

        sign = pauli_string.sign.real
        generator_matrix[i, -1] = int(sign)

        for j in range(n):
            pauli = pauli_string[j]
            generator_matrix[i, j] = int(pauli)

    return StabilizerState(generator_matrix, find_basis_state=True)

ComputationalBasisState = Bitstring

class StabilizerSubspace:

    S: NDArray[np.complex128]
    H: NDArray[np.complex128]
    N: int

    def __init__(self, states: List[StabilizerState], pauli_hamiltonian: List[tuple[np.complex128, str]]):
        self.states = states
        self.pauli_hamiltonian = pauli_hamiltonian
        self.S = self.build_overlap_matrix(states)
        self.H = self.build_hamiltonian_matrix(states, pauli_hamiltonian)
        self.N = len(states)

    @property
    def n_samples(self):
        return len(self.states)

    @classmethod
    def from_state(cls, state: Statevector, n_samples: int, pauli_hamiltonian: List[tuple[np.complex128, str]], *, max_attempts: int = 1000) -> 'StabilizerSubspace':
        states = []
        n_sampled = 0
        i = 0
        while n_sampled < n_samples and i < max_attempts:
            sample = CliffordShadow.sample_state(state)
            stabilizer_state = stabilizer_from_stim_tableau(sample)
            if stabilizer_state not in states:
                states.append(stabilizer_state)
                n_sampled += 1
            i += 1
        return cls(states, pauli_hamiltonian)

    @staticmethod
    def build_overlap_matrix(states: List[StabilizerState]) -> NDArray[np.complex128]:

        N = len(states)
        S = np.zeros((N, N), dtype=np.complex128)

        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states):
                S[i,j] = s_i.inner_product(s_j)

        return S

    @staticmethod
    def build_hamiltonian_matrix(states: List[StabilizerState], pauli_hamiltonian: List[tuple[np.complex128, str]]) -> NDArray[np.complex128]:

        N = len(states)
        H = np.zeros((N, N), dtype=np.complex128)

        for j, s_j in enumerate(states):
            projected_states = s_j.apply_hamiltonian(pauli_hamiltonian)
            for s in projected_states:
                for i, s_i in enumerate(states):
                    E = s_i.inner_product(s, phase=True) * projected_states[s]
                    H[i, j] += E

        return H

    def optimize_coefficients(self, reg: Optional[float] = None):
        if reg:
            S = self.S + reg * np.eye(self.N)
        else:
            S = self.S
        eigvals, eigvecs = eigh(self.H, S)
        self.energy = eigvals[0].real
        self.coeffs = eigvecs[:, 0]
        return self.energy, self.coeffs
    
    def add_samples_from_state(self, state: Statevector, n_samples: int) -> None:
        for _ in range(n_samples):
            sample = CliffordShadow.sample_state(state)
            stabilizer = stabilizer_from_stim_tableau(sample)
            if stabilizer not in self.states:
                self.states.append(stabilizer)

        self.S = self.build_overlap_matrix(self.states)
        self.H = self.build_hamiltonian_matrix(self.states, self.pauli_hamiltonian)

class ComputationalSubspace:

    H: NDArray[np.complex128]
    N: int

    def __init__(self, states: List[ComputationalBasisState], pauli_hamiltonian: List[tuple[np.complex128, str]]):
        self.H = self.build_hamiltonian_matrix(states, pauli_hamiltonian)
        self.N = len(states)

    @staticmethod
    def build_hamiltonian_matrix(states: List[ComputationalBasisState], pauli_hamiltonian: List[tuple[np.complex128, str]]) -> NDArray[np.complex128]:

        N = len(states)
        H = np.zeros((N, N), dtype=np.complex128)

        stabilizers = [b.to_stabilizers() for b in states]
        tableaus = [stim.Tableau.from_stabilizers(s) for s in stabilizers]
        stabilizers_states = [stabilizer_from_stim_tableau(tab) for tab in tableaus]

        for j, s_j in enumerate(stabilizers_states):
            projected_states = s_j.apply_hamiltonian(pauli_hamiltonian)
            for s in projected_states:
                for i, s_i in enumerate(stabilizers_states):
                    E = s_i.inner_product(s, phase=True) * projected_states[s]
                    H[i, j] += E

        return H

    @classmethod
    def from_state(cls, state: Statevector, n_samples: int, pauli_hamiltonian: List[tuple[np.complex128, str]], *, max_attempts: int = 1000):
        states = []
        n_sampled = 0
        i = 0
        while n_sampled < n_samples and i < max_attempts:
            sample = ComputationalShadow.sample_state(state)
            if sample not in states:
                states.append(sample)
                n_sampled += 1
            i += 1
        return cls(states, pauli_hamiltonian)

    
    def optimize_coefficients(self):
        """Solve Hc = ESc with regularization."""
        eigvals, eigvecs = eigh(self.H)
        self.energy = eigvals[0].real
        self.coeffs = eigvecs[:, 0]
        return self.energy, self.coeffs

if __name__ == "__main__":

    from qiskit_nature.second_q.operators import FermionicOp
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit.quantum_info import Statevector
    from scipy.linalg import eigh
    from shades.shadows import CliffordShadow

    fermionic_op = FermionicOp({
        "+_0 -_2": -1.0, "+_2 -_0": -1.0,
        "+_1 -_3": -1.0, "+_3 -_1": -1.0,
        "+_0 -_0 +_1 -_1": 2.0,
        "+_2 -_2 +_3 -_3": 2.0,
    }, num_spin_orbitals=4)

    qubit_op = JordanWignerMapper().map(fermionic_op)
    pauli_hamiltonian = [(coeff, label) for label, coeff in qubit_op.label_iter()]

    matrix = qubit_op.to_matrix()
    eigenvalues, eigenvectors = eigh(matrix)

    ground_energy = eigenvalues[0]
    ground_state = Statevector(eigenvectors[:, 0])

    N = 20

    sub = StabilizerSubspace.from_state(ground_state, N, pauli_hamiltonian)

    # check S
    assert np.allclose(np.diag(sub.S), np.ones(N)), 'Not normalized!'
    assert np.all(np.abs(sub.S) <= 1), 'Not normalized!'
    assert np.allclose(sub.S, sub.S.conj().T), 'S is not Hermitian!'

    # check H
    assert np.allclose(sub.H, sub.H.conj().T), 'H is not Hermitian!'

    E, coeffs = sub.optimize_coefficients()

    print(E, coeffs)
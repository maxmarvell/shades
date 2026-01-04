def ising_chain(n_qubits, J=1.0, h=1.0, periodic=False):
    """H = -J Σ Z_i Z_{i+1} - h Σ X_i"""
    terms = []
    
    for i in range(n_qubits - 1 + periodic):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        pauli[(i + 1) % n_qubits] = 'Z'
        terms.append((-J, ''.join(pauli)))
    
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'X'
        terms.append((-h, ''.join(pauli)))
    
    return terms


def heisenberg_chain(n_qubits, Jx=1.0, Jy=1.0, Jz=1.0, periodic=False):
    """H = Σ (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})"""
    terms = []
    
    for i in range(n_qubits - 1 + periodic):
        j = (i + 1) % n_qubits
        for pauli_char, coeff in [('X', Jx), ('Y', Jy), ('Z', Jz)]:
            if coeff == 0:
                continue
            pauli = ['I'] * n_qubits
            pauli[i] = pauli_char
            pauli[j] = pauli_char
            terms.append((coeff, ''.join(pauli)))
    
    return terms


def toric_code(Lx, Ly):
    """
    Toric code on Lx x Ly lattice with periodic boundaries.
    Qubits live on edges: n_qubits = 2 * Lx * Ly
    H = -Σ A_v (vertex/star) - Σ B_p (plaquette)
    """
    n_qubits = 2 * Lx * Ly
    
    def edge_index(x, y, direction):
        return 2 * ((y % Ly) * Lx + (x % Lx)) + direction
    
    terms = []
    
    for x in range(Lx):
        for y in range(Ly):
            edges = [
                edge_index(x, y, 0),      # right
                edge_index(x - 1, y, 0),  # left
                edge_index(x, y, 1),      # up
                edge_index(x, y - 1, 1),  # down
            ]
            pauli = ['I'] * n_qubits
            for e in edges:
                pauli[e] = 'X'
            terms.append((-1.0, ''.join(pauli)))
    
    for x in range(Lx):
        for y in range(Ly):
            edges = [
                edge_index(x, y, 0),      # bottom
                edge_index(x, y + 1, 0),  # top
                edge_index(x, y, 1),      # left
                edge_index(x + 1, y, 1),  # right
            ]
            pauli = ['I'] * n_qubits
            for e in edges:
                pauli[e] = 'Z'
            terms.append((-1.0, ''.join(pauli)))
    
    return terms


if __name__ == "__main__":

    from shades.stabilizer_subspace import StabilizerSubspace
    from shades.utils import pauli_terms_to_matrix
    from qiskit.quantum_info import Statevector
    from scipy.linalg import eigh

    terms = ising_chain(3, h=0, J=1)

    matrix = pauli_terms_to_matrix(terms)
    eigenvalues, eigenvectors = eigh(matrix)

    ground_energy = eigenvalues[0]
    ground_state = Statevector(eigenvectors[:, 0])

    print(ground_energy)

    ss = StabilizerSubspace.from_state(ground_state, n_samples=3, pauli_hamiltonian=terms)

    print(ss.optimize_coefficients())
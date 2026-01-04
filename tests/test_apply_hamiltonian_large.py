"""Tests for apply_hamiltonian with large and complex Hamiltonians.

Tests realistic quantum chemistry and condensed matter Hamiltonians with
many terms and multi-qubit operators.
"""

import pytest
import numpy as np
import stim
from shades.stabilizer.stabilizer_state import StabilizerState
from shades.stabilizer_subspace import stabilizer_from_stim_tableau


class TestLargeHamiltonians:
    """Test apply_hamiltonian with realistic large Hamiltonians."""

    def test_transverse_field_ising_chain(self):
        """Test transverse-field Ising model: H = -∑ZZ - h∑X."""
        # Create 4-qubit state |0000⟩
        generator_matrix = np.array([
            [3, 0, 0, 0, 1],
            [0, 3, 0, 0, 1],
            [0, 0, 3, 0, 1],
            [0, 0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        # H = -ZZ_I_I - _ZZ_I - __ZZ_I - ___ZZ - X___ - _X__ - __X_ - ___X
        hamiltonian = [
            (-1.0, 'ZZII'),  # Nearest-neighbor interactions
            (-1.0, 'IZZI'),
            (-1.0, 'IIZZ'),
            (-1.0, 'XIII'),  # Transverse field
            (-1.0, 'IXII'),
            (-1.0, 'IIXI'),
            (-1.0, 'IIIX'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Should produce multiple states
        assert len(result) > 0

        # Check total coefficient magnitude
        total_coeff = sum(abs(c)**2 for c in result.values())
        assert total_coeff > 0

        # Verify against exact calculation
        psi = state.get_statevector()

        # Build exact Hamiltonian matrix
        from scipy.sparse import csr_matrix
        n_qubits = 4
        dim = 2**n_qubits

        # Pauli matrices
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])

        def pauli_to_matrix(pauli_string):
            """Convert Pauli string to matrix."""
            pauli_dict = {'I': I, 'X': X, 'Z': Z}
            # Reverse string to match stabilizer qubit ordering (qubit 0 is rightmost)
            pauli_string = pauli_string[::-1]
            result = pauli_dict[pauli_string[0]]
            for p in pauli_string[1:]:
                result = np.kron(result, pauli_dict[p])
            return result

        H_exact = np.zeros((dim, dim), dtype=complex)
        for coeff, pauli_str in hamiltonian:
            H_exact += coeff * pauli_to_matrix(pauli_str)

        # Apply to state
        H_psi_exact = H_exact @ psi

        # Reconstruct from stabilizer results
        H_psi_stab = np.zeros(dim, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        # Compare
        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10), \
            f"Mismatch in H|ψ⟩!\nMax diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}"

    def test_heisenberg_hamiltonian(self):
        """Test Heisenberg model: H = ∑(XX + YY + ZZ)."""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        stabs = [
            stim.PauliString('+XX'),
            stim.PauliString('+ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Heisenberg Hamiltonian on 2 qubits
        hamiltonian = [
            (1.0, 'XX'),
            (1.0, 'YY'),
            (1.0, 'ZZ'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        ZZ = np.kron(Z, Z)

        H_exact = XX + YY + ZZ
        H_psi_exact = H_exact @ psi

        # Reconstruct
        H_psi_stab = np.zeros(4, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_molecular_hamiltonian_h2(self):
        """Test a realistic H2 molecular Hamiltonian."""
        # Simple 4-qubit H2 Hamiltonian in Jordan-Wigner encoding
        # This is a simplified version with representative terms
        hamiltonian = [
            (-1.0523, 'IIII'),  # Nuclear repulsion + constant
            (0.3979, 'ZIIY'),   # One-body terms
            (-0.3979, 'IYII'),
            (0.0113, 'ZXZX'),   # Two-body terms
            (-0.0113, 'ZXIX'),
            (0.1810, 'ZZII'),
        ]

        # Start with Hartree-Fock state |1100⟩ (2 electrons in lowest orbitals)
        generator_matrix = np.array([
            [3, 0, 0, 0, -1],  # Z with eigenvalue -1 → |1⟩
            [0, 3, 0, 0, -1],  # Z with eigenvalue -1 → |1⟩
            [0, 0, 3, 0, 1],   # Z with eigenvalue +1 → |0⟩
            [0, 0, 0, 3, 1],   # Z with eigenvalue +1 → |0⟩
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        def pauli_to_matrix(pauli_string):
            pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
            # Reverse string to match stabilizer qubit ordering (qubit 0 is rightmost)
            pauli_string = pauli_string[::-1]
            result = pauli_dict[pauli_string[0]]
            for p in pauli_string[1:]:
                result = np.kron(result, pauli_dict[p])
            return result

        H_exact = np.zeros((16, 16), dtype=complex)
        for coeff, pauli_str in hamiltonian:
            H_exact += coeff * pauli_to_matrix(pauli_str)

        H_psi_exact = H_exact @ psi

        # Reconstruct
        H_psi_stab = np.zeros(16, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        max_diff = np.max(np.abs(H_psi_exact - H_psi_stab))
        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10), \
            f"Max diff: {max_diff}"

    def test_many_body_hamiltonian(self):
        """Test Hamiltonian with many terms."""
        # GHZ state (|000⟩ + |111⟩)/√2
        stabs = [
            stim.PauliString('+XXX'),
            stim.PauliString('+ZZ_'),
            stim.PauliString('+_ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Hamiltonian with 20+ terms
        hamiltonian = [
            # Single-qubit terms
            (0.5, 'XII'), (0.5, 'IXI'), (0.5, 'IIX'),
            (0.3, 'YII'), (0.3, 'IYI'), (0.3, 'IIY'),
            (0.7, 'ZII'), (0.7, 'IZI'), (0.7, 'IIZ'),
            # Two-qubit terms
            (0.2, 'XXI'), (0.2, 'IXX'), (0.2, 'XIX'),
            (0.1, 'YYI'), (0.1, 'IYY'), (0.1, 'YIY'),
            (0.4, 'ZZI'), (0.4, 'IZZ'), (0.4, 'ZIZ'),
            # Three-qubit terms
            (0.15, 'XXX'),
            (0.25, 'ZZZ'),
            (0.1, 'XYZ'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        def pauli_to_matrix(pauli_string):
            pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
            # Reverse string to match stabilizer qubit ordering (qubit 0 is rightmost)
            pauli_string = pauli_string[::-1]
            result = pauli_dict[pauli_string[0]]
            for p in pauli_string[1:]:
                result = np.kron(result, pauli_dict[p])
            return result

        H_exact = np.zeros((8, 8), dtype=complex)
        for coeff, pauli_str in hamiltonian:
            H_exact += coeff * pauli_to_matrix(pauli_str)

        H_psi_exact = H_exact @ psi

        # Reconstruct
        H_psi_stab = np.zeros(8, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-8)


class TestComplexCoefficients:
    """Test Hamiltonians with complex coefficients."""

    def test_complex_coefficients(self):
        """Test Hamiltonian with complex coefficients."""
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state = StabilizerState(gen_0)

        # H = (1+i)X + (1-i)Z
        hamiltonian = [
            (1+1j, 'X'),
            (1-1j, 'Z'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])

        H_exact = (1+1j)*X + (1-1j)*Z
        H_psi_exact = H_exact @ psi

        H_psi_stab = np.zeros(2, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_imaginary_hamiltonian(self):
        """Test purely imaginary Hamiltonian."""
        # |+⟩ state
        plus_tab = stim.Tableau.from_stabilizers([stim.PauliString('+X')])
        state = stabilizer_from_stim_tableau(plus_tab)

        # H = iY + iZ
        hamiltonian = [
            (1j, 'Y'),
            (1j, 'Z'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        H_exact = 1j*Y + 1j*Z
        H_psi_exact = H_exact @ psi

        H_psi_stab = np.zeros(2, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)


class TestSparseHamiltonians:
    """Test Hamiltonians that produce sparse results."""

    def test_sparse_result(self):
        """Test Hamiltonian where many terms cancel."""
        # Create |0⟩
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state = StabilizerState(gen_0)

        # H = X - X + 2X = 2X (partial cancellation)
        hamiltonian = [
            (1.0, 'X'),
            (-1.0, 'X'),
            (2.0, 'X'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Should give 2|1⟩
        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 2.0)

    def test_complete_cancellation(self):
        """Test Hamiltonian with complete cancellation."""
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state = StabilizerState(gen_0)

        # H = X - X = 0
        hamiltonian = [
            (1.0, 'X'),
            (-1.0, 'X'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Should be empty
        assert len(result) == 0


class TestSymmetryPreservation:
    """Test that Hamiltonians preserve symmetries correctly."""

    def test_symmetric_hamiltonian_on_symmetric_state(self):
        """Test symmetric Hamiltonian on symmetric state."""
        # |+⟩⊗|+⟩ state (symmetric under qubit exchange)
        stabs = [
            stim.PauliString('+X_'),
            stim.PauliString('+_X'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Symmetric Hamiltonian H = XX + YY + ZZ
        hamiltonian = [
            (1.0, 'XX'),
            (1.0, 'YY'),
            (1.0, 'ZZ'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Result should also be symmetric
        # Verify by checking it's a valid quantum state
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        ZZ = np.kron(Z, Z)

        H_exact = XX + YY + ZZ
        H_psi_exact = H_exact @ psi

        # Reconstruct
        H_psi_stab = np.zeros(4, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)


class TestEdgeCases:
    """Test edge cases with large Hamiltonians."""

    def test_identity_heavy_hamiltonian(self):
        """Test Hamiltonian dominated by identity terms."""
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state = StabilizerState(gen_0)

        # H = 100*I + 0.01*X
        hamiltonian = [
            (100.0, 'I'),
            (0.01, 'X'),
        ]

        result = state.apply_hamiltonian(hamiltonian)

        # Should have two terms: 100|0⟩ + 0.01|1⟩
        assert len(result) == 2

    def test_all_pauli_combinations_3qubit(self):
        """Test with all single-qubit Pauli combinations on 3 qubits."""
        # |000⟩ state
        generator_matrix = np.array([
            [3, 0, 0, 1],
            [0, 3, 0, 1],
            [0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        # All single-qubit Pauli terms
        hamiltonian = []
        for i, op in enumerate(['X', 'Y', 'Z']):
            for pos in range(3):
                pauli_str = ['I', 'I', 'I']
                pauli_str[pos] = op
                hamiltonian.append((0.1 * (i+1), ''.join(pauli_str)))

        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        def pauli_to_matrix(pauli_string):
            pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
            # Reverse string to match stabilizer qubit ordering (qubit 0 is rightmost)
            pauli_string = pauli_string[::-1]
            result = pauli_dict[pauli_string[0]]
            for p in pauli_string[1:]:
                result = np.kron(result, pauli_dict[p])
            return result

        H_exact = np.zeros((8, 8), dtype=complex)
        for coeff, pauli_str in hamiltonian:
            H_exact += coeff * pauli_to_matrix(pauli_str)

        H_psi_exact = H_exact @ psi

        H_psi_stab = np.zeros(8, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        max_diff = np.max(np.abs(H_psi_exact - H_psi_stab))
        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10), \
            f"Max diff: {max_diff}"


class TestExpectationValues:
    """Test computing expectation values ⟨ψ|H|ψ⟩ with large Hamiltonians."""

    def test_energy_expectation(self):
        """Test energy expectation value calculation."""
        # Bell state
        stabs = [
            stim.PauliString('+XX'),
            stim.PauliString('+ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Hamiltonian
        hamiltonian = [
            (1.0, 'XX'),
            (1.0, 'ZZ'),
            (0.5, 'XI'),
            (0.5, 'IX'),
        ]

        # Apply H to state
        result = state.apply_hamiltonian(hamiltonian)

        # Compute ⟨ψ|H|ψ⟩
        expectation = 0
        for s, coeff in result.items():
            overlap = state.inner_product(s, phase=True)
            expectation += np.real(overlap * coeff)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.eye(2)

        XX = np.kron(X, X)
        ZZ = np.kron(Z, Z)
        XI = np.kron(X, I)
        IX = np.kron(I, X)

        H_exact = XX + ZZ + 0.5*XI + 0.5*IX
        expectation_exact = np.real(np.vdot(psi, H_exact @ psi))

        assert np.isclose(expectation, expectation_exact, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

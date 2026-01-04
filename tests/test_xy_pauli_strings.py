"""Tests for apply_hamiltonian with X and Y terms in multi-qubit Pauli strings.

Focus on testing the phase computation for X and Y operators on various states.
"""

import pytest
import numpy as np
import stim
from shades.stabilizer.stabilizer_state import StabilizerState
from shades.stabilizer_subspace import stabilizer_from_stim_tableau

class TestXYPauliStrings:
    """Test X and Y operators in multi-qubit Pauli strings."""

    def test_xy_on_computational_basis(self):
        """Test XY on |00⟩."""
        # |00⟩ state
        generator_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        hamiltonian = [(1.0, 'XY')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        XY = np.kron(X, Y)

        H_psi_exact = XY @ psi

        # Reconstruct from stabilizer
        H_psi_stab = np.zeros(4, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nXY|00⟩ test:")
        print(f"  Exact result: {H_psi_exact}")
        print(f"  Stabilizer result: {H_psi_stab}")
        print(f"  Match: {np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10), \
            f"Mismatch! Diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}"

    def test_xyx_on_ghz(self):
        """Test XYX on GHZ state."""
        # GHZ state (|000⟩ + |111⟩)/√2
        stabs = [
            stim.PauliString('+XXX'),
            stim.PauliString('+ZZ_'),
            stim.PauliString('+_ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        hamiltonian = [(1.0, 'XYX')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        I = np.eye(2)

        XYX = np.kron(np.kron(X, Y), X)
        H_psi_exact = XYX @ psi

        # Reconstruct
        H_psi_stab = np.zeros(8, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nXYX on GHZ test:")
        print(f"  Exact result: {H_psi_exact}")
        print(f"  Stabilizer result: {H_psi_stab}")
        print(f"  Match: {np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_xyxy_four_qubit(self):
        """Test XYXY on 4-qubit state |0000⟩."""
        generator_matrix = np.array([
            [3, 0, 0, 0, 1],
            [0, 3, 0, 0, 1],
            [0, 0, 3, 0, 1],
            [0, 0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        hamiltonian = [(1.0, 'XYXY')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])

        XYXY = np.kron(np.kron(np.kron(X, Y), X), Y)
        H_psi_exact = XYXY @ psi

        # Reconstruct
        H_psi_stab = np.zeros(16, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nXYXY|0000⟩ test:")
        print(f"  Exact norm: {np.linalg.norm(H_psi_exact)}")
        print(f"  Stabilizer norm: {np.linalg.norm(H_psi_stab)}")
        print(f"  Match: {np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_xxyy_on_bell_state(self):
        """Test XXYY on Bell state |Φ+⟩⊗|Φ+⟩."""
        # Create |Φ+⟩⊗|Φ+⟩ = (|0000⟩ + |0011⟩ + |1100⟩ + |1111⟩)/2
        # This is stabilized by X_X_, _X_X, ZZ__, __ZZ
        stabs = [
            stim.PauliString('+X_X_'),
            stim.PauliString('+_X_X'),
            stim.PauliString('+ZZ__'),
            stim.PauliString('+__ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        hamiltonian = [(1.0, 'XXYY')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])

        XXYY = np.kron(np.kron(np.kron(X, X), Y), Y)
        H_psi_exact = XXYY @ psi

        # Reconstruct
        H_psi_stab = np.zeros(16, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nXXYY on |Φ+⟩⊗|Φ+⟩ test:")
        print(f"  Exact result norm: {np.linalg.norm(H_psi_exact)}")
        print(f"  Stabilizer result norm: {np.linalg.norm(H_psi_stab)}")
        print(f"  Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_yxyx_on_plus_states(self):
        """Test YXYX on |++++⟩."""
        # |++++⟩ = |+⟩⊗|+⟩⊗|+⟩⊗|+⟩
        stabs = [
            stim.PauliString('+X___'),
            stim.PauliString('+_X__'),
            stim.PauliString('+__X_'),
            stim.PauliString('+___X'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        hamiltonian = [(1.0, 'YXYX')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify with exact calculation
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])

        YXYX = np.kron(np.kron(np.kron(Y, X), Y), X)
        H_psi_exact = YXYX @ psi

        # Reconstruct
        H_psi_stab = np.zeros(16, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nYXYX|++++⟩ test:")
        print(f"  Exact result: {H_psi_exact}")
        print(f"  Stabilizer result: {H_psi_stab}")
        print(f"  Match: {np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_mixed_xyz_string(self):
        """Test mixed XYZ string on 3-qubit state."""
        # |000⟩ state
        generator_matrix = np.array([
            [3, 0, 0, 1],
            [0, 3, 0, 1],
            [0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        # Test XYZ
        hamiltonian = [(1.0, 'XYZ')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        XYZ = np.kron(np.kron(X, Y), Z)
        H_psi_exact = XYZ @ psi

        H_psi_stab = np.zeros(8, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nXYZ|000⟩ test:")
        print(f"  Exact: {H_psi_exact}")
        print(f"  Stabilizer: {H_psi_stab}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_all_y_string(self):
        """Test all-Y string YYY on various states."""
        # Test on |000⟩
        generator_matrix = np.array([
            [3, 0, 0, 1],
            [0, 3, 0, 1],
            [0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        hamiltonian = [(1.0, 'YYY')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        Y = np.array([[0, -1j], [1j, 0]])
        YYY = np.kron(np.kron(Y, Y), Y)
        H_psi_exact = YYY @ psi

        H_psi_stab = np.zeros(8, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nYYY|000⟩ test:")
        print(f"  Exact: {H_psi_exact}")
        print(f"  Stabilizer: {H_psi_stab}")
        print(f"  Diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_alternating_xy(self):
        """Test alternating X and Y: XYXYXY on 6-qubit |000000⟩."""
        n_qubits = 6
        generator_matrix = np.array([
            [3] + [0]*(n_qubits-1) + [1],
            [0, 3] + [0]*(n_qubits-2) + [1],
            [0, 0, 3] + [0]*(n_qubits-3) + [1],
            [0, 0, 0, 3] + [0]*(n_qubits-4) + [1],
            [0, 0, 0, 0, 3] + [0]*(n_qubits-5) + [1],
            [0, 0, 0, 0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        hamiltonian = [(1.0, 'XYXYXY')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])

        # Build XYXYXY
        XYXYXY = X
        for i, op in enumerate(['Y', 'X', 'Y', 'X', 'Y']):
            if op == 'X':
                XYXYXY = np.kron(XYXYXY, X)
            else:
                XYXYXY = np.kron(XYXYXY, Y)

        H_psi_exact = XYXYXY @ psi

        H_psi_stab = np.zeros(2**n_qubits, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\nXYXYXY|000000⟩ test:")
        print(f"  Number of resulting states: {len(result)}")
        print(f"  Exact norm: {np.linalg.norm(H_psi_exact)}")
        print(f"  Stabilizer norm: {np.linalg.norm(H_psi_stab)}")
        print(f"  Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)


class TestXYCombinations:
    """Test combinations of X and Y in Hamiltonians with multiple terms."""

    def test_x_plus_y_each_qubit(self):
        """Test H = X_I + Y_I + _XI + _YI on 2-qubit |00⟩."""
        generator_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        hamiltonian = [
            (1.0, 'XI'),
            (1.0, 'YI'),
            (1.0, 'IX'),
            (1.0, 'IY'),
        ]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        I = np.eye(2)

        XI = np.kron(X, I)
        YI = np.kron(Y, I)
        IX = np.kron(I, X)
        IY = np.kron(I, Y)

        H_exact = XI + YI + IX + IY
        H_psi_exact = H_exact @ psi

        H_psi_stab = np.zeros(4, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\n(XI + YI + IX + IY)|00⟩ test:")
        print(f"  Number of terms: {len(result)}")
        print(f"  Match: {np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_xy_plus_yx(self):
        """Test H = XY + YX on |00⟩."""
        generator_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(generator_matrix)

        hamiltonian = [
            (1.0, 'XY'),
            (1.0, 'YX'),
        ]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])

        XY = np.kron(X, Y)
        YX = np.kron(Y, X)

        H_exact = XY + YX
        H_psi_exact = H_exact @ psi

        H_psi_stab = np.zeros(4, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\n(XY + YX)|00⟩ test:")
        print(f"  Exact: {H_psi_exact}")
        print(f"  Stabilizer: {H_psi_stab}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

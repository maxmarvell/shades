"""Detailed phase tracking tests for apply_hamiltonian method.

This test file specifically focuses on verifying phase tracking correctness
for large and complex Pauli strings.
"""

import pytest
import numpy as np
import stim
from shades.stabilizer.stabilizer_state import StabilizerState


class TestPhaseTracking:
    """Test phase tracking in apply_hamiltonian for various Pauli strings."""

    def test_single_y_operator_phase(self):
        """Test that Y operator produces correct phase."""
        # |0⟩ state
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state = StabilizerState(gen_0)

        # Apply Y
        hamiltonian = [(1.0, 'Y')]
        result = state.apply_hamiltonian(hamiltonian)

        # Y|0⟩ = i|1⟩
        # Check coefficient
        print("\n=== Single Y test ===")
        for s, coeff in result.items():
            print(f"State: {s.generator_matrix}")
            print(f"Coefficient: {coeff}")
            print(f"Expected: 1j")
            assert np.isclose(coeff, 1j, atol=1e-10), \
                f"Expected 1j, got {coeff}"

    def test_iy_operator_phase(self):
        """Test IY on 2-qubit system."""
        # |00⟩ state
        gen_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply IY (Y on second qubit)
        hamiltonian = [(1.0, 'IY')]
        result = state.apply_hamiltonian(hamiltonian)

        # IY|00⟩ = i|01⟩
        print("\n=== IY test ===")
        for s, coeff in result.items():
            print(f"Coefficient: {coeff}")
            print(f"Expected: 1j")
            assert np.isclose(coeff, 1j, atol=1e-10)

    def test_yi_operator_phase(self):
        """Test YI on 2-qubit system."""
        # |00⟩ state
        gen_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply YI (Y on first qubit)
        hamiltonian = [(1.0, 'YI')]
        result = state.apply_hamiltonian(hamiltonian)

        # YI|00⟩ = i|10⟩
        print("\n=== YI test ===")
        for s, coeff in result.items():
            print(f"Coefficient: {coeff}")
            print(f"Expected: 1j")
            assert np.isclose(coeff, 1j, atol=1e-10)

    def test_ziiy_operator_phase(self):
        """Test ZIIY on 4-qubit system - from failing H2 test."""
        # |1100⟩ state (HF state for H2)
        gen_matrix = np.array([
            [3, 0, 0, 0, -1],  # Z eigenvalue -1 → |1⟩
            [0, 3, 0, 0, -1],  # Z eigenvalue -1 → |1⟩
            [0, 0, 3, 0, 1],   # Z eigenvalue +1 → |0⟩
            [0, 0, 0, 3, 1],   # Z eigenvalue +1 → |0⟩
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply ZIIY
        hamiltonian = [(1.0, 'ZIIY')]
        result = state.apply_hamiltonian(hamiltonian)

        # Compute exact result
        psi = state.get_statevector()
        I = np.eye(2)
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        # Reverse to match stabilizer qubit ordering: ZIIY -> YIIZ
        ZIIY = np.kron(np.kron(np.kron(Y, I), I), Z)
        H_psi_exact = ZIIY @ psi

        # Reconstruct from stabilizer
        H_psi_stab = np.zeros(16, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print("\n=== ZIIY test ===")
        print(f"Exact result:\n{H_psi_exact}")
        print(f"Stabilizer result:\n{H_psi_stab}")
        print(f"Difference:\n{H_psi_exact - H_psi_stab}")
        print(f"Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        # Check specific coefficients
        for s, coeff in result.items():
            print(f"\nState basis state eigenvalues: {s.basis_state.generator_matrix[:, -1]}")
            print(f"Coefficient: {coeff}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10), \
            f"Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}"

    def test_iyii_operator_phase(self):
        """Test IYII on 4-qubit system - from failing H2 test."""
        # |1100⟩ state
        gen_matrix = np.array([
            [3, 0, 0, 0, -1],
            [0, 3, 0, 0, -1],
            [0, 0, 3, 0, 1],
            [0, 0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply IYII
        hamiltonian = [(1.0, 'IYII')]
        result = state.apply_hamiltonian(hamiltonian)

        # Exact calculation
        psi = state.get_statevector()
        I = np.eye(2)
        Y = np.array([[0, -1j], [1j, 0]])

        # Reverse to match stabilizer qubit ordering: IYII -> IIYI
        IYII = np.kron(np.kron(np.kron(I, I), Y), I)
        H_psi_exact = IYII @ psi

        # Reconstruct
        H_psi_stab = np.zeros(16, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print("\n=== IYII test ===")
        print(f"Exact result:\n{H_psi_exact}")
        print(f"Stabilizer result:\n{H_psi_stab}")
        print(f"Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_multiple_y_operators(self):
        """Test multiple Y operators in same Pauli string."""
        # |00⟩ state
        gen_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply YY
        hamiltonian = [(1.0, 'YY')]
        result = state.apply_hamiltonian(hamiltonian)

        # YY|00⟩ = (iY_0)(iY_1)|00⟩ = -1|11⟩
        # Actually: Y⊗Y|00⟩ = (iσ_y)⊗(iσ_y)|00⟩ = i²|11⟩ = -|11⟩
        psi = state.get_statevector()
        Y = np.array([[0, -1j], [1j, 0]])
        YY = np.kron(Y, Y)
        H_psi_exact = YY @ psi

        H_psi_stab = np.zeros(4, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print("\n=== YY test ===")
        print(f"Exact result: {H_psi_exact}")
        print(f"Stabilizer result: {H_psi_stab}")
        for s, coeff in result.items():
            print(f"Coefficient: {coeff}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_xyz_combination(self):
        """Test XYZ on 3-qubit state."""
        # |000⟩ state
        gen_matrix = np.array([
            [3, 0, 0, 1],
            [0, 3, 0, 1],
            [0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply XYZ
        hamiltonian = [(1.0, 'XYZ')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        # Reverse to match stabilizer qubit ordering: XYZ -> ZYX
        XYZ = np.kron(np.kron(Z, Y), X)
        H_psi_exact = XYZ @ psi

        H_psi_stab = np.zeros(8, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print("\n=== XYZ test ===")
        print(f"Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_long_pauli_string_with_y(self):
        """Test a long Pauli string with Y operators."""
        # 6-qubit state |000000⟩
        gen_matrix = np.array([
            [3, 0, 0, 0, 0, 0, 1],
            [0, 3, 0, 0, 0, 0, 1],
            [0, 0, 3, 0, 0, 0, 1],
            [0, 0, 0, 3, 0, 0, 1],
            [0, 0, 0, 0, 3, 0, 1],
            [0, 0, 0, 0, 0, 3, 1],
        ], dtype=np.int8)
        state = StabilizerState(gen_matrix)

        # Apply XYZXYZ
        hamiltonian = [(1.0, 'XYZXYZ')]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        pauli_str = 'XYZXYZ'
        # Reverse to match stabilizer qubit ordering
        pauli_str = pauli_str[::-1]
        op = pauli_dict[pauli_str[0]]
        for p in pauli_str[1:]:
            op = np.kron(op, pauli_dict[p])

        H_psi_exact = op @ psi

        H_psi_stab = np.zeros(64, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print("\n=== XYZXYZ test ===")
        print(f"Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)

    def test_y_on_excited_state(self):
        """Test Y operator on |1⟩ state."""
        # |1⟩ state
        gen_1 = np.array([[3, -1]], dtype=np.int8)
        state = StabilizerState(gen_1)

        # Apply Y
        hamiltonian = [(1.0, 'Y')]
        result = state.apply_hamiltonian(hamiltonian)

        # Y|1⟩ = -i|0⟩
        print("\n=== Y on |1⟩ test ===")
        for s, coeff in result.items():
            print(f"Coefficient: {coeff}")
            print(f"Expected: -1j")
            assert np.isclose(coeff, -1j, atol=1e-10)

    def test_phase_from_z_eigenvalue(self):
        """Test that Z eigenvalues affect Y phase correctly."""
        # Test Y on both |0⟩ and |1⟩
        for eigenvalue, expected_phase in [(1, 1j), (-1, -1j)]:
            gen = np.array([[3, eigenvalue]], dtype=np.int8)
            state = StabilizerState(gen)

            hamiltonian = [(1.0, 'Y')]
            result = state.apply_hamiltonian(hamiltonian)

            print(f"\n=== Y on eigenvalue {eigenvalue} ===")
            for s, coeff in result.items():
                print(f"Coefficient: {coeff}")
                print(f"Expected: {expected_phase}")
                assert np.isclose(coeff, expected_phase, atol=1e-10)


class TestComplexPauliStrings:
    """Test complex Pauli strings with multiple qubits."""

    def test_alternating_xyz_8qubits(self):
        """Test XYZXYZXY on 8-qubit state."""
        n_qubits = 8
        gen_matrix = np.eye(n_qubits + 1, dtype=np.int8)
        gen_matrix[:, :-1] *= 3
        gen_matrix[:, -1] = 1
        gen_matrix = gen_matrix[:n_qubits, :]

        state = StabilizerState(gen_matrix)

        # Apply XYZXYZXY
        pauli_str = 'XYZXYZXY'
        hamiltonian = [(1.0, pauli_str)]
        result = state.apply_hamiltonian(hamiltonian)

        # Verify
        psi = state.get_statevector()
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        # Reverse to match stabilizer qubit ordering
        pauli_str_reversed = pauli_str[::-1]
        op = pauli_dict[pauli_str_reversed[0]]
        for p in pauli_str_reversed[1:]:
            op = np.kron(op, pauli_dict[p])

        H_psi_exact = op @ psi

        H_psi_stab = np.zeros(2**n_qubits, dtype=complex)
        for s, coeff in result.items():
            s_vec = s.get_statevector()
            H_psi_stab += coeff * s_vec

        print(f"\n=== {pauli_str} test ===")
        print(f"Max diff: {np.max(np.abs(H_psi_exact - H_psi_stab))}")

        assert np.allclose(H_psi_exact, H_psi_stab, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

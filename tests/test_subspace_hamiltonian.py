"""Tests for StabilizerSubspace Hamiltonian matrix construction.

Verify that the Hamiltonian matrix H_ij = ⟨i|H|j⟩ is Hermitian.
"""

import pytest
import numpy as np
import stim
from shades.stabilizer.stabilizer_state import StabilizerState
from shades.stabilizer_subspace import StabilizerSubspace
from shades.stabilizer_subspace import stabilizer_from_stim_tableau

class TestSubspaceHermiticity:
    """Test that subspace Hamiltonian matrices are Hermitian."""

    def test_simple_z_hamiltonian(self):
        """Test H = Z on computational basis states."""
        # Create |0⟩ and |1⟩
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        gen_1 = np.array([[3, -1]], dtype=np.int8)

        state_0 = StabilizerState(gen_0)
        state_1 = StabilizerState(gen_1)

        states = [state_0, state_1]
        hamiltonian = [(1.0, 'Z')]

        # Build matrices
        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== Z Hamiltonian Test ===")
        print(f"H matrix:\n{subspace.H}")
        print(f"H†:\n{subspace.H.conj().T}")
        print(f"H - H†:\n{subspace.H - subspace.H.conj().T}")

        # Check Hermiticity
        assert np.allclose(subspace.H, subspace.H.conj().T), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}"

        # Check expected values
        # H[0,0] = ⟨0|Z|0⟩ = +1
        # H[1,1] = ⟨1|Z|1⟩ = -1
        # H[0,1] = ⟨0|Z|1⟩ = 0
        # H[1,0] = ⟨1|Z|0⟩ = 0
        expected = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert np.allclose(subspace.H, expected), \
            f"H matrix doesn't match expected!\nGot:\n{subspace.H}\nExpected:\n{expected}"

    def test_x_hamiltonian_plus_minus(self):
        """Test H = X on |+⟩ and |-⟩ states."""
        # |+⟩ and |-⟩
        plus_tab = stim.Tableau.from_stabilizers([stim.PauliString('+X')])
        minus_tab = stim.Tableau.from_stabilizers([stim.PauliString('-X')])

        state_plus = stabilizer_from_stim_tableau(plus_tab)
        state_minus = stabilizer_from_stim_tableau(minus_tab)

        states = [state_plus, state_minus]
        hamiltonian = [(1.0, 'X')]

        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== X Hamiltonian Test ===")
        print(f"H matrix:\n{subspace.H}")
        print(f"H†:\n{subspace.H.conj().T}")

        # Check Hermiticity
        assert np.allclose(subspace.H, subspace.H.conj().T), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}"

        # Expected: H[0,0] = ⟨+|X|+⟩ = +1, H[1,1] = ⟨-|X|-⟩ = -1
        expected = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert np.allclose(subspace.H, expected)

    def test_off_diagonal_terms(self):
        """Test Hamiltonian with off-diagonal terms."""
        # Create |0⟩ and |1⟩
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        gen_1 = np.array([[3, -1]], dtype=np.int8)

        state_0 = StabilizerState(gen_0)
        state_1 = StabilizerState(gen_1)

        states = [state_0, state_1]
        # H = X creates off-diagonal terms
        hamiltonian = [(1.0, 'X')]

        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== Off-Diagonal Test (X on |0⟩,|1⟩) ===")
        print(f"H matrix:\n{subspace.H}")
        print(f"H†:\n{subspace.H.conj().T}")
        print(f"H - H†:\n{subspace.H - subspace.H.conj().T}")

        # Check Hermiticity
        assert np.allclose(subspace.H, subspace.H.conj().T), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}"

        # Expected: X flips states, so H[0,1] = ⟨0|X|1⟩ = 1, H[1,0] = ⟨1|X|0⟩ = 1
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert np.allclose(subspace.H, expected)

    def test_bell_state_hamiltonian(self):
        """Test with Bell states."""
        # |Φ+⟩ and |Φ-⟩
        phi_plus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XX'),
                stim.PauliString('+ZZ'),
            ])
        )

        phi_minus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XX'),
                stim.PauliString('-ZZ'),
            ])
        )

        states = [phi_plus, phi_minus]
        # H = ZZ
        hamiltonian = [(1.0, 'ZZ')]

        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== Bell State ZZ Hamiltonian ===")
        print(f"H matrix:\n{subspace.H}")
        print(f"H†:\n{subspace.H.conj().T}")

        assert np.allclose(subspace.H, subspace.H.conj().T), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}"

        # Expected: |Φ+⟩ is +1 eigenstate, |Φ-⟩ is -1 eigenstate
        expected = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert np.allclose(subspace.H, expected)

    def test_complex_hamiltonian(self):
        """Test with Hamiltonian having multiple terms."""
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        gen_1 = np.array([[3, -1]], dtype=np.int8)

        state_0 = StabilizerState(gen_0)
        state_1 = StabilizerState(gen_1)

        states = [state_0, state_1]
        # H = Z + X
        hamiltonian = [(1.0, 'Z'), (1.0, 'X')]

        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== Z + X Hamiltonian ===")
        print(f"H matrix:\n{subspace.H}")
        print(f"H†:\n{subspace.H.conj().T}")

        assert np.allclose(subspace.H, subspace.H.conj().T), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}"

    def test_y_operator_phases(self):
        """Test that Y operator phases are handled correctly."""
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        gen_1 = np.array([[3, -1]], dtype=np.int8)

        state_0 = StabilizerState(gen_0)
        state_1 = StabilizerState(gen_1)

        states = [state_0, state_1]
        hamiltonian = [(1.0, 'Y')]

        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== Y Hamiltonian ===")
        print(f"H matrix:\n{subspace.H}")
        print(f"H†:\n{subspace.H.conj().T}")
        print(f"H - H†:\n{subspace.H - subspace.H.conj().T}")

        # Check element by element
        for i in range(2):
            for j in range(2):
                print(f"H[{i},{j}] = {subspace.H[i,j]}")
                print(f"H[{j},{i}]* = {subspace.H[j,i].conj()}")
                print(f"Difference: {subspace.H[i,j] - subspace.H[j,i].conj()}")

        assert np.allclose(subspace.H, subspace.H.conj().T, atol=1e-10), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}"


class TestSubspaceOverlapMatrix:
    """Test that overlap matrix S is Hermitian."""

    def test_overlap_hermiticity(self):
        """Test S_ij = ⟨i|j⟩ is Hermitian."""
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        gen_1 = np.array([[3, -1]], dtype=np.int8)

        state_0 = StabilizerState(gen_0)
        state_1 = StabilizerState(gen_1)

        states = [state_0, state_1]
        hamiltonian = [(1.0, 'Z')]  # Dummy Hamiltonian

        subspace = StabilizerSubspace(states, hamiltonian)

        print("\n=== Overlap Matrix ===")
        print(f"S matrix:\n{subspace.S}")
        print(f"S†:\n{subspace.S.conj().T}")

        assert np.allclose(subspace.S, subspace.S.conj().T), \
            f"S is not Hermitian!\nS =\n{subspace.S}\nS† =\n{subspace.S.conj().T}"


class TestManualHamiltonianConstruction:
    """Manually verify Hamiltonian matrix construction step by step."""

    def test_manual_construction_trace(self):
        """Trace through the construction manually to find issues."""
        # Simple case: |0⟩ and |1⟩ with H = Y
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        gen_1 = np.array([[3, -1]], dtype=np.int8)

        state_0 = StabilizerState(gen_0)
        state_1 = StabilizerState(gen_1)

        states = [state_0, state_1]
        hamiltonian = [(1.0, 'Y')]

        print("\n=== Manual Construction Trace ===")
        print("States: |0⟩ and |1⟩")
        print("Hamiltonian: Y")

        # Manually compute H matrix
        H_manual = np.zeros((2, 2), dtype=np.int8)

        for j, s_j in enumerate(states):
            print(f"\n--- Processing column j={j} (state_{j}) ---")

            # Apply H to |j⟩
            projected_states = s_j.apply_hamiltonian(hamiltonian)
            print(f"H|{j}⟩ =", end="")
            for s, coeff in projected_states.items():
                # Identify which state this is
                if s == state_0:
                    print(f" {coeff}|0⟩", end="")
                elif s == state_1:
                    print(f" {coeff}|1⟩", end="")
            print()

            # Compute overlaps with all basis states
            for i, s_i in enumerate(states):
                for s, coeff in projected_states.items():
                    overlap = s_i.inner_product(s, phase=True)
                    contribution = overlap * coeff
                    H_manual[i, j] += contribution
                    print(f"  H[{i},{j}] += ⟨{i}|s⟩ * {coeff} = {overlap} * {coeff} = {contribution}")

        print(f"\nManual H matrix:\n{H_manual}")
        print(f"Manual H†:\n{H_manual.conj().T}")
        print(f"Is Hermitian: {np.allclose(H_manual, H_manual.conj().T)}")

        # Compare with automatic construction
        subspace = StabilizerSubspace(states, hamiltonian)
        print(f"\nAutomatic H matrix:\n{subspace.H}")
        print(f"Match: {np.allclose(H_manual, subspace.H)}")

        # Check individual elements
        print("\n=== Element-by-element check ===")
        for i in range(2):
            for j in range(2):
                print(f"H[{i},{j}] = {H_manual[i,j]:.6f}")
                print(f"H[{j},{i}]* = {H_manual[j,i].conj():.6f}")
                print(f"Hermitian? {np.isclose(H_manual[i,j], H_manual[j,i].conj())}")
                print()


class TestLargeSystemHermiticity:
    """Test Hermiticity on larger systems like transverse field Ising model."""

    def test_transverse_field_ising_hermiticity(self):
        """Test that subspace Hamiltonian is Hermitian for transverse field Ising model.

        The transverse field Ising model has the Hamiltonian:
        H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i

        We test on a chain of 6 qubits with random stabilizer states.
        """
        n_qubits = 6
        J = 1.0
        h = 0.5

        # Build transverse field Ising Hamiltonian
        hamiltonian = []

        # ZZ terms (nearest neighbor interactions)
        for i in range(n_qubits - 1):
            pauli_string = 'I' * i + 'ZZ' + 'I' * (n_qubits - i - 2)
            hamiltonian.append((-J, pauli_string))

        # X terms (transverse field)
        for i in range(n_qubits):
            pauli_string = 'I' * i + 'X' + 'I' * (n_qubits - i - 1)
            hamiltonian.append((-h, pauli_string))

        print(f"\n=== Transverse Field Ising Model ({n_qubits} qubits) ===")
        print(f"J = {J}, h = {h}")
        print(f"Number of terms: {len(hamiltonian)}")

        # Create stabilizer states for the subspace
        n_states = 4
        states = []

        # |000000⟩ - stabilized by Z on each qubit
        stabilizers_0 = [stim.PauliString(f'+{"I" * i}Z{"I" * (n_qubits - i - 1)}')
                         for i in range(n_qubits)]
        tab_0 = stim.Tableau.from_stabilizers(stabilizers_0)
        states.append(stabilizer_from_stim_tableau(tab_0))

        # |111111⟩ - stabilized by -Z on each qubit
        stabilizers_1 = [stim.PauliString(f'-{"I" * i}Z{"I" * (n_qubits - i - 1)}')
                         for i in range(n_qubits)]
        tab_1 = stim.Tableau.from_stabilizers(stabilizers_1)
        states.append(stabilizer_from_stim_tableau(tab_1))

        # |++++++⟩ - stabilized by X on each qubit (all in +X eigenstate)
        stabilizers_plus = [stim.PauliString(f'+{"I" * i}X{"I" * (n_qubits - i - 1)}')
                            for i in range(n_qubits)]
        tab_plus = stim.Tableau.from_stabilizers(stabilizers_plus)
        states.append(stabilizer_from_stim_tableau(tab_plus))

        # |------⟩ - stabilized by -X on each qubit (all in -X eigenstate)
        stabilizers_minus = [stim.PauliString(f'-{"I" * i}X{"I" * (n_qubits - i - 1)}')
                             for i in range(n_qubits)]
        tab_minus = stim.Tableau.from_stabilizers(stabilizers_minus)
        states.append(stabilizer_from_stim_tableau(tab_minus))

        print(f"Number of basis states: {n_states}")

        # Build subspace Hamiltonian
        subspace = StabilizerSubspace(states, hamiltonian)

        print(f"\nH matrix shape: {subspace.H.shape}")
        print(f"H matrix:\n{subspace.H}")
        print(f"\nH† matrix:\n{subspace.H.conj().T}")
        print(f"\nH - H†:\n{subspace.H - subspace.H.conj().T}")
        print(f"\nMax |H - H†|: {np.max(np.abs(subspace.H - subspace.H.conj().T))}")

        # Check Hermiticity
        assert np.allclose(subspace.H, subspace.H.conj().T, atol=1e-10), \
            f"H is not Hermitian!\nH =\n{subspace.H}\nH† =\n{subspace.H.conj().T}\nDiff:\n{subspace.H - subspace.H.conj().T}"

        # Verify it's real (transverse field Ising should have real Hamiltonian)
        assert np.allclose(subspace.H.imag, 0, atol=1e-10), \
            f"H has imaginary components (unexpected for TFIM):\n{subspace.H.imag}"

        print("\n✓ Hamiltonian is Hermitian and real")

        # Also verify overlap matrix is Hermitian
        assert np.allclose(subspace.S, subspace.S.conj().T, atol=1e-10), \
            f"Overlap matrix S is not Hermitian!"

        print("✓ Overlap matrix is Hermitian")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

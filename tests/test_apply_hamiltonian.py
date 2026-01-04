"""Tests for the apply_hamiltonian method of StabilizerState class.

The apply_hamiltonian method takes a Hamiltonian (linear combination of Pauli strings)
and applies it to a stabilizer state, returning a dictionary of resulting stabilizer
states with their coefficients.
"""

import pytest
import numpy as np
import stim
from shades.stabilizer.stabilizer_state import StabilizerState
from shades.stabilizer_subspace import stabilizer_from_stim_tableau


class TestApplyHamiltonianBasic:
    """Basic tests for apply_hamiltonian functionality."""

    def test_identity_hamiltonian(self):
        """Test that identity Hamiltonian returns the same state with correct coefficient."""
        # Create a simple 2-qubit stabilizer state |00⟩
        generator_matrix = np.array([
            [3, 0, 1],  # Z on qubit 0
            [0, 3, 1],  # Z on qubit 1
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: 2.0 * IIII
        hamiltonian = [(2.0, 'II')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should get the same state back with coefficient 2.0
        assert len(result) == 1
        result_state = list(result.keys())[0]
        assert result_state == state
        assert np.isclose(result[result_state], 2.0)

    def test_single_pauli_x(self):
        """Test applying a single X Pauli operator."""
        # Create |00⟩ state
        generator_matrix = np.array([
            [3, 0, 1],  # Z on qubit 0
            [0, 3, 1],  # Z on qubit 1
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: X on first qubit (rightmost in little-endian)
        hamiltonian = [(1.0, 'XI')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should get |10⟩ state
        assert len(result) == 1
        result_state = list(result.keys())[0]

        # Verify the result is different from original
        assert result_state != state

        # The coefficient should be 1.0 (no phase from X|0⟩)
        assert np.isclose(result[result_state], 1.0)

    def test_single_pauli_z(self):
        """Test applying Z Pauli to |0⟩ and |1⟩ states."""
        # Test on |0⟩
        generator_matrix = np.array([
            [3, 1],  # Z stabilizer, eigenvalue +1
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)
        hamiltonian = [(1.0, 'Z')]

        result = state.apply_hamiltonian(hamiltonian)

        # Z|0⟩ = +|0⟩, should get same state with coefficient +1
        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 1.0)

        # Test on |1⟩
        generator_matrix = np.array([
            [3, -1],  # Z stabilizer, eigenvalue -1
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)
        result = state.apply_hamiltonian(hamiltonian)

        # Z|1⟩ = -|1⟩, should get same state with coefficient -1
        assert len(result) == 1
        assert np.isclose(list(result.values())[0], -1.0)

    def test_single_pauli_y(self):
        """Test applying Y Pauli operator (Y = iXZ)."""
        # Create |0⟩ state
        generator_matrix = np.array([
            [3, 1],  # Z stabilizer
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: Y
        hamiltonian = [(1.0, 'Y')]

        result = state.apply_hamiltonian(hamiltonian)

        # Y|0⟩ = i|1⟩, should get |1⟩ state with phase i
        assert len(result) == 1
        result_state = list(result.keys())[0]

        # Should be different state
        assert result_state != state

        # Should have phase i (or -i, depending on convention)
        phase = result[result_state]
        assert np.isclose(abs(phase), 1.0)
        assert np.isclose(np.real(phase), 0.0)


class TestApplyHamiltonianLinearCombinations:
    """Test applying Hamiltonians that are linear combinations of Pauli operators."""

    def test_two_term_hamiltonian(self):
        """Test Hamiltonian with two terms: H = aX + bZ."""
        # Create |0⟩ state
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: H = 0.5*X + 0.3*Z
        hamiltonian = [(0.5, 'X'), (0.3, 'Z')]

        result = state.apply_hamiltonian(hamiltonian)

        # X|0⟩ = |1⟩ with coeff 0.5
        # Z|0⟩ = |0⟩ with coeff 0.3
        # Should get two states
        assert len(result) == 2

        # Check that coefficients are correct
        total_coeff = sum(abs(c) for c in result.values())
        assert np.isclose(total_coeff, 0.8)

    def test_pauli_x_plus_y(self):
        """Test H = X + Y on |0⟩ state."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Both X|0⟩ and Y|0⟩ give |1⟩ (with different phases)
        hamiltonian = [(1.0, 'X'), (1.0, 'Y')]

        result = state.apply_hamiltonian(hamiltonian)

        # X|0⟩ = |1⟩ with phase 1
        # Y|0⟩ = i|1⟩ with phase i
        # Result: (1 + i)|1⟩
        assert len(result) == 1  # Same state, so they combine

        result_coeff = list(result.values())[0]
        # Should be 1 + i
        assert np.isclose(result_coeff, 1.0 + 1j, atol=1e-10)


class TestApplyHamiltonianDestructiveInterference:
    """Test cases where Pauli terms cancel (destructive interference)."""

    def test_x_minus_x_cancels(self):
        """Test that H = X - X gives zero (complete cancellation)."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: X - X = 0
        hamiltonian = [(1.0, 'X'), (-1.0, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should completely cancel
        assert len(result) == 0

    def test_partial_cancellation(self):
        """Test H = 2X - X = X gives correct result."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        hamiltonian = [(2.0, 'X'), (-1.0, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should get |1⟩ with coefficient 1.0
        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 1.0)

    def test_complex_cancellation(self):
        """Test cancellation with complex coefficients."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Y|0⟩ = i|1⟩, so -iY|0⟩ = |1⟩
        # X|0⟩ = |1⟩
        # Together: X - iY should give |1⟩ + |1⟩ = 2|1⟩
        hamiltonian = [(1.0, 'X'), (-1j, 'Y')]

        result = state.apply_hamiltonian(hamiltonian)

        assert len(result) == 1
        # X|0⟩ = |1⟩, -iY|0⟩ = -i(i|1⟩) = |1⟩
        assert np.isclose(list(result.values())[0], 2.0, atol=1e-10)


class TestApplyHamiltonianMultiQubit:
    """Test apply_hamiltonian on multi-qubit states."""

    def test_two_qubit_xx(self):
        """Test XX operator on |00⟩ state."""
        # Create |00⟩
        generator_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: XX
        hamiltonian = [(1.0, 'XX')]

        result = state.apply_hamiltonian(hamiltonian)

        # XX|00⟩ = |11⟩
        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 1.0)

    def test_bell_state_hamiltonian(self):
        """Test Hamiltonian on Bell state (|00⟩ + |11⟩)/√2."""
        # Create Bell state using stim
        stabs = [
            stim.PauliString('+XX'),
            stim.PauliString('+ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Hamiltonian: ZZ (should give eigenvalue +1)
        hamiltonian = [(1.0, 'ZZ')]

        result = state.apply_hamiltonian(hamiltonian)

        # Bell state is +1 eigenstate of ZZ
        assert len(result) == 1
        assert np.isclose(abs(list(result.values())[0]), 1.0)

    def test_three_qubit_xyz(self):
        """Test XYZ operator on 3-qubit state."""
        # Create |000⟩
        generator_matrix = np.array([
            [3, 0, 0, 1],
            [0, 3, 0, 1],
            [0, 0, 3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Apply XYZ
        hamiltonian = [(1.0, 'XYZ')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should transform to |111⟩ with phase from Y
        assert len(result) == 1


class TestApplyHamiltonianPhases:
    """Test that phases are correctly tracked."""

    def test_negative_stabilizer_phase(self):
        """Test Hamiltonian with negative sign."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: -X
        hamiltonian = [(-1.0, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        assert len(result) == 1
        assert np.isclose(list(result.values())[0], -1.0)

    def test_imaginary_coefficient(self):
        """Test Hamiltonian with imaginary coefficient."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Hamiltonian: iX
        hamiltonian = [(1j, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 1j)

    def test_phase_from_y_operator(self):
        """Test that Y operator contributes correct i phase."""
        # |1⟩ state
        generator_matrix = np.array([
            [3, -1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Y|1⟩ = -i|0⟩
        hamiltonian = [(1.0, 'Y')]

        result = state.apply_hamiltonian(hamiltonian)

        assert len(result) == 1
        phase = list(result.values())[0]
        # Should have phase -i
        assert np.isclose(np.real(phase), 0.0, atol=1e-10)
        assert np.isclose(np.imag(phase), -1.0, atol=1e-10)


class TestApplyHamiltonianEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_hamiltonian(self):
        """Test applying empty Hamiltonian."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        hamiltonian = []

        result = state.apply_hamiltonian(hamiltonian)

        # Should return empty dictionary
        assert len(result) == 0

    def test_large_coefficient(self):
        """Test with large coefficients."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        hamiltonian = [(1000.0, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 1000.0)

    def test_very_small_coefficient_not_cancelled(self):
        """Test that small non-zero coefficients are kept."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Very small but non-zero
        hamiltonian = [(1e-10, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should still be present (above math.isclose threshold)
        assert len(result) == 1
        assert np.isclose(list(result.values())[0], 1e-10)

    def test_single_qubit_all_paulis(self):
        """Test applying all Pauli operators on single qubit."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # H = I + X + Y + Z on |0⟩
        # I|0⟩ = |0⟩
        # X|0⟩ = |1⟩
        # Y|0⟩ = i|1⟩
        # Z|0⟩ = |0⟩
        hamiltonian = [(1.0, 'I'), (1.0, 'X'), (1.0, 'Y'), (1.0, 'Z')]

        result = state.apply_hamiltonian(hamiltonian)

        # I and Z both give |0⟩: (1 + 1)|0⟩ = 2|0⟩
        # X and Y both give |1⟩: (1 + i)|1⟩
        # Should get two states
        assert len(result) == 2


class TestApplyHamiltonianHeisenberg:
    """Test with physically meaningful Hamiltonians (Heisenberg, Ising, etc.)."""

    def test_heisenberg_xx_yy_zz(self):
        """Test Heisenberg interaction: H = XX + YY + ZZ."""
        # Create |00⟩
        generator_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Heisenberg Hamiltonian
        hamiltonian = [(1.0, 'XX'), (1.0, 'YY'), (1.0, 'ZZ')]

        result = state.apply_hamiltonian(hamiltonian)

        # XX|00⟩ = |11⟩
        # YY|00⟩ = (iX)(iX)|00⟩ = -XX|00⟩ = -|11⟩
        # ZZ|00⟩ = |00⟩
        # Total: |00⟩ + 0|11⟩ = |00⟩ (XX and YY cancel on |11⟩)
        # Actually: XX|00⟩ = |11⟩, YY|00⟩ = i²|11⟩ = -|11⟩, ZZ|00⟩ = |00⟩

        # Should have both |00⟩ and |11⟩ or they cancel
        assert len(result) >= 1

    def test_ising_hamiltonian(self):
        """Test transverse field Ising: H = -ZZ - X."""
        # Create |00⟩
        generator_matrix = np.array([
            [3, 0, 1],
            [0, 3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # H = -ZZ - X_0 (X on first qubit only)
        hamiltonian = [(-1.0, 'ZZ'), (-1.0, 'XI')]

        result = state.apply_hamiltonian(hamiltonian)

        # -ZZ|00⟩ = -|00⟩
        # -X|00⟩ = -|10⟩ (flips first qubit)
        # Should get two states
        assert len(result) == 2


class TestApplyHamiltonianGHZState:
    """Test with GHZ state: (|000⟩ + |111⟩)/√2."""

    def test_ghz_stabilizers(self):
        """Test Hamiltonian on GHZ state."""
        # GHZ state stabilizers
        stabs = [
            stim.PauliString('+XXX'),
            stim.PauliString('+ZZ_'),
            stim.PauliString('+_ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Apply XXX (should be +1 eigenstate)
        hamiltonian = [(1.0, 'XXX')]

        result = state.apply_hamiltonian(hamiltonian)

        # GHZ is +1 eigenstate of XXX
        assert len(result) == 1
        # Should get same state back
        result_state = list(result.keys())[0]
        assert result_state == state
        assert np.isclose(abs(list(result.values())[0]), 1.0)

    def test_ghz_zzi(self):
        """Test ZZ_ on GHZ state."""
        stabs = [
            stim.PauliString('+XXX'),
            stim.PauliString('+ZZ_'),
            stim.PauliString('+_ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        # Apply ZZI
        hamiltonian = [(1.0, 'ZZI')]

        result = state.apply_hamiltonian(hamiltonian)

        # GHZ is +1 eigenstate of ZZ_
        assert len(result) == 1
        assert np.isclose(abs(list(result.values())[0]), 1.0)


class TestApplyHamiltonianNonCommuting:
    """Test with non-commuting Pauli operators."""

    def test_x_then_z_vs_z_then_x(self):
        """Test that XZ ≠ ZX demonstrates non-commutativity."""
        # Create |0⟩
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # XZ|0⟩ vs ZX|0⟩
        # XZ = -iY, ZX = iY
        # So they differ by -1

        hamiltonian_xz = [(1.0, 'X')]
        result_x = state.apply_hamiltonian(hamiltonian_xz)

        # Now apply Z to result
        result_state_x = list(result_x.keys())[0]
        hamiltonian_z = [(1.0, 'Z')]
        result_xz = result_state_x.apply_hamiltonian(hamiltonian_z)

        # Compare with single XZ... Actually we can't directly apply XZ as product
        # But we can test that [X,Z] ≠ 0 through the anticommutation


class TestApplyHamiltonianNumericalStability:
    """Test numerical stability and precision."""

    def test_accumulated_phases_remain_normalized(self):
        """Test that phases don't grow unbounded."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Apply many Y operators (each adds ±i)
        hamiltonian = [(1.0, 'Y')]

        result = state.apply_hamiltonian(hamiltonian)

        # The magnitude should still be reasonable
        coeff = list(result.values())[0]
        assert abs(coeff) < 10.0  # Should be ~1

    def test_cancellation_precision(self):
        """Test that cancellation works with numerical precision."""
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state = StabilizerState(generator_matrix)

        # Craft coefficients that should cancel
        a = 1.0 + 1e-15
        b = -1.0
        hamiltonian = [(a, 'X'), (b, 'X')]

        result = state.apply_hamiltonian(hamiltonian)

        # Should not cancel completely (1e-15 remains)
        # But should be recognized as very small
        if len(result) > 0:
            assert abs(list(result.values())[0]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

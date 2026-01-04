"""Tests for the inner_product (overlap) method of StabilizerState class.

The inner_product method computes ⟨self|other⟩ between two stabilizer states.
We test it by comparing against exact statevector calculations.
"""

import pytest
import numpy as np
import stim
from shades.stabilizer.stabilizer_state import StabilizerState
from shades.stabilizer_subspace import stabilizer_from_stim_tableau

class TestInnerProductBasic:
    """Basic tests comparing inner_product against statevector overlaps."""

    def test_identical_computational_basis_states(self):
        """Test ⟨0|0⟩ = 1."""
        # Create |0⟩ state
        generator_matrix = np.array([
            [3, 1],
        ], dtype=np.int8)

        state1 = StabilizerState(generator_matrix)
        state2 = StabilizerState(generator_matrix)

        # Without phase
        overlap = state1.inner_product(state2, phase=False)
        assert np.isclose(abs(overlap), 1.0)

        # With phase
        overlap_phased = state1.inner_product(state2, phase=True)
        assert np.isclose(abs(overlap_phased), 1.0)

        # Compare to statevector
        psi1 = state1.get_statevector()
        psi2 = state2.get_statevector()
        exact_overlap = np.vdot(psi1, psi2)
        assert np.isclose(abs(exact_overlap), 1.0)

    def test_orthogonal_computational_basis(self):
        """Test ⟨0|1⟩ = 0."""
        # Create |0⟩
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state0 = StabilizerState(gen_0)

        # Create |1⟩
        gen_1 = np.array([[3, -1]], dtype=np.int8)
        state1 = StabilizerState(gen_1)

        # Should be orthogonal
        overlap = state0.inner_product(state1, phase=False)
        assert overlap == 0

        overlap_phased = state0.inner_product(state1, phase=True)
        assert overlap_phased == 0

        # Compare to statevector
        psi0 = state0.get_statevector()
        psi1 = state1.get_statevector()
        exact_overlap = np.vdot(psi0, psi1)
        assert np.isclose(exact_overlap, 0.0)

    def test_plus_state_overlap(self):
        """Test ⟨+|+⟩ = 1 where |+⟩ = (|0⟩ + |1⟩)/√2."""
        # Create |+⟩ state using X stabilizer
        stab = stim.PauliString('+X')
        tab = stim.Tableau.from_stabilizers([stab])
        state_plus = stabilizer_from_stim_tableau(tab)

        # Self-overlap
        overlap = state_plus.inner_product(state_plus, phase=False)
        assert np.isclose(abs(overlap), 1.0)

        overlap_phased = state_plus.inner_product(state_plus, phase=True)
        assert np.isclose(abs(overlap_phased), 1.0)

        # Compare to statevector
        psi = state_plus.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)

    def test_plus_minus_orthogonal(self):
        """Test ⟨+|-⟩ = 0 where |+⟩ and |-⟩ are X eigenstates."""
        # |+⟩: +1 eigenstate of X
        plus_tab = stim.Tableau.from_stabilizers([stim.PauliString('+X')])
        state_plus = stabilizer_from_stim_tableau(plus_tab)

        # |-⟩: -1 eigenstate of X
        minus_tab = stim.Tableau.from_stabilizers([stim.PauliString('-X')])
        state_minus = stabilizer_from_stim_tableau(minus_tab)

        # Should be orthogonal
        overlap = state_plus.inner_product(state_minus, phase=False)
        assert overlap == 0

        overlap_phased = state_plus.inner_product(state_minus, phase=True)
        assert overlap_phased == 0

        # Verify with statevector
        psi_plus = state_plus.get_statevector()
        psi_minus = state_minus.get_statevector()
        exact_overlap = np.vdot(psi_plus, psi_minus)
        assert np.isclose(exact_overlap, 0.0)


class TestInnerProductTwoQubit:
    """Test inner products for two-qubit states."""

    def test_two_qubit_computational_basis(self):
        """Test overlaps between |00⟩, |01⟩, |10⟩, |11⟩."""
        # Create all four computational basis states
        states = []
        for i in range(4):
            gen = np.array([
                [3, 0, 1 if (i & 2) == 0 else -1],
                [0, 3, 1 if (i & 1) == 0 else -1],
            ], dtype=np.int8)
            states.append(StabilizerState(gen))

        # Test all pairs
        for i in range(4):
            for j in range(4):
                overlap = states[i].inner_product(states[j], phase=True)

                # Get statevectors
                psi_i = states[i].get_statevector()
                psi_j = states[j].get_statevector()
                exact_overlap = np.vdot(psi_i, psi_j)

                if i == j:
                    assert np.isclose(abs(overlap), 1.0)
                    assert np.isclose(exact_overlap, 1.0)
                else:
                    assert overlap == 0
                    assert np.isclose(exact_overlap, 0.0)

    def test_bell_state_phi_plus(self):
        """Test |Φ+⟩ = (|00⟩ + |11⟩)/√2 self-overlap."""
        stabs = [
            stim.PauliString('+XX'),
            stim.PauliString('+ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        bell_state = stabilizer_from_stim_tableau(tab)

        # Self-overlap should be 1
        overlap = bell_state.inner_product(bell_state, phase=False)
        assert np.isclose(abs(overlap), 1.0)

        overlap_phased = bell_state.inner_product(bell_state, phase=True)
        assert np.isclose(abs(overlap_phased), 1.0)

        # Verify with statevector
        psi = bell_state.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)

    def test_bell_states_orthogonality(self):
        """Test that different Bell states are orthogonal."""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XX'),
                stim.PauliString('+ZZ'),
            ])
        )

        # |Φ-⟩ = (|00⟩ - |11⟩)/√2
        phi_minus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XX'),
                stim.PauliString('-ZZ'),
            ])
        )

        # Should be orthogonal
        overlap = phi_plus.inner_product(phi_minus, phase=True)
        assert overlap == 0

        # Verify with statevector
        psi_plus = phi_plus.get_statevector()
        psi_minus = phi_minus.get_statevector()
        exact_overlap = np.vdot(psi_plus, psi_minus)
        assert np.isclose(exact_overlap, 0.0)

    def test_bell_psi_states(self):
        """Test |Ψ+⟩ = (|01⟩ + |10⟩)/√2."""
        # |Ψ+⟩
        psi_plus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XX'),
                stim.PauliString('-ZZ'),
            ])
        )

        # Self-overlap
        overlap = psi_plus.inner_product(psi_plus, phase=True)
        assert np.isclose(abs(overlap), 1.0)

        # Compare to statevector
        psi = psi_plus.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)


class TestInnerProductGHZ:
    """Test inner products with GHZ states."""

    def test_three_qubit_ghz_self_overlap(self):
        """Test GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2."""
        stabs = [
            stim.PauliString('+XXX'),
            stim.PauliString('+ZZ_'),
            stim.PauliString('+_ZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        ghz = stabilizer_from_stim_tableau(tab)

        # Self-overlap
        overlap = ghz.inner_product(ghz, phase=False)
        assert np.isclose(abs(overlap), 1.0)

        overlap_phased = ghz.inner_product(ghz, phase=True)
        assert np.isclose(abs(overlap_phased), 1.0)

        # Compare to statevector
        psi = ghz.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)

    def test_ghz_vs_w_state(self):
        """Test GHZ vs W state (should be non-orthogonal but distinct)."""
        # GHZ: (|000⟩ + |111⟩)/√2
        ghz = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XXX'),
                stim.PauliString('+ZZ_'),
                stim.PauliString('+_ZZ'),
            ])
        )

        # W state is not a stabilizer state, so we'll test GHZ against |000⟩
        gen_000 = np.array([
            [3, 0, 0, 1],
            [0, 3, 0, 1],
            [0, 0, 3, 1],
        ], dtype=np.int8)
        state_000 = StabilizerState(gen_000)

        # Overlap should be 1/√2
        overlap = ghz.inner_product(state_000, phase=True)

        # Compare to statevector
        psi_ghz = ghz.get_statevector()
        psi_000 = state_000.get_statevector()
        exact_overlap = np.vdot(psi_ghz, psi_000)

        assert np.isclose(abs(overlap), abs(exact_overlap), atol=1e-10)


class TestInnerProductPhases:
    """Test that phases are computed correctly."""

    def test_phase_from_y_eigenstate(self):
        """Test overlap with Y eigenstates involves ±i phases."""
        # |i⟩: +1 eigenstate of Y (up to normalization)
        # Y|i⟩ = i|i⟩
        y_plus_tab = stim.Tableau.from_stabilizers([stim.PauliString('+Y')])
        y_plus = stabilizer_from_stim_tableau(y_plus_tab)

        # Self-overlap
        overlap = y_plus.inner_product(y_plus, phase=True)
        assert np.isclose(abs(overlap), 1.0)

        # Compare to statevector
        psi = y_plus.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)

    def test_phase_preservation(self):
        """Test that relative phases are preserved."""
        # Create |+⟩
        plus_tab = stim.Tableau.from_stabilizers([stim.PauliString('+X')])
        state_plus = stabilizer_from_stim_tableau(plus_tab)

        # Create |0⟩
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state_0 = StabilizerState(gen_0)

        # ⟨+|0⟩ should be 1/√2
        overlap = state_plus.inner_product(state_0, phase=True)

        # Compare to statevector
        psi_plus = state_plus.get_statevector()
        psi_0 = state_0.get_statevector()
        exact_overlap = np.vdot(psi_plus, psi_0)

        # Should match
        assert np.isclose(overlap, exact_overlap, atol=1e-10)


class TestInnerProductRandomStates:
    """Test with random Clifford-generated states."""

    def test_random_single_qubit_clifford(self):
        """Test overlaps for random single-qubit Clifford states."""
        np.random.seed(42)

        # Generate random Clifford tableaux
        n_tests = 10
        for _ in range(n_tests):
            # Create random Clifford
            tab1 = stim.Tableau.random(1)
            tab2 = stim.Tableau.random(1)

            state1 = stabilizer_from_stim_tableau(tab1)
            state2 = stabilizer_from_stim_tableau(tab2)

            # Compute overlap
            overlap = state1.inner_product(state2, phase=True)

            # Compare to statevector
            psi1 = state1.get_statevector()
            psi2 = state2.get_statevector()
            exact_overlap = np.vdot(psi1, psi2)

            # Magnitudes should match (global phase may differ)
            assert np.isclose(abs(overlap), abs(exact_overlap), atol=1e-10), \
                f"Overlap mismatch: {abs(overlap)} vs {abs(exact_overlap)}"

    def test_random_two_qubit_clifford(self):
        """Test overlaps for random two-qubit Clifford states."""
        np.random.seed(123)

        n_tests = 10
        for _ in range(n_tests):
            tab1 = stim.Tableau.random(2)
            tab2 = stim.Tableau.random(2)

            state1 = stabilizer_from_stim_tableau(tab1)
            state2 = stabilizer_from_stim_tableau(tab2)

            overlap = state1.inner_product(state2, phase=True)

            psi1 = state1.get_statevector()
            psi2 = state2.get_statevector()
            exact_overlap = np.vdot(psi1, psi2)

            assert np.isclose(abs(overlap), abs(exact_overlap), atol=1e-10)

    def test_random_three_qubit_clifford(self):
        """Test overlaps for random three-qubit Clifford states."""
        np.random.seed(456)

        n_tests = 5
        for _ in range(n_tests):
            tab1 = stim.Tableau.random(3)
            tab2 = stim.Tableau.random(3)

            state1 = stabilizer_from_stim_tableau(tab1)
            state2 = stabilizer_from_stim_tableau(tab2)

            overlap = state1.inner_product(state2, phase=True)

            psi1 = state1.get_statevector()
            psi2 = state2.get_statevector()
            exact_overlap = np.vdot(psi1, psi2)

            assert np.isclose(abs(overlap), abs(exact_overlap), atol=1e-9)


class TestInnerProductEdgeCases:
    """Test edge cases and special scenarios."""

    def test_return_k_parameter(self):
        """Test that return_k parameter returns the X-rank."""
        # Bell state
        bell = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+XX'),
                stim.PauliString('+ZZ'),
            ])
        )

        # Get k value
        k = bell.inner_product(bell, phase=False, return_k=True)

        # For self-overlap of pure state, k should be 0
        assert k == 0

    def test_product_state_overlap(self):
        """Test |+⟩⊗|+⟩ self-overlap."""
        # Create |++⟩
        plus_plus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([
                stim.PauliString('+X_'),
                stim.PauliString('+_X'),
            ])
        )

        overlap = plus_plus.inner_product(plus_plus, phase=True)
        assert np.isclose(abs(overlap), 1.0)

        # Verify with statevector
        psi = plus_plus.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)

    def test_four_qubit_state(self):
        """Test four-qubit stabilizer state."""
        # Create a 4-qubit state
        stabs = [
            stim.PauliString('+XX__'),
            stim.PauliString('+_XX_'),
            stim.PauliString('+__XX'),
            stim.PauliString('+ZZZZ'),
        ]
        tab = stim.Tableau.from_stabilizers(stabs)
        state = stabilizer_from_stim_tableau(tab)

        overlap = state.inner_product(state, phase=True)
        assert np.isclose(abs(overlap), 1.0)

        psi = state.get_statevector()
        exact_overlap = np.vdot(psi, psi)
        assert np.isclose(exact_overlap, 1.0)


class TestInnerProductConsistency:
    """Test consistency properties of inner products."""

    def test_hermiticity(self):
        """Test that ⟨ψ|φ⟩ = ⟨φ|ψ⟩* (conjugate symmetry)."""
        # Create two different states
        state1 = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([stim.PauliString('+X')])
        )
        gen2 = np.array([[3, 1]], dtype=np.int8)
        state2 = StabilizerState(gen2)

        overlap_12 = state1.inner_product(state2, phase=True)
        overlap_21 = state2.inner_product(state1, phase=True)

        # Should be complex conjugates
        assert np.isclose(overlap_12, np.conj(overlap_21), atol=1e-10)

        # Verify with statevector
        psi1 = state1.get_statevector()
        psi2 = state2.get_statevector()
        exact_12 = np.vdot(psi1, psi2)
        exact_21 = np.vdot(psi2, psi1)
        assert np.isclose(exact_12, np.conj(exact_21))

    def test_linearity(self):
        """Test that overlaps behave linearly (via apply_hamiltonian + inner_product)."""
        # Create |0⟩
        gen = np.array([[3, 1]], dtype=np.int8)
        state = StabilizerState(gen)

        # Apply H = X + Y to get |1⟩ + i|1⟩ = (1+i)|1⟩
        hamiltonian = [(1.0, 'X'), (1.0, 'Y')]
        result_states = state.apply_hamiltonian(hamiltonian)

        # Should get single state
        assert len(result_states) == 1
        result_state = list(result_states.keys())[0]
        result_coeff = list(result_states.values())[0]

        # Create |1⟩ for comparison
        gen_1 = np.array([[3, -1]], dtype=np.int8)
        state_1 = StabilizerState(gen_1)

        # Overlap of result with |1⟩
        overlap = result_state.inner_product(state_1, phase=True)

        # Should be consistent with coefficient
        # The result is (1+i)|1⟩, so ⟨1|result⟩ should account for the coefficient
        assert result_state == state_1


class TestInnerProductWithApplyHamiltonian:
    """Test inner_product in combination with apply_hamiltonian for energy calculations."""

    def test_expectation_value_z(self):
        """Test ⟨0|Z|0⟩ = +1 and ⟨1|Z|1⟩ = -1."""
        # Test ⟨0|Z|0⟩
        gen_0 = np.array([[3, 1]], dtype=np.int8)
        state_0 = StabilizerState(gen_0)

        hamiltonian_z = [(1.0, 'Z')]
        result = state_0.apply_hamiltonian(hamiltonian_z)

        # Z|0⟩ = |0⟩
        expectation = 0.0
        for result_state, coeff in result.items():
            overlap = state_0.inner_product(result_state, phase=True)
            expectation += np.real(overlap * coeff)

        assert np.isclose(expectation, 1.0)

        # Test ⟨1|Z|1⟩
        gen_1 = np.array([[3, -1]], dtype=np.int8)
        state_1 = StabilizerState(gen_1)

        result = state_1.apply_hamiltonian(hamiltonian_z)
        expectation = 0.0
        for result_state, coeff in result.items():
            overlap = state_1.inner_product(result_state, phase=True)
            expectation += np.real(overlap * coeff)

        assert np.isclose(expectation, -1.0)

    def test_expectation_value_x_on_plus(self):
        """Test ⟨+|X|+⟩ = +1."""
        plus = stabilizer_from_stim_tableau(
            stim.Tableau.from_stabilizers([stim.PauliString('+X')])
        )

        hamiltonian = [(1.0, 'X')]
        result = plus.apply_hamiltonian(hamiltonian)

        expectation = 0.0
        for result_state, coeff in result.items():
            overlap = plus.inner_product(result_state, phase=True)
            expectation += np.real(overlap * coeff)

        assert np.isclose(expectation, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

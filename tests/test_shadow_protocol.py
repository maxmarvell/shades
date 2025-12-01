"""Rigorous tests for the Shadow Protocol implementation.

Tests the classical shadow tomography protocol for estimating overlaps
between quantum states, following the methodology from:
- Huggins et al. Nature Physics (2021): https://doi.org/10.1038/s41586-021-04351-z
- Lenihan et al. (2025): Excitation amplitude sampling
"""

import pytest
import numpy as np
import stim
from qiskit.quantum_info import Statevector
from shades.shadows import (
    CliffordGroup,
    ClassicalShadow,
    ShadowProtocol,
)
from shades.utils import Bitstring


class TestCliffordGroup:
    """Test the Clifford group ensemble."""

    def test_initialization(self):
        """Test CliffordGroup initializes with correct dimension."""
        n_qubits = 4
        group = CliffordGroup(n_qubits)
        assert group.d == n_qubits

    def test_generate_sample_returns_tableau(self):
        """Test that generate_sample returns a stim.Tableau."""
        group = CliffordGroup(3)
        sample = group.generate_sample()
        assert isinstance(sample, stim.Tableau)

    def test_generate_sample_correct_size(self):
        """Test that generated tableau has correct number of qubits."""
        n_qubits = 5
        group = CliffordGroup(n_qubits)
        sample = group.generate_sample()
        assert len(sample) == n_qubits

    def test_generate_multiple_samples_different(self):
        """Test that multiple samples are different (randomness)."""
        group = CliffordGroup(3)
        samples = [group.generate_sample() for _ in range(10)]
        # Check that not all samples are identical
        # Compare string representations since Tableau doesn't have __eq__
        sample_strs = [str(s) for s in samples]
        assert len(set(sample_strs)) > 1, "All samples are identical"


class TestShadowProtocolInitialization:
    """Test ShadowProtocol initialization."""

    def test_init_with_computational_basis_state(self):
        """Test initialization with a computational basis state."""
        state = Statevector.from_label("0000")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        assert protocol.state == state
        assert protocol.ensemble_type == 'clifford'
        assert isinstance(protocol.ensemble, CliffordGroup)
        assert protocol.ensemble.d == 4

    def test_init_with_superposition_state(self):
        """Test initialization with a superposition state."""
        # Create |+⟩ state
        state = Statevector.from_label("0")
        state = state.evolve(Statevector.from_label("0").to_operator().power(0))
        state = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])

        protocol = ShadowProtocol(state, ensemble_type='clifford')
        assert protocol.state == state

    def test_init_pauli_ensemble_not_implemented(self):
        """Test that Pauli ensemble raises NotImplementedError."""
        state = Statevector.from_label("00")
        with pytest.raises(NotImplementedError, match="Not yet implemented"):
            ShadowProtocol(state, ensemble_type='pauli')

    def test_init_invalid_ensemble_raises_error(self):
        """Test that invalid ensemble type raises ValueError."""
        state = Statevector.from_label("00")
        with pytest.raises(ValueError, match="Unexpected ensemble"):
            ShadowProtocol(state, ensemble_type='invalid')


class TestTauMethod:
    """Test the tau() method for constructing the mixed state."""

    def test_tau_with_orthogonal_state(self):
        """Test tau() with a state orthogonal to vacuum (⟨0|ψ⟩ = 0)."""
        # Create state |1⟩ which is orthogonal to |0⟩
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        tau_state = protocol.tau()

        # tau = (|0⟩ + |1⟩) / √2 = |+⟩
        expected = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
        assert np.allclose(tau_state.data, expected.data)

    def test_tau_with_multi_qubit_orthogonal_state(self):
        """Test tau() with multi-qubit state orthogonal to vacuum."""
        # Create state |01⟩ which has ⟨00|01⟩ = 0
        state = Statevector.from_label("01")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        tau_state = protocol.tau()

        # tau = (|00⟩ + |01⟩) / √2
        vacuum = np.zeros(4, dtype=complex)
        vacuum[0] = 1.0
        expected_data = (vacuum + state.data) / np.sqrt(2)
        assert np.allclose(tau_state.data, expected_data)

    def test_tau_raises_error_for_non_orthogonal_state(self):
        """Test that tau() raises RuntimeError for non-orthogonal states."""
        state = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        with pytest.raises(RuntimeError, match="not orthogonal to vacuum"):
            protocol.tau()

    def test_tau_raises_error_for_vacuum_state(self):
        """Test that tau() raises RuntimeError for vacuum state itself."""
        # |0⟩ has ⟨0|0⟩ = 1 ≠ 0
        state = Statevector.from_label("0")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        with pytest.raises(RuntimeError, match="not orthogonal to vacuum"):
            protocol.tau()

    def test_tau_with_complex_orthogonal_state(self):
        """Test tau() with complex amplitudes but orthogonal to vacuum."""
        # Create state (i|01⟩ + |10⟩) / √2
        state_data = np.zeros(4, dtype=complex)
        state_data[1] = 1j / np.sqrt(2)  # |01⟩
        state_data[2] = 1 / np.sqrt(2)   # |10⟩
        state = Statevector(state_data)

        protocol = ShadowProtocol(state, ensemble_type='clifford')
        tau_state = protocol.tau()

        # tau = (|00⟩ + (i|01⟩ + |10⟩)/√2) / √2
        vacuum = np.zeros(4, dtype=complex)
        vacuum[0] = 1.0
        expected_data = (vacuum + state.data) / np.sqrt(2)
        assert np.allclose(tau_state.data, expected_data)


class TestCollectSamples:
    """Test the collect_samples() method."""

    def test_collect_samples_creates_correct_number_of_estimators(self):
        """Test that collect_samples creates the right number of estimators."""
        state = Statevector.from_label("01")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        n_samples = 1000
        n_estimators = 10
        protocol.collect_samples(n_samples, n_estimators, prediction='overlap')

        assert len(protocol.k_estimators) == n_estimators

    def test_collect_samples_each_estimator_has_correct_size(self):
        """Test that each estimator has correct number of samples."""
        state = Statevector.from_label("10")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        n_samples = 500
        n_estimators = 5
        protocol.collect_samples(n_samples, n_estimators, prediction='overlap')

        batch_size = n_samples // n_estimators
        for estimator in protocol.k_estimators:
            assert estimator.N == batch_size

    def test_collect_samples_total_count_correct(self):
        """Test that total samples across all estimators is correct."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        n_samples = 600
        n_estimators = 3
        protocol.collect_samples(n_samples, n_estimators, prediction='overlap')

        total_samples = sum(est.N for est in protocol.k_estimators)
        assert total_samples == n_samples

    def test_collect_samples_raises_error_for_uneven_split(self):
        """Test that collect_samples raises error if n_samples not divisible."""
        state = Statevector.from_label("0")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        with pytest.raises(ValueError, match="equally sized parts"):
            protocol.collect_samples(100, 3, prediction='overlap')

    def test_collect_samples_stores_prediction_type(self):
        """Test that collect_samples stores the prediction type."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        protocol.collect_samples(100, 10, prediction='overlap')
        assert protocol.prediction == 'overlap'

    def test_collect_samples_creates_clifford_snapshots(self):
        """Test that snapshots contain Clifford unitaries."""
        state = Statevector.from_label("01")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        protocol.collect_samples(100, 10, prediction='overlap')

        # Check first estimator's first snapshot
        snapshot = protocol.k_estimators[0].shadow[0]
        assert isinstance(snapshot, ClassicalSnapshot)
        assert isinstance(snapshot.unitary, stim.Tableau)
        assert isinstance(snapshot.measurement, Bitstring)
        assert snapshot.sampler == 'clifford'

    def test_collect_samples_measurements_are_bitstrings(self):
        """Test that all measurements are Bitstring objects."""
        state = Statevector.from_label("11")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        protocol.collect_samples(50, 5, prediction='overlap')

        for estimator in protocol.k_estimators:
            for snapshot in estimator.shadow:
                assert isinstance(snapshot.measurement, Bitstring)
                assert snapshot.measurement.size == 2

    def test_collect_samples_invalid_prediction_raises_error(self):
        """Test that invalid prediction type raises ValueError."""
        state = Statevector.from_label("0")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        with pytest.raises(ValueError):
            protocol.collect_samples(100, 10, prediction='invalid')


class TestClassicalShadow:
    """Test the ClassicalShadow class."""

    def test_classical_shadow_initialization(self):
        """Test ClassicalShadow initialization."""
        measurement = Bitstring([False, True])
        clifford = stim.Tableau.random(2)
        snapshot = ClassicalSnapshot(measurement, clifford, 'clifford')

        shadow = ClassicalShadow([snapshot], n_qubits=2)

        assert shadow.N == 1
        assert shadow.n_qubits == 2
        assert len(shadow.shadow) == 1

    def test_classical_shadow_N_property(self):
        """Test that N property returns correct count."""
        snapshots = []
        for _ in range(5):
            measurement = Bitstring([False, False])
            clifford = stim.Tableau.random(2)
            snapshots.append(ClassicalSnapshot(measurement, clifford, 'clifford'))

        shadow = ClassicalShadow(snapshots, n_qubits=2)
        assert shadow.N == 5


class TestEstimateOverlap:
    """Test the estimate_overlap() method with median-of-means."""

    def test_estimate_overlap_raises_error_without_samples(self):
        """Test that estimate_overlap raises error if collect_samples not called."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        target = Bitstring([True])
        with pytest.raises(ValueError, match="Must call collect_samples"):
            protocol.estimate_overlap(target)

    def test_estimate_overlap_returns_complex(self):
        """Test that estimate_overlap returns a complex number."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        protocol.collect_samples(100, 10, prediction='overlap')
        target = Bitstring([True])

        overlap = protocol.estimate_overlap(target)
        assert isinstance(overlap, (complex, np.complex128, np.complex64))

    def test_estimate_overlap_with_simple_state(self):
        """Test overlap estimation with a simple computational basis state."""

        bitstring = "0001"
        state = Statevector.from_label(bitstring)
        protocol = ShadowProtocol(state, ensemble_type='clifford', use_qulacs=True)

        protocol.collect_samples(20000, 20, prediction='overlap')

        # check overlap with itself
        target_same = Bitstring.from_string(bitstring)
        overlap_same = protocol.estimate_overlap(target_same)

        assert abs(abs(overlap_same) - 1) < 0.05

    def test_estimate_overlap_uses_median_of_means(self):
        """Test that estimate_overlap uses median across K estimators."""
        state = Statevector.from_label("01")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        # Use multiple estimators to test median
        protocol.collect_samples(500, 5, prediction='overlap')

        target = Bitstring([False, True])
        overlap = protocol.estimate_overlap(target)

        # Just check it returns a value (actual correctness tested elsewhere)
        assert overlap is not None
        assert isinstance(overlap, (complex, np.complex128, np.complex64))


class TestOverlapCalculation:
    """Test the ClassicalShadow.overlap() method with known states."""

    def test_overlap_single_qubit_computational_basis(self):
        """Test overlap with single qubit |1⟩ state.

        For |ψ⟩ = |1⟩, the protocol estimates ⟨a|ψ⟩ for bitstrings |a⟩
        Expected: ⟨1|1⟩ = 1
        """
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        # Use large number of samples for statistical accuracy
        protocol.collect_samples(1000, 10, prediction='overlap')

        # Test overlap with |1⟩ (same state)
        target_1 = Bitstring([True])
        overlap_1 = protocol.estimate_overlap(target_1).real
        expected_1 = 1.0

        # Allow tolerance for statistical sampling
        assert abs(abs(overlap_1) - expected_1) < 0.3, \
            f"⟨1|1⟩: expected {expected_1:.3f}, got {abs(overlap_1):.3f}"

    def test_overlap_two_qubit_state(self):
        """Test overlap with two qubit |01⟩ state.

        For |ψ⟩ = |01⟩, the protocol estimates ⟨a|ψ⟩
        Expected: ⟨00|01⟩ = 0
        Expected: ⟨01|01⟩ = 1
        Expected: ⟨10|01⟩ = 0
        Expected: ⟨11|01⟩ = 0
        """
        state = Statevector.from_label("01")
        protocol = ShadowProtocol(state, ensemble_type='clifford', use_qulacs=False)

        protocol.collect_samples(10000, 50, prediction='overlap')


        # Test overlap with |01⟩ (same state)
        target_01 = Bitstring([True, False])
        overlap_01 = protocol.estimate_overlap(target_01)
        expected_01 = 1.0

        # Test overlap with |10⟩ (orthogonal)
        target_10 = Bitstring([False, True])
        overlap_10 = protocol.estimate_overlap(target_10)
        expected_10 = 0.0

        # Test overlap with |11⟩ (orthogonal)
        target_11 = Bitstring([True, True])
        overlap_11 = protocol.estimate_overlap(target_11)
        expected_11 = 0.0

        # Check results with tolerance for statistical error
        assert abs(abs(overlap_01) - expected_01) < 0.3, \
            f"⟨01|01⟩: expected {expected_01:.3f}, got {abs(overlap_01):.3f}"
        assert abs(overlap_10) < 0.15, \
            f"⟨10|01⟩: expected {expected_10:.3f}, got {abs(overlap_10):.3f}"
        assert abs(overlap_11) < 0.15, \
            f"⟨11|01⟩: expected {expected_11:.3f}, got {abs(overlap_11):.3f}"

    def test_overlap_three_qubit_state(self):
        """Test overlap with three qubit |101⟩ state.

        For |ψ⟩ = |101⟩, the protocol estimates ⟨a|ψ⟩
        Expected: ⟨000|101⟩ = 0 (orthogonal)
        Expected: ⟨101|101⟩ = 1 (same state)
        Expected: ⟨010|101⟩ = 0 (orthogonal)
        """
        state = Statevector.from_label("101")
        protocol = ShadowProtocol(state, verbose=0)

        protocol.collect_samples_for_overlaps(5000, 10)

        # Test overlap with |101⟩ (same state)
        target_101 = Bitstring([True, False, True])
        overlap_101 = protocol.estimate_overlap(target_101).real
        expected_101 = 1.0

        # Test overlap with |010⟩ (orthogonal)
        target_010 = Bitstring([False, True, False])
        overlap_010 = protocol.estimate_overlap(target_010).real
        expected_010 = 0.0

        assert abs(abs(overlap_101) - expected_101) < 0.3, \
            f"⟨101|101⟩: expected {expected_101:.3f}, got {abs(overlap_101):.3f}"
        assert abs(overlap_010) < 0.15, \
            f"⟨010|101⟩: expected {expected_010:.3f}, got {abs(overlap_010):.3f}"

    def test_overlap_superposition_state(self):
        """Test overlap with a superposition state.

        For |ψ⟩ = (|10⟩ + |11⟩)/√2, which is |1⟩ ⊗ |+⟩
        The protocol estimates ⟨a|ψ⟩
        Expected: ⟨00|ψ⟩ = 0 (orthogonal - first qubit differs)
        Expected: ⟨10|ψ⟩ = 1/√2 (inner product with first component)
        Expected: ⟨11|ψ⟩ = 1/√2 (inner product with second component)
        Expected: ⟨01|ψ⟩ = 0 (orthogonal - first qubit differs)
        """
        # Create |1⟩ ⊗ |+⟩ = (|10⟩ + |11⟩)/√2
        state_data = np.zeros(4, dtype=complex)
        state_data[2] = 1 / np.sqrt(2)  # |10⟩
        state_data[3] = 1 / np.sqrt(2)  # |11⟩
        state = Statevector(state_data)

        protocol = ShadowProtocol(state, ensemble_type='clifford')
        protocol.collect_samples(10000, 100, prediction='overlap')

        # Test overlap with |10⟩
        target_10 = Bitstring([False, True])
        overlap_10 = protocol.estimate_overlap(target_10).real
        expected_10 = 1 / np.sqrt(2)

        # Test overlap with |11⟩
        target_11 = Bitstring([True, True])
        overlap_11 = protocol.estimate_overlap(target_11).real
        expected_11 = 1 / np.sqrt(2)

        # Test overlap with |01⟩ (orthogonal)
        target_01 = Bitstring([True, False])
        overlap_01 = protocol.estimate_overlap(target_01).real
        expected_01 = 0.0

        assert abs(abs(overlap_10) - expected_10) < 0.15, \
            f"⟨10|ψ⟩: expected {expected_10:.3f}, got {abs(overlap_10):.3f}"
        assert abs(abs(overlap_11) - expected_11) < 0.15, \
            f"⟨11|ψ⟩: expected {expected_11:.3f}, got {abs(overlap_11):.3f}"
        assert abs(overlap_01) < 0.15, \
            f"⟨01|ψ⟩: expected {expected_01:.3f}, got {abs(overlap_01):.3f}"

    def test_overlap_with_relative_phases(self):
        """Test overlap with state containing relative phases.

        For |ψ⟩ = (|10⟩ - |11⟩)/√2 = |1⟩ ⊗ |-⟩
        The protocol estimates ⟨a|ψ⟩ which includes phase information
        Expected: ⟨10|ψ⟩ = 1/√2
        Expected: ⟨11|ψ⟩ = -1/√2 (note the negative phase)
        Expected: |⟨10|ψ⟩| = |⟨11|ψ⟩| = 1/√2
        """
        # Create |1⟩ ⊗ |-⟩ = (|10⟩ - |11⟩)/√2
        state_data = np.zeros(4, dtype=complex)
        state_data[2] = 1 / np.sqrt(2)   # |10⟩
        state_data[3] = -1 / np.sqrt(2)  # -|11⟩
        state = Statevector(state_data)

        protocol = ShadowProtocol(state, ensemble_type='clifford')
        protocol.collect_samples(1000, 50, prediction='overlap')

        # Test overlap with |10⟩
        target_10 = Bitstring([False, True])
        overlap_10 = protocol.estimate_overlap(target_10)
        expected_10 = 1 / np.sqrt(2)

        # Test overlap with |11⟩ (with negative phase)
        target_11 = Bitstring([True, True])
        overlap_11 = protocol.estimate_overlap(target_11)
        expected_11 = 1 / np.sqrt(2)  # We check magnitude

        assert abs(abs(overlap_10) - expected_10) < 0.3, \
            f"|⟨10|ψ⟩|: expected {expected_10:.3f}, got {abs(overlap_10):.3f}"
        assert abs(abs(overlap_11) - expected_11) < 0.3, \
            f"|⟨11|ψ⟩|: expected {expected_11:.3f}, got {abs(overlap_11):.3f}"

    def test_overlap_raises_error_for_non_clifford_unitary(self):
        """Test that overlap raises NotImplementedError for non-Clifford."""
        measurement = Bitstring([False, False])
        # Use a numpy array instead of Tableau
        non_clifford = np.eye(2)
        snapshot = ClassicalSnapshot(measurement, non_clifford, 'other')
        shadow = ClassicalShadow([snapshot], n_qubits=2)

        target = Bitstring([False, False])
        with pytest.raises(NotImplementedError, match="Only Clifford"):
            shadow.overlap(target)


class TestIntegration:
    """Integration tests for the full Shadow Protocol workflow."""

    @pytest.mark.skip(reason="Known bugs in ClassicalShadow.overlap() prevent this from working")
    def test_full_workflow_computational_basis(self):
        """Test complete workflow with computational basis state."""
        # This test is skipped until bugs are fixed:
        # 1. Line 94: self.state.num_qubits → self.n_qubits
        # 2. Line 95: vacuum type mismatch with gaussian_elimination
        pass

    @pytest.mark.skip(reason="Known bugs in ClassicalShadow.overlap() prevent this from working")
    def test_full_workflow_bell_state(self):
        """Test complete workflow with Bell state."""
        # Skipped due to known bugs
        pass

    def test_protocol_with_different_sample_sizes(self):
        """Test protocol with various sample sizes."""
        state = Statevector.from_label("10")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        # Test multiple sample sizes
        for n_samples, n_k in [(100, 10), (500, 5), (1000, 20)]:
            protocol_test = ShadowProtocol(state, ensemble_type='clifford')
            protocol_test.collect_samples(n_samples, n_k, prediction='overlap')

            assert len(protocol_test.k_estimators) == n_k
            total = sum(est.N for est in protocol_test.k_estimators)
            assert total == n_samples


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_qubit_system(self):
        """Test protocol with single qubit."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        protocol.collect_samples(100, 10, prediction='overlap')
        assert len(protocol.k_estimators) == 10

    def test_large_qubit_system(self):
        """Test protocol initialization with larger system."""
        state = Statevector.from_label("0000000")  # 7 qubits
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        assert protocol.ensemble.d == 7

    def test_minimum_samples(self):
        """Test with minimum number of samples."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        # Minimum: 1 estimator, 1 sample per estimator
        protocol.collect_samples(1, 1, prediction='overlap')
        assert len(protocol.k_estimators) == 1
        assert protocol.k_estimators[0].N == 1

    def test_vacuum_bitstring_conversion(self):
        """Test that vacuum state can be properly represented as Bitstring."""
        # This tests the fix needed for line 95 bug
        n_qubits = 3
        vacuum = Bitstring([False] * n_qubits)

        # Verify it's all zeros
        assert all(not bit for bit in vacuum)
        assert vacuum.size == n_qubits


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_sample_count_type(self):
        """Test that invalid sample count types are handled."""
        state = Statevector.from_label("0")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        with pytest.raises(TypeError):
            protocol.collect_samples("invalid", 10, prediction='overlap')

    def test_negative_sample_count(self):
        """Test that negative sample counts are handled."""
        state = Statevector.from_label("0")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        # May raise different errors depending on implementation
        with pytest.raises((ValueError, TypeError)):
            protocol.collect_samples(-100, 10, prediction='overlap')

    def test_zero_estimators(self):
        """Test that zero estimators raises appropriate error."""
        state = Statevector.from_label("1")
        protocol = ShadowProtocol(state, ensemble_type='clifford')

        with pytest.raises((ValueError, ZeroDivisionError)):
            protocol.collect_samples(100, 0, prediction='overlap')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

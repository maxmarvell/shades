"""
Sample from Clifford group and verify fidelity-based sampling distribution.

This script demonstrates that when sampling from the Clifford group and measuring
a quantum state, the resulting stabilizer states appear with probability proportional
to their fidelity with the original state.

Theoretical background:
- Each Clifford unitary U maps the state |ψ⟩ to U|ψ⟩
- Measurement collapses U|ψ⟩ to a computational basis state |b⟩
- The pair (U, b) defines a stabilizer state via the stabilizers U†(Z_i ± I)
- The probability of observing (U, b) is |⟨b|U|ψ⟩|²
- This is proportional to the fidelity between |ψ⟩ and the stabilizer state
"""

import numpy as np
import stim
from qiskit.quantum_info import Statevector, random_statevector
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import qulacs

from shades.utils import bitstring_to_stabilizers, compute_x_rank, canonicalize, tableau_to_qulacs_circuit


def generate_random_state(n_qubits: int, seed: int = None) -> Statevector:
    """Generate a random quantum state over n qubits.

    Args:
        n_qubits: Number of qubits
        seed: Random seed for reproducibility

    Returns:
        Random statevector
    """
    if seed is not None:
        np.random.seed(seed)
    return random_statevector(2**n_qubits)


def sample_clifford_and_measure(
    state: Statevector,
) -> Tuple[stim.Tableau, int]:
    """Apply a random Clifford and measure in computational basis.

    Args:
        state: Input quantum state

    Returns:
        Tuple of (Clifford tableau, measurement outcome as int)
    """
    n_qubits = state.num_qubits

    # Generate random Clifford tableau
    tableau = stim.Tableau.random(n_qubits)

    qulacs_state = qulacs.QuantumState(n_qubits)
    qulacs_state.load(state)
    circuit = tableau_to_qulacs_circuit(tableau, n_qubits)
    circuit.update_quantum_state(qulacs_state)
    sample = qulacs_state.sampling(1)[0]

    return tableau, sample


def create_stabilizer_snapshot(
    tableau: stim.Tableau,
    bitstring: int,
    n_qubits: int = None,
) -> stim.Tableau:
    """Create a stabilizer snapshot from a Clifford and measurement.

    The snapshot represents the stabilizer state that would be reconstructed
    from this measurement outcome after applying the Clifford.

    Args:
        tableau: The Clifford tableau that was applied
        bitstring: The measurement outcome as int (little-endian)
        n_qubits: Number of qubits (derived from tableau if not given)

    Returns:
        Stabilizer tableau representing the snapshot
    """
    if n_qubits is None:
        n_qubits = len(tableau)

    # Convert measurement to stabilizers (Z_i eigenstates)
    stabilizers = bitstring_to_stabilizers(bitstring, n_qubits)

    # Transform back through inverse Clifford
    U_inv = tableau.inverse()
    transformed_stabilizers = [U_inv(s) for s in stabilizers]

    # Canonicalize and create tableau
    canonical_stabilizers = canonicalize(transformed_stabilizers)
    snapshot = stim.Tableau.from_stabilizers(canonical_stabilizers)

    return snapshot


def compute_stabilizer_fidelity(state: Statevector, tableau: stim.Tableau) -> float:
    """Compute fidelity between a state and a stabilizer state.

    Args:
        state: Original quantum state
        tableau: Stabilizer tableau defining the stabilizer state

    Returns:
        Fidelity (between 0 and 1)
    """
    n_qubits = state.num_qubits

    # Convert tableau to state vector
    s = tableau.to_state_vector()

    # Compute overlap <state|s>
    overlap = np.vdot(state.data, s)

    # Fidelity is |overlap|^2
    fidelity = np.abs(overlap) ** 2

    return float(np.real(fidelity))


def collect_clifford_samples(
    state: Statevector,
    n_samples: int,
    use_qulacs: bool = True,
    verbose: bool = True
) -> List[stim.Tableau]:
    """Collect samples by applying random Cliffords and measuring.

    Args:
        state: Quantum state to sample from
        n_samples: Number of samples to collect
        use_qulacs: Use Qulacs for faster simulation
        verbose: Print progress

    Returns:
        List of stabilizer snapshots
    """
    n_qubits = state.num_qubits
    snapshots = []

    if verbose:
        print(f"\nCollecting {n_samples} Clifford samples on {n_qubits} qubits...")
        print(f"Backend: {'Qulacs' if use_qulacs else 'Qiskit'}")

    for i in range(n_samples):
        # Sample random Clifford and measure
        tableau, bitstring = sample_clifford_and_measure(state)

        # Create stabilizer snapshot
        snapshot = create_stabilizer_snapshot(tableau, bitstring)
        snapshots.append(snapshot)

        # Progress reporting
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            progress = (i + 1) / n_samples * 100
            print(f"  Progress: {i+1}/{n_samples} ({progress:.0f}%)")

    if verbose:
        print(f"Collected {len(snapshots)} samples")

    return snapshots


def analyze_stabilizer_distribution(
    state: Statevector,
    snapshots: List[stim.Tableau],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze the distribution of stabilizer states.

    Args:
        state: Original quantum state
        snapshots: List of stabilizer snapshots
        verbose: Print analysis

    Returns:
        Tuple of (fidelities, counts) arrays
    """
    if verbose:
        print(f"\nAnalyzing {len(snapshots)} snapshots...")

    # Group snapshots by unique stabilizer states
    tableau_counts = defaultdict(int)
    tableau_map = {}

    for snapshot in snapshots:
        # Create signature from stabilizers
        stabilizers = snapshot.to_stabilizers()
        signature = tuple(str(s) for s in stabilizers)

        tableau_counts[signature] += 1
        if signature not in tableau_map:
            tableau_map[signature] = snapshot

    if verbose:
        print(f"Found {len(tableau_counts)} unique stabilizer states")

    # Compute fidelity for each unique state
    fidelities = []
    counts = []

    for i, (signature, count) in enumerate(tableau_counts.items()):
        tableau = tableau_map[signature]
        fidelity = compute_stabilizer_fidelity(state, tableau)
        fidelities.append(fidelity)
        counts.append(count)

        if verbose and (i + 1) % max(1, len(tableau_counts) // 10) == 0:
            print(f"  Progress: {i+1}/{len(tableau_counts)} ({(i+1)/len(tableau_counts)*100:.0f}%)")

    fidelities = np.array(fidelities)
    counts = np.array(counts)

    if verbose:
        print("\nFidelity Analysis:")
        print(f"  Mean fidelity (weighted): {np.average(fidelities, weights=counts):.4f}")
        print(f"  Max fidelity: {np.max(fidelities):.4f}")
        print(f"  Min fidelity: {np.min(fidelities):.4f}")
        print(f"\nSampling frequency:")
        print(f"  Total samples: {np.sum(counts)}")
        print(f"  Most frequent state: {np.max(counts)} samples")
        print(f"  Least frequent state: {np.min(counts)} samples")

        # Correlation analysis
        correlation = np.corrcoef(fidelities, counts)[0, 1]
        print(f"\nCorrelation (fidelity vs count): {correlation:.4f}")

        # Top states by fidelity
        sorted_indices = np.argsort(fidelities)[::-1]
        print("\nTop 5 states by fidelity:")
        for i in range(min(5, len(fidelities))):
            idx = sorted_indices[i]
            print(f"  Rank {i+1}: Fidelity = {fidelities[idx]:.4f}, Count = {counts[idx]}")

    return fidelities, counts


def plot_fidelity_vs_frequency(
    fidelities: np.ndarray,
    counts: np.ndarray,
    n_qubits: int,
    n_samples: int,
    save_path: str = None
):
    """Plot the relationship between fidelity and sampling frequency.

    Args:
        fidelities: Array of fidelities
        counts: Array of counts for each stabilizer state
        n_qubits: Number of qubits
        n_samples: Total number of samples
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    ax1.scatter(fidelities, counts, alpha=0.6, s=50)
    ax1.set_xlabel('Fidelity F(|ψ⟩, |stabilizer⟩)', fontsize=12)
    ax1.set_ylabel('Sampling Count', fontsize=12)
    ax1.set_title(f'Fidelity vs Sampling Frequency\n({n_qubits} qubits, {n_samples} samples)',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)

    correlation = np.corrcoef(fidelities, counts)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]
    ax2.hist(fidelities, weights=counts, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Fidelity F(|ψ⟩, |stabilizer⟩)', fontsize=12)
    ax2.set_ylabel('Total Samples', fontsize=12)
    ax2.set_title('Distribution of All Samples\n(many low-F states × few each)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[2]
    ax3.hist(fidelities, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax3.set_xlabel('Fidelity F(|ψ⟩, |stabilizer⟩)', fontsize=12)
    ax3.set_ylabel('Number of Unique States', fontsize=12)
    ax3.set_title('Distribution of Unique States\n(unweighted)', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()


def main():
    """Run the Clifford sampling experiment."""

    n_qubits = 3
    n_samples = 500000
    use_qulacs = True
    seed = 42

    print("=" * 70)
    print("Clifford Group Sampling: Fidelity-Based Distribution")
    print("=" * 70)

    print(f"\nGenerating random {n_qubits}-qubit state (seed={seed})...")
    state = generate_random_state(n_qubits, seed=seed)
    print(f"State generated. Norm: {np.linalg.norm(state.data):.6f}")

    print("\n" + "-" * 70)
    snapshots = collect_clifford_samples(
        state, n_samples, use_qulacs=use_qulacs, verbose=True
    )

    print("\n" + "-" * 70)
    fidelities, counts = analyze_stabilizer_distribution(state, snapshots, verbose=True)

    print("\n" + "-" * 70)
    print("Generating plots...")
    plot_fidelity_vs_frequency(fidelities, counts, n_qubits, n_samples)

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)

    print("\nStatistical Interpretation:")
    print("If sampling is correct, we expect:")
    print("  1. Positive correlation between fidelity and count")
    print("  2. Higher-fidelity stabilizer states appear more frequently")
    print("  3. Distribution weighted toward high-fidelity states")

    correlation = np.corrcoef(fidelities, counts)[0, 1]
    if correlation > 0.5:
        print(f"\n✓ Strong positive correlation ({correlation:.3f}) confirms the theory!")
    elif correlation > 0.3:
        print(f"\n✓ Moderate positive correlation ({correlation:.3f}) supports the theory.")
        print("  (Increase n_samples for stronger evidence)")
    else:
        print(f"\n⚠ Weak correlation ({correlation:.3f}). Consider increasing n_samples.")


if __name__ == "__main__":
    main()

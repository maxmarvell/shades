"""Subspace Expansion Convergence Analysis.

This script demonstrates how the StabilizerSubspace energy estimates converge
to the exact ground state energy as the number of sampled stabilizer states increases.

The analysis systematically varies N (number of stabilizer states from 1 to 25),
constructs a StabilizerSubspace from Clifford shadow samples, and compares the
optimized energy to the exact result from full diagonalization.

Key convergence properties tested:
- Energy estimates approach exact value as N → ∞
- Error decreases with increasing subspace dimension
- Multiple runs provide statistical error bars
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import Statevector
from scipy.linalg import eigh

from shades.stabilizer_subspace import StabilizerSubspace, compose_tableau_bitstring
from shades.stabilizer import StabilizerState
from shades.tomography import CliffordShadow


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Sample schedule (number of stabilizer states to test)
SAMPLE_SCHEDULE = np.arange(1, 26)  # N = 1, 2, 3, ..., 25

# Number of independent runs per N (for statistics)
N_RUNS = 10

# Regularization for optimize_coefficients
REGULARIZATION = 1e-10

# Output configuration
OUTPUT_DIR = "./results/expansion_convergence"

# Plotting parameters
FIGURE_SIZE = (12, 4)
PLOT_DPI = 300


def main():
    """Run subspace expansion convergence analysis."""
    print("=" * 70)
    print("Subspace Expansion Convergence Analysis")
    print("=" * 70)

    # ========================================================================
    # Step 1: Define Hamiltonian (same as stabilizer_subspace.py)
    # ========================================================================
    print("\nSetting up Hamiltonian...")

    # Define fermionic Hamiltonian (4 spin-orbitals, H2-like system)
    fermionic_op = FermionicOp({
        "+_0 -_2": -1.0, "+_2 -_0": -1.0,      # Hopping terms
        "+_1 -_3": -1.0, "+_3 -_1": -1.0,
        "+_0 -_0 +_1 -_1": 2.0,                # Coulomb repulsion
        "+_2 -_2 +_3 -_3": 2.0,
    }, num_spin_orbitals=4)

    # Convert to qubit Hamiltonian via Jordan-Wigner mapping
    qubit_op = JordanWignerMapper().map(fermionic_op)
    pauli_hamiltonian = [(coeff, label) for label, coeff in qubit_op.label_iter()]

    # Get exact ground state (reference for convergence)
    matrix = qubit_op.to_matrix()
    eigenvalues, eigenvectors = eigh(matrix)
    ground_energy_exact = eigenvalues[0]
    ground_state = Statevector(eigenvectors[:, 0])

    print(f"Exact ground state energy: {ground_energy_exact:.10f}")
    print(f"System size: {len(ground_state)} dimensional Hilbert space")

    # ========================================================================
    # Step 2: Main convergence loop
    # ========================================================================
    print(f"\nRunning convergence analysis...")
    print(f"Sample schedule: N = {SAMPLE_SCHEDULE[0]} to {SAMPLE_SCHEDULE[-1]}")
    print(f"Runs per sample count: {N_RUNS}")
    print("-" * 70)

    # Storage for results
    all_energies = np.empty((len(SAMPLE_SCHEDULE), N_RUNS))
    all_errors = np.empty((len(SAMPLE_SCHEDULE), N_RUNS))

    for i, N in enumerate(SAMPLE_SCHEDULE):
        print(f"\nN = {N} stabilizer states")

        for run in range(N_RUNS):
            # Sample N stabilizer states from ground state via Clifford shadows
            samples = [CliffordShadow.sample_state(ground_state) for _ in range(N)]

            # Compose tableaus with bitstrings to create stabilizer snapshots
            tabs = [compose_tableau_bitstring(tab, b) for b, tab in samples]

            # Convert to StabilizerState objects
            states = [StabilizerState.from_stim_tableau(t) for t in tabs]

            # Construct subspace
            subspace = StabilizerSubspace(states, pauli_hamiltonian)

            # Optional validation (can be disabled for performance)
            if N <= 5 and run == 0:  # Only check for small N, first run
                try:
                    assert np.allclose(np.diag(subspace.S), np.ones(N), atol=1e-10), \
                        f'States not normalized! Diag(S) = {np.diag(subspace.S)}'
                    assert np.allclose(subspace.S, subspace.S.conj().T, atol=1e-10), \
                        'S not Hermitian!'
                    assert np.allclose(subspace.H, subspace.H.conj().T, atol=1e-10), \
                        'H not Hermitian!'
                except AssertionError as e:
                    print(f"  Warning: {e}")

            # Optimize coefficients to find ground state in subspace
            energy, coeffs = subspace.optimize_coefficients(reg=REGULARIZATION)

            # Store results
            all_energies[i, run] = energy
            all_errors[i, run] = abs(energy - ground_energy_exact)

            if run < 3 or run == N_RUNS - 1:  # Print first 3 and last run
                print(f"  Run {run+1}/{N_RUNS}: E = {energy:.8f}, "
                      f"Error = {all_errors[i, run]:.2e}")
            elif run == 3:
                print(f"  ...")

    # ========================================================================
    # Step 3: Statistical analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)

    # Compute statistics across runs
    mean_energies = np.mean(all_energies, axis=1)
    std_energies = np.std(all_energies, axis=1, ddof=1)  # Use ddof=1 for sample std
    sem_energies = std_energies / np.sqrt(N_RUNS)

    mean_errors = np.mean(all_errors, axis=1)
    std_errors = np.std(all_errors, axis=1, ddof=1)
    sem_errors = std_errors / np.sqrt(N_RUNS)

    # Print summary table
    print(f"{'N':<5} {'Mean Energy':<18} {'Std Energy':<15} {'Mean Error':<15}")
    print("-" * 70)
    for i, N in enumerate(SAMPLE_SCHEDULE):
        print(f"{N:<5} {mean_energies[i]:<18.10f} {std_energies[i]:<15.2e} "
              f"{mean_errors[i]:<15.2e}")

    print("\n" + "=" * 70)
    print(f"Exact ground state energy: {ground_energy_exact:.10f}")
    print(f"Final estimate (N={SAMPLE_SCHEDULE[-1]}): {mean_energies[-1]:.10f} "
          f"± {sem_energies[-1]:.2e}")
    print(f"Final error: {mean_errors[-1]:.2e} ± {sem_errors[-1]:.2e}")

    # ========================================================================
    # Step 4: Data export
    # ========================================================================
    print(f"\nSaving results to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save summary statistics
    summary_df = pd.DataFrame({
        'N_states': SAMPLE_SCHEDULE,
        'mean_energy': mean_energies,
        'std_energy': std_energies,
        'sem_energy': sem_energies,
        'mean_error': mean_errors,
        'std_error': std_errors,
        'sem_error': sem_errors,
        'exact_energy': ground_energy_exact
    })
    summary_path = os.path.join(OUTPUT_DIR, "convergence_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary statistics: {summary_path}")

    # Save all raw data
    all_runs_df = pd.DataFrame(
        all_energies,
        columns=[f'run_{i+1}' for i in range(N_RUNS)],
        index=SAMPLE_SCHEDULE
    )
    all_runs_df.index.name = 'N_states'
    raw_path = os.path.join(OUTPUT_DIR, "all_runs.csv")
    all_runs_df.to_csv(raw_path)
    print(f"  Raw data: {raw_path}")

    # ========================================================================
    # Step 5: Visualization
    # ========================================================================
    print("\nGenerating convergence plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # Plot 1: Energy convergence
    ax1.axhline(ground_energy_exact, color='red', linestyle='--',
                label='Exact', linewidth=2, zorder=1)
    ax1.errorbar(SAMPLE_SCHEDULE, mean_energies, yerr=sem_energies,
                 marker='o', capsize=4, label='Subspace estimate',
                 linewidth=1.5, markersize=6, zorder=2)
    ax1.set_xlabel('Number of stabilizer states (N)', fontsize=11)
    ax1.set_ylabel('Ground state energy', fontsize=11)
    ax1.set_title('Subspace Expansion Convergence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error vs N (log-log)
    ax2.loglog(SAMPLE_SCHEDULE, mean_errors, marker='o', label='Mean error',
               linewidth=1.5, markersize=6)
    ax2.fill_between(SAMPLE_SCHEDULE,
                      np.maximum(mean_errors - sem_errors, 1e-12),  # Avoid log(0)
                      mean_errors + sem_errors,
                      alpha=0.3)

    # Reference line: 1/sqrt(N) scaling (if first point is reasonable)
    if mean_errors[0] > 0:
        reference_scaling = mean_errors[0] / np.sqrt(SAMPLE_SCHEDULE / SAMPLE_SCHEDULE[0])
        ax2.loglog(SAMPLE_SCHEDULE, reference_scaling,
                   'k--', alpha=0.5, linewidth=1.5, label=r'$1/\sqrt{N}$ reference')

    ax2.set_xlabel('Number of stabilizer states (N)', fontsize=11)
    ax2.set_ylabel(r'$|E - E_{\mathrm{exact}}|$', fontsize=11)
    ax2.set_title('Error Convergence', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, "convergence_plot.png")
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

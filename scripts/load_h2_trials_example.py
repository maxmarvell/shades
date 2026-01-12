"""Example script demonstrating how to load and analyze saved H2 stretching trial data.

This script shows how to:
1. Load the .npz file containing all individual trial data
2. Access the different arrays and metadata
3. Perform custom analysis on individual trials
4. Generate additional plots
"""

import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze_trials(npz_filename):
    """Load and analyze H2 stretching trial data from .npz file.

    Args:
        npz_filename: Path to the .npz file (e.g., 'h2_stretching_N8_all_trials.npz')
    """

    # Load the data
    print(f"Loading data from: {npz_filename}")
    data = np.load(npz_filename)

    # Display what's available
    print("\nAvailable arrays in .npz file:")
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim > 0:
            print(f"  {key:20s} shape: {arr.shape}")
        else:
            print(f"  {key:20s} value: {arr}")

    # Extract the arrays
    distances = data['distances']
    exact_fci = data['exact_fci']
    all_energies = data['all_energies']  # Shape: (n_distances, n_simulations)
    all_c0 = data['all_c0']
    all_c2_norms = data['all_c2_norms']

    # Extract metadata
    n_samples = data['n_samples']
    n_estimators = data['n_estimators']
    n_simulations = data['n_simulations']
    n_hydrogen = data['n_hydrogen']

    print(f"\nExperiment configuration:")
    print(f"  H{n_hydrogen} molecule")
    print(f"  {n_samples} samples/run, {n_estimators} estimators")
    print(f"  {n_simulations} independent trials per distance")
    print(f"  {len(distances)} distances: {distances}")

    # Example analysis: histogram of errors at each distance
    print("\n" + "="*60)
    print("Example Analysis: Error Distribution")
    print("="*60)

    for i, d in enumerate(distances):
        trial_energies = all_energies[i, :]
        exact = exact_fci[i]
        errors = trial_energies - exact

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        print(f"Distance {d:.2f} Å:")
        print(f"  Mean error:  {mean_error:+.2e} Ha")
        print(f"  Std error:   {std_error:.2e} Ha")
        print(f"  Min error:   {np.min(errors):+.2e} Ha")
        print(f"  Max error:   {np.max(errors):+.2e} Ha")

    # Example plot: Error distribution for middle distance
    mid_idx = len(distances) // 2
    mid_distance = distances[mid_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Histogram of errors
    trial_energies = all_energies[mid_idx, :]
    exact = exact_fci[mid_idx]
    errors = (trial_energies - exact) * 1000  # Convert to mHa

    axes[0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Exact')
    axes[0].set_xlabel('Error (mHa)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution at {mid_distance:.2f} Å')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Correlation between |c0| and error
    c0_vals = all_c0[mid_idx, :]

    axes[1].scatter(c0_vals, errors, alpha=0.5, s=20)
    axes[1].set_xlabel(r'$|c_0|$ (HF overlap)')
    axes[1].set_ylabel('Error (mHa)')
    axes[1].set_title(f'Error vs HF Overlap at {mid_distance:.2f} Å')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('h2_trial_analysis.png', dpi=150)
    print(f"\nPlot saved to: h2_trial_analysis.png")
    plt.show()

    # Example: Access individual trial data
    print("\n" + "="*60)
    print("Example: Individual Trial Access")
    print("="*60)

    # Get the first trial at the first distance
    trial_idx = 0
    dist_idx = 0

    print(f"Trial {trial_idx} at distance {distances[dist_idx]:.2f} Å:")
    print(f"  Energy:    {all_energies[dist_idx, trial_idx]:.8f} Ha")
    print(f"  |c0|:      {all_c0[dist_idx, trial_idx]:.6f}")
    print(f"  ||c2||:    {all_c2_norms[dist_idx, trial_idx]:.6f}")
    print(f"  Exact FCI: {exact_fci[dist_idx]:.8f} Ha")
    print(f"  Error:     {(all_energies[dist_idx, trial_idx] - exact_fci[dist_idx])*1000:.2f} mHa")

    return data


if __name__ == "__main__":
    # Example usage
    # Replace with your actual filename
    npz_file = "h2_stretching_N8_all_trials.npz"

    try:
        data = load_and_analyze_trials(npz_file)
        print("\n✓ Successfully loaded and analyzed trial data!")
    except FileNotFoundError:
        print(f"\n✗ File not found: {npz_file}")
        print("  Run h2_stretching.py first to generate the data file.")

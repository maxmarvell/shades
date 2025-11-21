"""
Plot Brueckner Convergence Results

This script loads and plots the convergence data from Brueckner orbital optimization,
comparing TrivialEstimator (exact) and ShadowEstimator (sampling-based) results.

Usage:
    python scripts/plot_brueckner_convergence.py [--data-file PATH] [--output-dir PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Plotting configuration
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8),
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})


def load_convergence_data(data_file: Path):
    """Load convergence data from npz file."""
    data = np.load(data_file)

    # Extract data
    result = {
        # Trivial estimator (exact)
        'energies_trivial': data['energies_trivial'],
        'c0_values_trivial': data['c0_values_trivial'],
        'c1_norms_trivial': data['c1_norms_trivial'],
        # Shadow estimator (mean and std)
        'energies_shadow_mean': data['energies_shadow_mean'],
        'energies_shadow_std': data['energies_shadow_std'],
        'c0_shadow_mean': data['c0_shadow_mean'],
        'c0_shadow_std': data['c0_shadow_std'],
        'c1_norms_shadow_mean': data['c1_norms_shadow_mean'],
        'c1_norms_shadow_std': data['c1_norms_shadow_std'],
        # Metadata
        'n_samples': int(data['n_samples']),
        'n_k_estimators': int(data['n_k_estimators']),
        'n_repetitions': int(data['n_repetitions']),
    }

    # Handle legacy data format - always reconstruct iterations from data length
    # This ensures compatibility with old data files where iterations array might be mismatched
    result['iterations'] = np.arange(len(data['energies_trivial']))

    # HOTFIX: Remove iteration 0 (duplicate of iteration 1 due to callback timing bug)
    # Skip the first data point which is erroneously duplicated
    if len(result['iterations']) > 1:
        result['energies_trivial'] = result['energies_trivial'][1:]
        result['c0_values_trivial'] = result['c0_values_trivial'][1:]
        result['c1_norms_trivial'] = result['c1_norms_trivial'][1:]
        result['energies_shadow_mean'] = result['energies_shadow_mean'][1:]
        result['energies_shadow_std'] = result['energies_shadow_std'][1:]
        result['c0_shadow_mean'] = result['c0_shadow_mean'][1:]
        result['c0_shadow_std'] = result['c0_shadow_std'][1:]
        result['c1_norms_shadow_mean'] = result['c1_norms_shadow_mean'][1:]
        result['c1_norms_shadow_std'] = result['c1_norms_shadow_std'][1:]
        # Renumber iterations to start from 1
        result['iterations'] = np.arange(1, len(result['energies_trivial']) + 1)

    return result


def plot_energy_convergence(data, output_dir: Path):
    """Plot energy convergence with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = data['iterations']
    E_trivial = data['energies_trivial']
    E_shadow_mean = data['energies_shadow_mean']
    E_shadow_std = data['energies_shadow_std']

    # Get FCI energy (should be constant, take from trivial estimator)
    E_fci = E_trivial[-1]  # Assuming converged value is close to FCI

    # Plot exact FCI reference
    ax.axhline(E_fci, color='gray', linestyle='--', linewidth=1.5,
               label=f'Exact FCI: {E_fci:.8f} Ha', zorder=1)

    # Plot Trivial estimator (exact)
    ax.plot(iterations, E_trivial, 'o-', color='C0',
            label='Trivial Estimator (Exact)', zorder=3)

    # Plot Shadow estimator with error bars
    ax.errorbar(iterations, E_shadow_mean, yerr=E_shadow_std,
                fmt='s-', color='C1', capsize=4, capthick=1.5,
                label=f'Shadow Estimator (N={data["n_samples"]}, K={data["n_k_estimators"]}, n={data["n_repetitions"]})',
                zorder=2)

    ax.set_xlabel('Brueckner Iteration')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('Brueckner Orbital Optimization: Energy Convergence')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set y-axis limits to ±1 Ha around FCI energy
    ax.set_ylim(E_fci - 1.0, E_fci + 1.0)

    plt.tight_layout()

    output_file = output_dir / 'energy_convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    return fig


def plot_c1_convergence(data, output_dir: Path):
    """Plot ||c1|| norm convergence."""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = data['iterations']
    c1_trivial = data['c1_norms_trivial']
    c1_shadow_mean = data['c1_norms_shadow_mean']
    c1_shadow_std = data['c1_norms_shadow_std']

    # Plot Trivial estimator (exact)
    ax.semilogy(iterations, c1_trivial, 'o-', color='C0',
                label='Trivial Estimator (Exact)')

    # Plot Shadow estimator with error bars
    ax.errorbar(iterations, c1_shadow_mean, yerr=c1_shadow_std,
                fmt='s-', color='C1', capsize=4, capthick=1.5,
                label=f'Shadow Estimator (n={data["n_repetitions"]} reps)')

    ax.set_xlabel('Brueckner Iteration')
    ax.set_ylabel('||c₁|| (Singles Norm)')
    ax.set_title('Brueckner Orbital Optimization: Singles Amplitude Convergence')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_file = output_dir / 'c1_convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    return fig


def plot_error_analysis(data, output_dir: Path):
    """Plot error between Shadow and Trivial estimators."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    iterations = data['iterations']
    E_trivial = data['energies_trivial']
    E_shadow_mean = data['energies_shadow_mean']
    E_shadow_std = data['energies_shadow_std']
    c1_trivial = data['c1_norms_trivial']

    # Energy error
    energy_error = E_shadow_mean - E_trivial
    energy_error_mHa = energy_error * 1000  # Convert to milli-Hartree
    energy_error_std_mHa = E_shadow_std * 1000

    ax1.errorbar(iterations, energy_error_mHa, yerr=energy_error_std_mHa,
                 fmt='o-', color='C2', capsize=4, capthick=1.5,
                 label='Shadow - Trivial')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Brueckner Iteration')
    ax1.set_ylabel('Energy Error (mHa)')
    ax1.set_title('Shadow Estimator Error vs Trivial (Exact)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Error vs ||c1|| (to see if error correlates with reference quality)
    ax2.errorbar(c1_trivial, energy_error_mHa, yerr=energy_error_std_mHa,
                 fmt='o', color='C3', capsize=4, capthick=1.5,
                 label='Shadow Error')

    # Add iteration labels
    for i, (x, y) in enumerate(zip(c1_trivial, energy_error_mHa)):
        ax2.annotate(f'{int(iterations[i])}', (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.7)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('||c₁|| (Singles Norm)')
    ax2.set_ylabel('Energy Error (mHa)')
    ax2.set_title('Shadow Error vs Reference Quality')
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_file = output_dir / 'error_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    return fig


def plot_c0_convergence(data, output_dir: Path):
    """Plot reference overlap |c0| convergence."""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = data['iterations']
    c0_trivial = data['c0_values_trivial']
    c0_shadow_mean = data['c0_shadow_mean']
    c0_shadow_std = data['c0_shadow_std']

    # Plot Trivial estimator (exact)
    ax.plot(iterations, c0_trivial, 'o-', color='C0',
            label='Trivial Estimator (Exact)')

    # Plot Shadow estimator with error bars
    ax.errorbar(iterations, c0_shadow_mean, yerr=c0_shadow_std,
                fmt='s-', color='C1', capsize=4, capthick=1.5,
                label=f'Shadow Estimator (n={data["n_repetitions"]} reps)')

    ax.set_xlabel('Brueckner Iteration')
    ax.set_ylabel('|c₀| (Reference Overlap)')
    ax.set_title('Brueckner Orbital Optimization: Reference Determinant Overlap')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / 'c0_convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    return fig


def plot_variance_analysis(data, output_dir: Path):
    """Plot variance (standard deviation) evolution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    iterations = data['iterations']
    E_shadow_std = data['energies_shadow_std']
    c1_shadow_std = data['c1_norms_shadow_std']
    c1_trivial = data['c1_norms_trivial']

    # Energy variance vs iteration
    ax1.plot(iterations, E_shadow_std * 1000, 'o-', color='C4',
             label='Energy Std Dev')
    ax1.set_xlabel('Brueckner Iteration')
    ax1.set_ylabel('Energy Std Dev (mHa)')
    ax1.set_title(f'Shadow Estimator Variance (n={data["n_repetitions"]} repetitions)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Variance vs reference quality
    ax2.semilogy(c1_trivial, E_shadow_std * 1000, 'o-', color='C5',
                 label='Energy Std Dev vs ||c₁||')

    # Add iteration labels
    for i, (x, y) in enumerate(zip(c1_trivial, E_shadow_std * 1000)):
        ax2.annotate(f'{int(iterations[i])}', (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.7)

    ax2.set_xlabel('||c₁|| (Singles Norm)')
    ax2.set_ylabel('Energy Std Dev (mHa)')
    ax2.set_title('Shadow Variance vs Reference Quality')
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_file = output_dir / 'variance_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    return fig


def plot_all(data_file: Path, output_dir: Path):
    """Generate all plots."""
    print(f"Loading data from: {data_file}")
    data = load_convergence_data(data_file)

    print(f"\nDataset info:")
    print(f"  Iterations: {len(data['iterations'])}")
    print(f"  Shadow samples: N={data['n_samples']}, K={data['n_k_estimators']}")
    print(f"  Repetitions per iteration: {data['n_repetitions']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in: {output_dir}")

    # Generate all plots
    plot_energy_convergence(data, output_dir)
    plot_c1_convergence(data, output_dir)
    plot_c0_convergence(data, output_dir)
    plot_error_analysis(data, output_dir)
    plot_variance_analysis(data, output_dir)

    print("\nAll plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Brueckner convergence analysis results'
    )
    parser.add_argument(
        '--data-file',
        type=Path,
        default=Path('results/brueckner_convergence/convergence_data.npz'),
        help='Path to convergence data .npz file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save plots (default: same as data file directory)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = args.data_file.parent / 'plots'
    else:
        output_dir = args.output_dir

    # Generate plots
    plot_all(args.data_file, output_dir)


if __name__ == '__main__':
    main()

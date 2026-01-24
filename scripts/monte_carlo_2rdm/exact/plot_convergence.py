"""Convergence analysis for exact estimator Monte Carlo 2-RDM.

Plots energy error and RDM2 Frobenius error vs MC iterations
using exact wavefunction coefficients (no shadow noise).
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
from plotting_config import setup_plotting_style, save_figure


def load_data(data_dir: Path):
    """Load results from npz and metadata from json."""
    data = np.load(data_dir / "data.npz", allow_pickle=True)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    return data, metadata


def plot_convergence(data_dir: Path, output_dir: Path = None):
    """Plot convergence of energy and RDM2 error vs MC iterations."""
    data, metadata = load_data(data_dir)

    if output_dir is None:
        output_dir = data_dir

    plot_dir = output_dir / "convergence"
    plot_dir.mkdir(exist_ok=True)

    n_mc_iters = np.array(metadata["n_mc_iters"])
    E_fci = metadata["E_fci_hartree"]
    n_runs = metadata["n_runs"]

    E_tot = data["E_tot"]  # shape: (n_mc, n_runs)
    rel_frob = data["rel_frob_rdm2"]  # shape: (n_mc, n_runs)

    setup_plotting_style()
    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Energy error
    ax1 = axes[0]
    E_mean = E_tot.mean(axis=1)
    E_std = E_tot.std(axis=1, ddof=1)
    E_sem = E_std / np.sqrt(n_runs)
    error_mean = E_mean - E_fci

    ax1.errorbar(n_mc_iters, error_mean, yerr=E_sem, fmt="o-", capsize=3,
                 linewidth=1.5, markersize=5, label="Exact estimator")
    ax1.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel(r"Number of MC Steps ($N_\mathrm{MC}$)")
    ax1.set_ylabel(r"$E_\mathrm{MC} - E_\mathrm{FCI}$ (Ha)")
    ax1.set_xscale("log")
    ax1.set_title("Energy Error vs MC Steps")
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: RDM2 Frobenius error
    ax2 = axes[1]
    frob_mean = rel_frob.mean(axis=1)
    frob_std = rel_frob.std(axis=1, ddof=1)
    frob_sem = frob_std / np.sqrt(n_runs)

    ax2.errorbar(n_mc_iters, frob_mean, yerr=frob_sem, fmt="o-", capsize=3,
                 linewidth=1.5, markersize=5, label="Exact estimator")
    ax2.set_xlabel(r"Number of MC Steps ($N_\mathrm{MC}$)")
    ax2.set_ylabel(r"$\|\Delta\Gamma\|_F / \|\Gamma_\mathrm{ref}\|_F$")
    ax2.set_xscale("log")
    ax2.set_title("RDM2 Relative Frobenius Error vs MC Steps")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    save_figure(plot_dir / "convergence.pdf")
    save_figure(plot_dir / "convergence.png", dpi=300)
    save_figure(plot_dir / "convergence.svg")

    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("Convergence Summary")
    print("=" * 60)
    print(f"E_FCI: {E_fci:.10f} Ha")
    print(f"Final E_MC: {E_mean[-1]:.10f} ± {E_sem[-1]:.2e} Ha")
    print(f"Final error: {error_mean[-1]:.2e} Ha")
    print(f"Final rel. Frob. error: {frob_mean[-1]:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence for exact estimator")
    parser.add_argument("data_dir", type=Path, help="Directory containing data.npz and metadata.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    plot_convergence(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

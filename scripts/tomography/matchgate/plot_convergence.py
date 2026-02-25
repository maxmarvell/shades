"""Plot energy convergence and variance scaling from shadow budget sweep.

Loads results from a directory containing data.npz and metadata.json
(as produced by convergence.py) and produces a two-panel figure:
  - Upper: mean energy vs shadow budget with std-dev error bars
  - Lower: variance vs shadow budget on a log-log scale with fitted C/N line

The prefactor C in Var(E) = C / N_s is extracted via a least-squares fit
in log-space and annotated on the figure.

Usage:
    python scripts/tomography/matchgate/plot_convergence.py <results_directory>

Example:
    python scripts/tomography/matchgate/plot_convergence.py results/tomography/matchgate/convergence/2026-02-18_12-00-00/
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from plotting_config import setup_plotting_style, save_figure


def load_data(directory):
    """Load data.npz and metadata.json from the given directory."""
    npz_path = os.path.join(directory, "data.npz")
    meta_path = os.path.join(directory, "metadata.json")

    if not os.path.isfile(npz_path):
        sys.exit(f"Error: {npz_path} not found.")
    if not os.path.isfile(meta_path):
        sys.exit(f"Error: {meta_path} not found.")

    data = np.load(npz_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return data, metadata


def fit_prefactor(budgets, variance):
    """Fit Var = C / N^alpha via linear regression in log-space.

    Returns (C, alpha) where alpha should be close to 1.0.
    """
    log_n = np.log(budgets)
    log_var = np.log(variance)
    # log(Var) = log(C) - alpha * log(N)
    A = np.column_stack([np.ones_like(log_n), -log_n])
    (log_c, alpha), _, _, _ = np.linalg.lstsq(A, log_var, rcond=None)
    return np.exp(log_c), alpha


def plot_convergence(directory, output_filename=None):
    """Plot energy convergence and variance scaling against shadow budget."""
    data, metadata = load_data(directory)

    budgets = np.array(metadata["n_shadow_samples"], dtype=float)
    energies = data["energy"]  # shape: (n_budgets, n_simulations)
    fci_energy = float(data["fci_energy"])
    hf_energy = float(data["hf_energy"])

    mean_energy = np.mean(energies, axis=1)
    std_energy = np.std(energies, axis=1)
    variance = std_energy ** 2

    C, alpha = fit_prefactor(budgets, variance)

    n_hydrogen = metadata.get("n_hydrogen", "?")
    distance = metadata.get("interatomic_distance_angstrom", "?")

    setup_plotting_style()

    fig, (ax_energy, ax_var) = plt.subplots(
        2, 1, figsize=(3.5, 4.5),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
    )

    # --- Upper panel: mean energy vs budget ---
    ax_energy.axhline(fci_energy, color="k", linewidth=1.0, linestyle="-", label="FCI (exact)")
    ax_energy.axhline(hf_energy, color="k", linewidth=1.0, linestyle="--", label="Hartree-Fock")
    ax_energy.errorbar(
        budgets, mean_energy, yerr=std_energy,
        fmt="o", markersize=3, capsize=3, capthick=1, linewidth=1,
        color="tab:blue", label="Shadow estimate",
    )

    ax_energy.set_xscale("log")
    ax_energy.set_xlabel(r"Shadow samples $N_s$")
    ax_energy.set_ylabel(r"Energy (Ha)")
    ax_energy.legend(frameon=True, fancybox=False, edgecolor="gray")
    ax_energy.set_title(
        rf"$\mathrm{{H}}_{{{n_hydrogen}}}$ / {metadata.get('basis_set', 'sto-3g')}"
        rf" ($d = {distance}$ \AA)"
    )

    # --- Lower panel: variance scaling (log-log) ---
    ax_var.loglog(
        budgets, variance,
        "o", markersize=4, color="tab:blue", label=r"$\mathrm{Var}(\hat{E})$",
    )

    # Fitted C / N^alpha line
    n_fit = np.geomspace(budgets[0], budgets[-1], 200)
    var_fit = C / n_fit ** alpha
    ax_var.loglog(
        n_fit, var_fit,
        "k--", linewidth=1.0,
        label=rf"Fit: $C/N_s^{{\alpha}}$",
    )

    ax_var.annotate(
        rf"$C = {C:.1f}$, $\alpha = {alpha:.2f}$",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    ax_var.set_xlabel(r"Shadow samples $N_s$")
    ax_var.set_ylabel(r"Variance (Ha$^2$)")
    ax_var.legend(frameon=True, fancybox=False, edgecolor="gray", loc="lower left")

    if output_filename is None:
        output_filename = os.path.join(directory, "convergence")

    save_figure(output_filename)
    plt.close(fig)

    print(f"Fitted prefactor: C = {C:.4f}, exponent: alpha = {alpha:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence from shadow budget sweep results directory."
    )
    parser.add_argument("directory", help="Path to results directory containing data.npz and metadata.json")
    parser.add_argument("-o", "--output", default=None, help="Output figure path without extension (default: <directory>/convergence)")
    args = parser.parse_args()

    plot_convergence(args.directory, output_filename=args.output)


if __name__ == "__main__":
    main()

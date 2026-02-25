"""Plot variance scaling with system size from matchgate shadow data.

Loads results from a directory containing data.npz and metadata.json
(as produced by system_size.py) and produces a two-panel figure:
  - Upper: mean energy error vs number of orbitals with std-dev error bars
  - Lower: variance vs number of orbitals on a log-log scale with n^2 reference

Usage:
    python scripts/tomography/matchgate/plot_system_size.py <results_directory>

Example:
    python scripts/tomography/matchgate/plot_system_size.py results/tomography/matchgate/system_size/2026-02-18_12-00-00/
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


def plot_system_size(directory, output_filename=None):
    """Plot energy error and variance scaling against system size."""
    data, metadata = load_data(directory)

    system_sizes = np.array(metadata["system_sizes"])
    n_orbitals = system_sizes  # sto-3g: 1 orbital per H atom
    fci_energies = data["fci_energy"]

    n_shadows = metadata.get("n_shadow_samples", "?")
    distance = metadata.get("interatomic_distance_angstrom", "?")

    mean_energies = np.empty(len(system_sizes))
    std_energies = np.empty(len(system_sizes))
    variances = np.empty(len(system_sizes))

    for j, n_h in enumerate(system_sizes):
        energies = data[f"energy_H{n_h}"]
        mean_energies[j] = np.mean(energies)
        std_energies[j] = np.std(energies)
        variances[j] = std_energies[j] ** 2

    setup_plotting_style()

    fig, (ax_err, ax_var) = plt.subplots(
        2, 1, figsize=(3.5, 4.5),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
    )

    # --- Upper panel: mean error vs system size ---
    errors = mean_energies - fci_energies
    ax_err.errorbar(
        n_orbitals, errors, yerr=std_energies,
        fmt="o", markersize=4, capsize=3, capthick=1, linewidth=1,
        color="tab:blue",
    )
    ax_err.axhline(0, color="k", linewidth=0.8, linestyle="-")
    ax_err.set_xlabel(r"Number of orbitals $n$")
    ax_err.set_ylabel(r"$\langle\hat{E}\rangle - E_{\mathrm{FCI}}$ (Ha)")
    ax_err.set_title(
        rf"$N_s = {n_shadows}$, $d = {distance}$ \AA, sto-3g"
    )

    # --- Lower panel: variance scaling (log-log) ---
    ax_var.loglog(
        n_orbitals, variances,
        "o", markersize=4, color="tab:blue", label=r"$\mathrm{Var}(\hat{E})$",
    )

    # n^2 reference line anchored at the first data point
    ref = variances[0] * (n_orbitals / n_orbitals[0]) ** 2
    ax_var.loglog(
        n_orbitals, ref,
        "k--", linewidth=1.0, label=r"$\propto n^2$",
    )

    ax_var.set_xlabel(r"Number of orbitals $n$")
    ax_var.set_ylabel(r"Variance (Ha$^2$)")
    ax_var.legend(frameon=True, fancybox=False, edgecolor="gray")

    if output_filename is None:
        output_filename = os.path.join(directory, "system_size")

    save_figure(output_filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot system size scaling from matchgate shadow results directory."
    )
    parser.add_argument("directory", help="Path to results directory containing data.npz and metadata.json")
    parser.add_argument("-o", "--output", default=None, help="Output figure path without extension (default: <directory>/system_size)")
    args = parser.parse_args()

    plot_system_size(args.directory, output_filename=args.output)


if __name__ == "__main__":
    main()

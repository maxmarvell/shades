"""Plot potential energy surface from shadow tomography data.

Loads results from a directory containing data.npz and metadata.json
(as produced by matchgate.py) and plots the estimated energy with
variance-based error bars against interatomic distance, compared to
exact FCI and Hartree-Fock reference curves.

Usage:
    python scripts/tomography/plot_pes.py <results_directory>

Example:
    python scripts/tomography/plot_pes.py results/tomography/matchgate/system_size/2026-02-17_12-00-00/
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


def plot_pes(directory, output_filename=None):
    """Plot the potential energy surface with variance error bars."""
    data, metadata = load_data(directory)

    distances = np.array(metadata["interatomic_distances_angstrom"])
    energies = data["energy"]  # shape: (n_distances, n_simulations)
    fci_energy = data["fci_energy"]
    hf_energy = data["hf_energy"]

    mean_energy = np.mean(energies, axis=1)
    std_energy = np.std(energies, axis=1)

    n_samples = metadata.get("n_shadow_samples", "?")
    n_runs = metadata.get("n_runs", energies.shape[1])
    n_hydrogen = metadata.get("n_hydrogen", "?")

    setup_plotting_style()

    fig, (ax_pes, ax_err) = plt.subplots(
        2, 1, figsize=(3.5, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # --- Upper panel: PES ---
    ax_pes.plot(distances, fci_energy, "k-", linewidth=1.2, label="FCI (exact)")
    ax_pes.plot(distances, hf_energy, "k--", linewidth=1.0, label="Hartree-Fock")
    ax_pes.errorbar(
        distances, mean_energy, yerr=std_energy,
        fmt="o", markersize=3, capsize=3, capthick=1, linewidth=1,
        color="tab:blue", label=rf"Shadow ($N_s={n_samples}$)",
    )

    ax_pes.set_ylabel(r"Energy (Ha)")
    ax_pes.legend(frameon=True, fancybox=False, edgecolor="gray")
    ax_pes.set_title(
        rf"$\mathrm{{H}}_{{{n_hydrogen}}}$ / {metadata.get('basis_set', 'sto-3g')}"
    )

    # --- Lower panel: error ---
    error = mean_energy - fci_energy
    ax_err.errorbar(
        distances, error, yerr=std_energy,
        fmt="o", markersize=3, capsize=3, capthick=1, linewidth=1,
        color="tab:blue",
    )
    ax_err.axhline(0, color="k", linewidth=0.8, linestyle="-")
    ax_err.set_xlabel(r"Interatomic distance (\AA)")
    ax_err.set_ylabel(r"$E - E_{\mathrm{FCI}}$ (Ha)")

    if output_filename is None:
        output_filename = os.path.join(directory, "pes")

    save_figure(output_filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PES from shadow tomography results directory."
    )
    parser.add_argument("directory", help="Path to results directory containing data.npz and metadata.json")
    parser.add_argument("-o", "--output", default=None, help="Output figure path without extension (default: <directory>/pes)")
    args = parser.parse_args()

    plot_pes(args.directory, output_filename=args.output)


if __name__ == "__main__":
    main()

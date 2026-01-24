"""Convergence curves with fixed shadow samples.

For each fixed n_shadow_samples value, plot energy error vs n_mc_steps.
This reveals how MC sampling converges when shadow noise is held constant.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotting_config import setup_plotting_style, save_figure


def load_data(data_dir: Path):
    """Load results from npz and metadata from json."""
    data = np.load(data_dir / "data.npz", allow_pickle=True)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    return data, metadata


def plot_convergence_fixed_shadows(data_dir: Path, output_dir: Path = None):
    """Plot energy error convergence for each fixed shadow sample count."""
    data, metadata = load_data(data_dir)

    if output_dir is None:
        output_dir = data_dir

    plot_dir = output_dir / "convergence_fixed_shadows"
    plot_dir.mkdir(exist_ok=True)

    n_mc_iters = np.array(metadata["n_mc_iters"])
    n_shadow_samples = np.array(metadata["n_shadow_samples"])
    E_fci = metadata["E_fci_hartree"]
    n_runs = metadata["n_runs"]

    E_tot = data["E_tot"]  # shape: (n_shadows, n_mc, n_runs)

    setup_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_shadow_samples)))

    # Plot 1: Absolute energy error
    ax1 = axes[0]
    for i, n_shad in enumerate(n_shadow_samples):
        E_mean = E_tot[i].mean(axis=1)  # mean over runs
        E_std = E_tot[i].std(axis=1, ddof=1)
        E_sem = E_std / np.sqrt(n_runs)

        error_mean = E_mean - E_fci

        ax1.errorbar(
            n_mc_iters,
            error_mean,
            yerr=E_sem,
            fmt="o-",
            capsize=3,
            color=colors[i],
            label=rf"$N_\mathrm{{shad}} = {n_shad:,}$",
            linewidth=1.5,
            markersize=5,
        )

    ax1.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel(r"Number of MC Steps ($N_\mathrm{MC}$)")
    ax1.set_ylabel(r"$E_\mathrm{MC} - E_\mathrm{FCI}$ (Ha)")
    ax1.set_xscale("log")
    ax1.set_title("Energy Error vs MC Steps")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: Relative Frobenius error of RDM2
    ax2 = axes[1]
    rel_frob = data["rel_frob_rdm2"]  # shape: (n_shadows, n_mc, n_runs)

    for i, n_shad in enumerate(n_shadow_samples):
        frob_mean = rel_frob[i].mean(axis=1)
        frob_std = rel_frob[i].std(axis=1, ddof=1)
        frob_sem = frob_std / np.sqrt(n_runs)

        ax2.errorbar(
            n_mc_iters,
            frob_mean,
            yerr=frob_sem,
            fmt="o-",
            capsize=3,
            color=colors[i],
            label=rf"$N_\mathrm{{shad}} = {n_shad:,}$",
            linewidth=1.5,
            markersize=5,
        )

    ax2.set_xlabel(r"Number of MC Steps ($N_\mathrm{MC}$)")
    ax2.set_ylabel(r"$\|\Delta\Gamma\|_F / \|\Gamma_\mathrm{ref}\|_F$")
    ax2.set_xscale("log")
    ax2.set_title("RDM2 Relative Frobenius Error vs MC Steps")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    save_figure(plot_dir / "convergence_fixed_shadows.pdf")
    save_figure(plot_dir / "convergence_fixed_shadows.png", dpi=300)
    save_figure(plot_dir / "convergence_fixed_shadows.svg")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence curves with fixed shadow samples"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing data.npz and metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as data_dir)",
    )
    args = parser.parse_args()

    plot_convergence_fixed_shadows(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

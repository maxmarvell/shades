"""Variance decomposition analysis.

Analyzes how the total variance in energy estimates decomposes as a function
of each parameter. This helps understand which source of stochasticity
(MC sampling vs shadow sampling) dominates the error.
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


def plot_variance_decomposition(data_dir: Path, output_dir: Path = None):
    """Plot variance decomposition across MC and shadow parameters."""
    data, metadata = load_data(data_dir)

    if output_dir is None:
        output_dir = data_dir

    plot_dir = output_dir / "variance_decomposition"
    plot_dir.mkdir(exist_ok=True)

    n_mc_iters = np.array(metadata["n_mc_iters"])
    n_shadow_samples = np.array(metadata["n_shadow_samples"])
    E_fci = metadata["E_fci_hartree"]

    E_tot = data["E_tot"]  # shape: (n_shadows, n_mc, n_runs)

    setup_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Variance vs MC steps (averaged over shadow samples)
    ax1 = axes[0, 0]
    var_vs_mc = E_tot.var(axis=2)  # variance over runs: (n_shadows, n_mc)
    var_vs_mc_mean = var_vs_mc.mean(axis=0)  # average over shadow samples
    var_vs_mc_std = var_vs_mc.std(axis=0, ddof=1)

    ax1.errorbar(
        n_mc_iters,
        var_vs_mc_mean,
        yerr=var_vs_mc_std,
        fmt="o-",
        capsize=4,
        color="tab:blue",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel(r"Number of MC Steps ($N_\mathrm{MC}$)")
    ax1.set_ylabel(r"Var$(E)$ (Ha$^2$)")
    ax1.set_xscale("log")
    ax1.set_title(r"Variance vs MC Steps (avg over $N_\mathrm{shad}$)")
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: Variance vs shadow samples (averaged over MC steps)
    ax2 = axes[0, 1]
    var_vs_shad_mean = var_vs_mc.mean(axis=1)  # average over MC steps
    var_vs_shad_std = var_vs_mc.std(axis=1, ddof=1)

    ax2.errorbar(
        n_shadow_samples,
        var_vs_shad_mean,
        yerr=var_vs_shad_std,
        fmt="s-",
        capsize=4,
        color="tab:orange",
        linewidth=2,
        markersize=6,
    )
    ax2.set_xlabel(r"Number of Shadow Samples ($N_\mathrm{shad}$)")
    ax2.set_ylabel(r"Var$(E)$ (Ha$^2$)")
    ax2.set_xscale("log")
    ax2.set_title(r"Variance vs Shadow Samples (avg over $N_\mathrm{MC}$)")
    ax2.grid(True, alpha=0.3, which="both")

    # Plot 3: Variance heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(
        var_vs_mc,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(),
    )
    ax3.set_xticks(range(len(n_mc_iters)))
    ax3.set_xticklabels([f"{n//1000}k" for n in n_mc_iters])
    ax3.set_yticks(range(len(n_shadow_samples)))
    ax3.set_yticklabels([f"{n//1000}k" for n in n_shadow_samples])
    ax3.set_xlabel(r"$N_\mathrm{MC}$")
    ax3.set_ylabel(r"$N_\mathrm{shad}$")
    ax3.set_title(r"Variance Heatmap (Ha$^2$)")
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label(r"Var$(E)$")

    # Plot 4: Bias^2 vs Variance decomposition
    ax4 = axes[1, 1]

    # Compute bias (systematic error) and variance for each configuration
    E_mean = E_tot.mean(axis=2)  # (n_shadows, n_mc)
    bias_sq = (E_mean - E_fci) ** 2
    variance = E_tot.var(axis=2)
    mse = bias_sq + variance

    # Flatten for scatter plot
    bias_sq_flat = bias_sq.flatten()
    variance_flat = variance.flatten()
    mse_flat = mse.flatten()

    # Color by total cost (n_mc * n_shadow)
    costs = np.outer(n_shadow_samples, n_mc_iters).flatten()

    sc = ax4.scatter(
        variance_flat,
        bias_sq_flat,
        c=np.log10(costs),
        cmap="plasma",
        s=60,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
    )
    cbar4 = plt.colorbar(sc, ax=ax4)
    cbar4.set_label(r"$\log_{10}(N_\mathrm{MC} \times N_\mathrm{shad})$")

    # Add diagonal lines for constant MSE
    max_val = max(variance_flat.max(), bias_sq_flat.max())
    for mse_level in [1e-4, 1e-3, 1e-2]:
        x = np.linspace(0, mse_level, 100)
        y = mse_level - x
        ax4.plot(x, y, "--", color="gray", alpha=0.5, linewidth=1)
        ax4.text(
            mse_level * 0.7,
            mse_level * 0.3,
            f"MSE={mse_level:.0e}",
            fontsize=6,
            color="gray",
        )

    ax4.set_xlabel(r"Variance (Ha$^2$)")
    ax4.set_ylabel(r"Bias$^2$ (Ha$^2$)")
    ax4.set_title("Bias-Variance Tradeoff")
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    save_figure(plot_dir / "variance_decomposition.pdf")
    save_figure(plot_dir / "variance_decomposition.png", dpi=300)
    save_figure(plot_dir / "variance_decomposition.svg")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Variance Decomposition Summary")
    print("=" * 60)
    print(f"\nVariance range: {variance_flat.min():.2e} - {variance_flat.max():.2e} Ha^2")
    print(f"Bias^2 range:   {bias_sq_flat.min():.2e} - {bias_sq_flat.max():.2e} Ha^2")
    print(f"MSE range:      {mse_flat.min():.2e} - {mse_flat.max():.2e} Ha^2")

    # Identify which parameter dominates
    var_ratio = variance_flat / mse_flat
    print(f"\nVariance/MSE ratio: {var_ratio.mean():.2%} (avg)")
    print(f"Bias^2/MSE ratio:   {(1 - var_ratio).mean():.2%} (avg)")


def main():
    parser = argparse.ArgumentParser(description="Plot variance decomposition analysis")
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

    plot_variance_decomposition(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

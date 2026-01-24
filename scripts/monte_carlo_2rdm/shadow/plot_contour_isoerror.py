"""Contour plot with iso-error lines.

Creates a contour plot showing energy error as a function of both parameters,
with contour lines indicating combinations that achieve the same accuracy.
Useful for budget allocation decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotting_config import setup_plotting_style, save_figure


def load_data(data_dir: Path):
    """Load results from npz and metadata from json."""
    data = np.load(data_dir / "data.npz", allow_pickle=True)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    return data, metadata


def plot_contour_isoerror(data_dir: Path, output_dir: Path = None):
    """Plot contour map with iso-error lines."""
    data, metadata = load_data(data_dir)

    if output_dir is None:
        output_dir = data_dir

    plot_dir = output_dir / "contour_isoerror"
    plot_dir.mkdir(exist_ok=True)

    n_mc_iters = np.array(metadata["n_mc_iters"])
    n_shadow_samples = np.array(metadata["n_shadow_samples"])
    E_fci = metadata["E_fci_hartree"]

    E_tot = data["E_tot"]  # shape: (n_shadows, n_mc, n_runs)
    rel_frob = data["rel_frob_rdm2"]

    # Compute mean error for each configuration
    E_mean = E_tot.mean(axis=2)  # (n_shadows, n_mc)
    energy_error = np.abs(E_mean - E_fci)
    frob_error = rel_frob.mean(axis=2)

    setup_plotting_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Create meshgrid for plotting (in log space for better visualization)
    log_mc = np.log10(n_mc_iters)
    log_shad = np.log10(n_shadow_samples)

    # Interpolate to finer grid for smoother contours
    interp_energy = RegularGridInterpolator(
        (log_shad, log_mc), np.log10(energy_error), method="linear"
    )
    interp_frob = RegularGridInterpolator(
        (log_shad, log_mc), np.log10(frob_error), method="linear"
    )

    # Fine grid
    log_mc_fine = np.linspace(log_mc.min(), log_mc.max(), 100)
    log_shad_fine = np.linspace(log_shad.min(), log_shad.max(), 100)
    LOG_MC_FINE, LOG_SHAD_FINE = np.meshgrid(log_mc_fine, log_shad_fine)

    # Interpolate
    points = np.stack([LOG_SHAD_FINE.ravel(), LOG_MC_FINE.ravel()], axis=-1)
    energy_error_fine = 10 ** interp_energy(points).reshape(LOG_MC_FINE.shape)
    frob_error_fine = 10 ** interp_frob(points).reshape(LOG_MC_FINE.shape)

    # ===== Plot 1: Energy error contours =====
    ax1 = axes[0]

    # Filled contours
    levels_energy = np.logspace(
        np.floor(np.log10(energy_error.min())),
        np.ceil(np.log10(energy_error.max())),
        15,
    )
    cf1 = ax1.contourf(
        10**LOG_MC_FINE,
        10**LOG_SHAD_FINE,
        energy_error_fine,
        levels=levels_energy,
        cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(),
    )

    # Contour lines
    cs1 = ax1.contour(
        10**LOG_MC_FINE,
        10**LOG_SHAD_FINE,
        energy_error_fine,
        levels=[1e-3, 5e-3, 1e-2, 2e-2, 5e-2],
        colors="white",
        linewidths=1.5,
    )
    ax1.clabel(cs1, inline=True, fontsize=7, fmt="%.0e")

    # Mark actual data points
    MC_GRID, SHAD_GRID = np.meshgrid(n_mc_iters, n_shadow_samples)
    ax1.scatter(
        MC_GRID.ravel(),
        SHAD_GRID.ravel(),
        c="red",
        s=30,
        marker="x",
        linewidths=1.5,
        zorder=10,
    )

    ax1.set_xlabel(r"$N_\mathrm{MC}$")
    ax1.set_ylabel(r"$N_\mathrm{shad}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Energy Error (Ha)")
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label(r"$|E - E_\mathrm{FCI}|$")

    # ===== Plot 2: RDM2 Frobenius error contours =====
    ax2 = axes[1]

    levels_frob = np.logspace(
        np.floor(np.log10(frob_error.min())),
        np.ceil(np.log10(frob_error.max())),
        15,
    )
    cf2 = ax2.contourf(
        10**LOG_MC_FINE,
        10**LOG_SHAD_FINE,
        frob_error_fine,
        levels=levels_frob,
        cmap="plasma",
        norm=plt.matplotlib.colors.LogNorm(),
    )

    cs2 = ax2.contour(
        10**LOG_MC_FINE,
        10**LOG_SHAD_FINE,
        frob_error_fine,
        levels=[0.1, 0.2, 0.3, 0.5],
        colors="white",
        linewidths=1.5,
    )
    ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.2f")

    ax2.scatter(
        MC_GRID.ravel(),
        SHAD_GRID.ravel(),
        c="cyan",
        s=30,
        marker="x",
        linewidths=1.5,
        zorder=10,
    )

    ax2.set_xlabel(r"$N_\mathrm{MC}$")
    ax2.set_ylabel(r"$N_\mathrm{shad}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(r"RDM2 Relative Frobenius Error")
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label(r"$\|\Delta\Gamma\|_F / \|\Gamma_\mathrm{ref}\|_F$")

    # ===== Plot 3: Cost-efficiency contours =====
    ax3 = axes[2]

    # Total cost = N_MC * N_shad
    COST_FINE = 10**LOG_MC_FINE * 10**LOG_SHAD_FINE

    # Efficiency = 1 / (error * cost) -- higher is better
    efficiency = 1.0 / (energy_error_fine * COST_FINE)

    cf3 = ax3.contourf(
        10**LOG_MC_FINE,
        10**LOG_SHAD_FINE,
        efficiency,
        levels=20,
        cmap="RdYlGn",
        norm=plt.matplotlib.colors.LogNorm(),
    )

    # Add iso-cost lines
    cost_levels = [1e7, 5e7, 1e8, 5e8, 1e9]
    for cost in cost_levels:
        mc_line = np.logspace(log_mc.min(), log_mc.max(), 100)
        shad_line = cost / mc_line
        valid = (shad_line >= 10 ** log_shad.min()) & (shad_line <= 10 ** log_shad.max())
        if valid.any():
            ax3.plot(
                mc_line[valid],
                shad_line[valid],
                "--",
                color="black",
                alpha=0.5,
                linewidth=1,
            )
            # Label at midpoint
            mid_idx = len(mc_line[valid]) // 2
            ax3.text(
                mc_line[valid][mid_idx],
                shad_line[valid][mid_idx] * 1.1,
                f"{cost:.0e}",
                fontsize=6,
                ha="center",
            )

    ax3.scatter(
        MC_GRID.ravel(),
        SHAD_GRID.ravel(),
        c="blue",
        s=30,
        marker="x",
        linewidths=1.5,
        zorder=10,
    )

    ax3.set_xlabel(r"$N_\mathrm{MC}$")
    ax3.set_ylabel(r"$N_\mathrm{shad}$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title("Cost Efficiency (dashed = iso-cost)")
    cbar3 = plt.colorbar(cf3, ax=ax3)
    cbar3.set_label(r"$1 / (\mathrm{error} \times \mathrm{cost})$")

    plt.tight_layout()

    save_figure(plot_dir / "contour_isoerror.pdf")
    save_figure(plot_dir / "contour_isoerror.png", dpi=300)
    save_figure(plot_dir / "contour_isoerror.svg")

    plt.show()

    # Print optimal configurations
    print("\n" + "=" * 60)
    print("Optimal Configurations (Pareto Front)")
    print("=" * 60)

    # Find Pareto-optimal points (minimize both error and cost)
    costs = np.outer(n_shadow_samples, n_mc_iters).flatten()
    errors = energy_error.flatten()

    # Sort by cost
    sorted_idx = np.argsort(costs)
    pareto_idx = []
    min_error = np.inf

    for idx in sorted_idx:
        if errors[idx] < min_error:
            pareto_idx.append(idx)
            min_error = errors[idx]

    print(f"\n{'N_shad':>10} {'N_MC':>10} {'Cost':>12} {'Error (Ha)':>12}")
    print("-" * 50)
    for idx in pareto_idx:
        i_shad = idx // len(n_mc_iters)
        i_mc = idx % len(n_mc_iters)
        print(
            f"{n_shadow_samples[i_shad]:>10,} {n_mc_iters[i_mc]:>10,} "
            f"{costs[idx]:>12,.0f} {errors[idx]:>12.2e}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot contour map with iso-error lines")
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

    plot_contour_isoerror(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

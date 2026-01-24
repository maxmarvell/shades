"""Log-log scaling analysis.

Plot log(error) vs log(n_samples) for both parameters to determine
the convergence rate. A slope of -0.5 indicates 1/sqrt(N) scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotting_config import setup_plotting_style, save_figure


def load_data(data_dir: Path):
    """Load results from npz and metadata from json."""
    data = np.load(data_dir / "data.npz", allow_pickle=True)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    return data, metadata


def fit_power_law(x, y):
    """Fit y = a * x^b in log space, return (a, b, r_squared)."""
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
    return 10**intercept, slope, r_value**2


def plot_scaling_loglog(data_dir: Path, output_dir: Path = None):
    """Plot log-log scaling to determine convergence rates."""
    data, metadata = load_data(data_dir)

    if output_dir is None:
        output_dir = data_dir

    plot_dir = output_dir / "scaling_loglog"
    plot_dir.mkdir(exist_ok=True)

    n_mc_iters = np.array(metadata["n_mc_iters"])
    n_shadow_samples = np.array(metadata["n_shadow_samples"])
    E_fci = metadata["E_fci_hartree"]
    n_runs = metadata["n_runs"]

    E_tot = data["E_tot"]  # shape: (n_shadows, n_mc, n_runs)
    rel_frob = data["rel_frob_rdm2"]

    setup_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ===== Row 1: Energy error scaling =====

    # Plot 1: Error vs MC steps (at highest shadow count)
    ax1 = axes[0, 0]
    idx_max_shad = -1  # highest shadow samples
    E_slice = E_tot[idx_max_shad]  # (n_mc, n_runs)
    E_mean = E_slice.mean(axis=1)
    E_std = E_slice.std(axis=1, ddof=1)
    error_mean = np.abs(E_mean - E_fci)

    ax1.errorbar(
        n_mc_iters,
        error_mean,
        yerr=E_std / np.sqrt(n_runs),
        fmt="o",
        capsize=4,
        color="tab:blue",
        markersize=8,
        label="Data",
    )

    # Fit power law
    a, b, r2 = fit_power_law(n_mc_iters, error_mean)
    x_fit = np.logspace(np.log10(n_mc_iters.min()), np.log10(n_mc_iters.max()), 100)
    ax1.plot(
        x_fit,
        a * x_fit**b,
        "--",
        color="tab:red",
        linewidth=2,
        label=rf"Fit: $\propto N_\mathrm{{MC}}^{{{b:.2f}}}$ ($R^2={r2:.3f}$)",
    )

    # Reference 1/sqrt(N) scaling
    ref_scale = error_mean[0] * np.sqrt(n_mc_iters[0] / x_fit)
    ax1.plot(x_fit, ref_scale, ":", color="gray", alpha=0.7, label=r"$1/\sqrt{N}$ reference")

    ax1.set_xlabel(r"$N_\mathrm{MC}$")
    ax1.set_ylabel(r"$|E - E_\mathrm{FCI}|$ (Ha)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(rf"Energy Error Scaling ($N_\mathrm{{shad}}={n_shadow_samples[idx_max_shad]:,}$)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: Error vs shadow samples (at highest MC count)
    ax2 = axes[0, 1]
    idx_max_mc = -1  # highest MC steps
    E_slice = E_tot[:, idx_max_mc, :]  # (n_shadows, n_runs)
    E_mean = E_slice.mean(axis=1)
    E_std = E_slice.std(axis=1, ddof=1)
    error_mean = np.abs(E_mean - E_fci)

    ax2.errorbar(
        n_shadow_samples,
        error_mean,
        yerr=E_std / np.sqrt(n_runs),
        fmt="s",
        capsize=4,
        color="tab:orange",
        markersize=8,
        label="Data",
    )

    # Fit power law
    a, b, r2 = fit_power_law(n_shadow_samples, error_mean)
    x_fit = np.logspace(
        np.log10(n_shadow_samples.min()), np.log10(n_shadow_samples.max()), 100
    )
    ax2.plot(
        x_fit,
        a * x_fit**b,
        "--",
        color="tab:red",
        linewidth=2,
        label=rf"Fit: $\propto N_\mathrm{{shad}}^{{{b:.2f}}}$ ($R^2={r2:.3f}$)",
    )

    # Reference 1/sqrt(N) scaling
    ref_scale = error_mean[0] * np.sqrt(n_shadow_samples[0] / x_fit)
    ax2.plot(x_fit, ref_scale, ":", color="gray", alpha=0.7, label=r"$1/\sqrt{N}$ reference")

    ax2.set_xlabel(r"$N_\mathrm{shad}$")
    ax2.set_ylabel(r"$|E - E_\mathrm{FCI}|$ (Ha)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(rf"Energy Error Scaling ($N_\mathrm{{MC}}={n_mc_iters[idx_max_mc]:,}$)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, which="both")

    # ===== Row 2: RDM2 Frobenius error scaling =====

    # Plot 3: Frobenius error vs MC steps
    ax3 = axes[1, 0]
    frob_slice = rel_frob[idx_max_shad]  # (n_mc, n_runs)
    frob_mean = frob_slice.mean(axis=1)
    frob_std = frob_slice.std(axis=1, ddof=1)

    ax3.errorbar(
        n_mc_iters,
        frob_mean,
        yerr=frob_std / np.sqrt(n_runs),
        fmt="o",
        capsize=4,
        color="tab:blue",
        markersize=8,
        label="Data",
    )

    a, b, r2 = fit_power_law(n_mc_iters, frob_mean)
    x_fit = np.logspace(np.log10(n_mc_iters.min()), np.log10(n_mc_iters.max()), 100)
    ax3.plot(
        x_fit,
        a * x_fit**b,
        "--",
        color="tab:red",
        linewidth=2,
        label=rf"Fit: $\propto N_\mathrm{{MC}}^{{{b:.2f}}}$ ($R^2={r2:.3f}$)",
    )

    ref_scale = frob_mean[0] * np.sqrt(n_mc_iters[0] / x_fit)
    ax3.plot(x_fit, ref_scale, ":", color="gray", alpha=0.7, label=r"$1/\sqrt{N}$ reference")

    ax3.set_xlabel(r"$N_\mathrm{MC}$")
    ax3.set_ylabel(r"$\|\Delta\Gamma\|_F / \|\Gamma_\mathrm{ref}\|_F$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title(rf"RDM2 Error Scaling ($N_\mathrm{{shad}}={n_shadow_samples[idx_max_shad]:,}$)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, which="both")

    # Plot 4: Frobenius error vs shadow samples
    ax4 = axes[1, 1]
    frob_slice = rel_frob[:, idx_max_mc, :]  # (n_shadows, n_runs)
    frob_mean = frob_slice.mean(axis=1)
    frob_std = frob_slice.std(axis=1, ddof=1)

    ax4.errorbar(
        n_shadow_samples,
        frob_mean,
        yerr=frob_std / np.sqrt(n_runs),
        fmt="s",
        capsize=4,
        color="tab:orange",
        markersize=8,
        label="Data",
    )

    a, b, r2 = fit_power_law(n_shadow_samples, frob_mean)
    x_fit = np.logspace(
        np.log10(n_shadow_samples.min()), np.log10(n_shadow_samples.max()), 100
    )
    ax4.plot(
        x_fit,
        a * x_fit**b,
        "--",
        color="tab:red",
        linewidth=2,
        label=rf"Fit: $\propto N_\mathrm{{shad}}^{{{b:.2f}}}$ ($R^2={r2:.3f}$)",
    )

    ref_scale = frob_mean[0] * np.sqrt(n_shadow_samples[0] / x_fit)
    ax4.plot(x_fit, ref_scale, ":", color="gray", alpha=0.7, label=r"$1/\sqrt{N}$ reference")

    ax4.set_xlabel(r"$N_\mathrm{shad}$")
    ax4.set_ylabel(r"$\|\Delta\Gamma\|_F / \|\Gamma_\mathrm{ref}\|_F$")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_title(rf"RDM2 Error Scaling ($N_\mathrm{{MC}}={n_mc_iters[idx_max_mc]:,}$)")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    save_figure(plot_dir / "scaling_loglog.pdf")
    save_figure(plot_dir / "scaling_loglog.png", dpi=300)
    save_figure(plot_dir / "scaling_loglog.svg")

    plt.show()

    # Print scaling summary
    print("\n" + "=" * 60)
    print("Scaling Analysis Summary")
    print("=" * 60)
    print("\nExpected: slope = -0.5 for 1/sqrt(N) convergence")
    print("(More negative = faster convergence)")


def main():
    parser = argparse.ArgumentParser(description="Plot log-log scaling analysis")
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

    plot_scaling_loglog(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""2-RDM validity metrics.

Checks N-representability conditions on the estimated 2-RDM:
1. Trace condition: Tr(Γ) = N(N-1)
2. Positive semidefiniteness: min eigenvalue >= 0
3. Antisymmetry: Γ(pq|rs) = -Γ(qp|rs)
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


def compute_trace(rdm2: np.ndarray) -> float:
    """Compute trace of 2-RDM: Tr(Γ) = Σ_pq Γ(pq|pq)."""
    return np.einsum("pqpq->", rdm2)


def compute_min_eigenvalue(rdm2: np.ndarray) -> float:
    """Compute minimum eigenvalue of 2-RDM as a pair-index matrix."""
    norb = rdm2.shape[0]
    # Reshape to (norb^2, norb^2) matrix
    rdm2_matrix = rdm2.reshape(norb * norb, norb * norb)
    # Symmetrize for numerical stability
    rdm2_matrix = 0.5 * (rdm2_matrix + rdm2_matrix.T)
    eigvals = np.linalg.eigvalsh(rdm2_matrix)
    return eigvals.min()


def compute_antisymmetry_violation(rdm2: np.ndarray) -> float:
    """Compute relative antisymmetry violation: ||Γ - antisym(Γ)||_F / ||Γ||_F."""
    # Antisymmetrize: Γ_antisym(pq|rs) = 0.5 * (Γ(pq|rs) - Γ(qp|rs))
    rdm2_antisym = 0.5 * (rdm2 - rdm2.transpose(1, 0, 2, 3))
    violation = np.linalg.norm(rdm2 - rdm2_antisym)
    norm = np.linalg.norm(rdm2)
    return violation / norm if norm > 0 else 0.0


def plot_rdm2_validity(data_dir: Path, output_dir: Path = None):
    """Plot 2-RDM validity metrics."""
    data, metadata = load_data(data_dir)

    if output_dir is None:
        output_dir = data_dir

    plot_dir = output_dir / "rdm2_validity"
    plot_dir.mkdir(exist_ok=True)

    n_mc_iters = np.array(metadata["n_mc_iters"])
    n_shadow_samples = np.array(metadata["n_shadow_samples"])
    n_runs = metadata["n_runs"]

    # H4 chain has 4 electrons
    n_elec = 4
    expected_trace = n_elec * (n_elec - 1)

    rdm2_all = data["rdm2"]  # shape: (n_shadows, n_mc, n_runs, norb, norb, norb, norb)

    # Compute metrics for each configuration
    n_shad, n_mc, n_run = rdm2_all.shape[:3]

    trace_err = np.zeros((n_shad, n_mc, n_run))
    min_eig = np.zeros((n_shad, n_mc, n_run))
    antisym_viol = np.zeros((n_shad, n_mc, n_run))

    for i in range(n_shad):
        for j in range(n_mc):
            for k in range(n_run):
                rdm2 = rdm2_all[i, j, k]
                trace_err[i, j, k] = (compute_trace(rdm2) - expected_trace) / expected_trace
                min_eig[i, j, k] = compute_min_eigenvalue(rdm2)
                antisym_viol[i, j, k] = compute_antisymmetry_violation(rdm2)

    setup_plotting_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_shadow_samples)))

    # Plot 1: Trace error
    ax1 = axes[0]
    for i, n_shad_val in enumerate(n_shadow_samples):
        trace_mean = trace_err[i].mean(axis=1)
        trace_std = trace_err[i].std(axis=1, ddof=1)
        trace_sem = trace_std / np.sqrt(n_runs)

        ax1.errorbar(
            n_mc_iters,
            trace_mean,
            yerr=trace_sem,
            fmt="o-",
            capsize=3,
            color=colors[i],
            label=rf"$N_\mathrm{{shad}} = {n_shad_val:,}$",
            linewidth=1.5,
            markersize=5,
        )

    ax1.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel(r"$N_\mathrm{MC}$")
    ax1.set_ylabel(r"$(\mathrm{Tr}(\Gamma) - N(N-1)) / N(N-1)$")
    ax1.set_xscale("log")
    ax1.set_title("Trace Error")
    ax1.legend(fontsize=6, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: Minimum eigenvalue
    ax2 = axes[1]
    for i, n_shad_val in enumerate(n_shadow_samples):
        eig_mean = min_eig[i].mean(axis=1)
        eig_std = min_eig[i].std(axis=1, ddof=1)
        eig_sem = eig_std / np.sqrt(n_runs)

        ax2.errorbar(
            n_mc_iters,
            eig_mean,
            yerr=eig_sem,
            fmt="o-",
            capsize=3,
            color=colors[i],
            label=rf"$N_\mathrm{{shad}} = {n_shad_val:,}$",
            linewidth=1.5,
            markersize=5,
        )

    ax2.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel(r"$N_\mathrm{MC}$")
    ax2.set_ylabel(r"$\lambda_\mathrm{min}(\Gamma)$")
    ax2.set_xscale("log")
    ax2.set_title("Minimum Eigenvalue (P-condition)")
    ax2.legend(fontsize=6, loc="best")
    ax2.grid(True, alpha=0.3, which="both")

    # Plot 3: Antisymmetry violation
    ax3 = axes[2]
    for i, n_shad_val in enumerate(n_shadow_samples):
        antisym_mean = antisym_viol[i].mean(axis=1)
        antisym_std = antisym_viol[i].std(axis=1, ddof=1)
        antisym_sem = antisym_std / np.sqrt(n_runs)

        ax3.errorbar(
            n_mc_iters,
            antisym_mean,
            yerr=antisym_sem,
            fmt="o-",
            capsize=3,
            color=colors[i],
            label=rf"$N_\mathrm{{shad}} = {n_shad_val:,}$",
            linewidth=1.5,
            markersize=5,
        )

    ax3.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel(r"$N_\mathrm{MC}$")
    ax3.set_ylabel(r"$\|\Gamma - \Gamma_\mathrm{antisym}\|_F / \|\Gamma\|_F$")
    ax3.set_xscale("log")
    ax3.set_title("Antisymmetry Violation")
    ax3.legend(fontsize=6, loc="best")
    ax3.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    save_figure(plot_dir / "rdm2_validity.pdf")
    save_figure(plot_dir / "rdm2_validity.png", dpi=300)
    save_figure(plot_dir / "rdm2_validity.svg")

    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("2-RDM Validity Summary")
    print("=" * 60)
    print(f"\nExpected trace: {expected_trace} (N={n_elec})")
    print(f"Trace error range: {trace_err.min():.2e} to {trace_err.max():.2e}")
    print(f"Min eigenvalue range: {min_eig.min():.4f} to {min_eig.max():.4f}")
    print(f"Antisymmetry violation range: {antisym_viol.min():.2e} to {antisym_viol.max():.2e}")


def main():
    parser = argparse.ArgumentParser(description="Plot 2-RDM validity metrics")
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

    plot_rdm2_validity(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

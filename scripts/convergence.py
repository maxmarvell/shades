"""Shadow Tomography Convergence Analysis.

This script demonstrates the convergence of shadow state tomography estimates
to the exact FCI ground state energy as the number of shadow samples increases.

The analysis systematically varies the number of measurement samples (shots)
and compares the shadow-estimated energy to the exact FCI result, computing
both absolute and relative errors. This validates that shadow tomography
provides statistically sound estimates that improve with more measurements.

Key convergence properties tested:
- Energy estimates approach exact FCI value as N_samples → ∞
- Error decreases approximately as 1/√N (statistical sampling bound)
- Median-of-means provides robust estimation against outliers
- Multiple simulation runs reduce statistical noise and provide error bars
"""

from pyscf import gto, scf
from shades.estimators import ShadowEstimator
from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting_config import setup_plotting_style, save_figure
import os
from utils import get_output_directory

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Output configuration
DEFAULT_OUTPUT_DIR = "./results/convergence"


# Simulation parameters
DEFAULT_N_RUNS = 10  # Number of simulation runs per shot count (for statistics)
N_K_ESTIMATORS = 20  # Number of k-estimators for median-of-means
N_JOBS = 8  # Number of parallel jobs for estimation
USE_QULACS = True  # Use Qulacs backend for faster simulation

# Molecular system parameters
N_HYDROGEN_ATOMS = 8  # Number of hydrogen atoms in chain
BOND_LENGTH = 3.5  # Bond length in Angstroms
BASIS_SET = "sto-3g"  # Basis set for quantum chemistry

# Shadow sampling schedule (number of shots to test)
SHOT_SCHEDULE = np.array([500, 1000, 2000, 4000, 8000, 16000], dtype=int)

# Plotting parameters
FIGURE_SIZE = (10, 4)
PLOT_DPI = 300


def create_result_folder_name(n_runs):
    # Format: H4_d0.50_sto-3g_nruns5
    folder_name = f"H{N_HYDROGEN_ATOMS}_d{BOND_LENGTH:.2f}_{BASIS_SET}_nruns{n_runs}"
    return folder_name


def create_output_filename(base_name, extension):
    return f"{base_name}{extension}"


def main():
    """Run convergence analysis and generate plots."""
    print("=" * 70)
    print("Shadow CI Convergence Analysis")
    print("=" * 70)

    # Prompt for number of simulation runs
    print("\nSimulation Parameters")
    print("-" * 70)
    n_runs_input = input(
        f"Enter number of simulation runs per shot count (default: {DEFAULT_N_RUNS}): "
    ).strip()
    n_runs = int(n_runs_input) if n_runs_input else DEFAULT_N_RUNS
    print(f"Will run {n_runs} simulations per shot count for statistical analysis")

    # Get base output directory from user
    base_output_dir = get_output_directory(DEFAULT_OUTPUT_DIR)

    result_folder_name = create_result_folder_name(n_runs)
    output_dir = os.path.join(base_output_dir, result_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    mol = gto.Mole()
    atom = make_hydrogen_chain(N_HYDROGEN_ATOMS, BOND_LENGTH)
    mol.build(atom=atom, basis=BASIS_SET)
    print(f"\nMolecule: H{N_HYDROGEN_ATOMS} chain (bond length = {BOND_LENGTH:.2f} Å)")
    print(f"Basis set: {BASIS_SET}")

    mf = scf.UHF(mol)
    mf.run()

    shots = SHOT_SCHEDULE
    all_energies = np.empty((len(shots), n_runs), dtype=float)

    print("\n" + "=" * 70)
    print("Running Shadow Tomography Convergence Study")
    print("=" * 70)
    print(f"Shot schedule: {shots.tolist()}")
    print(f"K-estimators: {N_K_ESTIMATORS}, Jobs: {N_JOBS}, QULACS: {USE_QULACS}")

    fci_solver = FCISolver(mf)
    estimator = ShadowEstimator(mf, solver=fci_solver, verbose=1)
    E_exact = estimator.E_exact
    E_hf = estimator.E_hf

    print("\n" + "=" * 70)
    print("Reference Energies")
    print("=" * 70)
    print(f"Hartree-Fock Energy:   {E_hf:.10f} Ha")
    print(f"Exact FCI Energy:      {E_exact:.10f} Ha")
    print(f"Correlation Energy:    {E_exact - E_hf:.10f} Ha")
    print("=" * 70)

    for i, n_shots in enumerate(shots):
        print(f"\n[{i + 1}/{len(shots)}] Running with {n_shots:,} shadow samples...")

        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ", flush=True)
            E_approx, _, _, _ = estimator.run(
                n_samples=n_shots,
                n_k_estimators=N_K_ESTIMATORS,
                n_jobs=N_JOBS
            )
            all_energies[i, run] = np.real(E_approx)
            print(f"E = {np.real(E_approx):.8f} Ha")

            estimator.clear_sample()

    mean_energies = np.mean(all_energies, axis=1)
    std_energies = np.std(all_energies, axis=1, ddof=1)
    sem_energies = std_energies / np.sqrt(n_runs)

    abs_errors_all_runs = np.abs(all_energies - E_exact)
    mean_abs_errors = np.mean(abs_errors_all_runs, axis=1)
    std_abs_errors = np.std(abs_errors_all_runs, axis=1, ddof=1)
    sem_abs_errors = std_abs_errors / np.sqrt(n_runs)

    results_df = pd.DataFrame(
        {
            "n_samples": shots,
            "mean_energy": mean_energies,
            "std_energy": std_energies,
            "sem_energy": sem_energies,
            "mean_abs_error": mean_abs_errors,
            "std_abs_error": std_abs_errors,
            "sem_abs_error": sem_abs_errors,
            "rel_error_percent": np.abs((mean_energies - E_exact) / (E_exact - E_hf)) * 100,
        }
    )

    print("\n" + "=" * 70)
    print("Convergence Summary")
    print("=" * 70)
    print(
        f"{'Samples':>12} {'Mean Energy (Ha)':>18} {'SEM':>12} {'Mean |ΔE|':>15} {'SEM |ΔE|':>12} {'Rel Error (%)':>15}"
    )
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(
            f"{row['n_samples']:>12.0f} {row['mean_energy']:>18.8f} {row['sem_energy']:>12.2e} "
            f"{row['mean_abs_error']:>15.2e} {row['sem_abs_error']:>12.2e} {row['rel_error_percent']:>15.4f}"
        )
    print("=" * 70)

    # === Plotting Section ===
    print("\n" + "=" * 70)
    print("Generating Convergence Plots")
    print("=" * 70)

    setup_plotting_style()
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # Plot mean energies with error bars (standard error of mean)
    ax1.errorbar(
        results_df["n_samples"],
        results_df["mean_energy"],
        yerr=results_df["sem_energy"],
        fmt="o-",
        label="Shadow Estimate (mean ± SEM)",
        linewidth=2,
        markersize=8,
        color="C1",
        capsize=5,
        capthick=2,
    )
    ax1.axhline(E_exact, linestyle="--", label="Exact FCI", linewidth=2, color="C0", alpha=0.8)
    ax1.set_xlabel(r"Number of Shadow Samples")
    ax1.set_ylabel(r"Ground State Energy (Ha)")
    ax1.set_xscale("log")
    ax1.set_title("Energy Convergence")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot mean absolute errors with proper uncertainty
    ax2.errorbar(
        results_df["n_samples"],
        results_df["mean_abs_error"],
        yerr=results_df["sem_abs_error"],
        fmt="o-",
        label="Mean |Error| ± SEM",
        linewidth=2,
        markersize=8,
        color="C2",
        capsize=5,
        capthick=2,
    )

    reference_scaling = results_df["mean_abs_error"].iloc[0] * np.sqrt(
        results_df["n_samples"].iloc[0] / results_df["n_samples"]
    )
    ax2.loglog(
        results_df["n_samples"],
        reference_scaling,
        "--",
        label=r"$1/\sqrt{N}$ scaling",
        linewidth=2,
        color="gray",
        alpha=0.6,
    )

    ax2.set_xlabel(r"Number of Shadow Samples")
    ax2.set_ylabel(r"Absolute Error (Ha)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Error Scaling")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    # Save figures to user-specified directory
    pdf_filename = create_output_filename("convergence_analysis", ".pdf")
    png_filename = create_output_filename("convergence_analysis", ".png")
    pdf_path = os.path.join(output_dir, pdf_filename)
    png_path = os.path.join(output_dir, png_filename)
    save_figure(pdf_path)
    save_figure(png_path, dpi=PLOT_DPI)
    print(f"Plots saved: {pdf_filename}, {png_filename}")

    plt.show()

    print("\nSaving numerical results...")

    # Create metadata dictionary
    metadata = {
        "system": f"H{N_HYDROGEN_ATOMS} chain",
        "bond_length_angstrom": BOND_LENGTH,
        "basis_set": BASIS_SET,
        "n_runs": n_runs,
        "k_estimators": N_K_ESTIMATORS,
        "n_jobs": N_JOBS,
        "use_qulacs": USE_QULACS,
        "E_hf_hartree": E_hf,
        "E_fci_hartree": E_exact,
        "E_corr_hartree": E_exact - E_hf,
    }

    # Save metadata as JSON
    import json

    metadata_filename = create_output_filename("metadata", ".json")
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_filename}")

    # Save summary statistics as CSV using pandas
    summary_filename = create_output_filename("convergence_summary", ".csv")
    summary_path = os.path.join(output_dir, summary_filename)
    results_df.to_csv(summary_path, index=False, float_format="%.10e")
    print(f"Summary statistics saved to {summary_filename}")

    # Save all individual run data as CSV using pandas
    all_runs_filename = create_output_filename("convergence_all_runs", ".csv")
    all_runs_path = os.path.join(output_dir, all_runs_filename)

    # Create DataFrame with all runs
    all_runs_df = pd.DataFrame(all_energies, columns=[f"run_{i + 1}" for i in range(n_runs)])
    all_runs_df.insert(0, "n_samples", shots)
    all_runs_df.to_csv(all_runs_path, index=False, float_format="%.10f")
    print(f"All run data saved to {all_runs_filename}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

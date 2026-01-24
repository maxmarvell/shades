"""Monte Carlo RDM2 Convergence Analysis.

This script demonstrates the convergence of Monte Carlo RDM2 estimation
as the number of MC iterations increases. The MC estimator computes
a correlation-focused 2-RDM.

Two convergence metrics are tracked:
1. RDM2 Frobenius norm distance to a high-iteration reference
2. Representative matrix element convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging
from pyscf import gto, scf
from pyscf.fci import direct_spin1

from shades.solvers import FCISolver
from shades.estimators import ShadowEstimator
from shades.utils import make_hydrogen_chain
from shades.monte_carlo import MPSSampler, MonteCarloEstimator

from plotting_config import setup_plotting_style, save_figure
from utils import spinorb_to_spatial_chem, doubles_energy, total_energy_from_rdm12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

RUN_COMMENT = "Check that using exact coefficients the Monte Carlo sampler yields a valid."

DEFAULT_OUTPUT_DIR = f"./results/rdm2_convergence/shadow/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 20
N_MC_ITERS = [5000, 10_000, 15_000, 25_000, 50_000]
N_SHADOWS = [1000, 4000, 10_000, 20_000, 50_000]
N_K_ESTIMATORS = 20

N_HYDROGEN = 4
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"

FIGURE_SIZE = (10, 4)
PLOT_DPI = 300

def main():

    """Run RDM2 convergence analysis."""
    print("=" * 70)
    print("Monte Carlo RDM2 Convergence Analysis")
    print("=" * 70)

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    hstring = make_hydrogen_chain(N_HYDROGEN, BOND_LENGTH)
    mol = gto.Mole()
    mol.build(atom=hstring, basis=BASIS_SET, verbose=0)

    mf = scf.RHF(mol)
    mf.run()

    print(f"\nMolecule: H{N_HYDROGEN} chain (bond length = {BOND_LENGTH:.2f} A)")
    print(f"Basis set: {BASIS_SET}")

    fci_solver = FCISolver(mf)
    fci_solver.solve()

    E_fci = fci_solver.energy
    E_hf = mf.e_tot

    norb = mf.mo_coeff.shape[1]
    nelec = mf.mol.nelec
    rdm1, rdm2_ref = direct_spin1.make_rdm12(
        fci_solver.civec, norb, nelec
    )
    E_double_ref = doubles_energy(rdm2_ref, mf)

    print("\n" + "=" * 70)
    print("Reference Energies")
    print("=" * 70)
    print(f"Hartree-Fock Energy:      {E_hf:.10f} Ha")
    print(f"Exact FCI Energy:         {E_fci:.10f} Ha")
    print(f"Correlation Energy:       {E_fci - E_hf:.10f} Ha")
    print("=" * 70)

    # sampler = MetropolisSampler(estimator, _gen_single_site_hops, auto_corr_iters=1000)
    sampler = MPSSampler(mf)

    results = {
        'rdm2': np.empty((len(N_SHADOWS), len(N_MC_ITERS), N_RUNS, norb, norb, norb, norb), dtype=np.float32),
        'E_tot': np.empty((len(N_SHADOWS), len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'E_doubles': np.empty((len(N_SHADOWS), len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'rel_err_E2': np.empty((len(N_SHADOWS), len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'rel_frob_rdm2': np.empty((len(N_SHADOWS), len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'max_abs_rdm2': np.empty((len(N_SHADOWS), len(N_MC_ITERS), N_RUNS), dtype=np.float64),
    }

    print("\n" + "=" * 70)
    print("Running Monte Carlo Convergence Study")
    print("=" * 70)
    print(f"Sample schedule: {N_SHADOWS}")
    print(f"Iteration schedule: {N_MC_ITERS}")
    print(f"Runs per iteration count: {N_RUNS}")

    shadow1 = ShadowEstimator(mf, fci_solver)
    shadow2 = ShadowEstimator(mf, fci_solver)

    for l, n_samples in enumerate(N_SHADOWS):

        print(f"\n{'='*70}")
        print(f"N Shadow samples: {n_samples}")
        print(f"{'='*70}")

        for i, n_iters in enumerate(N_MC_ITERS):
            print(f"MC Iterations: {n_iters:,}")

            for j, run in enumerate(range(N_RUNS)):
                print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)

                shadow1.sample(n_samples//2, N_K_ESTIMATORS)
                shadow2.sample(n_samples//2, N_K_ESTIMATORS)
                estimator = (shadow1, shadow2)

                mc = MonteCarloEstimator(estimator, sampler)
                rdm2_mc = mc.estimate_2rdm(max_iters=n_iters)
                rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)

                E_doubles = doubles_energy(rdm2, mf)
                rel_err = np.abs(E_double_ref - E_doubles) / np.abs(E_double_ref)

                diff = rdm2 - rdm2_ref
                frob = np.linalg.norm(diff)
                rel_frob = frob / np.linalg.norm(rdm2_ref)
                max_abs = np.max(np.abs(diff))

                results['rdm2'][l, i, j] = rdm2.astype(np.float32)
                results['E_doubles'][l, i, j] = E_doubles
                results['rel_err_E2'][l, i, j] = rel_err
                results['rel_frob_rdm2'][l, i, j] = rel_frob
                results['max_abs_rdm2'][l, i, j] = max_abs

                E = total_energy_from_rdm12(rdm1, rdm2, mf)
                print(f"E: {E}, E_doubles = {E_doubles}, rel_err = {rel_err} ||dRDM2||_F = {rel_frob:.4e}")

                results['E_tot'][l, i, j] = E

                shadow1.clear_sample()
                shadow2.clear_sample()

    npz_path = os.path.join(output_dir, "data.npz")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_mc_iters": np.asarray(N_MC_ITERS, dtype=np.int64),
        'n_shadow_samples': np.asarray(N_SHADOWS, dtype=np.int64),
        "E_hf_hartree": float(E_hf),
        "E_fci_hartree": float(E_fci),
        "E_corr_hartree": float(E_fci - E_hf),
        'n_k_estimators': int(N_K_ESTIMATORS),
        "run_comment": RUN_COMMENT
    }

    meta_np = {k: np.asarray(v, dtype=object) for k, v in metadata.items()}

    np.savez_compressed(npz_path, **results, **meta_np)
    print(f"Saved: {npz_path}")

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    print("\n" + "=" * 70)
    print("Generating Convergence Plots")
    print("=" * 70)

    npz_path = os.path.join(output_dir, "data.npz")
    results = np.load(npz_path, allow_pickle=True)

    n_samples_arr = np.asarray(results['n_shadow_samples'])

    rel_frob_mean = results['rel_frob_rdm2'].mean(axis=(1, 2))
    rel_frob_std  = results['rel_frob_rdm2'].std(axis=(1, 2), ddof=1)
    rel_frob_sem  = rel_frob_std / np.sqrt(N_RUNS)

    rel_E2_mean = results['rel_err_E2'].mean(axis=(1, 2))
    rel_E2_std  = results['rel_err_E2'].std(axis=(1, 2), ddof=1)
    rel_E2_sem  = rel_E2_std / np.sqrt(N_RUNS)

    setup_plotting_style()
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))


    # --- Plot 1: Relative Frobenius error of RDM2 ---
    ax1.errorbar(
        n_samples_arr,
        rel_frob_mean,
        yerr=rel_frob_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$\|\Delta\Gamma\|_F / \|\Gamma_{\mathrm{ref}}\|_F$',
        linewidth=2, markersize=8,
    )

    # ref_scaling = rel_frob_mean[0] * np.sqrt(n_samples_arr[0] / n_samples_arr)
    # ax1.loglog(
    #     n_samples_arr, ref_scaling, '--',
    #     label=r'$1/\sqrt{N}$ scaling', linewidth=2, alpha=0.6
    # )

    ax1.set_xlabel('Number of Shadow Samples')
    ax1.set_ylabel('Relative Frobenius Error')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('RDM2 Relative Frobenius Convergence')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')

    # --- Plot 2: Relative two-electron energy error (E2 / doubles) ---
    ax2.errorbar(
        n_samples_arr,
        rel_E2_mean,
        yerr=rel_E2_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$|E_{2,\mathrm{MC}}-E_{2,\mathrm{ref}}|/|E_{2,\mathrm{ref}}|$',
        linewidth=2, markersize=8,
    )

    # Compute mean and SEM for E_doubles across runs
    E_doubles_mean = results['E_doubles'].mean(axis=(1, 2))  # Average over MC iters and runs
    E_doubles_std = results['E_doubles'].std(axis=(1, 2), ddof=1)
    E_doubles_sem = E_doubles_std / np.sqrt(N_RUNS)


    # ref_scaling_rel = rel_E2_mean[0] * np.sqrt(n_samples_arr[0] / n_samples_arr)
    # ax2.loglog(
    #     n_samples_arr, ref_scaling_rel, '--',
    #     label=r'$1/\sqrt{N}$ scaling', linewidth=2, alpha=0.6
    # )

    ax1.set_xlabel('Number of Shadow Samples')
    ax2.set_ylabel('Relative $E_2$ Error')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Two-electron Energy Convergence')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')

    ax3.errorbar(
        n_samples_arr,
        E_doubles_mean,
        yerr=E_doubles_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$E_2^{\mathrm{MC}}$',
        linewidth=2, markersize=8,
    )
    ax3.axhline(E_double_ref, color='r', linestyle='--', linewidth=2, label=r'$E_2^{\mathrm{ref}}$')

    ax3.set_xlabel('Number of Shadow Samples')
    ax3.set_ylabel(r'$E_2$ (Hartree)')
    ax3.set_xscale('log')
    ax3.set_title('Two-electron Energy vs Reference')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'rdm2_convergence.pdf')
    png_path = os.path.join(output_dir, 'rdm2_convergence.png')
    svg_path = os.path.join(output_dir, 'rdm2_convergence.svg')
    save_figure(pdf_path)
    save_figure(png_path, dpi=PLOT_DPI)
    save_figure(svg_path)
    print(f"Plots saved: rdm2_convergence.pdf, rdm2_convergence.png, rdm2_convergence.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

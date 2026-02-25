"""Bias test: signed E2 errors for mean vs median-of-means.

Plots signed (not absolute) deviations from the reference two-electron
energy across many runs for each system size, for both the plain
arithmetic mean and median-of-means MC estimators. If unbiased, the
signed errors should scatter symmetrically around zero.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime
from pyscf import gto, scf
from pyscf.fci import direct_spin1

from shades.solvers import FCISolver
from shades.estimators import ShadowEstimator
from shades.utils import make_hydrogen_chain
from shades.monte_carlo import MPSSampler, MonteCarloEstimator

from plotting_config import setup_plotting_style, save_figure
from utils import spinorb_to_spatial_chem, doubles_energy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

RUN_COMMENT = "Bias test: signed E2 errors for mean vs median-of-means."

DEFAULT_OUTPUT_DIR = f"./results/rdm2_bias_test/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 20
N_MC_ITERS = 5000
N_SHADOWS = 5000
N_K_ESTIMATORS = 20
N_MC_BATCHES = 50

N_HYDROGEN = [2, 4, 6]
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"


def main():
    print("=" * 70)
    print("Bias Test: Mean vs Median-of-Means")
    print("=" * 70)

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    print(f"\nSystem sizes (N_H): {N_HYDROGEN}")
    print(f"Shadow samples: {N_SHADOWS}, K = {N_K_ESTIMATORS}")
    print(f"MC iterations: {N_MC_ITERS}")
    print(f"  Mean: arithmetic average over all {N_MC_ITERS} iters")
    print(f"  MoM:  {N_MC_BATCHES} batches of {N_MC_ITERS // N_MC_BATCHES} iters, element-wise median")
    print(f"Runs per system size: {N_RUNS}")

    all_results = {}
    reference_data = {}

    for n_h in N_HYDROGEN:
        print(f"\n{'='*70}")
        print(f"System: H{n_h} chain (bond length = {BOND_LENGTH:.2f} A)")
        print(f"{'='*70}")

        hstring = make_hydrogen_chain(n_h, BOND_LENGTH)
        mol = gto.Mole()
        mol.build(atom=hstring, basis=BASIS_SET, verbose=0)

        mf = scf.RHF(mol)
        mf.run()

        fci_solver = FCISolver(mf)
        fci_solver.solve()
        E_fci = fci_solver.energy
        E_hf = mf.e_tot

        norb = mf.mo_coeff.shape[1]
        nelec = mf.mol.nelec
        _, rdm2_ref = direct_spin1.make_rdm12(fci_solver.civec, norb, nelec)
        E_double_ref = doubles_energy(rdm2_ref, mf)

        print(f"Basis set: {BASIS_SET}, norb = {norb}")
        print(f"Hartree-Fock Energy:      {E_hf:.10f} Ha")
        print(f"Exact FCI Energy:         {E_fci:.10f} Ha")
        print(f"E2 reference:             {E_double_ref:.10f} Ha")

        reference_data[n_h] = {
            'E_hf': E_hf,
            'E_fci': E_fci,
            'E_corr': E_fci - E_hf,
            'E_double_ref': E_double_ref,
            'norb': norb,
        }

        results = {
            'E2_mean': np.empty(N_RUNS, dtype=np.float64),
            'E2_mom': np.empty(N_RUNS, dtype=np.float64),
            'signed_err_mean': np.empty(N_RUNS, dtype=np.float64),
            'signed_err_mom': np.empty(N_RUNS, dtype=np.float64),
        }

        sampler = MPSSampler(mf)
        shadow1 = ShadowEstimator(mf, fci_solver)
        shadow2 = ShadowEstimator(mf, fci_solver)

        for j in range(N_RUNS):
            print(f"  Run {j+1}/{N_RUNS}...", end=" ", flush=True)

            shadow1.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
            shadow2.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
            estimator = (shadow1, shadow2)

            # Plain mean
            mc_mean = MonteCarloEstimator(estimator, sampler)
            rdm2_mc = mc_mean.estimate_2rdm(max_iters=N_MC_ITERS)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)
            E2_mean = doubles_energy(rdm2, mf)

            # Median-of-means (same shadow samples)
            mc_mom = MonteCarloEstimator(estimator, sampler)
            rdm2_mc_mom = mc_mom.estimate_2rdm(max_iters=N_MC_ITERS, n_batches=N_MC_BATCHES)
            rdm2_mom = spinorb_to_spatial_chem(rdm2_mc_mom, norb)
            E2_mom = doubles_energy(rdm2_mom, mf)

            err_mean = E2_mean - E_double_ref
            err_mom = E2_mom - E_double_ref

            results['E2_mean'][j] = E2_mean
            results['E2_mom'][j] = E2_mom
            results['signed_err_mean'][j] = err_mean
            results['signed_err_mom'][j] = err_mom

            print(
                f"E2_mean = {E2_mean:.6f} (err = {err_mean:+.4e}), "
                f"E2_mom = {E2_mom:.6f} (err = {err_mom:+.4e})"
            )

            shadow1.clear_sample()
            shadow2.clear_sample()

        all_results[n_h] = results

    # Save data
    npz_path = os.path.join(output_dir, "data.npz")
    npz_data = {}
    for n_h in N_HYDROGEN:
        for key, arr in all_results[n_h].items():
            npz_data[f"H{n_h}_{key}"] = arr
    np.savez_compressed(npz_path, **npz_data)
    print(f"\nSaved: {npz_path}")

    metadata = {
        "system": "H chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_mc_iters": int(N_MC_ITERS),
        "n_shadow_samples": int(N_SHADOWS),
        "n_k_estimators": int(N_K_ESTIMATORS),
        "n_mc_batches": int(N_MC_BATCHES),
        "comments": RUN_COMMENT,
        "reference_data": {str(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in reference_data.items()},
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Plot
    setup_plotting_style()
    n_sizes = len(N_HYDROGEN)
    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5), squeeze=False)
    axes = axes[0]

    for idx, n_h in enumerate(N_HYDROGEN):
        ax = axes[idx]
        err_mean = all_results[n_h]['signed_err_mean']
        err_mom = all_results[n_h]['signed_err_mom']

        x_mean = np.ones(N_RUNS) * 0.8
        x_mom = np.ones(N_RUNS) * 2.2

        # Individual runs as scatter
        ax.scatter(x_mean, err_mean, alpha=0.5, s=30, color='C0', zorder=3)
        ax.scatter(x_mom, err_mom, alpha=0.5, s=30, color='C1', zorder=3)

        # Mean +/- SEM as crosshair
        for x, errs, color, label in [
            (0.8, err_mean, 'C0', 'Mean'),
            (2.2, err_mom, 'C1', 'MoM'),
        ]:
            mu = errs.mean()
            sem = errs.std(ddof=1) / np.sqrt(N_RUNS)
            ax.errorbar(x, mu, yerr=2 * sem, fmt='D', color=color, markersize=8,
                        capsize=6, capthick=2, linewidth=2, zorder=4, label=f'{label}: $\\bar{{\\epsilon}}={mu:.4f}$')

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_xticks([0.8, 2.2])
        ax.set_xticklabels(['Mean', 'MoM'])
        ax.set_xlim(0, 3)
        ax.set_ylabel(r'$E_2^{\mathrm{MC}} - E_2^{\mathrm{ref}}$ (Ha)')
        ax.set_title(f'H{n_h}')
        ax.legend(loc='best', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Signed $E_2$ Error: Mean vs Median-of-Means', y=1.02)
    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'bias_test.pdf')
    png_path = os.path.join(output_dir, 'bias_test.png')
    save_figure(pdf_path)
    save_figure(png_path, dpi=300)

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"{'System':<8} {'Method':<8} {'Mean err':>12} {'SEM':>12} {'Std':>12} {'N>0':>6} {'N<0':>6}")
    print("-" * 70)
    for n_h in N_HYDROGEN:
        for method, key in [('Mean', 'signed_err_mean'), ('MoM', 'signed_err_mom')]:
            errs = all_results[n_h][key]
            mu = errs.mean()
            sem = errs.std(ddof=1) / np.sqrt(N_RUNS)
            std = errs.std(ddof=1)
            n_pos = np.sum(errs > 0)
            n_neg = np.sum(errs < 0)
            print(f"H{n_h:<7} {method:<8} {mu:>+12.6f} {sem:>12.6f} {std:>12.6f} {n_pos:>6} {n_neg:>6}")

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

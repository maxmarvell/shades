"""Monitor MC 2-RDM convergence with shadow estimators.

Compares plain mean vs median-of-means estimation to see whether
the median approach is robust against the heavy-tailed outliers
from shadow overlap noise.
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
from shades.monte_carlo import MonteCarloEstimator, MPSSampler

from plotting_config import setup_plotting_style, save_figure
from utils import spinorb_to_spatial_chem, doubles_energy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

RUN_COMMENT = "Compare plain mean vs median-of-means MC 2-RDM estimation."

DEFAULT_OUTPUT_DIR = f"./results/rdm2_monitor/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 5
N_MC_ITERS = 5000
N_SHADOWS = 1000
N_K_ESTIMATORS = 20
N_MC_BATCHES = 50  # for median-of-means: 5000/50 = 100 iters per batch
LOG_EVERY = 50     # for plain mean: log every 50 iters

N_HYDROGEN = 6
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"


def main():
    print("=" * 70)
    print("Monitor MC 2-RDM Convergence: Mean vs Median-of-Means")
    print("=" * 70)

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    hstring = make_hydrogen_chain(N_HYDROGEN, BOND_LENGTH)
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
    rdm2_ref_norm = np.linalg.norm(rdm2_ref)

    print(f"\nMolecule: H{N_HYDROGEN} chain (bond length = {BOND_LENGTH:.2f} A)")
    print(f"Basis set: {BASIS_SET}, norb = {norb}")
    print(f"Hartree-Fock Energy:      {E_hf:.10f} Ha")
    print(f"Exact FCI Energy:         {E_fci:.10f} Ha")
    print(f"Correlation Energy:       {E_fci - E_hf:.10f} Ha")
    print(f"\nShadow samples: {N_SHADOWS}, K = {N_K_ESTIMATORS}")
    print(f"MC iterations: {N_MC_ITERS}")
    print(f"  Plain mean: logging every {LOG_EVERY} iters")
    print(f"  Median-of-means: {N_MC_BATCHES} batches of {N_MC_ITERS // N_MC_BATCHES} iters")
    print(f"Runs: {N_RUNS}")

    sampler = MPSSampler(mf)

    # --- Storage for plain mean ---
    n_checkpoints_mean = N_MC_ITERS // LOG_EVERY
    iters_mean = np.arange(1, n_checkpoints_mean + 1) * LOG_EVERY
    all_rel_frob_mean = np.empty((N_RUNS, n_checkpoints_mean), dtype=np.float64)
    all_rel_err_E2_mean = np.empty((N_RUNS, n_checkpoints_mean), dtype=np.float64)

    # --- Storage for median-of-means ---
    batch_size = N_MC_ITERS // N_MC_BATCHES
    iters_mom = np.arange(1, N_MC_BATCHES + 1) * batch_size
    all_rel_frob_mom = np.empty((N_RUNS, N_MC_BATCHES), dtype=np.float64)
    all_rel_err_E2_mom = np.empty((N_RUNS, N_MC_BATCHES), dtype=np.float64)

    shadow1 = ShadowEstimator(mf, fci_solver)
    shadow2 = ShadowEstimator(mf, fci_solver)

    for run in range(N_RUNS):
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{N_RUNS}")
        print(f"{'='*70}")

        shadow1.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
        shadow2.sample(N_SHADOWS // 2, N_K_ESTIMATORS)

        # --- Plain mean ---
        print("  [Plain mean]")
        run_rel_frob = []
        run_rel_err_E2 = []

        def on_iter_mean(i, gamma):
            if (i + 1) % LOG_EVERY != 0:
                return
            rdm2 = spinorb_to_spatial_chem(gamma, norb)
            rel_frob = np.linalg.norm(rdm2 - rdm2_ref) / rdm2_ref_norm
            E_doubles = doubles_energy(rdm2, mf)
            rel_err = np.abs(E_double_ref - E_doubles) / np.abs(E_double_ref)
            run_rel_frob.append(rel_frob)
            run_rel_err_E2.append(rel_err)
            print(f"    iter {i+1:5d}: rel_frob = {rel_frob:.4e}, rel_err_E2 = {rel_err:.4e}")

        mc = MonteCarloEstimator((shadow1, shadow2), sampler)
        mc.estimate_2rdm(max_iters=N_MC_ITERS, callback=on_iter_mean)

        all_rel_frob_mean[run] = run_rel_frob
        all_rel_err_E2_mean[run] = run_rel_err_E2

        # --- Median-of-means (same shadow samples) ---
        print("  [Median-of-means]")
        run_rel_frob_mom = []
        run_rel_err_E2_mom = []

        def on_batch(i, gamma):
            rdm2 = spinorb_to_spatial_chem(gamma, norb)
            rel_frob = np.linalg.norm(rdm2 - rdm2_ref) / rdm2_ref_norm
            E_doubles = doubles_energy(rdm2, mf)
            rel_err = np.abs(E_double_ref - E_doubles) / np.abs(E_double_ref)
            run_rel_frob_mom.append(rel_frob)
            run_rel_err_E2_mom.append(rel_err)
            batch_num = len(run_rel_frob_mom)
            print(f"    batch {batch_num:3d}/{N_MC_BATCHES}: rel_frob = {rel_frob:.4e}, rel_err_E2 = {rel_err:.4e}")

        mc2 = MonteCarloEstimator((shadow1, shadow2), sampler)
        mc2.estimate_2rdm(max_iters=N_MC_ITERS, n_batches=N_MC_BATCHES, callback=on_batch)

        all_rel_frob_mom[run] = run_rel_frob_mom
        all_rel_err_E2_mom[run] = run_rel_err_E2_mom

        shadow1.clear_sample()
        shadow2.clear_sample()

    # Save data
    npz_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        iters_mean=iters_mean,
        rel_frob_mean=all_rel_frob_mean,
        rel_err_E2_mean=all_rel_err_E2_mean,
        iters_mom=iters_mom,
        rel_frob_mom=all_rel_frob_mom,
        rel_err_E2_mom=all_rel_err_E2_mom,
    )
    print(f"\nSaved: {npz_path}")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_mc_iters": int(N_MC_ITERS),
        "n_shadow_samples": int(N_SHADOWS),
        "n_k_estimators": int(N_K_ESTIMATORS),
        "n_mc_batches": int(N_MC_BATCHES),
        "log_every": int(LOG_EVERY),
        "E_hf_hartree": float(E_hf),
        "E_fci_hartree": float(E_fci),
        "E_corr_hartree": float(E_fci - E_hf),
        "E_double_ref": float(E_double_ref),
        "comments": RUN_COMMENT,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Plot
    setup_plotting_style()
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric_mean, metric_mom, ylabel, title in [
        (axes[0], all_rel_frob_mean, all_rel_frob_mom,
         r'$\|\Delta\Gamma\|_F / \|\Gamma_{\mathrm{ref}}\|_F$',
         'RDM2 Relative Frobenius Error'),
        (axes[1], all_rel_err_E2_mean, all_rel_err_E2_mom,
         r'$|E_{2}^{\mathrm{MC}} - E_{2}^{\mathrm{ref}}| / |E_{2}^{\mathrm{ref}}|$',
         r'Relative $E_2$ Error'),
    ]:
        for r in range(N_RUNS):
            ax.plot(iters_mean, metric_mean[r], alpha=0.25, linewidth=0.6, color='C0')
            ax.plot(iters_mom, metric_mom[r], alpha=0.25, linewidth=0.6, color='C1')
        ax.plot(iters_mean, metric_mean.mean(axis=0), 'C0-', linewidth=2, label='mean')
        ax.plot(iters_mom, metric_mom.mean(axis=0), 'C1-', linewidth=2, label='median-of-means')
        ax.set_xlabel('MC Iteration')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'convergence_mean_vs_mom.pdf')
    png_path = os.path.join(output_dir, 'convergence_mean_vs_mom.png')
    save_figure(pdf_path)
    save_figure(png_path, dpi=300)

    plt.show()

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

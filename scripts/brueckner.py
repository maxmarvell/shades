import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf

from shades.brueckner import brueckner_cycle, rotate_mf
from shades.estimators import ShadowEstimator, ExactEstimator
from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain
from plotting_config import setup_plotting_style, save_figure

# =============================================================================
# CONSTANTS - Adjust these for different runs
# =============================================================================

# Molecule parameters
N_ATOMS = 4
BOND_LENGTH = 0.73
BASIS = "sto-3g"
SPIN = 0 

# Artificial rotation parameters
O_TARGET = 0.75

# Shadow protocol parameters
N_SAMPLES = 2000
N_K_ESTIMATORS = 20
N_SHOTS_REPETITIONS = 100  # Number of times to repeat shadow estimation
USE_QULACS = True
N_JOBS = 8

# Brueckner optimization parameters
MAX_BRUECKNER_ITER = 10
DAMPING = 0.8
USE_DIIS = False
CONVERGENCE_THRESHOLD = 1e-8  # ||c1|| threshold for convergence
METHOD = "taylor"

# Output parameters
VERBOSE = 1  # 0=WARNING, 1=INFO, 2=DEBUG
OUTPUT_DIR = Path("results/brueckner_convergence/engineered")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if VERBOSE >= 2 else logging.INFO if VERBOSE == 1 else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "brueckner_convergence.log", mode="w"),
    ],
    force=True,  # Remove any existing handlers
)

logger = logging.getLogger(__name__)


def main():

    logger.info("=" * 80)
    logger.info("BRUECKNER CONVERGENCE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Molecule: H{N_ATOMS} chain, R={BOND_LENGTH} Å, basis={BASIS}, spin={SPIN}")
    logger.info(f"Shadow: N={N_SAMPLES}, K={N_K_ESTIMATORS}, repetitions={N_SHOTS_REPETITIONS}")
    logger.info(f"Brueckner: max_iter={MAX_BRUECKNER_ITER}, damping={DAMPING}, DIIS={USE_DIIS}")
    logger.info(f"Convergence threshold: ||c1|| < {CONVERGENCE_THRESHOLD:.6e}")
    logger.info("=" * 80 + "\n")

    mol = gto.Mole()
    atom = make_hydrogen_chain(n_atoms=N_ATOMS, bond_length=BOND_LENGTH)
    mol.build(atom=atom, basis=BASIS, spin=SPIN, verbose=0)
    mf = scf.RHF(mol)
    mf.run()

    lam = np.sqrt(1.0 / O_TARGET - 1.0) 

    nocc, _ = mf.mol.nelec         # number of occupied spatial orbitals
    norb = mf.mo_coeff.shape[1]
    nvirt = norb - nocc

    t1 = np.zeros((nocc, nvirt))
    t1[nocc-1, 0] = lam

    mf_rot = rotate_mf(mf, t1, canonicalise=True, damping=0.0)

    # Use a single solver instance for both estimators to ensure state synchronization
    solver = FCISolver(mf_rot)

    logger.info("Computing exact FCI energy...")
    _, fci_energy = solver.solve()
    logger.info(f"Exact FCI energy: {fci_energy:.10f} Ha")

    # Both estimators share the same solver instance
    exact_estimator = ExactEstimator(mf_rot, solver, verbose=0)
    shadow_estimator = ShadowEstimator(mf_rot, solver, verbose=0)

    c0 = exact_estimator.estimate_c0()
    logger.info(f"Rotated c0: {c0:.10f}")

    # Pre-allocate arrays to store convergence data
    # Shape: (n_brueckner_iterations, n_shots_repetitions)
    E_samples = np.zeros((MAX_BRUECKNER_ITER, N_SHOTS_REPETITIONS))
    c0_samples = np.zeros((MAX_BRUECKNER_ITER, N_SHOTS_REPETITIONS))

    E_exact = np.zeros(MAX_BRUECKNER_ITER)
    c0_exact = np.zeros(MAX_BRUECKNER_ITER)
    c1_norms = np.zeros(MAX_BRUECKNER_ITER)

    # Shape: (n_brueckner_iterations,) for mean/std statistics
    E_means = np.zeros(MAX_BRUECKNER_ITER)
    E_stds = np.zeros(MAX_BRUECKNER_ITER)
    c0_means = np.zeros(MAX_BRUECKNER_ITER)
    c0_stds = np.zeros(MAX_BRUECKNER_ITER)

    counter = 0  # Use list for mutable counter in closure

    def callback(E, c0, norm):
        nonlocal counter
        logger.info(f"--- Brueckner iteration {counter + 1} ---")
        logger.info(f"Exact: E = {E:.10f} Ha, c0 = {c0:.8f}, ||c1|| = {norm:.8e}")

        E_exact[counter] = E
        c0_exact[counter] = c0
        c1_norms[counter] = norm
     
        # Update shadow estimator with the same mf that exact_estimator now has
        # (after brueckner_cycle's update_reference() call)
        shadow_estimator.update_reference(exact_estimator.mf)

        for i in range(N_SHOTS_REPETITIONS):
            shadow_estimator.clear_sample()
            E_estimate, c0_estimate, _, _ = shadow_estimator.run(
                n_samples=N_SAMPLES,
                n_k_estimators=N_K_ESTIMATORS,
                n_jobs=N_JOBS,
                use_qulacs=USE_QULACS,
                calc_c1=True
            )

            # Store individual samples
            E_samples[counter, i] = E_estimate
            c0_samples[counter, i] = c0_estimate

        # Compute and store statistics
        E_means[counter] = np.mean(E_samples[counter, :])
        E_stds[counter] = np.std(E_samples[counter, :])
        c0_means[counter] = np.mean(c0_samples[counter, :])
        c0_stds[counter] = np.std(c0_samples[counter, :])

        logger.info(f"Shadow: E = {E_means[counter]:.10f} ± {E_stds[counter]:.10f} Ha")
        logger.info(f"Shadow: c0 = {c0_means[counter]:.8f} ± {c0_stds[counter]:.8f}")


        counter += 1

        return False


    brueckner_cycle(
        mf=mf_rot,
        estimator=exact_estimator,
        canonicalise=False,
        damping=DAMPING,
        use_diis=USE_DIIS,
        max_iter=MAX_BRUECKNER_ITER,
        callback_fn=callback,
        verbose=0,
        method=METHOD
    )

    n_iters = counter
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Brueckner cycle completed after {n_iters} iterations")
    logger.info(f"Saving results to {OUTPUT_DIR}")

    np.savez(
        OUTPUT_DIR / "convergence_data.npz",
        E_samples=E_samples[:n_iters, :],
        c0_samples=c0_samples[:n_iters, :],
        E_means=E_means[:n_iters],
        E_stds=E_stds[:n_iters],
        c0_means=c0_means[:n_iters],
        c0_stds=c0_stds[:n_iters],
        E_exacts=E_exact[:n_iters],
        c0_exacts=c0_exact[:n_iters],
        c1_norms=c1_norms[:n_iters],
        fci_energy=fci_energy,
        n_atoms=N_ATOMS,
        bond_length=BOND_LENGTH,
        basis=BASIS,
        n_samples=N_SAMPLES,
        n_k_estimators=N_K_ESTIMATORS,
        n_shots_repetitions=N_SHOTS_REPETITIONS,
    )

    logger.info(f"Results saved successfully!")

    # =========================================================================
    # Generate plots
    # =========================================================================
    logger.info("Generating convergence plots...")
    setup_plotting_style()

    # Extract data for plotting
    iterations = np.arange(1, n_iters + 1)
    E_mean = E_means[:n_iters]
    E_std = E_stds[:n_iters]
    c0_mean = c0_means[:n_iters]
    c0_std = c0_stds[:n_iters]
    E_ex = E_exact[:n_iters]
    c0_ex = c0_exact[:n_iters]
    c1_norm = c1_norms[:n_iters]

    # Figure 1: Energy and c0 convergence vs Brueckner iteration
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Subplot 1: Energy estimates
    ax1.errorbar(iterations, E_mean, yerr=E_std, fmt='o-', label=r'Shadow estimate',
                 capsize=5, markersize=5, alpha=0.8)
    ax1.plot(iterations, E_ex, 's-', label=r'Exact', markersize=5)
    ax1.axhline(fci_energy, color='k', linestyle='--', linewidth=1.5, label=r'FCI energy')
    ax1.set_xlabel(r'Brueckner iteration')
    ax1.set_ylabel(r'Energy (Ha)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: c0 values
    ax2.errorbar(iterations, c0_mean, yerr=c0_std, fmt='o-', label=r'Shadow estimate',
                 capsize=5, markersize=5, alpha=0.8)
    ax2.plot(iterations, c0_ex, 's-', label=r'Exact', markersize=5)
    ax2.set_xlabel(r'Brueckner iteration')
    ax2.set_ylabel(r'$c_0$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(OUTPUT_DIR / "convergence_vs_iteration.svg")
    plt.close()

    # Figure 2: Energy vs c0 (exact)
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 8))

    # Subplot 1: Mean energy with error bars vs exact c0
    ax3.errorbar(c0_ex, E_mean, yerr=E_std, fmt='o', capsize=5, markersize=6, alpha=0.8)
    # Add iteration labels
    for i, (c0_val, E_val) in enumerate(zip(c0_ex, E_mean)):
        ax3.annotate(f'{i+1}', (c0_val, E_val), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, alpha=0.7)
    ax3.axhline(fci_energy, color='k', linestyle='--', linewidth=1.5, label=r'FCI energy')
    ax3.set_xlabel(r'Exact $c_0$')
    ax3.set_ylabel(r'Mean energy (Ha)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 2: Variance magnitude vs exact c0
    ax4.plot(c0_ex, E_std, 'o-', markersize=6)
    for i, (c0_val, std_val) in enumerate(zip(c0_ex, E_std)):
        ax4.annotate(f'{i+1}', (c0_val, std_val), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, alpha=0.7)
    ax4.set_xlabel(r'Exact $c_0$')
    ax4.set_ylabel(r'Energy std dev (Ha)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(OUTPUT_DIR / "convergence_vs_c0.svg")
    plt.close()

    # Figure 3: Energy vs c1_norm
    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(8, 8))

    # Subplot 1: Mean energy with error bars vs c1_norm
    ax5.errorbar(c1_norm, E_mean, yerr=E_std, fmt='o', capsize=5, markersize=6, alpha=0.8)
    # Add iteration labels
    for i, (norm_val, E_val) in enumerate(zip(c1_norm, E_mean)):
        ax5.annotate(f'{i+1}', (norm_val, E_val), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, alpha=0.7)
    ax5.axhline(fci_energy, color='k', linestyle='--', linewidth=1.5, label=r'FCI energy')
    ax5.set_xlabel(r'$\|c_1\|$')
    ax5.set_ylabel(r'Mean energy (Ha)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Subplot 2: Variance magnitude vs c1_norm
    ax6.plot(c1_norm, E_std, 'o-', markersize=6)
    for i, (norm_val, std_val) in enumerate(zip(c1_norm, E_std)):
        ax6.annotate(f'{i+1}', (norm_val, std_val), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, alpha=0.7)
    ax6.set_xlabel(r'$\|c_1\|$')
    ax6.set_ylabel(r'Energy std dev (Ha)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(OUTPUT_DIR / "convergence_vs_c1norm.svg")
    plt.close()

    logger.info("All plots generated successfully!")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

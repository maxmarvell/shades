"""
Brueckner Convergence Analysis Script

This script demonstrates and analyzes the convergence behavior of Brueckner orbital
optimization using both TrivialEstimator (exact) and ShadowEstimator (sampling-based).

The key idea is to rotate orbitals to minimize single excitation amplitudes (c1),
which should improve the reference state and reduce the correlation energy needed
from double excitations.
"""

import logging
from copy import copy
from pathlib import Path
from typing import Optional

import numpy as np
from pyscf import gto, scf

from shades.brueckner import brueckner_cycle
from shades.estimators import ShadowEstimator, TrivialEstimator
from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain

# =============================================================================
# CONSTANTS - Adjust these for different runs
# =============================================================================

# Molecule parameters
N_ATOMS = 5
BOND_LENGTH = 1.5
BASIS = "sto-3g"
SPIN = 1  # Unpaired electrons for UHF

# Shadow protocol parameters
N_SAMPLES = 1000
N_K_ESTIMATORS = 20
N_SHOTS_REPETITIONS = 100  # Number of times to repeat shadow estimation
USE_QULACS = True
N_JOBS = 8

# Brueckner optimization parameters
MAX_BRUECKNER_ITER = 5
DAMPING = 0.7
USE_DIIS = False
CONVERGENCE_THRESHOLD = 1e-8  # ||c1|| threshold for convergence

# Output parameters
VERBOSE = 1  # 0=WARNING, 1=INFO, 2=DEBUG
OUTPUT_DIR = Path("results/brueckner_convergence")

# =============================================================================
# Logging Setup
# =============================================================================

# Create output directory first
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if VERBOSE >= 2 else logging.INFO if VERBOSE == 1 else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "brueckner_convergence.log", mode="w"),
    ],
)

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================


def setup_molecule_and_mf(n_atoms: int, bond_length: float, basis: str, spin: int):
    """Setup molecule and mean-field calculation."""
    logger.info(f"Setting up {n_atoms}-atom H chain, bond_length={bond_length}, spin={spin}")

    atom = make_hydrogen_chain(n_atoms, bond_length=bond_length)
    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, spin=spin, verbose=0)

    # Use UHF for spin != 0
    if spin != 0:
        mf = scf.UHF(mol)
        logger.info("Using UHF reference")
    else:
        mf = scf.RHF(mol)
        logger.info("Using RHF reference")

    mf.run()
    logger.info(f"Initial HF energy: {mf.e_tot:.10f} Ha")

    return mol, mf


def setup_estimators(mf, verbose: int):
    """Setup both trivial and shadow estimators with FCI solver."""
    logger.info("Setting up FCI solver and estimators")

    # Setup FCI solver
    solver = FCISolver(mf)

    # Trivial estimator (exact)
    trivial_estimator = TrivialEstimator(mf, solver, verbose=verbose)
    logger.info("Created TrivialEstimator (exact)")

    # Shadow estimator (sampling-based)
    # Note: ShadowEstimator only takes mf, solver, verbose in __init__
    # Sampling parameters are passed to run() method
    shadow_estimator = ShadowEstimator(mf, solver, verbose=verbose)
    logger.info(
        f"Created ShadowEstimator (will use N={N_SAMPLES}, K={N_K_ESTIMATORS}, "
        f"qulacs={USE_QULACS}, n_jobs={N_JOBS} when run)"
    )

    return solver, trivial_estimator, shadow_estimator


class BruecknerCallbackTracker:
    """Callback function class to track convergence and update shadow estimator."""

    def __init__(
        self,
        trivial_estimator: TrivialEstimator,
        shadow_estimator: ShadowEstimator,
        n_repetitions: int,
        convergence_threshold: float = 1e-6,
    ):
        """
        Args:
            trivial_estimator: Exact estimator driving Brueckner cycle
            shadow_estimator: Shadow estimator to update alongside
            n_repetitions: Number of shadow estimation runs per iteration
            convergence_threshold: ||c1|| threshold for convergence
        """
        self.trivial_estimator = trivial_estimator
        self.shadow_estimator = shadow_estimator
        self.n_repetitions = n_repetitions
        self.convergence_threshold = convergence_threshold

        # Storage for convergence data
        self.iteration = 0
        self.energies_trivial = []
        self.energies_shadow = []
        self.c0_values_trivial = []
        self.c0_values_shadow = []
        self.c1_norms_trivial = []
        self.c1_norms_shadow = []

        logger.info(f"Initialized BruecknerCallbackTracker with {n_repetitions} shadow repetitions")

    def __call__(self, E_trivial: float, c0_trivial: float, c1_norm_trivial: float) -> bool:
        """
        Callback function called after each Brueckner iteration.

        Args:
            E_trivial: Energy from trivial estimator
            c0_trivial: Reference overlap from trivial estimator
            c1_norm_trivial: ||c1|| from trivial estimator

        Returns:
            True if converged, False otherwise
        """
        self.iteration += 1

        logger.info("=" * 80)
        logger.info(f"Brueckner Iteration {self.iteration} - Post-rotation Analysis")
        logger.info("=" * 80)

        # Store trivial estimator results
        self.energies_trivial.append(E_trivial)
        self.c0_values_trivial.append(np.abs(c0_trivial))
        self.c1_norms_trivial.append(c1_norm_trivial)

        logger.info(f"Trivial Estimator Results:")
        logger.info(f"  Energy: {E_trivial:.10f} Ha")
        logger.info(f"  |c0|: {np.abs(c0_trivial):.10f}")
        logger.info(f"  ||c1||: {c1_norm_trivial:.6e}")

        # Update shadow estimator reference to match trivial estimator
        # Copy the rotated mean-field object from trivial estimator
        logger.debug("Updating shadow estimator reference from trivial estimator")
        self.shadow_estimator.update_reference(self.trivial_estimator.mf)

        # Run shadow estimation multiple times and collect statistics
        logger.info(f"\nRunning {self.n_repetitions} shadow estimation repetitions...")
        shadow_energies = []
        shadow_c0s = []
        shadow_c1_norms = []

        for rep in range(self.n_repetitions):
            logger.debug(f"  Shadow repetition {rep + 1}/{self.n_repetitions}")

            # Clear previous samples and run new estimation
            self.shadow_estimator.clear_sample()

            E_shadow, c0_shadow, c1_shadow, _ = self.shadow_estimator.run(
                n_samples=N_SAMPLES,
                n_k_estimators=N_K_ESTIMATORS,
                n_jobs=N_JOBS,
                use_qulacs=USE_QULACS,
                calc_c1=True
            )

            if isinstance(c1_shadow, tuple):
                c1_norm_shadow = np.sqrt(
                    np.linalg.norm(c1_shadow[0]) ** 2 + np.linalg.norm(c1_shadow[1]) ** 2
                )
            else:
                c1_norm_shadow = np.linalg.norm(c1_shadow)

            shadow_energies.append(E_shadow)
            shadow_c0s.append(np.abs(c0_shadow))
            shadow_c1_norms.append(c1_norm_shadow)

            logger.debug(f"    E={E_shadow:.10f}, |c0|={np.abs(c0_shadow):.6f}, ||c1||={c1_norm_shadow:.6e}")

        # Compute statistics
        E_shadow_mean = np.mean(shadow_energies)
        E_shadow_std = np.std(shadow_energies)
        c0_shadow_mean = np.mean(shadow_c0s)
        c0_shadow_std = np.std(shadow_c0s)
        c1_norm_shadow_mean = np.mean(shadow_c1_norms)
        c1_norm_shadow_std = np.std(shadow_c1_norms)

        self.energies_shadow.append((E_shadow_mean, E_shadow_std))
        self.c0_values_shadow.append((c0_shadow_mean, c0_shadow_std))
        self.c1_norms_shadow.append((c1_norm_shadow_mean, c1_norm_shadow_std))

        logger.info(f"\nShadow Estimator Results (n={self.n_repetitions} reps):")
        logger.info(f"  Energy: {E_shadow_mean:.10f} ± {E_shadow_std:.6e} Ha")
        logger.info(f"  |c0|: {c0_shadow_mean:.10f} ± {c0_shadow_std:.6e}")
        logger.info(f"  ||c1||: {c1_norm_shadow_mean:.6e} ± {c1_norm_shadow_std:.6e}")

        # Compute errors
        energy_error = E_shadow_mean - E_trivial
        c0_error = c0_shadow_mean - np.abs(c0_trivial)
        c1_norm_error = c1_norm_shadow_mean - c1_norm_trivial

        logger.info(f"\nShadow vs Trivial Errors:")
        logger.info(f"  ΔE: {energy_error:.6e} Ha ({energy_error * 27.211:.6e} eV)")
        logger.info(f"  Δ|c0|: {c0_error:.6e}")
        logger.info(f"  Δ||c1||: {c1_norm_error:.6e}")

        # Check convergence
        converged = c1_norm_trivial < self.convergence_threshold
        if converged:
            logger.info(f"\n{'*' * 80}")
            logger.info(f"CONVERGED: ||c1|| = {c1_norm_trivial:.6e} < {self.convergence_threshold:.6e}")
            logger.info(f"{'*' * 80}")

        logger.info("=" * 80 + "\n")

        return converged

    def save_results(self, output_dir: Path):
        """Save convergence data to file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "convergence_data.npz"

        # Convert shadow results to arrays
        energies_shadow_mean = np.array([x[0] for x in self.energies_shadow])
        energies_shadow_std = np.array([x[1] for x in self.energies_shadow])
        c0_shadow_mean = np.array([x[0] for x in self.c0_values_shadow])
        c0_shadow_std = np.array([x[1] for x in self.c0_values_shadow])
        c1_norms_shadow_mean = np.array([x[0] for x in self.c1_norms_shadow])
        c1_norms_shadow_std = np.array([x[1] for x in self.c1_norms_shadow])

        # Create iterations array starting from 0 (initial state)
        # self.iteration is the count of callback calls (starts at 0 for initial, increments in callback)
        n_datapoints = len(self.energies_trivial)
        iterations = np.arange(n_datapoints)

        np.savez(
            output_file,
            iterations=iterations,
            # Trivial estimator
            energies_trivial=np.array(self.energies_trivial),
            c0_values_trivial=np.array(self.c0_values_trivial),
            c1_norms_trivial=np.array(self.c1_norms_trivial),
            # Shadow estimator
            energies_shadow_mean=energies_shadow_mean,
            energies_shadow_std=energies_shadow_std,
            c0_shadow_mean=c0_shadow_mean,
            c0_shadow_std=c0_shadow_std,
            c1_norms_shadow_mean=c1_norms_shadow_mean,
            c1_norms_shadow_std=c1_norms_shadow_std,
            # Metadata
            n_samples=N_SAMPLES,
            n_k_estimators=N_K_ESTIMATORS,
            n_repetitions=self.n_repetitions,
        )

        logger.info(f"Saved convergence data to {output_file}")

    def print_summary(self):
        """Print a summary of the convergence behavior."""
        logger.info("\n" + "=" * 80)
        logger.info("CONVERGENCE SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Total iterations: {self.iteration}")

        if len(self.energies_trivial) > 0:
            E_initial = self.energies_trivial[0]
            E_final = self.energies_trivial[-1]
            c1_initial = self.c1_norms_trivial[0]
            c1_final = self.c1_norms_trivial[-1]

            logger.info(f"\nTrivial Estimator:")
            logger.info(f"  Initial energy: {E_initial:.10f} Ha")
            logger.info(f"  Final energy: {E_final:.10f} Ha")
            logger.info(f"  Energy change: {E_final - E_initial:.6e} Ha")
            logger.info(f"  Initial ||c1||: {c1_initial:.6e}")
            logger.info(f"  Final ||c1||: {c1_final:.6e}")
            logger.info(f"  ||c1|| reduction: {(c1_initial - c1_final) / c1_initial * 100:.2f}%")

        if len(self.energies_shadow) > 0:
            E_shadow_initial_mean = self.energies_shadow[0][0]
            E_shadow_final_mean = self.energies_shadow[-1][0]
            E_shadow_final_std = self.energies_shadow[-1][1]
            c1_shadow_initial_mean = self.c1_norms_shadow[0][0]
            c1_shadow_final_mean = self.c1_norms_shadow[-1][0]
            c1_shadow_final_std = self.c1_norms_shadow[-1][1]

            logger.info(f"\nShadow Estimator (final iteration, n={self.n_repetitions} reps):")
            logger.info(f"  Initial energy: {E_shadow_initial_mean:.10f} Ha")
            logger.info(f"  Final energy: {E_shadow_final_mean:.10f} ± {E_shadow_final_std:.6e} Ha")
            logger.info(f"  Energy change: {E_shadow_final_mean - E_shadow_initial_mean:.6e} Ha")
            logger.info(f"  Initial ||c1||: {c1_shadow_initial_mean:.6e}")
            logger.info(f"  Final ||c1||: {c1_shadow_final_mean:.6e} ± {c1_shadow_final_std:.6e}")

            # Final error comparison
            final_energy_error = E_shadow_final_mean - E_final
            logger.info(f"\nFinal Shadow vs Trivial Error:")
            logger.info(f"  ΔE: {final_energy_error:.6e} Ha ({final_energy_error * 27.211:.6e} eV)")
            logger.info(f"  ΔE/σ: {final_energy_error / E_shadow_final_std:.2f}")

        logger.info("=" * 80 + "\n")


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Main execution function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BRUECKNER CONVERGENCE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Molecule: H{N_ATOMS} chain, R={BOND_LENGTH} Å, basis={BASIS}, spin={SPIN}")
    logger.info(f"Shadow: N={N_SAMPLES}, K={N_K_ESTIMATORS}, repetitions={N_SHOTS_REPETITIONS}")
    logger.info(f"Brueckner: max_iter={MAX_BRUECKNER_ITER}, damping={DAMPING}, DIIS={USE_DIIS}")
    logger.info(f"Convergence threshold: ||c1|| < {CONVERGENCE_THRESHOLD:.6e}")
    logger.info("=" * 80 + "\n")

    # Setup molecule and mean-field
    mol, mf = setup_molecule_and_mf(N_ATOMS, BOND_LENGTH, BASIS, SPIN)

    # Setup estimators
    solver, trivial_estimator, shadow_estimator = setup_estimators(mf, verbose=VERBOSE)

    # Get exact FCI energy for reference
    logger.info("Computing exact FCI energy...")
    _, fci_energy = solver.solve()
    logger.info(f"Exact FCI energy: {fci_energy:.10f} Ha\n")

    # Create callback tracker
    callback = BruecknerCallbackTracker(
        trivial_estimator=trivial_estimator,
        shadow_estimator=shadow_estimator,
        n_repetitions=N_SHOTS_REPETITIONS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
    )

    # Run initial estimation on unrotated (HF) orbitals
    logger.info("=" * 80)
    logger.info("INITIAL STATE ANALYSIS (before Brueckner optimization)")
    logger.info("=" * 80)

    # Get trivial estimator results for initial state
    logger.info("Running TrivialEstimator on initial HF orbitals...")
    E_trivial_init, c0_trivial_init, c1_trivial_init, _ = trivial_estimator.run(calc_c1=True)

    if isinstance(c1_trivial_init, tuple):
        c1_norm_trivial_init = np.sqrt(
            np.linalg.norm(c1_trivial_init[0]) ** 2 + np.linalg.norm(c1_trivial_init[1]) ** 2
        )
    else:
        c1_norm_trivial_init = np.linalg.norm(c1_trivial_init)

    logger.info(f"Initial HF State - Trivial Estimator:")
    logger.info(f"  Energy: {E_trivial_init:.10f} Ha")
    logger.info(f"  |c0|: {np.abs(c0_trivial_init):.10f}")
    logger.info(f"  ||c1||: {c1_norm_trivial_init:.6e}")

    # Store initial trivial results
    callback.energies_trivial.append(E_trivial_init)
    callback.c0_values_trivial.append(np.abs(c0_trivial_init))
    callback.c1_norms_trivial.append(c1_norm_trivial_init)

    # Run shadow estimator on initial state
    logger.info(f"\nRunning {N_SHOTS_REPETITIONS} shadow estimation repetitions on initial HF state...")
    shadow_energies_init = []
    shadow_c0s_init = []
    shadow_c1_norms_init = []

    for rep in range(N_SHOTS_REPETITIONS):
        logger.debug(f"  Shadow repetition {rep + 1}/{N_SHOTS_REPETITIONS}")

        shadow_estimator.clear_sample()

        E_shadow, c0_shadow, c1_shadow, _ = shadow_estimator.run(
            n_samples=N_SAMPLES,
            n_k_estimators=N_K_ESTIMATORS,
            n_jobs=N_JOBS,
            use_qulacs=USE_QULACS,
            calc_c1=True
        )

        if isinstance(c1_shadow, tuple):
            c1_norm_shadow = np.sqrt(
                np.linalg.norm(c1_shadow[0]) ** 2 + np.linalg.norm(c1_shadow[1]) ** 2
            )
        else:
            c1_norm_shadow = np.linalg.norm(c1_shadow)

        shadow_energies_init.append(E_shadow)
        shadow_c0s_init.append(np.abs(c0_shadow))
        shadow_c1_norms_init.append(c1_norm_shadow)

    # Store initial shadow results
    E_shadow_mean_init = np.mean(shadow_energies_init)
    E_shadow_std_init = np.std(shadow_energies_init)
    c0_shadow_mean_init = np.mean(shadow_c0s_init)
    c0_shadow_std_init = np.std(shadow_c0s_init)
    c1_norm_shadow_mean_init = np.mean(shadow_c1_norms_init)
    c1_norm_shadow_std_init = np.std(shadow_c1_norms_init)

    callback.energies_shadow.append((E_shadow_mean_init, E_shadow_std_init))
    callback.c0_values_shadow.append((c0_shadow_mean_init, c0_shadow_std_init))
    callback.c1_norms_shadow.append((c1_norm_shadow_mean_init, c1_norm_shadow_std_init))
    callback.iteration = 0  # Mark this as iteration 0

    logger.info(f"\nInitial HF State - Shadow Estimator (n={N_SHOTS_REPETITIONS} reps):")
    logger.info(f"  Energy: {E_shadow_mean_init:.10f} ± {E_shadow_std_init:.6e} Ha")
    logger.info(f"  |c0|: {c0_shadow_mean_init:.10f} ± {c0_shadow_std_init:.6e}")
    logger.info(f"  ||c1||: {c1_norm_shadow_mean_init:.6e} ± {c1_norm_shadow_std_init:.6e}")

    energy_error_init = E_shadow_mean_init - E_trivial_init
    logger.info(f"\nInitial Shadow vs Trivial Error:")
    logger.info(f"  ΔE: {energy_error_init:.6e} Ha ({energy_error_init * 27.211:.6e} eV)")
    logger.info("=" * 80 + "\n")

    # Run Brueckner cycle with trivial estimator driving the optimization
    logger.info("Starting Brueckner orbital optimization...")
    logger.info(f"Driver: TrivialEstimator (exact c1 amplitudes)")
    logger.info(f"Follower: ShadowEstimator (updated each iteration)\n")

    brueckner_cycle(
        mf=mf,
        estimator=trivial_estimator,
        canonicalise=True,
        damping=DAMPING,
        use_diis=USE_DIIS,
        max_iter=MAX_BRUECKNER_ITER,
        callback_fn=callback,
        verbose=VERBOSE,
    )

    # Print summary and save results
    callback.print_summary()
    callback.save_results(OUTPUT_DIR)

    logger.info(f"Results saved to {OUTPUT_DIR}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()

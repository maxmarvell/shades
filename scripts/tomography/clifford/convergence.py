"""Shadow Shot Budget Convergence via Clifford Shadow Tomography.

This script studies how the energy estimate converges as the number of
Clifford shadow samples increases, for a fixed hydrogen chain geometry.
The results can be used to verify the expected Var ~ 1/N scaling.

The workflow:
1. Build a fixed hydrogen chain with specified geometry
2. Compute Hartree-Fock reference and exact FCI energy
3. For each shadow shot budget N_s:
   - Perform multiple independent shadow estimations
   - Record energies
4. Save results to disk for further analysis
"""

import os
import json
from datetime import datetime
from pyscf import gto, scf
from shades.estimators import ShadowEstimator, ExactEstimator
from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain
import numpy as np
import logging

N_HYDROGEN = 6
INTERATOMIC_DISTANCE = 1.50
BASIS_SET = "sto-3g"

SHADOW_SAMPLES = [500, 1000, 2000, 4000, 8000, 16000]
N_K_ESTIMATORS = 20
N_SIMULATIONS = 10
N_JOBS = 8

RUN_COMMENT = "Convergence study: energy variance vs Clifford shadow shot budget."
OUTPUT_DIR = f"./results/tomography/clifford/convergence/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
      force=True,
)
logger = logging.getLogger(__name__)

def main():
    """Run shadow budget convergence analysis."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_budgets = len(SHADOW_SAMPLES)

    estimated_energies = np.empty((n_budgets, N_SIMULATIONS))

    logger.info("=" * 80)
    logger.info("Shadow Budget Convergence Study (Clifford)")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Number of H atoms:     {N_HYDROGEN}")
    logger.info(f"  - Interatomic distance:  {INTERATOMIC_DISTANCE:.2f} Å")
    logger.info(f"  - Basis set:             {BASIS_SET}")
    logger.info(f"  - K-estimators:          {N_K_ESTIMATORS}")
    logger.info(f"  - Shadow budgets:        {SHADOW_SAMPLES}")
    logger.info(f"  - Independent runs:      {N_SIMULATIONS}")
    logger.info("=" * 80)

    mol_string = make_hydrogen_chain(N_HYDROGEN, INTERATOMIC_DISTANCE)
    mol = gto.Mole()
    mol.build(atom=mol_string, basis=BASIS_SET, verbose=0)

    mf = scf.RHF(mol)
    mf.run()

    fci_solver = FCISolver(mf)
    estimator = ShadowEstimator(mf, solver=fci_solver, verbose=4)
    exact_estimator = ExactEstimator(mf, solver=fci_solver)

    fci_energy = estimator.E_exact
    hf_energy = estimator.E_hf
    exact_c0 = exact_estimator.estimate_c0()
    logger.info(f"  HF Energy:  {hf_energy:.8f} Ha")
    logger.info(f"  FCI Energy: {fci_energy:.8f} Ha")
    logger.info(f"  Exact c0:   {exact_c0:.8f}")

    for j, n_shadows in enumerate(SHADOW_SAMPLES):
        logger.info(f"[{j+1}/{n_budgets}] Shadow budget N_s = {n_shadows}")

        energies = np.empty(N_SIMULATIONS)
        c0_values = np.empty(N_SIMULATIONS)

        for i in range(N_SIMULATIONS):
            energies[i], c0, _, _ = estimator.run(
                n_samples=n_shadows,
                n_k_estimators=N_K_ESTIMATORS,
                n_jobs=N_JOBS,
            )
            c0_values[i] = c0
            estimator.clear_sample()

            logger.info(
                f"    Run {i+1}/{N_SIMULATIONS}: E = {energies[i]:.8f}, "
                f"c0 = {c0:.6f} (exact = {exact_c0:.6f}, err = {c0 - exact_c0:+.4e})"
            )

        estimated_energies[j] = energies

        mean_e = np.mean(energies)
        std_e = np.std(energies)
        mean_c0 = np.mean(c0_values)
        std_c0 = np.std(c0_values)
        error = mean_e - fci_energy
        logger.info(f"  Shadow Mean:  {mean_e:.8f} Ha")
        logger.info(f"  Shadow Std:   {std_e:.8f} Ha")
        logger.info(f"  Mean Error:   {error:+.2e} Ha")
        logger.info(f"  c0 mean:      {mean_c0:.6f} (exact: {exact_c0:.6f}, bias: {mean_c0 - exact_c0:+.4e})")
        logger.info(f"  c0 std:       {std_c0:.6f}")

    results = {
        'energy': estimated_energies,
        'fci_energy': np.array(fci_energy),
        'hf_energy': np.array(hf_energy),
    }

    npz_path = os.path.join(OUTPUT_DIR, "data.npz")

    metadata = {
        "system": "H chain",
        "interatomic_distance_angstrom": INTERATOMIC_DISTANCE,
        "basis_set": str(BASIS_SET),
        "n_runs": N_SIMULATIONS,
        "n_hydrogen": N_HYDROGEN,
        "n_shadow_samples": SHADOW_SAMPLES,
        "n_k_estimators": N_K_ESTIMATORS,
        "n_jobs": N_JOBS,
        "comments": RUN_COMMENT,
    }

    np.savez_compressed(npz_path, **results)
    logger.info(f"Saved: {npz_path}")

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()

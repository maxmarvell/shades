"""Shadow Shot Budget Convergence via Matchgate Shadow Tomography.

This script studies how the energy estimate converges as the number of
matchgate shadow samples increases, for a fixed hydrogen chain geometry.
The results can be used to verify the expected Var ∝ 1/N scaling.

The workflow:
1. Build a fixed hydrogen chain with specified geometry
2. Compute Hartree-Fock reference and exact FCI energy
3. For each shadow shot budget N_s:
   - Perform multiple independent shadow estimations
   - Record energies and RDMs
4. Save results to disk for further analysis
"""

import os
import json
from datetime import datetime
from pyscf import gto, scf
from shades.estimators import MatchgateEstimator
from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain
import numpy as np
import logging

N_HYDROGEN = 4
INTERATOMIC_DISTANCE = 1.00
BASIS_SET = "sto-3g"

SHADOW_SAMPLES = [100, 200, 500, 1000, 2000, 5000, 10000]
N_SIMULATIONS = 100

RUN_COMMENT = "Convergence study: energy variance vs shadow shot budget."
OUTPUT_DIR = f"./results/tomography/matchgate/convergence/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

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

    norb = N_HYDROGEN
    n_budgets = len(SHADOW_SAMPLES)

    estimated_energies = np.empty((n_budgets, N_SIMULATIONS))
    estimated_rdm1 = np.empty((n_budgets, N_SIMULATIONS, norb, norb))
    estimated_rdm2 = np.empty((n_budgets, N_SIMULATIONS, norb, norb, norb, norb))

    logger.info("=" * 80)
    logger.info("Shadow Budget Convergence Study")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Number of H atoms:     {N_HYDROGEN}")
    logger.info(f"  - Interatomic distance:  {INTERATOMIC_DISTANCE:.2f} Å")
    logger.info(f"  - Basis set:             {BASIS_SET}")
    logger.info(f"  - Shadow budgets:        {SHADOW_SAMPLES}")
    logger.info(f"  - Independent runs:      {N_SIMULATIONS}")
    logger.info("=" * 80)

    mol_string = make_hydrogen_chain(N_HYDROGEN, INTERATOMIC_DISTANCE)
    mol = gto.Mole()
    mol.build(atom=mol_string, basis=BASIS_SET, verbose=0)

    mf = scf.RHF(mol)
    mf.run()

    fci_solver = FCISolver(mf)
    estimator = MatchgateEstimator(mf, solver=fci_solver, verbose=4)

    fci_energy = estimator.E_exact
    hf_energy = estimator.E_hf
    logger.info(f"  HF Energy:  {hf_energy:.8f} Ha")
    logger.info(f"  FCI Energy: {fci_energy:.8f} Ha")

    for j, n_shadows in enumerate(SHADOW_SAMPLES):
        logger.info(f"[{j+1}/{n_budgets}] Shadow budget N_s = {n_shadows}")

        energies = np.empty(N_SIMULATIONS)
        rdm1 = np.empty((N_SIMULATIONS, norb, norb))
        rdm2 = np.empty((N_SIMULATIONS, norb, norb, norb, norb))

        for i in range(N_SIMULATIONS):
            energies[i], rdm1[i], rdm2[i], _ = estimator.run(n_samples=n_shadows)

            if (i + 1) % 10 == 0:
                logger.info(f"    Completed {i+1}/{N_SIMULATIONS} runs")

        estimated_energies[j] = energies
        estimated_rdm1[j] = rdm1
        estimated_rdm2[j] = rdm2

        mean_e = np.mean(energies)
        std_e = np.std(energies)
        error = mean_e - fci_energy
        logger.info(f"  Shadow Mean:  {mean_e:.8f} Ha")
        logger.info(f"  Shadow Std:   {std_e:.8f} Ha")
        logger.info(f"  Mean Error:   {error:+.2e} Ha")

    results = {
        'energy': estimated_energies,
        'rdm1': estimated_rdm1,
        'rdm2': estimated_rdm2,
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

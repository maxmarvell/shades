"""Variance Scaling with System Size via Matchgate Shadow Tomography.

This script studies how the energy estimator variance scales with the
number of orbitals (system size) at a fixed shadow sample budget and
bond length. It runs H2, H4, H6, H8, H10 chains to check whether
variance follows the expected Var ∝ n^2 / N_s scaling law.

The workflow:
1. For each system size (n_hydrogen = 2, 4, 6, 8, 10):
   - Build hydrogen chain at fixed bond length
   - Compute Hartree-Fock reference and exact FCI energy
   - Perform multiple independent shadow estimations
   - Record energies and RDMs
2. Save results to disk for further analysis
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

SYSTEM_SIZES = [2, 4, 6, 8]
INTERATOMIC_DISTANCE = 1.00
BASIS_SET = "sto-3g"

N_SHADOWS = 10000
N_SIMULATIONS = 10

RUN_COMMENT = "System size scaling study: energy variance vs number of orbitals."
OUTPUT_DIR = f"./results/tomography/matchgate/system_size/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
      force=True,
)
logger = logging.getLogger(__name__)

def main():
    """Run system size scaling analysis."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_sizes = len(SYSTEM_SIZES)

    fci_energies = np.empty(n_sizes)
    hf_energies = np.empty(n_sizes)

    logger.info("=" * 80)
    logger.info("System Size Scaling Study")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - System sizes (H atoms): {SYSTEM_SIZES}")
    logger.info(f"  - Interatomic distance:   {INTERATOMIC_DISTANCE:.2f} Å")
    logger.info(f"  - Basis set:              {BASIS_SET}")
    logger.info(f"  - Shadow samples:         {N_SHADOWS}")
    logger.info(f"  - Independent runs:       {N_SIMULATIONS}")
    logger.info("=" * 80)

    all_energies = {}

    for j, n_hydrogen in enumerate(SYSTEM_SIZES):
        logger.info(f"[{j+1}/{n_sizes}] H{n_hydrogen} ({n_hydrogen} orbitals)")

        mol_string = make_hydrogen_chain(n_hydrogen, INTERATOMIC_DISTANCE)
        mol = gto.Mole()
        mol.build(atom=mol_string, basis=BASIS_SET, verbose=0)

        mf = scf.RHF(mol)
        mf.run()
 
        fci_solver = FCISolver(mf)
        estimator = MatchgateEstimator(mf, solver=fci_solver, verbose=4)

        fci_energies[j] = estimator.E_exact
        hf_energies[j] = estimator.E_hf
        logger.info(f"  HF Energy:  {hf_energies[j]:.8f} Ha")
        logger.info(f"  FCI Energy: {fci_energies[j]:.8f} Ha")

        energies = np.empty(N_SIMULATIONS)

        logger.info(f"  Running {N_SIMULATIONS} shadow estimations...")
        for i in range(N_SIMULATIONS):
            energies[i], _, _, _ = estimator.run(n_samples=N_SHADOWS)

            if (i + 1) % 10 == 0:
                logger.info(f"    Completed {i+1}/{N_SIMULATIONS} runs")

        all_energies[f"energy_H{n_hydrogen}"] = energies

        mean_e = np.mean(energies)
        std_e = np.std(energies)
        error = mean_e - fci_energies[j]
        logger.info(f"  Shadow Mean:  {mean_e:.8f} Ha")
        logger.info(f"  Shadow Std:   {std_e:.8f} Ha")
        logger.info(f"  Mean Error:   {error:+.2e} Ha")

    results = {
        **all_energies,
        'fci_energy': fci_energies,
        'hf_energy': hf_energies,
    }

    npz_path = os.path.join(OUTPUT_DIR, "data.npz")

    metadata = {
        "system": "H chain",
        "interatomic_distance_angstrom": INTERATOMIC_DISTANCE,
        "basis_set": str(BASIS_SET),
        "n_runs": N_SIMULATIONS,
        "system_sizes": SYSTEM_SIZES,
        "n_shadow_samples": N_SHADOWS,
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

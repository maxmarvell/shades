"""H2 Potential Energy Surface via Shadow Tomography.

This script computes the potential energy surface (PES) of an n hydrogen chain (nH)
as a function of internuclear distance using matchgate shadow state tomography. The results
are compared against exact Full Configuration Interaction (FCI) calculations to
demonstrate the accuracy and statistical properties of the shadow tomography method.

The workflow:
1. For each interatomic distance:
   - Build H2 molecule with specified geometry
   - Compute Hartree-Fock reference and molecular Hamiltonian
   - Run exact FCI calculation for ground truth
   - Perform multiple shadow tomography estimations
2. Compute mean and standard deviation of shadow estimates
3. Plot PES comparing shadow tomography vs exact FCI
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
INTERATOMIC_DISTANCES = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
BASIS_SET = "sto-3g"

N_SHADOWS = 1000
N_SIMULATIONS = 100

RUN_COMMENT = "Preliminary run to verify correctness of matchgate shadow sampling."
OUTPUT_DIR = f"./results/tomography/matchgate/system_size/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
      force=True,
)
# logging.getLogger("shades").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Run H streching analysis and generate PES plot."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exact_fci = np.empty_like(INTERATOMIC_DISTANCES)
    exact_hf = np.empty_like(INTERATOMIC_DISTANCES)
    estimated_mean = np.empty_like(INTERATOMIC_DISTANCES)
    estimated_std = np.empty_like(INTERATOMIC_DISTANCES)

    estimated_energies = np.empty((len(INTERATOMIC_DISTANCES), N_SIMULATIONS))

    norb = N_HYDROGEN
    estimated_rdm1 = np.empty((len(INTERATOMIC_DISTANCES), N_SIMULATIONS, norb, norb))
    estimated_rdm2 = np.empty((len(INTERATOMIC_DISTANCES), N_SIMULATIONS, norb, norb, norb, norb))

    logger.info("=" * 80)
    logger.info("H2 Potential Energy Surface via Shadow Tomography")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Samples per run:       {N_SHADOWS}")
    logger.info(f"  - Independent runs:      {N_SIMULATIONS}")
    logger.info(f"  - Number of H atoms:     {N_HYDROGEN}")
    logger.info(f"  - Distances (Å):         {INTERATOMIC_DISTANCES}")
    logger.info("=" * 80)

    for j, d in enumerate(INTERATOMIC_DISTANCES):
        logger.info(f"[{j+1}/{len(INTERATOMIC_DISTANCES)}] Processing distance = {d:.2f} Å")

        energies = np.empty(N_SIMULATIONS)
        rdm1 = np.empty((N_SIMULATIONS, norb, norb))
        rdm2 = np.empty((N_SIMULATIONS, norb, norb, norb, norb))

        mol_string = make_hydrogen_chain(N_HYDROGEN, d)
        mol = gto.Mole()
        mol.build(atom=mol_string, basis=BASIS_SET, verbose=0)

        mf = scf.RHF(mol)
        mf.run()

        fci_solver = FCISolver(mf)
        estimator = MatchgateEstimator(mf, solver=fci_solver, verbose=4)
        logger.info(f"  FCI Energy (exact): {estimator.E_exact:.8f} Ha")

        logger.info(f"  Running {N_SIMULATIONS} shadow estimations...")
        for i in range(N_SIMULATIONS):
            energies[i], rdm1[i], rdm2[i], _ = estimator.run(n_samples=N_SHADOWS)

            if (i + 1) % 1 == 0:
                logger.info(f"    Completed {i+1}/{N_SIMULATIONS} runs")

        estimated_energies[j] = energies
        estimated_rdm1[j] = rdm1
        estimated_rdm2[j] = rdm2

        exact_fci[j] = estimator.E_exact
        exact_hf[j] = estimator.E_hf
        estimated_mean[j] = np.mean(energies)
        estimated_std[j] = np.std(energies)

        error = estimated_mean[j] - exact_fci[j]
        logger.info(f"  Shadow Mean:  {estimated_mean[j]:.8f} Ha")
        logger.info(f"  Shadow Std:   {estimated_std[j]:.8f} Ha")
        logger.info(f"  Mean Error:   {error:+.2e} Ha")

    results = {
        'energy': estimated_energies,
        'rdm1': estimated_rdm1,
        'rdm2': estimated_rdm2,
        'fci_energy': exact_fci,
        'hf_energy': exact_hf,
        "interatomic_distances": np.array(INTERATOMIC_DISTANCES)
    }

    npz_path = os.path.join(OUTPUT_DIR, "data.npz")

    metadata = {
        "system": "H chain",
        "interatomic_distances_angstrom": INTERATOMIC_DISTANCES,
        "basis_set": str(BASIS_SET),
        "n_runs": N_SIMULATIONS,
        "n_hydrogen": N_HYDROGEN,
        "n_shadow_samples": N_SHADOWS,
        "comments": RUN_COMMENT
    }

    np.savez_compressed(npz_path, **results)
    logger.info(f"Saved: {npz_path}")

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
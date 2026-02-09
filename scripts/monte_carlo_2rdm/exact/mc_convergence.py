"""Monte Carlo RDM2 Convergence Analysis.

This script demonstrates the convergence of Monte Carlo RDM2 estimation
as the number of MC iterations increases. The MC estimator computes
a correlation-focused 2-RDM.

Two convergence metrics are tracked:
1. RDM2 Frobenius norm distance to a high-iteration reference
2. Representative matrix element convergence
"""

import numpy as np
import os
import json
from datetime import datetime
import logging
from pyscf import gto, scf
from pyscf.fci import direct_spin1

from shades.solvers import FCISolver
from shades.estimators import ExactEstimator
from shades.utils import make_hydrogen_chain
from shades.monte_carlo import MPSSampler, MonteCarloEstimator

from utils import spinorb_to_spatial_chem, doubles_energy, total_energy_from_rdm12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

RUN_COMMENT = "Check that using exact coefficients the Monte Carlo sampler yields a valid 2-RDM sampling directly from a trial MPS state."

DEFAULT_OUTPUT_DIR = f"./results/rdm2_convergence/exact/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 20
N_MC_ITERS = [500, 1000, 1500, 2500, 5000]

N_HYDROGEN = 8
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

    estimator = ExactEstimator(mf, fci_solver)
    sampler = MPSSampler(mf)

    results = {
        'rdm2': np.empty((len(N_MC_ITERS), N_RUNS, norb, norb, norb, norb), dtype=np.float32),
        'E_tot': np.empty((len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'E_doubles': np.empty((len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'rel_err_E2': np.empty((len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'rel_frob_rdm2': np.empty((len(N_MC_ITERS), N_RUNS), dtype=np.float64),
        'max_abs_rdm2': np.empty((len(N_MC_ITERS), N_RUNS), dtype=np.float64),
    }

    print("\n" + "=" * 70)
    print("Running Monte Carlo Convergence Study")
    print("=" * 70)
    print(f"Iteration schedule: {N_MC_ITERS}")
    print(f"Runs per iteration count: {N_RUNS}")


    for i, n_iters in enumerate(N_MC_ITERS):
        print(f"MC Iterations: {n_iters:,}")

        for j, run in enumerate(range(N_RUNS)):
            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)

            mc = MonteCarloEstimator(estimator, sampler)
            rdm2_mc = mc.estimate_2rdm(max_iters=n_iters)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)

            E_doubles = doubles_energy(rdm2, mf)
            rel_err = np.abs(E_double_ref - E_doubles) / np.abs(E_double_ref)

            diff = rdm2 - rdm2_ref
            frob = np.linalg.norm(diff)
            rel_frob = frob / np.linalg.norm(rdm2_ref)
            max_abs = np.max(np.abs(diff))

            results['rdm2'][i, j] = rdm2.astype(np.float32)
            results['E_doubles'][i, j] = E_doubles
            results['rel_err_E2'][i, j] = rel_err
            results['rel_frob_rdm2'][i, j] = rel_frob
            results['max_abs_rdm2'][i, j] = max_abs

            E = total_energy_from_rdm12(rdm1, rdm2, mf)
            print(f"E: {E}, E_doubles = {E_doubles}, rel_err = {rel_err} ||dRDM2||_F = {rel_frob:.4e}")

            results['E_tot'][i, j] = E

    npz_path = os.path.join(output_dir, "data.npz")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_mc_iters": N_MC_ITERS,
        "E_hf_hartree": float(E_hf),
        "E_fci_hartree": float(E_fci),
        "E_corr_hartree": float(E_fci - E_hf),
        "run_comment": RUN_COMMENT
    }

    np.savez_compressed(npz_path, **results)
    print(f"Saved: {npz_path}")

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    npz_path = os.path.join(output_dir, "data.npz")
    results = np.load(npz_path, allow_pickle=True)

if __name__ == "__main__":
    main()

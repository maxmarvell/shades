"""Monte Carlo RDM2 System Size Scaling Analysis (Exact Estimator).

Uses the exact estimator (direct statevector overlaps) with an MPS surrogate
sampler to isolate the behaviour of the MC method itself from shadow noise.
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
from shades.estimators import ExactEstimator
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

RUN_COMMENT = "Exact estimator + MPS sampler: isolate MC variance from shadow noise."

DEFAULT_OUTPUT_DIR = f"./results/rdm2_scaling/exact/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

# Fixed parameters
N_RUNS = 10
N_MC_ITERS = 10000

# System size sweep
N_HYDROGEN = [2, 4, 6, 8]
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"


def main():
    print("=" * 70)
    print("Monte Carlo RDM2 System Size Scaling (Exact Estimator)")
    print("=" * 70)

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    print(f"\nSystem sizes (N_H): {N_HYDROGEN}")
    print(f"MC iterations: {N_MC_ITERS}")
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
        rdm1, rdm2_ref = direct_spin1.make_rdm12(
            fci_solver.civec, norb, nelec
        )
        E_double_ref = doubles_energy(rdm2_ref, mf)

        print(f"Basis set: {BASIS_SET}")
        print(f"Number of orbitals: {norb}")
        print(f"Hartree-Fock Energy:      {E_hf:.10f} Ha")
        print(f"Exact FCI Energy:         {E_fci:.10f} Ha")
        print(f"Correlation Energy:       {E_fci - E_hf:.10f} Ha")

        reference_data[n_h] = {
            'E_hf': E_hf,
            'E_fci': E_fci,
            'E_corr': E_fci - E_hf,
            'E_double_ref': E_double_ref,
            'norb': norb,
        }

        results = {
            'E_tot': np.empty(N_RUNS, dtype=np.float64),
            'E_doubles': np.empty(N_RUNS, dtype=np.float64),
            'rel_err_E2': np.empty(N_RUNS, dtype=np.float64),
            'rel_frob_rdm2': np.empty(N_RUNS, dtype=np.float64),
            'max_abs_rdm2': np.empty(N_RUNS, dtype=np.float64),
        }

        estimator = ExactEstimator(mf, fci_solver)
        sampler = MPSSampler(mf)

        for j in range(N_RUNS):
            print(f"  Run {j+1}/{N_RUNS}...", end=" ", flush=True)

            mc = MonteCarloEstimator(estimator, sampler)
            rdm2_mc = mc.estimate_2rdm(max_iters=N_MC_ITERS)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)

            E_doubles = doubles_energy(rdm2, mf)
            rel_err = np.abs(E_double_ref - E_doubles) / np.abs(E_double_ref)

            diff = rdm2 - rdm2_ref
            frob = np.linalg.norm(diff)
            rel_frob = frob / np.linalg.norm(rdm2_ref)
            max_abs = np.max(np.abs(diff))

            E = total_energy_from_rdm12(rdm1, rdm2, mf)
            print(f"E: {E:.6f}, E_doubles = {E_doubles:.6f}, rel_err = {rel_err:.4e}, ||dRDM2||_F = {rel_frob:.4e}")

            results['E_tot'][j] = E
            results['E_doubles'][j] = E_doubles
            results['rel_err_E2'][j] = rel_err
            results['rel_frob_rdm2'][j] = rel_frob
            results['max_abs_rdm2'][j] = max_abs

        all_results[n_h] = results

    # Save results
    npz_path = os.path.join(output_dir, "data.npz")

    npz_data = {}
    for n_h in N_HYDROGEN:
        for key, arr in all_results[n_h].items():
            npz_data[f"H{n_h}_{key}"] = arr

    metadata = {
        "system": "H chain",
        "estimator": "exact",
        "sampler": "MPS (DMRG)",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_hydrogen": [int(x) for x in N_HYDROGEN],
        "n_mc_iters": N_MC_ITERS,
        "comments": RUN_COMMENT,
        "reference_data": {str(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in reference_data.items()},
    }

    np.savez_compressed(npz_path, **npz_data)
    print(f"\nSaved: {npz_path}")

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    n_h_arr = np.array(N_HYDROGEN)

    rel_frob_mean = np.array([all_results[n_h]['rel_frob_rdm2'].mean() for n_h in N_HYDROGEN])
    rel_frob_std = np.array([all_results[n_h]['rel_frob_rdm2'].std(ddof=1) for n_h in N_HYDROGEN])
    rel_frob_sem = rel_frob_std / np.sqrt(N_RUNS)

    rel_E2_mean = np.array([all_results[n_h]['rel_err_E2'].mean() for n_h in N_HYDROGEN])
    rel_E2_std = np.array([all_results[n_h]['rel_err_E2'].std(ddof=1) for n_h in N_HYDROGEN])
    rel_E2_sem = rel_E2_std / np.sqrt(N_RUNS)

    E_doubles_mean = np.array([all_results[n_h]['E_doubles'].mean() for n_h in N_HYDROGEN])
    E_doubles_std = np.array([all_results[n_h]['E_doubles'].std(ddof=1) for n_h in N_HYDROGEN])
    E_doubles_sem = E_doubles_std / np.sqrt(N_RUNS)

    E_double_refs = np.array([reference_data[n_h]['E_double_ref'] for n_h in N_HYDROGEN])

    setup_plotting_style()
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.errorbar(
        n_h_arr, rel_frob_mean, yerr=rel_frob_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$\|\Delta\Gamma\|_F / \|\Gamma_{\mathrm{ref}}\|_F$',
        linewidth=2, markersize=8,
    )
    ax1.set_xlabel('Number of Hydrogen Atoms')
    ax1.set_ylabel('Relative Frobenius Error')
    ax1.set_yscale('log')
    ax1.set_title('RDM2 Error vs System Size (Exact)')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.errorbar(
        n_h_arr, rel_E2_mean, yerr=rel_E2_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$|E_{2,\mathrm{MC}}-E_{2,\mathrm{ref}}|/|E_{2,\mathrm{ref}}|$',
        linewidth=2, markersize=8,
    )
    ax2.set_xlabel('Number of Hydrogen Atoms')
    ax2.set_ylabel(r'Relative $E_2$ Error')
    ax2.set_yscale('log')
    ax2.set_title(r'Two-electron Energy Error vs System Size (Exact)')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')

    ax3.errorbar(
        n_h_arr, E_doubles_mean, yerr=E_doubles_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$E_2^{\mathrm{MC}}$',
        linewidth=2, markersize=8,
    )
    ax3.plot(n_h_arr, E_double_refs, 's--', color='r', linewidth=2, markersize=8, label=r'$E_2^{\mathrm{ref}}$')
    ax3.set_xlabel('Number of Hydrogen Atoms')
    ax3.set_ylabel(r'$E_2$ (Hartree)')
    ax3.set_title('Two-electron Energy vs System Size (Exact)')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'system_size_scaling_exact.pdf')
    png_path = os.path.join(output_dir, 'system_size_scaling_exact.png')
    svg_path = os.path.join(output_dir, 'system_size_scaling_exact.svg')
    save_figure(pdf_path)
    save_figure(png_path, dpi=300)
    save_figure(svg_path)

    plt.show()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

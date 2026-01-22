"""Monte Carlo RDM2 System Size Scaling Analysis."""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent block2 segfault with libomp

import numpy as np
import matplotlib.pyplot as plt
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

RUN_COMMENT = "First test run to observe scaling with system size."

DEFAULT_OUTPUT_DIR = f"./results/rdm2_scaling/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

# Fixed parameters
N_RUNS = 10
N_MC_ITERS = 10000
N_SHADOWS = 10000
N_K_ESTIMATORS = 40

# System size sweep
N_HYDROGEN = [4, 6, 8]
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"

FIGURE_SIZE = (10, 4)
PLOT_DPI = 300


def _spinorb_to_spatial_chem(
    rdm2_so: np.ndarray,
    norb: int
) -> np.ndarray:

    rdm2_aa = rdm2_so[:norb, :norb, :norb, :norb]
    rdm2_ab = rdm2_so[:norb, norb:, :norb, norb:]
    rdm2_bb = rdm2_so[norb:, norb:, norb:, norb:]

    dm2aa = rdm2_aa.transpose(0, 2, 1, 3)  # (p,q,r,s) -> (p,r,q,s)
    dm2ab = rdm2_ab.transpose(0, 2, 1, 3)
    dm2bb = rdm2_bb.transpose(0, 2, 1, 3)
    return dm2aa + dm2bb + dm2ab + dm2ab.transpose(1, 0, 3, 2)


def doubles_energy(
    rdm2: np.ndarray,
    mf
) -> np.ndarray:

    from pyscf import ao2mo

    norb = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mf.mol, mf.mo_coeff)
    eri = ao2mo.restore(1, eri, norb)

    return 0.5 * np.einsum("ijkl,ijkl->", eri, rdm2)


def total_energy_from_rdm12(dm1, dm2, mf):
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    # eri in MO basis (chemist)
    from pyscf import ao2mo
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), h1.shape[0])
    e1 = np.einsum("pq,pq->", h1, dm1)
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eri, dm2)
    return e1 + e2 + mf.mol.energy_nuc()


def main():
    """Run RDM2 system size scaling analysis."""
    print("=" * 70)
    print("Monte Carlo RDM2 System Size Scaling Analysis")
    print("=" * 70)

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    print("\n" + "=" * 70)
    print("Running Monte Carlo System Size Study")
    print("=" * 70)
    print(f"System sizes (N_H): {N_HYDROGEN}")
    print(f"Shadow samples: {N_SHADOWS}")
    print(f"MC iterations: {N_MC_ITERS}")
    print(f"Runs per system size: {N_RUNS}")

    # Store results per system size
    all_results = {}
    reference_data = {}

    for n_h in N_HYDROGEN:
        print(f"\n{'='*70}")
        print(f"System: H{n_h} chain (bond length = {BOND_LENGTH:.2f} A)")
        print(f"{'='*70}")

        # Build molecule for this system size
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

        # Store reference data
        reference_data[n_h] = {
            'E_hf': E_hf,
            'E_fci': E_fci,
            'E_corr': E_fci - E_hf,
            'E_double_ref': E_double_ref,
            'norb': norb,
        }

        # Initialize results for this system size
        results = {
            'E_tot': np.empty(N_RUNS, dtype=np.float64),
            'E_doubles': np.empty(N_RUNS, dtype=np.float64),
            'rel_err_E2': np.empty(N_RUNS, dtype=np.float64),
            'rel_frob_rdm2': np.empty(N_RUNS, dtype=np.float64),
            'max_abs_rdm2': np.empty(N_RUNS, dtype=np.float64),
        }

        sampler = MPSSampler(mf)
        shadow1 = ShadowEstimator(mf, fci_solver)
        shadow2 = ShadowEstimator(mf, fci_solver)

        for j in range(N_RUNS):
            print(f"  Run {j+1}/{N_RUNS}...", end=" ", flush=True)

            shadow1.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
            shadow2.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
            estimator = (shadow1, shadow2)

            mc = MonteCarloEstimator(estimator, sampler)
            rdm2_mc = mc.estimate_2rdm(max_iters=N_MC_ITERS)
            rdm2 = _spinorb_to_spatial_chem(rdm2_mc, norb)

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

            shadow1.clear_sample()
            shadow2.clear_sample()

        all_results[n_h] = results

    # Save results
    npz_path = os.path.join(output_dir, "data.npz")

    # Flatten results for npz storage
    npz_data = {}
    for n_h in N_HYDROGEN:
        for key, arr in all_results[n_h].items():
            npz_data[f"H{n_h}_{key}"] = arr

    np.savez_compressed(npz_path, **npz_data)
    print(f"\nSaved: {npz_path}")

    # Save metadata
    metadata = {
        "system": "H chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_hydrogen": N_HYDROGEN,
        "n_mc_iters": N_MC_ITERS,
        "n_shadow_samples": N_SHADOWS,
        "n_k_estimators": int(N_K_ESTIMATORS),
        "comments": RUN_COMMENT,
        "reference_data": {str(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in reference_data.items()},
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Convergence Plots")
    print("=" * 70)

    n_h_arr = np.array(N_HYDROGEN)

    # Compute statistics across runs for each system size
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

    # --- Plot 1: Relative Frobenius error of RDM2 vs system size ---
    ax1.errorbar(
        n_h_arr,
        rel_frob_mean,
        yerr=rel_frob_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$\|\Delta\Gamma\|_F / \|\Gamma_{\mathrm{ref}}\|_F$',
        linewidth=2, markersize=8,
    )
    ax1.set_xlabel('Number of Hydrogen Atoms')
    ax1.set_ylabel('Relative Frobenius Error')
    ax1.set_yscale('log')
    ax1.set_title('RDM2 Error vs System Size')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')

    # --- Plot 2: Relative two-electron energy error vs system size ---
    ax2.errorbar(
        n_h_arr,
        rel_E2_mean,
        yerr=rel_E2_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$|E_{2,\mathrm{MC}}-E_{2,\mathrm{ref}}|/|E_{2,\mathrm{ref}}|$',
        linewidth=2, markersize=8,
    )
    ax2.set_xlabel('Number of Hydrogen Atoms')
    ax2.set_ylabel('Relative $E_2$ Error')
    ax2.set_yscale('log')
    ax2.set_title('Two-electron Energy Error vs System Size')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')

    # --- Plot 3: E_doubles MC vs reference ---
    ax3.errorbar(
        n_h_arr,
        E_doubles_mean,
        yerr=E_doubles_sem,
        fmt='o-', capsize=5, capthick=2,
        label=r'$E_2^{\mathrm{MC}}$',
        linewidth=2, markersize=8,
    )
    ax3.plot(n_h_arr, E_double_refs, 's--', color='r', linewidth=2, markersize=8, label=r'$E_2^{\mathrm{ref}}$')
    ax3.set_xlabel('Number of Hydrogen Atoms')
    ax3.set_ylabel(r'$E_2$ (Hartree)')
    ax3.set_title('Two-electron Energy vs System Size')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'system_size_scaling.pdf')
    png_path = os.path.join(output_dir, 'system_size_scaling.png')
    svg_path = os.path.join(output_dir, 'system_size_scaling.svg')
    save_figure(pdf_path)
    save_figure(png_path, dpi=PLOT_DPI)
    save_figure(svg_path)
    print(f"Plots saved: system_size_scaling.pdf, system_size_scaling.png, system_size_scaling.svg")

    plt.show()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

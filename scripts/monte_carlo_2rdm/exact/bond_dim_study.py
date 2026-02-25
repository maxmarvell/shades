"""Bond dimension study: MPS sampler quality vs exact MC 2-RDM accuracy.

No shadows involved — uses exact overlaps only. Isolates the effect of
MPS sampler fidelity on the MC energy estimate. If the MPS distribution
introduces bias, it will show up here.
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
from utils import spinorb_to_spatial_chem, doubles_energy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

RUN_COMMENT = "Exact estimator only: isolate MPS sampler bias from shadow noise."

DEFAULT_OUTPUT_DIR = f"./results/bond_dim_study/exact/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 20
N_MC_ITERS = 5000

BOND_DIMS = [10, 25, 50, 100, 200]

N_HYDROGEN = 8
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"


def det_to_int(det):
    norb = det.shape[0]
    alpha = det & 1
    beta = (det >> 1) & 1
    res = sum(1 << i for i in range(norb) if alpha[i] == 1)
    res += sum(1 << i for i in range(norb, 2 * norb) if beta[i - norb] == 1)
    return res


def compute_distribution_metrics(fci_probs, mps_sampler):
    mps_dict = {}
    for det, prob in zip(mps_sampler.dets, mps_sampler.probs):
        key = det_to_int(det)
        mps_dict[key] = mps_dict.get(key, 0.0) + prob

    all_keys = set(range(len(fci_probs)))
    all_keys.update(mps_dict.keys())

    fid = 0.0
    tvd = 0.0
    for k in all_keys:
        p = fci_probs[k] if k < len(fci_probs) else 0.0
        q = mps_dict.get(k, 0.0)
        fid += np.sqrt(p * q)
        tvd += abs(p - q)

    # Mass missing from MPS
    fci_keys = {k for k in range(len(fci_probs)) if fci_probs[k] > 1e-16}
    missing_mass = sum(fci_probs[k] for k in fci_keys if k not in mps_dict)

    return fid ** 2, 0.5 * tvd, missing_mass


def main():
    print("=" * 80)
    print("Bond Dimension Study: Exact Estimator + MPS Sampler")
    print("=" * 80)

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

    fci_probs = np.abs(fci_solver.state.data) ** 2

    print(f"\nH{N_HYDROGEN} chain, r = {BOND_LENGTH} A, {BASIS_SET}")
    print(f"norb = {norb}, E_HF = {E_hf:.10f}, E_FCI = {E_fci:.10f}")
    print(f"E2_ref = {E_double_ref:.10f}")
    print(f"\nBond dims: {BOND_DIMS}")
    print(f"MC iters: {N_MC_ITERS}, Runs: {N_RUNS}")

    estimator = ExactEstimator(mf, fci_solver)

    fidelities = np.empty(len(BOND_DIMS))
    tvds = np.empty(len(BOND_DIMS))
    missing_masses = np.empty(len(BOND_DIMS))
    signed_err = np.empty((len(BOND_DIMS), N_RUNS))

    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        print(f"\n{'='*80}")
        print(f"Bond dimension: {bond_dim}")
        print(f"{'='*80}")

        sampler = MPSSampler(mf, max_bond_dim=bond_dim)

        fid, tvd, mm = compute_distribution_metrics(fci_probs, sampler)
        fidelities[bd_idx] = fid
        tvds[bd_idx] = tvd
        missing_masses[bd_idx] = mm
        print(f"  Fidelity = {fid:.10f}, TVD = {tvd:.6e}, missing mass = {mm:.6e}")
        print(f"  Num MPS dets: {len(sampler.dets)}")

        for j in range(N_RUNS):
            mc = MonteCarloEstimator(estimator, sampler)
            rdm2_mc = mc.estimate_2rdm(max_iters=N_MC_ITERS)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)
            E2 = doubles_energy(rdm2, mf)
            err = E2 - E_double_ref
            signed_err[bd_idx, j] = err
            print(f"  Run {j+1}/{N_RUNS}: E2 = {E2:.6f}, err = {err:+.4e}")

    # Save data
    npz_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        bond_dims=np.array(BOND_DIMS),
        fidelities=fidelities,
        tvds=tvds,
        missing_masses=missing_masses,
        signed_err=signed_err,
    )
    print(f"\nSaved: {npz_path}")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_mc_iters": int(N_MC_ITERS),
        "bond_dims": BOND_DIMS,
        "E_hf": float(E_hf),
        "E_fci": float(E_fci),
        "E_double_ref": float(E_double_ref),
        "comments": RUN_COMMENT,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Plots
    setup_plotting_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bd_arr = np.array(BOND_DIMS)

    # Plot 1: MPS distribution quality
    ax = axes[0]
    ax.plot(bd_arr, 1 - fidelities, 'o-', linewidth=2, markersize=8, label='$1 - F$')
    ax.plot(bd_arr, tvds, 's--', linewidth=2, markersize=8, label='TVD')
    ax.plot(bd_arr, missing_masses, '^:', linewidth=2, markersize=8, label='Missing mass')
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel('Distance from FCI')
    ax.set_yscale('log')
    ax.set_title('MPS Distribution Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Signed E2 error scatter
    ax = axes[1]
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        x = np.full(N_RUNS, bond_dim) + np.random.uniform(-2, 2, N_RUNS)
        ax.scatter(x, signed_err[bd_idx], alpha=0.4, s=20, color='C0')
        mu = signed_err[bd_idx].mean()
        sem = signed_err[bd_idx].std(ddof=1) / np.sqrt(N_RUNS)
        ax.errorbar(bond_dim, mu, yerr=2 * sem, fmt='D', color='C1',
                    markersize=8, capsize=5, capthick=2, linewidth=2, zorder=5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel(r'$E_2^{\mathrm{MC}} - E_2^{\mathrm{ref}}$ (Ha)')
    ax.set_title('Signed $E_2$ Error (Exact Overlaps)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Mean bias and std vs bond dim
    ax = axes[2]
    means = signed_err.mean(axis=1)
    stds = signed_err.std(axis=1, ddof=1)
    sems = stds / np.sqrt(N_RUNS)
    ax.errorbar(bd_arr, means, yerr=2 * sems, fmt='o-', color='C0',
                linewidth=2, markersize=8, capsize=5, capthick=2, label='Mean bias $\\pm$ 2 SEM')
    ax.fill_between(bd_arr, means - stds, means + stds, alpha=0.2, color='C0', label='$\\pm$ 1 std')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel(r'$E_2$ error (Ha)')
    ax.set_title('Bias and Variance vs Bond Dim')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'bond_dim_exact.pdf')
    png_path = os.path.join(output_dir, 'bond_dim_exact.png')
    save_figure(pdf_path)
    save_figure(png_path, dpi=300)

    plt.show()

    # Summary
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"{'D':>6} {'Fidelity':>12} {'TVD':>12} {'Miss mass':>12} | "
          f"{'Mean err':>12} {'SEM':>12} {'Std':>12} {'N>0':>5} {'N<0':>5}")
    print("-" * 90)
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        errs = signed_err[bd_idx]
        print(
            f"{bond_dim:>6} "
            f"{fidelities[bd_idx]:>12.8f} {tvds[bd_idx]:>12.4e} {missing_masses[bd_idx]:>12.4e} | "
            f"{errs.mean():>+12.6f} {errs.std(ddof=1)/np.sqrt(N_RUNS):>12.6f} "
            f"{errs.std(ddof=1):>12.6f} "
            f"{np.sum(errs > 0):>5} {np.sum(errs < 0):>5}"
        )

    print("\nDone!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

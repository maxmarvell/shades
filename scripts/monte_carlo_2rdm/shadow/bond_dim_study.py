"""Bond dimension study: MPS quality vs MC 2-RDM accuracy.

Sweeps the DMRG bond dimension used in the MPS sampler and measures:
1. MPS distribution fidelity to FCI (independent of shadow/MC)
2. MC 2-RDM accuracy using exact overlaps (isolates MPS effect)
3. MC 2-RDM accuracy using shadow overlaps, both mean and median-of-means

This disentangles the MPS sampler quality from the shadow estimator noise.
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
from shades.estimators import ShadowEstimator, ExactEstimator
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

RUN_COMMENT = "Bond dimension sweep: MPS fidelity and MC 2-RDM accuracy."

DEFAULT_OUTPUT_DIR = f"./results/bond_dim_study/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 10
N_MC_ITERS = 5000
N_SHADOWS = 1000
N_K_ESTIMATORS = 20
N_MC_BATCHES = 50

BOND_DIMS = [10, 25, 50, 100, 200, 300]

N_HYDROGEN = 6
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"


def det_to_int(det):
    norb = det.shape[0]
    alpha = det & 1
    beta = (det >> 1) & 1
    res = sum(1 << i for i in range(norb) if alpha[i] == 1)
    res += sum(1 << i for i in range(norb, 2 * norb) if beta[i - norb] == 1)
    return res


def compute_fidelity(fci_probs, mps_sampler, n_qubits):
    """Compute fidelity between FCI and MPS probability distributions."""
    mps_dict = {}
    for det, prob in zip(*mps_sampler.get_distribution()):
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

    return fid ** 2, 0.5 * tvd


def main():
    print("=" * 80)
    print("Bond Dimension Study: MPS Quality vs MC 2-RDM Accuracy")
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
    n_qubits = 2 * norb
    nelec = mf.mol.nelec
    _, rdm2_ref = direct_spin1.make_rdm12(fci_solver.civec, norb, nelec)
    E_double_ref = doubles_energy(rdm2_ref, mf)

    fci_probs = np.abs(fci_solver.state.data) ** 2

    print(f"\nH{N_HYDROGEN} chain, r = {BOND_LENGTH} A, {BASIS_SET}")
    print(f"norb = {norb}, n_qubits = {n_qubits}")
    print(f"E_HF = {E_hf:.10f}, E_FCI = {E_fci:.10f}")
    print(f"E2_ref = {E_double_ref:.10f}")
    print(f"\nBond dims: {BOND_DIMS}")
    print(f"MC iters: {N_MC_ITERS}, Runs: {N_RUNS}")
    print(f"Shadows: {N_SHADOWS}, K = {N_K_ESTIMATORS}")
    print(f"MoM batches: {N_MC_BATCHES}")

    # Storage
    fidelities = np.empty(len(BOND_DIMS))
    tvds = np.empty(len(BOND_DIMS))

    exact_signed_err = np.empty((len(BOND_DIMS), N_RUNS))
    shadow_mean_signed_err = np.empty((len(BOND_DIMS), N_RUNS))
    shadow_mom_signed_err = np.empty((len(BOND_DIMS), N_RUNS))

    exact_estimator = ExactEstimator(mf, fci_solver)

    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        print(f"\n{'='*80}")
        print(f"Bond dimension: {bond_dim}")
        print(f"{'='*80}")

        sampler = MPSSampler(mf, max_bond_dim=bond_dim)

        fid, tvd = compute_fidelity(fci_probs, sampler, n_qubits)
        fidelities[bd_idx] = fid
        tvds[bd_idx] = tvd
        print(f"  Fidelity = {fid:.10f}, TVD = {tvd:.6e}")

        shadow1 = ShadowEstimator(mf, fci_solver)
        shadow2 = ShadowEstimator(mf, fci_solver)

        for j in range(N_RUNS):
            print(f"  Run {j+1}/{N_RUNS}...", end=" ", flush=True)

            # Exact estimator + MPS sampler
            mc_exact = MonteCarloEstimator(exact_estimator, sampler)
            rdm2_mc = mc_exact.estimate_2rdm(max_iters=N_MC_ITERS)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)
            E2_exact = doubles_energy(rdm2, mf)
            exact_signed_err[bd_idx, j] = E2_exact - E_double_ref

            # Shadow estimator + MPS sampler (mean)
            shadow1.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
            shadow2.sample(N_SHADOWS // 2, N_K_ESTIMATORS)

            mc_shadow_mean = MonteCarloEstimator((shadow1, shadow2), sampler)
            rdm2_mc = mc_shadow_mean.estimate_2rdm(max_iters=N_MC_ITERS)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)
            E2_shadow_mean = doubles_energy(rdm2, mf)
            shadow_mean_signed_err[bd_idx, j] = E2_shadow_mean - E_double_ref

            # Shadow estimator + MPS sampler (median-of-means, same shadows)
            mc_shadow_mom = MonteCarloEstimator((shadow1, shadow2), sampler)
            rdm2_mc = mc_shadow_mom.estimate_2rdm(max_iters=N_MC_ITERS, n_batches=N_MC_BATCHES)
            rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)
            E2_shadow_mom = doubles_energy(rdm2, mf)
            shadow_mom_signed_err[bd_idx, j] = E2_shadow_mom - E_double_ref

            shadow1.clear_sample()
            shadow2.clear_sample()

            print(
                f"exact = {E2_exact - E_double_ref:+.4e}, "
                f"shadow_mean = {E2_shadow_mean - E_double_ref:+.4e}, "
                f"shadow_mom = {E2_shadow_mom - E_double_ref:+.4e}"
            )

    # Save data
    npz_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        bond_dims=np.array(BOND_DIMS),
        fidelities=fidelities,
        tvds=tvds,
        exact_signed_err=exact_signed_err,
        shadow_mean_signed_err=shadow_mean_signed_err,
        shadow_mom_signed_err=shadow_mom_signed_err,
    )
    print(f"\nSaved: {npz_path}")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "n_mc_iters": int(N_MC_ITERS),
        "n_shadow_samples": int(N_SHADOWS),
        "n_k_estimators": int(N_K_ESTIMATORS),
        "n_mc_batches": int(N_MC_BATCHES),
        "bond_dims": BOND_DIMS,
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
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel('Distance from FCI')
    ax.set_yscale('log')
    ax.set_title('MPS Distribution Quality')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Signed E2 error vs bond dim (scatter + mean/SEM)
    ax = axes[1]
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        x_exact = bond_dim - 2
        x_mean = bond_dim
        x_mom = bond_dim + 2

        ax.scatter(np.full(N_RUNS, x_exact), exact_signed_err[bd_idx], alpha=0.3, s=15, color='C0')
        ax.scatter(np.full(N_RUNS, x_mean), shadow_mean_signed_err[bd_idx], alpha=0.3, s=15, color='C1')
        ax.scatter(np.full(N_RUNS, x_mom), shadow_mom_signed_err[bd_idx], alpha=0.3, s=15, color='C2')

    # Mean +/- 2 SEM
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        for x_off, data, color, label in [
            (-2, exact_signed_err, 'C0', 'Exact' if bd_idx == 0 else None),
            (0, shadow_mean_signed_err, 'C1', 'Shadow mean' if bd_idx == 0 else None),
            (2, shadow_mom_signed_err, 'C2', 'Shadow MoM' if bd_idx == 0 else None),
        ]:
            mu = data[bd_idx].mean()
            sem = data[bd_idx].std(ddof=1) / np.sqrt(N_RUNS)
            ax.errorbar(bond_dim + x_off, mu, yerr=2 * sem, fmt='D', color=color,
                        markersize=6, capsize=4, capthick=1.5, linewidth=1.5,
                        label=label)

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel(r'$E_2^{\mathrm{MC}} - E_2^{\mathrm{ref}}$ (Ha)')
    ax.set_title('Signed $E_2$ Error')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Std of E2 error vs bond dim
    ax = axes[2]
    ax.plot(bd_arr, exact_signed_err.std(axis=1, ddof=1), 'o-', color='C0', linewidth=2, markersize=8, label='Exact')
    ax.plot(bd_arr, shadow_mean_signed_err.std(axis=1, ddof=1), 's-', color='C1', linewidth=2, markersize=8, label='Shadow mean')
    ax.plot(bd_arr, shadow_mom_signed_err.std(axis=1, ddof=1), '^-', color='C2', linewidth=2, markersize=8, label='Shadow MoM')
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel(r'Std of $E_2$ error (Ha)')
    ax.set_yscale('log')
    ax.set_title('Estimator Variance')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'bond_dim_study.pdf')
    png_path = os.path.join(output_dir, 'bond_dim_study.png')
    save_figure(pdf_path)
    save_figure(png_path, dpi=300)

    plt.show()

    # Summary table
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"{'D':>6} {'Fidelity':>12} {'TVD':>12} | "
          f"{'Exact bias':>12} {'Exact std':>12} | "
          f"{'Mean bias':>12} {'Mean std':>12} | "
          f"{'MoM bias':>12} {'MoM std':>12}")
    print("-" * 100)
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        print(
            f"{bond_dim:>6} "
            f"{fidelities[bd_idx]:>12.8f} {tvds[bd_idx]:>12.4e} | "
            f"{exact_signed_err[bd_idx].mean():>+12.6f} {exact_signed_err[bd_idx].std(ddof=1):>12.6f} | "
            f"{shadow_mean_signed_err[bd_idx].mean():>+12.6f} {shadow_mean_signed_err[bd_idx].std(ddof=1):>12.6f} | "
            f"{shadow_mom_signed_err[bd_idx].mean():>+12.6f} {shadow_mom_signed_err[bd_idx].std(ddof=1):>12.6f}"
        )

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
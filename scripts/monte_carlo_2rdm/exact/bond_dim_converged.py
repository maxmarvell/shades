"""Bond dimension study with convergence: run exact MC until E2 stabilizes.

For each bond dimension, runs the exact estimator + MPS sampler until
the running E2 estimate converges (relative change falls below threshold
over a window), then reports the final converged value.

This reveals whether the MPS distribution introduces a systematic bias
in the *converged* estimate, independent of MC sampling noise.
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

RUN_COMMENT = "Exact estimator converged: isolate MPS bias at convergence."

DEFAULT_OUTPUT_DIR = f"./results/bond_dim_converged/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_RUNS = 10
MAX_MC_ITERS = 100000
CONV_WINDOW = 500      # check convergence over this many iterations
CONV_THRESHOLD = 1e-6  # relative change in E2 over window
CHECK_EVERY = 100      # how often to evaluate E2

BOND_DIMS = [50, 100, 200, 300, 400, 500]

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

    fci_keys = {k for k in range(len(fci_probs)) if fci_probs[k] > 1e-16}
    missing_mass = sum(fci_probs[k] for k in fci_keys if k not in mps_dict)

    return fid ** 2, 0.5 * tvd, missing_mass


def main():
    print("=" * 80)
    print("Bond Dimension Study: Exact Estimator, Converged")
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
    print(f"Max MC iters: {MAX_MC_ITERS}")
    print(f"Convergence: rel change < {CONV_THRESHOLD} over window of {CONV_WINDOW} iters")
    print(f"Runs: {N_RUNS}")

    estimator = ExactEstimator(mf, fci_solver)

    fidelities = np.empty(len(BOND_DIMS))
    tvds = np.empty(len(BOND_DIMS))
    missing_masses = np.empty(len(BOND_DIMS))
    converged_E2 = np.full((len(BOND_DIMS), N_RUNS), np.nan)
    converged_err = np.full((len(BOND_DIMS), N_RUNS), np.nan)
    converged_at = np.full((len(BOND_DIMS), N_RUNS), MAX_MC_ITERS, dtype=int)

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

        for j in range(N_RUNS):
            # Track E2 history for convergence check
            e2_history = []
            final_E2 = [None]
            final_iter = [MAX_MC_ITERS]

            n_checks = CONV_WINDOW // CHECK_EVERY

            def on_iter(i, gamma):
                if (i + 1) % CHECK_EVERY != 0:
                    return

                rdm2 = spinorb_to_spatial_chem(gamma, norb)
                E2 = doubles_energy(rdm2, mf)
                e2_history.append(E2)

                if len(e2_history) >= n_checks:
                    recent = e2_history[-n_checks:]
                    old_val = recent[0]
                    new_val = recent[-1]
                    if abs(old_val) > 1e-15:
                        rel_change = abs(new_val - old_val) / abs(old_val)
                    else:
                        rel_change = abs(new_val - old_val)

                    if rel_change < CONV_THRESHOLD:
                        final_E2[0] = new_val
                        final_iter[0] = i + 1
                        raise StopIteration

            mc = MonteCarloEstimator(estimator, sampler)
            rdm2_mc = mc.estimate_2rdm(max_iters=MAX_MC_ITERS, callback=on_iter)

            if final_E2[0] is None:
                rdm2 = spinorb_to_spatial_chem(rdm2_mc, norb)
                final_E2[0] = doubles_energy(rdm2, mf)

            E2 = final_E2[0]
            err = E2 - E_double_ref
            converged_E2[bd_idx, j] = E2
            converged_err[bd_idx, j] = err
            converged_at[bd_idx, j] = final_iter[0]

            status = "converged" if final_iter[0] < MAX_MC_ITERS else "max iters"
            print(f"  Run {j+1}/{N_RUNS}: E2 = {E2:.8f}, err = {err:+.6e}, "
                  f"iters = {final_iter[0]} ({status})")

    # Save data
    npz_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        bond_dims=np.array(BOND_DIMS),
        fidelities=fidelities,
        tvds=tvds,
        missing_masses=missing_masses,
        converged_E2=converged_E2,
        converged_err=converged_err,
        converged_at=converged_at,
    )
    print(f"\nSaved: {npz_path}")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "n_runs": int(N_RUNS),
        "max_mc_iters": int(MAX_MC_ITERS),
        "conv_window": int(CONV_WINDOW),
        "conv_threshold": float(CONV_THRESHOLD),
        "check_every": int(CHECK_EVERY),
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    bd_arr = np.array(BOND_DIMS)

    # Plot 1: MPS distribution quality
    ax = axes[0, 0]
    ax.plot(bd_arr, 1 - fidelities, 'o-', linewidth=2, markersize=8, label='$1 - F$')
    ax.plot(bd_arr, tvds, 's--', linewidth=2, markersize=8, label='TVD')
    ax.plot(bd_arr, missing_masses, '^:', linewidth=2, markersize=8, label='Missing mass')
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel('Distance from FCI')
    ax.set_yscale('log')
    ax.set_title('MPS Distribution Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Converged signed E2 error
    ax = axes[0, 1]
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        x = np.full(N_RUNS, bond_dim) + np.random.uniform(-2, 2, N_RUNS)
        ax.scatter(x, converged_err[bd_idx], alpha=0.4, s=20, color='C0')
        mu = converged_err[bd_idx].mean()
        sem = converged_err[bd_idx].std(ddof=1) / np.sqrt(N_RUNS)
        ax.errorbar(bond_dim, mu, yerr=2 * sem, fmt='D', color='C1',
                    markersize=8, capsize=5, capthick=2, linewidth=2, zorder=5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel(r'$E_2^{\mathrm{conv}} - E_2^{\mathrm{ref}}$ (Ha)')
    ax.set_title('Converged $E_2$ Error')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Convergence iteration count
    ax = axes[1, 0]
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        x = np.full(N_RUNS, bond_dim) + np.random.uniform(-2, 2, N_RUNS)
        ax.scatter(x, converged_at[bd_idx], alpha=0.4, s=20, color='C0')
    ax.plot(bd_arr, converged_at.mean(axis=1), 'D-', color='C1', markersize=8, linewidth=2)
    ax.axhline(MAX_MC_ITERS, color='r', linestyle=':', linewidth=1, label=f'max ({MAX_MC_ITERS})')
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel('Iterations to converge')
    ax.set_title('Convergence Speed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Bias and std
    ax = axes[1, 1]
    means = converged_err.mean(axis=1)
    stds = converged_err.std(axis=1, ddof=1)
    sems = stds / np.sqrt(N_RUNS)
    ax.errorbar(bd_arr, means, yerr=2 * sems, fmt='o-', color='C0',
                linewidth=2, markersize=8, capsize=5, capthick=2, label='Mean bias $\\pm$ 2 SEM')
    ax.fill_between(bd_arr, means - stds, means + stds, alpha=0.2, color='C0', label='$\\pm$ 1 std')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Max Bond Dimension')
    ax.set_ylabel(r'Converged $E_2$ error (Ha)')
    ax.set_title('Bias and Variance at Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'bond_dim_converged.pdf')
    png_path = os.path.join(output_dir, 'bond_dim_converged.png')
    save_figure(pdf_path)
    save_figure(png_path, dpi=300)

    plt.show()

    # Summary
    print("\n" + "=" * 100)
    print("Summary (converged values)")
    print("=" * 100)
    print(f"{'D':>6} {'Fidelity':>12} {'TVD':>12} {'Miss mass':>12} | "
          f"{'Mean err':>12} {'SEM':>12} {'Std':>12} {'Mean iters':>12}")
    print("-" * 100)
    for bd_idx, bond_dim in enumerate(BOND_DIMS):
        errs = converged_err[bd_idx]
        print(
            f"{bond_dim:>6} "
            f"{fidelities[bd_idx]:>12.8f} {tvds[bd_idx]:>12.4e} {missing_masses[bd_idx]:>12.4e} | "
            f"{errs.mean():>+12.8f} {errs.std(ddof=1)/np.sqrt(N_RUNS):>12.8f} "
            f"{errs.std(ddof=1):>12.8f} "
            f"{converged_at[bd_idx].mean():>12.0f}"
        )

    print("\nDone!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

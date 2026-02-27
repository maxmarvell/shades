"""Compare MC 2-RDM estimation: shadow estimator vs exact estimator.

For an H8 chain, run the MonteCarloEstimator with:
  1. ExactEstimator (direct statevector overlaps) — the gold standard
  2. ShadowEstimator with increasing shadow counts — should converge to exact

Both use the same MPS importance sampler and median-of-means MC aggregation.
MC iterations run until convergence (relative change in 2-RDM Frobenius norm
falls below a threshold over a window of batches).
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
from shades.monte_carlo import MonteCarloEstimator, MPSSampler

from plotting_config import setup_plotting_style, save_figure
from utils import spinorb_to_spatial_chem, doubles_energy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

DEFAULT_OUTPUT_DIR = f"./results/shadow_vs_exact/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_HYDROGEN = 8
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"

SHADOW_COUNTS = [1000, 2000, 5000, 10000, 20000]
N_K_ESTIMATORS = 20

MPS_BOND_DIM = 300
MPS_PROB_CUTOFF = None

N_MC_ITERS = 50000
N_MC_BATCHES = 1
N_MC_BATCHES_PARALLEL = 100
N_WORKERS = 8
CONV_WINDOW = 20
CONV_THRESHOLD_MEAN = 1e-2
CONV_THRESHOLD_MOM = 5e-4

N_RUNS = 5


LOG_EVERY = 100  # for plain mean: log every N iters


def run_mc_converged(mc, max_iters, n_batches, rdm2_ref, rdm2_ref_norm, mf, norb, label,
                     E2_ref, conv_threshold=None, parallel=False, n_workers=1):
    """Run MC 2-RDM estimation with convergence checking.

    When n_batches=1, uses plain mean (unbiased). When n_batches>1, uses
    median-of-means (robust but biased for skewed distributions).

    If parallel=True, uses estimate_2rdm_parallel with pre-computed overlaps
    and multiprocessing workers for tensor assembly.

    Returns (rdm2_spatial, doubles_E, converged_iter, trajectory).
    """
    from pyscf import ao2mo
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), norb)

    trajectory = []  # (global_iter, rel_frob, rel_err_E2)
    recent_frobs = []
    converged_at = max_iters
    if conv_threshold is None:
        conv_threshold = CONV_THRESHOLD_MEAN if n_batches <= 1 else CONV_THRESHOLD_MOM

    def _doubles_energy_fast(rdm2):
        return 0.5 * np.einsum("ijkl,ijkl->", eri_mo, rdm2)

    def on_callback(i, gamma):
        nonlocal converged_at
        rdm2 = spinorb_to_spatial_chem(gamma, norb)
        rel_frob = np.linalg.norm(rdm2 - rdm2_ref) / rdm2_ref_norm
        E2 = _doubles_energy_fast(rdm2)
        rel_err = np.abs(E2 - E2_ref) / np.abs(E2_ref)

        # For plain mean, only log periodically
        if not parallel and n_batches <= 1 and (i + 1) % LOG_EVERY != 0:
            return

        trajectory.append((i + 1, float(rel_frob), float(rel_err)))

        recent_frobs.append(rel_frob)
        if len(recent_frobs) > CONV_WINDOW:
            recent_frobs.pop(0)

        checkpoint = len(trajectory)
        if checkpoint % 50 == 0 or checkpoint <= 5:
            print(f"    {label} iter {i+1:5d}: "
                  f"rel_frob={rel_frob:.4e}, rel_err_E2={rel_err:.4e}")

        if len(recent_frobs) == CONV_WINDOW:
            window = np.array(recent_frobs)
            half = CONV_WINDOW // 2
            mean_old = np.mean(window[:half])
            mean_new = np.mean(window[half:])
            rel_change = np.abs(mean_new - mean_old) / (np.abs(mean_old) + 1e-30)
            if rel_change < conv_threshold:
                converged_at = i + 1
                print(f"    {label} converged at iter {i+1}, "
                      f"rel_change={rel_change:.2e}")
                raise StopIteration

    if parallel:
        gamma = mc.estimate_2rdm_parallel(
            max_iters=max_iters, n_batches=n_batches, n_workers=n_workers, callback=on_callback,
        )
    else:
        gamma = mc.estimate_2rdm(max_iters=max_iters, n_batches=n_batches, callback=on_callback)
    rdm2 = spinorb_to_spatial_chem(gamma, norb)
    E2 = doubles_energy(rdm2, mf)
    return rdm2, float(E2), converged_at, trajectory


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Shadow vs Exact MC 2-RDM Estimation")
    print("=" * 70)

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
    E2_ref = doubles_energy(rdm2_ref, mf)
    rdm2_ref_norm = np.linalg.norm(rdm2_ref)

    print(f"\nMolecule: H{N_HYDROGEN} chain (r = {BOND_LENGTH:.2f} A)")
    print(f"Basis: {BASIS_SET}, norb = {norb}")
    print(f"E_HF  = {E_hf:.10f} Ha")
    print(f"E_FCI = {E_fci:.10f} Ha")
    print(f"E_corr = {E_fci - E_hf:.10f} Ha")
    print(f"E2_ref = {E2_ref:.10f} Ha")
    print(f"\nMC: {N_MC_ITERS} iters, {N_MC_BATCHES} batches (exact), "
          f"{N_MC_BATCHES_PARALLEL} batches (shadow parallel, {N_WORKERS} workers)")
    print(f"Convergence: window={CONV_WINDOW}, "
          f"threshold_mean={CONV_THRESHOLD_MEAN}, threshold_mom={CONV_THRESHOLD_MOM}")
    print(f"Shadow counts to test: {SHADOW_COUNTS}")
    print(f"Runs per setting: {N_RUNS}")

    sampler = MPSSampler(mf, max_bond_dim=MPS_BOND_DIM, prob_cutoff=MPS_PROB_CUTOFF)

    # --- Exact estimator runs ---
    print(f"\n{'='*70}")
    print("EXACT ESTIMATOR")
    print(f"{'='*70}")

    exact1 = ExactEstimator(mf, fci_solver)
    exact2 = ExactEstimator(mf, fci_solver)
    exact_results = []

    for run in range(N_RUNS):
        print(f"\n  Run {run + 1}/{N_RUNS}")
        mc = MonteCarloEstimator((exact1, exact2), sampler)
        rdm2, E2, conv_iter, traj = run_mc_converged(
            mc, N_MC_ITERS, 1, rdm2_ref, rdm2_ref_norm, mf, norb, "exact", E2_ref,
        )
        rel_frob = np.linalg.norm(rdm2 - rdm2_ref) / rdm2_ref_norm
        rel_err_E2 = np.abs(E2 - E2_ref) / np.abs(E2_ref)
        exact_results.append({
            "rel_frob": rel_frob,
            "rel_err_E2": rel_err_E2,
            "E2": E2,
            "converged_at": conv_iter,
            "trajectory": traj,
        })
        print(f"    Final: rel_frob={rel_frob:.4e}, rel_err_E2={rel_err_E2:.4e}, E2={E2:.8f}")

    # --- Shadow estimator runs ---
    shadow_results = {}

    for n_shadows in SHADOW_COUNTS:
        print(f"\n{'='*70}")
        print(f"SHADOW ESTIMATOR: N_shadows = {n_shadows}")
        print(f"{'='*70}")

        runs = []
        for run in range(N_RUNS):
            print(f"\n  Run {run + 1}/{N_RUNS}")

            shadow1 = ShadowEstimator(mf, fci_solver)
            shadow2 = ShadowEstimator(mf, fci_solver)
            shadow1.sample(n_shadows // 2, N_K_ESTIMATORS)
            shadow2.sample(n_shadows // 2, N_K_ESTIMATORS)

            mc = MonteCarloEstimator((shadow1, shadow2), sampler)
            rdm2, E2, conv_iter, traj = run_mc_converged(
                mc, N_MC_ITERS, N_MC_BATCHES_PARALLEL, rdm2_ref, rdm2_ref_norm, mf, norb,
                f"shadow({n_shadows})", E2_ref,
                parallel=True, n_workers=N_WORKERS,
            )
            rel_frob = np.linalg.norm(rdm2 - rdm2_ref) / rdm2_ref_norm
            rel_err_E2 = np.abs(E2 - E2_ref) / np.abs(E2_ref)
            runs.append({
                "rel_frob": rel_frob,
                "rel_err_E2": rel_err_E2,
                "E2": E2,
                "converged_at": conv_iter,
                "trajectory": traj,
            })
            print(f"    Final: rel_frob={rel_frob:.4e}, rel_err_E2={rel_err_E2:.4e}, E2={E2:.8f}")

            shadow1.clear_sample()
            shadow2.clear_sample()

        shadow_results[n_shadows] = runs

    # --- Save results ---
    save_data = {
        "system": f"H{N_HYDROGEN}",
        "bond_length": BOND_LENGTH,
        "basis": BASIS_SET,
        "E_hf": float(E_hf),
        "E_fci": float(E_fci),
        "E2_ref": float(E2_ref),
        "n_mc_iters": N_MC_ITERS,
        "n_mc_batches": N_MC_BATCHES,
        "n_k_estimators": N_K_ESTIMATORS,
        "mps_bond_dim": MPS_BOND_DIM,
        "mps_prob_cutoff": MPS_PROB_CUTOFF,
        "conv_window": CONV_WINDOW,
        "conv_threshold_mean": CONV_THRESHOLD_MEAN,
        "conv_threshold_mom": CONV_THRESHOLD_MOM,
        "n_runs": N_RUNS,
        "shadow_counts": SHADOW_COUNTS,
        "exact_results": exact_results,
        "shadow_results": {str(k): v for k, v in shadow_results.items()},
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_dir}results.json")

    # --- Plot ---
    setup_plotting_style()

    exact_frobs = [r["rel_frob"] for r in exact_results]
    exact_E2_errs = [r["rel_err_E2"] for r in exact_results]

    shadow_frob_med = []
    shadow_frob_lo = []
    shadow_frob_hi = []
    shadow_E2_med = []
    shadow_E2_lo = []
    shadow_E2_hi = []

    for ns in SHADOW_COUNTS:
        frobs = [r["rel_frob"] for r in shadow_results[ns]]
        e2errs = [r["rel_err_E2"] for r in shadow_results[ns]]
        shadow_frob_med.append(np.median(frobs))
        shadow_frob_lo.append(np.percentile(frobs, 25))
        shadow_frob_hi.append(np.percentile(frobs, 75))
        shadow_E2_med.append(np.median(e2errs))
        shadow_E2_lo.append(np.percentile(e2errs, 25))
        shadow_E2_hi.append(np.percentile(e2errs, 75))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Left: relative Frobenius error
    ax = axes[0]
    ax.errorbar(
        SHADOW_COUNTS, shadow_frob_med,
        yerr=[np.array(shadow_frob_med) - np.array(shadow_frob_lo),
              np.array(shadow_frob_hi) - np.array(shadow_frob_med)],
        fmt="o-", ms=4, capsize=3, label="Shadow"
    )
    ax.axhline(np.median(exact_frobs), color="k", ls="--", lw=1, label="Exact")
    ax.axhspan(np.percentile(exact_frobs, 25), np.percentile(exact_frobs, 75),
               alpha=0.15, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shadow samples $N$")
    ax.set_ylabel(r"Rel. Frobenius error in $\Gamma^{(2)}$")
    ax.set_title("2-RDM accuracy")
    ax.legend(fontsize=8)

    # Right: relative E2 error
    ax = axes[1]
    ax.errorbar(
        SHADOW_COUNTS, shadow_E2_med,
        yerr=[np.array(shadow_E2_med) - np.array(shadow_E2_lo),
              np.array(shadow_E2_hi) - np.array(shadow_E2_med)],
        fmt="o-", ms=4, capsize=3, label="Shadow"
    )
    ax.axhline(np.median(exact_E2_errs), color="k", ls="--", lw=1, label="Exact")
    ax.axhspan(np.percentile(exact_E2_errs, 25), np.percentile(exact_E2_errs, 75),
               alpha=0.15, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shadow samples $N$")
    ax.set_ylabel(r"Rel. error in $E_2$")
    ax.set_title("Doubles energy accuracy")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(os.path.join(output_dir, "shadow_vs_exact.pdf"))
    save_figure(os.path.join(output_dir, "shadow_vs_exact.png"))
    plt.close()

    # Convergence trajectory plot (one run per setting)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]
    traj = exact_results[0]["trajectory"]
    iters, frobs, _ = zip(*traj)
    ax.plot(iters, frobs, "k-", lw=1, label="Exact")

    for ns in SHADOW_COUNTS:
        traj = shadow_results[ns][0]["trajectory"]
        iters, frobs, _ = zip(*traj)
        ax.plot(iters, frobs, lw=1, label=f"$N={ns}$")

    ax.set_xlabel("MC iteration")
    ax.set_ylabel(r"Rel. Frobenius error")
    ax.set_yscale("log")
    ax.set_title("Convergence (run 1)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    traj = exact_results[0]["trajectory"]
    iters, _, e2errs = zip(*traj)
    ax.plot(iters, e2errs, "k-", lw=1, label="Exact")

    for ns in SHADOW_COUNTS:
        traj = shadow_results[ns][0]["trajectory"]
        iters, _, e2errs = zip(*traj)
        ax.plot(iters, e2errs, lw=1, label=f"$N={ns}$")

    ax.set_xlabel("MC iteration")
    ax.set_ylabel(r"Rel. error in $E_2$")
    ax.set_yscale("log")
    ax.set_title("Convergence (run 1)")
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    save_figure(os.path.join(output_dir, "convergence_traces.pdf"))
    save_figure(os.path.join(output_dir, "convergence_traces.png"))
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()

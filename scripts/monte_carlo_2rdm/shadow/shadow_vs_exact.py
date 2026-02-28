"""Systematic evaluation: shadow vs exact MC 2-RDM estimation for H4.

Compares three estimator configurations:
  1. Exact (gold standard): sequential estimate_2rdm() with convergence callback
  2. Shadow + mean: parallel estimate_2rdm_parallel() with n_batches=1
  3. Shadow + MoM: parallel estimate_2rdm_parallel() with n_batches>1

All three use the same MPS sampler (fixed high bond dimension known to be unbiased).
Goal: demonstrate that the MC 2-RDM estimation is unbiased when the MPS bond
dimension is sufficient.
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

# --- System parameters ---
N_HYDROGEN = 4
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"

# --- MPS sampler ---
MPS_BOND_DIM = 300
MPS_PROB_CUTOFF = None

# --- Shadow parameters ---
SHADOW_COUNTS = [1000, 5000, 10000, 20000, 30000]
N_K_ESTIMATORS = 20

# --- MC parameters for shadow (parallel, fixed budget) ---
N_MC_ITERS = 50000
N_MC_BATCHES_MOM = 50
N_WORKERS = 8

# --- MC parameters for exact (sequential with convergence) ---
MAX_MC_ITERS_EXACT = 100000
CONV_WINDOW = 500
CONV_THRESHOLD = 1e-6
CHECK_EVERY = 100

N_RUNS = 20


def run_exact_converged(mc, mf, norb, E2_ref, rdm2_ref, run_idx):
    """Run exact MC with convergence callback (sequential).

    Returns dict with E2, signed_err, rel_frob, converged_at.
    """
    rdm2_ref_norm = np.linalg.norm(rdm2_ref)
    n_checks = CONV_WINDOW // CHECK_EVERY

    e2_history = []
    final_E2 = [None]
    final_iter = [MAX_MC_ITERS_EXACT]

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

    gamma = mc.estimate_2rdm(max_iters=MAX_MC_ITERS_EXACT, callback=on_iter)

    if final_E2[0] is None:
        rdm2 = spinorb_to_spatial_chem(gamma, norb)
        final_E2[0] = doubles_energy(rdm2, mf)

    rdm2_final = spinorb_to_spatial_chem(gamma, norb)
    E2 = final_E2[0]
    signed_err = E2 - E2_ref
    rel_frob = np.linalg.norm(rdm2_final - rdm2_ref) / rdm2_ref_norm
    converged_at = final_iter[0]

    status = "converged" if converged_at < MAX_MC_ITERS_EXACT else "max iters"
    print(f"  Run {run_idx + 1}/{N_RUNS}: E2={E2:.8f}, err={signed_err:+.6e}, "
          f"rel_frob={rel_frob:.4e}, iters={converged_at} ({status})")

    return {
        "E2": float(E2),
        "signed_err": float(signed_err),
        "rel_frob": float(rel_frob),
        "converged_at": int(converged_at),
    }


def run_shadow_parallel(mc, mf, norb, E2_ref, rdm2_ref, n_batches, run_idx, label):
    """Run shadow MC with parallel estimation (fixed budget).

    Returns dict with E2, signed_err, rel_frob.
    """
    rdm2_ref_norm = np.linalg.norm(rdm2_ref)

    gamma = mc.estimate_2rdm_parallel(
        max_iters=N_MC_ITERS,
        n_batches=n_batches,
        n_workers=N_WORKERS,
    )

    rdm2 = spinorb_to_spatial_chem(gamma, norb)
    E2 = float(doubles_energy(rdm2, mf))
    signed_err = E2 - E2_ref
    rel_frob = np.linalg.norm(rdm2 - rdm2_ref) / rdm2_ref_norm

    print(f"  Run {run_idx + 1}/{N_RUNS} ({label}): E2={E2:.8f}, "
          f"err={signed_err:+.6e}, rel_frob={rel_frob:.4e}")

    return {
        "E2": float(E2),
        "signed_err": float(signed_err),
        "rel_frob": float(rel_frob),
    }


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Shadow vs Exact MC 2-RDM Estimation (Systematic)")
    print("=" * 70)

    # --- Setup molecule ---
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
    E2_ref = float(doubles_energy(rdm2_ref, mf))
    rdm2_ref_norm = float(np.linalg.norm(rdm2_ref))

    print(f"\nMolecule: H{N_HYDROGEN} chain (r = {BOND_LENGTH:.2f} A)")
    print(f"Basis: {BASIS_SET}, norb = {norb}")
    print(f"E_HF  = {E_hf:.10f} Ha")
    print(f"E_FCI = {E_fci:.10f} Ha")
    print(f"E_corr = {E_fci - E_hf:.10f} Ha")
    print(f"E2_ref = {E2_ref:.10f} Ha")
    print(f"\nExact: max {MAX_MC_ITERS_EXACT} iters, convergence window={CONV_WINDOW}, "
          f"threshold={CONV_THRESHOLD}, check_every={CHECK_EVERY}")
    print(f"Shadow: {N_MC_ITERS} iters, {N_WORKERS} workers")
    print(f"Shadow counts: {SHADOW_COUNTS}, K estimators: {N_K_ESTIMATORS}")
    print(f"MoM batches: {N_MC_BATCHES_MOM}")
    print(f"Runs per setting: {N_RUNS}")

    sampler = MPSSampler(mf, max_bond_dim=MPS_BOND_DIM, prob_cutoff=MPS_PROB_CUTOFF)

    n_shadow_counts = len(SHADOW_COUNTS)

    # Storage arrays
    exact_E2 = np.empty(N_RUNS)
    exact_signed_err = np.empty(N_RUNS)
    exact_rel_frob = np.empty(N_RUNS)
    exact_converged_at = np.empty(N_RUNS, dtype=int)

    shadow_mean_E2 = np.empty((n_shadow_counts, N_RUNS))
    shadow_mean_signed_err = np.empty((n_shadow_counts, N_RUNS))
    shadow_mean_rel_frob = np.empty((n_shadow_counts, N_RUNS))

    shadow_mom_E2 = np.empty((n_shadow_counts, N_RUNS))
    shadow_mom_signed_err = np.empty((n_shadow_counts, N_RUNS))
    shadow_mom_rel_frob = np.empty((n_shadow_counts, N_RUNS))

    # ========== Phase 1: Exact estimator ==========
    print(f"\n{'='*70}")
    print("PHASE 1: EXACT ESTIMATOR (converged, sequential)")
    print(f"{'='*70}")

    exact1 = ExactEstimator(mf, fci_solver)
    exact2 = ExactEstimator(mf, fci_solver)

    for run in range(N_RUNS):
        mc = MonteCarloEstimator((exact1, exact2), sampler)
        result = run_exact_converged(mc, mf, norb, E2_ref, rdm2_ref, run)
        exact_E2[run] = result["E2"]
        exact_signed_err[run] = result["signed_err"]
        exact_rel_frob[run] = result["rel_frob"]
        exact_converged_at[run] = result["converged_at"]

    # ========== Phase 2: Shadow estimator ==========
    for sc_idx, n_shadows in enumerate(SHADOW_COUNTS):
        print(f"\n{'='*70}")
        print(f"PHASE 2: SHADOW ESTIMATOR, N_shadows = {n_shadows}")
        print(f"{'='*70}")

        for run in range(N_RUNS):
            # Create fresh shadow estimators and sample
            shadow1 = ShadowEstimator(mf, fci_solver)
            shadow2 = ShadowEstimator(mf, fci_solver)
            shadow1.sample(n_shadows // 2, N_K_ESTIMATORS)
            shadow2.sample(n_shadows // 2, N_K_ESTIMATORS)

            # (a) Mean path (n_batches=1)
            mc_mean = MonteCarloEstimator((shadow1, shadow2), sampler)
            res_mean = run_shadow_parallel(
                mc_mean, mf, norb, E2_ref, rdm2_ref, n_batches=1,
                run_idx=run, label="mean",
            )
            shadow_mean_E2[sc_idx, run] = res_mean["E2"]
            shadow_mean_signed_err[sc_idx, run] = res_mean["signed_err"]
            shadow_mean_rel_frob[sc_idx, run] = res_mean["rel_frob"]

            # (b) MoM path (n_batches=N_MC_BATCHES_MOM)
            mc_mom = MonteCarloEstimator((shadow1, shadow2), sampler)
            res_mom = run_shadow_parallel(
                mc_mom, mf, norb, E2_ref, rdm2_ref, n_batches=N_MC_BATCHES_MOM,
                run_idx=run, label="MoM",
            )
            shadow_mom_E2[sc_idx, run] = res_mom["E2"]
            shadow_mom_signed_err[sc_idx, run] = res_mom["signed_err"]
            shadow_mom_rel_frob[sc_idx, run] = res_mom["rel_frob"]

            shadow1.clear_sample()
            shadow2.clear_sample()

    # ========== Save data ==========
    npz_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        exact_E2=exact_E2,
        exact_signed_err=exact_signed_err,
        exact_rel_frob=exact_rel_frob,
        exact_converged_at=exact_converged_at,
        shadow_mean_E2=shadow_mean_E2,
        shadow_mean_signed_err=shadow_mean_signed_err,
        shadow_mean_rel_frob=shadow_mean_rel_frob,
        shadow_mom_E2=shadow_mom_E2,
        shadow_mom_signed_err=shadow_mom_signed_err,
        shadow_mom_rel_frob=shadow_mom_rel_frob,
    )
    print(f"\nSaved: {npz_path}")

    metadata = {
        "system": f"H{N_HYDROGEN} chain",
        "bond_length_angstrom": float(BOND_LENGTH),
        "basis_set": str(BASIS_SET),
        "norb": int(norb),
        "E_hf": float(E_hf),
        "E_fci": float(E_fci),
        "E2_ref": float(E2_ref),
        "rdm2_ref_norm": rdm2_ref_norm,
        "mps_bond_dim": int(MPS_BOND_DIM),
        "mps_prob_cutoff": MPS_PROB_CUTOFF,
        "shadow_counts": SHADOW_COUNTS,
        "n_k_estimators": int(N_K_ESTIMATORS),
        "n_mc_iters": int(N_MC_ITERS),
        "n_mc_batches_mom": int(N_MC_BATCHES_MOM),
        "n_workers": int(N_WORKERS),
        "max_mc_iters_exact": int(MAX_MC_ITERS_EXACT),
        "conv_window": int(CONV_WINDOW),
        "conv_threshold": float(CONV_THRESHOLD),
        "check_every": int(CHECK_EVERY),
        "n_runs": int(N_RUNS),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # ========== Plots (4-panel figure) ==========
    setup_plotting_style()

    exact_mean_err = exact_signed_err.mean()
    exact_sem = exact_signed_err.std(ddof=1) / np.sqrt(N_RUNS)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # --- Top-left: Signed E2 error vs shadow count (mean path) ---
    ax = axes[0, 0]
    for sc_idx, ns in enumerate(SHADOW_COUNTS):
        errs = shadow_mean_signed_err[sc_idx]
        x_jitter = ns * np.ones(N_RUNS) + np.random.uniform(-0.02 * ns, 0.02 * ns, N_RUNS)
        ax.scatter(x_jitter, errs, alpha=0.3, s=12, color="C0", zorder=2)
        mu = errs.mean()
        sem = errs.std(ddof=1) / np.sqrt(N_RUNS)
        ax.errorbar(ns, mu, yerr=2 * sem, fmt="D", color="C1", markersize=5,
                    capsize=4, capthick=1.5, linewidth=1.5, zorder=5)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.axhspan(exact_mean_err - 2 * exact_sem, exact_mean_err + 2 * exact_sem,
               alpha=0.15, color="gray", label="Exact $\\pm 2$ SEM")
    ax.set_xscale("log")
    ax.set_xlabel(r"Shadow samples $N$")
    ax.set_ylabel(r"$E_2 - E_2^{\mathrm{ref}}$ (Ha)")
    ax.set_title("Signed $E_2$ error (mean)")
    ax.legend(fontsize=7)

    # --- Top-right: Signed E2 error vs shadow count (MoM path) ---
    ax = axes[0, 1]
    for sc_idx, ns in enumerate(SHADOW_COUNTS):
        errs = shadow_mom_signed_err[sc_idx]
        x_jitter = ns * np.ones(N_RUNS) + np.random.uniform(-0.02 * ns, 0.02 * ns, N_RUNS)
        ax.scatter(x_jitter, errs, alpha=0.3, s=12, color="C0", zorder=2)
        mu = errs.mean()
        sem = errs.std(ddof=1) / np.sqrt(N_RUNS)
        ax.errorbar(ns, mu, yerr=2 * sem, fmt="D", color="C1", markersize=5,
                    capsize=4, capthick=1.5, linewidth=1.5, zorder=5)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.axhspan(exact_mean_err - 2 * exact_sem, exact_mean_err + 2 * exact_sem,
               alpha=0.15, color="gray", label="Exact $\\pm 2$ SEM")
    ax.set_xscale("log")
    ax.set_xlabel(r"Shadow samples $N$")
    ax.set_ylabel(r"$E_2 - E_2^{\mathrm{ref}}$ (Ha)")
    ax.set_title("Signed $E_2$ error (MoM)")
    ax.legend(fontsize=7)

    # --- Bottom-left: Relative Frobenius error vs shadow count ---
    ax = axes[1, 0]
    for label_name, data, color, marker in [
        ("Mean", shadow_mean_rel_frob, "C0", "o"),
        ("MoM", shadow_mom_rel_frob, "C2", "s"),
    ]:
        medians = np.median(data, axis=1)
        q25 = np.percentile(data, 25, axis=1)
        q75 = np.percentile(data, 75, axis=1)
        ax.errorbar(
            SHADOW_COUNTS, medians,
            yerr=[medians - q25, q75 - medians],
            fmt=f"{marker}-", ms=4, capsize=3, color=color, label=label_name,
        )
    exact_frob_med = np.median(exact_rel_frob)
    exact_frob_q25 = np.percentile(exact_rel_frob, 25)
    exact_frob_q75 = np.percentile(exact_rel_frob, 75)
    ax.axhline(exact_frob_med, color="k", ls="--", lw=1, label="Exact")
    ax.axhspan(exact_frob_q25, exact_frob_q75, alpha=0.15, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shadow samples $N$")
    ax.set_ylabel(r"Rel. Frobenius error in $\Gamma^{(2)}$")
    ax.set_title("2-RDM accuracy")
    ax.legend(fontsize=7)

    # --- Bottom-right: Relative E2 error (absolute) vs shadow count ---
    ax = axes[1, 1]
    for label_name, data_signed, color, marker in [
        ("Mean", shadow_mean_signed_err, "C0", "o"),
        ("MoM", shadow_mom_signed_err, "C2", "s"),
    ]:
        abs_rel_err = np.abs(data_signed) / np.abs(E2_ref)
        medians = np.median(abs_rel_err, axis=1)
        q25 = np.percentile(abs_rel_err, 25, axis=1)
        q75 = np.percentile(abs_rel_err, 75, axis=1)
        ax.errorbar(
            SHADOW_COUNTS, medians,
            yerr=[medians - q25, q75 - medians],
            fmt=f"{marker}-", ms=4, capsize=3, color=color, label=label_name,
        )
    exact_abs_rel = np.abs(exact_signed_err) / np.abs(E2_ref)
    exact_are_med = np.median(exact_abs_rel)
    exact_are_q25 = np.percentile(exact_abs_rel, 25)
    exact_are_q75 = np.percentile(exact_abs_rel, 75)
    ax.axhline(exact_are_med, color="k", ls="--", lw=1, label="Exact")
    ax.axhspan(exact_are_q25, exact_are_q75, alpha=0.15, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shadow samples $N$")
    ax.set_ylabel(r"Rel. error in $E_2$")
    ax.set_title("Doubles energy accuracy")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(os.path.join(output_dir, "shadow_vs_exact.pdf"))
    save_figure(os.path.join(output_dir, "shadow_vs_exact.png"))
    plt.close()

    # ========== Summary table ==========
    print("\n" + "=" * 90)
    print("Summary Statistics (Signed E2 Error)")
    print("=" * 90)
    print(f"{'Setting':<24} {'Mean err':>12} {'SEM':>12} {'Std':>12} {'N>0':>6} {'N<0':>6}")
    print("-" * 90)

    errs = exact_signed_err
    mu = errs.mean()
    sem = errs.std(ddof=1) / np.sqrt(N_RUNS)
    std = errs.std(ddof=1)
    n_pos = int(np.sum(errs > 0))
    n_neg = int(np.sum(errs < 0))
    print(f"{'Exact (converged)':<24} {mu:>+12.6f} {sem:>12.6f} {std:>12.6f} {n_pos:>6} {n_neg:>6}")

    for sc_idx, ns in enumerate(SHADOW_COUNTS):
        for method, data in [("Mean", shadow_mean_signed_err), ("MoM", shadow_mom_signed_err)]:
            errs = data[sc_idx]
            mu = errs.mean()
            sem = errs.std(ddof=1) / np.sqrt(N_RUNS)
            std = errs.std(ddof=1)
            n_pos = int(np.sum(errs > 0))
            n_neg = int(np.sum(errs < 0))
            label = f"Shadow {method} N={ns}"
            print(f"{label:<24} {mu:>+12.6f} {sem:>12.6f} {std:>12.6f} {n_pos:>6} {n_neg:>6}")

    print("\n" + "=" * 90)
    print("Done!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 90)


if __name__ == "__main__":
    main()

"""Simultaneous variance scaling of median-of-means shadow estimates.

For an H8 chain, we repeatedly collect classical shadows and use the
median-of-means protocol to estimate overlaps with an increasing number
M of Slater determinants.  Classical shadow theory predicts that the
sample complexity for *simultaneous* estimation of M observables grows
as O(log M).  Equivalently, for a fixed shadow budget the maximum
estimation error across M determinants should scale as sqrt(log M).

Protocol
--------
1. Set up H8 / STO-3G, run FCI via FCISolver.
2. Build determinant set: HF + singles + doubles excitations.
3. Repeat R times:
   a. Collect N shadow samples split into K median-of-means estimators.
   b. For each determinant, call protocol.estimate_overlap() which
      computes the median of K group means — the actual MoM estimator.
   c. Record the MoM estimate for every determinant.
4. Across R repeats, compute per-determinant variance and absolute
   error of the MoM estimator.
5. For subset sizes M = 1, 2, 5, ..., M_total (ordered by decreasing
   exact overlap magnitude), record max/mean variance and error.
6. Fit and plot: max error ~ a * sqrt(log M) + b.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime
from pyscf import gto, scf

from shades.solvers import FCISolver
from shades.excitations import get_hf_reference, get_singles, get_doubles
from shades.utils import make_hydrogen_chain
from shades.tomography import ShadowProtocol

from plotting_config import setup_plotting_style, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

DEFAULT_OUTPUT_DIR = f"./results/variance_scaling/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

N_HYDROGEN = 8
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"

N_SHADOWS = 5000
N_K_ESTIMATORS = 20

N_REPEATS = 10


def collect_determinants(mf):
    """Return sorted array of unique determinant integers (HF + singles + doubles)."""
    hf = get_hf_reference(mf)
    singles = get_singles(mf)
    doubles = get_doubles(mf)

    det_set = {hf}
    for exc in singles:
        det_set.add(exc.bitstring)
    for exc in doubles:
        det_set.add(exc.bitstring)

    return np.array(sorted(det_set), dtype=np.int64)


def exact_overlaps(statevector, dets):
    """Compute exact overlaps Re(<det|psi>) for each determinant."""
    data = statevector.data
    return np.array([data[d].real for d in dets])


def make_subset_sizes(M_total):
    """Generate a geometric sequence of subset sizes from 1 to M_total."""
    sizes = sorted(set(
        [1, 2, 5] +
        list(np.unique(np.geomspace(1, M_total, num=25).astype(int))) +
        [M_total]
    ))
    return [s for s in sizes if s <= M_total]


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Median-of-Means Simultaneous Variance Scaling with Set Size M")
    print("=" * 70)

    hstring = make_hydrogen_chain(N_HYDROGEN, BOND_LENGTH)
    mol = gto.Mole()
    mol.build(atom=hstring, basis=BASIS_SET, verbose=0)
    mf = scf.RHF(mol)
    mf.run()

    fci_solver = FCISolver(mf)
    psi, E_fci = fci_solver.solve()

    norb = mf.mo_coeff.shape[1]
    n_qubits = 2 * norb
    print(f"\nMolecule: H{N_HYDROGEN} chain (r = {BOND_LENGTH:.2f} A)")
    print(f"Basis: {BASIS_SET}, norb = {norb}, nqubits = {n_qubits}")
    print(f"E_HF  = {mf.e_tot:.10f} Ha")
    print(f"E_FCI = {E_fci:.10f} Ha")

    dets = collect_determinants(mf)
    M_total = len(dets)
    print(f"\nDeterminants: {M_total} (HF + singles + doubles)")

    exact = exact_overlaps(psi, dets)
    order = np.argsort(-np.abs(exact))
    dets = dets[order]
    exact = exact[order]
    print(f"Largest  |<det|psi>| = {np.abs(exact[0]):.6f}")
    print(f"Smallest |<det|psi>| = {np.abs(exact[-1]):.2e}")

    print(f"\nShadow budget: N = {N_SHADOWS}, K = {N_K_ESTIMATORS} "
          f"({N_SHADOWS // N_K_ESTIMATORS} samples/estimator)")
    print(f"Repeats: {N_REPEATS}")

    subset_sizes = make_subset_sizes(M_total)
    print(f"Subset sizes: {subset_sizes}")

    # (N_REPEATS, M_total) — one MoM estimate per det per repeat
    all_mom_estimates = np.empty((N_REPEATS, M_total))

    for rep in range(N_REPEATS):
        print(f"\n--- Repeat {rep + 1}/{N_REPEATS} ---")

        protocol = ShadowProtocol(psi)
        protocol.collect_samples_for_overlaps(N_SHADOWS, N_K_ESTIMATORS)

        print("  Computing median-of-means overlaps...")
        for j, det in enumerate(dets):
            all_mom_estimates[rep, j] = protocol.estimate_overlap(int(det))
        protocol._close_pool()

        max_err = np.max(np.abs(all_mom_estimates[rep] - exact))
        print(f"  max |MoM - exact| = {max_err:.4e}")

    # Per-determinant statistics across repeats
    mom_var = np.var(all_mom_estimates, axis=0)       # (M_total,) variance of MoM estimator
    mom_mean = np.mean(all_mom_estimates, axis=0)     # (M_total,) mean MoM estimate
    mom_bias = mom_mean - exact                       # (M_total,) bias
    mom_mse = np.mean((all_mom_estimates - exact[None, :]) ** 2, axis=0)  # (M_total,) MSE

    # For each repeat: per-determinant absolute error
    abs_errors = np.abs(all_mom_estimates - exact[None, :])  # (N_REPEATS, M_total)

    # Compute scaling metrics for each subset size
    max_var = np.empty(len(subset_sizes))
    mean_var = np.empty(len(subset_sizes))
    max_mse = np.empty(len(subset_sizes))

    # Per-repeat max error for each subset size → gives distribution
    max_err_runs = np.empty((N_REPEATS, len(subset_sizes)))

    for i, M in enumerate(subset_sizes):
        max_var[i] = np.max(mom_var[:M])
        mean_var[i] = np.mean(mom_var[:M])
        max_mse[i] = np.max(mom_mse[:M])
        for rep in range(N_REPEATS):
            max_err_runs[rep, i] = np.max(abs_errors[rep, :M])

    M_arr = np.array(subset_sizes, dtype=float)
    max_err_med = np.median(max_err_runs, axis=0)
    max_err_lo = np.percentile(max_err_runs, 25, axis=0)
    max_err_hi = np.percentile(max_err_runs, 75, axis=0)
    mean_err_med = np.median(np.mean(abs_errors, axis=1, keepdims=True)
                             * np.ones((1, len(subset_sizes))), axis=0)

    # Compute mean abs error per subset (not just global mean)
    mean_err_per_M = np.empty(len(subset_sizes))
    for i, M in enumerate(subset_sizes):
        mean_err_per_M[i] = np.median(np.mean(abs_errors[:, :M], axis=1))

    # Fit max error vs sqrt(log M)
    mask = M_arr > 1
    sqrt_log_M = np.sqrt(np.log(M_arr[mask]))
    fit_err = np.polyfit(sqrt_log_M, max_err_med[mask], 1)
    fit_var = np.polyfit(np.log(M_arr[mask]), max_var[mask], 1)
    print(f"\nLinear fit: max |err| ~ {fit_err[0]:.4e} * sqrt(ln M) + {fit_err[1]:.4e}")
    print(f"Linear fit: max var  ~ {fit_var[0]:.4e} * ln(M) + {fit_var[1]:.4e}")

    results = {
        "system": f"H{N_HYDROGEN}",
        "bond_length": BOND_LENGTH,
        "basis": BASIS_SET,
        "n_shadows": N_SHADOWS,
        "n_k_estimators": N_K_ESTIMATORS,
        "n_repeats": N_REPEATS,
        "n_determinants_total": int(M_total),
        "subset_sizes": [int(s) for s in subset_sizes],
        "max_mom_var": max_var.tolist(),
        "mean_mom_var": mean_var.tolist(),
        "max_mom_mse": max_mse.tolist(),
        "max_err_median": max_err_med.tolist(),
        "max_err_q25": max_err_lo.tolist(),
        "max_err_q75": max_err_hi.tolist(),
        "mean_err_per_M": mean_err_per_M.tolist(),
        "fit_max_err_slope": float(fit_err[0]),
        "fit_max_err_intercept": float(fit_err[1]),
        "fit_max_var_slope": float(fit_var[0]),
        "fit_max_var_intercept": float(fit_var[1]),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}results.json")

    # --- Plots ---
    setup_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Left panel: MoM variance scaling
    ax = axes[0]
    ax.plot(M_arr, max_var, "o-", ms=3, label=r"$\max_j \mathrm{Var}[\hat{c}_j^{\mathrm{MoM}}]$")
    ax.plot(M_arr, mean_var, "s--", ms=3, label=r"$\mathrm{mean}_j \mathrm{Var}$")
    M_fit = np.linspace(2, M_arr[-1], 200)
    ax.plot(M_fit, fit_var[0] * np.log(M_fit) + fit_var[1], "k:",
            label=rf"${fit_var[0]:.2e} \ln M + b$", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Subset size $M$")
    ax.set_ylabel(r"Variance of MoM estimator")
    ax.set_title("MoM variance vs set size")
    ax.legend(fontsize=7)

    # Right panel: simultaneous max error scaling
    ax = axes[1]
    ax.plot(M_arr, max_err_med, "o-", ms=3,
            label=r"$\mathrm{median}_r \, \max_j |\hat{c}_j - c_j|$")
    ax.fill_between(M_arr, max_err_lo, max_err_hi, alpha=0.25)
    ax.plot(M_arr, mean_err_per_M, "s--", ms=3, label=r"$\mathrm{mean}_j |\hat{c}_j - c_j|$")
    ax.plot(M_fit, fit_err[0] * np.sqrt(np.log(M_fit)) + fit_err[1], "k:",
            label=rf"${fit_err[0]:.2e} \sqrt{{\ln M}} + b$", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Subset size $M$")
    ax.set_ylabel("Absolute error")
    ax.set_title("Simultaneous MoM error")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(os.path.join(output_dir, "variance_scaling.pdf"))
    plt.close()

    # Dedicated fit plot: max error vs sqrt(ln M)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(sqrt_log_M, max_err_med[mask], "o", ms=4)
    ax.plot(sqrt_log_M, fit_err[0] * sqrt_log_M + fit_err[1], "r-",
            label=rf"slope $= {fit_err[0]:.3e}$")
    ax.set_xlabel(r"$\sqrt{\ln M}$")
    ax.set_ylabel(r"$\max_j |\hat{c}_j - c_j|$ (median over repeats)")
    ax.set_title(r"Max MoM error vs $\sqrt{\ln M}$")
    ax.legend()
    fig.tight_layout()
    save_figure(os.path.join(output_dir, "max_err_vs_sqrt_logM.pdf"))
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()

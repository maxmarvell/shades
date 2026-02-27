"""Profile Monte Carlo 2-RDM estimation with shadow + MPS sampler.

Breaks down wall-clock time per stage for H-chain system sizes 2, 4, 6, 8.
Includes per-iteration timing traces and cache saturation analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "monte_carlo_2rdm", "shadow"))

import time
import cProfile
import pstats
import io
import math
import numpy as np
import logging
from pyscf import gto, scf

from shades.solvers import FCISolver
from shades.estimators import ShadowEstimator
from shades.utils import make_hydrogen_chain
from shades.monte_carlo import MPSSampler, MonteCarloEstimator, _gen_single_site_hops, _gen_double_site_hops

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

N_HYDROGEN_SIZES = [2, 4, 6, 8]
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"

N_SHADOWS = 200
N_K_ESTIMATORS = 20
N_MC_ITERS = 200  # reduced for profiling
N_OVERLAP_BENCH = 500  # number of overlap calls to benchmark
MPS_BOND_DIM = 300  # match production system_size.py


def time_it(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def profile_system(n_h):
    print(f"\n{'='*70}")
    print(f"  PROFILING H{n_h} chain  ({BASIS_SET}, r={BOND_LENGTH} A)")
    print(f"{'='*70}")

    hstring = make_hydrogen_chain(n_h, BOND_LENGTH)
    mol = gto.Mole()
    mol.build(atom=hstring, basis=BASIS_SET, verbose=0)
    mf = scf.RHF(mol)
    mf.run()
    norb = mf.mo_coeff.shape[1]
    n_qubits = 2 * norb
    nelec = mf.mol.nelec
    print(f"  norb={norb}, n_qubits={n_qubits}, nelec={nelec}")

    # --- Stage 1: FCI solver ---
    t0 = time.perf_counter()
    fci_solver = FCISolver(mf)
    fci_solver.solve()
    t_fci = time.perf_counter() - t0
    print(f"\n  [1] FCI solve:                {t_fci:10.3f} s")

    # --- Stage 2: MPS sampler (DMRG + CSF extraction) ---
    t0 = time.perf_counter()
    mps_sampler = MPSSampler(mf, max_bond_dim=MPS_BOND_DIM)
    t_mps = time.perf_counter() - t0
    print(f"  [2] MPS sampler (DMRG+CSF):   {t_mps:10.3f} s")
    print(f"       num determinants:         {len(mps_sampler.get_distribution()[0])}")

    # --- Stage 3: Shadow sampling ---
    shadow1 = ShadowEstimator(mf, fci_solver)
    shadow2 = ShadowEstimator(mf, fci_solver)

    t0 = time.perf_counter()
    shadow1.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
    t_shadow1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    shadow2.sample(N_SHADOWS // 2, N_K_ESTIMATORS)
    t_shadow2 = time.perf_counter() - t0

    t_shadow = t_shadow1 + t_shadow2
    print(f"  [3] Shadow sampling (2x{N_SHADOWS//2}):  {t_shadow:10.3f} s")
    print(f"       per-sample:               {t_shadow / N_SHADOWS * 1e3:10.3f} ms")

    # --- Stage 4: Overlap estimation benchmark ---
    # Pick a set of bitstrings to estimate overlaps for
    hf_int = sum(1 << i for i in range(nelec[0])) + sum(1 << (norb + i) for i in range(nelec[1]))
    test_bitstrings = [hf_int]
    # Add some single-hop bitstrings
    hops, _ = _gen_single_site_hops(hf_int, n_qubits)
    for h in hops[:min(20, len(hops))]:
        test_bitstrings.append(h)

    shadow1._amplitude_cache = {}  # clear cache
    n_calls = min(N_OVERLAP_BENCH, len(test_bitstrings))
    overlap_times = []
    for bs in test_bitstrings[:n_calls]:
        t0 = time.perf_counter()
        shadow1.estimate_overlap(bs)
        overlap_times.append(time.perf_counter() - t0)
    # clear cache and do a second pass to get uncached times
    shadow1._amplitude_cache = {}
    uncached_times = []
    for bs in test_bitstrings[:n_calls]:
        t0 = time.perf_counter()
        shadow1.estimate_overlap(bs)
        uncached_times.append(time.perf_counter() - t0)

    print(f"  [4] Overlap estimation ({n_calls} calls):")
    print(f"       mean per call:            {np.mean(uncached_times)*1e3:10.3f} ms")
    print(f"       median per call:          {np.median(uncached_times)*1e3:10.3f} ms")
    print(f"       total:                    {sum(uncached_times):10.3f} s")

    # --- Stage 5: Hop generation benchmark ---
    t0 = time.perf_counter()
    for _ in range(100):
        _gen_single_site_hops(hf_int, n_qubits)
    t_single_hops = (time.perf_counter() - t0) / 100
    n_single_hops = len(_gen_single_site_hops(hf_int, n_qubits)[0])

    t0 = time.perf_counter()
    for _ in range(100):
        _gen_double_site_hops(hf_int, n_qubits)
    t_double_hops = (time.perf_counter() - t0) / 100
    n_double_hops = len(_gen_double_site_hops(hf_int, n_qubits)[0])

    print(f"  [5] Hop generation (from HF ref):")
    print(f"       single hops:              {n_single_hops:5d}  ({t_single_hops*1e6:.1f} us)")
    print(f"       double hops:              {n_double_hops:5d}  ({t_double_hops*1e6:.1f} us)")

    # --- Stage 6: MPS sampling benchmark ---
    t0 = time.perf_counter()
    for _ in range(1000):
        mps_sampler.sample()
    t_mps_sample = (time.perf_counter() - t0) / 1000
    print(f"  [6] MPS sampler.sample():      {t_mps_sample*1e6:10.1f} us/call")

    # --- Stage 7: Full MC iteration profiling with cProfile + per-iteration timing ---
    print(f"\n  [7] Full MC estimate_2rdm ({N_MC_ITERS} iters):")
    shadow1._amplitude_cache = {}
    shadow2._amplitude_cache = {}
    estimator = (shadow1, shadow2)
    mc = MonteCarloEstimator(estimator, mps_sampler)

    # Per-iteration timing trace
    iter_times = []
    cache_sizes = []  # (iter, shadow1_cache_size, shadow2_cache_size)
    trace_interval = max(1, N_MC_ITERS // 20)  # record cache sizes ~20 times
    _last_time = [time.perf_counter()]

    def timing_callback(i, gamma):
        now = time.perf_counter()
        iter_times.append(now - _last_time[0])
        _last_time[0] = now
        if i % trace_interval == 0 or i == N_MC_ITERS - 1:
            cache_sizes.append((i, len(shadow1._amplitude_cache), len(shadow2._amplitude_cache)))

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    rdm2 = mc.estimate_2rdm(max_iters=N_MC_ITERS, callback=timing_callback)
    t_mc = time.perf_counter() - t0
    pr.disable()

    print(f"       total wall time:          {t_mc:10.3f} s")
    print(f"       per iteration:            {t_mc/N_MC_ITERS*1e3:10.3f} ms")

    # Estimate overlap calls per iteration
    n_overlaps_per_iter = n_single_hops + n_double_hops + 2  # +2 for c_n and density-density c_m
    print(f"       est. overlap calls/iter:  ~{n_single_hops + n_double_hops + 2}")
    print(f"       est. overlap time/iter:   {(n_single_hops + n_double_hops + 2)*np.mean(uncached_times)*1e3:.1f} ms (uncached)")

    # Per-iteration timing analysis
    if len(iter_times) >= 20:
        cold_times = iter_times[:10]
        warm_times = iter_times[-10:]
        print(f"\n  [7a] Per-iteration timing (cold vs warm cache):")
        print(f"       first 10 iters (cold):    {np.mean(cold_times)*1e3:10.3f} ms/iter  (std: {np.std(cold_times)*1e3:.3f} ms)")
        print(f"       last 10 iters (warm):     {np.mean(warm_times)*1e3:10.3f} ms/iter  (std: {np.std(warm_times)*1e3:.3f} ms)")
        print(f"       speedup (cold/warm):      {np.mean(cold_times)/np.mean(warm_times):10.1f}x")
    if iter_times:
        print(f"       min iter time:            {min(iter_times)*1e3:10.3f} ms")
        print(f"       max iter time:            {max(iter_times)*1e3:10.3f} ms")

    # Cache growth trace
    if cache_sizes:
        print(f"\n  [7b] Cache growth trace:")
        print(f"       {'iter':>6}  {'shadow1':>8}  {'shadow2':>8}  {'total':>8}")
        for it, s1, s2 in cache_sizes:
            print(f"       {it:>6}  {s1:>8}  {s2:>8}  {s1+s2:>8}")

    # Print top cProfile entries
    print(f"\n  --- cProfile top 20 (cumulative) ---")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print(f"\n  --- cProfile top 20 (tottime) ---")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(20)
    print(s.getvalue())

    # --- Stage 8: Cache and Hilbert space analysis ---
    nalpha, nbeta = nelec
    hilbert_size = math.comb(norb, nalpha) * math.comb(norb, nbeta)
    n_mps_dets = len(mps_sampler.get_distribution()[0])
    s1_cache = len(shadow1._amplitude_cache)
    s2_cache = len(shadow2._amplitude_cache)

    print(f"  [8] Cache & Hilbert space analysis:")
    print(f"       Hilbert space size:       {hilbert_size}")
    print(f"       MPS unique determinants:  {n_mps_dets}")
    print(f"       shadow1 cache size:       {s1_cache}")
    print(f"       shadow2 cache size:       {s2_cache}")
    print(f"       total cached overlaps:    {s1_cache + s2_cache}")
    print(f"       cache / Hilbert %%:        {(s1_cache + s2_cache) / (2 * hilbert_size) * 100:.1f}%")

    # Estimate time for 100K iterations given cache behaviour
    if len(iter_times) >= 20:
        warm_rate = np.mean(warm_times)
        cold_rate = np.mean(cold_times)
        # Rough model: first ~cache_size iters are cold, rest are warm
        n_cold = max(s1_cache, s2_cache)  # approximate cold iterations
        n_warm_100k = max(0, 100_000 - n_cold)
        est_100k = n_cold * cold_rate + n_warm_100k * warm_rate
        print(f"\n  [8a] Projected wall time for 100K iters:")
        print(f"       est. cold iters:          {n_cold}")
        print(f"       est. warm iters:          {n_warm_100k}")
        print(f"       est. total time:          {est_100k:.1f} s  ({est_100k/3600:.2f} h)")
        print(f"       (assumes cache saturates at ~{n_cold} iters)")

    return {
        "n_h": n_h,
        "norb": norb,
        "n_qubits": n_qubits,
        "t_fci": t_fci,
        "t_mps_init": t_mps,
        "t_shadow": t_shadow,
        "t_overlap_mean_ms": np.mean(uncached_times) * 1e3,
        "t_single_hops_us": t_single_hops * 1e6,
        "t_double_hops_us": t_double_hops * 1e6,
        "n_single_hops": n_single_hops,
        "n_double_hops": n_double_hops,
        "t_mps_sample_us": t_mps_sample * 1e6,
        "t_mc_total": t_mc,
        "t_mc_per_iter_ms": t_mc / N_MC_ITERS * 1e3,
        "n_mps_dets": n_mps_dets,
        "hilbert_size": hilbert_size,
        "s1_cache": s1_cache,
        "s2_cache": s2_cache,
        "cold_ms": np.mean(cold_times) * 1e3 if len(iter_times) >= 20 else None,
        "warm_ms": np.mean(warm_times) * 1e3 if len(iter_times) >= 20 else None,
    }


# Workaround: FCISolver doesn't have a tap() method
def main():
    all_results = []
    for n_h in N_HYDROGEN_SIZES:
        result = profile_system(n_h)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    header = (
        f"{'N_H':>4} {'norb':>5} {'nq':>4} {'FCI':>8} {'MPS init':>9} {'Shadow':>8} "
        f"{'Overlap':>10} {'MC total':>9} {'MC/iter':>10} {'#s_hop':>7} {'#d_hop':>7} "
        f"{'cold':>8} {'warm':>8}"
    )
    units = (
        f"{'':>4} {'':>5} {'':>4} {'(s)':>8} {'(s)':>9} {'(s)':>8} "
        f"{'(ms/call)':>10} {'(s)':>9} {'(ms)':>10} {'':>7} {'':>7} "
        f"{'(ms)':>8} {'(ms)':>8}"
    )
    print(header)
    print(units)
    print("-" * 110)
    for r in all_results:
        cold_str = f"{r['cold_ms']:>8.1f}" if r.get('cold_ms') is not None else f"{'N/A':>8}"
        warm_str = f"{r['warm_ms']:>8.1f}" if r.get('warm_ms') is not None else f"{'N/A':>8}"
        print(
            f"{r['n_h']:>4} {r['norb']:>5} {r['n_qubits']:>4} "
            f"{r['t_fci']:>8.3f} {r['t_mps_init']:>9.3f} {r['t_shadow']:>8.3f} "
            f"{r['t_overlap_mean_ms']:>10.3f} {r['t_mc_total']:>9.3f} {r['t_mc_per_iter_ms']:>10.3f} "
            f"{r['n_single_hops']:>7} {r['n_double_hops']:>7} "
            f"{cold_str} {warm_str}"
        )

    # Bottleneck analysis
    print("\n" + "=" * 110)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 110)
    for r in all_results:
        total = r['t_fci'] + r['t_mps_init'] + r['t_shadow'] + r['t_mc_total']
        print(f"\n  H{r['n_h']} (total ~ {total:.1f}s):")
        print(f"    FCI solve:     {r['t_fci']:8.3f}s  ({r['t_fci']/total*100:5.1f}%)")
        print(f"    MPS init:      {r['t_mps_init']:8.3f}s  ({r['t_mps_init']/total*100:5.1f}%)")
        print(f"    Shadow sample: {r['t_shadow']:8.3f}s  ({r['t_shadow']/total*100:5.1f}%)")
        print(f"    MC estimate:   {r['t_mc_total']:8.3f}s  ({r['t_mc_total']/total*100:5.1f}%)")
        print(f"    Overlap calls dominate MC: ~{r['n_single_hops'] + r['n_double_hops']} calls/iter x {r['t_overlap_mean_ms']:.2f} ms = {(r['n_single_hops'] + r['n_double_hops']) * r['t_overlap_mean_ms']:.0f} ms/iter")
        print(f"    Hilbert space: {r['hilbert_size']}, MPS dets: {r['n_mps_dets']}, cached: {r['s1_cache']}+{r['s2_cache']}")


if __name__ == "__main__":
    main()

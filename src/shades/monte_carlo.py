from itertools import product, combinations
from functools import lru_cache
from multiprocessing import Pool
from typing import List, Tuple, Callable, Any, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import random
from pyscf.scf.hf import RHF
from pyscf.scf.uhf import UHF
from pyscf import ao2mo
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from shades.estimators import AbstractEstimator


@lru_cache(maxsize=4096)
def _gen_single_site_hops(
    reference: int,
    n_qubits: int,
    symm_type: str = 'SZ'
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """
    Given a reference single slater determinant as a bitstring, generate all
    possible nearest neighbours and the hop required to get there.
    """

    assert n_qubits % 2 == 0

    res = []
    n_spatial = n_qubits // 2

    alpha_occ = [i for i in range(n_spatial) if (reference >> i) & 1]
    alpha_vac = [i for i in range(n_spatial) if not (reference >> i) & 1]
    alpha_hops = [(o, v) for o, v in product(alpha_occ, alpha_vac)]
    res += [reference ^ (1 << i) ^ (1 << j) for i, j in alpha_hops]

    if symm_type == 'SU2': return tuple(res), tuple(alpha_hops)

    beta_occ = [i for i in range(n_spatial, n_qubits) if (reference >> i) & 1]
    beta_vac = [i for i in range(n_spatial, n_qubits) if not (reference >> i) & 1]
    beta_hops = [(o, v) for o, v in product(beta_occ, beta_vac)]
    res += [reference ^ (1 << i) ^ (1 << j) for i in beta_occ for j in beta_vac]

    return tuple(res), tuple(alpha_hops + beta_hops)


@lru_cache(maxsize=4096)
def _gen_double_site_hops(
    reference: int,
    n_qubits: int,
    symm_type: str = 'SZ'
) -> Tuple[Tuple[int, ...], Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]]:
    """
    Given a reference single slater determinant as a bitstring, generate all
    possible next-nearest nearest neighbours and the hops required to get there.
    """

    #TODO: not tested for UHF where n_alpha and n_beta are not the same
    assert n_qubits % 2 == 0
    norb = n_qubits // 2
    res = []

    alpha_occ = [i for i in range(norb) if (reference >> i) & 1]
    alpha_vac = [i for i in range(norb) if not (reference >> i) & 1]
    double_alpha = [(o, v) for o, v in product(combinations(alpha_occ, 2), combinations(alpha_vac, 2))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in double_alpha]

    beta_occ = [i for i in range(norb, n_qubits) if (reference >> i) & 1]
    beta_vac = [i for i in range(norb, n_qubits) if not (reference >> i) & 1]

    alpha_beta = [(o, v) for o, v in product(product(alpha_occ, beta_occ), product(alpha_vac, beta_vac))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in alpha_beta]

    if symm_type == 'SU2': return tuple(res), tuple(double_alpha + alpha_beta)

    double_beta = [(o, v) for o, v in product(combinations(beta_occ, 2), combinations(beta_vac, 2))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in double_beta]

    # beta_alpha = [(o, v) for o, v in product(product(beta_occ, alpha_occ), product(beta_vac, alpha_vac))]
    # res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in beta_alpha]

    return tuple(res), tuple(double_alpha + alpha_beta + double_beta)


def _compute_fermionic_sign(
    m: int,
    t: Tuple[Tuple[int, int], Tuple[int, int]],
    n: Optional[int] = None
):
    sign = 1
    (i, j), (k, l) = t

    # annhilate at l
    if not (m >> l) & 1:
        return 0
    sign *= (-1) ** (m & ((1 << l) - 1)).bit_count()
    m ^= (1 << l)

    # annhilate at k
    if not (m >> k) & 1:
        return 0

    sign *= (-1) ** (m & ((1 << k) - 1)).bit_count()
    m ^= (1 << k)

    # create at j
    if (m >> j) & 1:
        return 0
    sign *= (-1) ** (m & ((1 << j) - 1)).bit_count()
    m ^= (1 << j)
    
    # create at i
    if (m >> i) & 1:
        return 0
    sign *= (-1) ** (m & ((1 << i) - 1)).bit_count()
    m ^= (1 << i)

    # validate final state if provided
    if n is not None and m != n:
        raise ValueError('The referenced hop does not meet the target state')
    
    return sign


def _assemble_2rdm_from_cache(
    n: int,
    bias_prob: Optional[float],
    overlaps_n: dict,
    overlaps_m: dict,
    n_qubits: int,
) -> np.ndarray:
    """Assemble a 2-RDM estimator from pre-computed overlap dicts (no estimator calls)."""
    gamma = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    c_n = overlaps_n[n]
    if np.abs(c_n) < 1e-15:
        return gamma

    w = (np.abs(c_n) ** 2) / bias_prob if bias_prob is not None else 1.0

    double_hops, double_transitions = _gen_double_site_hops(n, n_qubits, symm_type='SZ')
    single_hops, single_transitions = _gen_single_site_hops(n, n_qubits)

    for m, t in zip(double_hops, double_transitions):
        (i, j), (k, l) = t
        c_m = overlaps_m[m]
        gamma[i, j, l, k] = _compute_fermionic_sign(m, t) * (c_m / c_n) * w

    for m, hop in zip(single_hops, single_transitions):
        i, k = hop
        occupied = [j for j in range(n_qubits) if (m >> j) & 1 and (n >> j) & 1]
        c_m = overlaps_m[m]
        for j in occupied:
            t = ((i, j), (k, j))
            gamma[i, j, j, k] = _compute_fermionic_sign(m, t) * (c_m / c_n) * w

    occupied = [i for i in range(n_qubits) if (n >> i) & 1]
    c_m = overlaps_m[n]
    for i, j in combinations(occupied, 2):
        gamma[i, j, i, j] = (c_m / c_n) * w

    return gamma - gamma.transpose(1, 0, 2, 3) - gamma.transpose(0, 1, 3, 2) + gamma.transpose(1, 0, 3, 2)


def _process_batch(args):
    """Process a batch of MC samples using pre-computed overlaps. Runs in a worker process."""
    samples, overlaps_n, overlaps_m, n_qubits = args
    batch_mean = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for j, (det, prob) in enumerate(samples):
        estimate = _assemble_2rdm_from_cache(det, prob, overlaps_n, overlaps_m, n_qubits)
        batch_mean += (estimate - batch_mean) / (j + 1)
    return batch_mean


class AbstractStochasticSampler(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def sample(self) -> tuple[int, Optional[float]]:
        pass


class WavefunctionSampler(AbstractStochasticSampler):

    def __init__(self, state: np.ndarray, n_qubits: int):
        self.n_qubits = n_qubits
        self.states = list(range(2**self.n_qubits))
        self.probabilities = np.abs(state)**2
        self.probabilities /= self.probabilities.sum()


    def sample(self) -> tuple[int, Optional[float]]:
        return np.random.choice(self.states, p=self.probabilities), None
    

class MetropolisSampler(AbstractStochasticSampler):

    n: Optional[int]

    def __init__(
        self, 
        estimator: AbstractEstimator, 
        transition_fn: Callable[[int, int], Tuple[List[int], List[Tuple[int, int]]]],
        auto_corr_iters: int = 100
    ):
        self.estimator = estimator
        self.n_qubits = estimator.n_qubits
        self.transition_fn = transition_fn
        self.auto_corr_iters = auto_corr_iters

        if not isinstance(self.estimator.mf, RHF):
            raise NotImplementedError('Not implemented yet for unrestricted Hartree-Fock!')

        self.nelec = self.estimator.mf.mol.nelec
        self.n = None


    def initialise(self) -> int:
        # get random valid initial state in the 
        # get random n_qubit bitstring with Hamming weight m 
        # TODO this is only implemented for RHF
        norb = self.n_qubits // 2
        nalpha, nbeta = self.nelec
        n = sum(1 << p for p in random.sample(range(norb), nalpha))
        n += sum(1 << (p + norb) for p in random.sample(range(norb), nbeta))
        return n
    

    def sample(self) -> tuple[int, float]:

        if self.n is None:
            self.n = self.initialise()

        for _ in range(self.auto_corr_iters):
            m, t = self._propose_candidate(self.n)
            if self._accept(self.n, m, t):
                self.n = m

        return self.n, None
    

    def _propose_candidate(self, n: int) -> Tuple[int, Any]:
        candidates, transitions = self.transition_fn(n, self.n_qubits)
        i = random.choice(range(len(candidates)))
        return candidates[i], transitions[i]
    

    def _accept(self, n: int, m: int, t: Any) -> bool:
        # TODO: In general we also need to account fot the transition amplitudes T(m->n)/T(n->m)
        c_n = self.estimator.estimate_overlap(n)
        c_m = self.estimator.estimate_overlap(m)
        P = min(1.0, (c_m * np.conj(c_m)) / (c_n * np.conj(c_n)))
        return random.random() <= P


class MPSSampler(AbstractStochasticSampler):

    def __init__(
        self,
        mf: Union[RHF, UHF],
        max_bond_dim: int = 200,
        batch_size: int = 1000,
        prob_cutoff: Optional[float] = None,
    ):

        if isinstance(mf, UHF):
            raise NotImplementedError()

        self.norb = mf.mo_coeff.shape[1]
        nelec = mf.mol.nelectron
        self.batch_size = batch_size
        self.prob_cutoff = prob_cutoff

        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.kernel(mf.mol, mf.mo_coeff)
        e_core = mf.mol.energy_nuc()

        schedule = [d for d in [50, 100, 200, max_bond_dim] if d <= max_bond_dim]
        if not schedule or schedule[-1] != max_bond_dim:
            schedule.append(max_bond_dim)

        self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ)
        self.driver.initialize_system(n_sites=self.norb, n_elec=nelec, spin=0)
        self.mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=e_core, iprint=0)
        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=max_bond_dim + 50, nroots=1)
        self.driver.dmrg(
            self.mpo, self.ket,
            bond_dims=schedule,
            noises=[1e-4, 1e-5, 0],
            thrds=[1e-8],
            n_sweeps=20,
            iprint=0
        )

        self._cat_dets: Optional[np.ndarray] = None
        self._cat_probs: Optional[np.ndarray] = None

        if prob_cutoff is not None:
            dets, coeffs = self.driver.get_csf_coefficients(self.ket, cutoff=prob_cutoff, iprint=0)
            int_dets = np.array([self._det_to_int(d) for d in dets])
            probs = np.abs(coeffs) ** 2
            probs /= probs.sum()
            self._cat_dets = int_dets
            self._cat_probs = probs
            logger.info(
                f"MPSSampler: prob_cutoff={prob_cutoff}, "
                f"kept {len(int_dets)} determinants (truncated from full distribution)"
            )
        else:
            self._buffer_dets: List[int] = []
            self._buffer_probs: List[float] = []
            self._refill_buffer()

    def _det_to_int(self, det: np.ndarray) -> int:
        norb = self.norb
        alpha = det & 1
        beta = (det >> 1) & 1
        res = sum(1 << i for i in range(norb) if alpha[i] == 1)
        res += sum(1 << i for i in range(norb, 2 * norb) if beta[i - norb] == 1)
        return res

    def _refill_buffer(self):
        dets, coeffs = self.driver.sample_csf_coefficients(
            self.ket, n_sample=self.batch_size, iprint=0
        )
        probs = np.abs(coeffs) ** 2
        self._buffer_dets = [self._det_to_int(d) for d in dets]
        self._buffer_probs = probs.tolist()
        self._buffer_idx = 0

    def get_distribution(self, cutoff: float = 1e-12):
        """Extract the full determinant distribution for analysis.

        Returns (dets, probs) arrays like the old implementation, using
        get_csf_coefficients with the given cutoff.
        """
        if self._cat_dets is not None:
            return self._cat_dets, self._cat_probs
        dets, coeffs = self.driver.get_csf_coefficients(
            self.ket, cutoff=cutoff, iprint=0
        )
        probs = np.abs(coeffs) ** 2
        probs /= probs.sum()
        return dets, probs

    def sample(self) -> tuple[int, float]:
        if self._cat_dets is not None:
            idx = np.random.choice(len(self._cat_dets), p=self._cat_probs)
            return int(self._cat_dets[idx]), float(self._cat_probs[idx])
        if self._buffer_idx >= len(self._buffer_dets):
            self._refill_buffer()
        det = self._buffer_dets[self._buffer_idx]
        prob = self._buffer_probs[self._buffer_idx]
        self._buffer_idx += 1
        return det, prob


IndependentEstimators = tuple[AbstractEstimator, AbstractEstimator]


class MonteCarloEstimator:

    estimators: tuple[AbstractEstimator, Optional[AbstractEstimator]]

    def __init__(
        self,
        estimators: Union[AbstractEstimator, IndependentEstimators],
        sampler: Optional[AbstractStochasticSampler] = None
    ):

        if isinstance(estimators, tuple):
            if len(estimators) != 2:
                raise ValueError("If passing tuple, must provide exactly 2 estimators")
            self.estimators = estimators
            self.estimator = estimators[0]
        else:
            self.estimators = (estimators, None)
            self.estimator = estimators

        self.n_qubits = self.estimator.n_qubits

        if sampler is None:
            sampler = WavefunctionSampler(self.estimator.solver.state.data, self.n_qubits)
        self.sampler = sampler


    def _compute_2rdm_estimator(self, n: int, bias_prob: Optional[float] = None):

        n_qubits = self.n_qubits
        gamma = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        estimator_n, estimator_m = self.estimators
        if estimator_m is None:
            estimator_m = estimator_n  # fall back to single estimator (biased)

        c_n = estimator_n.estimate_overlap(n)

        if np.abs(c_n) < 1e-15:
            return gamma

        w = (np.abs(c_n) ** 2) / bias_prob if bias_prob is not None else 1.0

        # collect all unique m values needed
        unique_m = set()

        double_hops, double_transitions = _gen_double_site_hops(n, n_qubits, symm_type='SZ')
        for m in double_hops:
            unique_m.add(m)

        single_hops, single_transitions = _gen_single_site_hops(n, n_qubits)
        for m in single_hops:
            unique_m.add(m)

        unique_m.add(n)  # needed for density-density terms

        # compute overlaps once per unique m
        overlap_cache = {}
        for m in unique_m:
            overlap_cache[m] = estimator_m.estimate_overlap(m)

        # double hops
        for m, t in zip(double_hops, double_transitions):
            (i, j), (k, l) = t
            c_m = overlap_cache[m]
            gamma[i, j, l, k] = _compute_fermionic_sign(m, t) * (c_m/c_n) * w

        # single hops
        for m, hop in zip(single_hops, single_transitions):
            i, k = hop
            occupied = [j for j in range(n_qubits) if (m >> j) & 1 and (n >> j) & 1]
            c_m = overlap_cache[m]
            for j in occupied:
                t = ((i, j), (k, j))
                gamma[i, j, j, k] = _compute_fermionic_sign(m, t) * (c_m/c_n) * w

        # density-density terms
        occupied = [i for i in range(n_qubits) if (n >> i) & 1]
        c_m = overlap_cache[n]
        for i, j in combinations(occupied, 2):
            gamma[i, j, i, j] = (c_m/c_n) * w

        # antisymmetrise
        return gamma - gamma.transpose(1,0,2,3) - gamma.transpose(0,1,3,2) + gamma.transpose(1,0,3,2)


    def estimate_2rdm(
        self,
        *,
        max_iters: int = 10000,
        n_batches: int = 1,
        callback: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> np.ndarray:

        n_qubits = self.n_qubits

        if n_batches <= 1:
            gamma = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
            for i in range(max_iters):
                n, p = self.sampler.sample()
                estimate = self._compute_2rdm_estimator(n, p)
                gamma += (estimate - gamma) / (i + 1)
                if callback is not None:
                    try:
                        callback(i, gamma)
                    except StopIteration:
                        break
            return gamma

        if max_iters % n_batches != 0:
            raise ValueError("max_iters must be divisible by n_batches")

        batch_size = max_iters // n_batches
        batch_means = np.zeros((n_batches, n_qubits, n_qubits, n_qubits, n_qubits))

        for b in range(n_batches):
            batch_mean = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
            for j in range(batch_size):
                n, p = self.sampler.sample()
                estimate = self._compute_2rdm_estimator(n, p)
                batch_mean += (estimate - batch_mean) / (j + 1)
            batch_means[b] = batch_mean

            if callback is not None:
                gamma = np.median(batch_means[:b + 1], axis=0)
                global_iter = (b + 1) * batch_size - 1
                try:
                    callback(global_iter, gamma)
                except StopIteration:
                    return gamma

        return np.median(batch_means, axis=0)

    def _collect_unique_bitstrings(self, sampled_dets: list) -> set:
        """Scan all sampled determinants and their hop targets, return all unique bitstrings."""
        unique = set()
        for det in sampled_dets:
            unique.add(det)
            single_hops, _ = _gen_single_site_hops(det, self.n_qubits)
            for m in single_hops:
                unique.add(m)
            double_hops, _ = _gen_double_site_hops(det, self.n_qubits, symm_type='SZ')
            for m in double_hops:
                unique.add(m)
        return unique

    def estimate_2rdm_parallel(
        self,
        *,
        max_iters: int = 10000,
        n_batches: int = 100,
        n_workers: int = 4,
        callback: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Parallel 2-RDM estimation using pre-computed overlaps and multiprocessing.

        Phase 1: Pre-sample all determinants sequentially.
        Phase 2: Pre-compute all unique overlaps sequentially (populates cache).
        Phase 3: Distribute batches to worker processes for tensor assembly.
        """
        if max_iters % n_batches != 0:
            raise ValueError("max_iters must be divisible by n_batches")

        n_qubits = self.n_qubits
        estimator_n, estimator_m = self.estimators
        if estimator_m is None:
            estimator_m = estimator_n

        # Phase 1: Pre-sample all determinants
        logger.info("Phase 1: Pre-sampling %d determinants", max_iters)
        all_samples = [self.sampler.sample() for _ in range(max_iters)]

        # Phase 2: Pre-compute all unique overlaps
        # Enable intra-estimator parallelism (e.g. across K shadow estimators)
        if hasattr(estimator_n, 'n_workers'):
            estimator_n.n_workers = n_workers
        if hasattr(estimator_m, 'n_workers'):
            estimator_m.n_workers = n_workers

        sampled_dets = [det for det, _ in all_samples]
        unique_bs = self._collect_unique_bitstrings(sampled_dets)
        logger.info("Phase 2: Computing overlaps for %d unique bitstrings", len(unique_bs))

        overlaps_n = {}
        overlaps_m = {}
        for bs in unique_bs:
            overlaps_n[bs] = estimator_n.estimate_overlap(bs)
            overlaps_m[bs] = estimator_m.estimate_overlap(bs)

        # Phase 3: Parallel tensor assembly
        batch_size = max_iters // n_batches
        chunks = [
            all_samples[b * batch_size : (b + 1) * batch_size]
            for b in range(n_batches)
        ]

        logger.info(
            "Phase 3: Assembling %d batches (size %d) with %d workers",
            n_batches, batch_size, n_workers,
        )

        batch_means_all = []

        # Process in waves of n_workers to support callbacks between waves
        wave_size = n_workers
        for wave_start in range(0, n_batches, wave_size):
            wave_end = min(wave_start + wave_size, n_batches)
            wave_chunks = chunks[wave_start:wave_end]

            worker_args = [
                (chunk, overlaps_n, overlaps_m, n_qubits)
                for chunk in wave_chunks
            ]

            with Pool(processes=min(n_workers, len(wave_chunks))) as pool:
                wave_results = pool.map(_process_batch, worker_args)

            batch_means_all.extend(wave_results)

            if callback is not None:
                gamma = np.median(np.array(batch_means_all), axis=0)
                global_iter = wave_end * batch_size
                try:
                    callback(global_iter, gamma)
                except StopIteration:
                    return gamma

        return np.median(np.array(batch_means_all), axis=0)


if __name__ == "__main__":
    
    from shades.utils import make_hydrogen_chain
    from shades.solvers import FCISolver
    from pyscf import gto, scf

    N_HYDROGEN = 10
    BOND_LENGTH = 1.5
    BASIS_SET = "sto-3g"

    hstring = make_hydrogen_chain(N_HYDROGEN, BOND_LENGTH)
    mol = gto.Mole()
    mol.build(atom=hstring, basis=BASIS_SET, verbose=0)

    mf = scf.RHF(mol)
    mf.run()

    fci = FCISolver(mf)
    fci.solve()

    mps = MPSSampler(mf)

    norb = mf.mo_coeff.shape[1]
    n_qubits = 2 * norb

    amp = fci.state.data
    idx = np.argwhere(np.abs(amp)**2 > 1e-16)   # compare probs to probs
    fci_coeffs = {format(k0, f"0{n_qubits}b")[::-1]: float(np.abs(amp[k0])**2) for k0 in idx[:,0]}

    def _block_2_to_idx(det):
        norb = det.shape[0]
        alpha = det & 1
        beta = (det >> 1) & 1
        res = sum(1 << i for i in range(norb) if alpha[i] == 1)
        res += sum (1 << i for i in range(norb, 2*norb) if beta[i-norb] == 1)
        return res

    mps_dets, mps_probs = mps.get_distribution()
    mps_coeffs = {format(_block_2_to_idx(k), f"0{n_qubits}b")[::-1]: v for k, v in zip(mps_dets, mps_probs)}
    # fci_coeffs = dict(zip(idx, fci.state.data[idx]))
    # mps_coeffs = dict(zip(mps.dets, mps.probs))

    k1 = set(fci_coeffs)
    k2 = set(mps_coeffs)

    print("common:", len(k1 & k2))
    print("only in fci:", k1 - k2)
    print("only in mps:", k2 - k1)

    for k in sorted(k1 & k2):
        v1 = complex(fci_coeffs[k])
        v2 = complex(mps_coeffs[k])
        abs_err = abs(v1 - v2)
        rel_err = abs_err / max(abs(v2), 1e-15)

        print(
            k,
            f"psi1={v1:.6e}",
            f"psi2={v2:.6e}",
            f"|Δ|={abs_err:.3e}",
            f"rel={rel_err:.3e}",
        )


    only = sorted([(k, mps_coeffs[k]) for k in (k2 - k1)], key=lambda x: -x[1])
    for k, p in only[:20]:
        print(k, p)

    mass_only_mps = sum(mps_coeffs[k] for k in (k2 - k1))
    mass_common   = sum(mps_coeffs[k] for k in (k1 & k2))
    print("mass_only_mps:", mass_only_mps)
    print("mass_common:", mass_common)

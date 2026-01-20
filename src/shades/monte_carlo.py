from itertools import product, combinations
from typing import List, Tuple, Callable, Any, Optional, Union
import numpy as np

from abc import ABC, abstractmethod
import random
from pyscf.scf.hf import RHF
from pyscf.scf.uhf import UHF
from pyscf import ao2mo
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from shades.estimators import AbstractEstimator
from shades.utils import Bitstring

def _gen_single_site_hops(
    reference: int, 
    n_qubits: int
) -> Tuple[List[int], List[Tuple[int, int]]]:
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
    
    beta_occ = [i for i in range(n_spatial, n_qubits) if (reference >> i) & 1]
    beta_vac = [i for i in range(n_spatial, n_qubits) if not (reference >> i) & 1]
    beta_hops = [(o, v) for o, v in product(beta_occ, beta_vac)]
    res += [reference ^ (1 << i) ^ (1 << j) for i in beta_occ for j in beta_vac]
    
    return res, alpha_hops + beta_hops


def _gen_double_site_hops(
    reference: int, 
    n_qubits: int
) -> Tuple[List[int], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Given a reference single slater determinant as a bitstring, generate all
    possible next-nearest nearest neighbours and the hops required to get there.
    """
    
    #TODO: not tested for UHF where n_alpha and n_beta are not the same
    assert n_qubits % 2 == 0

    n_spatial = n_qubits // 2
    res = []

    alpha_occ = [i for i in range(n_spatial) if (reference >> i) & 1]
    alpha_vac = [i for i in range(n_spatial) if not (reference >> i) & 1]
    double_alpha = [(o, v) for o, v in product(combinations(alpha_occ, 2), combinations(alpha_vac, 2))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in double_alpha]


    beta_occ = [i for i in range(n_spatial, n_qubits) if (reference >> i) & 1]
    beta_vac = [i for i in range(n_spatial, n_qubits) if not (reference >> i) & 1]
    double_beta = [(o, v) for o, v in product(combinations(beta_occ, 2), combinations(beta_vac, 2))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in double_beta]


    alpha_beta = [(o, v) for o, v in product(product(alpha_occ, beta_occ), product(alpha_vac, beta_vac))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in alpha_beta]

    
    return res, double_alpha + double_beta + alpha_beta    


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


class AbstractStochasticSampler(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def sample(self) -> tuple[int, float]:
        pass


class WavefunctionSampler(AbstractStochasticSampler):

    def __init__(self, estimator: AbstractEstimator):
        self.estimator = estimator
        self.n_qubits = estimator.n_qubits
        self.states = list(range(2**self.n_qubits))

        amplitudes = []
        for state in self.states:
            bitstring = Bitstring.from_int(state, self.n_qubits)
            overlap = self.estimator.estimate_overlap(bitstring)
            amplitudes.append(overlap)

        amplitudes = np.array(amplitudes)
        self.probabilities = np.abs(amplitudes)**2
        self.probabilities /= self.probabilities.sum()


    def sample(self) -> tuple[int, float]:
        return np.random.choice(self.states, p=self.probabilities), 1.0
    

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
            m, t = self._propose_candidate(n)
            if self._accept(self.n, m, t):
                self.n = m

        return self.n, 1.0
    

    def _propose_candidate(self, n: int) -> Tuple[int, Any]:
        candidates, transitions = self.transition_fn(n, self.n_qubits)
        i = random.choice(range(len(candidates)))
        return candidates[i], transitions[i]
    

    def _accept(self, n: int, m: int, t: Any) -> bool:
        # TODO: In general we also need to account fot the transition amplitudes T(m->n)/T(n->m)
        c_n = self.estimator.estimate_overlap(Bitstring.from_int(n, self.n_qubits))
        c_m = self.estimator.estimate_overlap(Bitstring.from_int(m, self.n_qubits))
        P = min(1.0, (c_m * np.conj(c_m)) / (c_n * np.conj(c_n)))
        return random.random() <= P


class DMRGSampler(AbstractStochasticSampler):

    def __init__(self, mf: Union[RHF, UHF]):

        if isinstance(mf, UHF):
            raise NotImplementedError()
        
        norb = mf.mo_coeff.shape[1]
        nelec = mf.mol.nelectron

        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.kernel(mol, mf.mo_coeff)
        e_core = mol.energy_nuc()

        self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ)
        self.driver.initialize_system(n_sites=norb, n_elec=nelec, spin=0)
        self.mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=e_core, iprint=0)
        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
        self.driver.dmrg(
            self.mpo, self.ket,
            bond_dims=[50, 100, 200],
            noises=[1e-4, 1e-5, 0],
            thrds=[1e-8],
            n_sweeps=20,
            iprint=0
        )

    def sample(self) -> tuple[int, float]:
        
        cfgs, coeffs = self.driver.sample_csf_coefficients(
            self.ket, 
            n_sample=1,
            iprint=0
        )

        norb = cfgs.shape[1]

        cfg = cfgs[0]
        alpha = cfg & 1
        beta = (cfg >> 1) & 1


        res = sum(1 << i for i in range(norb) if alpha[i] == 1)
        res += sum (1 << i for i in range(norb, 2*norb) if beta[i-norb] == 1)

        return res, np.abs(coeffs[0]) ** 2

type IndependentEstimators = tuple[AbstractEstimator, AbstractEstimator]

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
            sampler = WavefunctionSampler(self.estimator)
        self.sampler = sampler


    def _compute_2rdm_estimator(self, n: int, bias_prob: Optional[float] = None):

        n_qubits = self.n_qubits
        gamma = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        estimator_n, estimator_m = self.estimators
        if estimator_m is None:
            estimator_m = estimator_n  # fall back to single estimator (biased)

        c_n = estimator_n.estimate_overlap(Bitstring.from_int(n, n_qubits))

        if bias_prob:
            w = (np.abs(c_n) ** 2) / bias_prob
        else:
            w = 1

        # double hops
        hops, transitions = _gen_double_site_hops(n, n_qubits)
        for m, t in zip(hops, transitions):
            (i, j), (k, l) = t
            c_m = estimator_m.estimate_overlap(Bitstring.from_int(m, n_qubits))
            gamma[i, j, k, l] = _compute_fermionic_sign(m, t) * (c_m/c_n) * w

        # single hops
        hops, transitions = _gen_single_site_hops(n, n_qubits)
        for m, t in zip(hops, transitions):
            i, k = t
            occupied = [j for j in range(n_qubits) if (m >> j) & 1 and j != k]
            for j in occupied:
                t = ((i, j), (k, j))
                c_m = estimator_m.estimate_overlap(Bitstring.from_int(m, n_qubits))
                gamma[i, j, k, j] = _compute_fermionic_sign(m, t) * (c_m/c_n) * w


        # get density-density terms
        occupied = [i for i in range(n_qubits) if (n >> i) & 1]
        for i, j in combinations(occupied, 2):
            gamma[i, j, i, j] = -1.0 * w

        # antisymmetrise
        return gamma - gamma.transpose(1,0,2,3) - gamma.transpose(0,1,3,2) + gamma.transpose(1,0,3,2)


    def estimate_2rdm(self, *, max_iters: int = 10000) -> np.ndarray:

        n_qubits = self.n_qubits
        gamma = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        for i in range(max_iters):

            n, p = self.sampler.sample()

            estimate = self._compute_2rdm_estimator(n, p)
            gamma += (estimate - gamma) / (i + 1)

        # return _spinorb_to_spatial_2rdm(gamma, norb=n_qubits // 2)
        # TODO convert this gamma into spatial orbital basis for RHF
        return gamma
    

def _spatial_to_spinorb_rdm2(rdm2_spatial: np.ndarray, norb: int) -> np.ndarray:
    norb_spin = 2 * norb
    rdm2_spin = np.zeros((norb_spin, norb_spin, norb_spin, norb_spin))

    # Alpha-alpha block: indices [0:norb, 0:norb, 0:norb, 0:norb]
    rdm2_spin[:norb, :norb, :norb, :norb] = rdm2_spatial

    # Beta-beta block: indices [norb:, norb:, norb:, norb:]
    rdm2_spin[norb:, norb:, norb:, norb:] = rdm2_spatial

    # Alpha-beta and beta-alpha blocks
    # For different spins, only direct term (no exchange)
    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    # Alpha(i), Beta(j), Alpha(k), Beta(l)
                    rdm2_spin[i, j + norb, k, l + norb] = rdm2_spatial[i, j, k, l]
                    # Beta(i), Alpha(j), Beta(k), Alpha(l)
                    rdm2_spin[i + norb, j, k + norb, l] = rdm2_spatial[i, j, k, l]

    return rdm2_spin

def _spinorb_to_spatial_2rdm(gamma_so, norb):
    gamma_spatial = np.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    gamma_spatial[p,q,r,s] += gamma_so[p, q, r, s]
                    gamma_spatial[p,q,r,s] += gamma_so[p+norb, q+norb, r+norb, s+norb]
                    gamma_spatial[p,q,r,s] += gamma_so[p, q+norb, r, s+norb]
                    gamma_spatial[p,q,r,s] += gamma_so[p+norb, q, r+norb, s]
    
    return gamma_spatial


if __name__ == "__main__":

    from pyscf import gto, scf
    from pyscf.fci import direct_spin1
    from shades.solvers import FCISolver
    from shades.estimators import ExactEstimator
    from shades.utils import make_hydrogen_chain

    N_HYDROGEN = 4
    d = 1.5

    mol_string = make_hydrogen_chain(N_HYDROGEN, d)
    mol = gto.Mole()
    mol.build(atom=mol_string, basis="sto-3g", verbose=0)

    mf = scf.RHF(mol)
    mf.run()

    model = DMRGSampler(mf)
    model.sample()

    fci_solver = FCISolver(mf)
    estimator = ExactEstimator(mf, solver=fci_solver)

    norb = mf.mo_coeff.shape[1]
    nelec = mf.mol.nelec
    (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb) = direct_spin1.make_rdm12s(
        fci_solver.civec, norb, nelec
    )

    mc = MonteCarloEstimator(estimator, sampler=model)
    mc_rdm2 = mc.estimate_2rdm(max_iters=100000)

    rdm2 = fci_solver.get_rdm2()

    print("PySCF rdm2aa[0,1,0,1]:", rdm2aa[0,1,0,1])
    print("PySCF rdm2aa[0,0,1,1]:", rdm2aa[0,0,1,1])
    print("PySCF rdm2ab[0,0,0,0]:", rdm2ab[0,0,0,0])

    mc_aa = mc_rdm2[:norb, :norb, :norb, :norb]
    mc_ab = mc_rdm2[:norb, norb:, :norb, norb:]

    print("\nYour mc_aa[0,1,0,1]:", mc_aa[0,1,0,1])
    print("Your mc_aa[0,0,1,1]:", mc_aa[0,0,1,1])
    print("Your mc_ab[0,0,0,0]:", mc_ab[0,0,0,0])


    mc_aa_converted = mc_aa.transpose(0, 3, 1, 2)
    diff = np.abs(rdm2aa - mc_aa_converted)

    # Find the worst element
    print(np.max(diff))
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print("Max diff at index:", idx)
    print("PySCF value:", rdm2aa[idx])
    print("MC value:", mc_aa_converted[idx])

    for p, q, r, s in [(0,0,0,0), (0,1,1,0), (1,0,0,1), (1,1,0,0)]:
        print(f"[{p},{q},{r},{s}]: PySCF={rdm2aa[p,q,r,s]:.4f}, MC={mc_aa_converted[p,q,r,s]:.4f}")

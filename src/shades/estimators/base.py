from abc import ABC, abstractmethod
from shades.solvers import GroundStateSolver
from pyscf import scf
import logging
import time
from typing import Union
import numpy as np

from shades.excitations import (
    doubles_to_t2,
    get_doubles,
    get_hf_reference,
    get_singles,
    singles_to_t1,
)
from shades.utils import Bitstring, compute_correlation_energy

logger = logging.getLogger(__name__)

class AbstractEstimator(ABC):

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF], solver: GroundStateSolver, verbose: int = 0):
        self.solver = solver
        self.trial, self.E_exact = solver.solve()
        self.E_hf = mf.e_tot
        self.mf = mf
        self.verbose = verbose
        self.n_qubits = 2 * self.mf.mol.nao

    def update_reference(self, new_mf: Union[scf.hf.RHF, scf.uhf.UHF]):
        self.mf = new_mf
        self.solver.mf = new_mf  # Update solver's mf reference for Brueckner iterations
        self.E_hf = new_mf.e_tot
        self.trial, self.E_exact = self.solver.solve()

    def run(self, *, calc_c1: bool = False):

        logger.info("Estimating HF reference overlap (c0)...")
        t_start = time.perf_counter()

        c0 = self.estimate_c0()

        t_elapsed = time.perf_counter() - t_start
        logger.info(f"c0 = {c0:.6f} ({t_elapsed:.3f}s)")

        if calc_c1:
            n_singles = len(get_singles(self.mf))
            logger.info(f"Estimating {n_singles} single excitation amplitudes (c1)...")
            t_start = time.perf_counter()

            c1 = self.estimate_c1()

            t_elapsed = time.perf_counter() - t_start
            avg_time = t_elapsed / n_singles if n_singles > 0 else 0
            logger.info(
                f"Estimated {n_singles} singles in {t_elapsed:.2f}s ({avg_time * 1000:.1f} ms/exc)"
            )
        else:
            c1 = None

        n_doubles = len(get_doubles(self.mf))
        logger.info(f"Estimating {n_doubles} double excitation amplitudes (c2)...")
        t_start = time.perf_counter()

        c2 = self.estimate_c2()

        t_elapsed = time.perf_counter() - t_start
        avg_time = t_elapsed / n_doubles if n_doubles > 0 else 0
        logger.info(
            f"Estimated {n_doubles} doubles in {t_elapsed:.2f}s ({avg_time * 1000:.1f} ms/exc)"
        )

        logger.debug("Computing correlation energy...")
        e_corr = compute_correlation_energy(self.mf, c0, c1, c2)
        e_total = self.E_hf + e_corr

        logger.info(f"HF Energy: {self.E_hf:.8f} Ha")
        logger.info(f"Correlation Energy: {e_corr:.8f} Ha")
        logger.info(f"Total Energy: {e_total:.8f} Ha")
        if hasattr(self, "E_exact"):
            error = e_total - self.E_exact
            logger.info(f"Exact FCI Energy: {self.E_exact:.8f} Ha")
            logger.info(f"Error: {error:+.2e} Ha")

        return e_total, c0, c1, c2

    @abstractmethod
    def estimate_overlap(self, a: Bitstring) -> np.float64:
        pass

    def estimate_c0(self)-> np.float64:
        psi0 = get_hf_reference(self.mf)
        overlap = self.estimate_overlap(psi0)
        return np.abs(overlap)

    def estimate_c1(self):

        def f(bitstring):
            nonlocal c
            c += 1
            if c % max(1, n_exc // 10) == 0:
                progress = c / n_exc * 100
                logger.debug(f"Singles progress: {c}/{n_exc} ({progress:.0f}%)")
            return self.estimate_overlap(bitstring)

        norb = self.mf.mol.nao

        if isinstance(self.mf, scf.hf.RHF):
            nocc, _ = self.mf.mol.nelec
            nvirt = norb - nocc

            singles = get_singles(self.mf)

            n_exc = len(singles)
            c = 0

            t1 = singles_to_t1(singles, f, nocc, nvirt)
            return t1
        
        elif isinstance(self.mf, scf.uhf.UHF):
            nocc_a, nocc_b = self.mf.mol.nelec
            nvirt_a = norb - nocc_a
            nvirt_b = norb - nocc_b

            singles = get_singles(self.mf)
            n_exc = len(singles)
            c = 0

            s_a = [ex for ex in singles if ex.spin == 'alpha']
            s_b = [ex for ex in singles if ex.spin == 'beta']

            t1_a = singles_to_t1(s_a, f, nocc_a, nvirt_a)
            t1_b = singles_to_t1(s_b, f, nocc_b, nvirt_b)

            return t1_a, t1_b
        
        else:
            raise RuntimeError()

    def estimate_c2(self):

        def f(bitstring):
            nonlocal c
            c += 1
            if c % max(1, n_exc // 10) == 0:
                progress = c / n_exc * 100
                logger.debug(f"Doubles progress: {c}/{n_exc} ({progress:.0f}%)")
            return self.estimate_overlap(bitstring)

        norb = self.mf.mol.nao

        if isinstance(self.mf, scf.hf.RHF):
            nocc, _ = self.mf.mol.nelec
            nvirt = norb - nocc

            doubles = get_doubles(self.mf)

            n_exc = len(doubles)
            c = 0
            t2 = doubles_to_t2(
                doubles, f, nocc, nvirt, spin_case="alpha-beta", symmetry_restricted=True
            )

            return t2

        elif isinstance(self.mf, scf.uhf.UHF):
            nocc_a, nocc_b = self.mf.mol.nelec
            nvirt_a = norb - nocc_a
            nvirt_b = norb - nocc_b

            doubles = get_doubles(self.mf, symmetry_restricted=True)
            n_exc = len(doubles)
            c = 0

            s_aa = [ex for ex in doubles if ex.spin_case == 'alpha-alpha']
            s_bb = [ex for ex in doubles if ex.spin_case == 'beta-beta']
            s_ab = [ex for ex in doubles if ex.spin_case == 'alpha-beta']

            t2_aa = doubles_to_t2(s_aa, f, nocc_a, nvirt_a, spin_case='alpha-alpha', symmetry_restricted=True)
            t2_bb = doubles_to_t2(s_bb, f, nocc_b, nvirt_b, spin_case='beta-beta', symmetry_restricted=True)
            t2_ab = doubles_to_t2(s_ab, f, (nocc_a, nocc_b), (nvirt_a, nvirt_b), spin_case='alpha-beta')

            return t2_aa, t2_bb, t2_ab
        
        else:
            raise RuntimeError()
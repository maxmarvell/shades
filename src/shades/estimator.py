import logging
import time
from typing import Union

import numpy as np
from pyscf import scf

from shades.excitations import (
    doubles_to_t2,
    get_doubles,
    get_hf_reference,
    get_singles,
    singles_to_t1,
)
from shades.shadows import ShadowProtocol
from shades.solvers import GroundStateSolver
from shades.utils import compute_correlation_energy

logger = logging.getLogger(__name__)


class GroundStateEstimator:
    """Ground state energy estimator using shadow tomography.

    Uses the 'mixed' energy estimator to approximate corrections to the ground state
    wavefunction of a HF mean-field Hamiltonian via classical shadow tomography.

    Args:
        hamiltonian: Molecular Hamiltonian object
        solver: Ground state solver (e.g., FCISolver, VQESolver)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)
    """

    def __init__(
        self, mf: Union[scf.hf.RHF, scf.uhf.UHF], solver: GroundStateSolver, verbose: int = 0
    ):
        self.trial, self.E_exact = solver.solve()
        self.E_hf = mf.e_tot
        self.mf = mf
        norb = mf.mo_coeff.shape[0]
        self.n_qubits = 2 * norb
        self.verbose = verbose

        # Configure logging based on verbosity
        if verbose >= 2:
            logging.basicConfig(
                level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
            )
        elif verbose >= 1:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

        logger.debug(f"Initialized: {self.n_qubits} qubits ({norb} orbitals)")
        logger.debug(f"HF Energy: {self.E_hf:.8f} Ha")
        logger.debug(f"FCI Energy: {self.E_exact:.8f} Ha")
        logger.debug(f"Correlation Energy: {self.E_exact - self.E_hf:.8f} Ha")

    def estimate_ground_state(
        self,
        n_samples: int,
        n_k_estimators: int,
        *,
        n_jobs: int = 1,
        use_qualcs: bool = True,
        calc_c1: bool = False,
    ):
        """Estimate ground state energy and excitation amplitudes via shadow tomography.

        Args:
            n_samples: Total number of shadow measurements
            n_k_estimators: Number of median-of-means estimators
            n_jobs: Number of parallel workers (default: 1)
            use_qualcs: Use Qulacs backend for faster sampling (default: True)

        Returns:
            Tuple of (energy, c0_overlap, singles_amplitudes, doubles_amplitudes)
        """

        logger.debug(f"Shadow samples: {n_samples:,}, bins: {n_k_estimators}, workers: {n_jobs}")
        logger.debug(f"Backend: {'Qulacs' if use_qualcs else 'Qiskit'}")

        # Phase 1: Collect shadow samples
        logger.info("Collecting shadow samples...")
        t_start = time.perf_counter()

        protocol = ShadowProtocol(
            self.trial, n_jobs=n_jobs, use_qulacs=use_qualcs, verbose=self.verbose - 1
        )
        protocol.collect_samples(n_samples, n_k_estimators, prediction="overlap")

        t_elapsed = time.perf_counter() - t_start
        throughput = n_samples / t_elapsed
        logger.info(
            f"Collected {n_samples:,} samples in {t_elapsed:.2f}s ({throughput:.0f} samples/s)"
        )

        # Phase 2: Estimate HF overlap (c0)
        logger.info("Estimating HF reference overlap (c0)...")
        t_start = time.perf_counter()

        c0 = self.estimate_c0(protocol)

        t_elapsed = time.perf_counter() - t_start
        logger.info(f"c0 = {c0:.6f} ({t_elapsed:.3f}s)")

        # Phase 3: Estimate single excitations (c1)
        if calc_c1:
            n_singles = len(get_singles(self.mf))
            logger.info(f"Estimating {n_singles} single excitation amplitudes (c1)...")
            t_start = time.perf_counter()

            c1 = self.estimate_t1(protocol)

            t_elapsed = time.perf_counter() - t_start
            avg_time = t_elapsed / n_singles if n_singles > 0 else 0
            logger.info(
                f"Estimated {n_singles} singles in {t_elapsed:.2f}s ({avg_time * 1000:.1f} ms/exc)"
            )
        else:
            c1 = None

        # Phase 4: Estimate double excitations (c2)
        n_doubles = len(get_doubles(self.mf))
        logger.info(f"Estimating {n_doubles} double excitation amplitudes (c2)...")
        t_start = time.perf_counter()

        c2 = self.estimate_t2(protocol)

        t_elapsed = time.perf_counter() - t_start
        avg_time = t_elapsed / n_doubles if n_doubles > 0 else 0
        logger.info(
            f"Estimated {n_doubles} doubles in {t_elapsed:.2f}s ({avg_time * 1000:.1f} ms/exc)"
        )

        # Compute correlation energy
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

    def estimate_c0(self, protocol: ShadowProtocol) -> np.float64:
        psi0 = get_hf_reference(self.mf)
        overlap = protocol.estimate_overlap(psi0)
        return overlap.real

    def estimate_t1(self, protocol: ShadowProtocol) -> np.ndarray:
        """Estimate single excitation amplitudes and return in tensor form.

        For RHF systems, only unique excitations are measured (alpha only),
        reducing the number of shadow measurements by a factor of 2.

        Returns:
            SingleAmplitudes: Amplitudes in t1[i,a] format (nocc, nvirt)
        """

        def f(bitstring):
            nonlocal c
            c += 1
            if (c + 1) % max(1, n_exc // 10) == 0:
                progress = (c + 1) / n_exc * 100
                logger.debug(f"Singles progress: {c + 1}/{n_exc} ({progress:.0f}%)")
            return protocol.estimate_overlap(bitstring).real

        nocc, _ = self.mf.mol.nelec
        norb = self.mf.mol.nao
        nvirt = norb - nocc

        excitations = get_singles(self.mf)

        n_exc = len(excitations)
        c = 0

        t1 = singles_to_t1(excitations, f, nocc, nvirt)
        return t1

    def estimate_t2(self, protocol: ShadowProtocol) -> np.ndarray:
        """Estimate double excitation amplitudes and return in tensor form.

        Returns:
            DoubleAmplitudes: Amplitudes in t2[i,j,a,b] format (nocc, nocc, nvirt, nvirt)
        """

        def f(bitstring):
            nonlocal c
            c += 1
            if (c + 1) % max(1, n_exc // 10) == 0:
                progress = (c + 1) / n_exc * 100
                logger.debug(f"Doubles progress: {c + 1}/{n_exc} ({progress:.0f}%)")
            return protocol.estimate_overlap(bitstring).real

        if isinstance(self.mf, scf.hf.RHF):
            nocc, _ = self.mf.mol.nelec
            norb = self.mf.mol.nao
            nvirt = norb - nocc

            doubles = get_doubles(self.mf)

            n_exc = len(doubles)
            c = 0
            t2 = doubles_to_t2(
                doubles, f, nocc, nvirt, spin_case="alpha-beta", symmetry_restricted=True
            )

        elif isinstance(self.mf, scf.uhf.UHF):
            raise NotImplementedError()

        return t2


if __name__ == "__main__":
    pass

import numpy as np
import logging
import time
from typing import Union

from pyscf import scf

from shades.solvers import GroundStateSolver
from shades.tomography.matchgate import (
    MatchgateShadow,
    estimate_one_rdm,
    estimate_two_rdm,
)
from shades.utils import spinorb_to_spatial_2rdm, total_energy_from_rdms

logger = logging.getLogger(__name__)


class MatchgateEstimator:
    """Estimates RDMs and energy using matchgate shadow tomography.

    Unlike the overlap-based estimators (ShadowEstimator, ExactEstimator), this
    estimator constructs the full 1-RDM and 2-RDM from matchgate shadow snapshots
    and contracts them with molecular integrals to obtain the total energy.
    """

    def __init__(
        self,
        mf: Union[scf.hf.RHF, scf.uhf.UHF],
        solver: GroundStateSolver,
        *,
        verbose: int = 0,
    ):
        self.mf = mf
        self.solver = solver
        self.trial, self.E_exact = solver.solve()
        self.E_hf = mf.e_tot
        self.n_qubits = 2 * mf.mol.nao
        self.verbose = verbose
        self._shadow = None

    def _estimate_one_rdm(self, shadow: MatchgateShadow) -> np.ndarray:
        rdm1_spinorb = estimate_one_rdm(shadow)
        norb = shadow.n_qubits // 2
        rdm1_aa = rdm1_spinorb[:norb, :norb]
        rdm1_bb = rdm1_spinorb[norb:, norb:]
        return (rdm1_aa + rdm1_bb).real

    def run(
        self,
        *,
        n_samples: int,
        symmetry: str = "RHF",
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Collect matchgate shadows, estimate RDMs, and compute the energy.

        Args:
            n_samples: Number of matchgate shadow snapshots to collect.
            symmetry: Symmetry to exploit ("RHF" or "none").

        Returns:
            Tuple of (total_energy, spatial_1rdm, spatial_2rdm, spinorb_2rdm).
        """
        logger.info(f"Collecting {n_samples} matchgate shadow samples...")
        t_start = time.perf_counter()

        self._shadow = MatchgateShadow(self.trial)
        self._shadow.run(n_samples)

        t_elapsed = time.perf_counter() - t_start
        throughput = n_samples / t_elapsed
        logger.info(f"Collected {n_samples:,} samples in {t_elapsed:.2f}s ({throughput:.0f} samples/s)")

        logger.info("Estimating 1-RDM from matchgate shadows...")
        t_start = time.perf_counter()
        rdm1_spatial = self._estimate_one_rdm(self._shadow)
        t_elapsed = time.perf_counter() - t_start
        logger.info(f"Estimated 1-RDM in {t_elapsed:.2f}s")

        logger.info("Estimating 2-RDM from matchgate shadows...")
        t_start = time.perf_counter()
        rdm2_spinorb = estimate_two_rdm(self._shadow, symmetry=symmetry)
        t_elapsed = time.perf_counter() - t_start
        logger.info(f"Estimated 2-RDM in {t_elapsed:.2f}s")

        norb = self.mf.mol.nao
        rdm2_spatial = spinorb_to_spatial_2rdm(rdm2_spinorb.real, norb)

        e_total = total_energy_from_rdms(rdm1_spatial, rdm2_spatial, self.mf)

        logger.info(f"HF Energy: {self.E_hf:.8f} Ha")
        logger.info(f"Total Energy: {e_total:.8f} Ha")
        logger.info(f"Exact FCI Energy: {self.E_exact:.8f} Ha")
        logger.info(f"Error: {e_total - self.E_exact:+.2e} Ha")

        return e_total, rdm1_spatial, rdm2_spatial, rdm2_spinorb

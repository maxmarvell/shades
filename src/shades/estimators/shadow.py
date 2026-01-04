from shades.estimators import AbstractEstimator
from shades.solvers import GroundStateSolver
from shades.utils import Bitstring
from shades.shadows import ShadowProtocol
from pyscf import scf
from typing import Union, Optional
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class ShadowEstimator(AbstractEstimator):
    protocol: Optional[ShadowProtocol]
    n_workers: int = 1

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF], solver: GroundStateSolver, *, verbose: int = 0):
        super().__init__(mf, solver, verbose)
        self.protocol = None

    def update_reference(self, new_mf: Union[scf.hf.RHF, scf.uhf.UHF]):
        """Update the mean-field reference and clear cached shadow samples."""
        super().update_reference(new_mf)
        self.clear_sample()

    def run(
        self,
        *,
        n_samples: int,
        n_k_estimators: int,
        n_jobs: int = 1,
        calc_c1=False,
    ):
        logger.debug(f"Shadow samples: {n_samples:,}, bins: {n_k_estimators}")

        self.n_workers = n_jobs

        if self.protocol is None:

            logger.info("Collecting shadow samples...")
            t_start = time.perf_counter()

            self.sample(n_samples, n_k_estimators)

            t_elapsed = time.perf_counter() - t_start
            throughput = n_samples / t_elapsed
            logger.info(
                f"Collected {n_samples:,} samples in {t_elapsed:.2f}s ({throughput:.0f} samples/s)"
            )

        return super().run(calc_c1=calc_c1)
    
    def clear_sample(self) -> None:
        self.protocol = None

    def sample(self, n_samples: int, n_k_estimators: int) -> None:
        self.protocol = ShadowProtocol(self.trial)
        self.protocol.collect_samples_for_overlaps(n_samples, n_k_estimators)

    def estimate_overlap(self, a: Bitstring) -> np.float64:
        if not self.protocol:
            raise RuntimeError()
        return self.protocol.estimate_overlap(a, n_jobs=self.n_workers).real

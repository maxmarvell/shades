from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
import numpy as np
from qiskit.quantum_info import Statevector
import qulacs
import stim
import time
import logging
from multiprocessing import Pool

from shades.utils import bitstring_to_stabilizers, measurement_to_int, gaussian_elimination, gaussian_elimination_fast, compute_x_rank, canonicalize
from shades.tomography.clifford import (
    CliffordChannel,
    CliffordSnapshot,
)
from shades.tomography.matchgate import MatchgateSnapshot

logger = logging.getLogger(__name__)


def _reconstruct_stabilizer_tableau(snapshot: CliffordSnapshot) -> stim.Tableau:
    """Reconstruct the stabilizer tableau U†|b><b|U from a CliffordSnapshot."""
    U_inv = snapshot.tableau.inverse()
    stabilizers = bitstring_to_stabilizers(snapshot.bitstring, snapshot.n_qubits)
    transformed = [U_inv(s) for s in stabilizers]
    canonical = canonicalize(transformed)
    return stim.Tableau.from_stabilizers(canonical)


@dataclass
class _PrecomputedSnapshot:
    """Per-snapshot data that is independent of the target bitstring."""
    stabilizers: List[stim.PauliString]
    mag_squared: float
    ref_state: int
    phase_0: complex
    n_qubits: int
    # Precomputed numpy arrays for fast gaussian elimination
    stab_x_bits: Optional[np.ndarray] = None      # (n_stab, n_qubits) bool
    stab_z_bits: Optional[np.ndarray] = None      # (n_stab, n_qubits) bool
    stab_signs: Optional[np.ndarray] = None       # (n_stab,) complex
    stab_pauli_types: Optional[np.ndarray] = None  # (n_stab, n_qubits) uint8


@dataclass
class ClassicalShadow:
    """A classical shadow: a collection of snapshots collected by applying
    random unitaries from a CliffordChannel, measuring, and storing the
    channel-specific snapshot representation.
    """

    snapshots: List[Union[CliffordSnapshot, MatchgateSnapshot]]
    channel: CliffordChannel
    n_qubits: int

    _precomputed: Optional[List[_PrecomputedSnapshot]] = None

    @classmethod
    def from_state(
        cls,
        state: Statevector,
        n_samples: int,
        channel: Optional[CliffordChannel] = None,
    ):
        n = state.num_qubits
        if channel is None:
            channel = CliffordChannel(n)

        qulacs_template = qulacs.QuantumState(n)
        qulacs_template.load(state.data)

        snapshots = []
        for _ in range(n_samples):
            qulacs_state = qulacs_template.copy()
            snapshot = channel(qulacs_state)
            snapshots.append(snapshot)

        return cls(snapshots=snapshots, channel=channel, n_qubits=n)

    @property
    def N(self) -> int:
        return len(self.snapshots)

    def _ensure_precomputed(self):
        """Precompute per-snapshot data that is reused across all overlap queries."""
        if self._precomputed is not None:
            return
        vacuum = 0
        targets = list(range(self.n_qubits))
        self._precomputed = []
        for snapshot in self.snapshots:
            stab_tableau = _reconstruct_stabilizer_tableau(snapshot)
            stabilizers = stab_tableau.to_stabilizers()
            x_rank = compute_x_rank(stabilizers)
            mag_squared = 2 ** (-x_rank)

            sim = stim.TableauSimulator()
            sim.do_tableau(stab_tableau, targets=targets)
            measurement = sim.measure_many(*range(self.n_qubits))
            ref_state = measurement_to_int(measurement)

            n_stab = len(stabilizers)
            nq = self.n_qubits
            stab_x_bits = np.empty((n_stab, nq), dtype=bool)
            stab_z_bits = np.empty((n_stab, nq), dtype=bool)
            stab_signs = np.empty(n_stab, dtype=complex)
            stab_pauli_types = np.empty((n_stab, nq), dtype=np.uint8)

            for idx, s in enumerate(stabilizers):
                x_arr, z_arr = s.to_numpy()
                stab_x_bits[idx] = x_arr
                stab_z_bits[idx] = z_arr
                stab_signs[idx] = s.sign
                for q in range(nq):
                    x, z = x_arr[q], z_arr[q]
                    if not x and not z:
                        stab_pauli_types[idx, q] = 0  # I
                    elif x and not z:
                        stab_pauli_types[idx, q] = 1  # X
                    elif x and z:
                        stab_pauli_types[idx, q] = 2  # Y
                    else:
                        stab_pauli_types[idx, q] = 3  # Z

            phase_0 = gaussian_elimination_fast(
                stab_x_bits, stab_pauli_types, stab_signs,
                ref_state, vacuum, nq,
            )

            self._precomputed.append(_PrecomputedSnapshot(
                stabilizers=stabilizers,
                mag_squared=mag_squared,
                ref_state=ref_state,
                phase_0=phase_0,
                n_qubits=self.n_qubits,
                stab_x_bits=stab_x_bits,
                stab_z_bits=stab_z_bits,
                stab_signs=stab_signs,
                stab_pauli_types=stab_pauli_types,
            ))

    def estimate_observable(self, observable: stim.PauliString) -> complex:
        """Estimate the expectation value of a Pauli observable.

        Averages per-snapshot estimates, then applies channel inversion.
        """
        raw = np.mean([s.estimate(observable) for s in self.snapshots])
        return self.channel.invert(raw)

    def overlap(self, a: int):
        """Estimate overlap using the Clifford stabilizer formalism.

        This method is specific to Clifford shadows and uses Gaussian
        elimination on the reconstructed stabilizer tableau.
        """
        self._ensure_precomputed()
        overlaps = np.empty(len(self._precomputed))
        for i, pre in enumerate(self._precomputed):
            phase_a = gaussian_elimination_fast(
                pre.stab_x_bits, pre.stab_pauli_types, pre.stab_signs,
                pre.ref_state, a, pre.n_qubits,
            )
            overlaps[i] = (pre.mag_squared * phase_a * pre.phase_0.conjugate()).real

        return 2 * (2**self.n_qubits + 1) * np.mean(overlaps)

    @staticmethod
    def sample_state(
        state: Statevector,
        channel: Optional[CliffordChannel] = None,
    ) -> stim.Tableau:
        """Sample a single snapshot and return the reconstructed stabilizer tableau.

        Kept for backward compatibility with stabilizer_subspace.py.
        """
        n = state.num_qubits
        if channel is None:
            channel = CliffordChannel(n)
        qulacs_state = qulacs.QuantumState(n)
        qulacs_state.load(state.data)
        snapshot = channel(qulacs_state)
        return _reconstruct_stabilizer_tableau(snapshot)


# Backward compatibility alias
CliffordShadow = ClassicalShadow


@dataclass
class ComputationalShadow:

    snapshots: List[int]
    n_qubits: int

    @classmethod
    def from_state(cls, state: Statevector, n_samples: int):
        n = state.num_qubits

        qulacs_state = qulacs.QuantumState(n)
        qulacs_state.load(state.data)

        snapshots = []
        for _ in range(n_samples):
            sample = qulacs_state.sampling(1)[0]
            snapshots.append(sample)

        return cls(snapshots=snapshots, n_qubits=n)

    @staticmethod
    def sample_state(state: Statevector) -> int:
        n = state.num_qubits
        qulacs_state = qulacs.QuantumState(n)
        qulacs_state.load(state.data)
        return qulacs_state.sampling(1)[0]


def _compute_single_estimator_overlap(args):
    """Compute overlap for a single K-estimator."""
    estimator, target = args
    return estimator.overlap(target)



class PredictionTask(Enum):
    OVERLAP = "overlap"
    OBSERVABLE = "observable"


def _prepare_tau(state: Statevector):
    if not np.allclose(state.data[0], 0):
        raise RuntimeError("tau() assumes ⟨0|ψ⟩ = 0 so that |τ⟩ is a valid equal superposition.")
    vacuum = np.zeros(2**state.num_qubits, dtype=complex); vacuum[0] = 1.0
    tau = (vacuum + state.data) / np.sqrt(2)
    return Statevector(tau)


@dataclass
class ShadowProtocol:
    state: Statevector
    channel: Optional[CliffordChannel] = None

    _k_estimators: Optional[List[ClassicalShadow]] = None
    _task: Optional[PredictionTask] = None
    _pool: Optional[Pool] = None
    _pool_size: int = 0

    def __post_init__(self):
        if self.channel is None:
            self.channel = CliffordChannel(self.state.num_qubits)

    def __del__(self):
        self._close_pool()

    def _close_pool(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            self._pool_size = 0

    def _get_pool(self, n_jobs: int) -> Pool:
        if self._pool is None or self._pool_size != n_jobs:
            self._close_pool()
            self._pool = Pool(processes=n_jobs)
            self._pool_size = n_jobs
        return self._pool

    def collect_samples_for_overlaps(self, n_samples: int, n_estimators: int):
        """Collect shadows by sampling from tau for overlap estimation."""
        tau = _prepare_tau(self.state)
        self._k_estimators = self._collect(tau, n_samples, n_estimators)
        self._task = PredictionTask.OVERLAP

    def collect_samples_for_observables(self, n_samples: int, n_estimators: int):
        """Collect shadows by sampling from state for observable estimation."""
        self._k_estimators = self._collect(self.state, n_samples, n_estimators)
        self._task = PredictionTask.OBSERVABLE

    def _collect(self, state: Statevector, n_samples: int, n_estimators: int):

        if n_samples <= 0 or n_estimators <= 0:
            raise ValueError("n_samples and n_estimators must be positive.")
        if n_samples % n_estimators != 0:
            raise ValueError("The shadow must be split into K equally sized parts.")

        samples_per_estimator = n_samples // n_estimators
        logger.debug(
            f"Collecting {n_samples:,} shadow samples across {n_estimators} estimators "
            f"({samples_per_estimator:,} samples/estimator)"
        )

        k_estimators = []
        for i in range(n_estimators):
            logger.debug(f"Collecting estimator {i+1}/{n_estimators}...")
            t_start = time.perf_counter()

            shadow = ClassicalShadow.from_state(state, samples_per_estimator, self.channel)
            k_estimators.append(shadow)

            t_elapsed = time.perf_counter() - t_start
            throughput = samples_per_estimator / t_elapsed
            logger.debug(
                f"Estimator {i+1}/{n_estimators} complete: "
                f"{samples_per_estimator:,} samples in {t_elapsed:.2f}s ({throughput:.0f} samples/s)"
            )

        logger.debug(f"Shadow collection complete: {n_estimators} estimators ready")
        return k_estimators

    def estimate_overlap(self, a: int, *, n_jobs: int = 1):

        if self._k_estimators is None:
            raise ValueError("Must call collect_samples before estimating anything")

        if self._task is not PredictionTask.OVERLAP:
            raise RuntimeError("Incorrect shados sampled!")

        t_start = time.perf_counter()

        if n_jobs > 1 and len(self._k_estimators) > 1:
            args_list = [(estimator, a) for estimator in self._k_estimators]
            pool = self._get_pool(n_jobs)
            means = pool.map(_compute_single_estimator_overlap, args_list)
        else:
            means = []
            for estimator in self._k_estimators:
                overlap_val = estimator.overlap(a)
                means.append(overlap_val)

        result = np.median(means)

        t_elapsed = time.perf_counter() - t_start
        mode = "parallel" if n_jobs > 1 and len(self._k_estimators) > 1 else "serial"
        logger.debug(
            f"Computed overlap in {t_elapsed*1000:.2f} ms "
            f"({len(self._k_estimators)} estimators, {mode})"
        )

        return result

    def estimate_stabilizer_overlap(self, S: stim.Tableau):
        pass

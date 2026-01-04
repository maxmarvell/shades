from dataclasses import dataclass
from typing import Union, List, Optional, Any
from enum import Enum
import numpy as np
from qiskit.quantum_info import Statevector, Clifford
import qulacs
import time

from abc import ABC, abstractmethod
import stim
from multiprocessing import Pool

from shades.utils import Bitstring, gaussian_elimination, compute_x_rank, canonicalize

class AbstractEnsemble(ABC):

    def __init__(self, d: int):
        self.d = d

    @abstractmethod
    def generate_sample(self) -> stim.Tableau | np.ndarray:
        """Gives a sample according to the ensemble"""

class CliffordGroup(AbstractEnsemble):
    def generate_sample(self) -> stim.Tableau:
        return stim.Tableau.random(self.d)

@dataclass
class AbstractShadow(ABC):
    snapshots: List
    n_qubits: int

    @property
    def N(self) -> int:
        """Total number of shadow measurements."""
        return len(self.snapshots)

    @classmethod
    @abstractmethod
    def from_state(cls, state: Statevector, n_samples: int):
        pass

    @abstractmethod
    def overlap(self, a: Bitstring):
        pass

    @staticmethod
    @abstractmethod
    def sample_state(
        state: Statevector
    ) -> tuple[Bitstring, Any]:
        pass



@dataclass
class CliffordShadow:

    snapshots: Union[List[stim.Tableau]]
    n_qubits: int

    @classmethod
    def from_state(cls, state: Statevector, n_samples: int):
        snapshots = []
        for _ in range(n_samples):
            snapshot = cls.sample_state(state)
            snapshots.append(snapshot)
        
        return cls(snapshots=snapshots, n_qubits=state.num_qubits)
    
    @property
    def N(self) -> int:
        """Total number of shadow measurements."""
        return len(self.snapshots)

    def overlap(self, a: Bitstring):
        overlaps = []
        vacuum = Bitstring([False] * self.n_qubits, endianess='little')

        for snapshot in self.snapshots:
            
            stabilizers = snapshot.to_stabilizers()
            x_rank = compute_x_rank(stabilizers)
            mag = 2 ** (-x_rank / 2)

            sim = stim.TableauSimulator()
            sim.do_tableau(snapshot, targets=list(range(self.n_qubits)))
            measurement = sim.measure_many(*range(self.n_qubits))
            bitstring = Bitstring(measurement, endianess='little')

            phase_a = gaussian_elimination(stabilizers, bitstring, a)
            phase_0 = gaussian_elimination(stabilizers, bitstring, vacuum)

            overlaps.append(mag**2 * phase_a * phase_0.conjugate())

        return 2 * (2**self.n_qubits + 1) * np.mean(overlaps)
    
    @staticmethod
    def sample_state(
        state: Statevector
    ) -> stim.Tableau:
        
        n = state.num_qubits
        tab = stim.Tableau.random(n)

        clifford = _tableau_to_qiskit_clifford(tab)
        qulacs_state = qulacs.QuantumState(n)
        qulacs_state.load(state)
        circuit = _clifford_to_qulacs_circuit(clifford, n)
        circuit.update_quantum_state(qulacs_state)
        sample = qulacs_state.sampling(1)[0]
        b = Bitstring.from_int(sample, size=n, endianess='little')

        U_inv = tab.inverse()
        stabilizers = b.to_stabilizers()
        transformed_stabilizers = [U_inv(s) for s in stabilizers]
        canonical_stabilizers = canonicalize(transformed_stabilizers)

        return stim.Tableau.from_stabilizers(canonical_stabilizers)

@dataclass
class MatchgateShadow:
    pass


@dataclass
class ComputationalShadow:

    snapshots: List[Bitstring]
    n_qubits: int

    @classmethod
    def from_state(cls, state: Statevector, n_samples: int):
        snapshots = []
        for _ in range(n_samples):
            b = cls.sample_state(state)
            snapshots.append(b)

        return cls(snapshots=snapshots, n_qubits=state.num_qubits)

    @staticmethod
    def sample_state(state: Statevector) -> Bitstring:
        n = state.num_qubits
        qulacs_state = qulacs.QuantumState(n)
        qulacs_state.load(state)
        sample = qulacs_state.sampling(1)[0]
        b = Bitstring.from_int(sample, size=n, endianess='little')
        return b

def _compute_single_estimator_overlap(args):
    """Compute overlap for a single K-estimator (for parallelization).

    Args:
        args: Tuple of (estimator, target_bitstring)

    Returns:
        Complex overlap value
    """
    estimator, target = args
    return estimator.overlap(target)

def _tableau_to_qiskit_clifford(tab: stim.Tableau) -> Clifford:
    """Convert a stim.Tableau to a Qiskit Clifford via symplectic generators."""
    n = len(tab)
    destabs =  [str(x).replace("_","I") for x in [tab.x_output(k) for k in range(n)]]
    stabs = [str(x).replace("_","I") for x in tab.to_stabilizers()]
    return Clifford(destabs + stabs)

def _clifford_to_qulacs_circuit(cliff: Clifford, n_qubits: int) -> qulacs.QuantumCircuit:
    """Convert Qiskit Clifford to Qulacs circuit."""

    import textwrap
    from qiskit import qasm2
    from qulacs.converter import convert_QASM_to_qulacs_circuit

    circuit = qulacs.QuantumCircuit(n_qubits)

    # from c.lenihan
    qasm = qasm2.dumps(cliff.to_circuit())
    qasm = textwrap.dedent(qasm).strip()
    circuit = convert_QASM_to_qulacs_circuit(qasm.splitlines())

    return circuit

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
    verbose: int

    _k_estimators: Optional[List[CliffordShadow]] = None
    _task: Optional[PredictionTask] = None


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

        k_estimators = []
        for _ in range(n_estimators):
            shadow = CliffordShadow.from_state(state, n_samples)
            k_estimators.append(shadow)

        return k_estimators
    
    def estimate_overlap(self, a: Bitstring, *, n_jobs: int = 1):

        if self._k_estimators is None:
            raise ValueError("Must call collect_samples before estimating anything")
        
        if self._task is not PredictionTask.OVERLAP:
            raise RuntimeError("Incorrect shados sampled!")

        if self.verbose >= 3:
            t_start = time.perf_counter()

        if n_jobs > 1 and len(self._k_estimators) > 1:
            args_list = [(estimator, a) for estimator in self._k_estimators]
            with Pool(processes=n_jobs) as pool:
                means = pool.map(_compute_single_estimator_overlap, args_list)
        else:
            means = []
            for estimator in self._k_estimators:
                overlap_val = estimator.overlap(a)
                means.append(overlap_val)

        result = np.median(means)

        if self.verbose >= 3:
            t_elapsed = time.perf_counter() - t_start
            mode = "parallel" if self.n_jobs > 1 and len(self._k_estimators) > 1 else "serial"
            print(f"    [Overlap Estimation] Computed overlap in {t_elapsed*1000:.2f} ms ({len(self._k_estimators)} estimators, {mode})")

        return result




if __name__ == "__main__":
    pass
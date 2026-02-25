from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import qulacs
import stim
from qiskit.quantum_info import Statevector

from shades.utils import bitstring_to_stabilizers, measurement_to_int, canonicalize, gaussian_elimination, compute_x_rank, tableau_to_qulacs_circuit


@dataclass
class CliffordSnapshot:
    """A single Clifford shadow snapshot storing the sampled unitary and
    measurement outcome."""
    tableau: stim.Tableau
    bitstring: int
    n_qubits: int

    def estimate(self, observable: stim.PauliString) -> complex:
        """Compute the raw (pre-inversion) single-snapshot estimator for an observable.

        Returns tr(O * U†|b><b|U) = <b|U O U†|b>.
        If the rotated observable P = U O U† is not diagonal (has X or Y),
        the expectation is 0.  Otherwise it is sign(P) * prod(-1)^{b_q}
        for each qubit q where P has a Z.
        """
        rotated = self.tableau(observable)

        result = rotated.sign
        for q in range(self.n_qubits):
            pauli = rotated[q]  # 0=I, 1=X, 2=Y, 3=Z
            if pauli == 1 or pauli == 2:
                return 0.0
            if pauli == 3:
                if (self.bitstring >> q) & 1:
                    result *= -1
        return complex(result)

    def overlap(self, a: int) -> complex:

        overlaps = []
        vacuum = 0

        stab_tableau = self._reconstruct_stabilizer_tableau()

        stabilizers = stab_tableau.to_stabilizers()
        x_rank = compute_x_rank(stabilizers)
        mag = 2 ** (-x_rank / 2)

        sim = stim.TableauSimulator()
        sim.do_tableau(stab_tableau, targets=list(range(self.n_qubits)))
        measurement = sim.measure_many(*range(self.n_qubits))
        ref_state = measurement_to_int(measurement)

        phase_a = gaussian_elimination(stabilizers, ref_state, a, self.n_qubits)
        phase_0 = gaussian_elimination(stabilizers, ref_state, vacuum, self.n_qubits)

        overlaps.append(mag**2 * phase_a * phase_0.conjugate())

        return 2 * (2**self.n_qubits + 1) * np.mean(overlaps)

    def _reconstruct_stabilizer_tableau(self) -> stim.Tableau:
        """Reconstruct the stabilizer tableau U†|b><b|U from a CliffordSnapshot."""
        U_inv = self.tableau.inverse()
        stabilizers = bitstring_to_stabilizers(self.bitstring, self.n_qubits)
        transformed = [U_inv(s) for s in stabilizers]
        canonical = canonicalize(transformed)
        return stim.Tableau.from_stabilizers(canonical)

@dataclass
class CliffordChannel():
    """Uniform random Clifford channel (the standard shadow tomography ensemble)."""

    n_qubits: int

    def sample(self) -> stim.Tableau:
        return stim.Tableau.random(self.n_qubits)

    def __call__(self, state: qulacs.QuantumState) -> CliffordSnapshot:
        tab = self.sample()
        circuit = tableau_to_qulacs_circuit(tab, self.n_qubits)
        circuit.update_quantum_state(state)
        outcome = state.sampling(1)[0]
        return CliffordSnapshot(tableau=tab, bitstring=outcome, n_qubits=self.n_qubits)

    def invert(self, raw_estimate: complex) -> complex:
        return (2**self.n_qubits + 1) * raw_estimate
    

class CliffordShadow():

    n_qubits: int
    snapshots: list[CliffordSnapshot] | None

    def __init__(self, state: Statevector):
        self.state = state
        self.n_qubits = state.num_qubits
        self.channel = CliffordChannel(self.n_qubits)
        self.snapshots = None

    def run(self, n_samples: int):
        self.snapshots = []
        for _ in range(n_samples):
            snapshot = self.channel(self.state)
            self.snapshots.append(snapshot)
        return self.snapshots
    
    def estimate_overlap(self, a: int, n_estimators: int) -> complex:

        assert len(self.snapshots) % n_estimators != 0, ""
        assert self.snapshots is not None

        chunk_size = len(self.snapshots) // n_estimators
        estimators = [self.snapshots[i*chunk_size:(i+1)*chunk_size] for i in range(n_estimators)]

        means = []
        for estimator in estimators:
            mean = np.mean([p() for p in estimator])

        return np.mean([p.estimate_majorana(S) for p in self.snapshots])

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import Optional
import numpy as np
import networkx as nx

class WeylDistribution:
    pass

class BellDifferenceSampler:

    def __init__(self, state: Statevector, m_clique: Optional[int] = None):
        self.state = state

        if m_clique:
            self.m_clique = m_clique
        else:
            raise NotImplementedError("TODO: Use formula from theorem 4.9 to choose reasonable m_cliques.")
        
    def bell_basis_measurement(self) -> np.ndarray:

        n_qubits = self.state.num_qubits

        copy1 = QuantumRegister(n_qubits, 'copy1')
        copy2 = QuantumRegister(n_qubits, 'copy2')

        classical = ClassicalRegister(2 * n_qubits, 'meas')
        
        qc = QuantumCircuit(copy1, copy2, classical)
        qc.prepare_state(self.state, copy1[:])
        qc.prepare_state(self.state, copy2[:])
        
        for i in range(n_qubits):
            qc.cx(copy1[i], copy2[i])
            qc.h(copy1[i])
            qc.measure(copy1[i], classical[i])
            qc.measure(copy2[i], classical[n_qubits + i])

        simulator = AerSimulator()
        result = simulator.run(qc, shots=1).result()
        bitstring = list(result.get_counts().keys())[0]

        bits = np.array([int(b) for b in bitstring[::-1]], dtype=np.uint8)

        a = bits[:n_qubits]
        b = bits[n_qubits:]
        return np.concatenate([a, b])
        
    def run(self):

        x = self.bell_basis_measurement()
        y = self.bell_basis_measurement()

        return (x + y) % 2


def symplectic_product(x: np.ndarray, y: np.ndarray) -> np.uint8:

    # TODO: check size of input arrays and type etx

    n = x.size // 2
    a, b = x[:n], x[n:]
    c, d = y[:n], y[n:]
    s = np.bitwise_xor.reduce(np.bitwise_and(a, d))
    s ^= np.bitwise_xor.reduce(np.bitwise_and(b, c))
    return np.uint8(s & 1)


def stabilizer_state_approzimation(state: Statevector):

    G = nx.Graph()

    n_qubits = state.num_qubits
    m_clique = 1000

    sampler = BellDifferenceSampler(state, m_clique=1000)

    for _ in m_clique:
        sample = sampler.run()
        G.add_node(sample)
        for node in G.nodes:
            if node is not sample:
                if symplectic_product(sample, node) == 1:
                    pass

            

    # weyl_samples = np.empty((m_clique, 2*n_qubits), dtype=np.uint8)




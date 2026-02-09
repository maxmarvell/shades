from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import Optional
import numpy as np
import networkx as nx

from shades.tomography import ShadowProtocol
from shades.utils import canonicalize
import stim

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
        """Perform Bell basis measurements on two copies of the state.

        Returns:
            Array of integers representing 2n-bit measurement outcomes.
            Bit layout: bits 0..n-1 are from copy1 (a), bits n..2n-1 are from copy2 (b).
        """
        n_qubits = self.state.num_qubits

        copy1 = QuantumRegister(n_qubits, 'copy1')
        copy2 = QuantumRegister(n_qubits, 'copy2')
        classical = ClassicalRegister(2 * n_qubits, 'meas')

        qc = QuantumCircuit(copy1, copy2, classical)
        qc.initialize(self.state.data, copy1[:])
        qc.initialize(self.state.data, copy2[:])

        for i in range(n_qubits):
            qc.cx(copy1[i], copy2[i])
            qc.h(copy1[i])
            qc.measure(copy1[i], classical[i])
            qc.measure(copy2[i], classical[n_qubits + i])

        simulator = AerSimulator()
        result = simulator.run(qc, shots=self.m_clique, memory=True).result()
        bitstrings = result.get_memory()

        return np.array([int(bs[::-1], 2) for bs in bitstrings], dtype=np.uint64)
        
    def run(self) -> np.ndarray:
        """Sample Weyl operators by taking pairwise differences of Bell measurements.

        Returns:
            Array of unique non-zero integers representing 2n-bit Weyl operators.
            The identity (0) is omitted as it commutes with everything.
        """
        x = self.bell_basis_measurement()
        y = self.bell_basis_measurement()
        weyl_operators = np.bitwise_xor(x, y)
        unique_ops = np.unique(weyl_operators)
        return unique_ops[unique_ops != 0]


def symplectic_product(x: int, y: int, n_qubits: int) -> int:
    """Compute symplectic product of two Weyl operators.

    The symplectic product ω(x,y) for x=(a,b) and y=(c,d) is computed as:
    ω(x,y) = (a·d ⊕ b·c) mod 2

    where · is bitwise AND and ⊕ is XOR (parity) of all bits.

    Args:
        x: First Weyl operator as 2n-bit integer
        y: Second Weyl operator as 2n-bit integer
        n_qubits: Number of qubits (n)

    Returns:
        Symplectic product (0 or 1)
    """

    mask = (1 << n_qubits) - 1
    a = x & mask
    b = x >> n_qubits
    c = y & mask
    d = y >> n_qubits
    result = (a & d) ^ (b & c)
    return bin(result).count('1') & 1

def flip_sign(p: stim.PauliString) -> stim.PauliString:
    return stim.PauliString(str(p).replace("+", "-", 1)) if str(p).startswith("+") else stim.PauliString(str(p).replace("-", "+", 1))

def stabilizer_states_from_generators(gens):
    n = len(gens)
    for mask in range(1 << n):
        signed = []
        for i, g in enumerate(gens):
            if (mask >> i) & 1:
                signed.append(flip_sign(g))
            else:
                signed.append(g)
        # This defines the unique joint +/− eigenstate (if consistent)
        tab = stim.Tableau.from_stabilizers(signed)
        yield mask, tab


def stabilizer_state_approximation(state: Statevector, m_clique: int = 1000, m_shadow: int = 1000):

    G = nx.Graph()

    n_qubits = state.num_qubits

    sampler = BellDifferenceSampler(state, m_clique=m_clique)
    weyl_samples = sampler.run()

    for x in weyl_samples:
        G.add_node(x)
        for y in G.nodes:
            if y != x:
                if symplectic_product(x, y, n_qubits) == 0:
                    G.add_edge(x, y)

    
    shadow = ShadowProtocol(state)
    shadow.collect_samples_for_observables(m_shadow, n_estimators=20)

    # for C in nx.find_cliques(G):
    #     C = list(C)        # safe even if already a list
    #     _save_graph(G, C)

    ovlps = []
    stabilizers = []

    for C in nx.find_cliques(G):

        if len(C) < n_qubits: continue

        stabs = _build_generator_from_clique(C, n_qubits)
        gens = list(filter(lambda s: s.weight != 0, stabs))
        dim = len(gens)

        # check Lagrangian
        if dim < n_qubits: 
            continue
        elif dim > n_qubits:
            raise RuntimeError("Overcomplete generators for stabilizer state")

        for m, S in stabilizer_states_from_generators(gens):

            stabilizers.append(S)

            phi = S.to_state_vector()
            ovlp = np.abs(np.vdot(phi, state.data))**2  
            ovlps.append(ovlp)   

        print()

    return ovlps, stabilizers
    print(ovlps, stabilizers)

def _build_generator_from_clique(C, n_qubits):

    M = []
    for c in C:
        M.append(_weyl_to_stim_paulistring(c, n_qubits))

    return canonicalize(M)

def _weyl_to_stim_paulistring(x: int, n: int):
    """
    x_bits: iterable of length 2n of 0/1 (or a numpy array / python int bits unpacked)
            convention: x = (a | b) where a = x_bits[:n], b = x_bits[n:].
    n: number of qubits
    """
    chars = []
    for j in range(n):
        a_j = (x >> j) & 1
        b_j = (x >> (j + n)) & 1

        if a_j == 0 and b_j == 0:
            chars.append('I')
        elif a_j == 1 and b_j == 0:
            chars.append('X')
        elif a_j == 0 and b_j == 1:
            chars.append('Z')
        else:  # a_j == 1 and b_j == 1
            chars.append('Y')

    return stim.PauliString(''.join(chars))

def _save_graph(G, C):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))

    pos = nx.spring_layout(G, seed=0)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=500,
        font_size=8
    )

    # draw clique nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=C,
        node_color="tab:red",
        node_size=800
    )

    # draw clique edges
    clique_edges = [(u, v) for u in C for v in C if u != v]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=clique_edges,
        edge_color="tab:red",
        width=2.5
    )

    plt.tight_layout()
    plt.savefig("graph_highlight_clique.png", dpi=300)
    plt.close()


if __name__ == "__main__":

    n_qubits = 4
    psi = Statevector.from_label('0110')
    stabilizer_state_approximation(psi, m_clique=30)
    
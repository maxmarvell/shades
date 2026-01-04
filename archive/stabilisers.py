import stim
from typing import List
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from shades.utils import Bitstring

G = np.array([0, 0, 0, 0,
                   0, -1, 1, 0,
                   0, -1, 1, 0,
                   0, 1, -1, 0],
                  dtype=np.int8)

class StabilizerState:
    """Stabilizer methods"""

    x: NDArray[np.bool_]
    z: NDArray[np.bool_]
    s: NDArray[np.bool_]


    def __init__(self, 
                 x: NDArray[np.bool_],
                 z: NDArray[np.bool_],
                 s: NDArray[np.bool_],
                 *, 
                 canonicalise: bool = True):
        
        self.n = len(x[0])
        if canonicalise:
            self.x, self.z, self.s = self.canonicalise(x, z, s)
            self.canonical = True
        else:
            self.x, self.z, self.s = x, z, s
            self.canonical = False

    @staticmethod
    def canonicalise(x_matrix: NDArray[np.bool_], z_matrix: NDArray[np.bool_], signs: NDArray[np.bool_]):

        x = x_matrix.copy()
        z = z_matrix.copy()
        s = signs.copy()

        k, n = x.shape

        def rowswap(i: int, j: int):
            x[i], x[j] = x[j], x[i]
            z[i], z[j] = z[j], z[i]
            s[i], s[j] = s[j], s[i]

        def rowmult(i: int, j: int):

            def g(x1, z1, x2, z2):
                x1_int = x1.astype(np.int8)
                z1_int = z1.astype(np.int8)
                x2_int = x2.astype(np.int8)
                z2_int = z2.astype(np.int8)

                return (
                    (x1_int & z1_int) * (z2_int - x2_int) +
                    (x1_int & ~z1_int) * (2*x2_int*z2_int - z2_int) +
                    (~x1_int & z1_int) * (x2_int - 2*x2_int*z2_int)
                )

            g_sum = np.sum(g(x[i], z[i], x[j], z[j]))

            total = 2 * s[j] + 2 * s[i] + g_sum
            s[j] = ((total % 4) >= 2)

            x[j] ^= x[i]
            z[j] ^= z[i]

        # Gaussian elimination: X-block
        pivot = 0
        for col in range(n):

            found = None
            for row in range(pivot, k):
                if x[row, col]:
                    found = row
                    break
            
            if found is not None:
                rowswap(pivot, found)
                for row in range(k):
                    if row != pivot and x[row, col]:
                        rowmult(pivot, row)
                pivot += 1
        
        # Gaussian elimination: Z-block
        for col in range(n):

            found = None
            for row in range(pivot, k):
                if z[row, col] and not x[row, col]:
                    found = row
                    break
            
            if found is not None:
                rowswap(pivot, found)
                for row in range(k):
                    if row != pivot and z[row, col]:
                        rowmult(pivot, row)
                pivot += 1
        
        return x, z, s

    @classmethod
    def from_stim_tableau(cls, tableau: stim.Tableau, *, canonicalise: bool = False):

        n = len(tableau)

        x_matrix = np.zeros((n, n), dtype=np.bool_)
        z_matrix = np.zeros((n, n), dtype=np.bool_)
        signs = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            pauli_string = tableau.z_output(i)

            # Extract sign: +1 -> False, -1 -> True
            if pauli_string.sign == -1:
                signs[i] = True
            elif pauli_string.sign != 1:
                raise ValueError(f"Stabilizer {i} has complex sign {pauli_string.sign}. Only real signs (±1) are supported.")

            # Extract X and Z components for each qubit
            for j in range(n):
                pauli = pauli_string[j]  
                if pauli == 1: 
                    x_matrix[i, j] = True
                elif pauli == 2:  
                    x_matrix[i, j] = True
                    z_matrix[i, j] = True
                elif pauli == 3: 
                    z_matrix[i, j] = True

        return cls(x_matrix, z_matrix, signs, canonicalise=canonicalise)
    
    @property
    def statevector(self) -> NDArray[np.complex128]:
        tab = stim.Tableau.from_stabilizers(self.generators)
        return tab.to_state_vector()
    
    @staticmethod
    def conjguate(generators: List[stim.PauliString], gate: str, targets: List[int]):
        """
        Get generator matrix after conjugating by a named Clifford gate operation.
        """
        n = len(generators[0])
        tableau = stim.Tableau(n)
        tableau.append(stim.Tableau.from_named_gate(gate), targets)
        return [tableau(ps) for ps in generators]
    
    def find_basis_state(self) -> List[bool]:
        sim = stim.TableauSimulator()
        tab = stim.Tableau.from_stabilizers(tab)
        # sim.do_tableau(tab, targets=)


if __name__ == "__main__":

    tab = stim.Tableau.random(5)#


    from shades.stabilizer import StabilizerState as Old


    old_stabilizer = Old.from_stim_tableau(tab)
    stabilizer = StabilizerState.from_stim_tableau(tab, canonicalise=True)

    print(stabilizer)

    
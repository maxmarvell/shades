import stim
from typing import List

class Stabilizer:
    """Stabilizer methods"""

    def __init__(self, 
                 generators: List[stim.PauliString], 
                 *, 
                 canonicalise: bool = True):
        
        if not generators:
            raise ValueError("Need at least one generator")
        
        self.n = len(generators[0])
        if canonicalise:
            self.generators = self.canonicalise(generators)
        else:
            self.generators = generators


    @staticmethod
    def canonicalise(stabilizers: List[stim.PauliString]):
        """Convert stim stabilizers to canonical form using Gaussian elimination.

        Takes a list of stim.PauliString stabilizers and reduces them to
        canonical row echelon form following the algorithm from
        https://arxiv.org/pdf/1711.07848

        Args:
            stabilizers: List of stim.PauliString objects representing stabilizers

        Returns:
            Canonical stabilizer matrix where each row is [pauli_ops..., phase]
            Example: [['X', 'I', 'I', 1], ['I', 'Z', 'I', -1]]

        Example:
            >>> # For Bell state |Φ⁺⟩ with stabilizers XX and ZZ
            >>> stabs = [stim.PauliString("XX"), stim.PauliString("ZZ")]
            >>> canonical = canonicalize(stabs)
            >>> # Returns: [['X', 'X', 1], ['Z', 'Z', 1]]
        """

        def rowswap(i: int, j: int):
            canonicalised[i], canonicalised[j] = canonicalised[j], canonicalised[i]

        def rowmult(i: int, j: int):
            canonicalised[j] = canonicalised[i] * canonicalised[j]

        canonicalised = [s.copy() for s in stabilizers]
        nq = len(canonicalised[0])
        nr = len(canonicalised)

        # X-block
        i = 0
        for j in range(nq):
            k = next((k for k in range(i, nr) if canonicalised[k][j] in {1, 2}), None)
            if k is not None:
                rowswap(i, k)
                for m in range(nr):
                    if m != i and canonicalised[m][j] in {1, 2}:
                        rowmult(i, m)
                i += 1

        # Z-block
        for j in range(nq):
            k = next((k for k in range(i, nr) if canonicalised[k][j] == 3), None)
            if k is not None:
                rowswap(i, k)
                for m in range(nr):
                    if m != i and canonicalised[m][j] in {2, 3}:
                        rowmult(i, m)
                i += 1


        return canonicalised


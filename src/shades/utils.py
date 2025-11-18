from dataclasses import dataclass
from typing import List
import stim

import numpy as np
from numpy.typing import NDArray
from typing import Union, Optional
from pyscf import scf
from pyscf import ao2mo

@dataclass
class Bitstring():

    array: NDArray[np.bool_]
    endianess: str = 'little'

    def __post_init__(self):
        """Validate endianness parameter and ensure numpy array."""
        if self.endianess not in ('big', 'little'):
            raise ValueError(f"endianess must be 'big' or 'little', got '{self.endianess}'")
        if not isinstance(self.array, np.ndarray):
            self.array = np.array(self.array, dtype=np.bool_)

    @property
    def size(self) -> int:
        return len(self.array)

    def __getitem__(self, key):
        """Allow indexing as array[i]"""
        result = self.array[key]
        if isinstance(result, np.bool_):
            return bool(result)
        return result

    def __setitem__(self, key, value):
        """Allow item assignment as array[i] = value"""
        self.array[key] = value

    def __eq__(self, other):
        """Check equality with another Bitstring."""
        if not isinstance(other, Bitstring):
            return False
        return np.array_equal(self.array, other.array) and self.endianess == other.endianess

    def __iter__(self):
        """Allow iteration over bits."""
        return iter(self.array)

    def __len__(self):
        """Allow len() to work."""
        return len(self.array)

    def to_string(self) -> str:
        """Convert bitstring to string representation."""
        if self.endianess == 'big':
            return ''.join('1' if bit else '0' for bit in self.array)
        else:
            return ''.join('1' if bit else '0' for bit in self.array[::-1])

    def to_stabilizers(self) -> List[stim.PauliString]:
        """Convert to list of stabilizer generators."""
        out = []
        n = self.size
        for i, bit in enumerate(self.array):
            label = ['I'] * n
            label[n - 1 - i] = 'Z'             
            p = stim.PauliString(''.join(label))
            if bit:
                p *= -1                  
            out.append(p)
        return out

    def to_int(self) -> int:
        """Convert bitstring to integer.

        For both endianness: to_string() already handles reversal, so we just convert.
        """
        if self.endianess == 'little':
            value = sum((1 << i) if bit else 0 for i, bit in enumerate(self.array))
        else:
            n = len(self.array)
            value = sum((1 << (n - 1 - i)) if bit else 0 for i, bit in enumerate(self.array))
        return value

    def to_array(self) -> NDArray[np.int_]:
        """Convert to numpy array of integers (0s and 1s)."""
        return self.array.astype(np.int_)

    @classmethod
    def from_string(cls, s: str, endianess: str = 'little') -> 'Bitstring':
        """Create a Bitstring from a string like '0101'.

        Args:
            s: Binary string (e.g., '0101')
            endianess: 'little' (default) or 'big'
        """
        if endianess == "little":
            bits = np.array([b == '1' for b in s[::-1]], dtype=np.bool_)
        else:
            bits = np.array([b == '1' for b in s], dtype=np.bool_)
        return cls(bits, endianess=endianess)

    @classmethod
    def from_int(cls, value: int, size: int, endianess: str = 'little') -> 'Bitstring':
        """Create a Bitstring from an integer with specified size.

        Args:
            value: Non-negative integer to convert
            size: Number of bits in the output
            endianess: 'big' (default) or 'little'

        For big-endian: MSB is leftmost bit (standard binary)
        For little-endian: array is stored reversed
        """
        bits = np.array([(value >> i) & 1 for i in range(size)], dtype=np.bool_)
        if endianess == "big":
            bits = bits[::-1]
        return cls(bits, endianess=endianess)

    @classmethod
    def random(cls, size: int, rng=None, endianess: str = 'little') -> 'Bitstring':
        """Create a random Bitstring of given size.

        Args:
            size: Number of bits
            rng: Optional numpy random generator (for reproducibility)
            endianess: 'big' (default) or 'little'

        Returns:
            Random Bitstring
        """
        if rng is None:
            rng = np.random.default_rng()
        bits = rng.choice([True, False], size=size).astype(np.bool_)
        return cls(bits, endianess=endianess)

    def copy(self) -> 'Bitstring':
        """Return a deep copy of the Bitstring."""
        return Bitstring(self.array.copy(), endianess=self.endianess)
    
    def convert_endian(self) -> 'Bitstring':
        """Convert representation so integer value is preserved across endianness flip."""
        target = 'little' if self.endianess == 'big' else 'big'
        return Bitstring(self.array[::-1], endianess=target)
    
@dataclass
class SingleExcitation:
    """Represents a single excitation with its indices and bitstring."""
    occ: int  # occupied orbital index
    virt: int  # virtual orbital index
    spin: str  # 'alpha' or 'beta'
    bitstring: Bitstring
    n: int # number of occ alpha or beta orbitals

    def __repr__(self) -> str:
        spin_symbol = 'α' if self.spin == 'alpha' else 'β'
        return f"{self.occ}{spin_symbol} → {self.virt+self.n}{spin_symbol}"

def gaussian_elimination(
    stabilizers: List[stim.PauliString],
        ref_state: Bitstring,
        target_state: Bitstring
    ) -> complex:
    """Compute the phase of overlap between a stabilizer and a target basis state.

    If no overlap simply returns 0 + 0j

    Args:
        stabilizers: List of canonical stabilizers (reduced stabilizer matrix)
        ref_state: Reference computational basis state (measurement outcome)
        target_state: Target computational basis state for overlap computation

    Returns:
        Complex phase factor for the overlap
    """
    phase = 1 + 0j
    interm_state = ref_state.copy()
    target_str = target_state.copy()
    n_qubits = target_str.size

    for n in range(n_qubits):
        if target_str[n] != interm_state[n]:
            for m in range(len(stabilizers)):
                stabilizer = stabilizers[m]
                x_bits, _ = stabilizer.to_numpy()

                if not x_bits[n]: continue

                all_left_false = True
                for k in range(n):
                    if x_bits[k]:
                        all_left_false = False
                        break

                if all_left_false:
                    interm_state, phase = apply_stabilizer_to_state(interm_state, stabilizer, phase)
                    break

        # no overlap
        if target_str[n] != interm_state[n]: return 0j

    return phase

def apply_stabilizer_to_state(
        state: Bitstring,
        stabilizer: stim.PauliString,
        phase: complex
    ) -> tuple[Bitstring, complex]:
        """Apply a Pauli string (stabilizer) to a computational basis state.

        Args:
            state: Computational basis state as Bitstring object
            stabilizer: Pauli string to apply
            phase: Accumulated phase factor

        Returns:
            Tuple of (new_state, new_phase)
        """
        n_qubits = state.size
        post_state = state

        # Apply the stabilizer's sign/phase
        new_phase = phase * stabilizer.sign

        # Apply each Pauli operator
        for n in range(n_qubits):
            pauli = stabilizer[n]
            bit = post_state[n]

            if pauli == 1: # X
                post_state[n] = not bit
            elif pauli == 2: # Y
                post_state[n] = not bit
                new_phase = new_phase * (1j) * (-1) ** (int(bit))
            elif pauli == 3: # Z
                new_phase = new_phase * (-1) ** (int(bit))
            elif pauli != 0: # I
                raise ValueError(f"Unrecognized pauli operator {pauli}.")

        return post_state, new_phase

def compute_x_rank(canonical_stabilizers: List[stim.PauliString]) -> int:
    """Compute the X-rank given a set of canonical stabilizers."""
    x_rank = 0
    for stab in canonical_stabilizers:
        x_bits, _ = stab.to_numpy()
        for bit in x_bits:
            if bit: x_rank += 1; break
    return x_rank

def canonicalize(stabilizers: List[stim.PauliString]) -> List[stim.PauliString]:
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
        canonicalized[i], canonicalized[j] = canonicalized[j], canonicalized[i]

    def rowmult(i: int, j: int):
        canonicalized[j] = canonicalized[i] * canonicalized[j] 

    canonicalized = [s.copy() for s in stabilizers]
    nq = len(canonicalized[0])      # number of qubits = length of PauliString
    nr = len(canonicalized)         # number of generators (rows)

    # X-block
    i = 0
    for j in range(nq):
        k = next((k for k in range(i, nr) if canonicalized[k][j] in {1, 2}), None)
        if k is not None:
            rowswap(i, k)
            for m in range(nr):
                if m != i and canonicalized[m][j] in {1, 2}:
                    rowmult(i, m)
            i += 1

    # Z-block
    for j in range(nq):
        k = next((k for k in range(i, nr) if canonicalized[k][j] == 3), None)
        if k is not None:
            rowswap(i, k)
            for m in range(nr):
                if m != i and canonicalized[m][j] in {2, 3}:
                    rowmult(i, m)
            i += 1


    return canonicalized

def make_hydrogen_chain(n_atoms: int, bond_length: float = 0.50) -> str:
    """Generate a linear hydrogen chain with fixed interatomic distance.

    Creates a string representation of N hydrogen atoms arranged linearly
    along the z-axis with equal spacing, suitable for PySCF molecule input.

    Args:
        n_atoms: Number of hydrogen atoms in the chain (must be >= 1)
        bond_length: Interatomic distance in Angstroms (default: 0.50)

    Returns:
        String in PySCF format: "H 0 0 0; H 0 0 d; H 0 0 2d; ..."

    Examples:
        >>> make_hydrogen_chain(2, 0.50)
        'H 0 0 0; H 0 0 0.50'

        >>> make_hydrogen_chain(4, 0.74)
        'H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22'
    """
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
    if bond_length <= 0:
        raise ValueError(f"bond_length must be positive, got {bond_length}")

    atoms = []
    for i in range(n_atoms):
        z_coord = i * bond_length
        atoms.append(f"H 0 0 {z_coord:.10g}")

    return "; ".join(atoms)

def compute_correlation_energy(
        mf: Union[scf.hf.RHF, scf.uhf.UHF],
        c0: float,
        c1: Optional[Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]],
        c2: Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]
    ) -> float:

    e_singles = 0
    norb = mf.mol.nao

    if isinstance(mf, scf.hf.RHF):

        nocc, _ = mf.mol.nelec

        if c1 is not None:
            fock_ao = mf.get_fock()
            fock_mo = mf.mo_coeff.T @ fock_ao @ mf.mo_coeff
            f_ov = fock_mo[:nocc, nocc:]
            e_singles = 2.0 * np.sum(f_ov * c1) / c0

        mo_coeff = mf.mo_coeff
        eri_ao = mf.mol.intor("int2e")
        eri_mo = ao2mo.full(eri_ao, mo_coeff)
        eri = eri_mo.reshape(norb, norb, norb, norb)
        g_ovvo = eri[:nocc, nocc:, nocc:, :nocc]
        e_doubles = (
            2.0 * np.einsum("ijab,iabj->", c2, g_ovvo)
            - np.einsum("ijab,ibaj->", c2, g_ovvo)
        ) / c0
    
    elif isinstance(mf, scf.uhf.UHF): 

        nocc_a, nocc_b = mf.mol.nelec
        mo_a, mo_b = mf.mo_coeff
        
        if c1 is not None:

            fock_a, fock_b = mf.get_fock()
            c1_a, c1_b = c1

            fock_mo = mo_a.T @ fock_a @ mo_a
            f_ov = fock_mo[:nocc_a, nocc_a:]
            e_singles += np.sum(f_ov * c1_a) / c0

            fock_mo = mo_b.T @ fock_b @ mo_b
            f_ov = fock_mo[:nocc_b, nocc_b:]
            e_singles += np.sum(f_ov * c1_b) / c0

        eri_ao = mf.mol.intor("int2e")

        eri_aaaa = ao2mo.general(
            eri_ao, 
            [mo_a, mo_a, mo_a, mo_a],
            compact=False
        )
        eri_aaaa = eri_aaaa.reshape(norb, norb, norb, norb)
        g_ovvo_aa = eri_aaaa[:nocc_a, nocc_a:, nocc_a:, :nocc_a]

        eri_bbbb = ao2mo.general(
            eri_ao,
            [mo_b, mo_b, mo_b, mo_b],
            compact=False
        )
        eri_bbbb = eri_bbbb.reshape(norb, norb, norb, norb)
        g_ovvo_bb = eri_bbbb[:nocc_b, nocc_b:, nocc_b:, :nocc_b]

        eri_aabb = ao2mo.general(
            eri_ao,
            [mo_a, mo_a, mo_b, mo_b],
            compact=False
        )
        eri_aabb = eri_aabb.reshape(norb, norb, norb, norb)
        g_ovvo_ab = eri_aabb[:nocc_a, nocc_a:, nocc_b:, :nocc_b]

        c2_aa, c2_bb, c2_ab = c2

        e_aa = (
            np.einsum("ijab,iabj->", c2_aa, g_ovvo_aa) 
            - np.einsum("ijab,ibaj->", c2_aa, g_ovvo_aa)
        ) / 4
        
        e_bb = (
            np.einsum("ijab,iabj->", c2_bb, g_ovvo_bb)
            - np.einsum("ijab,ibaj->", c2_bb, g_ovvo_bb)
        ) / 4

        e_ab = np.einsum("ijab,iabj->", c2_ab, g_ovvo_ab)
        
        e_doubles = (e_aa + e_bb + e_ab) / c0

    else:
        raise ValueError("Correlation energy formula only implemented for RHF and UHF")

    return e_singles + e_doubles

if __name__ == "__main__":
    
    from shades.utils import make_hydrogen_chain
    from pyscf import gto, scf

    atom = make_hydrogen_chain(4, bond_length=3.5)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g", verbose=0, spin=0)
    mf = scf.UHF(mol)
    mf.run()

    from shades.solvers import FCISolver
    fci = FCISolver(mf)

    from shades.estimators import TrivialEstimator

    estimator = TrivialEstimator(mf, fci)

    t2_aa, t2_bb, t2_ab = estimator.estimate_c2()

    
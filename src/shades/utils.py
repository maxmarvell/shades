from typing import List
import stim
import qulacs

import numpy as np
from numpy.typing import NDArray
from typing import Union, Optional
from pyscf import scf
from pyscf import ao2mo

def bitstring_to_stabilizers(value: int, n_qubits: int) -> List[stim.PauliString]:
    """Convert an integer bitstring to a list of stabilizer generators.

    Args:
        value: Integer representing the computational basis state (little-endian).
        n_qubits: Number of qubits.

    Returns:
        List of stim.PauliString stabilizers, one per qubit.
    """
    out = []
    for i in range(n_qubits):
        label = ['I'] * n_qubits
        label[i] = 'Z'
        p = stim.PauliString(''.join(label))
        if (value >> i) & 1:
            p *= -1
        out.append(p)
    return out


def measurement_to_int(measurement) -> int:
    """Convert a measurement result (list of bools) to an integer (little-endian)."""
    value = 0
    for i, bit in enumerate(measurement):
        if bit:
            value |= (1 << i)
    return value


def gaussian_elimination(
    stabilizers: List[stim.PauliString],
    ref_state: int,
    target_state: int,
    n_qubits: int,
) -> complex:
    """Compute the phase of overlap between a stabilizer and a target basis state.

    If no overlap simply returns 0 + 0j

    Args:
        stabilizers: List of canonical stabilizers (reduced stabilizer matrix)
        ref_state: Reference computational basis state as int (little-endian)
        target_state: Target computational basis state as int (little-endian)
        n_qubits: Number of qubits

    Returns:
        Complex phase factor for the overlap
    """
    phase = 1 + 0j
    interm = ref_state
    target = target_state

    for n in range(n_qubits):
        if ((target >> n) & 1) != ((interm >> n) & 1):
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
                    interm, phase = apply_stabilizer_to_state(interm, stabilizer, phase, n_qubits)
                    break

        # no overlap
        if ((target >> n) & 1) != ((interm >> n) & 1): return 0j

    return phase

def apply_stabilizer_to_state(
        state: int,
        stabilizer: stim.PauliString,
        phase: complex,
        n_qubits: int,
    ) -> tuple[int, complex]:
        """Apply a Pauli string (stabilizer) to a computational basis state.

        Args:
            state: Computational basis state as int (little-endian)
            stabilizer: Pauli string to apply
            phase: Accumulated phase factor
            n_qubits: Number of qubits

        Returns:
            Tuple of (new_state, new_phase)
        """
        post_state = state

        # Apply the stabilizer's sign/phase
        new_phase = phase * stabilizer.sign

        # Apply each Pauli operator
        for n in range(n_qubits):
            pauli = stabilizer[n]
            bit = (post_state >> n) & 1

            if pauli == 1: # X
                post_state ^= (1 << n)
            elif pauli == 2: # Y
                post_state ^= (1 << n)
                new_phase = new_phase * (1j) * (-1) ** bit
            elif pauli == 3: # Z
                new_phase = new_phase * (-1) ** bit
            elif pauli != 0: # I
                raise ValueError(f"Unrecognized pauli operator {pauli}.")

        return post_state, new_phase

def apply_stabilizer_to_state_fast(
    state: int,
    pauli_types: np.ndarray,
    sign: complex,
    phase: complex,
    n_qubits: int,
) -> tuple[int, complex]:
    """Apply a stabilizer (as numpy arrays) to a computational basis state.

    Args:
        state: Computational basis state as int (little-endian)
        pauli_types: Array of shape (n_qubits,) uint8 with 0=I, 1=X, 2=Y, 3=Z
        sign: Sign of the stabilizer
        phase: Accumulated phase factor
        n_qubits: Number of qubits

    Returns:
        Tuple of (new_state, new_phase)
    """
    post_state = state
    new_phase = phase * sign

    for n in range(n_qubits):
        pauli = pauli_types[n]
        bit = (post_state >> n) & 1

        if pauli == 1:  # X
            post_state ^= (1 << n)
        elif pauli == 2:  # Y
            post_state ^= (1 << n)
            new_phase = new_phase * (1j) * (-1) ** bit
        elif pauli == 3:  # Z
            new_phase = new_phase * (-1) ** bit

    return post_state, new_phase


def gaussian_elimination_fast(
    stab_x_bits: np.ndarray,
    stab_pauli_types: np.ndarray,
    stab_signs: np.ndarray,
    ref_state: int,
    target_state: int,
    n_qubits: int,
) -> complex:
    """Fast version of gaussian_elimination using precomputed numpy arrays.

    Args:
        stab_x_bits: Array shape (n_stab, n_qubits) bool - X bits per stabilizer
        stab_pauli_types: Array shape (n_stab, n_qubits) uint8 - Pauli type per qubit
        stab_signs: Array shape (n_stab,) complex - sign per stabilizer
        ref_state: Reference computational basis state as int (little-endian)
        target_state: Target computational basis state as int (little-endian)
        n_qubits: Number of qubits

    Returns:
        Complex phase factor for the overlap
    """
    phase = 1 + 0j
    interm = ref_state
    target = target_state
    n_stab = stab_x_bits.shape[0]

    for n in range(n_qubits):
        if ((target >> n) & 1) != ((interm >> n) & 1):
            for m in range(n_stab):
                if not stab_x_bits[m, n]:
                    continue
                if n > 0 and np.any(stab_x_bits[m, :n]):
                    continue
                interm, phase = apply_stabilizer_to_state_fast(
                    interm, stab_pauli_types[m], stab_signs[m], phase, n_qubits,
                )
                break

        if ((target >> n) & 1) != ((interm >> n) & 1):
            return 0j

    return phase


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


def spinorb_to_spatial_2rdm(
    rdm2_so: np.ndarray,
    norb: int,
) -> np.ndarray:
    """Convert a spin-orbital 2-RDM to a spatial-orbital 2-RDM in chemist's notation.

    Assumes spin-orbital ordering [α₀, ..., α_{n-1}, β₀, ..., β_{n-1}].
    Sums over αα, ββ, and αβ spin blocks with the appropriate transpositions.

    Args:
        rdm2_so: Spin-orbital 2-RDM, shape (2*norb, 2*norb, 2*norb, 2*norb).
        norb: Number of spatial orbitals.

    Returns:
        Spatial 2-RDM in chemist's notation, shape (norb, norb, norb, norb).
    """
    rdm2_aa = rdm2_so[:norb, :norb, :norb, :norb]
    rdm2_ab = rdm2_so[:norb, norb:, :norb, norb:]
    rdm2_bb = rdm2_so[norb:, norb:, norb:, norb:]

    dm2aa = rdm2_aa.transpose(0, 2, 1, 3)
    dm2ab = rdm2_ab.transpose(0, 2, 1, 3)
    dm2bb = rdm2_bb.transpose(0, 2, 1, 3)
    return dm2aa + dm2bb + dm2ab + dm2ab.transpose(1, 0, 3, 2)


def doubles_energy(
    rdm2: np.ndarray,
    mf: Union[scf.hf.RHF, scf.uhf.UHF],
) -> float:
    """Compute the two-electron energy from a spatial 2-RDM.

    E₂ = 0.5 * Σ_{pqrs} g_{pqrs} Γ_{pqrs}

    Args:
        rdm2: Spatial 2-RDM in chemist's notation, shape (norb, norb, norb, norb).
        mf: PySCF mean-field object (RHF or UHF).

    Returns:
        Two-electron energy in Hartrees.
    """
    norb = mf.mo_coeff.shape[1] if isinstance(mf, scf.hf.RHF) else mf.mo_coeff[0].shape[1]
    mo = mf.mo_coeff if isinstance(mf, scf.hf.RHF) else mf.mo_coeff[0]
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mo), norb)
    return 0.5 * np.einsum("ijkl,ijkl->", eri, rdm2)


def total_energy_from_rdms(
    dm1: np.ndarray,
    dm2: np.ndarray,
    mf: Union[scf.hf.RHF, scf.uhf.UHF],
) -> float:
    """Compute the total electronic energy from 1-RDM and 2-RDM.

    E = Σ_{pq} h_{pq} γ_{pq} + 0.5 * Σ_{pqrs} g_{pqrs} Γ_{pqrs} + E_nuc

    Args:
        dm1: Spatial 1-RDM, shape (norb, norb).
        dm2: Spatial 2-RDM in chemist's notation, shape (norb, norb, norb, norb).
        mf: PySCF mean-field object (RHF or UHF).

    Returns:
        Total energy in Hartrees.
    """
    mo = mf.mo_coeff if isinstance(mf, scf.hf.RHF) else mf.mo_coeff[0]
    h1 = mo.T @ mf.get_hcore() @ mo
    norb = h1.shape[0]
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mo), norb)
    e1 = np.einsum("pq,pq->", h1, dm1)
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eri, dm2)
    return e1 + e2 + mf.mol.energy_nuc()


def pauli_terms_to_matrix(
        hamiltonian: List[tuple[str, float]]
    ) -> NDArray[np.complex128]:
    """Convert a Hamiltonian represented as Pauli terms to a full matrix.

    Args:
        hamiltonian: List of (coefficient, pauli_string) tuples
                    Examples: [(1.0, 'XY'), (0.5, 'ZI'), (2.0, 'XX')]

    Returns:
        Full matrix representation of the Hamiltonian as a 2^n x 2^n complex array

    Example:
        >>> # Hamiltonian H = X⊗Y + 0.5*Z⊗I
        >>> hamiltonian = [(1.0, 'XY'), (0.5, 'ZI')]
        >>> matrix = pauli_terms_to_matrix(hamiltonian)
        >>> matrix.shape
        (4, 4)
    """
    if not hamiltonian:
        raise ValueError("Hamiltonian must contain at least one term")

    # Get number of qubits from first term
    n_qubits = len(hamiltonian[0][1])
    dim = 2 ** n_qubits

    # Validate all terms have same length
    for coeff, pauli_string in hamiltonian:
        if len(pauli_string) != n_qubits:
            raise ValueError(
                f"All Pauli strings must have the same length. "
                f"Expected {n_qubits}, got {len(pauli_string)} for '{pauli_string}'"
            )

    # Define single-qubit Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # Build full Hamiltonian matrix
    H = np.zeros((dim, dim), dtype=np.complex128)

    for coeff, pauli_string in hamiltonian:
        # Build tensor product for this term
        term_matrix = np.array([[1.0]], dtype=np.complex128)
        for pauli_char in pauli_string:
            if pauli_char not in pauli_map:
                raise ValueError(
                    f"Invalid Pauli character '{pauli_char}'. "
                    f"Must be one of 'I', 'X', 'Y', 'Z'"
                )
            term_matrix = np.kron(term_matrix, pauli_map[pauli_char])

        H += coeff * term_matrix

    return H

def tableau_to_qulacs_circuit(tab: stim.Tableau, n_qubits: int) -> qulacs.QuantumCircuit:
    """Convert a stim.Tableau directly to a qulacs QuantumCircuit.

    Uses stim's canonical decomposition into {H, S, CX} gates,
    bypassing the Qiskit/QASM intermediate representation.

    Note: stim batches gate targets, e.g. ``S 0 0 1`` means apply S
    to qubit 0 twice then to qubit 1. Two-qubit gates (CX) are batched
    in pairs of targets.
    """
    stim_circuit = tab.to_circuit()
    circuit = qulacs.QuantumCircuit(n_qubits)

    for op in stim_circuit:
        name = op.name
        targets = [t.qubit_value for t in op.targets_copy()]

        if name == "H":
            for q in targets:
                circuit.add_H_gate(q)
        elif name == "S":
            for q in targets:
                circuit.add_S_gate(q)
        elif name == "CX":
            for i in range(0, len(targets), 2):
                circuit.add_CNOT_gate(targets[i], targets[i + 1])
        else:
            raise ValueError(f"Unexpected gate in Clifford decomposition: {name}")

    return circuit


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

    from shades.estimators import ExactEstimator

    estimator = ExactEstimator(mf, fci)

    t2_aa, t2_bb, t2_ab = estimator.estimate_c2()

    
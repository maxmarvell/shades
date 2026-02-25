from typing import List, Callable, Literal, Union, Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyscf import scf
from pyscf.ci.cisd import tn_addrs_signs

type SingleSpinCase = Literal["alpha", "beta"]
type DoubleSpinCase = Literal["alpha-alpha", "beta-beta", "alpha-beta"]


def get_hf_reference(mf: Union[scf.hf.RHF, scf.uhf.UHF]) -> int:
    nalpha, nbeta = mf.mol.nelec
    norb = mf.mol.nao
    return sum(1 << i for i in range(nalpha)) + sum(1 << (i + norb) for i in range(nbeta))


@dataclass
class SingleExcitation:
    """Represents a single excitation with its indices and bitstring."""

    occ: int  # occupied orbital index
    virt: int  # virtual orbital index
    spin: SingleSpinCase  # 'alpha' or 'beta'
    bitstring: int
    n: int  # number of occ alpha or beta orbitals

    def __repr__(self) -> str:
        spin_symbol = "α" if self.spin == "alpha" else "β"
        return f"{self.occ}{spin_symbol} → {self.virt + self.n}{spin_symbol}"


def get_singles(
    mf: Union[scf.hf.RHF, scf.uhf.UHF],
    *,
    spin_cases: Optional[List[SingleSpinCase]] = None,
    mp2_screening: bool = False,  # need to implement this
) -> List[SingleExcitation]:
    if spin_cases is None:  # if none just assume default
        if isinstance(mf, scf.hf.RHF):
            spin_cases = ["alpha"]
        else:
            spin_cases = ["alpha", "beta"]

    excitations = []
    reference = get_hf_reference(mf)

    n_alpha, n_beta = mf.mol.nelec
    norb = mf.mol.nao

    if "alpha" in spin_cases:
        occupied_alpha = list(range(n_alpha))
        virtual_alpha = list(range(n_alpha, norb))
        for i in occupied_alpha:
            for a in virtual_alpha:
                excited_state = (reference & ~(1 << i)) | (1 << a)
                excitations.append(
                    SingleExcitation(
                        occ=i, virt=a - n_alpha, spin="alpha", bitstring=excited_state, n=n_alpha
                    )
                )

    if "beta" in spin_cases:
        n_qubits = 2 * norb

        occupied_beta = list(range(norb, norb + n_beta))
        virtual_beta = list(range(norb + n_beta, n_qubits))

        for i in occupied_beta:
            for a in virtual_beta:
                excited_state = (reference & ~(1 << i)) | (1 << a)
                excitations.append(
                    SingleExcitation(
                        occ=i - norb,
                        virt=a - norb - n_beta,
                        spin="beta",
                        bitstring=excited_state,
                        n=n_beta,
                    )
                )

    return excitations


def singles_to_t1(
    excitations: List[SingleExcitation],
    amplitude_fn: Callable[[int], np.float64],
    nocc: int,
    nvirt: int,
) -> NDArray[np.float64]:
    c1 = np.zeros((nocc, nvirt), dtype=np.float64)
    for ex in excitations:
        i = ex.occ
        a = ex.virt
        c1[i, a] = amplitude_fn(ex.bitstring)

    norb = nocc + nvirt
    _, t1sign = tn_addrs_signs(norb, nocc, 1)
    t1 = (c1.reshape(-1) * t1sign).reshape(nocc, nvirt)
    return t1


@dataclass
class DoubleExcitation:
    """Represents a double excitation with its indices and bitstring."""

    occ1: int
    occ2: int
    virt1: int
    virt2: int
    spin_case: DoubleSpinCase  # 'alpha-alpha', 'beta-beta', or 'alpha-beta'
    bitstring: int
    n1: int  # number of alpha or beta orbitals
    n2: int  # number of alpha or beta orbitals

    def __repr__(self) -> str:
        if self.spin_case == "alpha-alpha":
            return (
                f"({self.occ1}α,{self.occ2}α) → ({self.virt1 + self.n1}α,{self.virt2 + self.n2}α)"
            )
        elif self.spin_case == "beta-beta":
            return (
                f"({self.occ1}β,{self.occ2}β) → ({self.virt1 + self.n1}β,{self.virt2 + self.n2}β)"
            )
        else:  # alpha-beta
            return (
                f"({self.occ1}α,{self.occ2}β) → ({self.virt1 + self.n1}α,{self.virt2 + self.n2}β)"
            )


def get_doubles(
    mf: Union[scf.hf.RHF, scf.uhf.UHF],
    *,
    spin_cases: Optional[List[DoubleSpinCase]] = None,
    symmetry_restricted: Optional[bool] = None,
    mp2_screening: bool = False,  # need to implement this
) -> List[DoubleExcitation]:
    
    if spin_cases is None:  # if none just assume default
        if isinstance(mf, scf.hf.RHF):
            spin_cases = ["alpha-beta"]
        else:
            spin_cases = ["alpha-alpha", "beta-beta", "alpha-beta"]

    if symmetry_restricted is None:
        if isinstance(mf, scf.hf.RHF):
            symmetry_restricted = True
        else:
            symmetry_restricted = False

    excitations = []

    n_alpha, n_beta = mf.mol.nelec
    norb = mf.mol.nao
    n_qubits = 2 * norb

    reference = get_hf_reference(mf)

    # alpha spin indices
    occupied_alpha = list(range(n_alpha))
    virtual_alpha = list(range(n_alpha, norb))

    # beta spin indices
    occupied_beta = list(range(norb, norb + n_beta))
    virtual_beta = list(range(norb + n_beta, n_qubits))

    # alpha-beta mixed excitations
    if "alpha-beta" in spin_cases:
        for idx_i, i in enumerate(occupied_alpha):
            idx_i = idx_i + 1 if symmetry_restricted and isinstance(mf, scf.hf.RHF) else 0
            for j in occupied_beta[idx_i:]:
                if i + norb == j and symmetry_restricted and isinstance(mf, scf.hf.RHF):
                    continue  # not allowed
                for a in virtual_alpha:
                    for b in virtual_beta:
                        if a + norb == b and symmetry_restricted and isinstance(mf, scf.hf.RHF):
                            continue  # not allowed
                        excited_state = (reference & ~(1 << i) & ~(1 << j)) | (1 << a) | (1 << b)
                        excitations.append(
                            DoubleExcitation(
                                occ1=i,
                                occ2=j - norb,
                                virt1=a - n_alpha,
                                virt2=b - norb - n_beta,
                                spin_case="alpha-beta",
                                bitstring=excited_state,
                                n1=n_alpha,
                                n2=n_beta,
                            )
                        )

        # get double excitations from same orbital
        if symmetry_restricted and isinstance(mf, scf.hf.RHF):
            for i in occupied_alpha:
                for a in virtual_alpha:
                    excited_state = (reference & ~(1 << i) & ~(1 << (i + norb))) | (1 << a) | (1 << (a + norb))
                    excitations.append(
                        DoubleExcitation(
                            occ1=i,
                            occ2=i,
                            virt1=a - n_alpha,
                            virt2=a - n_alpha,
                            spin_case="alpha-beta",
                            bitstring=excited_state,
                            n1=n_alpha,
                            n2=n_alpha,
                        )
                    )

    if "alpha-alpha" in spin_cases:
        for idx_i, i in enumerate(occupied_alpha):
            idx_i = idx_i + 1 if symmetry_restricted else 0
            for j in occupied_alpha[idx_i:]:
                if i == j: continue
                for idx_a, a in enumerate(virtual_alpha):
                    idx_a = idx_a + 1 if symmetry_restricted else 0
                    for b in virtual_alpha[idx_a:]:
                        if a == b: continue
                        excited_state = (reference & ~(1 << i) & ~(1 << j)) | (1 << a) | (1 << b)
                        excitations.append(
                            DoubleExcitation(
                                occ1=i,
                                occ2=j,
                                virt1=a - n_alpha,
                                virt2=b - n_alpha,
                                spin_case="alpha-alpha",
                                bitstring=excited_state,
                                n1=n_alpha,
                                n2=n_alpha,
                            )
                        )

    if "beta-beta" in spin_cases:
        for idx_i, i in enumerate(occupied_beta):
            for j in occupied_beta[idx_i + 1 :]:
                for idx_a, a in enumerate(virtual_beta):
                    for b in virtual_beta[idx_a + 1 :]:
                        excited_state = (reference & ~(1 << i) & ~(1 << j)) | (1 << a) | (1 << b)
                        excitations.append(
                            DoubleExcitation(
                                occ1=i - norb,
                                occ2=j - norb,
                                virt1=a - norb - n_beta,
                                virt2=b - norb - n_beta,
                                spin_case="beta-beta",
                                bitstring=excited_state,
                                n1=n_beta,
                                n2=n_beta,
                            )
                        )

    return excitations


def doubles_to_t2(
    excitations: List[DoubleExcitation],
    amplitude_fn: Callable[[int], np.float64],
    nocc: Union[int, tuple[int, int]],
    nvirt: Union[int, tuple[int, int]],
    spin_case: DoubleSpinCase,
    symmetry_restricted: bool = False,
) -> NDArray[np.float64]:
    if isinstance(nocc, int):
        nocc1, nocc2 = nocc, nocc
    else:
        nocc1, nocc2 = nocc

    if isinstance(nvirt, int):
        nvirt1, nvirt2 = nvirt, nvirt
    else:
        nvirt1, nvirt2 = nvirt

    c2 = np.zeros((nocc1, nvirt1, nocc2, nvirt2), dtype=np.float64)
    for ex in excitations:
        c2[ex.occ1, ex.virt1, ex.occ2, ex.virt2] = amplitude_fn(ex.bitstring)

    if spin_case in ["alpha-alpha", "beta-beta"] and symmetry_restricted:
        for ex in excitations:
            val = c2[ex.occ1, ex.virt1, ex.occ2, ex.virt2]
            c2[ex.occ2, ex.virt1, ex.occ1, ex.virt2] = - val
            c2[ex.occ1, ex.virt2, ex.occ2, ex.virt1] = - val
            c2[ex.occ2, ex.virt2, ex.occ1, ex.virt1] = val

        c2 = - c2

    if spin_case in ["alpha-beta"] and symmetry_restricted:
        for ex in excitations:
            val = c2[ex.occ1, ex.virt1, ex.occ2, ex.virt2]
            c2[ex.occ2, ex.virt2, ex.occ1, ex.virt1] = val

    norb1 = nocc1 + nvirt1
    norb2 = nocc2 + nvirt2
    _, t1sign1 = tn_addrs_signs(norb1, nocc1, 1)
    _, t1sign2 = tn_addrs_signs(norb2, nocc2, 1)

    c2_flat = c2.reshape(nocc1 * nvirt1, nocc2 * nvirt2)
    t2_flat = np.outer(t1sign1, t1sign2) * c2_flat
    t2 = t2_flat.reshape(nocc1, nvirt1, nocc2, nvirt2).transpose(0, 2, 1, 3)
    return t2


if __name__ == "__main__":
    from shades.utils import make_hydrogen_chain
    from pyscf import gto, scf

    atom = make_hydrogen_chain(4, bond_length=1.0)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g", verbose=0, spin=0)
    mf = scf.UHF(mol)
    mf.run()

    get_hf_reference(mf)

    doubles = get_doubles(mf, spin_cases=["alpha-alpha"])

    print(doubles)

    n_alpha, n_beta = mf.mol.nelec
    norb = mf.mol.nao

    nocc_a, nvirt_a = n_alpha, norb - n_alpha
    nocc_b, nvirt_b = n_beta, norb - n_beta

    from shades.solvers import FCISolver

    fci = FCISolver(mf)
    state, _ = fci.solve()

    doubles_to_t2(
        doubles, lambda b: state.data[b].real, nocc_a, nocc_b, spin_case="alpha-alpha"
    )

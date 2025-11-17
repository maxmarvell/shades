from typing import List, Callable, Literal, Union
from dataclasses import dataclass
from shadow_ci.utils import Bitstring
import numpy as np
from numpy.typing import NDArray
from pyscf import scf

def get_hf_reference(mf: Union[scf.hf.RHF, scf.uhf.UHF]) -> Bitstring:
    n_alpha, n_beta = mf.mol.nelec
    norb = mf.mo_coeff.shape[0]
    alpha_string = [True] * n_alpha + [False] * (norb - n_alpha) 
    beta_string = [True] * n_beta + [False] * (norb - n_beta) 
    return Bitstring(alpha_string + beta_string, endianess='little')

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

class SinglesSector:
    """Sector of single excitations with symmetry information."""

    excitations: List[SingleExcitation]
    spin_type: Literal["RHF", "UHF"]

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF], *, symmetry_reduced: bool = True):

        self.symmetry_reduced = symmetry_reduced
        if isinstance(mf, scf.hf.RHF):
            self.spin_type = "RHF"
        else:
            self.spin_type = "UHF"

        n_alpha, n_beta = mf.mol.nelec
        norb = mf.mo_coeff.shape[0]
        

        self.nocc = n_alpha
        self.nvirt = norb - n_alpha

        self.excitations = self._get_excitations(mf)

    @property
    def n_unique(self) -> int:
        """Number of unique excitations stored."""
        return len(self.excitations)
    
    @property
    def n_total(self) -> int:
        """Total excitations including symmetry equivalents."""
        if self.spin_type == "RHF":
            return len(self.excitations) * 2  # α + β
        return len(self.excitations)
    
    def expand_to_full(self) -> "SinglesSector":
        """Expand symmetry-reduced set to include all spin channels."""
        raise NotImplementedError()
    
    def excitations_to_t1(self, amplitude_fn: Callable[[Bitstring], np.float64]) -> NDArray[np.float64]:
        """Convert list of coefficients to structured tensor."""

        t1 = np.zeros((self.nocc, self.nvirt), dtype=np.float64)
        for ex in self.excitations:
            i = ex.occ
            a = ex.virt
            t1[i,a] = amplitude_fn(ex.bitstring)

        return t1

    def _get_excitations(self, mf) -> List[SingleExcitation]:

        excitations = []
        reference = get_hf_reference(mf)

        n_alpha, n_beta = mf.mol.nelec
        norb = mf.mo_coeff.shape[0]

        occupied_alpha = list(range(n_alpha))
        virtual_alpha = list(range(n_alpha, norb))

        # alpha excitations
        for i in occupied_alpha:
            for a in virtual_alpha:
                excited_state = reference.copy()
                excited_state[i] = False
                excited_state[a] = True
                excitations.append(SingleExcitation(
                    occ=i,
                    virt=a - n_alpha,
                    spin='alpha',
                    bitstring=excited_state,
                    n=n_alpha
                ))

        if isinstance(mf, scf.hf.RHF) and self.symmetry_reduced: return excitations

        n_qubits = 2 * norb

        occupied_beta = list(range(norb, norb + n_beta))
        virtual_beta = list(range(norb + n_beta, n_qubits))

        # beta excitations
        for i in occupied_beta:
            for a in virtual_beta:
                excited_state = reference.copy()
                excited_state[i] = False
                excited_state[a] = True
                excitations.append(SingleExcitation(
                    occ=i - norb,
                    virt=a - norb - n_beta,
                    spin='beta',
                    bitstring=excited_state, 
                    n=n_beta
                ))

        return excitations
        
@dataclass
class DoubleExcitation:
    """Represents a double excitation with its indices and bitstring."""
    occ1: int
    occ2: int
    virt1: int
    virt2: int
    spin_case: str  # 'alpha-alpha', 'beta-beta', or 'alpha-beta'
    bitstring: Bitstring
    n1: int # number of alpha or beta orbitals 
    n2: int # number of alpha or beta orbitals

    def __repr__(self) -> str:
        if self.spin_case == 'alpha-alpha':
            return f"({self.occ1}α,{self.occ2}α) → ({self.virt1 + self.n1}α,{self.virt2 + self.n2}α)"
        elif self.spin_case == 'beta-beta':
            return f"({self.occ1}β,{self.occ2}β) → ({self.virt1 + self.n1}β,{self.virt2 + self.n2}β)"
        else:  # alpha-beta
            return f"({self.occ1}α,{self.occ2}β) → ({self.virt1 + self.n1}α,{self.virt2 + self.n2}β)"

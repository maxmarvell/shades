from shades.solvers.base import GroundStateSolver
from typing import Tuple, Union, Optional
import numpy as np
from qiskit.quantum_info import Statevector
from pyscf import scf, fci, ci

from shades.excitations import get_hf_reference, get_doubles, get_singles, singles_to_t1, doubles_to_t2

class FCISolver(GroundStateSolver):

    civec: Optional[np.ndarray]

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF]):
        super().__init__(mf)
        self.civec = None

    def solve(self) -> Tuple[np.float64, np.ndarray]:
        self.energy, self.civec = fci.FCI(self.mf).kernel()
        self.state = self._civec_to_statevector(self.civec)
        return self.state, self.energy

    def _civec_to_statevector(self, civec: np.ndarray) -> Statevector:
        """Convert PySCF FCI CI vector to Qiskit Statevector.

        PySCF FCI stores the wavefunction in a compressed format over
        determinants. This method expands it to the full 2^n qubit basis.

        Args:
            civec: CI vector from PySCF FCI solver

        Returns:
            Qiskit Statevector in computational basis
        """

        n_alpha, n_beta = self.mf.mol.nelec
        norb = self.mf.mol.nao
        n_qubits = 2 * norb

        full_statevector = np.zeros(2**n_qubits, dtype=complex)

        if isinstance(self.mf, scf.hf.RHF):

            alpha_strings = fci.cistring.make_strings(range(norb), n_alpha)
            beta_strings = fci.cistring.make_strings(range(norb), n_beta)

            for i_alpha, alpha_str in enumerate(alpha_strings):
                for i_beta, beta_str in enumerate(beta_strings):
                    ci_coeff = civec[i_alpha, i_beta]

                    qubit_index = int(alpha_str) + (int(beta_str) << norb)

                    full_statevector[qubit_index] = ci_coeff
        else:

            alpha_strings = fci.cistring.make_strings(range(norb), n_alpha)
            beta_strings = fci.cistring.make_strings(range(norb), n_beta)

            civec_flat = civec.ravel()

            for i_alpha, alpha_str in enumerate(alpha_strings):
                for i_beta, beta_str in enumerate(beta_strings):
                    idx = i_alpha * len(beta_strings) + i_beta
                    ci_coeff = civec_flat[idx]

                    qubit_index = int(alpha_str) + (int(beta_str) << norb)
                    full_statevector[qubit_index] = ci_coeff

        return Statevector(full_statevector)

    def get_configuration_interaction(self):
        if not self.state: self.solve()
        c0 = self._get_reference()
        c1 = self._get_singles()
        return self.energy, c0, c1, None

    def _get_reference(self) -> np.float64:
        ref = get_hf_reference(self.mf)
        return self.state[ref.to_int()].real

    def _get_singles(self) -> np.ndarray:

        nocc, _ = self.mf.mol.nelec
        norb = self.mf.mol.nao
        nvirt = norb - nocc

        singles = get_singles(self.mf)
        t1 = singles_to_t1(singles, lambda b: self.state[b.to_int()].real, nocc, norb)
        return t1

    def estimate(
        self, mf: Union[scf.hf.RHF, scf.uhf.UHF]
    ) -> tuple[np.float64, np.float64, np.ndarray, np.ndarray]:
        """Estimate ground state properties for a given mean-field object.

        This method can be used as an estimator_fn for Brueckner optimization.

        Args:
            mf: Mean-field object (RHF or UHF)

        Returns:
            Tuple of (energy, c0, c1, c2) where c2 is currently None
        """
        self.mf = mf
        self.solve()
        return self.get_configuration_interaction()

if __name__ == "__main__":

    from shades.utils import make_hydrogen_chain
    from pyscf import gto

    atom = make_hydrogen_chain(6)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g", verbose=0)

    mf = scf.RHF(mol)
    mf.run()

    solver = FCISolver(mf)

    solver.solve()


    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nocc, _ = mf.mol.nelec
    
    print(solver.get_configuration_interaction())


    myci = ci.CISD(mf).run()

    # Extract amplitudes from CI vector
    norb = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2

    # Unpack the CI vector
    c0, c1, c2 = myci.cisdvec_to_amplitudes(myci.ci, norb, nocc)
    print(f"c0 = {c0}")
    print(f"c1 (t1 amplitudes):\n{c1}")
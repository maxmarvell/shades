from shades.estimators import AbstractEstimator
from shades.solvers import GroundStateSolver
from shades.utils import Bitstring
from pyscf import scf
from typing import Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ExactEstimator(AbstractEstimator):

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF], solver: GroundStateSolver, *, verbose: int = 0):
        super().__init__(mf, solver, verbose)

    def estimate_overlap(self, a: Union[Bitstring, int]) -> np.float64:
        if isinstance(a, int):
            return self.trial.data[a].real
        idx = a.to_int()
        return self.trial.data[idx].real


if __name__ == "__main__":

    from shades.utils import make_hydrogen_chain
    from pyscf import gto, scf

    atom = make_hydrogen_chain(7, bond_length=3.5)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g", verbose=0, spin=1)
    mf = scf.UHF(mol)
    mf.run()

    from shades.solvers import FCISolver
    fci = FCISolver(mf)

    estimator = ExactEstimator(mf, fci)

    E, c0, c1, c2 = estimator.run(calc_c1=True)

    print(E)
    print(estimator.E_exact)
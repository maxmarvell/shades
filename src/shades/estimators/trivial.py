from shades.estimators import AbstractEstimator
from shades.solvers import GroundStateSolver
from shades.utils import Bitstring
from pyscf import scf
from typing import Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TrivialEstimator(AbstractEstimator):

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF], solver: GroundStateSolver):
        super().__init__(mf, solver)

    def estimate_overlap(self, a: Bitstring) -> np.float64:
        idx = a.to_int()  
        return self.trial[idx].real


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

    estimator = TrivialEstimator(mf, fci)

    E, c0, c1, c2 = estimator.run(calc_c1=True)

    print(E)
    print(estimator.E_exact)
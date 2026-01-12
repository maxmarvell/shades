from pyscf import scf
from itertools import product, combinations
from typing import Callable, List, Tuple
import numpy as np

from abc import ABC
import random

from shades.estimators import AbstractEstimator
from shades.excitations import (
    get_hf_reference
)
from shades.utils import Bitstring

def _gen_single_site_hops(
    reference: int, n_qubits: int
) -> Tuple[List[int], List[Tuple[int, int]]]:
    
    #TODO: this assumes spin symettry where number of alpha and beta orbitals are the same (I think)
    assert n_qubits % 2 == 0

    res = []
    n_spatial = n_qubits // 2

    alpha_occ = [i for i in range(n_spatial) if (reference >> i) & 1]
    alpha_vac = [i for i in range(n_spatial) if not (reference >> i) & 1]
    alpha_hops = [(o, v) for o, v in product(alpha_occ, alpha_vac)]
    res += [reference ^ (1 << i) ^ (1 << j) for i, j in alpha_hops]
    
    beta_occ = [i for i in range(n_spatial, n_qubits) if (reference >> i) & 1]
    beta_vac = [i for i in range(n_spatial, n_qubits) if not (reference >> i) & 1]
    beta_hops = [(o, v) for o, v in product(beta_occ, beta_vac)]
    res += [reference ^ (1 << i) ^ (1 << j) for i in beta_occ for j in beta_vac]
    
    return res, alpha_hops + beta_hops


def _gen_double_site_hops(
    reference: int, n_qubits: int
) -> Tuple[List[int], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    
    #TODO: this assumes spin symettry where number of alpha and beta orbitals are the same (I think)
    assert n_qubits % 2 == 0

    n_spatial = n_qubits // 2
    res = []

    alpha_occ = [i for i in range(n_spatial) if (reference >> i) & 1]
    alpha_vac = [i for i in range(n_spatial) if not (reference >> i) & 1]
    double_alpha = [(o, v) for o, v in product(combinations(alpha_occ, 2), combinations(alpha_vac, 2))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in double_alpha]


    beta_occ = [i for i in range(n_spatial, n_qubits) if (reference >> i) & 1]
    beta_vac = [i for i in range(n_spatial, n_qubits) if not (reference >> i) & 1]
    double_beta = [(o, v) for o, v in product(combinations(beta_occ, 2), combinations(beta_vac, 2))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in double_beta]


    alpha_beta = [(o, v) for o, v in product(product(alpha_occ, beta_occ), product(alpha_vac, beta_vac))]
    res += [reference ^ (1 << i[0]) ^ (1 << i[1]) ^ (1 << j[0]) ^ (1 << j[1]) for i, j in alpha_beta]

    
    return res, double_alpha + double_beta + alpha_beta


def spin_displacement(ref: Bitstring) -> Tuple[Bitstring, float]:
    candidates = _gen_single_site_hops(ref)
    return random.choice(candidates)

class MonteCarloEstimator:

    def __init__(self, estimator: AbstractEstimator, transition_fn: str = 'displacement'):

        self.estimator = estimator
        
        if transition_fn == 'displacement':
            self.T = spin_displacement
        else:
            raise NotImplementedError()

    
    def metropolis(self, n: Bitstring, m: Bitstring) -> bool:
        c_n = self.estimator.estimate_overlap(n)
        c_m = self.estimator.estimate_overlap(m)
        P = min(1, c_m*np.conj(c_m)/(c_n*np.conj(c_n)))
        return random.random() >= P
    
    @staticmethod
    def double_excitations(n: Bitstring) -> List[Bitstring]:
        res = _gen_double_site_hops(n.to_int(), n.size)
        return [Bitstring.from_int(res, size=n.size)]


    def estimate_rdm(self) -> np.ndarray:

        psi0 = get_hf_reference(self.estimator.mf)

        psi = spin_displacement(psi0)



        self.estimator.estimate_overlap()
        pass
    


if __name__ == "__main__":

    reference = Bitstring.from_string('10011001')

    result = _gen_single_site_hops(reference.to_int(), n_qubits=8)

    result = _gen_double_site_hops(reference.to_int(), n_qubits=8)

    print(result)
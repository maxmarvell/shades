import logging
from copy import copy
from typing import Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pyscf import lib, scf
from scipy.linalg import eigh, expm

from shades.estimators.base import AbstractEstimator

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


def make_kappa(t1: NDArray[np.float64]) -> NDArray[np.float64]:
    nocc, nvirt = t1.shape
    norb = nocc + nvirt

    kappa = np.zeros((norb, norb))
    kappa[-nvirt:, :nocc] = t1.conj().T
    kappa[:nocc, -nvirt:] = -t1

    return kappa


def rotate_mo_coeffs(
    C: Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]],
    t1: Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]],
    ovlp: Optional[NDArray[np.float64]],
    damping: float = 0.0,
    diis: Optional[lib.diis.DIIS] = None,
    method: Literal["expn", "taylor"] = "taylor",
):
    if C.ndim == 3:
        is_uhf = True
    elif C.ndim == 2:
        is_uhf = False
    else:
        raise ValueError()

    if is_uhf:
        nocc_a, nvirt_a = t1[0].shape
        nocc_b, nvirt_b = t1[1].shape
        norb = C[0].shape[-1]
    else:
        nocc, nvirt = t1.shape
        norb = C.shape[-1]

        if not nocc + nvirt == norb:
            raise ValueError("The shape of molecular orbital coefficients and t1 do not match!")

    if method == "expn":
        if is_uhf:
            # UHF case: handle alpha and beta spin channels separately
            kappa_a = make_kappa(t1[0])
            kappa_b = make_kappa(t1[1])

            # Apply unitary transformation: U = exp(kappa * (1 - damping))
            U_a = expm(kappa_a * (1 - damping))
            U_b = expm(kappa_b * (1 - damping))

            # Transform MO coefficients: correct order is U @ C, not C @ U
            bmo_a = U_a @ C[0]
            bmo_b = U_b @ C[1]

            # Extract occupied and virtual orbitals
            bmo_occ = [bmo_a[:, :nocc_a], bmo_b[:, :nocc_b]]
            bmo_vir = [bmo_a[:, nocc_a:], bmo_b[:, nocc_b:]]

            bmo = [np.hstack((bmo_occ[0], bmo_vir[0])), np.hstack((bmo_occ[1], bmo_vir[1]))]

        else:
            kappa = make_kappa(t1)
            U = expm(kappa * (1 - damping))
            bmo = C @ U 
            bmo_occ = bmo[:, :nocc]
            bmo_vir = bmo[:, nocc:]

    elif method == "taylor":
        if is_uhf:
            delta_occ = [
                (1 - damping) * np.dot(C[0][:, nocc_a:], t1[0].T),
                (1 - damping) * np.dot(C[1][:, nocc_b:], t1[1].T),
            ]

            bmo_occ = [C[0][:, :nocc_a] + delta_occ[0], C[1][:, :nocc_b] + delta_occ[1]]

            if ovlp is None:
                bmo_occ = [np.linalg.qr(bmo_occ[0])[0], np.linalg.qr(bmo_occ[1])[0]]
            else:
                dm_occ = [np.dot(bmo_occ[0], bmo_occ[0].T), np.dot(bmo_occ[1], bmo_occ[1].T)]
                v = [eigh(dm_occ[0], b=ovlp, type=2)[1], eigh(dm_occ[1], b=ovlp, type=2)[1]]
                bmo_occ = [v[0][:, -nocc_a:], v[1][:, -nocc_b:]]

            if diis:
                dm_occ = np.array(
                    [np.dot(bmo_occ[0], bmo_occ[0].T), np.dot(bmo_occ[1], bmo_occ[1].T)]
                )
                dm_occ = diis.update(dm_occ)
                v = [eigh(dm_occ[0], b=ovlp, type=2)[1], eigh(dm_occ[1], b=ovlp, type=2)[1]]
                bmo_occ = [v[0][:, -nocc_a:], v[1][:, -nocc_b:]]

            if ovlp is None:  # get virtuals by unitary completion
                dm_vir = [
                    np.eye(norb) - np.dot(bmo_occ[0], bmo_occ[0].T),
                    np.eye(norb) - np.dot(bmo_occ[1], bmo_occ[1].T),
                ]
            else:
                dm_vir = [
                    np.linalg.inv(ovlp) - np.dot(bmo_occ[0], bmo_occ[0].T),
                    np.linalg.inv(ovlp) - np.dot(bmo_occ[1], bmo_occ[1].T),
                ]

            v = [eigh(dm_vir[0], b=ovlp, type=2)[1], eigh(dm_vir[1], b=ovlp, type=2)[1]]
            bmo_vir = [v[0][:, -nvirt_a:], v[1][:, -nvirt_b:]]

            bmo = [np.hstack((bmo_occ[0], bmo_vir[0])), np.hstack((bmo_occ[1], bmo_vir[1]))]

        else:
            delta_occ = (1 - damping) * np.dot(C[:, nocc:], t1.T)  # multiply virtuals by t1.T
            bmo_occ = C[:, :nocc] + delta_occ

            if ovlp is None:
                bmo_occ = np.linalg.qr(bmo_occ)[0]
            else:
                dm_occ = np.dot(bmo_occ, bmo_occ.T)
                _, v = eigh(dm_occ, b=ovlp, type=2)
                bmo_occ = v[:, -nocc:]

            if diis:
                dm_occ = np.dot(bmo_occ, bmo_occ.T)
                dm_occ = diis.update(dm_occ)
                _, v = eigh(dm_occ, b=ovlp, type=2)
                bmo_occ = v[:, -nocc:]

            if ovlp is None:  # get virtuals by unitary completion
                dm_vir = np.eye(norb) - np.dot(bmo_occ, bmo_occ.T)
            else:
                dm_vir = np.linalg.inv(ovlp) - np.dot(bmo_occ, bmo_occ.T)

            _, v = eigh(dm_vir, b=ovlp, type=2)
            bmo_vir = v[:, -nvirt:]

            if not bmo_occ.shape[-1] == nocc:
                raise RuntimeError()
            if not bmo_vir.shape[-1] == nvirt:
                raise RuntimeError()

            bmo = np.hstack((bmo_occ, bmo_vir))

    else:
        raise ValueError(f"Unrecognised rotation method: {method}")

    # Verify orthonormality
    if is_uhf:
        for spin_idx in range(2):
            if ovlp is None:
                if not np.allclose(np.dot(bmo[spin_idx].T, bmo[spin_idx]), np.eye(norb)):
                    raise RuntimeError(
                        f"Brueckner molecular orbitals (spin {spin_idx}) no longer orthonormal!"
                    )
            else:
                if not np.allclose(
                    np.linalg.multi_dot((bmo[spin_idx].T, ovlp, bmo[spin_idx])), np.eye(norb)
                ):
                    raise RuntimeError(
                        f"Brueckner molecular orbitals (spin {spin_idx}) not orthonormal w.r.t overlap!"
                    )
    else:
        if ovlp is None:
            if not np.allclose(np.dot(bmo.T, bmo), np.eye(norb)):
                raise RuntimeError("Brueckner molecular orbitals no longer orthonormal!")
        else:
            if not np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(norb)):
                raise RuntimeError("Brueckner molecular orbitals not orthonormal w.r.t overlap!")

    return bmo_occ, bmo_vir


def canonicalise_bmo(
    h1e: NDArray[np.float64], bmo_occ: NDArray[np.float64], bmo_vir: NDArray[np.float64]
):
    _, r = np.linalg.eigh(np.linalg.multi_dot((bmo_occ.T, h1e, bmo_occ)))
    bmo_occ = np.dot(bmo_occ, r)
    _, r = np.linalg.eigh(np.linalg.multi_dot((bmo_vir.T, h1e, bmo_vir)))
    bmo_vir = np.dot(bmo_vir, r)
    return bmo_occ, bmo_vir


def rotate_mf(
    mf: Union[scf.hf.RHF, scf.uhf.UHF],
    t1: Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]],
    canonicalise: bool = True,
    damping: float = 0.0,
    diis: Optional[lib.diis.DIIS] = None,
    method: Literal["expn", "taylor"] = "taylor",
):
    mf = copy(mf)
    norb = mf.mol.nao

    mo_coeff = mf.mo_coeff

    if isinstance(mf, scf.hf.RHF):
        nocc, _ = mf.mol.nelec
        nvirt = norb - nocc

        if not t1.shape == (nocc, nvirt):
            raise ValueError("Incorrect shape for t1 amplitudes.")

    elif isinstance(mf, scf.uhf.UHF):
        mo_a, mo_b = mf.mo_coeff
        t1_a, t1_b = t1

        nocc_a, nocc_b = mf.mol.nelec
        nvirt_a = norb - nocc_a
        nvirt_b = norb - nocc_b

        if not t1_a.shape == (nocc_a, nvirt_a):
            raise ValueError("Incorrect shape for alpha t1 amplitudes.")

        if not t1_b.shape == (nocc_b, nvirt_b):
            raise ValueError("Incorrect shape for alpha t1 amplitudes.")

    else:
        raise ValueError()

    ovlp = mf.get_ovlp()
    if ovlp is not None and np.allclose(ovlp, np.eye(ovlp.shape[-1])):
        ovlp = None

    bmo_occ, bmo_vir = rotate_mo_coeffs(mo_coeff, t1, ovlp, damping, diis, method)

    if isinstance(mf, scf.hf.RHF):
        if canonicalise:
            fock = mf.get_fock()
            bmo_occ, bmo_vir = canonicalise_bmo(fock, bmo_occ, bmo_vir)

        bmo = np.hstack((bmo_occ, bmo_vir))
        if ovlp is None and not np.allclose(np.dot(bmo.T, bmo), np.eye(norb)):
            raise RuntimeError("Brueckner molecular orbitals no longer orthonormal!")
        else:
            assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(norb))

    else:
        bmo_occ_a, bmo_occ_b = bmo_occ
        bmo_vir_a, bmo_vir_b = bmo_vir

        if canonicalise:
            fock_a, fock_b = mf.get_fock()
            bmo_occ_a, bmo_vir_a = canonicalise_bmo(fock_a, bmo_occ_a, bmo_vir_a)
            bmo_occ_b, bmo_vir_b = canonicalise_bmo(fock_b, bmo_occ_b, bmo_vir_b)

        bmo = np.array([np.hstack((bmo_occ_a, bmo_vir_a)), np.hstack((bmo_occ_b, bmo_vir_b))])

    mf.mo_coeff = bmo
    mf.e_tot = mf.energy_tot()
    return mf


def brueckner_cycle(
    mf: Union[scf.hf.RHF, scf.uhf.UHF],
    estimator: AbstractEstimator,
    canonicalise: bool = True,
    damping: float = 0.0,
    *,
    use_diis: bool = True,
    max_iter: int = 10,
    callback_fn: Optional[Callable[[np.float64, np.float64, np.float64], bool]] = None,
    method: Literal["expn", "taylor"] = "taylor",
    verbose: int = 0
):
    """Perform Brueckner orbital optimization to minimize single excitation amplitudes.

    Args:
        mf: Mean-field object (RHF or UHF)
        estimator: Estimator instance that implements AbstractEstimator interface
        canonicalise: Whether to canonicalize orbitals each iteration
        damping: Damping factor for orbital rotation (0 = no damping, 1 = full damping)
        max_iter: Maximum number of iterations
        callback_fn: Optional callback function(E, c0, norm) -> bool for convergence check
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
    """

    if verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:  # verbose >= 2
        logger.setLevel(logging.DEBUG)

    converged = False

    diis = lib.diis.DIIS() if use_diis else None

    logger.info("Starting Brueckner orbital optimization")
    logger.info(f"  max_iter={max_iter}, damping={damping}")
    logger.info("")
    logger.info(f"{'Iter':<6} {'Energy':<18} {'c0':<16} {'||c1||':<14}")
    logger.info("-" * 70)

    for iteration in range(max_iter):
        estimator.update_reference(mf)
        E, c0, c1, _ = estimator.run(calc_c1=True)

        if isinstance(c1, tuple):
            norm = np.sqrt(np.linalg.norm(c1[0]) ** 2 + np.linalg.norm(c1[1]) ** 2)
        else:
            norm = np.linalg.norm(c1)

        t1 = c1

        logger.info(f"{iteration + 1:<6} {E:<18.10f} {np.abs(c0):<16.10f} {norm:<14.6e}")

        if not isinstance(c1, tuple):
            logger.debug(f"  c1 max: {np.max(np.abs(c1)):.6e}")
            logger.debug(f"  t1 max: {np.max(np.abs(t1)):.6e}")
        else:
            logger.debug(
                f"  c1_a max: {np.max(np.abs(c1[0])):.6e}, c1_b max: {np.max(np.abs(c1[1])):.6e}"
            )
            logger.debug(
                f"  t1_a max: {np.max(np.abs(t1[0])):.6e}, t1_b max: {np.max(np.abs(t1[1])):.6e}"
            )

        if callback_fn:
            converged = callback_fn(E, c0, norm)

        if converged:
            logger.info("-" * 70)
            logger.info(f"Converged after {iteration + 1} iterations")
            break

        mf = rotate_mf(mf, t1, canonicalise, damping, diis=diis, method=method)

    if not converged:
        logger.info("-" * 70)
        logger.warning(f"Did not converge after {max_iter} iterations")


if __name__ == "__main__":
    from shades.solvers import FCISolver
    from shades.estimators import TrivialEstimator
    from shades.utils import make_hydrogen_chain
    from pyscf import gto, scf

    # Example 1: RHF Brueckner cycle with TrivialEstimator
    print("=" * 60)
    print("Example 1: RHF Brueckner Cycle with TrivialEstimator")
    print("=" * 60)

    atom = make_hydrogen_chain(4, bond_length=0.73)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.run()


    O_target = 0.75
    lam = np.sqrt(1.0 / O_target - 1.0)

    nocc, _ = mf.mol.nelec  # number of occupied spatial orbitals
    norb = mf.mo_coeff.shape[1]
    nvirt = norb - nocc

    t1 = np.zeros((nocc, nvirt))
    t1[nocc - 1, 0] = lam

    mf_rot = rotate_mf(mf, t1, canonicalise=True, damping=0.0)

    solver = FCISolver(mf_rot)
    estimator = TrivialEstimator(mf_rot, solver, verbose=1)

    # # Define convergence callback
    def converged(E, c0, norm):
        print(estimator.E_exact)
        return norm < 1e-6

    brueckner_cycle(
        mf_rot,
        estimator,
        max_iter=10,
        canonicalise=True,
        use_diis=False,
        damping=0.8,
        callback_fn=converged,
        verbose=2,
        method="taylor"
    )

    # print("\n" + "=" * 60)
    # print("Example 2: UHF Brueckner Cycle")
    # print("=" * 60)

    # atom = make_hydrogen_chain(5, bond_length=1.5)
    # mol_uhf = gto.Mole()
    # mol_uhf.build(atom=atom, basis='sto-3g', spin=1, verbose=0)
    # mf_uhf = scf.UHF(mol_uhf)
    # mf_uhf.run()

    # solver_uhf = FCISolver(mf_uhf)
    # estimator_uhf = TrivialEstimator(mf_uhf, solver_uhf, verbose=1)

    # brueckner_cycle(mf_uhf, estimator_uhf, max_iter=50, damping=0.7, use_diis=False, callback_fn=converged, verbose=2)

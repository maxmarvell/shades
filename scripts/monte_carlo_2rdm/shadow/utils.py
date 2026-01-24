import numpy as np

def spinorb_to_spatial_chem(
    rdm2_so: np.ndarray,
    norb: int
) -> np.ndarray:
    
    rdm2_aa = rdm2_so[:norb, :norb, :norb, :norb]
    rdm2_ab = rdm2_so[:norb, norb:, :norb, norb:]
    rdm2_bb = rdm2_so[norb:, norb:, norb:, norb:]

    dm2aa = rdm2_aa.transpose(0, 2, 1, 3)  # (p,q,r,s) -> (p,r,q,s)
    dm2ab = rdm2_ab.transpose(0, 2, 1, 3)
    dm2bb = rdm2_bb.transpose(0, 2, 1, 3)
    return dm2aa + dm2bb + dm2ab + dm2ab.transpose(1, 0, 3, 2)


def doubles_energy(
    rdm2: np.ndarray,
    mf
) -> np.ndarray:
    
    from pyscf import ao2mo

    norb = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mf.mol, mf.mo_coeff)
    eri = ao2mo.restore(1, eri, norb) 

    return 0.5 * np.einsum("ijkl,ijkl->", eri, rdm2)


def total_energy_from_rdm12(dm1, dm2, mf):
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    # eri in MO basis (chemist)
    from pyscf import ao2mo
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), h1.shape[0])
    e1 = np.einsum("pq,pq->", h1, dm1)
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eri, dm2)
    return e1 + e2 + mf.mol.energy_nuc()

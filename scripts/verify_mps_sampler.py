"""Verify that MPSSampler reproduces the FCI wavefunction distribution.

Compares the MPS (DMRG) determinant probabilities against the exact FCI
statevector probabilities for hydrogen chains of increasing size.
"""

import numpy as np
import logging
from pyscf import gto, scf

from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain
from shades.monte_carlo import MPSSampler

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

N_HYDROGEN = [2, 4, 6, 8, 10, 12]
BOND_LENGTH = 1.5
BASIS_SET = "sto-3g"


def det_to_int(det):
    """Convert block2 determinant array to spin-orbital bitstring integer.

    block2 with SZ symmetry stores determinants as arrays of length norb
    where each element encodes occupancy: 0=empty, 1=alpha, 2=beta, 3=alpha+beta.
    """
    norb = det.shape[0]
    alpha = det & 1
    beta = (det >> 1) & 1
    res = sum(1 << i for i in range(norb) if alpha[i] == 1)
    res += sum(1 << i for i in range(norb, 2 * norb) if beta[i - norb] == 1)
    return res


def main():
    print("=" * 80)
    print("MPS Sampler Distribution Verification")
    print("=" * 80)

    for n_h in N_HYDROGEN:
        print(f"\n{'='*80}")
        print(f"H{n_h} chain (r = {BOND_LENGTH} A, {BASIS_SET})")
        print(f"{'='*80}")

        hstring = make_hydrogen_chain(n_h, BOND_LENGTH)
        mol = gto.Mole()
        mol.build(atom=hstring, basis=BASIS_SET, verbose=0)

        mf = scf.RHF(mol)
        mf.run()

        norb = mf.mo_coeff.shape[1]
        n_qubits = 2 * norb
        print(f"norb = {norb}, n_qubits = {n_qubits}")

        # FCI reference
        fci_solver = FCISolver(mf)
        fci_solver.solve()
        fci_vec = fci_solver.state.data
        fci_probs = np.abs(fci_vec) ** 2

        # Build FCI probability dict (only nonzero entries)
        fci_dict = {}
        for idx in range(len(fci_probs)):
            if fci_probs[idx] > 1e-16:
                fci_dict[idx] = fci_probs[idx]

        # MPS sampler
        mps = MPSSampler(mf)

        # Build MPS probability dict
        mps_dets, mps_probs = mps.get_distribution()
        mps_dict = {}
        for det, prob in zip(mps_dets, mps_probs):
            key = det_to_int(det)
            mps_dict[key] = mps_dict.get(key, 0.0) + prob

        # Compare
        all_keys = set(fci_dict.keys()) | set(mps_dict.keys())
        fci_only = set(fci_dict.keys()) - set(mps_dict.keys())
        mps_only = set(mps_dict.keys()) - set(fci_dict.keys())
        common = set(fci_dict.keys()) & set(mps_dict.keys())

        print(f"\nDeterminants:  FCI = {len(fci_dict)},  MPS = {len(mps_dict)},  common = {len(common)}")
        print(f"Only in FCI: {len(fci_only)},  Only in MPS: {len(mps_only)}")

        if fci_only:
            missing_mass = sum(fci_dict[k] for k in fci_only)
            print(f"Mass of FCI-only dets: {missing_mass:.6e}")

        if mps_only:
            extra_mass = sum(mps_dict[k] for k in mps_only)
            print(f"Mass of MPS-only dets: {extra_mass:.6e}")

        # Per-determinant comparison for common dets
        fci_arr = np.array([fci_dict.get(k, 0.0) for k in sorted(all_keys)])
        mps_arr = np.array([mps_dict.get(k, 0.0) for k in sorted(all_keys)])

        # Total variation distance: 0.5 * sum |p - q|
        tvd = 0.5 * np.sum(np.abs(fci_arr - mps_arr))

        # KL divergence: sum p * log(p/q) for p > 0, q > 0
        mask = (fci_arr > 1e-16) & (mps_arr > 1e-16)
        kl = np.sum(fci_arr[mask] * np.log(fci_arr[mask] / mps_arr[mask]))

        # Fidelity: (sum sqrt(p*q))^2
        fidelity = np.sum(np.sqrt(fci_arr * mps_arr)) ** 2

        print(f"\nDistribution metrics:")
        print(f"  Total variation distance: {tvd:.6e}")
        print(f"  KL divergence (FCI||MPS): {kl:.6e}")
        print(f"  Fidelity:                 {fidelity:.10f}")

        # Top determinants comparison
        top_fci = sorted(fci_dict.items(), key=lambda x: -x[1])[:10]
        print(f"\nTop 10 determinants by FCI probability:")
        print(f"  {'Bitstring':<{n_qubits+2}} {'p(FCI)':>12} {'p(MPS)':>12} {'Ratio':>10} {'Abs diff':>12}")
        print(f"  {'-'*(n_qubits+2+12+12+10+12+4)}")
        for key, p_fci in top_fci:
            p_mps = mps_dict.get(key, 0.0)
            bs = format(key, f"0{n_qubits}b")[::-1]
            ratio = p_mps / p_fci if p_fci > 1e-16 else float('inf')
            print(f"  {bs:<{n_qubits+2}} {p_fci:>12.6e} {p_mps:>12.6e} {ratio:>10.4f} {p_mps - p_fci:>+12.4e}")

        # Check for duplicate determinant mappings in MPS
        int_keys = [det_to_int(d) for d in mps_dets]
        n_unique = len(set(int_keys))
        if n_unique != len(int_keys):
            print(f"\n  WARNING: {len(int_keys) - n_unique} duplicate determinant mappings in MPS!")


if __name__ == "__main__":
    main()
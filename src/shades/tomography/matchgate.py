from dataclasses import dataclass
from itertools import product
from typing import List, Tuple, Sequence
from math import comb

import numpy as np
import qulacs
import stim
from qiskit.quantum_info import Statevector

from shades.utils import tableau_to_qulacs_circuit

def _pfaffian(A: np.ndarray) -> complex:
    A = np.array(A, dtype=complex)
    n = A.shape[0]
    if n % 2 == 1:
        return 0.0
    if n == 0:
        return 1.0

    pfaffian = 1.0 + 0j
    for k in range(0, n - 1, 2):
        pivot_idx = k + 1 + np.argmax(np.abs(A[k, k + 1:]))
        if A[k, pivot_idx] == 0.0:
            return 0.0

        if pivot_idx != k + 1:
            A[[k + 1, pivot_idx], :] = A[[pivot_idx, k + 1], :]
            A[:, [k + 1, pivot_idx]] = A[:, [pivot_idx, k + 1]]
            pfaffian *= -1

        pfaffian *= A[k, k + 1]

        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, A[k + 1, k + 2:])
            A[k + 2:, k + 2:] -= np.outer(A[k + 1, k + 2:], tau)

    return pfaffian

@dataclass
class MatchgateSnapshot:
    """A single matchgate shadow snapshot storing the compact signed
    permutation representation and measurement outcome."""
    permutation: np.ndarray   # shape (2n,), dtype int
    signs: np.ndarray         # shape (2n,), values ±1
    b: int                    # n measurement bits
    n_qubits: int

    def __post_init__(self):
        dim = 2 * self.n_qubits
        self._inv_perm = np.empty(dim, dtype=int)
        self._inv_perm[self.permutation] = np.arange(dim)

    def _rotated_covariance_entry(self, mu: int, nu: int) -> float:
        """Compute (Q^T C_b Q)_{μ,ν} via signed permutation lookup. O(1)."""

        p = self._inv_perm[mu]
        q = self._inv_perm[nu]

        if p // 2 != q // 2:
            return 0.0
        if p == q:
            return 0.0

        k = p // 2
        bit_k = (self.b >> k) & 1
        sign = 1 - 2 * bit_k  # (-1)^{b_k}

        if p < q:
            c = float(sign)
        else:
            c = float(-sign)

        return (self.signs[p] * self.signs[q] * c)

    def _rotated_covariance_submatrix(self, S: Sequence[int]) -> np.ndarray:
        """Build the |S|×|S| submatrix (Q^T C_b Q)|_S."""
        k = len(S)
        M = np.zeros((k, k), dtype=complex)
        for a in range(k):
            for b in range(a + 1, k):
                val = self._rotated_covariance_entry(S[a], S[b])
                M[a, b] = val
                M[b, a] = -val
        return M

    def estimate_majorana(self, S: Sequence[int]) -> complex:
        """Unbiased estimator for tr(γ_S ρ) from this single snapshot."""
        S = list(S)
        k = len(S)
        n = self.n_qubits
        assert k % 2 == 0, "|S| must be even"
        assert k <= 2 * n, f"|S|={k} exceeds 2n={2*n}"

        # Inverse channel coefficient: C(2n, k) / C(n, k/2)
        coeff = comb(2 * n, k) / comb(n, k // 2)

        # Build i · (Q^T C_b Q)|_S and take its Pfaffian
        C_sub = self._rotated_covariance_submatrix(S)
        return coeff * _pfaffian(1j * C_sub)

    

@dataclass
class MatchgateChannel:
    n_qubits: int

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        dim = 2 * self.n_qubits
        permutation = np.random.permutation(dim)
        signs = np.random.choice([-1, 1], size=dim)
        visited = np.zeros(dim, dtype=bool)
        cycles = 0
        for i in range(dim):
            if not visited[i]:
                cycles += 1
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = permutation[j]
        parity = 1 if ((dim - cycles) % 2 == 0) else -1  # sgn(perm)

        det_sign = parity * int(np.prod(signs))
        if det_sign == -1:
            signs[0] *= -1  # flip any one sign to fix determinant

        return permutation, signs

    def __call__(self, state: qulacs.QuantumState) -> MatchgateSnapshot:
        perm, signs = self.sample()
        tab = self._signed_perm_to_tableau(perm, signs)
        circuit = tableau_to_qulacs_circuit(tab, self.n_qubits)
        circuit.update_quantum_state(state)
        b = state.sampling(1)[0]
        return MatchgateSnapshot(
            permutation=perm, signs=signs, b=b, n_qubits=self.n_qubits,
        )

    def _majorana_pauli_strings(self) -> List[stim.PauliString]:
        n = self.n_qubits
        gammas = []
        for q in range(n):
            x_str = ['I'] * n
            for j in range(q):
                x_str[j] = 'Z'
            x_str[q] = 'X'
            gammas.append(stim.PauliString(''.join(x_str)))

            y_str = ['I'] * n
            for j in range(q):
                y_str[j] = 'Z'
            y_str[q] = 'Y'
            gammas.append(stim.PauliString(''.join(y_str)))
        return gammas

    def _signed_perm_to_tableau(
        self, permutation: np.ndarray, signs: np.ndarray
    ) -> stim.Tableau:
        dim = 2 * self.n_qubits
        gammas = self._majorana_pauli_strings()

        inv_perm = np.empty(dim, dtype=int)
        inv_perm[permutation] = np.arange(dim)

        transformed = []
        for j in range(dim):
            k = inv_perm[j]
            sign = int(signs[k])
            transformed.append(gammas[k] if sign == 1 else gammas[k] * -1)

        z_outputs = []
        for q in range(self.n_qubits):
            z_q = transformed[2 * q] * transformed[2 * q + 1] * (-1j)
            z_outputs.append(z_q)

        x_outputs = []
        for q in range(self.n_qubits):
            result = transformed[2 * q]
            for j in range(q - 1, -1, -1):
                result = z_outputs[j] * result
            x_outputs.append(result)

        return stim.Tableau.from_conjugated_generators(xs=x_outputs, zs=z_outputs)

class MatchgateShadow:

    snapshots: list[MatchgateSnapshot] | None

    def __init__(self, state: Statevector):
        self.state = state
        self.n_qubits = state.num_qubits
        self.channel = MatchgateChannel(self.n_qubits)
        self.snapshots = None

    def run(self, n_samples: int):
        qulacs_template = qulacs.QuantumState(self.n_qubits)
        qulacs_template.load(self.state.data)

        self.snapshots = []
        for _ in range(n_samples):
            qulacs_state = qulacs_template.copy()
            snapshot = self.channel(qulacs_state)
            self.snapshots.append(snapshot)
        return self.snapshots
    
    def estimate_majorana(self, S: Sequence[int]) -> complex:
        return np.mean([p.estimate_majorana(S) for p in self.snapshots])

    def _ensure_arrays(self):
        """Stack snapshot data into contiguous numpy arrays for vectorized access."""
        if hasattr(self, "_arr_inv_perms"):
            return
        n_samples = len(self.snapshots)
        dim = 2 * self.n_qubits
        self._arr_inv_perms = np.empty((n_samples, dim), dtype=np.intp)
        self._arr_signs = np.empty((n_samples, dim), dtype=np.float64)
        self._arr_bits = np.empty(n_samples, dtype=np.int64)
        for i, snap in enumerate(self.snapshots):
            self._arr_inv_perms[i] = snap._inv_perm
            self._arr_signs[i] = snap.signs
            self._arr_bits[i] = snap.b


def _vectorized_covariance_entry(
    inv_perms: np.ndarray,
    signs: np.ndarray,
    bits: np.ndarray,
    mu: int,
    nu: int,
) -> np.ndarray:
    """Compute (Q^T C_b Q)_{mu,nu} for all snapshots simultaneously.

    Returns array of shape (n_samples,).
    """
    p = inv_perms[:, mu]
    q = inv_perms[:, nu]
    same_pair = (p // 2 == q // 2) & (p != q)
    k = p // 2
    bit_k = (bits >> k) & 1
    sign = 1 - 2 * bit_k
    c = np.where(p < q, sign, -sign)
    return signs[np.arange(len(signs)), p] * signs[np.arange(len(signs)), q] * c * same_pair


def _batch_estimate_majoranas(
    shadow: MatchgateShadow, index_sets: set[tuple[int, ...]]
) -> dict[tuple[int, ...], complex]:
    """Estimate all requested Majorana expectations in one vectorized pass.

    Each key in index_sets is a sorted tuple of Majorana indices with even length.
    Returns a dict mapping each tuple to its estimated expectation value.
    """
    shadow._ensure_arrays()
    inv_perms = shadow._arr_inv_perms
    signs_arr = shadow._arr_signs
    bits = shadow._arr_bits
    n = shadow.n_qubits
    n_samples = len(shadow.snapshots)
    sample_idx = np.arange(n_samples)

    results = {}
    for S in index_sets:
        k = len(S)
        coeff = comb(2 * n, k) / comb(n, k // 2)

        if k == 2:
            a, b = S
            p = inv_perms[:, a]
            q = inv_perms[:, b]
            same_pair = (p // 2 == q // 2) & (p != q)
            kk = p // 2
            bit_k = (bits >> kk) & 1
            bsign = 1 - 2 * bit_k
            c = np.where(p < q, bsign, -bsign)
            entry = signs_arr[sample_idx, p] * signs_arr[sample_idx, q] * c * same_pair
            results[S] = coeff * np.mean(1j * entry)

        elif k == 4:
            a, b, c_, d = S
            pairs = [(a, b), (a, c_), (a, d), (b, c_), (b, d), (c_, d)]
            entries = np.empty((6, n_samples), dtype=np.float64)
            for idx, (mu, nu) in enumerate(pairs):
                p = inv_perms[:, mu]
                q = inv_perms[:, nu]
                same_pair = (p // 2 == q // 2) & (p != q)
                kk = p // 2
                bit_k = (bits >> kk) & 1
                bsign = 1 - 2 * bit_k
                c_val = np.where(p < q, bsign, -bsign)
                entries[idx] = signs_arr[sample_idx, p] * signs_arr[sample_idx, q] * c_val * same_pair

            # Closed-form Pfaffian for 4x4 antisymmetric (with 1j prefactor):
            # M_ij = 1j * entries[pair_index]
            # pf = M01*M23 - M02*M13 + M03*M12
            # With M_ij = 1j * e_ij: pf = (1j)^2 * (e01*e23 - e02*e13 + e03*e12) = -(...)
            m01, m02, m03, m12, m13, m23 = entries
            pf_per_sample = -(m01 * m23 - m02 * m13 + m03 * m12)
            results[S] = coeff * np.mean(pf_per_sample)

        else:
            results[S] = shadow.estimate_majorana(S)

    return results


def _creation_coeffs(j):
    """a†_j = (γ_{2j} - i γ_{2j+1}) / 2"""
    return [(2*j, 0.5), (2*j+1, -0.5j)]

def _annihilation_coeffs(j):
    """a_j = (γ_{2j} + i γ_{2j+1}) / 2"""
    return [(2*j, 0.5), (2*j+1, 0.5j)]

def _canonicalize_majorana_product(indices: list[int]) -> tuple[int, list[int]]:
    """Canonicalise γ_{i1} γ_{i2} ... γ_{ik} into sign * γ_S with S sorted.

    Handles anticommutation ({γ_μ, γ_ν} = 2δ_{μν}) by sorting and cancelling
    repeated pairs (γ_μ² = I).

    Returns (sign, sorted_indices) where sign is ±1.
    """
    indices = list(indices)
    n = len(indices)

    # Count inversions (each swap = sign flip due to anticommutation)
    sign = 1
    for i in range(n):
        for j in range(i + 1, n):
            if indices[i] > indices[j]:
                sign *= -1

    sorted_idx = sorted(indices)

    # Cancel adjacent duplicate pairs (γ_μ² = I)
    result = []
    i = 0
    while i < len(sorted_idx):
        if i + 1 < len(sorted_idx) and sorted_idx[i] == sorted_idx[i + 1]:
            i += 2
        else:
            result.append(sorted_idx[i])
            i += 1

    return sign, result


def estimate_one_rdm(shadow: MatchgateShadow) -> np.ndarray:
    """Estimate the full 1-RDM from a matchgate shadow using batched Majorana estimation.

    Returns D[p,q] = ⟨a†_p a_q⟩ in spin-orbital basis, shape (n, n).
    """
    n = shadow.n_qubits
    unique_sets: set[tuple[int, ...]] = set()

    # Collect all unique 2-Majorana index pairs needed
    for p in range(n):
        unique_sets.add((2 * p, 2 * p + 1))
        for q in range(p + 1, n):
            for a, b in [(2*p, 2*q), (2*p+1, 2*q+1), (2*p+1, 2*q), (2*p, 2*q+1)]:
                unique_sets.add((min(a, b), max(a, b)))

    majorana_values = _batch_estimate_majoranas(shadow, unique_sets)

    def _signed(a, b):
        if a < b:
            return majorana_values[(a, b)]
        else:
            return -majorana_values[(b, a)]

    rdm1 = np.zeros((n, n), dtype=complex)
    for p in range(n):
        rdm1[p, p] = 0.5 * (1.0 + 1j * majorana_values[(2 * p, 2 * p + 1)])
        for q in range(p + 1, n):
            c00 = _signed(2*p, 2*q)
            c11 = _signed(2*p+1, 2*q+1)
            c10 = _signed(2*p+1, 2*q)
            c01 = _signed(2*p, 2*q+1)
            val = 0.25 * ((c00 + c11) + 1j * (c10 - c01))
            rdm1[p, q] = val
            rdm1[q, p] = val.conjugate()

    return rdm1


def estimate_one_rdm_element(shadow: MatchgateShadow, p: int, q: int) -> complex:
    """Estimate ⟨a†_p a_q⟩ from a matchgate shadow (0-indexed orbitals, p <= q)."""
    if p == q:
        # n_p = a†_p a_p = (1 + i γ_{2p} γ_{2p+1}) / 2
        g = shadow.estimate_majorana([2 * p, 2 * p + 1])
        return 0.5 * (1.0 + 1j * g)

    def _signed_majorana(a, b):
        if a < b:
            return shadow.estimate_majorana([a, b])
        else:
            return -shadow.estimate_majorana([b, a])

    c00 = _signed_majorana(2*p,   2*q)
    c11 = _signed_majorana(2*p+1, 2*q+1)
    c10 = _signed_majorana(2*p+1, 2*q)
    c01 = _signed_majorana(2*p,   2*q+1)

    return 0.25 * ((c00 + c11) + 1j * (c10 - c01))


def estimate_two_rdm_element(
    shadow: MatchgateShadow, p: int, q: int, r: int, s: int
) -> complex:
    """Estimate ⟨a†_p a†_q a_r a_s⟩ from a matchgate shadow.

    Physicists' notation: p < q, r < s (canonical index order).
    Expands each ladder operator into Majoranas and sums over the
    2^4 = 16 terms, canonicalising each product before estimation.
    """
    ops = [
        _creation_coeffs(p), _creation_coeffs(q),
        _annihilation_coeffs(r), _annihilation_coeffs(s),
    ]

    result = 0.0 + 0j
    for mu_p, c_p in ops[0]:
        for mu_q, c_q in ops[1]:
            for mu_r, c_r in ops[2]:
                for mu_s, c_s in ops[3]:
                    coeff = c_p * c_q * c_r * c_s
                    sign, canonical = _canonicalize_majorana_product(
                        [mu_p, mu_q, mu_r, mu_s]
                    )
                    if len(canonical) == 0:
                        result += coeff * sign
                    elif len(canonical) % 2 != 0:
                        continue
                    else:
                        result += coeff * sign * shadow.estimate_majorana(canonical)
    return result


def _collect_rdm2_terms(
    element_indices: list[tuple[int, int, int, int]],
) -> tuple[
    dict[tuple[int, ...], complex],
    list[tuple[int, int, int, int]],
]:
    """Pre-compute all Majorana expansion terms for a set of 2-RDM elements.

    Returns:
        majorana_coeffs: dict mapping each unique canonical Majorana tuple to the
            accumulated complex coefficient (zero for identity contributions).
        identity_contrib: not returned separately; identity terms are accumulated
            into a per-element dict.

    Actually returns (unique_majorana_sets, element_terms) where:
        unique_majorana_sets: set of all unique sorted Majorana index tuples needed
        element_terms: list of (p,q,r,s, [(coeff, sign, canonical_tuple), ...]) for assembly
    """
    unique_sets: set[tuple[int, ...]] = set()
    element_terms = []

    for p, q, r, s in element_indices:
        ops = [
            _creation_coeffs(p), _creation_coeffs(q),
            _annihilation_coeffs(r), _annihilation_coeffs(s),
        ]
        terms = []
        identity_sum = 0.0 + 0j
        for (mu_p, c_p), (mu_q, c_q), (mu_r, c_r), (mu_s, c_s) in product(*ops):
            coeff = c_p * c_q * c_r * c_s
            sign, canonical = _canonicalize_majorana_product([mu_p, mu_q, mu_r, mu_s])
            if len(canonical) == 0:
                identity_sum += coeff * sign
            elif len(canonical) % 2 != 0:
                continue
            else:
                key = tuple(canonical)
                unique_sets.add(key)
                terms.append((coeff * sign, key))
        element_terms.append(((p, q, r, s), identity_sum, terms))

    return unique_sets, element_terms


def estimate_two_rdm(shadow: MatchgateShadow, *, symmetry: str = "RHF") -> np.ndarray:
    """Estimate the full two-particle reduced density matrix from a matchgate shadow.

    Returns Γ[p,q,r,s] = ⟨a†_p a†_q a_s a_r⟩

    Uses batched vectorized Majorana estimation for performance.
    """
    n = shadow.n_qubits
    rdm2 = np.zeros((n, n, n, n), dtype=complex)

    if symmetry != "RHF":
        element_indices = [
            (p, q, r, s)
            for p in range(n)
            for q in range(p + 1, n)
            for r in range(n)
            for s in range(r + 1, n)
        ]
    else:
        norb = n // 2
        element_indices = []
        for p in range(norb):
            for q in range(p + 1, norb):
                for r in range(norb):
                    for s in range(r + 1, norb):
                        element_indices.append((p, q, r, s))
        for p in range(norb):
            for q in range(norb, n):
                for r in range(norb):
                    for s in range(norb, n):
                        element_indices.append((p, q, r, s))

    unique_sets, element_terms = _collect_rdm2_terms(element_indices)
    majorana_values = _batch_estimate_majoranas(shadow, unique_sets)

    if symmetry != "RHF":
        for (p, q, r, s), identity_sum, terms in element_terms:
            val = identity_sum
            for coeff_sign, key in terms:
                val += coeff_sign * majorana_values[key]
            rdm2[p, q, s, r] = val
            rdm2[q, p, s, r] = -val
            rdm2[p, q, r, s] = -val
            rdm2[q, p, r, s] = val
    else:
        norb = n // 2
        for (p, q, r, s), identity_sum, terms in element_terms:
            val = identity_sum
            for coeff_sign, key in terms:
                val += coeff_sign * majorana_values[key]
            rdm2[p, q, s, r] = val
            rdm2[q, p, s, r] = -val
            rdm2[p, q, r, s] = -val
            rdm2[q, p, r, s] = val
            if p < norb and q < norb and r < norb and s < norb:
                po, qo, ro, so = p + norb, q + norb, r + norb, s + norb
                rdm2[po, qo, so, ro] = val
                rdm2[qo, po, so, ro] = -val
                rdm2[po, qo, ro, so] = -val
                rdm2[qo, po, ro, so] = val

    return rdm2
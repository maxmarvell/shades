

import pytest
import logging
import numpy as np

from pyscf import gto, scf, fci

from shades.tomography import ShadowProtocol
from shades.utils import make_hydrogen_chain
from shades.solvers.fci import civec_to_statevector

from shades.utils import Bitstring

logging.basicConfig(level=logging.INFO)


class TestUnbiasedEstimator:
    """Test that the shadow protocol is an unbiased estimator of c_n"""

    def _pick_test_dets(self, state, n_pick=8, min_prob=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        probs = np.abs(state) ** 2
        idx = np.where(probs > min_prob)[0]
        if len(idx) == 0:
            raise RuntimeError("No basis states above min_prob; lower threshold.")
        if len(idx) > n_pick:
            idx = rng.choice(idx, size=n_pick, replace=False)
        return [int(i) for i in idx]

    @pytest.mark.parametrize("n_hydrogen", [2, 4])
    def test_hydrogen_chain(self, n_hydrogen: int):

        hstring = make_hydrogen_chain(n_hydrogen, 1.5)
        mol = gto.Mole()
        mol.build(atom=hstring, basis="sto-3g", verbose=0)

        mf = scf.RHF(mol)
        mf.run()

        _, civec = fci.FCI(mf).kernel()

        norb = mf.mo_coeff.shape[1]
        nelec = mf.mol.nelec    
        psi = civec_to_statevector(civec, norb, nelec)

        tests = self._pick_test_dets(psi.data, min_prob=0)

        n_reps = 40
        n_samples = 10000
        n_estimators = 20
        z_thresh = 3.5

        n_qubits = 2*norb

        for n in tests:

            a = Bitstring.from_int(n, n_qubits)
            bitstr = format(n, f"0{n_qubits}b")[::-1]
            target = float(np.real(psi[n]))

            logging.info("=" * 60)
            logging.info(f"Testing estimator bias for |{bitstr}>")
            logging.info(f"Target amplitude (Re) = {target:.8e}")

            estimates = []
            for i in range(n_reps):
                protocol = ShadowProtocol(state=psi)
                protocol.collect_samples_for_overlaps(n_samples, n_estimators=n_estimators)
                est = float(protocol.estimate_overlap(a, n_jobs=8).real)
                if i < 5:
                    logging.info(f"    repetition {i}: {est}")
                elif i == 5:
                    logging.info(f"    repetition {i}: {est}...")
                estimates.append(est)

            estimates = np.array(estimates, dtype=float)
            mean = float(estimates.mean())
            std = float(estimates.std(ddof=1))
            se = std / np.sqrt(n_reps)
            bias = mean - target

            logging.info("-" * 60)
            logging.info(f"Mean estimate      = {mean:.8e}")
            logging.info(f"Std dev            = {std:.8e}")
            logging.info(f"Standard error     = {se:.8e}")
            logging.info(f"Bias (mean-target) = {bias:.8e}")

            # If variance is numerically ~0 (unlikely), fall back to absolute tolerance
            if se < 1e-12:
                assert abs(mean - target) < 1e-8
            else:
                z = abs(mean - target) / se
                assert z < z_thresh, (
                    f"Overlap appears biased for n={n} (|{format(n, f'0{n_qubits}b')[::-1]}|): "
                    f"target(Re)={target:.6e}, mean={mean:.6e}, SE={se:.3e}, z={z:.2f}"
                )


    @pytest.mark.parametrize("n_hydrogen", [2, 4])
    def test_shadow_protocol_estimates_are_uncorrelated(self, n_hydrogen: int):

        # --- build FCI statevector ---
        hstring = make_hydrogen_chain(n_hydrogen, 1.5)
        mol = gto.Mole()
        mol.build(atom=hstring, basis="sto-3g", verbose=0)

        mf = scf.RHF(mol).run()

        # PySCF FCI
        e, civec = fci.FCI(mf).kernel()

        norb = mf.mo_coeff.shape[1]
        nelec = mf.mol.nelec
        n_qubits = 2 * norb

        psi = civec_to_statevector(civec, norb, nelec)  # numpy array or Statevector-like

        # choose some determinants to test
        tests = self._pick_test_dets(np.asarray(psi), n_pick=6, min_prob=1e-7, seed=123)
        assert len(tests) >= 2

        # choose pairs (n,m)
        pairs = list(zip(tests[:-1], tests[1:]))

        # --- experiment parameters ---
        R = 200                # repetitions for covariance estimate
        n_samples = 5000       # per protocol
        n_estimators = 20
        n_jobs = 8

        # correlation threshold: use t-test for corr=0
        # t = r * sqrt((R-2)/(1-r^2)) ~ StudentT(df=R-2)
        # For R=200, |t|<3 is already very safe (~0.003 level)
        t_thresh = 3.0

        for n, m in pairs:
            a_n = Bitstring.from_int(n, n_qubits)
            a_m = Bitstring.from_int(m, n_qubits)

            cn = np.empty(R, dtype=float)
            cm = np.empty(R, dtype=float)

            bit_n = format(n, f"0{n_qubits}b")[::-1]
            bit_m = format(m, f"0{n_qubits}b")[::-1]

            logging.info("=" * 60)
            logging.info(f"Testing independence for pair |{bit_n}>, |{bit_m}>")
            logging.info(f"R={R}, n_samples={n_samples}, n_estimators={n_estimators}")

            for r in range(R):
                # independent protocols (fresh sampling)
                protA = ShadowProtocol(state=psi)
                protA.collect_samples_for_overlaps(n_samples, n_estimators=n_estimators)

                protB = ShadowProtocol(state=psi)
                protB.collect_samples_for_overlaps(n_samples, n_estimators=n_estimators)

                cn[r] = float(protA.estimate_overlap(a_n, n_jobs=n_jobs).real)
                cm[r] = float(protB.estimate_overlap(a_m, n_jobs=n_jobs).real)

                if r < 3:
                    logging.info(
                        f"  rep {r:3d}: cn={cn[r]: .6e}, cm={cm[r]: .6e}"
                    )
                elif r % 20 == 0:
                    logging.info(
                        f"  progress: rep {r:3d}/{R}, "
                        f"mean(cn)={cn[:r+1].mean(): .3e}, "
                        f"mean(cm)={cm[:r+1].mean(): .3e}"
                    )

            # empirical covariance/correlation
            cov = float(np.cov(cn, cm, ddof=1)[0, 1])
            corr = float(np.corrcoef(cn, cm)[0, 1])

            # significance of correlation
            denom = max(1e-15, 1.0 - corr * corr)
            t_stat = corr * np.sqrt((R - 2) / denom)

            logging.info(f"mean(cn)={cn.mean(): .6e}, std(cn)={cn.std(ddof=1): .6e}")
            logging.info(f"mean(cm)={cm.mean(): .6e}, std(cm)={cm.std(ddof=1): .6e}")
            logging.info(f"cov={cov: .6e}, corr={corr: .6e}, t={t_stat: .3f}")

            assert abs(t_stat) < t_thresh, (
                f"Overlap estimates appear correlated for pair |{bit_n}>,|{bit_m}>: "
                f"corr={corr:.3e}, t={t_stat:.2f} (R={R}). "
                "This suggests shared randomness / RNG reuse between protocols."
            )
import logging
import os
import json
from datetime import datetime
import numpy as np
from scipy.linalg import eigh
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

from shades._utils.models import ising_chain
from shades.utils import pauli_terms_to_matrix
from shades.subspace_expansion.bell_difference_sampler import stabilizer_state_approximation
from shades.stabilizer_subspace import stabilizer_from_stim_tableau, StabilizerSubspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

logger = logging.getLogger(__name__)

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("stim").setLevel(logging.WARNING)
logging.getLogger("qulacs").setLevel(logging.WARNING)

N_QUBITS = 6
TRANSVERSE_FIELD_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
M_CLIQUE = 40
M_SHADOW = 500
TOP_N_STATES = 6
N_TRIALS = 10


def run_bell_difference_expansion(ground_state: Statevector, exact_energy: float,
                                   tfim, m_clique: int, m_shadow: int, top_n: int):
    """Run Bell difference sampling and subspace expansion for given ground state."""

    ovlps, stabilizers = stabilizer_state_approximation(
        ground_state, m_clique=m_clique, m_shadow=m_shadow
    )

    if len(stabilizers) == 0:
        return None, 0

    ovlps = np.array(ovlps)
    sorted_indices = np.argsort(ovlps)[::-1]

    n_select = min(top_n, len(stabilizers))
    top_indices = sorted_indices[:n_select]
    top_tableaus = [stabilizers[i] for i in top_indices]

    stabilizer_states = []
    for tab in top_tableaus:
        stab_state = stabilizer_from_stim_tableau(tab)
        if stab_state not in stabilizer_states:
            stabilizer_states.append(stab_state)

    if len(stabilizer_states) < 2:
        return None, len(stabilizer_states)

    subspace = StabilizerSubspace(stabilizer_states, tfim)

    try:
        energy, _ = subspace.optimize_coefficients(reg=1e-8)
    except np.linalg.LinAlgError:
        return None, len(stabilizer_states)

    return energy, len(stabilizer_states)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/bell_difference_sampling/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")

    n_h = len(TRANSVERSE_FIELD_STRENGTHS)
    energies = np.full((n_h, N_TRIALS), np.nan)
    exact_energies = np.zeros(n_h)
    n_states_used = np.zeros((n_h, N_TRIALS), dtype=int)

    for j, h in enumerate(TRANSVERSE_FIELD_STRENGTHS):
        logger.info(f"\n{'='*50}")
        logger.info(f"h = {h} ({j+1}/{n_h})")
        logger.info(f"{'='*50}")

        tfim = ising_chain(n_qubits=N_QUBITS, h=h, J=1.0)
        matrix = pauli_terms_to_matrix(tfim)
        eigs, evs = eigh(matrix)

        exact_energies[j] = eigs[0]
        ground_state = Statevector(evs[:, 0])

        logger.info(f"Exact energy: {exact_energies[j]:.6f}")

        for trial in range(N_TRIALS):
            energy, n_states = run_bell_difference_expansion(
                ground_state=ground_state,
                exact_energy=exact_energies[j],
                tfim=tfim,
                m_clique=M_CLIQUE,
                m_shadow=M_SHADOW,
                top_n=TOP_N_STATES,
            )

            if energy is not None:
                energies[j, trial] = energy
                n_states_used[j, trial] = n_states
                rel_error = (energy - exact_energies[j]) / abs(exact_energies[j])
                logger.info(f"  Trial {trial+1}/{N_TRIALS}: E={energy:.6f}, "
                           f"rel_error={rel_error:.2e}, n_states={n_states}")
            else:
                logger.warning(f"  Trial {trial+1}/{N_TRIALS}: FAILED (n_states={n_states})")

    rel_errors = (energies - exact_energies[:, None]) / np.abs(exact_energies[:, None])

    mean_energies = np.nanmean(energies, axis=1)
    std_energies = np.nanstd(energies, axis=1)
    mean_rel_errors = np.nanmean(rel_errors, axis=1)
    std_rel_errors = np.nanstd(rel_errors, axis=1)
    min_rel_errors = np.nanmin(rel_errors, axis=1)
    max_rel_errors = np.nanmax(rel_errors, axis=1)
    success_rate = np.sum(~np.isnan(energies), axis=1) / N_TRIALS

    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'h':>6} | {'Exact E':>10} | {'Mean E':>10} | {'Std E':>10} | "
          f"{'Mean Rel Err':>12} | {'Success':>8}")
    print("-"*90)

    for j, h in enumerate(TRANSVERSE_FIELD_STRENGTHS):
        print(f"{h:>6.2f} | {exact_energies[j]:>10.4f} | {mean_energies[j]:>10.4f} | "
              f"{std_energies[j]:>10.4f} | {mean_rel_errors[j]:>12.2e} | "
              f"{success_rate[j]*100:>7.1f}%")

    np.savez(
        os.path.join(output_dir, "data.npz"),
        transverse_field_strengths=np.array(TRANSVERSE_FIELD_STRENGTHS),
        energies=energies,
        exact_energies=exact_energies,
        rel_errors=rel_errors,
        n_states_used=n_states_used,
        mean_energies=mean_energies,
        std_energies=std_energies,
        mean_rel_errors=mean_rel_errors,
        std_rel_errors=std_rel_errors,
        min_rel_errors=min_rel_errors,
        max_rel_errors=max_rel_errors,
        success_rate=success_rate,
    )
    logger.info(f"Data saved to {output_dir}/data.npz")

    metadata = {
        "timestamp": timestamp,
        "parameters": {
            "n_qubits": N_QUBITS,
            "n_trials": N_TRIALS,
            "m_clique": M_CLIQUE,
            "m_shadow": M_SHADOW,
            "top_n_states": TOP_N_STATES,
            "transverse_field_strengths": TRANSVERSE_FIELD_STRENGTHS,
        },
        "results_summary": {
            "mean_rel_errors": mean_rel_errors.tolist(),
            "std_rel_errors": std_rel_errors.tolist(),
            "success_rate": success_rate.tolist(),
        }
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {output_dir}/metadata.json")

    fig, ax = plt.subplots(figsize=(10, 6))

    valid_mask = ~np.isnan(mean_rel_errors)
    h_valid = np.array(TRANSVERSE_FIELD_STRENGTHS)[valid_mask]
    mean_valid = mean_rel_errors[valid_mask]
    min_valid = min_rel_errors[valid_mask]
    max_valid = max_rel_errors[valid_mask]

    ax.plot(h_valid, mean_valid, 'o-', markersize=8, linewidth=2, label='Mean', color='C0')
    ax.fill_between(h_valid, min_valid, max_valid,
                    alpha=0.3, color='C0', label='Min/Max')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Transverse field strength $h$', fontsize=14)
    ax.set_ylabel(r'Relative energy error $(E - E_{\mathrm{exact}}) / |E_{\mathrm{exact}}|$', fontsize=14)
    ax.set_title(f'Bell Difference Sampling: {N_QUBITS}-site TFIM ({N_TRIALS} trials)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_error.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "relative_error.svg"))
    logger.info(f"Plots saved to {output_dir}/relative_error.png and .svg")
    plt.show()


if __name__ == "__main__":
    main()

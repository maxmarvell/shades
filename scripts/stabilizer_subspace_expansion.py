import argparse
import logging
import numpy as np
from scipy.linalg import eigh
import numpy.ma as ma
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

from shades._utils.models import ising_chain
from shades.utils import pauli_terms_to_matrix
from shades.stabilizer_subspace import StabilizerSubspace, ComputationalSubspace
from plotting_config import setup_plotting_style

setup_plotting_style()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

logger = logging.getLogger(__name__)

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("scipy").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("stim").setLevel(logging.WARNING)
logging.getLogger("qulacs").setLevel(logging.WARNING)


TRANSVERSE_FIELD_STRENGTH = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
N_SAMPLES = np.array([16, 24, 32, 48, 64, 72, 90, 108])
N_RUNS = 30
N_QUBITS = 8

def main(output_dir=None):

    s_energies = np.empty((len(TRANSVERSE_FIELD_STRENGTH), len(N_SAMPLES), N_RUNS))
    cb_energies = np.empty((len(TRANSVERSE_FIELD_STRENGTH), len(N_SAMPLES), N_RUNS))
    exact_energies = np.empty((len(TRANSVERSE_FIELD_STRENGTH)))

    for j, h in enumerate(TRANSVERSE_FIELD_STRENGTH):

        logger.info(f'Running for h={h}...')

        tfim = ising_chain(n_qubits=N_QUBITS, h=h, J=1)
        matrix = pauli_terms_to_matrix(tfim)
        eigs, evs = eigh(matrix)

        exact_energies[j] = eigs[0]
        ground_state_vector = Statevector(evs[:, 0])

        for i, N in enumerate(N_SAMPLES):

            logger.info(f"N = {N} stabilizer states")
            failures = 0

            for run in range(N_RUNS):
                try:
                    subspace = StabilizerSubspace.from_state(
                        ground_state_vector, n_samples=N, pauli_hamiltonian=tfim
                    )
                
                    energy, _ = subspace.optimize_coefficients()
                    s_energies[j, i, run] = energy

                    violation = energy < exact_energies[j] - 1e-10
                    if violation:
                        logger.warning(f"Trial has violated variational principle, ground state energy is {exact_energies[j]} variational result is {energy}")
                    
                except np.linalg.LinAlgError as e:
                    failures += 1
                    continue

                except ValueError as e:
                    logger.warning(e)

                subspace_cb = ComputationalSubspace.from_state(
                    ground_state_vector, n_samples=N, pauli_hamiltonian=tfim
                )
                energy_cb, _ = subspace_cb.optimize_coefficients()
                cb_energies[j, i, run] = energy_cb
                
                error = exact_energies[j] - s_energies[j, i, run]
                error_cb = exact_energies[j] - cb_energies[j, i, run]
                logger.info(f"Run {run+1}: Error Stab={error:.2e}, Error CB={error_cb:.2e}")

        if failures > 0:
            logger.warning(f"{failures} independant samples failed for h={h}!")

    fig, axs = plt.subplots(len(TRANSVERSE_FIELD_STRENGTH), 1, figsize=(10, 12), sharex=True)

    # q1 = np.percentile(s_energies, 25, axis=2, keepdims=True)
    # q3 = np.percentile(s_energies, 75, axis=2, keepdims=True)
    # iqr = q3 - q1

    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr

    # masked_data = ma.masked_where((s_energies < lower_bound) | (s_energies > upper_bound), s_energies)

    mean_stabilizer_energies = np.mean(s_energies, axis=2)
    min_stabilizer_energies = s_energies.min(axis=2)
    max_stabilizer_energies = s_energies.max(axis=2)

    mean_computational_energies = np.mean(cb_energies, axis=2)
    min_computational_energies = cb_energies.min(axis=2)
    max_computational_energies = cb_energies.max(axis=2)

    
    for j, h in enumerate(TRANSVERSE_FIELD_STRENGTH):
        tfim = ising_chain(n_qubits=N_QUBITS, h=h, J=1)
        matrix = pauli_terms_to_matrix(tfim)
        eigs, evs = eigh(matrix)
        exact_energies[j] = eigs[0]

    for j, (ax, h) in enumerate(zip(axs, TRANSVERSE_FIELD_STRENGTH)):
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

        ax.set_xlim(min(N_SAMPLES), max(N_SAMPLES))

        ax.plot(N_SAMPLES, (mean_stabilizer_energies[j, :] - exact_energies[j]) / exact_energies[j], 
                marker='o', markersize=4, linewidth=2, label='Stabilizer subspace expansion')
        ax.fill_between(N_SAMPLES, (min_stabilizer_energies[j, :] - exact_energies[j]) / exact_energies[j], (max_stabilizer_energies[j, :] - exact_energies[j]) / exact_energies[j], alpha=0.2)

        ax.plot(N_SAMPLES, (mean_computational_energies[j, :] - exact_energies[j]) / exact_energies[j],  
                marker='o', markersize=4, linewidth=2, label='Computational basis subspace expansion')
        ax.fill_between(N_SAMPLES, (min_computational_energies[j, :] - exact_energies[j]) / exact_energies[j], (max_computational_energies[j, :] - exact_energies[j]) / exact_energies[j], alpha=0.2)

        ax.text(0.52, 0.90, f'h={h:.1f}', 
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if j == 0:
            ax.legend(fontsize=13, loc='best')


    fig.supylabel(r'$(E-E_{exact})/E_{exact}$', fontsize=16)
    axs[-1].set_xlabel('Number of unique stabilizer states in basis', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure and data if output directory is specified
    if output_dir is not None:
        import os
        import json
        from datetime import datetime

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save figure as SVG
        svg_path = os.path.join(output_dir, 'graph.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        logger.info(f"Figure saved to {svg_path}")

        # Save raw data as NPZ
        npz_path = os.path.join(output_dir, 'data.npz')
        np.savez(
            npz_path,
            stabilizer_energies=s_energies,
            computational_basis_energies=cb_energies,
            transverse_field_strengths=np.array(TRANSVERSE_FIELD_STRENGTH),
            n_samples=N_SAMPLES,
            exact_energies=exact_energies,
            n_runs=N_RUNS,
            n_qubits=N_QUBITS
        )
        logger.info(f"Data saved to {npz_path}")

        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, 'metadata.json')
        metadata = {
            "title": "Stabilizer Subspace Expansion Results",
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "parameters": {
                "n_qubits": N_QUBITS,
                "n_runs": N_RUNS,
                "transverse_field_strengths": TRANSVERSE_FIELD_STRENGTH,
                "n_samples": N_SAMPLES.tolist()
            },
            "files": {
                "graph": "graph.svg",
                "data": "data.npz"
            },
            "data_format": {
                "stabilizer_energies": {
                    "shape": list(s_energies.shape),
                    "description": "Energy estimates from stabilizer subspace expansion"
                },
                "computational_basis_energies": {
                    "shape": list(cb_energies.shape),
                    "description": "Energy estimates from computational basis expansion"
                },
                "transverse_field_strengths": {
                    "values": TRANSVERSE_FIELD_STRENGTH,
                    "description": "Transverse field strengths (h) used in TFIM"
                },
                "n_samples": {
                    "values": N_SAMPLES.tolist(),
                    "description": "Number of basis states sampled"
                },
                "exact_energies": {
                    "description": "Exact ground state energies for each h value"
                },
                "n_runs": {
                    "value": N_RUNS,
                    "description": "Number of independent runs per configuration"
                },
                "n_qubits": {
                    "value": N_QUBITS,
                    "description": "Number of qubits in the system"
                }
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run stabilizer subspace expansion analysis for transverse field Ising model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output files (graph.svg, data.npz, metadata.json). If not specified, only displays the plot.'
    )

    args = parser.parse_args()
    main(output_dir=args.output_dir)
"""H2 Potential Energy Surface via Shadow Tomography.

This script computes the potential energy surface (PES) of molecular hydrogen (H2)
as a function of internuclear distance using shadow state tomography. The results
are compared against exact Full Configuration Interaction (FCI) calculations to
demonstrate the accuracy and statistical properties of the shadow tomography method.

The workflow:
1. For each interatomic distance:
   - Build H2 molecule with specified geometry
   - Compute Hartree-Fock reference and molecular Hamiltonian
   - Run exact FCI calculation for ground truth
   - Perform multiple shadow tomography estimations
2. Compute mean and standard deviation of shadow estimates
3. Plot PES comparing shadow tomography vs exact FCI
4. Save results to disk for further analysis

References:
    - Huggins et al., "Virtual Distillation for Quantum Error Mitigation"
      Nature Physics (2021)
    - Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits"
      Phys. Rev. A 70, 052328 (2004)
"""

from pyscf import gto, scf
from shades.estimators.shadow import ShadowEstimator
from shades.solvers import FCISolver
from shades.utils import make_hydrogen_chain
import numpy as np
import matplotlib.pyplot as plt
from plotting_config import setup_plotting_style, save_figure
import logging

N_SAMPLES = 1000
N_ESTIMATORS = 20
N_SIMULATIONS = 100
N_HYDROGEN = 8
INTERATOMIC_DISTANCES = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25])
N_WORKERS = 8

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

def main():
    """Run H2 stretching analysis and generate PES plot."""

    exact_fci = np.empty_like(INTERATOMIC_DISTANCES)
    estimated_mean = np.empty_like(INTERATOMIC_DISTANCES)
    estimated_std = np.empty_like(INTERATOMIC_DISTANCES)

    # Store ALL individual trial data
    all_estimations = np.empty((len(INTERATOMIC_DISTANCES), N_SIMULATIONS))
    all_c0 = np.empty((len(INTERATOMIC_DISTANCES), N_SIMULATIONS))
    all_c2_norms = np.empty((len(INTERATOMIC_DISTANCES), N_SIMULATIONS))

    print("=" * 80)
    print(f"H2 Potential Energy Surface via Shadow Tomography")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Samples per run:       {N_SAMPLES}")
    print(f"  - Median-of-means bins:  {N_ESTIMATORS}")
    print(f"  - Independent runs:      {N_SIMULATIONS}")
    print(f"  - Number of H atoms:     {N_HYDROGEN}")
    print(f"  - Distances (Å):         {INTERATOMIC_DISTANCES}")
    print("=" * 80)

    for j, d in enumerate(INTERATOMIC_DISTANCES):
        print(f"\n[{j+1}/{len(INTERATOMIC_DISTANCES)}] Processing distance = {d:.2f} Å")

        estimations = np.empty(N_SIMULATIONS)
        c0_values = np.empty(N_SIMULATIONS)
        c2_norms = np.empty(N_SIMULATIONS)

        mol_string = make_hydrogen_chain(N_HYDROGEN, d)
        mol = gto.Mole()
        mol.build(atom=mol_string, basis="sto-3g")

        mf = scf.RHF(mol)
        mf.run()

        fci_solver = FCISolver(mf)
        estimator = ShadowEstimator(mf, solver=fci_solver, verbose=4)
        print(f"  FCI Energy (exact): {estimator.E_exact:.8f} Ha")

        print(f"  Running {N_SIMULATIONS} shadow estimations...")
        for i in range(N_SIMULATIONS):
            E, c0, _, c2 = estimator.run(n_samples=N_SAMPLES, n_k_estimators=N_ESTIMATORS, n_jobs=N_WORKERS)
            estimations[i] = E
            c0_values[i] = np.abs(c0)
            c2_norms[i] = np.linalg.norm(c2)

            if (i + 1) % 1 == 0:
                print(f"    Completed {i+1}/{N_SIMULATIONS} runs")

            estimator.clear_sample()

        # Store individual trial data for this distance
        all_estimations[j, :] = estimations
        all_c0[j, :] = c0_values
        all_c2_norms[j, :] = c2_norms

        exact_fci[j] = estimator.E_exact
        estimated_mean[j] = np.mean(estimations)
        estimated_std[j] = np.std(estimations)

        error = estimated_mean[j] - exact_fci[j]
        print(f"  Shadow Mean:  {estimated_mean[j]:.8f} Ha")
        print(f"  Shadow Std:   {estimated_std[j]:.8f} Ha")
        print(f"  Mean Error:   {error:+.2e} Ha")


    print("\n" + "=" * 80)
    print("Generating Potential Energy Surface Plot")
    print("=" * 80)

    setup_plotting_style()

    _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(INTERATOMIC_DISTANCES, exact_fci,
            'o-', label='Exact FCI', linewidth=2, markersize=6, color='C0')

    ax.errorbar(INTERATOMIC_DISTANCES, estimated_mean, yerr=estimated_std,
                fmt='s--', label='Shadow Estimate', linewidth=1.5,
                markersize=5, capsize=4, capthick=1.5, color='C1', alpha=0.8)

    ax.set_xlabel(r'Interatomic Distance (\AA)')
    ax.set_ylabel(r'Ground State Energy (Ha)')
    ax.set_title(f'H$_{{{N_HYDROGEN}}}$ Potential Energy Surface')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_prefix = f"h2_stretching_N{N_HYDROGEN}"
    save_figure(f"{output_prefix}.png", dpi=300)

    plt.show()

    print("\nSaving numerical results...")

    # Save summary statistics as text file (backward compatible)
    results = np.column_stack((INTERATOMIC_DISTANCES, exact_fci,
                               estimated_mean, estimated_std))
    header = "Distance(Å)  FCI_Energy(Ha)  Shadow_Mean(Ha)  Shadow_Std(Ha)"
    np.savetxt(f"{output_prefix}_data.txt", results,
               header=header, fmt='%.10f', delimiter='  ')
    print(f"Summary statistics saved to {output_prefix}_data.txt")

    # Save ALL individual trial data to .npz file
    npz_filename = f"{output_prefix}_all_trials.npz"
    np.savez(
        npz_filename,
        distances=INTERATOMIC_DISTANCES,
        exact_fci=exact_fci,
        all_energies=all_estimations,
        all_c0=all_c0,
        all_c2_norms=all_c2_norms,
        estimated_mean=estimated_mean,
        estimated_std=estimated_std,
        # Save metadata
        n_samples=N_SAMPLES,
        n_estimators=N_ESTIMATORS,
        n_simulations=N_SIMULATIONS,
        n_hydrogen=N_HYDROGEN
    )
    print(f"All trial data saved to {npz_filename}")
    print(f"  - Shape of all_energies: {all_estimations.shape}")
    print(f"  - Contains: distances, exact_fci, all_energies, all_c0, all_c2_norms, + metadata")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
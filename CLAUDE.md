# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shadow CI is a quantum chemistry research package that implements **classical shadow tomography** for ground state energy estimation. It combines quantum information techniques with traditional quantum chemistry (PySCF) to estimate molecular ground states and correlation energies using the shadow tomography protocol.

**Core Scientific Method**: The package uses the "mixed energy estimator" approach to approximate ground state wavefunctions via classical shadow measurements. It leverages Clifford group sampling to efficiently estimate excitation amplitudes (c1, c2) from quantum states, then contracts these with molecular integrals to compute correlation energies.

## Development Commands

### Installation
```bash
# Basic installation (includes all quantum chemistry dependencies)
pip install -e .

# Development mode (includes pytest, black, ruff, mypy)
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_shadow_protocol.py

# Run with coverage report
pytest --cov=shadow_ci --cov-report=term-missing

# Run single test by name
pytest tests/test_shadow_protocol.py::test_collect_samples
```

### Code Quality
```bash
# Format code with black (line length: 100)
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type checking
mypy src/
```

### Benchmarking
```bash
# Run all performance benchmarks
pytest benchmarks/ --benchmark-only

# Run specific benchmark class
pytest benchmarks/benchmark_ground_state.py::TestShadowProtocol --benchmark-only

# Run shadow scaling analysis
python benchmarks/benchmark_shadow_scaling.py
```

## Architecture Overview

### Core Components

**1. Molecular Hamiltonian ([hamiltonian.py](src/shadow_ci/hamiltonian.py))**
- Bridge between PySCF quantum chemistry and shadow tomography
- Stores one-electron (h1e) and two-electron (h2e) integrals in MO basis
- Generates single/double excitations with RHF symmetry optimization
- **Key convention**: Spin-orbital ordering is [α₀, α₁, ..., α_{n-1}, β₀, β₁, ..., β_{n-1}]
- **Bitstring convention**: Little-endian (bit i from right = orbital i)
- For RHF systems: Exploits spin symmetry to reduce unique excitations by ~40%

**2. Shadow Protocol ([shadows.py](src/shadow_ci/shadows.py))**
- Implements classical shadow tomography using Clifford group sampling
- Collects measurement snapshots: (bitstring, Clifford tableau) pairs
- Computes overlaps via stabilizer formalism and Gaussian elimination
- **Performance**: Supports both Qiskit and Qulacs backends; Qulacs is ~5-10x faster
- Parallelization: `n_jobs > 1` for multi-core sampling and estimation
- **Critical**: Uses median-of-means (K estimators) for robust statistical estimation

**3. Ground State Solvers ([solvers/](src/shadow_ci/solvers/))**
- Base class: `GroundStateSolver` (abstract interface)
- `FCISolver`: PySCF Full CI solver for exact ground states
- `VQESolver`: Variational Quantum Eigensolver with UCC ansatz
- All solvers convert final states to Qiskit `Statevector` format

**4. Ground State Estimator ([estimator.py](src/shadow_ci/estimator.py))**
- Main entry point for shadow-based energy estimation
- Workflow:
  1. Collect shadow samples from trial state
  2. Estimate HF reference overlap (c0)
  3. Estimate single excitation amplitudes (c1) - optional
  4. Estimate double excitation amplitudes (c2)
  5. Contract amplitudes with Fock matrix and ERIs to compute E_corr
- Returns: (total_energy, c0, c1, c2)

**5. Utilities ([utils.py](src/shadow_ci/utils.py))**
- `Bitstring`: Custom bitstring class with stabilizer conversion
- `SingleExcitation`, `DoubleExcitation`: Dataclasses for excitations
- `SingleAmplitudes`, `DoubleAmplitudes`: Amplitude tensor management
- Gaussian elimination for stabilizer phase computations
- Helper functions for HF reference states and excitation generation

### Data Flow

```
PySCF MeanField (mf)
    ↓
MolecularHamiltonian.from_pyscf(mf)
    ↓
GroundStateSolver.solve() → trial Statevector
    ↓
ShadowProtocol(trial).collect_samples(N, K)
    ↓
GroundStateEstimator.estimate_ground_state()
    ├─ estimate_reference_determinant() → c0
    ├─ estimate_first_order_interactions() → c1
    ├─ estimate_second_order_interaction() → c2
    └─ compute_correlation_energy(c0, c1, c2) → E_total
```

## Key Implementation Details

### RHF Symmetry Exploitation
For restricted Hartree-Fock systems, the code automatically:
- Returns only alpha single excitations (beta are identical by symmetry)
- Returns only alpha-alpha and alpha-beta double excitations (beta-beta are redundant)
- This reduces shadow measurement requirements significantly (~40% fewer overlaps)

### Fermion-to-Qubit Encoding
- Default: Jordan-Wigner (JW) encoding where qubit i ↔ spin-orbital i
- Support for Bravyi-Kitaev and Parity encodings (partially implemented)
- Bitstrings use little-endian convention matching PySCF occupation strings

### Shadow Protocol Performance
- **Critical**: Set `use_qulacs=True` for medium/large molecules (>6 qubits)
- Qulacs provides 5-10x speedup over Qiskit for state evolution
- For parallel sampling: set `n_jobs > 1` (works best with Qiskit backend for small systems)
- Verbosity levels: 0=silent, 1=basic progress, 2=detailed timing, 3=debug

### Amplitude Tensor Structure
- `SingleAmplitudes`: Shape (nocc, nvirt) - occupied → virtual transitions
- `DoubleAmplitudes`: Shape (nocc, nocc, nvirt, nvirt) - antisymmetrized
- For RHF: Amplitudes are automatically expanded to both spin channels when needed

## Common Workflows

### Running Ground State Estimation
```python
from pyscf import gto, scf
from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.solvers import FCISolver
from shadow_ci.estimator import GroundStateEstimator

# Setup molecule
mol = gto.Mole()
mol.build(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol).run()

# Get exact FCI ground state as trial state
solver = FCISolver(mf)

# Estimate via shadows
estimator = GroundStateEstimator(mf, solver, verbose=2)
energy, c0, c1, c2 = estimator.estimate_ground_state(
    n_samples=1000,
    n_k_estimators=10,
    n_jobs=4,
    use_qualcs=True
)
```

### Analyzing Convergence
See [scripts/convergence.py](scripts/convergence.py) for systematic convergence analysis varying N_samples.

### Benchmark Usage
```bash
# Quick benchmark of shadow protocol
pytest benchmarks/benchmark_ground_state.py::TestShadowProtocol --benchmark-only

# Full scaling analysis (generates CSV + plots)
python benchmarks/benchmark_shadow_scaling.py --n-samples 100 500 1000 2000
```

## File Organization

- `src/shadow_ci/`: Main package code
  - `hamiltonian.py`: Molecular Hamiltonian and excitation generation
  - `shadows.py`: Shadow protocol and overlap estimation
  - `estimator.py`: Main ground state estimator
  - `utils.py`: Bitstrings, excitations, amplitude tensors
  - `solvers/`: Ground state solver implementations
- `tests/`: Unit tests (pytest)
- `benchmarks/`: Performance benchmarks (pytest-benchmark)
- `scripts/`: Research scripts (convergence analysis, H2 stretching, etc.)
- `results/`: Output directory for computed results

## Important Conventions

### Energy Units
All energies are in **Hartrees** (atomic units).

### Integral Storage
- h1e: One-electron integrals in MO basis, shape (norb, norb)
- h2e: Two-electron integrals from `ao2mo.full()`, flattened 4D array
- Both use physicist's notation: (pq|rs)

### State Representation
- Quantum states: Qiskit `Statevector` objects
- Classical snapshots: `(Bitstring, stim.Tableau)` pairs
- Trial states MUST be in Jordan-Wigner encoding for current implementation

### Parallelization Notes
- Shadow sampling: Best with Qulacs backend, serial mode (n_jobs=1) for large systems
- For small systems (<8 qubits): Qiskit with n_jobs>1 can parallelize sampling
- Overlap estimation: Always parallelizes efficiently across K estimators when n_jobs>1

## Dependencies

**Core quantum libraries:**
- PySCF: Quantum chemistry (integrals, FCI, mean-field)
- Qiskit: Quantum circuits and state vectors
- Qiskit Nature: Molecular Hamiltonians and fermion-qubit mappings
- Qulacs: Fast quantum state simulation (recommended)
- Stim: Stabilizer circuit simulation and Clifford operations

**Performance:**
- Qulacs provides the best performance for shadow sampling (5-10x faster than Qiskit)

## Testing Strategy

Tests are organized by functionality:
- `test_shadow_protocol.py`: Shadow sampling and overlap estimation
- `test_bitstring.py`: Bitstring conversions and stabilizer formalism
- `test_stabilizer.py`: Clifford operations and canonicalization
- `test_double_amplitudes_*.py`: Amplitude tensor antisymmetry and RHF symmetry
- `test_rhf_optimization.py`: Validates RHF symmetry optimizations work correctly

Run tests before committing changes to ensure quantum chemistry conventions remain correct.

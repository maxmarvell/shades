# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shades is a quantum chemistry research package that implements **classical shadow tomography** for ground state energy estimation. It combines quantum information techniques with traditional quantum chemistry (PySCF) to estimate molecular ground states and correlation energies using the shadow tomography protocol.

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
pytest --cov=shades --cov-report=term-missing

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

**1. Excitation Generation ([excitations.py](src/shades/excitations.py))**
- Generates single and double excitations from PySCF mean-field objects
- Creates bitstring representations for each excitation
- Converts excitation lists to amplitude tensors (t1, t2)
- **Key convention**: Spin-orbital ordering is [α₀, α₁, ..., α_{n-1}, β₀, β₁, ..., β_{n-1}]
- **Bitstring convention**: Little-endian (bit i from right = orbital i)
- For RHF systems: Exploits spin symmetry to reduce unique excitations by ~40%
- Helper functions: `get_singles()`, `get_doubles()`, `get_hf_reference()`

**2. Shadow Protocol ([shadows.py](src/shades/shadows.py))**
- Implements classical shadow tomography using Clifford group sampling
- `ClassicalShadow`: Stores snapshots as lists of stim.Tableau objects
- `ShadowProtocol`: Orchestrates shadow collection and overlap estimation
  - Initialize with `ShadowProtocol(state)`
  - Call `collect_samples_for_overlaps(n_samples, n_estimators)` to collect shadows from tau state
  - Call `estimate_overlap(bitstring, n_jobs=1)` to estimate overlaps using median-of-means
- Computes overlaps via stabilizer formalism and Gaussian elimination
- **Performance**: Uses Qulacs backend for ~5-10x faster state evolution than Qiskit
- Parallelization: `n_jobs > 1` in `estimate_overlap()` for multi-core overlap computation
- **Critical**: Uses median-of-means (K estimators) for robust statistical estimation
- **Logging**: Uses Python logging module at DEBUG level for overlap timing information

**3. Ground State Solvers ([solvers/](src/shades/solvers/))**
- Base class: `GroundStateSolver` (abstract interface in [base.py](src/shades/solvers/base.py))
- `FCISolver`: PySCF Full CI solver for exact ground states
- `VQESolver`: Variational Quantum Eigensolver with UCC ansatz
- All solvers convert final states to Qiskit `Statevector` format
- Returns tuple: (statevector, exact_energy)

**4. Estimators ([estimators/](src/shades/estimators/))**
- Base class: `AbstractEstimator` in [base.py](src/shades/estimators/base.py)
  - Defines workflow: estimate_c0() → estimate_c1() → estimate_c2() → compute_correlation_energy()
  - Main method: `run(calc_c1=False)` returns (total_energy, c0, c1, c2)
- `ShadowEstimator`: Uses classical shadow tomography for overlap estimation
  - Initialize with `ShadowEstimator(mf, solver, verbose=0)`
  - Method: `run(n_samples, n_k_estimators, n_jobs=1, use_qulacs=True, calc_c1=False)`
  - Collects shadow samples once via internal `ShadowProtocol`, reuses for all excitation overlaps
  - `n_jobs` parameter controls parallelization during overlap estimation
  - `use_qulacs` currently always uses Qulacs for sampling (parameter retained for compatibility)
- `TrivialEstimator`: Direct state vector access for exact overlaps (no shadows, for testing)
  - Method: `run(calc_c1=False)`

**5. Brueckner Orbitals ([brueckner.py](src/shades/brueckner.py))**
- Implements Brueckner orbital transformations to minimize single excitations
- `brueckner_cycle(mf, estimator, ...)`: Iteratively rotates orbitals to drive t1 amplitudes to zero
- `rotate_mo_coeffs()`: Applies orbital rotations using exponential or Taylor expansion
- `rotate_mf()`: Creates new mean-field object with rotated molecular orbitals
- Supports both RHF and UHF references with optional DIIS convergence acceleration

**6. Utilities ([utils.py](src/shades/utils.py))**
- `Bitstring`: Custom bitstring class with stabilizer conversion and endianness support
- `gaussian_elimination()`: Computes overlap phases via stabilizer formalism
- `compute_correlation_energy()`: Contracts amplitudes with Fock matrix and ERIs
- `make_hydrogen_chain()`: Helper to construct hydrogen chain geometries
- Helper functions for stabilizer canonicalization and phase computation

### Data Flow

```
PySCF MeanField (mf: scf.RHF or scf.UHF)
    ↓
GroundStateSolver(mf).solve() → (trial Statevector, exact_energy)
    ↓
AbstractEstimator(mf, solver) [ShadowEstimator or TrivialEstimator]
    ↓
estimator.run(n_samples, n_k_estimators, ...)
    │
    ├─ [Shadow only] Create ShadowProtocol(trial_state)
    │       ↓
    │   collect_samples_for_overlaps(n_samples, n_k_estimators)
    │       ↓
    │   Creates K ClassicalShadow estimators with n_samples/K each
    │
    ├─ estimate_c0() → |⟨HF|ψ⟩|
    │       ↓
    │   get_hf_reference(mf) → bitstring
    │   estimate_overlap(bitstring) → c0
    │
    ├─ estimate_c1() → t1 amplitudes (optional)
    │       ↓
    │   get_singles(mf) → List[SingleExcitation]
    │   For each excitation: estimate_overlap(bitstring)
    │   singles_to_t1() → t1 tensor
    │
    ├─ estimate_c2() → t2 amplitudes
    │       ↓
    │   get_doubles(mf) → List[DoubleExcitation]
    │   For each excitation: estimate_overlap(bitstring)
    │   doubles_to_t2() → t2 tensor
    │
    └─ compute_correlation_energy(mf, c0, c1, c2) → E_corr
            ↓
        E_total = E_HF + E_corr
        return (E_total, c0, c1, c2)
```

### Brueckner Cycle Workflow (Optional)

```
Initial MeanField (mf₀)
    ↓
for iteration in range(max_iter):
    ↓
    estimator.update_reference(mfᵢ)
    ↓
    Estimator.run(calc_c1=True) → (E, c0, c1, c2)
    ↓
    if ||c1|| < threshold: converged ✓
    ↓
    rotate_mf(mfᵢ, c1, ...) → mfᵢ₊₁
    ↓
    mfᵢ = mfᵢ₊₁

Final: Brueckner orbitals where c1 ≈ 0
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
- **Critical**: Qulacs is used for state evolution during shadow sampling (5-10x faster than Qiskit)
- Parallelization: Set `n_jobs > 1` in `estimator.run()` to parallelize overlap estimation across K estimators
- The `use_qulacs` parameter in `ShadowEstimator.run()` is currently always True

### Logging Configuration
- The package uses Python's `logging` module for debug and informational messages
- **IMPORTANT**: Scripts must configure logging to see output. Add this to your scripts:
  ```python
  import logging
  logging.basicConfig(
      level=logging.DEBUG,  # Use DEBUG to see all messages, INFO for less verbose
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
      force=True,
  )
  ```
- Estimator verbosity levels are independent but still used for backward compatibility: 0=WARNING, 1=INFO, 2+=DEBUG

### Amplitude Tensor Structure
- **t1 tensors** (single excitations):
  - RHF: Shape (nocc, nvirt) - alpha electrons only
  - UHF: Tuple of (t1_alpha, t1_beta), shapes (nocc_a, nvirt_a) and (nocc_b, nvirt_b)
- **t2 tensors** (double excitations):
  - RHF: Shape (nocc, nocc, nvirt, nvirt) - antisymmetrized alpha-beta amplitudes
  - UHF: Tuple of (t2_aa, t2_bb, t2_ab) for all three spin cases
- For RHF: Amplitudes are automatically expanded to both spin channels in energy computation

## Common Workflows

### Running Shadow-Based Ground State Estimation
```python
from pyscf import gto, scf
from shades.solvers import FCISolver
from shades.estimators import ShadowEstimator

# Setup molecule
mol = gto.Mole()
mol.build(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol).run()

# Get exact FCI ground state as trial state
solver = FCISolver(mf)

# Create shadow estimator
estimator = ShadowEstimator(mf, solver, verbose=1)

# Estimate energy via shadow tomography
energy, c0, c1, c2 = estimator.run(
    n_samples=1000,
    n_k_estimators=10,
    n_jobs=4,
    use_qulacs=True,
    calc_c1=False  # Set True to include singles
)

print(f"Shadow Energy: {energy:.8f} Ha")
print(f"Exact FCI Energy: {estimator.E_exact:.8f} Ha")
print(f"Error: {energy - estimator.E_exact:.2e} Ha")
```

### Running Exact (Non-Shadow) Estimation
```python
from pyscf import gto, scf
from shades.solvers import FCISolver
from shades.estimators import TrivialEstimator

mol = gto.Mole()
mol.build(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol).run()

solver = FCISolver(mf)
estimator = TrivialEstimator(mf, solver, verbose=1)

# Direct overlap computation (no shadow samples needed)
energy, c0, c1, c2 = estimator.run(calc_c1=True)
```

### Running Brueckner Orbital Optimization
```python
from pyscf import gto, scf
from shades.solvers import FCISolver
from shades.estimators import ShadowEstimator
from shades.brueckner import brueckner_cycle

mol = gto.Mole()
mol.build(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol).run()

solver = FCISolver(mf)
estimator = ShadowEstimator(mf, solver, verbose=1)

# Define convergence callback
def converged(E, c0, norm):
    return norm < 1e-6

# Run Brueckner cycle to minimize singles
# Note: n_samples/n_k_estimators are passed to estimator.run() internally
brueckner_cycle(
    mf,
    estimator,
    max_iter=10,
    damping=0.8,
    use_diis=False,
    callback_fn=converged,
    verbose=2,
    method="taylor"
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

- `src/shades/`: Main package code
  - `excitations.py`: Excitation generation and amplitude tensor conversion
  - `shadows.py`: Shadow protocol, classical snapshots, and overlap estimation
  - `brueckner.py`: Brueckner orbital transformations and cycle optimization
  - `utils.py`: Bitstring class, stabilizer operations, helper functions
  - `stabilizer.py`: Stabilizer-related utilities
  - `estimators/`: Estimator implementations
    - `base.py`: Abstract estimator with shared workflow
    - `shadow.py`: Shadow tomography-based estimator
    - `trivial.py`: Exact overlap estimator (for testing/validation)
  - `solvers/`: Ground state solver implementations
    - `base.py`: Abstract solver interface
    - `fci.py`: PySCF Full CI solver
    - `vqe.py`: Variational Quantum Eigensolver
- `tests/`: Unit tests (pytest)
  - `test_shadow_protocol.py`: Shadow protocol and overlap estimation tests
  - `test_bitstring.py`: Bitstring class tests
  - `test_stabilizer.py`: Stabilizer formalism tests
- `benchmarks/`: Performance benchmarks (pytest-benchmark)
- `scripts/`: Research scripts
  - `convergence.py`: Shadow sampling convergence analysis
  - `brueckner.py`: Brueckner orbital optimization demonstrations
  - `h2_stretching.py`: H₂ potential energy surface
  - `sample_space.py`: Sample space exploration
- `results/`: Output directory for computed results

## Important Conventions

### Energy Units
All energies are in **Hartrees** (atomic units).

### Integral Storage
- One-electron integrals: Retrieved from `mf.get_hcore()` in MO basis
- Two-electron integrals: Retrieved from `ao2mo.full()` in MO basis
- Both use physicist's notation: ⟨pq|rs⟩
- Integrals are accessed directly from PySCF mean-field objects, not stored separately

### State Representation
- Quantum states: Qiskit `Statevector` objects (from solvers)
- Classical snapshots: Lists of `stim.Tableau` objects within `ClassicalShadow`
- Bitstrings: Custom `Bitstring` class with little-endian convention by default
- Trial states use Jordan-Wigner encoding (qubit i ↔ spin-orbital i)

### Parallelization Notes
- Shadow sampling: Uses Qulacs backend (currently not parallelized during sampling)
- Overlap estimation: Parallelizes efficiently across K estimators when `n_jobs > 1`
- The `n_jobs` parameter in `estimator.run()` controls parallelization during overlap estimation phase
- For best performance: Use `n_k_estimators >= n_jobs` to ensure all workers stay busy

## Dependencies

**Core quantum libraries:**
- PySCF: Quantum chemistry (integrals, FCI, mean-field)
- Qiskit: Quantum circuits and state vectors
- Qiskit Nature: Molecular Hamiltonians and fermion-qubit mappings
- Qulacs: Fast quantum state simulation (used for shadow sampling)
- Stim: Stabilizer circuit simulation and Clifford operations

**Performance:**
- Qulacs provides the best performance for shadow sampling (5-10x faster than Qiskit)

## Testing Strategy

Tests are organized by functionality:
- Shadow protocol tests: Sampling, overlap estimation, and median-of-means
- Bitstring tests: Conversions, stabilizer formalism, and endianness
- Stabilizer tests: Clifford operations, canonicalization, and Gaussian elimination
- Excitation tests: Single/double excitation generation and amplitude tensor conversions
- Estimator tests: Validate both shadow and trivial estimators produce correct results
- Solver tests: FCI and VQE solver correctness

Run tests before committing changes to ensure quantum chemistry conventions remain correct:
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/test_*.py    # Run specific test file
```

# CKKS Parameter Recommendation Tool

An automated, data-driven tool for selecting parameters of the CKKS homomorphic encryption scheme for privacy-preserving machine learning inference.

The tool analyzes the dataset and the model, generates and evaluates candidate parameter configurations, and presents the trade-off between runtime, precision, and security, with a recommended configuration for each security level.

---

## Overview

Homomorphic encryption enables secure outsourced computation, but choosing CKKS parameters requires expert knowledge. The tool:

- estimates precision requirements from the dataset and the model output, and takes the model's multiplicative depth as a workload constraint;
- generates a workload-specific search space of parameter sets `(N, Δ, Q)` that the library can instantiate;
- evaluates every candidate with encrypted inference on representative (synthetic) test vectors, so that no individual record is used during the search;
- computes, for each security target (128, 192, 256 bits), the Pareto front of runtime against error over every configuration that is **at least** that secure, and recommends its knee point (runtimes within 5% of each other are treated as equal, so a noise-level speed difference never costs precision);
- re-evaluates the recommendations on real dataset rows.

## How parameters are generated

Chains follow the SEAL / TenSEAL structure `(q0, q1, …, qL, special)`:

| Component | Rule |
|---|---|
| `q1 … qL` | one prime per rescale, sized like the scale `Δ`. TenSEAL rescales after **every** multiplication (ciphertext-plaintext included), so a model of depth `L` needs exactly `L` of them. |
| `q0` | holds the output after the last rescale: `log2 q0 ≥ log2 Δ + ⌈log2 max|f(x)|⌉ + 1`, with `max|f(x)|` taken from the plaintext model on the test vectors. |
| special prime | key-switching prime, at least as large as every other prime. |
| scale `Δ` | a grid starting at the data-derived lower bound (finest per-feature resolution plus a buffer, or `data.target_precision_bits`), plus, for each security level, the largest scale whose smallest valid chain fits that level's budget at this `N`. Every prime takes one 64-bit word, so runtime depends on `N` and the number of primes but hardly on the scale: a larger scale gives more precision at the same cost. Budget scales are also tried with one spare level, which gives intermediate products (scale Δ² before the first rescale) headroom when inputs are large. |
| `N` | every supported ring dimension whose 128-bit budget fits the chain. |

Chains that SEAL cannot build (a prime size outside 2–60 bits, or not enough primes `≡ 1 mod 2N` of a given size) are recorded in `infeasible.csv` and not evaluated.

**Model depth** (`model.multiplicative_depth`) is the number of rescales the encrypted forward pass performs: a linear model has depth 1; the MNIST network (3 matmuls, 2 squares) has depth 5.

## Security model

Security levels are assigned from the Homomorphic Encryption Standard tables as implemented in Microsoft SEAL (`hestdparms.h`): a ternary secret key, error standard deviation 3.2, and a classical attacker. The bound is the total size of the chain, special prime included. TenSEAL only enforces the 128-bit budget itself; the 192- and 256-bit labels are assigned by the tool. For newer guidance see Bossuat et al., *Security Guidelines for Implementing Homomorphic Encryption* (IACR CiC 2024).

The tool decrypts results to measure error. In a deployment, decrypted CKKS outputs must not be returned to an untrusted party without countermeasures (Li & Micciancio, EUROCRYPT 2021).

---

## Project Structure

```text
ckks_param_tool/
├── config/        # Experiment configuration (dataclasses, JSON loader)
├── data/          # Dataset loaders, precision estimation, test vector generation
├── encryption/    # TenSEAL CKKS backend
├── evaluation/    # Candidate and recommended-parameter evaluation
├── experiment/    # Experiment runner, result types, environment capture
├── models/        # Homomorphic-compatible models and training helpers
├── params/        # Parameter representation, security tables, prime feasibility
├── plotting/      # Visualization utilities
├── search/        # Search policies, search space, Pareto selection, baseline
└── cli.py         # Command line interface
benchmark_suite/   # Benchmarks: tool vs near-exhaustive search, test vectors, baseline
tests/             # Unit and end-to-end tests
legacy/            # Prototype scripts and the June 2026 results (pre-0.2, not reproducible with this version)
reproduce.py       # Re-creates all benchmark tables and figures
```

## Installation

```bash
git clone https://github.com/<your-username>/ckks-param-tool.git
cd ckks-param-tool

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"          # add ,mnist for the MNIST workload (torch)
# or, for the exact versions used for the results:
pip install -r requirements.txt && pip install -e .
```

Run the tests with `pytest`.

## Example Usage

```python
from pathlib import Path

from ckks_param_tool import Experiment
from ckks_param_tool import client_server_split
from ckks_param_tool import load_diabetes_data
from ckks_param_tool import train_linear_regression_model

# Load dataset
X, y = load_diabetes_data()

# Simulate client/server split (the server trains on 30%, the client holds the rest)
X_client, X_server, y_client, y_server = client_server_split(X, y)

# Train model on server-side data
model = train_linear_regression_model(X_server, y_server)

# Initialize experiment (config is optional: a dict, a JSON path, or an ExperimentConfig)
experiment = Experiment(X_client, model, config={"search": {"profile": "default"}})

# Run parameter selection and evaluation
experiment.run(verbose=True)

# Save results
experiment.save(Path("results"))
```

From the command line, for a built-in workload:

```bash
ckks-param-tool run --dataset diabetes --profile default --output results
```

Each saved run contains `config.json`, `param_search.csv` (every evaluated candidate), `infeasible.csv`, `recommended_params.json`, `run_info.json` (precision estimates, search time, software and hardware environment) and `plots/`.

### Configuration

| Section | Field | Default | Meaning |
|---|---|---|---|
| `data` | `num_samples` | 10 | number of representative test vectors |
| | `method` | `truncated_normal` | sampler (`truncated_normal`, `gaussian`, `uniform`, `student_t`) |
| | `max_decimal_bits` | 40 | cap on the estimated decimal precision |
| | `target_precision_bits` | `None` | use this instead of the data estimate to set the scales |
| | `random_state` | 42 | seed |
| `evaluation` | `num_samples` | 10 | real rows used to evaluate the recommendations |
| | `warmup_runs` | 1 | untimed inferences before timing a candidate |
| | `timing_repeats` | 1 | timed passes over the inputs |
| | `n_threads` | `None` | TenSEAL threads (`None` = all cores) |
| `search` | `profile` | `default` | `fast`, `default`, `full`, `fuller` (near-exhaustive) or `custom` |
| | `poly_degrees`, `num_scale_options` (0 = no limit), `scale_step`, `base_scale_buffer`, `min_scale_bits`, `budget_scales`, `budget_extra_levels`, `extra_levels`, `q0_deltas`, `max_qL` | from profile | override single fields of the profile |
| `plotting` | `error_tol` | `None` | hide candidates above this error in plots |

Runtime is the median wall-clock time of one encrypted inference (encrypt, evaluate, decrypt); key generation is excluded. Error is the worst-case absolute difference to the plaintext model; precision is `-log2(error)`.

## Reproducing the benchmarks

```bash
python reproduce.py                        # all workloads; MNIST takes several hours
python reproduce.py --datasets iris wine   # a subset
python reproduce.py --no-fuller            # skip the near-exhaustive comparison
```

For every workload this runs the tool, the near-exhaustive `fuller` search (every scale from 2^20 to the largest feasible one in 5-bit steps, independent of the data, up to 4 spare levels and 6 sizes of q0), the tool's candidates with random dataset rows and with dataset-agnostic vectors, and the tutorial-default baseline (`N=8192`, `[60, 40, 40, 60]`, `Δ=2^40`). `results/summary/` then contains:

- `search_efficiency` — candidate and search-time reduction, Pareto recall, ε-recall and normalized hypervolume ratio per security level. The tool's front is computed from the `fuller` run's measurements of the tool's own candidates (every tool candidate is also a `fuller` candidate), so the comparison is not affected by run-to-run timing noise.
- `test_vectors` — Spearman correlation, Pearson correlation and R² on log error, and coverage (how often the estimate is at least the real error) for representative and naive vectors against random rows.
- `baseline` — the tool's recommendations versus the tutorial defaults on the same real rows.

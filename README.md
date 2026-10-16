# CKKS Parameter Recommendation Tool

An automated, data-driven tool for selecting parameters of the CKKS homomorphic encryption scheme for privacy-preserving machine learning inference.

The tool analyzes the dataset and the model, generates and evaluates candidate parameter configurations, and presents the trade-off between runtime, precision, and security, with a recommended configuration for each security level.

---

## Overview

Homomorphic encryption enables secure outsourced computation, but choosing CKKS parameters requires expert knowledge. The tool:

- works from a **data descriptor** (per-feature mean, standard deviation, minimum, maximum and resolution) and the model; the client can compute the descriptor locally and share it instead of any record;
- bounds the model's outputs from the descriptor and takes the model's multiplicative depth as a workload constraint;
- generates a workload-specific search space of parameter sets `(N, Δ, Q)` that the library can instantiate;
- evaluates every candidate with encrypted inference on representative (synthetic) test vectors;
- computes, for each security target (128, 192, 256 bits), the Pareto front of runtime against error over every configuration that is **at least** that secure, and reports its two endpoints (fastest, most precise) and its knee point as a starting point (runtimes within 5% of each other count as equal, so a noise-level speed difference never costs precision);
- if the raw data is available, re-evaluates these configurations on real rows (error, prediction agreement, overflowing rows).

## How parameters are generated

Chains follow the SEAL / TenSEAL structure `(q0, q1, …, qL, special)`:

| Component | Rule |
|---|---|
| `q1 … qL` | one prime per rescale, sized like the scale `Δ`. TenSEAL rescales after **every** multiplication (ciphertext-plaintext included), so a model of depth `L` needs exactly `L` of them. |
| `q0` | holds the output after the last rescale: `log2 q0 ≥ log2 Δ + b_out + 1` (rounded up to a multiple of 10). `b_out` is the integer bits of the largest plaintext output on 10,000 samples drawn from the descriptor plus its stress vectors (each feature at its minimum or maximum, the others at their mean), plus a 3-bit safety margin (`data.output_margin_bits`). Only this minimum is used: extra bits in `q0` add headroom, not precision. |
| special prime | key-switching prime, at least as large as every other prime. Two sizes are tried: the smallest valid one (cheapest in the security budget) and 60 bits (least key-switching noise). |
| scale `Δ` | a lower bound (the data's finest per-feature resolution plus a buffer, or `data.target_precision_bits` if set), plus, for each security level, the largest scale whose smallest valid chain fits that level's budget at this `N` (the *budget scales*). Budget scales are also tried with one spare level. |
| `N` | every supported ring dimension whose 128-bit budget fits the chain. |

**Why only these scales.** Every prime takes one 64-bit word in the RNS representation, so runtime depends on `N` and the number of primes but hardly on the scale: for a given `N` and chain length a larger scale gives more precision at the same cost. Each `(N, chain length)` therefore has one Pareto-optimal configuration, the largest scale that works, and that is what the budget scales target. The lower bound guarantees a candidate that meets the data's precision.

**Overflow.** CKKS computes modulo `q`: every value is multiplied by the scale and must fit in the current modulus, otherwise it wraps around and decrypts to garbage (an error of about `q0 / Δ`). This can happen at the end (the output must fit in `q0`, which is why `q0` is sized from the output) and after a multiplication (see step-down). On real rows the tool reports the share of rows whose error is at least as large as their own output (and at least 1) as the *overflow rate*.

**Step-down.** Before the first rescale a value carries scale `Δ²`, so at large scales the top modulus can be too small for intermediate products (for example, large raw inputs times weights). If a candidate fails or returns a result whose error is at least as large as the output itself, the same structure (`N`, number of levels, `q0` rule, special prime rule) is retried 5 bits lower, until one works or the lower bound is reached. Every attempt is kept in `param_search.csv`.

Chains that SEAL cannot build (a prime size outside 2–60 bits, or not enough primes `≡ 1 mod 2N` of a given size) are recorded in `infeasible.csv` and not evaluated.

**Model depth** (`model.multiplicative_depth`) is the number of rescales the encrypted forward pass performs: a linear model has depth 1; the MNIST network (3 matmuls, 2 squares) has depth 5.

**Precision a classifier needs (information only).** A prediction flips when the error reaches the decision margin (`|logit|` for binary models, half the gap between the two largest logits otherwise). From the margins on the descriptor samples the tool estimates the precision that keeps 99% of predictions unchanged (`data.margin_quantile`) and reports it and draws it in the plots; it does not change the recommendations.

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

from ckks_param_tool import Experiment, DataDescriptor
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

# ...or without any record: the client shares only its descriptor
# descriptor = DataDescriptor.from_data(X_client); descriptor.save("descriptor.json")
# experiment = Experiment(DataDescriptor.load("descriptor.json"), model)

# Run parameter selection and evaluation
experiment.run(verbose=True)

# Save results
experiment.save(Path("results"))
```

From the command line, for a built-in workload:

```bash
ckks-param-tool run --dataset diabetes --profile default --output results
```

With a descriptor only, the results are reported on the test vectors. Each saved run contains `config.json`, `descriptor.json` (all the tool knew about the data), `param_search.csv` (every evaluated candidate, step-downs included), `infeasible.csv`, `recommended_params.json`, `run_info.json` (precision estimates, search time, number of step-downs, git commit, software and hardware environment) and `plots/`.

### Configuration

| Section | Field | Default | Meaning |
|---|---|---|---|
| `data` | `num_samples` | 10 | number of representative test vectors |
| | `method` | `truncated_normal` | sampler (`truncated_normal`, `gaussian`, `uniform`, `student_t`) |
| | `max_decimal_bits` | 40 | cap on any precision estimated from the data |
| | `target_precision_bits` | `None` | optional precision target: raises the lower bound on the scale, and the fastest configuration meeting it is reported per security level |
| | `magnitude_samples` | 10000 | plaintext-only samples used to bound outputs and margins |
| | `output_margin_bits` | 3 | safety bits added to the output bound that sizes `q0` |
| | `margin_quantile` | 0.01 | classifiers: estimated precision keeps `1 - q` of predictions |
| | `random_state` | 42 | seed |
| `evaluation` | `num_samples` | 100 | real rows used to evaluate the reported configurations (0 = all) |
| | `recommendation` | `knee` | recommended point per level: `knee`, `fastest` or `most_accurate` |
| | `warmup_runs` | 1 | untimed inferences before timing a candidate |
| | `timing_repeats` | 1 | timed passes over the inputs |
| | `n_threads` | `None` | TenSEAL threads (`None` = all cores) |
| `search` | `profile` | `default` | `fast` (N ≤ 16384, no spare levels), `default`, `full` (grid of 4 scales, `q0` offsets, every special prime size, one spare level), `fuller` (near-exhaustive) or `custom` |
| | `poly_degrees`, `num_scale_options` (grid scales above the lower bound; 1 = only the lower bound, 0 = no limit), `scale_step`, `base_scale_buffer`, `min_scale_bits`, `budget_scales`, `budget_extra_levels`, `extra_levels`, `q0_deltas`, `max_qL`, `special_primes` (`min_max` or `all`), `step_down_bits` (0 = off) | from profile | override single fields of the profile |
| `plotting` | `error_tol` | `None` | hide candidates above this error in plots |

Runtime is the median wall-clock time of one encrypted inference (encrypt, evaluate, decrypt); key generation is excluded. Error is the worst-case absolute difference to the plaintext model; precision is `-log2(error)`. A candidate whose error is at least the largest output magnitude is reported as a failure ("no precision left").

The knee recommendation is only a starting point: the tool's output is the trade-off itself, i.e. the evaluated candidates, the Pareto front for each security level and its endpoints.

## Reproducing the benchmarks

```bash
python reproduce.py                        # all workloads; MNIST takes several hours
python reproduce.py --datasets iris wine   # a subset
python reproduce.py --no-fuller            # skip the near-exhaustive comparison
```

For every workload this runs the tool, the near-exhaustive `fuller` search (every scale from 2^20 to the largest feasible one in 5-bit steps, independent of the data, up to 4 spare levels and 6 sizes of q0), the tool's candidates with random dataset rows and with dataset-agnostic vectors, and the tutorial-default baseline (`N=8192`, `[60, 40, 40, 60]`, `Δ=2^40`). `results/summary/` then contains:

- `search_efficiency` — candidate and search-time reduction, Pareto recall, ε-recall and normalized hypervolume ratio per security level. The tool's front is computed from the `fuller` run's measurements of the tool's own candidates (every tool candidate is also a `fuller` candidate), so the comparison is not affected by run-to-run timing noise. Recall counts the reference-front parameter sets the tool also contains; ε-recall counts a reference point as found if the tool has one at most 5% slower and at most 0.1 decades less accurate. Hypervolume is computed on (runtime, log10 error), normalized by the reference front, with the runtime axis spanning at least 5% of the fastest runtime so that noise-level differences on flat fronts are not amplified.
- `test_vectors` — Spearman correlation, Pearson correlation and R² on log error, and coverage (how often the estimate is at least the real error) for representative and naive vectors against random rows.
- `recommendations` — the knee point and the front's endpoints per security level, measured on all client rows (200 for MNIST): precision, runtime, prediction agreement, relative error and overflow rate.
- `baseline` — the tutorial defaults against every configuration the tool evaluated: the fastest one at least as precise, and the most precise one at least as fast. The baseline is measured with the same test vectors, and both sides are re-measured on the same real rows.

On flat fronts (all candidates with the same `N` and chain length) runtime differences are mostly measurement noise, so repeated timing (`evaluation.timing_repeats`) makes the search-efficiency numbers more stable.

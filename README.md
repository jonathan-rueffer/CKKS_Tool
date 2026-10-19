# CKKS Parameter Recommendation Tool

An automated, data-driven tool for selecting parameters of the CKKS homomorphic encryption scheme for privacy-preserving machine learning inference.

The tool analyzes the dataset and the model, generates and evaluates candidate parameter configurations, and presents the trade-off between runtime, precision, and security, with a recommended configuration for each security level.

---

## Overview

Homomorphic encryption enables secure outsourced computation, but choosing CKKS parameters requires expert knowledge. The tool:

- is run by the **client**, on a **data descriptor** (per-feature mean, standard deviation, minimum and maximum) that it computes locally, so no record enters the tool;
- treats the service's model as a black box with a known multiplicative depth, and bounds its outputs from the descriptor;
- generates a workload-specific search space of parameter sets `(N, Δ, Q)` that the library can instantiate;
- evaluates every candidate with encrypted inference on representative (synthetic) test vectors;
- computes, for each security target (128, 192, 256 bits), the Pareto front of runtime against error over every configuration that is **at least** that secure, and reports its two endpoints (fastest, most precise) and its knee point as a starting point (runtimes within 5% of each other count as equal, so a noise-level speed difference never costs precision);
- in the benchmark only, validates these configurations on real rows (error, prediction agreement, overflowing rows).

## How parameters are generated

Chains follow the SEAL / TenSEAL structure `(q0, q1, …, qL, special)`:

| Component | Rule |
|---|---|
| `q1 … qL` | one prime per rescale, sized like the scale `Δ`. TenSEAL rescales after **every** multiplication (ciphertext-plaintext included), so a model of depth `L` needs exactly `L` of them. |
| `q0` | holds the output after the last rescale: `log2 q0 ≥ log2 Δ + b_out + 1` (rounded up to a multiple of 10). `b_out` is the integer bits of the largest plaintext output on 10,000 samples drawn from the descriptor plus its stress vectors (each feature at its minimum or maximum, the others at their mean), plus a 3-bit safety margin (`data.output_margin_bits`). Only this minimum is used: extra bits in `q0` add headroom, not precision. |
| special prime | key-switching prime, at least as large as every other prime. Two sizes are tried: the smallest valid one (cheapest in the security budget) and 60 bits (least key-switching noise). |
| scale `Δ` | for each `N` and security level, the largest scale (multiple of 5 bits, at least 2^20) whose smallest valid chain fits that level's budget at this `N` (the *budget scales*), also tried with one spare level. A precision target (`data.target_precision_bits`) raises the lowest scale considered. |
| `N` | every supported ring dimension whose 128-bit budget fits the chain. |

**Why only these scales.** Every prime takes one 64-bit word in the RNS representation, so runtime depends on `N` and the number of primes but hardly on the scale: for a given `N` and chain length a larger scale gives more precision at the same cost. Each `(N, chain length)` therefore has one Pareto-optimal configuration, the largest scale that works, and that is what the budget scales target. Together they cover both ends of the trade-off: the budget scale of a small `N` at a high security level is the cheap, less precise option, that of a larger `N` the precise one.

**What the data decides.** The scales are not derived from the data: an earlier version used the finest per-feature resolution of the data as a lower bound on the scale, but input resolution says nothing about the precision an output needs, the bound almost never applied, and where it did it excluded the only fast configuration of a security level. The data instead sizes `q0` (through the output bound), supplies the test vectors that measure every candidate and drive the step-down, and, for classifiers, gives the decision margins (information only). A precision floor comes only from the user (`data.target_precision_bits`).

**Overflow.** CKKS computes modulo `q`: every value is multiplied by the scale and must fit in the current modulus, otherwise it wraps around and decrypts to garbage (an error of about `q0 / Δ`). This can happen at the end (the output must fit in `q0`, which is why `q0` is sized from the output) and after a multiplication (see step-down). On real rows the tool reports the share of rows whose error is at least as large as their own output (and at least 1) as the *overflow rate*.

**Step-down.** Before the first rescale a value carries scale `Δ²`, so at large scales the top modulus can be too small for intermediate products (for example, large raw inputs times weights). If a candidate fails or returns a result whose error is at least as large as the output itself, the same structure (`N`, number of levels, `q0` rule, special prime rule) is retried 5 bits lower, until one works or the lowest scale considered is reached. Every attempt is kept in `param_search.csv`.

Chains that SEAL cannot build (a prime size outside 2–60 bits, or not enough primes `≡ 1 mod 2N` of a given size) are recorded in `infeasible.csv` and not evaluated.

**Model depth** (`model.multiplicative_depth`) is the number of rescales the encrypted forward pass performs: a linear model has depth 1; the MNIST network (3 matmuls, 2 squares) has depth 5.

**Precision a classifier needs (information only).** A prediction flips when the error reaches the decision margin (`|logit|` for binary models, half the gap between the two largest logits otherwise). From the margins on the descriptor samples the tool estimates the precision that keeps 99% of predictions unchanged (`data.margin_quantile`) and reports it and draws it in the plots; it does not change the recommendations.

## Threat model

The tool is a subroutine of the **client's** interaction with a machine-learning service:

1. The client computes the descriptor of its data locally. The raw data never enters the tool.
2. The tool draws synthetic vectors from the descriptor and uses the service's model as a black box with a known multiplicative depth: it queries the model **in plaintext on synthetic vectors** (reference outputs, output size, decision margins) and **on ciphertexts** (the service evaluates its model homomorphically under each candidate's parameters; the client decrypts).
3. The client uses the chosen parameters for its real, encrypted requests.

Assumptions and consequences:

- The service answers plaintext queries on synthetic vectors. Those vectors reveal the client's per-feature statistics (the descriptor) to the service, but no record.
- Decrypted results stay with the client. Returning decrypted CKKS outputs to an untrusted party enables key-recovery attacks (Li & Micciancio, EUROCRYPT 2021).
- In this implementation both kinds of query are simulated locally, so runtimes are the service's computation (plus encryption and decryption); network and serialization costs come on top in a deployment.
- The descriptor treats features independently. For tabular data this is a good approximation; for structured inputs such as images it is not (see the benchmark results for MNIST).

## Security model

Security levels are assigned from the Homomorphic Encryption Standard tables as implemented in Microsoft SEAL (`hestdparms.h`): a ternary secret key, error standard deviation 3.2, and a classical attacker. The bound is the total size of the chain, special prime included. TenSEAL only enforces the 128-bit budget itself; the 192- and 256-bit labels are assigned by the tool. For newer guidance see Bossuat et al., *Security Guidelines for Implementing Homomorphic Encryption* (IACR CiC 2024).

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
benchmark_suite/   # Benchmarks: tool vs reference grid, test vectors, baselines, noise
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

# The client computes its descriptor locally; only the descriptor enters the tool
descriptor = DataDescriptor.from_data(X_client)

# Initialize experiment (config is optional: a dict, a JSON path, or an ExperimentConfig)
experiment = Experiment(descriptor, model, config={"search": {"profile": "default"}})

# Run parameter selection and evaluation
experiment.run(verbose=True)

# Save results
experiment.save(Path("results"))
```

From the command line, for a built-in workload:

```bash
ckks-param-tool run --dataset diabetes --profile default --output results
```

All results are measured on the test vectors. Each saved run contains `config.json`, `descriptor.json` (all the tool knew about the data), `param_search.csv` (every evaluated candidate, step-downs included), `infeasible.csv`, `recommended_params.json`, `run_info.json` (precision estimates, search time, number of step-downs, git commit, software and hardware environment) and `plots/`.

### Configuration

| Section | Field | Default | Meaning |
|---|---|---|---|
| `data` | `num_samples` | 10 | number of representative test vectors |
| | `method` | `truncated_normal` | sampler (`truncated_normal`, `gaussian`, `uniform`, `student_t`) |
| | `max_decimal_bits` | 40 | cap on the precision estimated from decision margins |
| | `target_precision_bits` | `None` | optional precision target: raises the lowest scale considered, and the fastest configuration meeting it is reported per security level |
| | `magnitude_samples` | 10000 | plaintext-only samples used to bound outputs and margins |
| | `output_margin_bits` | 3 | safety bits added to the output bound that sizes `q0` |
| | `margin_quantile` | 0.01 | classifiers: estimated precision keeps `1 - q` of predictions |
| | `random_state` | 42 | seed |
| `evaluation` | `num_samples` | 100 | benchmark validation only: real rows per configuration (0 = all) |
| | `recommendation` | `knee` | recommended point per level: `knee`, `fastest` or `most_accurate` |
| | `warmup_runs` | 1 | untimed inferences before timing a candidate |
| | `timing_repeats` | 1 | timed passes over the inputs |
| | `n_threads` | `None` | TenSEAL threads (`None` = all cores) |
| `search` | `profile` | `default` | `fast` (N ≤ 16384, no spare levels), `default`, `full` (adds a grid of 4 scales from the floor, `q0` offsets, every special prime size, one spare level), `broad_grid` (the benchmark's reference grid, see below) or `custom` |
| | `poly_degrees`, `min_scale_bits` (lowest scale considered, 20), `num_scale_options` (grid scales from that floor in addition to the budget scales, 0 = none), `full_grid` (grid up to the largest feasible scale), `scale_step`, `base_scale_buffer` (added to a precision target to get the floor), `budget_scales`, `budget_extra_levels`, `extra_levels`, `q0_deltas`, `max_qL`, `special_primes` (`min_max` or `all`), `step_down_bits` (0 = off) | from profile | override single fields of the profile |
| | `runtime_tolerance` | 0.05 | runtimes within this relative difference count as equal (knee, endpoints, ε-recall) |
| | `precision_tolerance_bits` | 1.5 | precisions within this many bits count as equal ("most precise" endpoint, ε-recall); twice the largest standard deviation measured by `benchmark_suite.noise` |
| | `min_precision_bits` | 0 | reported points (knee, endpoints) must reach this precision |
| | `measure_cost` | `true` | record key generation time and serialized key / ciphertext sizes per candidate |
| `plotting` | `error_tol` | `None` | hide candidates above this error in plots |

**Metrics.** Runtime is the median wall-clock latency of one encrypted inference (encrypt, evaluate, decrypt); key generation is excluded and reported separately. Error is the worst-case absolute difference to the plaintext model over every output slot and test vector; precision is `-log2(error)`, and *relative precision* is `-log2(error / max|output|)`, which is comparable across workloads with outputs of different sizes. A candidate whose error is at least the largest output magnitude is reported as a failure ("no precision left").

**Tuning cost.** Each candidate needs its own keys, which a client would send to the service, and several encrypted queries. For every candidate the tool records the key generation time, the serialized size of the public context with the relinearization and Galois keys (never the secret key), and the size of one input and one output ciphertext; `run_info.json` sums them over the search. For example, the evaluation keys alone are about 19 MiB at `N=8192` and about 540 MiB at `N=32768` with 7 primes.

The knee recommendation is only a starting point: the tool's output is the trade-off itself, i.e. the evaluated candidates, the Pareto front for each security level and its endpoints.

## Reproducing the benchmarks

```bash
python reproduce.py                        # all workloads; MNIST takes several hours
python reproduce.py --datasets iris wine   # a subset
python reproduce.py --no-broad-grid        # skip the reference-grid comparison
```

For every workload this runs the tool, a broad reference grid (`broad_grid`), the tool's candidates with random dataset rows and with dataset-agnostic vectors, and two baselines. `results/summary/` then contains the tables below (CSV and LaTeX) and figures.

**Workloads.** Eight datasets were used while the candidate-generation rules were developed: diabetes, california housing, concrete (regression), breast cancer, heart disease, ionosphere (binary), wine, iris (3 classes), plus MNIST with a depth-5 square-activation network. Three **held-out** datasets were not used during development and measure how the rules generalize: credit default (binary, 23 numeric features with values up to about 10^6), bank marketing (binary, categorical features one-hot encoded) and wine quality (regression). Tables mark them with `held_out`.

**Reference grid.** `broad_grid` is a broad but finite grid, not an exhaustive search: every scale from 2^20 to the largest feasible one in 5-bit steps (independent of the data), `q0` from its minimum in steps of 10 up to +50 bits, every special prime size in steps of 10, 0 to 4 spare levels, and `N` in {8192, 16384, 32768}. It uses the same `q0` rule as the tool. Every tool candidate is also a grid candidate (tested).

**Baselines.** *Tutorial defaults*, what a non-expert copies from the TenSEAL / SEAL examples (`N=8192`, `[60, 40, 40, 60]`, `Δ=2^40`); and *expert defaults*, the rule of thumb of OpenFHE's examples (60-bit first prime, 50-bit scaling primes without spare levels, 60-bit special prime, the smallest `N` whose 128-bit budget fits).

**Real rows.** Reported configurations and baselines are validated on the client's real rows (all of them up to 5,000, a seeded random subset beyond that; 200 for MNIST).

- `search_efficiency` — per security level: candidate and search-time reduction, exact Pareto recall, ε-recall, the multiplicative ε-indicator, and the normalized hypervolume ratio with and without the runtime floor. The tool's front is computed from the grid run's measurements of the tool's own candidates, so the comparison is not affected by run-to-run timing noise.
  - *Pareto recall*: the share of the grid front's parameter sets (`N`, `Δ`, `Q`) that the tool also evaluated.
  - *ε-recall*: the share of grid-front points for which the tool has one at most `runtime_tolerance` slower and at most `precision_tolerance_bits` less precise.
  - *ε-indicator* (Zitzler et al. 2003): the smallest factor by which the tool front's runtimes and errors must be divided to weakly dominate the whole grid front; 1 is perfect.
  - *Hypervolume ratio*: on (runtime, log10 error), normalized by the grid front's ideal and nadir points, reference point at 1.1. The runtime axis spans at least `runtime_tolerance` of the fastest runtime so that noise-level differences on flat fronts are not amplified; `hypervolume_ratio_no_floor` omits that.
- `test_vectors` — Spearman correlation, Pearson correlation and R² on log error, and coverage (how often the estimate is at least the real error) for representative and naive vectors against random rows.
- `recommendations` — the knee point and the front's endpoints per security level: precision on the test vectors and on real rows (absolute and relative), runtime, prediction agreement, relative error and overflow rate.
- `baseline` — each baseline against every configuration the tool evaluated: the fastest one at least as precise, and the most precise one at least as fast, measured on the same real rows.
- `tuning_cost` — candidates, encrypted queries, search and key generation time, and the data a client would exchange with the service (evaluation keys and ciphertexts).
- `model_quality` — plaintext accuracy or R² of each model on the client split.

**Measurement noise.** `python -m benchmark_suite.noise --datasets ... --repeats 5` measures the configurations the tool reports several times, each with new keys and encryption noise, and suggests `runtime_tolerance` and `precision_tolerance_bits` as twice the largest observed coefficient of variation and standard deviation. On four datasets the runtime varied by at most 0.8% and the precision by up to 0.76 bits, since the error is a maximum over 10 random test vectors.

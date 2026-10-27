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
- in the benchmark only, validates these configurations on real rows (error, prediction agreement, catastrophic deviations).

## How parameters are generated

Chains follow the SEAL / TenSEAL structure `(q0, q1, …, qL, special)`:

| Component | Rule |
|---|---|
| `q1 … qL` | one prime per rescale, sized like the scale `Δ`. TenSEAL rescales after **every** multiplication (ciphertext-plaintext included), so a model of depth `L` needs exactly `L` of them. |
| final modulus | holds the output after the last rescale: at least `log2 Δ + b_out + 1` bits. It is a single prime `q0` sized exactly for that, or, when one prime cannot hold it (every prime is at most 60 bits), two primes `(q0, 60)` with `q0 = max(B - 60, 20)` for a required size of `B` bits. A final modulus needing more than two primes (`B > 120`) is not supported; no candidate is generated. `b_out` is the integer bits of the largest plaintext output on 10,000 samples drawn from the descriptor plus its stress vectors (each feature at its minimum or maximum, the others at their mean), plus a 3-bit safety margin (`data.output_margin_bits`). Only this minimum is used, not rounded: extra bits add headroom, not precision, and at the high security levels every unused bit of the budget is precision lost. |
| special prime | key-switching prime, at least as large as every other prime. Two sizes are tried: the smallest valid one, sized exactly (cheapest in the security budget), and 60 bits (least key-switching noise). |
| scale `Δ` | for each `N` and security level, the largest scale (multiple of 5 bits, at least 2^20) whose smallest valid chain fits that level's budget at this `N` (the *budget scales*), also tried with one spare level. A precision target (`data.target_precision_bits`) raises the lowest scale considered. |
| `N` | every supported ring dimension whose 128-bit budget fits the chain. |

**Why only these scales.** Every prime takes one 64-bit word in the RNS representation, so runtime depends on `N` and the number of primes but hardly on the scale: for a given `N` and chain length a larger scale gives more precision at the same cost. Each `(N, chain length)` therefore has one Pareto-optimal configuration, the largest scale that works, and that is what the budget scales target. Together they cover both ends of the trade-off: the budget scale of a small `N` at a high security level is the cheap, less precise option, that of a larger `N` the precise one.

**The precision ceiling.** With a single `q0` the scale can never exceed `60 - b_out - 1` bits, however much of the security budget is left: from `N=16384` on, the 60-bit word limit, not the budget, caps precision. The two-prime final modulus lifts that limit at the cost of one prime (about 1.6x the runtime of a depth-1 model at `N=8192`). Precision then levels off at about 35-37 bits even at `Δ=2^60`, because CKKS encoding in SEAL / TenSEAL goes through double-precision numbers.

**What the data decides.** The scales are not derived from the data: an earlier version used the finest per-feature resolution of the data as a lower bound on the scale, but input resolution says nothing about the precision an output needs, the bound almost never applied, and where it did it excluded the only fast configuration of a security level. The data instead sizes `q0` (through the output bound), supplies the test vectors that measure every candidate and drive the step-down, and, for classifiers, gives the decision margins (information only). A precision floor comes only from the user (`data.target_precision_bits`).

**Overflow and catastrophic deviations.** CKKS computes modulo `q`. Additions, multiplications and rescalings preserve the message modulo the current modulus, so values may exceed it in the middle of a computation without harm (tested up to 19 bits above the headroom). An overflow happens only at **decryption**: when a coefficient of the decrypted plaintext polynomial, times the scale, reaches half the modulus left after the last rescale, the result wraps (it changes by a multiple of about `q_final / Δ`). The coefficients mix every slot, and TenSEAL leaves rotation partial sums and broadcast scalars (a bias is added to every slot) next to the result, so the largest coefficient is not the largest output. It can be much smaller (one value in one slot is spread over all `N` coefficients) or larger (a large dot product cancelled by a large bias leaves the partial sums plus the bias in the other slots: an output of 1 can need 4 bits of headroom). The output-based rule `log2 Δ + b_out + 1` is therefore a heuristic, not a bound.

A result is a *catastrophic deviation* if its error is at least as large as the output (no correct bit; per real row also at least 1), **or** at least `2^headroom` (`headroom = bits of the modulus left after the last rescale − log2 Δ − 1`): an overflow changes a value by a multiple of about `2^(headroom + 1)`, which noise never does. The second criterion catches overflows whose wrapped value keeps its sign (an output of 50 with 3 bits of headroom wraps to 2, an error of 48 < 50), which the first one misses. The check only sees the evaluated inputs and does not claim the cause. On real rows the benchmark reports the share of such rows as the *catastrophic rate*.

**Measured headroom.** For every evaluated candidate the tool also decrypts all slots of each result (client side, outside the timed region; `evaluation.measure_coefficients`, about 10% of the search time), computes the plaintext polynomial's coefficients from them (inverse canonical embedding; SEAL's slot `k` holds the value at the root `exp(iπ·5^k/N)`), and records the largest one (`max_coefficient`, in units of the scale) and `coefficient_headroom_bits = output headroom − log2(max_coefficient)`.

- *What it is:* the actual coefficients of the decrypted results, not a model of the operations, so it covers whatever the model's encrypted forward pass leaves in the ciphertext (biases, partial sums, broadcasts, activations).
- *When it is exact:* on the **evaluated inputs** only (the maximum over the test vectors, or over the real rows in the benchmark), assuming the result did not overflow (a wrapped result shows wrapped coefficients; such candidates fail the check below and carry no headroom), the slots' imaginary parts are noise (TenSEAL encodes real vectors), and up to the use of nominal prime sizes and scale (SEAL's primes lie just below `2^bits`; cases within about 0.1 bit of the threshold are not decided). Tested for broadcast and per-slot biases, single-slot dot products, matmuls, a two-prime final modulus, a spare level, a depth-3 network with a square activation and intermediate values far above the headroom (`tests/test_coefficient_headroom.py`), and against encryption in 60 of 60 randomized linear cases.
- *What it is not:* a prediction for inputs that were not evaluated. It cannot be computed without evaluating the model, and other inputs can have larger coefficients (tested).

**Step-down.** If a candidate fails or deviates catastrophically, it is retried 5 bits lower **with the same final modulus**, the same number of levels and the same special prime, until one works or the lowest scale considered is reached. Keeping the final modulus is what makes the retry useful: every step trades 5 bits of precision for 5 bits of output headroom. Re-sizing the final modulus for the lower scale, as earlier versions did, keeps the headroom constant and cannot fix an overflow. Only the scaling primes shrink, so the retried candidate is at least as secure as the original; its final modulus is larger than the minimum for its scale, so it is generally not a `broad_grid` candidate. Every attempt is kept in `param_search.csv`. In the published benchmark runs no candidate failed, so no step-down occurred.

Chains that SEAL cannot build (a prime size outside 2–60 bits, or not enough primes `≡ 1 mod 2N` of a given size) are recorded in `infeasible.csv` and not evaluated.

**What decides what.**

| quantity | role |
|---|---|
| `b_out` (largest plaintext output on descriptor samples, stress vectors and test vectors, plus the 3-bit margin) | **generation**: sizes the final modulus of every candidate |
| security budgets, prime feasibility | **generation**: candidates that exceed a budget or cannot be built are not evaluated |
| catastrophic deviation on the test vectors (error ≥ largest output, or ≥ `2^headroom`) | **rejection**: the candidate is a failure, never reported, and retried with a step-down |
| `coefficient_headroom_bits`, `max_coefficient`, `max_output` | **diagnostic only**: recorded per candidate; never used to generate, reject or rank |
| real-row catastrophic rate, prediction agreement | **benchmark validation only** (needs the client's records) |

A reported configuration can therefore keep less coefficient headroom than the 3-bit margin intended (it worked on the test vectors, with less spare room). Such points are still reported, with `coefficient_headroom_below_margin: true` in `recommended_params.json` and a warning in the log; on other inputs they may overflow. No reported configuration is guaranteed safe on inputs that were not evaluated.

**What SEAL enforces and what the tool decides.**

| | rule | enforced by |
|---|---|---|
| primes | 2–60 bits, `≡ 1 mod 2N`, distinct; found by searching downward from `2^bits` (reproduced in `params/primes.py` to check feasibility before key generation) | SEAL |
| 128-bit budget | total chain size (special prime included) at most the HE Standard's 128-bit bound for `N` | SEAL, when TenSEAL creates the context |
| 192- / 256-bit labels | the same tables at the higher levels | the tool (not enforced by TenSEAL) |
| special prime ≥ every other prime | SEAL's recommendation for key-switching noise | the tool |
| final modulus `≥ log2 Δ + b_out + 1` bits | the output-capacity heuristic above (SEAL does not check message sizes) | the tool |
| `b_out` and its 3-bit margin | empirical: plaintext outputs on descriptor samples, stress vectors and test vectors | the tool |
| scales | multiples of 5 bits from 2^20 to 2^60 | the tool |

**Search resolution and the margin (fixed for the published results).** The 5-bit scale lattice is a search-resolution choice, not an exhaustive search over valid scales: for 73% of the tool's candidates in the benchmark a larger scale in 1-bit steps also fits (1–4 bits more), which measured a median of 3.2 bits more precision at the same latency, but 12% of those measured 0.3–2.7 bits *worse* (unexplained); 1-bit scales are left as possible future work. The 3-bit output margin is a heuristic: it protects against outputs above the estimate (the raw estimate was exceeded on breast cancer, ionosphere and MNIST), and it excludes viable configurations. Example: on diabetes the expert configuration `(60, 50, 60)` at `2^50` needs only the raw 9 output bits and is excluded by the margin (63 bits needed); it showed no catastrophic deviation on the evaluated rows with 1.7 bits of coefficient headroom, which does not make it safe on other inputs.

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
- The descriptor itself reveals per-feature statistics. `DataDescriptor.private(X, epsilon, bounds)` releases an ε-differentially private descriptor instead (Laplace mechanism on the mean and second moment of every feature, data clipped to public per-feature `bounds`, which are released as the minimum and maximum; ε is split evenly over the 2·d statistics). The bounds must be public, e.g. domain knowledge, not computed from the private data.

## Security model

Security levels are assigned from the Homomorphic Encryption Standard tables as implemented in Microsoft SEAL (`hestdparms.h`): a ternary secret key, error standard deviation 3.2, and a classical attacker. The bound is the total size of the chain, special prime included; nominal prime sizes are used (SEAL's primes lie just below `2^bits`), which errs on the safe side. TenSEAL only enforces the 128-bit budget itself; the 192- and 256-bit labels are assigned by the tool. For newer guidance see Bossuat et al., *Security Guidelines for Implementing Homomorphic Encryption* (IACR CiC 2024).

These levels cover the lattice estimates of the HE Standard only. They do not cover attacks on CKKS that use decrypted results (IND-CPA^D, Li and Micciancio, Eurocrypt 2021): in the tool's workflow the client decrypts and nothing decrypted returns to the service, and a deployment that shares decryptions with the service needs additional countermeasures. The workflow also sends the service about 10^4 plaintext queries plus the test vectors, which reveal the model to the client (a linear model is determined by `d + 1` queries); the tool does not protect the model.

The results were produced with TenSEAL 0.3.18 built on Microsoft SEAL 4.3; `run_info.json` records the SEAL version (read from SEAL's serialization header) with the other package versions.

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
| `data` | `num_samples` | 50 | representative test vectors encrypted per candidate (the error is a maximum over them; more vectors, less noise) |
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
| | `poly_degrees`, `min_scale_bits` (lowest scale considered, 20), `num_scale_options` (grid scales from that floor in addition to the budget scales, 0 = none), `full_grid` (grid up to the largest feasible scale), `scale_step`, `base_scale_buffer` (added to a precision target to get the floor), `budget_scales`, `budget_extra_levels`, `extra_levels`, `q0_deltas`, `q0_rounding` (`exact` and/or `round10`), `two_prime_q0`, `max_qL`, `special_primes` (`min_max` or `all`), `step_down_bits` (0 = off) | from profile | override single fields of the profile |
| | `runtime_tolerance` | 0.05 | runtimes within this relative difference count as equal (knee, endpoints, ε-recall) |
| | `precision_tolerance_bits` | 1.5 | precisions within this many bits count as equal ("most precise" endpoint, ε-recall); twice the largest standard deviation measured by `benchmark_suite.noise` |
| | `min_precision_bits` | 0 | reported points (knee, endpoints) must reach this precision |
| | `measure_cost` | `true` | record key generation time and serialized key / ciphertext sizes per candidate |
| | `measure_coefficients` | `true` | record the largest plaintext coefficient and the coefficient headroom of every result (diagnostic only; outside the timed latency, about 10% of the search time; the published runs predate it) |
| `plotting` | `error_tol` | `None` | hide candidates above this error in plots |

**Metrics.** Runtime is the median wall-clock latency of one encrypted inference (encrypt, evaluate, decrypt) with client and service in one process: it excludes network transfer and serialization; key generation is excluded and reported separately. Error is the worst-case absolute difference to the plaintext model over every output slot and test vector; precision is `-log2(error)`, and *relative precision* is `-log2(error / max|output|)`, which is comparable across workloads with outputs of different sizes. A candidate that deviates catastrophically on the test vectors (see above) is reported as a failure.

**Tuning cost.** Each candidate needs its own keys, which a client would send to the service, and several encrypted queries. For every candidate the tool records the key generation time, the serialized size of the public context with the relinearization and Galois keys (never the secret key), and the size of one input and one output ciphertext; `run_info.json` sums them over the search. For example, the evaluation keys alone are about 19 MiB at `N=8192` and about 540 MiB at `N=32768` with 7 primes. These are upper bounds: TenSEAL generates Galois keys for every rotation, while SEAL and OpenFHE can generate only the rotations a model uses.

The knee recommendation is only a starting point: the tool's output is the trade-off itself, i.e. the evaluated candidates, the Pareto front for each security level and its endpoints.

## Reproducing the benchmarks

```bash
python reproduce.py                        # noise, benchmark, analysis; MNIST takes several hours
python reproduce.py --datasets iris wine   # a subset
python reproduce.py --no-broad-grid        # skip the reference-grid comparison
```

`reproduce.py` first measures the measurement noise and writes the resulting tolerances to the run's config, then runs the benchmark with it and analyses the results. For every workload the benchmark runs the tool, a broad reference grid (`broad_grid`), the tool's candidates with random dataset rows and with dataset-agnostic vectors, and two baselines. `results/summary/` then contains the tables below (CSV and LaTeX) and figures.

**Workloads.** Eight datasets were used while the candidate-generation rules were developed: diabetes, california housing, concrete (regression), breast cancer, heart disease, ionosphere (binary), wine, iris (3 classes), plus MNIST with a depth-5 square-activation network. Three **held-out** datasets were not used during development and measure how the rules generalize: credit default (binary, 23 numeric features with values up to about 10^6), bank marketing (binary, categorical features one-hot encoded) and wine quality (regression). Tables mark them with `held_out`.

**Reference grid.** `broad_grid` is a broad but finite grid, not an exhaustive search: every scale from 2^20 to 2^60 in 5-bit steps (independent of the data); `q0` both sized exactly and rounded up to a multiple of 10, each with offsets up to +50 bits in steps of 10; one- and two-prime final moduli; the smallest valid special prime and every multiple of 10 above it; 0 to 4 spare levels; and `N` in {8192, 16384, 32768}. It uses the same output bound as the tool. Every tool candidate is also a grid candidate (tested).

**Baselines.** *Tutorial defaults*, what a non-expert copies from the TenSEAL / SEAL examples (`N=8192`, `[60, 40, 40, 60]`, `Δ=2^40`); and *expert defaults*, the rule of thumb of OpenFHE's examples (60-bit first prime, 50-bit scaling primes without spare levels, 60-bit special prime, the smallest `N` whose 128-bit budget fits).

**Real rows.** Reported configurations and baselines are validated on the client's real rows (all of them up to 5,000, a seeded random subset beyond that; 200 for MNIST).

- `search_efficiency` — per security level: candidate and search-time reduction, exact Pareto recall, ε-recall, the multiplicative ε-indicator, and the normalized hypervolume ratio with and without the runtime floor. The tool's front is computed from the grid run's measurements of the tool's own candidates, so the comparison is not affected by run-to-run timing noise.
  - *Pareto recall*: the share of the grid front's parameter sets (`N`, `Δ`, `Q`) that the tool also evaluated.
  - *ε-recall*: with `R` the grid front, `T` the tool front, `τ_t = runtime_tolerance`, `τ_p = precision_tolerance_bits`: the share of `r ∈ R` for which some `t ∈ T` has `latency(t) ≤ (1 + τ_t)·latency(r)` and `precision(t) ≥ precision(r) − τ_p` (both inclusive).
  - *ε-indicator* (multiplicative, Zitzler et al. 2003): `I(T, R) = max_{r ∈ R} min_{t ∈ T} max(latency(t)/latency(r), error(t)/error(r))`. 1 means `T` weakly dominates `R`; below 1 `T` is better; e.g. 2 means some grid point needs twice the latency or error.
  - *Hypervolume ratio*: `HV(T)/HV(R)` on (latency, log10 error), both minimized; each axis normalized by the grid front's ideal (minimum) and span (maximum − minimum), reference point (1.1, 1.1); tool points beyond it contribute nothing, and the ratio can exceed 1. The latency span is at least a **fixed 5%** of the fastest latency so that noise-level differences on flat fronts are not amplified; `hypervolume_ratio_floor_tau` uses the calibrated `runtime_tolerance` instead and `hypervolume_ratio_no_floor` no floor (in the published runs all three are identical: every grid front spans more than 5%).
- `test_vectors` — Spearman correlation, Pearson correlation and R² on log error, and coverage (how often the estimate is at least the real error) for representative and naive vectors against random rows.
- `recommendations` — the knee point and the front's endpoints per security level, with whether each lies on the reference grid's Pareto front (`on_grid_front`, from the grid's own measurement) or near it (`near_grid_front`: no grid configuration is both more than the runtime tolerance faster and more than the precision tolerance more precise): precision on the test vectors and on real rows (absolute and relative), runtime, prediction agreement, relative error and catastrophic rate.
- `baseline` — each baseline against every configuration the tool evaluated: the fastest one at least as precise, and the most precise one at least as fast, measured on the same real rows.
- `tuning_cost` — candidates, encrypted queries, search and key generation time, and the data a client would exchange with the service (evaluation keys and ciphertexts).
- `model_quality` — plaintext accuracy or R² of each model on the client split.
- `validation_accounting` — what was measured how: test vectors per candidate (50; 10 for MNIST), encrypted candidates, and which configurations were also encrypted on real client rows (reported points, baselines and their matches, DP knee points) with the number of rows; the remaining candidates were measured on test vectors only. In the published runs: 22–25 candidates per workload, of which 8–14 configurations were encrypted on real rows (105–5,000 rows each; 200 MNIST digits).
- The MNIST headroom figures in `abstract_data` (digits "at risk") are **inferred** from plaintext logits and the nominal headroom, not confirmed by encrypting every digit (`python -m benchmark_suite.diagnostics mnist` does that).
- `failures` — per workload (grouped into development, held-out and MNIST): candidates that failed on the synthetic test vectors (catastrophic deviations and other failures), the step-downs they triggered, and the largest catastrophic rate of the reported configurations on real client rows.
- `latency_vs_prime_size` — within each (`N`, number of primes) group of the reference grid, configurations differ only in their prime sizes; the spread of their latencies (and its rank correlation with the scale) measures how much prime size matters. In earlier runs the median spread was 2.5% and the largest 7%, against 30-60% per additional prime.
- `dp_robustness` — the tool run with an ε-differentially private descriptor (ε = 1 and 0.1; public bounds simulated by each feature's range over the whole dataset): output bound, the knee point per security level, whether it equals the one chosen with the exact descriptor, and its precision, prediction agreement and catastrophic rate on real rows. MNIST is skipped.

**Measurement tolerances.** `runtime_tolerance` (τ_t) and `precision_tolerance_bits` (τ_p) are set from the noise study below: twice the 90th percentile, over the reported configurations, of the coefficient of variation of the latency and of the standard deviation of the precision across repeated measurements with fresh keys (published runs: τ_t = 1.04%, τ_p = 2.53 bits). They are used by the knee and endpoint selection, ε-recall and the near-front check; the hypervolume floor is the fixed 5% above; the baseline comparisons in `baseline.csv` are strict (no tolerance). Back-to-back repeats underestimate drift over a long run, so τ_t is a lower bound on timing noise.

**Measurement noise.** `python -m benchmark_suite.noise --datasets ... --repeats 5` measures the configurations the tool reports several times, each with new keys and encryption noise, and suggests `runtime_tolerance` and `precision_tolerance_bits` as twice the 90th percentile of the configurations' coefficients of variation and standard deviations (the maxima are reported too; with 5 repetitions a single outlier draw would otherwise set the tolerance); `--write-config` merges them into an experiment config.

**Evaluation order.** Candidates are evaluated in a seeded random order (`evaluation.random_state`). In scale-sorted order, drift of the machine over a long run (temperature, clock frequency) showed up as a spurious correlation between scale and latency. With 10 test vectors the precision varied by a median of 0.25 bits and at most 1.8 bits across key draws, and the runtime by at most 3.6%; hence the default of 50 test vectors and the tolerances of 5% and 1.5 bits.

**Diagnostics.** `python -m benchmark_suite.diagnostics {coefficients,dp,scales,rules,mnist} --results <run>` checks the tool's assumptions on an existing run without changing its search or metrics: the largest plaintext coefficient against the largest output (`coefficients`), the output bound under many differentially private descriptors (`dp`), what an exact (1-bit) largest scale would add (`scales`), how the rules the reference grid shares with the tool shape its front (`rules`), and an encrypted check of the MNIST digits whose logits exceed a configuration's headroom (`mnist`). Results go to `results/diagnostics/`.

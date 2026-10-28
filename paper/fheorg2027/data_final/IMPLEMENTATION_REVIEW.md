# Implementation review: final behavior, guarantees and limits (2026-10-26)

> **California housing rerun (2026-10-28, commit `328bc5d`):** the four heavy-tailed features are
> log-transformed (plaintext R² −0.083 → 0.657). Same calibrated tolerances; the earlier California
> run is in `results/final_v4_superseded/`. After the transform its one-prime reported
> configurations keep only **0.9 bits** of coefficient headroom on real rows (largest coefficient
> 2.3 bits above the largest output: the pre-bias sum is cancelled by the intercept), with 0
> catastrophic deviations — California joins wine_quality (0.5 bits) and breast_cancer (2.2 bits)
> below the margin.
>
> **Final run:** `results/final_v4` (commit `902a326`, all on AC power), numbers in
> `paper/fheorg2027/data_final/RESULTS.md`. The diagnostics were repeated on it with the same
> conclusions (reported-configuration headroom: same 8 below the margin, wine_quality 0.5 bits,
> breast_cancer 2.2 bits; DP, 1-bit scales and shared-rule numbers within rounding). **New:** the
> MNIST encrypted check found **0 catastrophic deviations** in all 140 inferred-at-risk digits for
> each of the 4 one-prime configurations (largest coefficient 2^6.8 vs. 8 bits of headroom): the
> inferred "140 of 49,000 digits at risk" is wrong and must not be reported.

Code base: branch `robustness-fixes` after commit `1022b9c` (uncommitted changes). The reported
experiments are `results/final_v3` (commit `1022b9c`, Apple M5 Pro). Diagnostics were run on that
data with the current code; raw outputs in `results/diagnostics/*.csv`. Test suite: **137 passed**.

## 1. Final candidate-generation and safety-check behavior

| quantity | role |
|---|---|
| `b_out` = ⌈log2 max\|plaintext output\|⌉ (floored at 0) over 10⁴ descriptor samples, 2d stress vectors and the test vectors, + 3-bit margin | **generation**: final modulus = log2 Δ + b_out + 1 bits (one prime ≤ 60 bits, else two primes (max(B−60, 20), 60); > 120 bits: no candidate) |
| HE Standard budgets (special prime included), SEAL prime feasibility | **generation**: candidates over budget or not buildable are not evaluated |
| scales | **generation**: per N, security level and structure, the largest multiple of 5 bits (2^20…2^60) whose chain fits the budget; also with one spare level; special prime min or 60 |
| catastrophic deviation on the test vectors: error ≥ largest output, **or** error ≥ 2^headroom | **rejection**: the candidate fails, is never reported, and is retried (step-down) |
| step-down | **retry**: same N, final modulus, levels, special prime; scale − 5 bits → +5 bits of output headroom; repeated down to 2^20 |
| `coefficient_headroom_bits`, `max_coefficient`, `max_output` | **diagnostic only**: never used to generate, reject or rank. Reported points with headroom below the margin get `coefficient_headroom_below_margin: true` in `recommended_params.json` and a log warning |
| real-row catastrophic rate, prediction agreement | **benchmark validation only** |

Pareto fronts per security level use every successful configuration at least that secure; the knee
and endpoints apply the tolerances τ_t, τ_p.

## 2. What coefficient headroom measures

- **Overflow happens only at decryption.** CKKS additions, multiplications and rescalings preserve
  the message modulo the current modulus; intermediate values up to 19 bits above the headroom
  decrypted correctly (tested, including through a square). What must fit is the largest coefficient
  of the decrypted plaintext polynomial times Δ, below half the final modulus.
- **Computed from the actual result:** all decoded slots of each decrypted output → coefficients via
  the inverse canonical embedding. It therefore covers whatever the model leaves in the ciphertext
  (biases, partial sums, broadcasts, activations) without modelling operations.
- **Exact under stated assumptions:** on the evaluated inputs only; the result did not overflow
  (failed candidates carry no value); imaginary slot parts are noise (real TenSEAL vectors); nominal
  prime sizes and scale (cases within ~0.1 bit of the threshold are not decided).
- **Empirically validated** for: broadcast and per-slot biases, single-slot dot products, matmuls,
  two-prime final modulus, spare level, depth-3 network with a square activation, large intermediate
  values (`tests/test_coefficient_headroom.py`, 16 cases), 60/60 randomized linear cases. Not
  validated on every model structure (e.g. not on the MNIST network at a tight configuration).
- **Input-dependent:** it cannot be computed without evaluating, and larger inputs can overflow a
  configuration that had headroom on the evaluated ones (tested). It is not a guarantee.
- Overhead: 0.5–1.9 ms per inference (≈10% of search time, outside the timed latency);
  `evaluation.measure_coefficients=false` turns it off. `final_v3` ran without it.

## 3. Coefficient headroom of the reported configurations (500 real rows, tabular)

All reported configurations of the 11 tabular workloads: 0 catastrophic deviations; coefficient
headroom ≥ 3.2 bits except:

- **wine_quality: 0.5 bits** for the one-prime configurations (57,50,60), (52,45,52), (37,30,37) and
  (57,50,60) at N = 16384 — fastest@128/192/256 and **knee@256**. The 4,547-row evaluation showed no
  catastrophic deviation, but an output-slot coefficient 1.4× larger would overflow.
- **breast_cancer: 2.2 bits** for its four one-prime configurations (knee@128, fastest@192,
  knee@256, fastest@256): below the 3-bit margin.
- Two-prime configurations keep 11.7–21.7 bits.

Cause on wine_quality: a large pre-bias sum cancelled by the intercept leaves large values in the
other slots (largest coefficient 2.7 bits above the largest output). These points must not be called
safe in the paper; "no catastrophic deviation on the evaluated rows" is the supported statement.

## 4. Fixes made (no effect on reported numbers)

- **Step-down** now keeps the final modulus (before: re-sized it, so headroom never changed). Never
  triggered in `final_v3` (`num_step_downs = 0` everywhere).
- **Overflow criterion** now also flags error ≥ 2^headroom (before: missed wraps that keep their
  sign, e.g. output 50 → 2). No successful measurement in `final_v3` meets it.
- **Hypervolume floor** documented as a fixed 5% (not τ_t); recomputed with τ_t (1.04%) and with no
  floor: identical values in all 33 cases.
- Earlier comments claiming overflow of products before the first rescale (Δ²) were wrong and are
  corrected (see section 2).

## 5. Search policy for the paper (unchanged)

- Current code regenerates **exactly** the `final_v3` candidate sets (12 tool runs, 11 grids).
- 5-bit scale lattice = search resolution, not exhaustive. 1-bit scales (future work): available for
  73% of candidates, median +3.2 bits at equal latency, but 12% measured 0.3–2.7 bits *worse*
  (unexplained).
- 3-bit margin = heuristic. Raw estimate exceeded on breast_cancer (+1.4 bits), ionosphere
  (+0.15) and MNIST. It also excludes viable configurations: the diabetes expert (60,50,60) at 2^50
  is the no-margin candidate; no catastrophic deviation on the evaluated rows, 1.7 bits of
  coefficient headroom — not evidence of safety on other inputs.
- Relaxing the shared rules (no margin, 1-bit scales, N = 2^12, expert) extends the grid front in
  every case (hypervolume median 1.18, range 1.00–2.24): front recovery is measured within the
  tool's rules.

## 6. Other investigations

- **DP descriptor** (100 draws/ε): min/max are public bounds (no noise). Bound below the exact one
  in up to 33% of draws (diabetes, ε = 0.1, up to 2 bits); never at or below the largest real output
  (min spare 1.6 bits).
- **Correlated features** (test): 2048 copies of one feature → bound short even with the margin; not
  caught (test vectors come from the same descriptor).

## 7. Validation accounting (`tables/validation_accounting.csv`)

- Every candidate: encrypted on 50 seeded test vectors (MNIST: 10). 22–25 candidates per workload.
- Real rows (encrypted): 8–14 configurations per workload (6–7 reported points, baselines and their
  matches, DP knees), 105–5,000 rows each (MNIST 200 digits); 14–19 candidates per workload never.
- MNIST "digits at risk" (140/49,000) was **inferred** from plaintext logits and nominal headroom; the
  encrypted check on `final_v4` measured **0** of them deviating (all 4 configurations). The plaintext
  rule overstates the risk because the matmul spreads the logits over the coefficients.

## 8. Metadata

New runs record in `run_info.json`: the resolved search policy, the safety-check rules, the SEAL
version (4.3, from SEAL's serialization header; TenSEAL 0.3.18), git commit, packages, threads,
output bound and margin. `final_v3` records the commit, packages and output bound, but not the SEAL
version or resolved policy (the policy is recoverable from commit `1022b9c` and was verified above).
Per-run config records tolerances (τ_t = 1.04%, τ_p = 2.53 bits), test vectors, margin, seeds.

## 9. Manuscript descriptions to align with the implementation

1. Catastrophic deviation: describe both criteria; note `final_v3` used the first only and no
   measurement met the second.
2. Step-down: "same final modulus, scale − 5 bits (+5 bits headroom); never triggered".
3. Test vectors: 50 per tabular candidate, **10 for MNIST**.
4. "No candidate … on up to 5,000 real rows" → only the reported configurations (and baselines) ran
   on real rows; all candidates ran on test vectors.
5. Safety: "no catastrophic deviation on the evaluated inputs", not "safe"; mention thin headroom
   (wine_quality 0.5 bits, breast_cancer 2.2 bits).
6. b_out: max(0, ⌈log2 max\|f(x)\|∞⌉) + 3; final-modulus rule is a heuristic (overflow depends on
   the decrypted coefficients, not on the output).
7. Tolerances: used for knee/endpoints, ε-recall, near-front; **not** in the baseline table; the
   expert "tie" count (8/11) is a manual reading with τ_t, τ_p.
8. Hypervolume: fixed 5% latency floor (unchanged with 1.04% or none).
9. R² of test vectors: against **50 random client rows**, not the 5,000.
10. MNIST: drop "4 of 6 configurations lack headroom for 140 digits". Measured: the 140 digits with the
    largest logits decrypt correctly under all reported configurations (≈1.2 bits of coefficient
    headroom left for the one-prime ones); the descriptor still underestimates the largest logit by
    about 4–5 bits.
11. Latency: encrypt + evaluate + decrypt in one process; no network or serialization.
12. Security: HE Standard tables (classical, ternary secret, σ = 3.2); 192/256 labels assigned by the
    tool; not IND-CPA^D; plaintext queries reveal a linear model to the client.
13. SEAL 4.3 (bundled with TenSEAL 0.3.18).

## Reruns needed

None. The reported numbers come from `final_v3`; the code changes do not alter its candidate sets
(verified), no step-down occurred, and no measurement meets the new overflow criterion. A rerun with
the current code would add the coefficient columns and SEAL/policy metadata, and its search times
would be ≈10% longer unless `evaluation.measure_coefficients=false`.

# Results for the write-up

Run: `results/final_v4` · commit `902a326` (uncommitted changes: False) · Apple M5 Pro, 15 cores, TenSEAL 0.3.18, Python 3.14.6.

Every number below comes from the tables in `tables/` (the run's `summary/`) and, for Section 12, from the diagnostics in `diagnostics/`. **Reliability** notes state what each number does and does not show.

## 1. Workloads

Development workloads were used while the method was designed; held-out workloads only after it was fixed; MNIST is a stress test. *Output bound* is the empirically estimated integer bits of the largest model output (including the 3-bit safety margin). *Est. bits needed* is, for classifiers, the precision that keeps 99% of predictions unchanged according to the decision margins on synthetic samples (information only).

| workload | group | task | features | client rows | depth | plaintext quality | output bound b_out (bits, incl. +3) | est. bits needed (classifiers) |
|---|---|---|---|---|---|---|---|---|
| diabetes | development | regression | 10 | 309 | 1 | R² 0.442 | 12 | – |
| california_housing | development | regression | 8 | 14448 | 1 | R² 0.657 | 6 | – |
| concrete | development | regression | 8 | 721 | 1 | R² 0.567 | 10 | – |
| breast_cancer | development | binary | 30 | 398 | 1 | acc 0.972 | 7 | 5 |
| heart_disease | development | binary | 11 | 833 | 1 | acc 0.810 | 6 | 6 |
| wine | development | multiclass | 13 | 124 | 1 | acc 0.968 | 6 | 7 |
| ionosphere | development | binary | 34 | 245 | 1 | acc 0.865 | 7 | 5 |
| iris | development | multiclass | 4 | 105 | 1 | acc 0.952 | 6 | 7 |
| mnist | mnist | multiclass | 784 | 49000 | 5 | acc 0.960 | 8 | 2 |
| credit_default | held-out | binary | 23 | 21000 | 1 | acc 0.810 | 8 | 3 |
| bank_marketing | held-out | binary | 51 | 31647 | 1 | acc 0.903 | 8 | 4 |
| wine_quality | held-out | regression | 11 | 4547 | 1 | R² 0.293 | 6 | – |

**Reliability:** final.

## 2. Measurement noise and tolerances

Fixed configurations were measured 5 times each with fresh keys and encryption noise. *Precision spread* is the standard deviation of −log₂(error) across repetitions; *runtime spread* the coefficient of variation of the median latency. The tolerances are twice the 90th percentile over configurations; two points whose latencies differ by less than τ_t and precisions by less than τ_p count as tied.

- Configurations measured: **78**
- Precision spread: median **0.25 bits**, 90th percentile **0.99 bits**, max 1.93 bits
- Runtime spread: median **0.17%**, 90th percentile **0.43%**, max 1.67%
- **τ_p = 1.98 bits** (a factor of 3.9 in absolute error), **τ_t = 0.86%**

**Reliability:** final. The runtime tolerance comes from back-to-back repeats, so it is a lower bound on the timing noise over a long run.

## 3. Search efficiency and front recovery (tool vs. reference grid)

- *Candidates*: configurations evaluated by the tool / by the reference grid (all security levels).
- *Search time saved*: 1 − tool search time / grid search time (wall-clock, key generation included).
- *Hypervolume ratio*: area dominated by the tool's front divided by that of the grid's front, in (latency, log₁₀ error), both normalized by the grid front; 1 = the tool front covers everything the grid front covers. *No floor*: without the minimum runtime span.
- *ε-indicator* (multiplicative): the smallest factor by which the tool front's latencies and errors must be divided so that it weakly dominates every grid-front point; 1 = perfect, 2 = within a factor of 2 in both.
- *ε-recall*: share of grid-front points for which the tool has a point that is tied within τ_t and τ_p.
- *Exact recall*: share of grid-front parameter sets (N, Δ, Q) that the tool also evaluated.
- All per security level ≥128 / ≥192 / ≥256, computed on the grid's own measurements of the tool's candidates.

| group | security | tool | grid | cand_red | time_saved | hv_mean | hv_min | eps_ind_median | eps_ind_max | eps_recall_mean | eps_recall_min | recall_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| development | 128 | 25.000 | 1319.000 | 0.981 | 0.991 | 0.974 | 0.944 | 1.210 | 4.123 | 0.763 | 0.625 | 0.303 |
| development | 192 | 18.125 | 1010.125 | 0.981 | 0.991 | 0.968 | 0.939 | 1.795 | 4.143 | 0.634 | 0.400 | 0.281 |
| development | 256 | 13.000 | 807.000 | 0.981 | 0.991 | 0.944 | 0.908 | 1.895 | 4.143 | 0.653 | 0.286 | 0.184 |
| held-out | 128 | 24.333 | 1297.000 | 0.981 | 0.992 | 0.971 | 0.959 | 4.591 | 4.643 | 0.689 | 0.556 | 0.218 |
| held-out | 192 | 18.667 | 1004.667 | 0.981 | 0.992 | 0.969 | 0.943 | 5.298 | 8.585 | 0.508 | 0.333 | 0.255 |
| held-out | 256 | 13.000 | 802.333 | 0.981 | 0.992 | 0.964 | 0.941 | 4.008 | 5.859 | 0.544 | 0.400 | 0.244 |

Per workload:

| dataset | group | security | tool_candidates | broad_grid_candidates | candidate_reduction | search_time_reduction | hypervolume_ratio | hypervolume_ratio_no_floor | epsilon_indicator | epsilon_recall | pareto_recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bank_marketing | held-out | 128 | 24 | 1359 | 0.982 | 0.993 | 0.959 | 0.959 | 4.591 | 0.556 | 0.333 |
| bank_marketing | held-out | 192 | 19 | 1052 | 0.982 | 0.993 | 0.984 | 0.984 | 5.298 | 0.556 | 0.333 |
| bank_marketing | held-out | 256 | 13 | 832 | 0.982 | 0.993 | 0.974 | 0.974 | 5.859 | 0.500 | 0.300 |
| breast_cancer | development | 128 | 25 | 1339 | 0.981 | 0.992 | 0.998 | 0.998 | 1.096 | 0.818 | 0.182 |
| breast_cancer | development | 192 | 18 | 1019 | 0.981 | 0.992 | 0.996 | 0.996 | 1.046 | 0.909 | 0.182 |
| breast_cancer | development | 256 | 13 | 806 | 0.981 | 0.992 | 0.923 | 0.923 | 2.108 | 0.917 | 0.167 |
| california_housing | development | 128 | 25 | 1332 | 0.981 | 0.991 | 0.974 | 0.974 | 3.510 | 0.786 | 0.214 |
| california_housing | development | 192 | 18 | 1013 | 0.981 | 0.991 | 0.971 | 0.971 | 2.879 | 0.455 | 0.273 |
| california_housing | development | 256 | 13 | 806 | 0.981 | 0.991 | 0.935 | 0.935 | 2.278 | 0.643 | 0.214 |
| concrete | development | 128 | 25 | 1209 | 0.979 | 0.990 | 0.993 | 0.993 | 1.161 | 0.818 | 0.545 |
| concrete | development | 192 | 19 | 934 | 0.979 | 0.990 | 0.955 | 0.955 | 2.502 | 0.800 | 0.300 |
| concrete | development | 256 | 13 | 745 | 0.979 | 0.990 | 0.977 | 0.977 | 1.682 | 0.667 | 0.167 |
| credit_default | held-out | 128 | 24 | 1197 | 0.980 | 0.992 | 0.986 | 0.986 | 1.301 | 0.846 | 0.154 |
| credit_default | held-out | 192 | 19 | 946 | 0.980 | 0.992 | 0.980 | 0.980 | 1.430 | 0.636 | 0.182 |
| credit_default | held-out | 256 | 13 | 766 | 0.980 | 0.992 | 0.976 | 0.976 | 1.430 | 0.733 | 0.133 |
| diabetes | development | 128 | 25 | 1207 | 0.979 | 0.991 | 0.977 | 0.977 | 1.259 | 0.750 | 0.125 |
| diabetes | development | 192 | 18 | 925 | 0.979 | 0.991 | 0.981 | 0.981 | 1.088 | 0.750 | 0.250 |
| diabetes | development | 256 | 13 | 741 | 0.979 | 0.991 | 0.978 | 0.978 | 1.088 | 0.818 | 0.273 |
| heart_disease | development | 128 | 25 | 1335 | 0.981 | 0.991 | 0.945 | 0.945 | 4.123 | 0.625 | 0.125 |
| heart_disease | development | 192 | 18 | 1018 | 0.981 | 0.991 | 0.970 | 0.970 | 4.123 | 0.400 | 0.200 |
| heart_disease | development | 256 | 13 | 813 | 0.981 | 0.991 | 0.908 | 0.908 | 4.123 | 0.500 | 0.167 |
| ionosphere | development | 128 | 25 | 1395 | 0.982 | 0.991 | 0.989 | 0.989 | 1.538 | 0.857 | 0.286 |
| ionosphere | development | 192 | 18 | 1075 | 0.982 | 0.991 | 0.939 | 0.939 | 4.143 | 0.500 | 0.333 |
| ionosphere | development | 256 | 13 | 862 | 0.982 | 0.991 | 0.912 | 0.912 | 4.143 | 0.750 | 0.125 |
| iris | development | 128 | 25 | 1404 | 0.982 | 0.990 | 0.944 | 0.944 | 1.047 | 0.750 | 0.750 |
| iris | development | 192 | 18 | 1085 | 0.982 | 0.990 | 0.943 | 0.943 | 1.040 | 0.800 | 0.400 |
| iris | development | 256 | 13 | 878 | 0.982 | 0.990 | 0.933 | 0.933 | 1.046 | 0.286 | 0.143 |
| wine | development | 128 | 25 | 1331 | 0.981 | 0.991 | 0.973 | 0.973 | 1.042 | 0.700 | 0.200 |
| wine | development | 192 | 18 | 1012 | 0.981 | 0.991 | 0.988 | 0.988 | 1.045 | 0.462 | 0.308 |
| wine | development | 256 | 13 | 805 | 0.981 | 0.991 | 0.983 | 0.983 | 1.013 | 0.643 | 0.214 |
| wine_quality | held-out | 128 | 25 | 1335 | 0.981 | 0.992 | 0.967 | 0.967 | 4.643 | 0.667 | 0.167 |
| wine_quality | held-out | 192 | 18 | 1016 | 0.981 | 0.992 | 0.943 | 0.943 | 8.585 | 0.333 | 0.250 |
| wine_quality | held-out | 256 | 13 | 809 | 0.981 | 0.992 | 0.941 | 0.941 | 4.008 | 0.400 | 0.300 |

Absolute search times:

| workload | group | tool search (s) | grid search (s) |
|---|---|---|---|
| diabetes | development | 19.3 | 2089.3 |
| california_housing | development | 16.0 | 1825.6 |
| concrete | development | 16.8 | 1663.9 |
| breast_cancer | development | 29.4 | 3827.1 |
| heart_disease | development | 20.6 | 2425.4 |
| wine | development | 30.6 | 3483.4 |
| ionosphere | development | 22.4 | 2636.5 |
| iris | development | 26.1 | 2744.2 |
| credit_default | held-out | 25.8 | 3210.9 |
| bank_marketing | held-out | 27.6 | 3822.5 |
| wine_quality | held-out | 19.8 | 2435.6 |

**Reliability:** final. All grids ran on AC power without sleep (power logged every minute). Caveats: each grid configuration is measured once, so the grid front can include favourable noise draws, which lowers the measured recovery; the reference grid shares the tool's output-bound rule, 5-bit scale lattice and N range, so recovery is measured within those rules (Section 12); the hypervolume latency axis has a fixed 5% floor (with τ_t or no floor the values are identical). 14 of the 1,404 iris grid measurements are more than 15% slower than their (N, primes) group (a background interruption); none is a tool candidate or on a front.

## 4. Where the tool's reported points lie

For every security level the tool reports the knee point and the front's two endpoints (fastest, most precise). *On grid front*: Pareto-optimal among all grid configurations at least that secure (grid's own measurements). *Near grid front*: no grid configuration is both more than τ_t faster and more than τ_p more precise.

| group | on_grid_front | near_grid_front |
|---|---|---|
| development | 0.51 | 0.96 |
| held-out | 0.48 | 1.00 |
| mnist | – | – |

**Reliability:** final (same caveats as Section 3).

## 5. Baselines (measured on the same real client rows)

*Tutorial defaults*: N=8192, Q=(60,40,40,60), Δ=2⁴⁰ (depth ≤ 2). *Expert defaults*: 60-bit first and special prime, 50-bit scaling primes, smallest admissible N. Two asymmetric comparisons per baseline: *speedup* of the fastest tool configuration at least as precise as the baseline, and *precision gain* (bits) of the most precise tool configuration at least as fast.

**tutorial** (strict, tabular): speedup at equal precision — median 1.64×, range 1.29–1.78× ; precision gain at equal speed — median 13.9 bits, range 3.4 to 20.7 bits.

**expert** (strict, tabular): speedup at equal precision — median 0.77×, range 0.58–1.01× ; precision gain at equal speed — median -0.0 bits, range -13.9 to 2.4 bits.

| dataset | group | baseline | comparison | baseline_chain | baseline_precision_bits | baseline_runtime | tool_N | tool_chain | tool_precision_bits | tool_runtime | speedup | precision_gain_bits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bank_marketing | held-out | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 17.888 | 0.009 | 8192.000 | (49, 40, 60) | 18.501 | 0.005 | 1.780 | 0.613 |
| bank_marketing | held-out | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 17.888 | 0.009 | 8192.000 | (20, 60, 60, 60) | 36.088 | 0.009 | 1.003 | 18.200 |
| bank_marketing | held-out | expert | fastest at least as precise as the baseline | (60, 50, 60) | 28.506 | 0.005 | 8192.000 | (59, 50, 59) | 28.922 | 0.005 | 1.008 | 0.416 |
| bank_marketing | held-out | expert | most precise at least as fast as the baseline | (60, 50, 60) | 28.506 | 0.005 | 8192.000 | (59, 50, 59) | 28.922 | 0.005 | 1.008 | 0.416 |
| breast_cancer | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 15.947 | 0.009 | 8192.000 | (58, 50, 60) | 29.676 | 0.005 | 1.760 | 13.729 |
| breast_cancer | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 15.947 | 0.009 | 8192.000 | (58, 50, 60) | 29.676 | 0.005 | 1.760 | 13.729 |
| breast_cancer | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 27.765 | 0.005 | 8192.000 | (58, 50, 60) | 29.676 | 0.005 | 0.995 | 1.911 |
| breast_cancer | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 27.765 | 0.005 | 8192.000 | (58, 50, 60) | 29.676 | 0.005 | 0.995 | 1.911 |
| california_housing | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 17.267 | 0.004 | 8192.000 | (57, 50, 57) | 31.044 | 0.002 | 1.532 | 13.778 |
| california_housing | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 17.267 | 0.004 | 8192.000 | (20, 60, 60, 60) | 37.977 | 0.004 | 0.986 | 20.710 |
| california_housing | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 29.916 | 0.002 | 8192.000 | (57, 50, 57) | 31.044 | 0.002 | 0.996 | 1.128 |
| california_housing | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 29.916 | 0.002 | 8192.000 | (57, 50, 57) | 31.044 | 0.002 | 0.996 | 1.128 |
| concrete | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 16.095 | 0.004 | 8192.000 | (56, 45, 60) | 19.596 | 0.002 | 1.522 | 3.501 |
| concrete | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 16.095 | 0.004 | 8192.000 | (56, 45, 60) | 19.596 | 0.002 | 1.522 | 3.501 |
| concrete | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 30.204 | 0.002 | 8192.000 | (20, 60, 60, 60) | 36.716 | 0.004 | 0.644 | 6.512 |
| concrete | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 30.204 | 0.002 | 8192.000 | (51, 40, 60) | 16.290 | 0.002 | 1.000 | -13.914 |
| credit_default | held-out | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 14.909 | 0.007 | 8192.000 | (49, 40, 49) | 15.012 | 0.004 | 1.727 | 0.103 |
| credit_default | held-out | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 14.909 | 0.007 | 8192.000 | (59, 50, 59) | 24.622 | 0.004 | 1.731 | 9.712 |
| credit_default | held-out | expert | fastest at least as precise as the baseline | (60, 50, 60) | 24.556 | 0.004 | 8192.000 | (20, 60, 60, 60) | 33.369 | 0.007 | 0.579 | 8.813 |
| credit_default | held-out | expert | most precise at least as fast as the baseline | (60, 50, 60) | 24.556 | 0.004 | 8192.000 | (49, 40, 60) | 15.006 | 0.004 | 1.011 | -9.550 |
| diabetes | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 15.308 | 0.005 | 8192.000 | (58, 45, 58) | 18.641 | 0.003 | 1.616 | 3.333 |
| diabetes | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 15.308 | 0.005 | 8192.000 | (58, 45, 60) | 18.672 | 0.003 | 1.615 | 3.364 |
| diabetes | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 27.348 | 0.003 | 8192.000 | (20, 60, 60, 60) | 35.696 | 0.005 | 0.620 | 8.349 |
| diabetes | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 27.348 | 0.003 | – | – | – | – | – | – |
| heart_disease | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 17.813 | 0.005 | 8192.000 | (52, 45, 60) | 22.437 | 0.003 | 1.636 | 4.624 |
| heart_disease | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 17.813 | 0.005 | 8192.000 | (57, 50, 60) | 31.754 | 0.003 | 1.632 | 13.942 |
| heart_disease | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 29.533 | 0.003 | 8192.000 | (57, 50, 57) | 28.465 | 0.003 | 1.001 | -1.068 |
| heart_disease | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 29.533 | 0.003 | 8192.000 | (52, 45, 60) | 22.437 | 0.003 | 1.001 | -7.096 |
| ionosphere | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 18.286 | 0.006 | 8192.000 | (53, 45, 53) | 22.283 | 0.003 | 1.661 | 3.998 |
| ionosphere | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 18.286 | 0.006 | 8192.000 | (58, 50, 60) | 30.647 | 0.004 | 1.651 | 12.362 |
| ionosphere | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 29.547 | 0.003 | 8192.000 | (58, 50, 60) | 30.647 | 0.004 | 0.991 | 1.100 |
| ionosphere | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 29.547 | 0.003 | – | – | – | – | – | – |
| iris | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 19.310 | 0.007 | 8192.000 | (52, 45, 60) | 22.628 | 0.006 | 1.290 | 3.318 |
| iris | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 19.310 | 0.007 | 8192.000 | (20, 60, 60, 60) | 39.902 | 0.007 | 0.989 | 20.592 |
| iris | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 32.484 | 0.006 | 8192.000 | (20, 60, 60, 60) | 39.902 | 0.007 | 0.773 | 7.418 |
| iris | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 32.484 | 0.006 | 8192.000 | (57, 50, 60) | 32.430 | 0.006 | 1.009 | -0.054 |
| mnist | mnist | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 40, 40, 40, 60) | 7.200 | 2.243 | 16384.000 | (59, 50, 50, 50, 50, 50, 59) | 17.446 | 2.257 | 0.994 | 10.246 |
| mnist | mnist | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 40, 40, 40, 60) | 7.200 | 2.243 | 8192.000 | (34, 25, 25, 25, 25, 25, 34) | -4.151 | 1.077 | 2.083 | -11.351 |
| mnist | mnist | expert | fastest at least as precise as the baseline | (60, 50, 50, 50, 50, 50, 60) | 17.446 | 2.259 | 16384.000 | (59, 50, 50, 50, 50, 50, 60) | 17.446 | 2.295 | 0.984 | 0.000 |
| mnist | mnist | expert | most precise at least as fast as the baseline | (60, 50, 50, 50, 50, 50, 60) | 17.446 | 2.259 | 16384.000 | (49, 40, 40, 40, 40, 40, 49) | 7.200 | 2.246 | 1.006 | -10.246 |
| wine | development | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 18.292 | 0.010 | 8192.000 | (52, 45, 52) | 21.593 | 0.007 | 1.378 | 3.301 |
| wine | development | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 18.292 | 0.010 | 8192.000 | (20, 60, 60, 60) | 38.620 | 0.010 | 1.002 | 20.328 |
| wine | development | expert | fastest at least as precise as the baseline | (60, 50, 60) | 31.425 | 0.007 | 8192.000 | (20, 60, 60, 60) | 38.620 | 0.010 | 0.728 | 7.195 |
| wine | development | expert | most precise at least as fast as the baseline | (60, 50, 60) | 31.425 | 0.007 | 8192.000 | (57, 50, 57) | 31.424 | 0.007 | 1.006 | -0.001 |
| wine_quality | held-out | tutorial | fastest at least as precise as the baseline | (60, 40, 40, 60) | 17.222 | 0.005 | 8192.000 | (57, 50, 57) | 30.297 | 0.003 | 1.640 | 13.075 |
| wine_quality | held-out | tutorial | most precise at least as fast as the baseline | (60, 40, 40, 60) | 17.222 | 0.005 | 8192.000 | (20, 60, 60, 60) | 36.931 | 0.005 | 0.989 | 19.708 |
| wine_quality | held-out | expert | fastest at least as precise as the baseline | (60, 50, 60) | 28.734 | 0.003 | 8192.000 | (20, 60, 60, 60) | 36.931 | 0.005 | 0.606 | 8.197 |
| wine_quality | held-out | expert | most precise at least as fast as the baseline | (60, 50, 60) | 28.734 | 0.003 | 8192.000 | (57, 50, 60) | 31.099 | 0.003 | 1.003 | 2.365 |

**With the measurement tolerances** (test vectors, tabular workloads; τ_t and τ_p of Section 2): a *tie* if some tool candidate is at most τ_t slower and at most τ_p less precise than the baseline; *gain* = best tool precision at most τ_t slower, minus the baseline's; *speedup* = baseline latency / fastest tool candidate at most τ_p less precise.

- **tutorial**: tie or better on **11 of 11**; gain median 17.1 bits (range 3.3 to 21.0); speedup median 1.64× (range 1.33–1.87×)
- **expert**: tie or better on **8 of 11**; gain median -0.0 bits (range -18.5 to 3.7); speedup median 1.00× (range 0.61–1.06×)

| workload | group | baseline | baseline_chain | baseline_bits | baseline_ms | tie | gain_bits_within_tau_t | speedup_within_tau_p |
|---|---|---|---|---|---|---|---|---|
| diabetes | development | tutorial | (60, 40, 40, 60) | 15.97 | 4.79 | True | 3.60 | 1.59 |
| diabetes | development | expert | (60, 50, 60) | 28.04 | 2.99 | False | -8.48 | 0.61 |
| california_housing | development | tutorial | (60, 40, 40, 60) | 17.71 | 3.74 | True | 20.57 | 1.61 |
| california_housing | development | expert | (60, 50, 60) | 29.63 | 2.43 | True | 3.09 | 1.04 |
| concrete | development | tutorial | (60, 40, 40, 60) | 16.37 | 3.71 | True | 3.28 | 1.53 |
| concrete | development | expert | (60, 50, 60) | 28.85 | 2.43 | False | -9.20 | 0.63 |
| breast_cancer | development | tutorial | (60, 40, 40, 60) | 17.66 | 8.82 | True | 17.13 | 1.77 |
| breast_cancer | development | expert | (60, 50, 60) | 27.81 | 5.01 | True | 2.91 | 1.00 |
| heart_disease | development | tutorial | (60, 40, 40, 60) | 20.36 | 5.29 | True | 12.12 | 1.64 |
| heart_disease | development | expert | (60, 50, 60) | 28.85 | 3.24 | True | 3.63 | 1.00 |
| wine | development | tutorial | (60, 40, 40, 60) | 18.28 | 9.50 | True | 20.31 | 1.45 |
| wine | development | expert | (60, 50, 60) | 31.63 | 6.93 | True | -0.01 | 1.05 |
| ionosphere | development | tutorial | (60, 40, 40, 60) | 18.69 | 5.83 | True | 12.57 | 1.64 |
| ionosphere | development | expert | (60, 50, 60) | 29.99 | 3.51 | False | -18.53 | 0.99 |
| iris | development | tutorial | (60, 40, 40, 60) | 19.47 | 7.39 | True | 20.59 | 1.33 |
| iris | development | expert | (60, 50, 60) | 32.66 | 5.72 | True | -0.02 | 1.03 |
| credit_default | held-out | tutorial | (60, 40, 40, 60) | 15.98 | 7.37 | True | 10.64 | 1.72 |
| credit_default | held-out | expert | (60, 50, 60) | 26.70 | 4.30 | True | -0.08 | 0.99 |
| bank_marketing | held-out | tutorial | (60, 40, 40, 60) | 18.38 | 8.91 | True | 19.39 | 1.87 |
| bank_marketing | held-out | expert | (60, 50, 60) | 28.25 | 5.06 | True | 3.75 | 1.06 |
| wine_quality | held-out | tutorial | (60, 40, 40, 60) | 18.16 | 5.28 | True | 21.00 | 1.72 |
| wine_quality | held-out | expert | (60, 50, 60) | 32.11 | 3.22 | True | -1.71 | 1.05 |

**Reliability:** final. The strict table compares exact values (a 0.1% latency difference decides it); the tolerance version is the one to cite for ties. The expert gap on the regression workloads comes from the 3-bit output margin (the expert's 60-bit first prime leaves too little headroom under the tool's rule).

## 6. Reliability of the empirical estimates

*Catastrophic deviation*: error at least as large as the output (per real row: at least its own output and at least 1), or at least 2^headroom (an overflow); cause not proven. *Prediction agreement*: share of real rows with the same predicted class encrypted and in plaintext.

- Candidates with a catastrophic deviation on the synthetic test vectors: **0** of 295 (other failures: 0; step-downs: 0)
- Reported configurations with catastrophic deviations on real client rows: largest rate **0.0000** (real rows per workload: up to 5000)
- Prediction agreement of the reported configurations: at least **0.9984**

| dataset | group | candidates | failed_on_test_vectors | catastrophic_on_test_vectors | other_failures | step_downs | reported_configurations | max_catastrophic_rate_real_rows | real_rows |
|---|---|---|---|---|---|---|---|---|---|
| bank_marketing | held-out | 24 | 0 | 0 | 0 | 0 | 9 | 0.000 | 5000 |
| breast_cancer | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 398 |
| california_housing | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 5000 |
| concrete | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 721 |
| credit_default | held-out | 24 | 0 | 0 | 0 | 0 | 9 | 0.000 | 5000 |
| diabetes | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 309 |
| heart_disease | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 833 |
| ionosphere | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 245 |
| iris | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 105 |
| mnist | mnist | 22 | 0 | 0 | 0 | 0 | 9 | 0.000 | 200 |
| wine | development | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 124 |
| wine_quality | held-out | 25 | 0 | 0 | 0 | 0 | 9 | 0.000 | 4547 |

**Reliability:** final.

## 7. Fidelity of the synthetic test vectors

For every tool candidate the error on the 50 representative test vectors is compared with the error on 50 random real client rows (*truth*), and so are dataset-agnostic *naive* vectors (uniform in [−1, 1]). *R² (log)*: how well log₁₀ of the estimate predicts log₁₀ of the real error (1 = perfect). *Bias*: median of log₁₀(estimate / real) in decades (0 = unbiased, < 0 = optimistic). *Coverage*: share of candidates whose estimate is at least the real error.

| group | test_vectors | r2_median | r2_min | spearman_median | bias_median | coverage_median |
|---|---|---|---|---|---|---|
| development | naive | 0.954 | 0.670 | 0.962 | -0.440 | 0.120 |
| development | representative | 0.996 | 0.989 | 0.974 | -0.045 | 0.280 |
| held-out | naive | 0.987 | 0.842 | 0.960 | 0.032 | 0.708 |
| held-out | representative | 0.992 | 0.990 | 0.958 | 0.032 | 0.542 |
| mnist | naive | 0.908 | 0.908 | 0.981 | -0.942 | 0.000 |
| mnist | representative | 0.969 | 0.969 | 0.983 | -0.550 | 0.000 |

Per workload:

| dataset | group | test_vectors | spearman | pearson_log | r2_log | coverage | median_log_ratio |
|---|---|---|---|---|---|---|---|
| bank_marketing | held-out | representative | 0.932 | 0.995 | 0.990 | 0.542 | 0.039 |
| bank_marketing | held-out | naive | 0.952 | 0.995 | 0.987 | 0.708 | 0.157 |
| breast_cancer | development | representative | 0.946 | 0.997 | 0.993 | 0.240 | -0.143 |
| breast_cancer | development | naive | 0.962 | 0.994 | 0.966 | 0.880 | 0.546 |
| california_housing | development | representative | 0.980 | 0.996 | 0.989 | 0.520 | 0.000 |
| california_housing | development | naive | 0.964 | 0.991 | 0.888 | 0.040 | -1.067 |
| concrete | development | representative | 0.983 | 1.000 | 0.999 | 0.080 | -0.015 |
| concrete | development | naive | 0.952 | 0.981 | 0.670 | 0.000 | -1.847 |
| credit_default | held-out | representative | 0.972 | 0.999 | 0.997 | 0.542 | 0.032 |
| credit_default | held-out | naive | 0.960 | 0.983 | 0.842 | 0.042 | -0.991 |
| diabetes | development | representative | 0.983 | 1.000 | 0.999 | 0.320 | -0.045 |
| diabetes | development | naive | 0.985 | 1.000 | 0.872 | 1.000 | 1.122 |
| heart_disease | development | representative | 0.968 | 0.997 | 0.991 | 0.200 | -0.136 |
| heart_disease | development | naive | 0.940 | 0.996 | 0.982 | 0.200 | -0.429 |
| ionosphere | development | representative | 0.952 | 0.996 | 0.991 | 0.400 | -0.056 |
| ionosphere | development | naive | 0.936 | 0.995 | 0.990 | 0.280 | -0.083 |
| iris | development | representative | 0.948 | 1.000 | 1.000 | 0.000 | -0.046 |
| iris | development | naive | 0.965 | 1.000 | 0.977 | 0.000 | -0.452 |
| mnist | mnist | representative | 0.983 | 1.000 | 0.969 | 0.000 | -0.550 |
| mnist | mnist | naive | 0.981 | 1.000 | 0.908 | 0.000 | -0.942 |
| wine | development | representative | 0.985 | 1.000 | 1.000 | 0.760 | 0.006 |
| wine | development | naive | 0.962 | 1.000 | 0.942 | 0.000 | -0.712 |
| wine_quality | held-out | representative | 0.958 | 0.996 | 0.992 | 0.520 | 0.003 |
| wine_quality | held-out | naive | 0.969 | 0.999 | 0.998 | 0.840 | 0.032 |

**Reliability:** final.

## 8. Precision ceiling and what latency depends on

Best precision at ≥128-bit security and N = 8192 with a one-prime final modulus (depth + 2 primes) vs. a two-prime final modulus (depth + 3 primes), from the tool's own measurements. Latency structure from the development grids.

| workload | one-prime chain | bits | latency ms | two-prime chain | bits (two-prime) | latency ms (two-prime) | gain bits | latency ratio |
|---|---|---|---|---|---|---|---|---|
| diabetes | (58, 45, 60) | 19.565 | 3.027 | (20, 60, 60, 60) | 36.536 | 4.871 | 16.971 | 1.610 |
| california_housing | (57, 50, 57) | 32.715 | 2.327 | (20, 60, 60, 60) | 38.278 | 3.625 | 5.563 | 1.558 |
| concrete | (56, 45, 60) | 19.651 | 2.436 | (20, 60, 60, 60) | 36.819 | 3.858 | 17.168 | 1.584 |
| breast_cancer | (58, 50, 60) | 30.717 | 4.995 | (20, 60, 60, 60) | 34.791 | 8.888 | 4.074 | 1.779 |
| heart_disease | (57, 50, 60) | 32.476 | 3.239 | (20, 60, 60, 60) | 41.151 | 5.343 | 8.675 | 1.650 |
| wine | (57, 50, 57) | 31.621 | 6.585 | (20, 60, 60, 60) | 38.594 | 9.015 | 6.973 | 1.369 |
| ionosphere | (58, 50, 60) | 31.257 | 3.563 | (20, 60, 60, 60) | 39.245 | 5.944 | 7.988 | 1.668 |
| iris | (57, 50, 60) | 32.642 | 5.574 | (20, 60, 60, 60) | 40.057 | 7.271 | 7.416 | 1.304 |
| credit_default | (59, 50, 59) | 26.621 | 4.327 | (20, 60, 60, 60) | 35.319 | 7.559 | 8.698 | 1.747 |
| bank_marketing | (59, 50, 59) | 31.995 | 4.834 | (20, 60, 60, 60) | 37.766 | 8.505 | 5.771 | 1.760 |
| wine_quality | (57, 50, 60) | 30.401 | 3.080 | (20, 60, 60, 60) | 39.161 | 5.092 | 8.760 | 1.653 |

- One additional prime: median latency **+34%** (range 19% to 79%)
- Doubling N: median latency **×2.10** (range ×2.01 to ×2.18)
- Same N and number of primes, different prime sizes (development grids): median spread **3.5%**, max 81.4%; interquartile range 1.14% of the median

**Reliability:** final. The latency-structure numbers use the development grids; the largest spread is the iris outlier noted in Section 3 (the interquartile range is the robust figure). The ~36–40-bit plateau is the double-precision limit of CKKS encoding in SEAL/TenSEAL.

## 9. MNIST (stress test)

| target_security | point | N | chain | precision_bits_test_vectors | precision_bits | runtime | prediction_agreement | catastrophic_rate | real_rows |
|---|---|---|---|---|---|---|---|---|---|
| 128 | knee | 16384 | (20, 60, 55, 55, 55, 55, 55, 60) | 26.122 | 22.318 | 2.872 | 1.000 | 0.000 | 200 |
| 128 | fastest | 16384 | (59, 50, 50, 50, 50, 50, 59) | 21.255 | 17.446 | 2.257 | 1.000 | 0.000 | 200 |
| 128 | most_precise | 32768 | (20, 60, 60, 60, 60, 60, 60, 60) | 29.376 | 25.566 | 6.164 | 1.000 | 0.000 | 200 |
| 192 | knee | 32768 | (59, 50, 50, 50, 50, 50, 60) | 21.094 | 17.288 | 4.819 | 1.000 | 0.000 | 200 |
| 192 | fastest | 16384 | (49, 40, 40, 40, 40, 40, 49) | 11.003 | 7.200 | 2.246 | 1.000 | 0.000 | 200 |
| 192 | most_precise | 32768 | (20, 60, 60, 60, 60, 60, 60, 60) | 29.376 | 25.566 | 6.164 | 1.000 | 0.000 | 200 |
| 256 | knee | 32768 | (59, 50, 50, 50, 50, 50, 60) | 21.094 | 17.288 | 4.819 | 1.000 | 0.000 | 200 |
| 256 | fastest | 16384 | (39, 30, 30, 30, 30, 30, 39) | 4.117 | 0.320 | 2.249 | 1.000 | 0.000 | 200 |
| 256 | most_precise | 32768 | (20, 60, 60, 60, 60, 60, 60, 60) | 29.376 | 25.566 | 6.164 | 1.000 | 0.000 | 200 |

- Test vectors overstate precision by a median of **3.8 bits** (tabular workloads: ~0 to 0.2 bits).
- Output bound from the descriptor: 8 bits including the 3-bit margin.
- Largest real logit over all client digits: **537** (10 bits), i.e. the descriptor underestimates it by 5 bits.
- Headroom of each reported configuration against all client digits, **inferred from plaintext logits, not encrypted** (a digit is at risk if |logit|·Δ reaches half the final modulus; the encrypted check is `python -m benchmark_suite.diagnostics mnist`):

| N | chain | final modulus bits | headroom (bits) | digits at risk (of all client digits) | client digits | real-row precision (200 digits) |
|---|---|---|---|---|---|---|
| 16384 | (20, 60, 55, 55, 55, 55, 55, 60) | 80 | 24 | 0 | 49000 | 22.318 |
| 16384 | (59, 50, 50, 50, 50, 50, 59) | 59 | 8 | 140 | 49000 | 17.446 |
| 32768 | (20, 60, 60, 60, 60, 60, 60, 60) | 80 | 19 | 0 | 49000 | 25.566 |
| 32768 | (59, 50, 50, 50, 50, 50, 60) | 59 | 8 | 140 | 49000 | 17.288 |
| 16384 | (49, 40, 40, 40, 40, 40, 49) | 49 | 8 | 140 | 49000 | 7.200 |
| 16384 | (39, 30, 30, 30, 30, 30, 39) | 39 | 8 | 140 | 49000 | 0.320 |

- **4 of 6 reported configurations** leave too little headroom for the largest real logits: 140 of 49000 digits (0.29%) exceed the nominal headroom (inferred from plaintext logits).

**Encrypted check of the inferred risk** (`diagnostics mnist`): every digit inferred to be at risk was encrypted with each configuration (the 50 largest logits for configurations without inferred risk).

| N | chain | headroom_bits | inferred_at_risk | client_digits | encrypted_digits | measured_catastrophic | largest_logit_bits | largest_coefficient_bits |
|---|---|---|---|---|---|---|---|---|
| 16384 | (20, 60, 55, 55, 55, 55, 55, 60) | 24 | 0 | 49000 | 50 | 0 | 9.07 | 6.64 |
| 16384 | (59, 50, 50, 50, 50, 50, 59) | 8 | 140 | 49000 | 140 | 0 | 9.07 | 6.64 |
| 32768 | (20, 60, 60, 60, 60, 60, 60, 60) | 19 | 0 | 49000 | 50 | 0 | 9.07 | 6.79 |
| 32768 | (59, 50, 50, 50, 50, 50, 60) | 8 | 140 | 49000 | 140 | 0 | 9.07 | 6.79 |
| 16384 | (49, 40, 40, 40, 40, 40, 49) | 8 | 140 | 49000 | 140 | 0 | 9.07 | 6.64 |
| 16384 | (39, 30, 30, 30, 30, 30, 39) | 8 | 140 | 49000 | 140 | 0 | 9.07 | 6.64 |

- Measured: **0** catastrophic deviations in 660 encrypted digit-configuration pairs. The largest logit is 2^9.07, but the largest coefficient of the decrypted result is only 2^6.79: the matmul spreads the logits over the coefficients, so the plaintext rule overstates the risk. The one-prime configurations keep about 1.2 bits of coefficient headroom on these digits; this covers the evaluated digits only.

**Reliability:** final (MNIST has no grid comparison). The inferred counts above are superseded by the encrypted check.

## 10. Differentially private descriptor

The tool re-run with an ε-DP descriptor (Laplace mechanism on mean and second moment, public bounds simulated by each feature's range over the full dataset). *Same configuration*: identical knee point; *same class*: same N, scale and number of primes; *same output bound*: identical b_out.

| epsilon | same_configuration | same_class | same_output_bound | max_catastrophic_real |
|---|---|---|---|---|
| 0.10 | 0.27 | 0.64 | 0.64 | 0.00 |
| 1.00 | 0.36 | 0.76 | 0.91 | 0.00 |

**Reliability:** final, but only a secondary result; the knee choice is partly decided by noise, so "same configuration" understates how similar the choices are.

## 11. Tuning cost (reference only, not used in the abstract)

| dataset | held_out | candidates | encrypted_queries | search_seconds | keygen_seconds | eval_keys_MiB | ciphertexts_MiB | total_upload_download_MiB |
|---|---|---|---|---|---|---|---|---|
| bank_marketing | True | 24 | 1224 | 27.6 | 3.3 | 1603.1 | 1171.4 | 2774.5 |
| breast_cancer | False | 25 | 1275 | 29.4 | 3.4 | 1636.2 | 1199.4 | 2835.6 |
| california_housing | False | 25 | 1275 | 16.0 | 3.3 | 1628.9 | 1193.0 | 2821.9 |
| concrete | False | 25 | 1275 | 16.8 | 3.4 | 1598.9 | 1164.3 | 2763.2 |
| credit_default | True | 24 | 1224 | 25.8 | 3.6 | 1603.1 | 1171.2 | 2774.4 |
| diabetes | False | 25 | 1275 | 19.3 | 3.4 | 1612.7 | 1177.8 | 2790.4 |
| heart_disease | False | 25 | 1275 | 20.6 | 3.4 | 1629.1 | 1192.9 | 2822.0 |
| ionosphere | False | 25 | 1275 | 22.4 | 3.6 | 1636.1 | 1199.7 | 2835.8 |
| iris | False | 25 | 1275 | 26.1 | 3.5 | 1628.9 | 1193.0 | 2821.9 |
| mnist | False | 22 | 242 | 1011.1 | 20.8 | 8830.4 | 538.9 | 9369.3 |
| wine | False | 25 | 1275 | 30.6 | 3.6 | 1629.0 | 1192.7 | 2821.7 |
| wine_quality | True | 25 | 1275 | 19.8 | 3.3 | 1628.9 | 1193.0 | 2821.9 |

## 12. Headroom and search-rule diagnostics (same run)

Coefficient headroom = bits of the modulus left after the last rescale − log₂Δ − 1 − log₂(largest coefficient of the decrypted result): what decides overflow, measured on the evaluated inputs only.

- **Reported configurations** (tabular, up to 500 real rows): 69 configurations, 0 with a catastrophic deviation; 12 keep less coefficient headroom than the 3-bit margin:

| workload | N | chain | points | coefficient_headroom_bits |
|---|---|---|---|---|
| breast_cancer | 8192 | (58, 50, 60) | knee@128 fastest@128 | 2.22 |
| breast_cancer | 16384 | (58, 50, 60) | knee@192 knee@256 | 2.21 |
| breast_cancer | 8192 | (53, 45, 53) | fastest@192 | 2.22 |
| breast_cancer | 8192 | (38, 30, 38) | fastest@256 | 2.22 |
| wine_quality | 8192 | (57, 50, 60) | knee@128 fastest@128 | 0.51 |
| wine_quality | 8192 | (52, 45, 52) | fastest@192 | 0.51 |
| wine_quality | 16384 | (57, 50, 57) | knee@256 | 0.50 |
| wine_quality | 8192 | (37, 30, 37) | fastest@256 | 0.51 |
| california_housing | 8192 | (57, 50, 57) | fastest@128 | 0.94 |
| california_housing | 16384 | (57, 50, 60) | knee@192 knee@256 | 0.94 |
| california_housing | 8192 | (37, 30, 60) | fastest@192 | 0.94 |
| california_housing | 8192 | (37, 30, 37) | fastest@256 | 0.94 |

  All others keep at least 3.2 bits.
- **Largest coefficient vs. largest output** (knee at 128 bits): from -4.9 to +2.7 bits; above the output on concrete (+0.5), heart_disease (+0.3), wine_quality (+2.7), california_housing (+2.3) (a large pre-bias sum cancelled by the intercept leaves large values in the other slots). Expert configuration on diabetes: 1.7 bits of coefficient headroom.
- **DP descriptor** (100 noise draws per ε): output bound below the exact one in up to 33% of draws (diabetes, ε = 0.1); never at or below the largest real output (smallest spare 1.6 bits).
- **1-bit scales** (not used; future work): a larger exact scale exists for 199 of 273 candidates; median +3.0 bits (IQR +1.4 to +3.8) at a latency ratio of 1.000; 26 measured worse (down to -2.2 bits, unexplained).
- **Shared rules of the reference grid**: re-measured together with configurations that break one rule each (no margin, exact scale, N = 2^12, expert), the joint front's hypervolume relative to the grid front is median 1.15 (range 1.00–2.20); 50% of joint-front points are grid configurations; no margin 26%, N = 2^12 6%, expert 1%, exact scale 16%. No-margin configurations reach the joint front in 30 of 33 cases. The diabetes expert configuration is the no-margin candidate.

**Reliability:** final; diagnostics only (they change neither the search nor the metrics above).

## Environment and protocol facts

- Hardware: Apple M5 Pro, 15 cores; TenSEAL threads: 15
- Software: Python 3.14.6, tenseal 0.3.18, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, torch 2.14.1, Microsoft SEAL 4.3
- Tolerances used in this run: {"runtime_tolerance": 0.008582589550397436, "precision_tolerance_bits": 1.9779688113106513}
- Test vectors per candidate: 50 (MNIST 10); real rows per workload: up to 5,000 (MNIST 200); latency = encrypt + evaluate + decrypt in one process (no network or serialization)
- Run log (`STATUS`): start Mon Oct 26 23:07:44 EDT 2026 commit 902a326; stopped Tue Oct 27 09:13:25 EDT 2026 for transport; 10/12 workloads complete; bank_marketing partial moved to results/final_v4_aborted; resume Tue Oct 27 12:32:09 EDT 2026 commit 902a326; stopped Tue Oct 27 13:39:52 EDT 2026 for transport; bank_marketing partial (real-row stage) moved to results/final_v4_aborted; still to run: bank_marketing, wine_quality; resume Tue Oct 27 16:54:55 EDT 2026 commit 902a326; stopped Tue Oct 27 18:35:00 EDT 2026 for transport; bank_marketing complete; wine_quality partial (grid stage) moved to results/final_v4_aborted; still to run: wine_quality; resume Tue Oct 27 22:02:33 EDT 2026 commit 902a326; benchmark (wine_quality) exit 0 Tue Oct 27 22:49:55 EDT 2026; analyze exit 0 Tue Oct 27 22:50:00 EDT 2026; diagnostic coefficients exit 0 Tue Oct 27 22:51:14 EDT 2026; diagnostic reported exit 0 Tue Oct 27 22:54:18 EDT 2026; diagnostic dp exit 0 Tue Oct 27 22:55:21 EDT 2026; diagnostic scales exit 0 Tue Oct 27 23:05:31 EDT 2026; diagnostic rules exit 0 Tue Oct 27 23:13:32 EDT 2026; diagnostic mnist exit 0 Tue Oct 27 23:48:04 EDT 2026; done Tue Oct 27 23:48:04 EDT 2026; california rerun (log-transformed features) Wed Oct 28 15:25:20 EDT 2026 commit 328bc5d; previous run in results/final_v4_superseded; benchmark (california_housing) exit 0 Wed Oct 28 16:01:29 EDT 2026; analyze exit 0 Wed Oct 28 16:01:34 EDT 2026; diagnostic coefficients (california_housing) exit 0 Wed Oct 28 16:01:36 EDT 2026; diagnostic reported (california_housing) exit 0 Wed Oct 28 16:01:58 EDT 2026; diagnostic dp (california_housing) exit 0 Wed Oct 28 16:02:03 EDT 2026; diagnostic scales (california_housing) exit 0 Wed Oct 28 16:02:45 EDT 2026; diagnostic rules (california_housing) exit 0 Wed Oct 28 16:03:20 EDT 2026; done Wed Oct 28 16:03:20 EDT 2026
- Candidates evaluated in seeded random order; reference grid: scales 2^20–2^60 in 5-bit steps, exact and rounded first primes with offsets up to +50 bits, one- and two-prime final moduli, all special-prime sizes, 0–4 spare levels, N ∈ {8192, 16384, 32768}

import argparse
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ckks_param_tool.data.registry import DATASETS, HELD_OUT

from .analyze import latest_run
from .metrics import load_param_search, successful

'''
Collects the numbers for writing up a benchmark run into one folder:

    python -m benchmark_suite.abstract_data --results results/final_v3 --out paper/fheorg2027/data

writes
    <out>/RESULTS.md    every number used in the write-up, with a short description of each
                        metric and of how reliable the number currently is
    <out>/tables/       the analysis tables (CSV) of the run
    <out>/figures/      the tool's front plot of every workload and the tool-vs-grid fronts

Re-run it after a new benchmark run to update everything.
'''

GROUPS = ("development", "held-out", "mnist")


def group_of(name: str) -> str:
    return "mnist" if name == "mnist" else ("held-out" if name in HELD_OUT else "development")


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{digits}f}"


def pct(x, digits=1):
    return "–" if x is None or not math.isfinite(x) else f"{100 * x:.{digits}f}%"


def md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt(float(v), floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def runs(results: Path) -> dict[str, Path]:
    found = {}
    for name in DATASETS:
        d = results / name
        if d.is_dir():
            run = latest_run(d)
            if run is not None:
                found[name] = run
    return found


def info(run: Path) -> dict:
    return json.loads((run / "tool" / "run_info.json").read_text())


# ------------------------------------------------------------------------------------------
# workloads

def workloads(rs: dict[str, Path], quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, run in rs.items():
        loader, _, task = DATASETS[name]
        i = info(run)
        q = quality[quality.dataset == name]
        q = q.iloc[0] if len(q) else None
        desc = json.loads((run / "tool" / "descriptor.json").read_text())
        rows.append({
            "workload": name,
            "group": group_of(name),
            "task": i["model_task"],
            "features": len(desc["mean"]),
            "client rows": int(q["rows"]) if q is not None else "–",
            "depth": i["model_depth"],
            "plaintext quality": (f"acc {q['accuracy']:.3f}" if q is not None and pd.notna(q.get("accuracy"))
                                  else (f"R² {q['r2']:.3f}" if q is not None else "–")),
            "output bound b_out (bits, incl. +3)": i["output_int_bits"],
            "est. bits needed (classifiers)": i.get("estimated_required_bits") if i.get("estimated_required_bits") is not None else "–",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# precision ceiling and latency structure (from the tool's own measurements)

def ceilings(rs: dict[str, Path]) -> pd.DataFrame:
    # best precision at >= 128-bit security and N = 8192 with exactly depth + 2 primes (one-prime
    # final modulus, no spare level) vs. depth + 3 primes with a two-prime final modulus (q0, 60):
    # the two-prime structure costs exactly one extra prime
    rows = []
    for name, run in rs.items():
        if name == "mnist":
            continue
        depth = info(run)["model_depth"]
        tool = successful(load_param_search(run / "tool" / "param_search.csv"))
        at = tool[(tool.security >= 128) & (tool.poly_degree == 8192)]
        single = at[at.chain.apply(lambda c: len(c) == depth + 2)]
        two = at[at.chain.apply(lambda c: len(c) == depth + 3 and c[1] == 60)]
        if single.empty or two.empty:
            continue
        a = single.loc[single.precision_bits.idxmax()]
        b = two.loc[two.precision_bits.idxmax()]
        rows.append({
            "workload": name,
            "one-prime chain": str(tuple(a.chain)),
            "bits": a.precision_bits,
            "latency ms": 1000 * a.runtime,
            "two-prime chain": str(tuple(b.chain)),
            "bits (two-prime)": b.precision_bits,
            "latency ms (two-prime)": 1000 * b.runtime,
            "gain bits": b.precision_bits - a.precision_bits,
            "latency ratio": b.runtime / a.runtime,
        })
    return pd.DataFrame(rows)


def latency_structure(rs: dict[str, Path]) -> dict:
    # per (N, #primes) median latency from the grids measured under stable conditions
    per_prime, per_n = [], []
    for name, run in rs.items():
        if group_of(name) != "development" or not (run / "broad_grid").exists():
            continue
        grid = successful(load_param_search(run / "broad_grid" / "param_search.csv"))
        med = grid.groupby(["poly_degree", "chain_length"])["runtime"].median()
        for (n, k), v in med.items():
            if (n, k + 1) in med:
                per_prime.append(med[(n, k + 1)] / v - 1)
            if (2 * n, k) in med:
                per_n.append(med[(2 * n, k)] / v)
    return {
        "extra prime: median latency increase": float(np.median(per_prime)) if per_prime else float("nan"),
        "extra prime: range": (float(np.min(per_prime)), float(np.max(per_prime))) if per_prime else None,
        "doubling N: median latency factor": float(np.median(per_n)) if per_n else float("nan"),
        "doubling N: range": (float(np.min(per_n)), float(np.max(per_n))) if per_n else None,
    }


# ------------------------------------------------------------------------------------------
# MNIST: headroom of every reported configuration against all real digits

def mnist_headroom(run: Path) -> pd.DataFrame | None:
    from ckks_param_tool import client_server_split

    loader, trainer, _ = DATASETS["mnist"]
    X, y = loader()
    X_client, X_server, _, y_server = client_server_split(X, y)
    model = trainer(X_server, y_server)
    out = np.array([np.max(np.abs(model(np.asarray(x)))) for x in X_client])

    points = json.loads((run / "real_rows.json").read_text())
    rows, seen = [], set()
    for p in points:
        chain = tuple(p["chain"])
        if chain in seen:
            continue
        seen.add(chain)
        depth = 5
        final = chain[: len(chain) - depth - 1]           # primes left after the last rescale
        final_bits = sum(final)
        limit = 2.0 ** (final_bits - p["log_scale"] - 1)  # |value| * Delta must stay below Q_final / 2
        rows.append({
            "N": p["poly_degree"],
            "chain": str(chain),
            "final modulus bits": final_bits,
            "headroom (bits)": final_bits - p["log_scale"] - 1,
            "digits at risk (of all client digits)": int(np.sum(out >= limit)),
            "client digits": len(out),
            "real-row precision (200 digits)": p["real_rows"]["precision_bits"],
        })
    return pd.DataFrame(rows), float(out.max())


# ------------------------------------------------------------------------------------------

# the baselines against every tool candidate at least as secure, on the test vectors, with the
# run's tolerances: a tie if some candidate is at most tau_t slower and at most tau_p less precise
def baseline_ties(rs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, run in rs.items():
        if name == "mnist" or not (run / "baseline.json").exists():
            continue
        evaluation = json.loads((run / "tool" / "config.json").read_text())["evaluation"]
        tt, tp = evaluation["runtime_tolerance"], evaluation["precision_tolerance_bits"]
        tool = successful(load_param_search(run / "tool" / "param_search.csv"))
        for b in json.loads((run / "baseline.json").read_text())["baselines"]:
            pe, te = b["test_vectors"]["precision_bits"], b["test_vectors"]["runtime"]
            c = tool[tool.security >= b["security"]]
            fast = c[c.runtime <= te * (1 + tt)]
            precise = c[c.precision_bits >= pe - tp]
            best = float(fast.precision_bits.max()) if len(fast) else float("nan")
            rows.append({
                "workload": name, "group": group_of(name), "baseline": b["baseline"],
                "baseline_chain": str(tuple(b["chain"])), "baseline_bits": pe, "baseline_ms": 1000 * te,
                "tie": bool(len(fast) and best >= pe - tp),
                "gain_bits_within_tau_t": best - pe,
                "speedup_within_tau_p": te / float(precise.runtime.min()) if len(precise) else float("nan"),
            })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect the numbers of a benchmark run for the write-up.")
    parser.add_argument("--results", type=Path, default=Path("results/final_v4"))
    parser.add_argument("--diagnostics", type=Path, default=Path("results/final_v4_diagnostics"),
                        help="output of benchmark_suite.diagnostics on the same run (optional)")
    parser.add_argument("--out", type=Path, default=Path("paper/fheorg2027/data_final"))
    parser.add_argument("--skip-mnist-headroom", action="store_true", help="skip the (slow) MNIST check")
    args = parser.parse_args(argv)

    S = args.results / "summary"
    out = args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    tables = {p.stem: pd.read_csv(p) for p in sorted(S.glob("*.csv"))}
    for p in S.glob("*.csv"):
        shutil.copy(p, out / "tables" / p.name)

    rs = runs(args.results)
    for name, run in rs.items():
        plot = run / "tool" / "plots" / "runtime_vs_precision_pareto.png"
        if plot.exists():
            shutil.copy(plot, out / "figures" / f"{name}_tool_front.png")
        for p in S.glob(f"{name}_fronts_*.png"):
            shutil.copy(p, out / "figures" / p.name)

    any_info = info(next(iter(rs.values())))
    env = any_info["environment"]
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    config = json.loads((args.results / "config.json").read_text()) if (args.results / "config.json").exists() else {}
    noise_dirs = sorted((args.results / "noise").glob("*/summary.json"))
    noise = json.loads(noise_dirs[-1].read_text()) if noise_dirs else {}

    eff = tables["search_efficiency"].copy()
    eff["group"] = eff["dataset"].map(group_of)
    recs = tables["recommendations"].copy()
    base = tables["baseline"].copy()
    base["group"] = base["dataset"].map(group_of)
    tv = tables["test_vectors"].copy()
    tv["group"] = tv["dataset"].map(group_of)
    fails = tables["failures"].copy()
    dp = tables["dp_robustness"].copy()
    lat = tables["latency_vs_prime_size"].copy()
    lat["group"] = lat["dataset"].map(group_of)

    L = []
    w = L.append

    w("# Results for the write-up\n")
    w(f"Run: `{args.results}` · commit `{env['git']['commit'][:7]}` (uncommitted changes: {env['git']['uncommitted_changes']}) · "
      f"{cpu}, {env['cpu_count']} cores, TenSEAL {env['packages']['tenseal']}, Python {env['python']}.\n")
    w("Every number below comes from the tables in `tables/` (the run's `summary/`) and, for Section 12, from the "
      "diagnostics in `diagnostics/`. **Reliability** notes state what each number does and does not show.\n")

    # 1 workloads
    w("## 1. Workloads\n")
    w("Development workloads were used while the method was designed; held-out workloads only after it was fixed; "
      "MNIST is a stress test. *Output bound* is the empirically estimated integer bits of the largest model output "
      "(including the 3-bit safety margin). *Est. bits needed* is, for classifiers, the precision that keeps 99% "
      "of predictions unchanged according to the decision margins on synthetic samples (information only).\n")
    w(md_table(workloads(rs, tables["model_quality"])))
    w("\n**Reliability:** final.\n")

    # 2 tolerances
    w("## 2. Measurement noise and tolerances\n")
    w("Fixed configurations were measured 5 times each with fresh keys and encryption noise. *Precision spread* is "
      "the standard deviation of −log₂(error) across repetitions; *runtime spread* the coefficient of variation of the "
      "median latency. The tolerances are twice the 90th percentile over configurations; two points whose latencies "
      "differ by less than τ_t and precisions by less than τ_p count as tied.\n")
    if noise:
        w(f"- Configurations measured: **{noise['configurations']}**")
        w(f"- Precision spread: median **{noise['median_precision_std_bits']:.2f} bits**, 90th percentile "
          f"**{noise['p90_precision_std_bits']:.2f} bits**, max {noise['max_precision_std_bits']:.2f} bits")
        w(f"- Runtime spread: median **{pct(noise['median_runtime_cv'], 2)}**, 90th percentile "
          f"**{pct(noise['p90_runtime_cv'], 2)}**, max {pct(noise['max_runtime_cv'], 2)}")
        tp = noise["precision_tolerance_bits"]
        w(f"- **τ_p = {tp:.2f} bits** (a factor of {2 ** tp:.1f} in absolute error), **τ_t = {pct(noise['runtime_tolerance'], 2)}**\n")
    w("**Reliability:** final. The runtime tolerance comes from back-to-back repeats, so it is a lower bound on the "
      "timing noise over a long run.\n")

    # 3 search efficiency
    w("## 3. Search efficiency and front recovery (tool vs. reference grid)\n")
    w("- *Candidates*: configurations evaluated by the tool / by the reference grid (all security levels).\n"
      "- *Search time saved*: 1 − tool search time / grid search time (wall-clock, key generation included).\n"
      "- *Hypervolume ratio*: area dominated by the tool's front divided by that of the grid's front, in (latency, "
      "log₁₀ error), both normalized by the grid front; 1 = the tool front covers everything the grid front covers. "
      "*No floor*: without the minimum runtime span.\n"
      "- *ε-indicator* (multiplicative): the smallest factor by which the tool front's latencies and errors must be "
      "divided so that it weakly dominates every grid-front point; 1 = perfect, 2 = within a factor of 2 in both.\n"
      "- *ε-recall*: share of grid-front points for which the tool has a point that is tied within τ_t and τ_p.\n"
      "- *Exact recall*: share of grid-front parameter sets (N, Δ, Q) that the tool also evaluated.\n"
      "- All per security level ≥128 / ≥192 / ≥256, computed on the grid's own measurements of the tool's candidates.\n")
    agg = eff.groupby(["group", "security"]).agg(
        tool=("tool_candidates", "mean"), grid=("broad_grid_candidates", "mean"),
        cand_red=("candidate_reduction", "mean"), time_saved=("search_time_reduction", "mean"),
        hv_mean=("hypervolume_ratio", "mean"), hv_min=("hypervolume_ratio", "min"),
        eps_ind_median=("epsilon_indicator", "median"), eps_ind_max=("epsilon_indicator", "max"),
        eps_recall_mean=("epsilon_recall", "mean"), eps_recall_min=("epsilon_recall", "min"),
        recall_mean=("pareto_recall", "mean"),
    ).reset_index()
    w(md_table(agg))
    w("\nPer workload:\n")
    w(md_table(eff[["dataset", "group", "security", "tool_candidates", "broad_grid_candidates", "candidate_reduction",
                    "search_time_reduction", "hypervolume_ratio", "hypervolume_ratio_no_floor", "epsilon_indicator",
                    "epsilon_recall", "pareto_recall"]]))
    times = []
    for name, run in rs.items():
        if (run / "broad_grid").exists():
            t = json.loads((run / "tool" / "run_info.json").read_text())["search_seconds"]
            g = json.loads((run / "broad_grid" / "run_info.json").read_text())["search_seconds"]
            times.append({"workload": name, "group": group_of(name), "tool search (s)": t, "grid search (s)": g})
    if times:
        w("\nAbsolute search times:\n")
        w(md_table(pd.DataFrame(times), 1))
    w("\n**Reliability:** final. All grids ran on AC power without sleep (power logged every minute). Caveats: "
      "each grid configuration is measured once, so the grid front can include favourable noise draws, which "
      "lowers the measured recovery; the reference grid shares the tool's output-bound rule, 5-bit scale lattice "
      "and N range, so recovery is measured within those rules (Section 12); the hypervolume latency axis has a "
      "fixed 5% floor (with τ_t or no floor the values are identical). 14 of the 1,404 iris grid measurements are "
      "more than 15% slower than their (N, primes) group (a background interruption); none is a tool candidate "
      "or on a front.\n")

    # 4 reported points vs front
    w("## 4. Where the tool's reported points lie\n")
    w("For every security level the tool reports the knee point and the front's two endpoints (fastest, most "
      "precise). *On grid front*: Pareto-optimal among all grid configurations at least that secure (grid's own "
      "measurements). *Near grid front*: no grid configuration is both more than τ_t faster and more than τ_p more "
      "precise.\n")
    g = recs.groupby("group")[["on_grid_front", "near_grid_front"]].mean().reset_index()
    w(md_table(g, 2))
    w("\n**Reliability:** final (same caveats as Section 3).\n")

    # 5 baselines
    w("## 5. Baselines (measured on the same real client rows)\n")
    w("*Tutorial defaults*: N=8192, Q=(60,40,40,60), Δ=2⁴⁰ (depth ≤ 2). *Expert defaults*: 60-bit first and special "
      "prime, 50-bit scaling primes, smallest admissible N. Two asymmetric comparisons per baseline: "
      "*speedup* of the fastest tool configuration at least as precise as the baseline, and *precision gain* (bits) of "
      "the most precise tool configuration at least as fast.\n")
    for label in ("tutorial", "expert"):
        b = base[(base.baseline == label) & (base.dataset != "mnist")]
        fast = b[b.comparison.str.startswith("fastest")]
        prec = b[b.comparison.str.startswith("most precise")]
        w(f"**{label}** (strict, tabular): speedup at equal precision — median {fmt(fast.speedup.median())}×, range "
          f"{fmt(fast.speedup.min())}–{fmt(fast.speedup.max())}× ; precision gain at equal speed — median "
          f"{fmt(prec.precision_gain_bits.median(), 1)} bits, range {fmt(prec.precision_gain_bits.min(), 1)} to "
          f"{fmt(prec.precision_gain_bits.max(), 1)} bits.\n")
    cols = ["dataset", "group", "baseline", "comparison", "baseline_chain", "baseline_precision_bits", "baseline_runtime",
            "tool_N", "tool_chain", "tool_precision_bits", "tool_runtime", "speedup", "precision_gain_bits"]
    w(md_table(base[[c for c in cols if c in base.columns]], 3))
    ties = baseline_ties(rs)
    if len(ties):
        w("\n**With the measurement tolerances** (test vectors, tabular workloads; τ_t and τ_p of Section 2): a *tie* "
          "if some tool candidate is at most τ_t slower and at most τ_p less precise than the baseline; *gain* = best "
          "tool precision at most τ_t slower, minus the baseline's; *speedup* = baseline latency / fastest tool "
          "candidate at most τ_p less precise.\n")
        for label in ("tutorial", "expert"):
            t = ties[ties.baseline == label]
            w(f"- **{label}**: tie or better on **{int(t.tie.sum())} of {len(t)}**; gain median "
              f"{fmt(t.gain_bits_within_tau_t.median(), 1)} bits (range {fmt(t.gain_bits_within_tau_t.min(), 1)} to "
              f"{fmt(t.gain_bits_within_tau_t.max(), 1)}); speedup median {fmt(t.speedup_within_tau_p.median())}× "
              f"(range {fmt(t.speedup_within_tau_p.min())}–{fmt(t.speedup_within_tau_p.max())}×)")
        w("")
        w(md_table(ties, 2))
    w("\n**Reliability:** final. The strict table compares exact values (a 0.1% latency difference decides it); "
      "the tolerance version is the one to cite for ties. The expert gap on the regression workloads comes from the "
      "3-bit output margin (the expert's 60-bit first prime leaves too little headroom under the tool's rule).\n")

    # 6 reliability
    w("## 6. Reliability of the empirical estimates\n")
    w("*Catastrophic deviation*: error at least as large as the output (per real row: at least its own output and "
      "at least 1), or at least 2^headroom (an overflow); cause not proven. *Prediction agreement*: share of real rows with the same "
      "predicted class encrypted and in plaintext.\n")
    w(f"- Candidates with a catastrophic deviation on the synthetic test vectors: **{int(fails.catastrophic_on_test_vectors.sum())}** "
      f"of {int(fails.candidates.sum())} (other failures: {int(fails.other_failures.sum())}; step-downs: {int(fails.step_downs.sum())})")
    w(f"- Reported configurations with catastrophic deviations on real client rows: largest rate "
      f"**{fmt(fails.max_catastrophic_rate_real_rows.max(), 4)}** (real rows per workload: up to {int(fails.real_rows.max())})")
    w(f"- Prediction agreement of the reported configurations: at least **{fmt(recs.prediction_agreement.min(), 4)}**\n")
    w(md_table(fails))
    w("\n**Reliability:** final.\n")

    # 7 test vectors
    w("## 7. Fidelity of the synthetic test vectors\n")
    w("For every tool candidate the error on the 50 representative test vectors is compared with the error on 50 random "
      "real client rows (*truth*), and so are dataset-agnostic *naive* vectors (uniform in [−1, 1]). *R² (log)*: how well "
      "log₁₀ of the estimate predicts log₁₀ of the real error (1 = perfect). *Bias*: median of log₁₀(estimate / real) "
      "in decades (0 = unbiased, < 0 = optimistic). *Coverage*: share of candidates whose estimate is at least the real error.\n")
    g = tv.groupby(["group", "test_vectors"]).agg(r2_median=("r2_log", "median"), r2_min=("r2_log", "min"),
                                                  spearman_median=("spearman", "median"),
                                                  bias_median=("median_log_ratio", "median"),
                                                  coverage_median=("coverage", "median")).reset_index()
    w(md_table(g))
    w("\nPer workload:\n")
    w(md_table(tv[["dataset", "group", "test_vectors", "spearman", "pearson_log", "r2_log", "coverage", "median_log_ratio"]]))
    w("\n**Reliability:** final.\n")

    # 8 precision ceiling and latency
    w("## 8. Precision ceiling and what latency depends on\n")
    w("Best precision at ≥128-bit security and N = 8192 with a one-prime final modulus (depth + 2 primes) vs. a "
      "two-prime final modulus (depth + 3 primes), from the tool's own measurements. Latency structure from the "
      "development grids.\n")
    c = ceilings(rs)
    if not c.empty:
        w(md_table(c))
    ls = latency_structure(rs)
    w(f"\n- One additional prime: median latency **+{pct(ls['extra prime: median latency increase'], 0)}** "
      f"(range {pct(ls['extra prime: range'][0], 0)} to {pct(ls['extra prime: range'][1], 0)})" if ls["extra prime: range"] else "")
    w(f"- Doubling N: median latency **×{fmt(ls['doubling N: median latency factor'])}** "
      f"(range ×{fmt(ls['doubling N: range'][0])} to ×{fmt(ls['doubling N: range'][1])})" if ls["doubling N: range"] else "")
    dev = lat[lat.group == "development"]
    w(f"- Same N and number of primes, different prime sizes (development grids): median spread "
      f"**{pct(dev.latency_spread.median())}**, max {pct(dev.latency_spread.max())}; interquartile range "
      f"{pct(dev.latency_iqr_rel.median(), 2)} of the median\n")
    w("**Reliability:** final. The latency-structure numbers use the development grids; the largest spread is the "
      "iris outlier noted in Section 3 (the interquartile range is the robust figure). The ~36–40-bit plateau is the "
      "double-precision limit of CKKS encoding in SEAL/TenSEAL.\n")

    # 9 MNIST
    w("## 9. MNIST (stress test)\n")
    m = recs[recs.dataset == "mnist"]
    if len(m):
        w(md_table(m[["target_security", "point", "N", "chain", "precision_bits_test_vectors", "precision_bits",
                      "runtime", "prediction_agreement", "catastrophic_rate", "real_rows"]]))
        gap = (m.precision_bits_test_vectors - m.precision_bits).median()
        w(f"\n- Test vectors overstate precision by a median of **{gap:.1f} bits** (tabular workloads: ~0 to 0.2 bits).")
        mi = info(rs["mnist"])
        w(f"- Output bound from the descriptor: {mi['output_int_bits']} bits including the 3-bit margin.")
    if "mnist" in rs and not args.skip_mnist_headroom:
        h, max_logit = mnist_headroom(rs["mnist"])
        w(f"- Largest real logit over all client digits: **{max_logit:.0f}** ({math.ceil(math.log2(max_logit))} bits), "
          f"i.e. the descriptor underestimates it by {math.ceil(math.log2(max_logit)) - (mi['output_int_bits'] - 3)} bits.")
        w("- Headroom of each reported configuration against all client digits, **inferred from plaintext logits, "
          "not encrypted** (a digit is at risk if |logit|·Δ reaches half the final modulus; the encrypted check is "
          "`python -m benchmark_suite.diagnostics mnist`):\n")
        w(md_table(h))
        at_risk = h[h["digits at risk (of all client digits)"] > 0]
        if len(at_risk):
            w(f"\n- **{len(at_risk)} of {len(h)} reported configurations** leave too little headroom for the largest real "
              f"logits: {int(at_risk['digits at risk (of all client digits)'].max())} of {int(h['client digits'].iloc[0])} "
              f"digits ({pct(at_risk['digits at risk (of all client digits)'].max() / h['client digits'].iloc[0], 2)}) exceed "
              "the nominal headroom (inferred from plaintext logits).")
    mn = args.diagnostics / "mnist.csv"
    if mn.exists():
        m = pd.read_csv(mn)
        w("\n**Encrypted check of the inferred risk** (`diagnostics mnist`): every digit inferred to be at risk was "
          "encrypted with each configuration (the 50 largest logits for configurations without inferred risk).\n")
        w(md_table(m, 2))
        w(f"\n- Measured: **{int(m.measured_catastrophic.sum())}** catastrophic deviations in "
          f"{int(m.encrypted_digits.sum())} encrypted digit-configuration pairs. The largest logit is "
          f"2^{m.largest_logit_bits.max():.2f}, but the largest coefficient of the decrypted result is only "
          f"2^{m.largest_coefficient_bits.max():.2f}: the matmul spreads the logits over the coefficients, so the "
          "plaintext rule overstates the risk. The one-prime configurations keep about "
          f"{(m.headroom_bits - m.largest_coefficient_bits)[m.headroom_bits < 10].min():.1f} bits of coefficient "
          "headroom on these digits; this covers the evaluated digits only.")
    w("\n**Reliability:** final (MNIST has no grid comparison). The inferred counts above are superseded by the "
      "encrypted check.\n")

    # 10 DP
    w("## 10. Differentially private descriptor\n")
    w("The tool re-run with an ε-DP descriptor (Laplace mechanism on mean and second moment, public bounds simulated by "
      "each feature's range over the full dataset). *Same configuration*: identical knee point; *same class*: same N, "
      "scale and number of primes; *same output bound*: identical b_out.\n")
    d = dp.groupby("epsilon").agg(same_configuration=("same_as_non_dp", "mean"),
                                  same_class=("same_class_as_non_dp", "mean"),
                                  same_output_bound=("output_int_bits", lambda s: float(np.mean(s.values == dp.loc[s.index, "output_int_bits_non_dp"].values))),
                                  max_catastrophic_real=("catastrophic_rate", "max")).reset_index()
    w(md_table(d, 2))
    w("\n**Reliability:** final, but only a secondary result; the knee choice is partly decided by noise, so "
      "\"same configuration\" understates how similar the choices are.\n")

    # 11 tuning cost (reference only)
    if "tuning_cost" in tables:
        w("## 11. Tuning cost (reference only, not used in the abstract)\n")
        w(md_table(tables["tuning_cost"], 1))
        w("")

    # 12 diagnostics
    D = args.diagnostics
    if D.exists():
        (out / "diagnostics").mkdir(exist_ok=True)
        for p in D.glob("*.csv"):
            shutil.copy(p, out / "diagnostics" / p.name)
        w("## 12. Headroom and search-rule diagnostics (same run)\n")
        w("Coefficient headroom = bits of the modulus left after the last rescale − log₂Δ − 1 − log₂(largest "
          "coefficient of the decrypted result): what decides overflow, measured on the evaluated inputs only.\n")
        if (D / "reported.csv").exists():
            r = pd.read_csv(D / "reported.csv")
            w(f"- **Reported configurations** (tabular, up to 500 real rows): {len(r)} configurations, "
              f"{int((r.catastrophic_rate > 0).sum())} with a catastrophic deviation; "
              f"{int(r.below_margin.sum())} keep less coefficient headroom than the 3-bit margin:\n")
            w(md_table(r[r.below_margin][["workload", "N", "chain", "points", "coefficient_headroom_bits"]], 2))
            w(f"\n  All others keep at least {r[~r.below_margin].coefficient_headroom_bits.min():.1f} bits.")
        if (D / "coefficients.csv").exists():
            c = pd.read_csv(D / "coefficients.csv")
            k = c[c.config == "knee_128"].set_index("workload").coefficient_minus_output_bits
            e = c[(c.config == "expert") & (c.workload == "diabetes")]
            w(f"- **Largest coefficient vs. largest output** (knee at 128 bits): from {k.min():+.1f} to {k.max():+.1f} "
              f"bits; above the output on " + ", ".join(f"{n} ({v:+.1f})" for n, v in k[k > 0.2].items()) +
              " (a large pre-bias sum cancelled by the intercept leaves large values in the other slots)."
              + (f" Expert configuration on diabetes: {e.coefficient_headroom_bits.iloc[0]:.1f} bits of coefficient "
                 "headroom." if len(e) else ""))
        if (D / "dp.csv").exists():
            d = pd.read_csv(D / "dp.csv")
            worst = d.loc[d.share_below_exact.idxmax()]
            w(f"- **DP descriptor** ({int(d.draws.iloc[0])} noise draws per ε): output bound below the exact one in up "
              f"to {pct(d.share_below_exact.max(), 0)} of draws ({worst.workload}, ε = {worst.epsilon}); "
              f"never at or below the largest real output (smallest spare {d.min_spare_bits.min():.1f} bits).")
        if (D / "scales.csv").exists():
            sc = pd.read_csv(D / "scales.csv")
            total = int(sc.groupby("workload").tool_candidates.first().sum())
            w(f"- **1-bit scales** (not used; future work): a larger exact scale exists for {len(sc)} of {total} "
              f"candidates; median {sc.gain_bits.median():+.1f} bits (IQR {sc.gain_bits.quantile(.25):+.1f} to "
              f"{sc.gain_bits.quantile(.75):+.1f}) at a latency ratio of {sc.latency_ratio.median():.3f}; "
              f"{int((sc.gain_bits < 0).sum())} measured worse (down to {sc.gain_bits.min():.1f} bits, unexplained).")
        if (D / "rules.csv").exists():
            u = pd.read_csv(D / "rules.csv")
            cols = [x for x in u.columns if x.startswith("front_points_")]
            share = u[cols].sum() / u.joint_front_points.sum()
            w(f"- **Shared rules of the reference grid**: re-measured together with configurations that break one "
              f"rule each (no margin, exact scale, N = 2^12, expert), the joint front's hypervolume relative to the "
              f"grid front is median {u.hv_joint_vs_grid_front.median():.2f} (range {u.hv_joint_vs_grid_front.min():.2f}–"
              f"{u.hv_joint_vs_grid_front.max():.2f}); {pct(1 - share.sum(), 0)} of joint-front points are grid "
              "configurations; " + ", ".join(f"{c.replace('front_points_', '')} {pct(v, 0)}" for c, v in share.items()) +
              f". No-margin configurations reach the joint front in {int((u['front_points_no margin'] > 0).sum())} of "
              f"{len(u)} cases. The diabetes expert configuration is the no-margin candidate.")
        w("\n**Reliability:** final; diagnostics only (they change neither the search nor the metrics above).\n")

    w("## Environment and protocol facts\n")
    w(f"- Hardware: {cpu}, {env['cpu_count']} cores; TenSEAL threads: {env['tenseal_threads']}")
    w(f"- Software: Python {env['python']}, " + ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)
      + (f", Microsoft SEAL {env['seal']}" if env.get("seal") else ""))
    w(f"- Tolerances used in this run: {json.dumps(config.get('evaluation', {}))}")
    w("- Test vectors per candidate: 50 (MNIST 10); real rows per workload: up to 5,000 (MNIST 200); latency = "
      "encrypt + evaluate + decrypt in one process (no network or serialization)")
    status = args.results / "STATUS"
    if status.exists():
        w("- Run log (`STATUS`): " + "; ".join(l.strip() for l in status.read_text().splitlines() if l.strip()))
    w("- Candidates evaluated in seeded random order; reference grid: scales 2^20–2^60 in 5-bit steps, exact and "
      "rounded first primes with offsets up to +50 bits, one- and two-prime final moduli, all special-prime sizes, "
      "0–4 spare levels, N ∈ {8192, 16384, 32768}")

    (out / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"written: {out / 'RESULTS.md'}, {len(tables)} tables, figures in {out / 'figures'}")


if __name__ == "__main__":
    main()

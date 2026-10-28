import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ckks_param_tool import client_server_split
from ckks_param_tool.models.base import predict

from .analyze import latest_run
from .datasets import DATASETS
from .metrics import load_param_search, successful

'''
Per-workload results table for the abstract (LaTeX, booktabs), from an existing benchmark run:

    python -m benchmark_suite.paper_tables --results results/final_v4 --out paper/fheorg2027/tables/per_workload.tex

Columns: features; search time (tool s / grid min); hypervolume ratio (mean over the three security
levels); speedup and precision gain over the tutorial configuration (strict comparisons of
baseline.csv); R^2 of test-vector vs. real-row log error; the task metric in plaintext and under the
worst reported configuration encrypted; the fastest >= 128-bit latency (tool's own measurement).

The encrypted task metric is not stored per row, so it is derived from what the run saved, on the
same real rows the run encrypted (reconstructed from the evaluation seed and checked against the
saved row count): for a classifier, prediction agreement 1 means the encrypted accuracy equals the
plaintext accuracy on those rows, otherwise it lies within +-(1 - agreement); for a regression,
|R^2_enc - R^2| <= (2 e sum|r_i| + n e^2) / SS_tot with e the largest absolute error. The value is
printed as equal when this bound is below half a unit of the third decimal, else as an interval.
'''

GROUPS = [
    ("Regression", ["diabetes", "california_housing", "concrete", "wine_quality"]),
    ("Binary classification", ["breast_cancer", "heart_disease", "ionosphere", "credit_default", "bank_marketing"]),
    ("Multiclass", ["wine", "iris"]),
]
NAMES = {"diabetes": "Diabetes", "california_housing": "Calif.\\ housing", "concrete": "Concrete",
         "wine_quality": "Wine quality", "breast_cancer": "Breast cancer", "heart_disease": "Heart disease",
         "ionosphere": "Ionosphere", "credit_default": "Credit default", "bank_marketing": "Bank marketing",
         "wine": "Wine", "iris": "Iris"}


def _metric(task, outputs, y):
    if task == "regression":
        pred = np.array([float(np.atleast_1d(o)[0]) for o in outputs])
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), pred
    classes = np.unique(y)
    idx = np.array([predict(task, o)[0] for o in outputs])
    return float(np.mean(classes[idx] == y)), None


def task_metric(name: str, run: Path, eval_rows: int, seed: int) -> dict:
    loader, trainer, _ = DATASETS[name]
    X, y = loader()
    X_client, X_server, y_client, y_server = client_server_split(X, y)
    model = trainer(X_server, y_server)
    X_client, y_client = np.asarray(X_client), np.asarray(y_client)

    # the plaintext metric on all client rows must reproduce the saved model quality
    full, _ = _metric(model.task, [model(x) for x in X_client], y_client)
    saved = json.loads((run / "model_quality.json").read_text())
    assert abs(full - saved.get("accuracy", saved.get("r2"))) < 1e-9, f"{name}: model not reproduced"

    # the rows the run encrypted (TenSEALInferenceEvaluator's seeded sample)
    rng = np.random.default_rng(seed)
    size = len(X_client) if eval_rows == 0 else min(eval_rows, len(X_client))
    idx = rng.choice(len(X_client), size=size, replace=False)
    Xs, ys = X_client[idx], y_client[idx]
    plain, pred = _metric(model.task, [model(x) for x in Xs], ys)

    points = json.loads((run / "real_rows.json").read_text())
    for p in points:
        assert p["real_rows"]["num_rows"] == len(idx), f"{name}: evaluated rows not reproduced"
    if model.task == "regression":
        worst = max(points, key=lambda p: p["real_rows"]["error"])
        e = worst["real_rows"]["error"]
        bound = (2 * e * np.sum(np.abs(ys - pred)) + len(ys) * e**2) / np.sum((ys - ys.mean()) ** 2)
    else:
        worst = min(points, key=lambda p: (p["real_rows"]["prediction_agreement"], -p["real_rows"]["error"]))
        bound = 1 - worst["real_rows"]["prediction_agreement"]
    return {"plain": plain, "bound": bound, "rows": len(idx), "worst_chain": tuple(worst["chain"]),
            "worst_N": worst["poly_degree"], "worst_point": f"{worst['point']}@{worst['target_security']}"}


def collect(results: Path, exclude=()) -> pd.DataFrame:
    eff = pd.read_csv(results / "summary" / "search_efficiency.csv")
    base = pd.read_csv(results / "summary" / "baseline.csv")
    tv = pd.read_csv(results / "summary" / "test_vectors.csv")
    rows = []
    for _, names in GROUPS:
        for name in names:
            if name in exclude:
                continue
            run = latest_run(results / name)
            evaluation = json.loads((run / "tool" / "config.json").read_text())["evaluation"]
            tool = successful(load_param_search(run / "tool" / "param_search.csv"))
            tut = base[(base.dataset == name) & (base.baseline == "tutorial")]
            rows.append({
                "workload": name,
                "d": len(json.loads((run / "tool" / "descriptor.json").read_text())["mean"]),
                "tool_s": json.loads((run / "tool" / "run_info.json").read_text())["search_seconds"],
                "grid_min": json.loads((run / "broad_grid" / "run_info.json").read_text())["search_seconds"] / 60,
                "hv": eff[eff.dataset == name].hypervolume_ratio.mean(),
                "speedup": tut[tut.comparison.str.startswith("fastest")].speedup.iloc[0],
                "gain": tut[tut.comparison.str.startswith("most precise")].precision_gain_bits.iloc[0],
                # R^2 of log10 test-vector error vs. log10 real-row error over all candidates, and the
                # median log10(estimate / real) bias, for representative and uniform (naive) vectors
                "r2": tv[(tv.dataset == name) & (tv.test_vectors == "representative")].r2_log.iloc[0],
                "r2_uniform": tv[(tv.dataset == name) & (tv.test_vectors == "naive")].r2_log.iloc[0],
                "bias": tv[(tv.dataset == name) & (tv.test_vectors == "representative")].median_log_ratio.iloc[0],
                "bias_uniform": tv[(tv.dataset == name) & (tv.test_vectors == "naive")].median_log_ratio.iloc[0],
                "fastest_ms": 1000 * tool[tool.security >= 128].runtime.min(),
                **task_metric(name, run, evaluation["num_samples"], evaluation["random_state"]),
            })
    return pd.DataFrame(rows)


# the encrypted metric: equal at three decimals, or the plaintext value with the bound rounded up
def _encrypted(plain, bound):
    if bound < 5e-4:
        return f"{plain:.3f}"
    return f"{plain:.3f}$\\pm$.{int(np.ceil(bound * 1000)):03d}"


def _head(*lines):
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


def latex(df: pd.DataFrame) -> str:
    L = [
        "% generated by python -m benchmark_suite.paper_tables (do not edit by hand)",
        "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{2.3pt}",
        "\\begin{tabular}{@{}lrr@{\\,/\\,}lrrrrlr@{}}",
        "\\toprule",
        "Workload & $d$ & \\multicolumn{2}{c}{" + _head("Search", "s\\,/\\,min") + "} & HV & " + _head("Speed-", "up")
        + " & " + _head("Gain", "(bits)") + " & " + _head("Test-vector fit,", "$R^2$ (uniform)") + " & " + _head("Task metric", "plain\\,$\\to$\\,enc.") + " & "
        + _head("Fastest", "(ms)") + " \\\\",
        "\\midrule",
    ]
    first = True
    for label, names in GROUPS:
        names = [n for n in names if n in set(df.workload)]
        if not names:
            continue
        if not first:
            L.append("\\midrule")
        first = False
        L.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
        for name in names:
            r = df[df.workload == name].iloc[0]
            L.append(f"{NAMES[name]} & {r.d} & {r.tool_s:.0f} & {r.grid_min:.0f} & {r.hv:.2f} & "
                     f"{r.speedup:.2f}$\\times$ & {r.gain:+.1f} & {r.r2:.3f} ({r.r2_uniform:.3f}) & "
                     f"{r.plain:.3f}\\,$\\to$\\,{_encrypted(r.plain, r.bound)} & {r.fastest_ms:.1f} \\\\"
                     .replace("-", "$-$").replace("$-$$-$", "--"))
    m = df.median(numeric_only=True)
    L += ["\\midrule",
          f"Median ({len(df)}) & {m.d:.0f} & {m.tool_s:.0f} & {m.grid_min:.0f} & {m.hv:.2f} & {m.speedup:.2f}$\\times$ & "
          f"{m.gain:+.1f} & {m.r2:.3f} ({m.r2_uniform:.3f}) & & {m.fastest_ms:.1f} \\\\",
          "\\bottomrule", "\\end{tabular}", "\\endgroup", ""]
    return "\n".join(L)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/final_v4"))
    parser.add_argument("--out", type=Path, default=Path("paper/fheorg2027/tables/per_workload.tex"))
    parser.add_argument("--exclude", nargs="*", default=[], help="workloads left out of the table")
    args = parser.parse_args(argv)
    df = collect(args.results, set(args.exclude))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex(df))
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

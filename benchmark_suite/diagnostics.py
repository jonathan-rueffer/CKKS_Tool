import argparse
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ckks_param_tool import DataDescriptor, client_server_split
from ckks_param_tool.config import DataConfig, EvaluationConfig
from ckks_param_tool.data.generators import estimate_output_int_bits, generate_test_vectors, magnitude_inputs
from ckks_param_tool.evaluation import TenSEALInferenceEvaluator
from ckks_param_tool.params import CKKSParams

from .analyze import latest_run
from .datasets import DATASETS

'''
Diagnostics of the tool's assumptions, run on the benchmark workloads and an existing benchmark
run (nothing here changes the tool's search or the reported metrics):

    python -m benchmark_suite.diagnostics coefficients --results results/final_v3
    python -m benchmark_suite.diagnostics reported --results results/final_v3
    python -m benchmark_suite.diagnostics dp    --results results/final_v3
    python -m benchmark_suite.diagnostics scales --results results/final_v3
    python -m benchmark_suite.diagnostics rules  --results results/final_v3
    python -m benchmark_suite.diagnostics mnist  --results results/final_v3

coefficients
        The tool sizes the final modulus for the largest plaintext OUTPUT, but an overflow
        depends on the largest coefficient of the decrypted plaintext, which mixes the outputs
        with the other slots (TenSEAL leaves rotation partial sums and broadcast biases there;
        see TenSEALBackend.max_abs_coefficient). Measures, on real client rows, the largest
        coefficient against the largest output for the reported knee point at 128 bits and the
        expert configuration, and the headroom left above the largest coefficient.
reported
        The coefficient headroom of every configuration the tool reported (tabular workloads)
        on up to 500 real client rows, and whether it is below the output margin.
dp      The differentially private descriptor takes min/max from the public bounds and adds
        Laplace noise to the means and second moments. Repeats the output-bound estimate for
        many noise draws and compares it with the exact descriptor's and with the largest real
        output.
scales  The budget scales are multiples of 5 bits (so that every candidate is also on the
        reference grid). For every tool candidate, finds the largest scale in 1-bit steps with
        the same N, structure and special-prime rule that still meets the candidate's security
        level, and measures both versions back to back on the tool's own test vectors: the
        precision and latency an exact-scale candidate would add, and how many candidates it
        costs.
rules   The reference grid shares the tool's candidate-generation rules (output bound with a
        3-bit margin, 5-bit scale lattice, N >= 2^13, 60-bit primes), so front recovery is
        measured within them. Re-measures the grid's Pareto-front configurations in one
        session together with configurations that each break one shared rule: no output margin,
        the exact (1-bit) largest scale, N = 2^12, and the expert configuration. Reports, per
        security level, which of them reach the joint front and how much they extend it
        (hypervolume relative to the grid front, both measured in this session).
mnist   The MNIST headroom check of abstract_data is inferred from plaintext logits (a digit is
        at risk if |logit| reaches 2^headroom). Encrypts every at-risk digit with each reported
        configuration and counts the digits that actually deviate catastrophically, with the
        largest coefficient headroom measured on the others.

Output: <out>/<diagnostic>.csv
'''

logger = logging.getLogger("diagnostics")

TABULAR = [n for n in DATASETS if n != "mnist"]


@lru_cache(maxsize=None)
def workload(name: str):
    loader, trainer, _ = DATASETS[name]
    X, y = loader()
    X_client, X_server, y_client, y_server = client_server_split(X, y)
    return np.asarray(X), np.asarray(X_client), trainer(X_server, y_server)


def _params(entry) -> CKKSParams:
    return CKKSParams(entry["poly_degree"], 2 ** entry["log_scale"], tuple(entry["chain"]))


def _bits(v: float) -> float:
    return math.log2(v) if v > 0 else float("-inf")


def coefficients(results: Path, rows: int = 200) -> pd.DataFrame:
    from ckks_param_tool.search import expert_baseline_params

    out = []
    for name in DATASETS:
        run = latest_run(results / name)
        if run is None:
            continue
        _, X_client, model = workload(name)
        with open(run / "real_rows.json") as f:
            knee = next(p for p in json.load(f) if p["point"] == "knee" and p["target_security"] == 128)
        with open(run / "tool" / "run_info.json") as f:
            b_out = json.load(f)["output_int_bits"]
        n = 10 if name == "mnist" else rows
        evaluator = TenSEALInferenceEvaluator(model, X_client, EvaluationConfig(num_samples=n, warmup_runs=0))
        depth = model.multiplicative_depth
        for label, params in (("knee_128", _params(knee)), ("expert", expert_baseline_params(depth))):
            r = evaluator.evaluate(params)
            headroom = params.output_headroom_bits(depth)
            row = {
                "workload": name, "config": label, "N": params.poly_degree, "chain": str(params.chain),
                "log_scale": params.log_scale, "b_out_estimate": b_out, "headroom_bits": headroom,
                "rows": r.num_rows, "status": r.status,
                "output_bits": _bits(r.max_output), "coefficient_bits": _bits(r.max_coefficient),
                "coefficient_minus_output_bits": _bits(r.max_coefficient) - _bits(r.max_output),
                "coefficient_headroom_bits": r.coefficient_headroom_bits, "catastrophic_rate": r.catastrophic_rate,
            }
            # linear models: the pre-bias sum and the bias, the two terms that can cancel
            W, b = getattr(model, "W", None), getattr(model, "b", None)
            if W is not None:
                xs = X_client[evaluator.sample_idx]
                row["pre_bias_bits"] = _bits(float(np.max(np.abs(xs @ W))))
                row["bias_bits"] = _bits(float(np.max(np.abs(b))))
            out.append(row)
            logger.info("[coefficients] %s %s: %.2f bits above the output, %.2f bits of headroom left",
                        name, label, row["coefficient_minus_output_bits"], row["coefficient_headroom_bits"])
    return pd.DataFrame(out)


def dp(results: Path, draws: int = 100, epsilons=(1.0, 0.1)) -> pd.DataFrame:
    config = DataConfig()
    out = []
    for name in TABULAR:
        X, X_client, model = workload(name)
        exact_desc = DataDescriptor.from_data(X_client)
        tv = generate_test_vectors(exact_desc, config)
        exact = estimate_output_int_bits(model, np.vstack([magnitude_inputs(exact_desc, config), tv]),
                                         config.output_margin_bits)
        real_bits = _bits(max(float(np.max(np.abs(model(x)))) for x in X_client))
        bounds = (np.min(X, axis=0), np.max(X, axis=0))
        for eps in epsilons:
            bits = []
            for seed in range(draws):
                desc = DataDescriptor.private(X_client, eps, bounds, rng=seed)
                tv = generate_test_vectors(desc, config)
                bits.append(estimate_output_int_bits(model, np.vstack([magnitude_inputs(desc, config), tv]),
                                                     config.output_margin_bits))
            bits = np.array(bits)
            out.append({
                "workload": name, "epsilon": eps, "draws": draws, "b_out_exact": exact,
                "real_output_bits": real_bits, "b_out_dp_min": int(bits.min()),
                "b_out_dp_median": float(np.median(bits)), "b_out_dp_max": int(bits.max()),
                "share_below_exact": float(np.mean(bits < exact)),
                "share_above_exact": float(np.mean(bits > exact)),
                "min_spare_bits": float(bits.min() - real_bits),
                "share_without_spare": float(np.mean(bits <= real_bits)),
            })
            logger.info("[dp] %s eps=%s: b_out %d..%d (exact %d), min spare %.2f bits", name, eps,
                        bits.min(), bits.max(), exact, bits.min() - real_bits)
    return pd.DataFrame(out)


def _exact_scale(policy, params, structure, depth, b_out):
    from ckks_param_tool.params import SECURITY_TABLES, chain_infeasibility_reason

    sec = params.security_score
    budget = SECURITY_TABLES[int(sec)][params.poly_degree]
    best = params
    for s in range(params.log_scale + 1, params.log_scale + 5):
        chain = policy.chain(s, depth, b_out, structure)
        if chain is None or sum(chain) > budget or chain_infeasibility_reason(params.poly_degree, chain):
            continue
        if structure.special == "max" and chain[-1] != policy.max_qL:
            continue
        best = CKKSParams(params.poly_degree, 2**s, chain)
    return best


def scales(results: Path) -> pd.DataFrame:
    from ckks_param_tool.config import load_experiment_config
    from ckks_param_tool.evaluation import TenSEALParameterEvaluator
    from ckks_param_tool.search import CKKSParameterSearchSpace, CKKSSearchPolicy

    out = []
    for name in TABULAR:
        run = latest_run(results / name)
        if run is None:
            continue
        _, _, model = workload(name)
        with open(run / "tool" / "run_info.json") as f:
            b_out = json.load(f)["output_int_bits"]
        with open(run / "tool" / "config.json") as f:
            config = load_experiment_config(json.load(f))
        descriptor = DataDescriptor.load(run / "tool" / "descriptor.json")
        evaluator = TenSEALParameterEvaluator(model, generate_test_vectors(descriptor, config.data),
                                              config.evaluation)
        depth = model.multiplicative_depth
        policy = CKKSSearchPolicy.default()
        space = CKKSParameterSearchSpace(depth, b_out, policy)
        candidates = space.generate()
        changed = 0
        for params in candidates:
            exact = _exact_scale(policy, params, space.structure_of[params], depth, b_out)
            if exact == params:
                continue
            changed += 1
            # back to back, each measured twice in alternating order, to keep drift out of the ratio
            a1, b1, b2, a2 = (evaluator.evaluate(p) for p in (params, exact, exact, params))
            out.append({
                "workload": name, "N": params.poly_degree, "security": params.security_score,
                "chain": str(params.chain), "exact_chain": str(exact.chain),
                "log_scale": params.log_scale, "exact_log_scale": exact.log_scale,
                "precision_bits": np.mean([a1.precision_bits, a2.precision_bits]),
                "exact_precision_bits": np.mean([b1.precision_bits, b2.precision_bits]),
                "gain_bits": np.mean([b1.precision_bits, b2.precision_bits]) - np.mean([a1.precision_bits, a2.precision_bits]),
                "latency_ratio": np.mean([b1.runtime, b2.runtime]) / np.mean([a1.runtime, a2.runtime]),
                "tool_candidates": len(candidates),
            })
        logger.info("[scales] %s: %d of %d candidates have a larger exact scale", name, changed, len(candidates))
    return pd.DataFrame(out)


def rules(results: Path) -> pd.DataFrame:
    from dataclasses import replace
    from ckks_param_tool.config import load_experiment_config
    from ckks_param_tool.evaluation import TenSEALParameterEvaluator
    from ckks_param_tool.plotting import results_to_dataframe
    from ckks_param_tool.search import CKKSParameterSearchSpace, CKKSSearchPolicy, expert_baseline_params
    from .metrics import at_least, hypervolume_ratio, keys, load_param_search, pareto_front, successful

    out = []
    for name in TABULAR:
        run = latest_run(results / name)
        if run is None or not (run / "broad_grid").exists():
            continue
        _, _, model = workload(name)
        depth = model.multiplicative_depth
        with open(run / "tool" / "run_info.json") as f:
            b_out = json.load(f)["output_int_bits"]
        with open(run / "tool" / "config.json") as f:
            config = load_experiment_config(json.load(f))
        descriptor = DataDescriptor.load(run / "tool" / "descriptor.json")
        evaluator = TenSEALParameterEvaluator(model, generate_test_vectors(descriptor, config.data), config.evaluation)

        grid = successful(load_param_search(run / "broad_grid" / "param_search.csv"))
        front_keys = set()
        for sec in (128, 192, 256):
            front_keys |= keys(pareto_front(at_least(grid, sec)))
        sources = {CKKSParams(n, s, c): "grid front" for n, s, c in front_keys}

        policy = CKKSSearchPolicy.default()
        relaxed = {
            "no margin": CKKSParameterSearchSpace(depth, b_out - config.data.output_margin_bits, policy).generate(),
            "N = 2^12": [p for p in CKKSParameterSearchSpace(
                depth, b_out, replace(policy, poly_degrees=(4096,) + policy.poly_degrees)).generate()
                if p.poly_degree == 4096],
            "expert": [expert_baseline_params(depth)],
        }
        space = CKKSParameterSearchSpace(depth, b_out, policy)
        relaxed["exact scale"] = [e for p in space.generate()
                                  if (e := _exact_scale(policy, p, space.structure_of[p], depth, b_out)) != p]
        for label, params in relaxed.items():
            for p in params:
                sources.setdefault(p, label)

        order = np.random.default_rng(0).permutation(len(sources))
        params = [list(sources)[i] for i in order]
        df = results_to_dataframe([(p, evaluator.evaluate(p)) for p in params])
        df["source"] = [sources[p] for p in params]
        df = successful(df)

        for sec in (128, 192, 256):
            level = at_least(df, sec)
            grid_front = pareto_front(level[level.source == "grid front"])
            joint = pareto_front(level)
            row = {"workload": name, "security": sec, "grid_front_points": len(grid_front),
                   "joint_front_points": len(joint),
                   "hv_joint_vs_grid_front": hypervolume_ratio(grid_front, joint, runtime_tol=0.05)}
            for label in relaxed:
                row[f"front_points_{label}"] = int((joint.source == label).sum())
            row["expert_on_joint_front"] = bool((joint.source == "expert").any())
            out.append(row)
        logger.info("[rules] %s: %d configurations measured", name, len(params))
    return pd.DataFrame(out)


def mnist(results: Path) -> pd.DataFrame:
    from ckks_param_tool.encryption import TenSEALBackend
    from ckks_param_tool.evaluation.encrypted_run import run_encrypted_inference
    from ckks_param_tool.evaluation.tenseal_inference_evaluator import task_metrics

    run = latest_run(results / "mnist")
    _, X_client, model = workload("mnist")
    depth = model.multiplicative_depth
    logits = np.array([np.max(np.abs(model(np.asarray(x)))) for x in X_client])
    order = np.argsort(-logits)

    out, seen = [], set()
    for p in json.loads((run / "real_rows.json").read_text()):
        params = _params(p)
        if params in seen:
            continue
        seen.add(params)
        headroom = params.output_headroom_bits(depth)
        at_risk = order[logits[order] >= 2.0 ** headroom]
        # configurations without inferred risk: the 50 largest logits as a check of the margin
        digits = at_risk if len(at_risk) else order[:50]
        backend = TenSEALBackend(params)
        coefficients = []
        _, _, outputs = run_encrypted_inference(backend, model, X_client[digits], warmup_runs=0,
                                                raise_on_catastrophic=False, coefficient_maxima=coefficients)
        catastrophic = task_metrics("multiclass", outputs, overflow_error=2.0 ** headroom)["catastrophic_rate"]
        out.append({
            "N": params.poly_degree, "chain": str(params.chain), "headroom_bits": headroom,
            "inferred_at_risk": len(at_risk), "client_digits": len(logits), "encrypted_digits": len(digits),
            "measured_catastrophic": int(round(catastrophic * len(digits))),
            "largest_logit_bits": float(np.log2(logits[digits].max())),
            "largest_coefficient_bits": float(np.log2(max(coefficients))),
        })
        logger.info("[mnist] %s: %d inferred at risk, %d measured catastrophic of %d encrypted", params.chain,
                    len(at_risk), out[-1]["measured_catastrophic"], len(digits))
    return pd.DataFrame(out)


# every configuration the tool reported (knee and endpoints, all security levels) on the tabular
# workloads, encrypted on up to `rows` real rows: the coefficient headroom, and whether it is below
# the output margin (what the tool would now flag as coefficient_headroom_below_margin)
def reported(results: Path, rows: int = 500) -> pd.DataFrame:
    out = []
    for name in TABULAR:
        run = latest_run(results / name)
        if run is None:
            continue
        _, X_client, model = workload(name)
        with open(run / "tool" / "run_info.json") as f:
            margin = json.load(f)["output_margin_bits"]
        evaluator = TenSEALInferenceEvaluator(model, X_client, EvaluationConfig(num_samples=rows, warmup_runs=0))
        points = {}
        for p in json.loads((run / "real_rows.json").read_text()):
            points.setdefault(_params(p), []).append(f"{p['point']}@{p['target_security']}")
        for params, labels in points.items():
            r = evaluator.evaluate(params)
            out.append({
                "workload": name, "N": params.poly_degree, "chain": str(params.chain), "points": " ".join(labels),
                "headroom_bits": params.output_headroom_bits(model.multiplicative_depth), "rows": r.num_rows,
                "status": r.status, "catastrophic_rate": r.catastrophic_rate,
                "coefficient_headroom_bits": r.coefficient_headroom_bits,
                "below_margin": bool(r.coefficient_headroom_bits < margin),
            })
        logger.info("[reported] %s: %d configurations", name, len(points))
    return pd.DataFrame(out)


DIAGNOSTICS = {"coefficients": coefficients, "reported": reported, "dp": dp, "scales": scales, "rules": rules, "mnist": mnist}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnostic", choices=sorted(DIAGNOSTICS))
    parser.add_argument("--results", type=Path, default=Path("results/final_v3"))
    parser.add_argument("--out", type=Path, default=Path("results/diagnostics"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    df = DIAGNOSTICS[args.diagnostic](args.results)
    df.to_csv(args.out / f"{args.diagnostic}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

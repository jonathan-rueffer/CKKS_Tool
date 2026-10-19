import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ckks_param_tool import Experiment, DataDescriptor, client_server_split
from ckks_param_tool.config import load_experiment_config
from ckks_param_tool.data.generators import generate_test_vectors
from ckks_param_tool.evaluation import TenSEALParameterEvaluator, TenSEALInferenceEvaluator
from ckks_param_tool.plotting import results_to_dataframe
from ckks_param_tool.search import tutorial_baseline_params, expert_baseline_params

from .datasets import DATASETS, NO_BROAD_GRID
from .metrics import dominating_configs

'''
Runs the benchmark for one or more workloads. For every dataset a run folder

    results/<dataset>/<timestamp>/
        tool/          Experiment with the default search policy
        broad_grid/    Experiment with the broad-grid policy (skipped for NO_BROAD_GRID)
        test_vectors/  the tool's candidates re-evaluated with random dataset rows and with
                       dataset-agnostic (naive) vectors
        real_rows.json the knee points and front endpoints measured on all client rows
        baseline.json  tutorial-default parameters vs the tool's evaluated configurations

is created. Analyse the results with `python -m benchmark_suite.analyze`.
'''

logger = logging.getLogger("benchmark")


def _evaluate_candidates(params_list, model, X, config, policy):
    vectors = generate_test_vectors(X, config.data, test_vector_policy=policy)
    evaluator = TenSEALParameterEvaluator(model, vectors, config.evaluation)
    return results_to_dataframe([(p, evaluator.evaluate(p)) for p in params_list])


# real rows used to validate the reported configurations and the baselines: all client rows up to
# MAX_REAL_ROWS (a seeded random subset beyond that), fewer where one inference takes seconds
MAX_REAL_ROWS = 5000
REAL_ROWS = {"mnist": 200}


def model_quality(model, X, y) -> dict:
    import numpy as np
    from ckks_param_tool.models.base import predict

    outputs = [model(np.asarray(x)) for x in X]
    if model.task == "regression":
        pred = np.array([float(np.atleast_1d(o)[0]) for o in outputs])
        y = np.asarray(y, dtype=float)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return {"task": "regression", "r2": float(r2), "rows": len(y)}

    classes = np.unique(y)
    pred_idx = np.array([predict(model.task, o)[0] for o in outputs])
    if model.task == "binary": # sklearn: logit > 0 means the second class
        pred = classes[pred_idx]
    else: # one output per class, in sorted class order
        pred = classes[pred_idx] if len(classes) > int(pred_idx.max()) else pred_idx
    return {"task": model.task, "accuracy": float(np.mean(pred == np.asarray(y))), "rows": len(y)}


def run_dataset(name: str, output_root: Path, config=None, broad_grid: bool = True) -> Path:
    loader, trainer, _ = DATASETS[name]
    config = load_experiment_config(config).to_dict()
    config["evaluation"]["num_samples"] = REAL_ROWS.get(name, MAX_REAL_ROWS)
    config = load_experiment_config(config)

    X, y = loader()
    X_client, X_server, y_client, y_server = client_server_split(X, y)
    model = trainer(X_server, y_server)

    run_dir = output_root / name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True)
    logger.info("[%s] writing to %s", name, run_dir)

    # plaintext model quality on the client split (labels are only used for this report)
    with open(run_dir / "model_quality.json", "w") as f:
        json.dump(model_quality(model, X_client, y_client), f, indent=2)

    # 1) the tool with its default policy
    # the client side: only the descriptor of the client split enters the tool
    descriptor = DataDescriptor.from_data(X_client)

    tool = Experiment(descriptor, model, config=config)
    tool.run()
    tool.save(run_dir / "tool", timestamped=False)
    logger.info("[%s] tool: %d candidates in %.1fs", name,
                len(tool.results.param_search_results), tool.results.search_seconds)

    # 2) the broad-grid search
    if broad_grid and name not in NO_BROAD_GRID:
        grid_cfg = config.to_dict()
        grid_cfg["search"] = {"profile": "broad_grid"}
        grid_cfg["evaluation"]["measure_cost"] = False # hundreds of candidates; not needed there
        grid_cfg = load_experiment_config(grid_cfg)
        full = Experiment(descriptor, model, config=grid_cfg)
        full.run()
        full.save(run_dir / "broad_grid", timestamped=False)
        logger.info("[%s] broad grid: %d candidates in %.1fs", name,
                    len(full.results.param_search_results), full.results.search_seconds)

    # 3) the same candidates evaluated with other test vectors
    tv_dir = run_dir / "test_vectors"
    tv_dir.mkdir()
    params_list = [p for p, _ in tool.results.param_search_results]
    for policy in ("random_rows", "naive"):
        _evaluate_candidates(params_list, model, X_client, config, policy).to_csv(
            tv_dir / f"{policy}.csv", index=False
        )
    logger.info("[%s] test vector comparison done", name)

    # 4) validation on real client rows (only possible in the benchmark, where the researcher
    #    has the plaintext model and the client's records): the knee point and the front's
    #    endpoints of every security level, each distinct configuration measured once
    inference = TenSEALInferenceEvaluator(model, X_client, config.evaluation)
    measured = {}

    def on_real_rows(params):
        if params not in measured:
            measured[params] = inference.evaluate(params).to_dict()
        return measured[params]

    real_rows = []
    for sec, params, _ in tool.results.recommended_params:
        points = [("knee", params)] + [(label, p) for s, label, p, _ in tool.results.front_points if s == sec]
        for label, p in points:
            real_rows.append({
                "target_security": sec,
                "point": label,
                "poly_degree": p.poly_degree,
                "log_scale": p.log_scale,
                "chain": list(p.chain),
                "security": p.security_score,
                "real_rows": on_real_rows(p),
            })
    with open(run_dir / "real_rows.json", "w") as f:
        json.dump(real_rows, f, indent=2)
    logger.info("[%s] real-row validation done", name)

    # 5) baselines vs the tool's search: the tutorial defaults (non-expert) and the expert rule of
    #    thumb. Each baseline is measured exactly like the candidates (same representative test
    #    vectors), compared with every candidate that is at least as secure, and the matching
    #    configurations are re-measured on the same real rows.
    test_vectors = generate_test_vectors(descriptor, config.data, test_vector_policy=tool.test_vector_policy)
    tv_evaluator = TenSEALParameterEvaluator(model, test_vectors, config.evaluation)

    results = tool.results_dataframe()
    by_key = {(p.poly_degree, p.scale, p.chain): p for p, _ in tool.results.param_search_results}

    def describe(row, label):
        if row is None:
            return {"comparison": label, "found": False}
        params = by_key[(row["poly_degree"], row["scale"], row["chain"])]
        return {
            "comparison": label,
            "found": True,
            "poly_degree": params.poly_degree,
            "log_scale": params.log_scale,
            "chain": list(params.chain),
            "security": params.security_score,
            "test_vectors": {"error": float(row["max_err"]), "precision_bits": float(row["precision_bits"]),
                             "runtime": float(row["runtime"])},
            "real_rows": on_real_rows(params),
        }

    baselines = []
    for label, make in (("tutorial", tutorial_baseline_params), ("expert", expert_baseline_params)):
        params = make(model.multiplicative_depth)
        tv = tv_evaluator.evaluate(params)
        candidates = results[(results["status"] == "success") & (results["security"] >= params.security_score)]
        fastest, most_precise = (dominating_configs(candidates, tv.error, tv.runtime)
                                 if tv.status == "success" else (None, None))
        baselines.append({
            "baseline": label,
            "poly_degree": params.poly_degree,
            "log_scale": params.log_scale,
            "chain": list(params.chain),
            "security": params.security_score,
            "test_vectors": {"error": tv.error, "precision_bits": tv.precision_bits,
                             "runtime": tv.runtime, "status": tv.status, "failure_reason": tv.failure_reason},
            "real_rows": on_real_rows(params),
            "tool": [
                describe(fastest, "fastest at least as precise as the baseline"),
                describe(most_precise, "most precise at least as fast as the baseline"),
            ],
        })

    with open(run_dir / "baseline.json", "w") as f:
        json.dump({"baselines": baselines}, f, indent=2)
    logger.info("[%s] baseline done", name)

    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the CKKS parameter tool benchmarks.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=None, help="optional experiment config JSON")
    parser.add_argument("--no-broad-grid", action="store_true", help="skip the broad-grid search")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    for name in args.datasets:
        run_dataset(name, args.output, config=args.config, broad_grid=not args.no_broad_grid)


if __name__ == "__main__":
    main()

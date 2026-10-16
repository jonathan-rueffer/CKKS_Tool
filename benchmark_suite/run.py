import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ckks_param_tool import Experiment, client_server_split
from ckks_param_tool.config import load_experiment_config
from ckks_param_tool.data.generators import generate_test_vectors
from ckks_param_tool.evaluation import TenSEALParameterEvaluator, TenSEALInferenceEvaluator
from ckks_param_tool.plotting import results_to_dataframe
from ckks_param_tool.search import tutorial_baseline_params

from .datasets import DATASETS, NO_FULLER
from .metrics import dominating_configs

'''
Runs the benchmark for one or more workloads. For every dataset a run folder

    results/<dataset>/<timestamp>/
        tool/          Experiment with the default search policy
        fuller/        Experiment with the near-exhaustive policy (skipped for NO_FULLER)
        test_vectors/  the tool's candidates re-evaluated with random dataset rows and with
                       dataset-agnostic (naive) vectors
        baseline.json  tutorial-default parameters vs the tool's evaluated configurations

is created. Analyse the results with `python -m benchmark_suite.analyze`.
'''

logger = logging.getLogger("benchmark")


def _evaluate_candidates(params_list, model, X, config, policy):
    vectors = generate_test_vectors(X, config.data, test_vector_policy=policy)
    evaluator = TenSEALParameterEvaluator(model, vectors, config.evaluation)
    return results_to_dataframe([(p, evaluator.evaluate(p)) for p in params_list])


def run_dataset(name: str, output_root: Path, config=None, fuller: bool = True) -> Path:
    loader, trainer, _ = DATASETS[name]
    config = load_experiment_config(config)

    X, y = loader()
    X_client, X_server, y_client, y_server = client_server_split(X, y)
    model = trainer(X_server, y_server)

    run_dir = output_root / name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True)
    logger.info("[%s] writing to %s", name, run_dir)

    # 1) the tool with its default policy
    tool = Experiment(X_client, model, config=config)
    tool.run()
    tool.save(run_dir / "tool", timestamped=False)
    logger.info("[%s] tool: %d candidates in %.1fs", name,
                len(tool.results.param_search_results), tool.results.search_seconds)

    # 2) the near-exhaustive search
    if fuller and name not in NO_FULLER:
        fuller_cfg = load_experiment_config({
            **config.to_dict(), "search": {"profile": "fuller"}
        })
        full = Experiment(X_client, model, config=fuller_cfg)
        full.run()
        full.save(run_dir / "fuller", timestamped=False)
        logger.info("[%s] fuller: %d candidates in %.1fs", name,
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

    # 4) tutorial defaults vs the tool's search. The baseline is measured exactly like the
    #    candidates (same representative test vectors), compared with every candidate that is at
    #    least as secure, and the matching configurations are re-measured on the same real rows.
    baseline_params = tutorial_baseline_params(model.multiplicative_depth)
    test_vectors = generate_test_vectors(X_client, config.data, test_vector_policy=tool.test_vector_policy)
    baseline_tv = TenSEALParameterEvaluator(model, test_vectors, config.evaluation).evaluate(baseline_params)

    inference = TenSEALInferenceEvaluator(model, X_client, config.evaluation)
    baseline_real = inference.evaluate(baseline_params)

    candidates = tool.results_dataframe()
    candidates = candidates[(candidates["status"] == "success")
                            & (candidates["security"] >= baseline_params.security_score)]
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
            "real_rows": inference.evaluate(params).to_dict(),
        }

    fastest, most_precise = dominating_configs(candidates, baseline_tv.error, baseline_tv.runtime)

    with open(run_dir / "baseline.json", "w") as f:
        json.dump({
            "baseline": {
                "poly_degree": baseline_params.poly_degree,
                "log_scale": baseline_params.log_scale,
                "chain": list(baseline_params.chain),
                "security": baseline_params.security_score,
                "test_vectors": {"error": baseline_tv.error, "precision_bits": baseline_tv.precision_bits,
                                 "runtime": baseline_tv.runtime, "status": baseline_tv.status},
                "real_rows": baseline_real.to_dict(),
            },
            "tool": [
                describe(fastest, "fastest at least as precise as the baseline"),
                describe(most_precise, "most precise at least as fast as the baseline"),
            ],
            # the knee recommendations, for reference only
            "knee_recommendations": [
                {
                    "target_security": sec,
                    "poly_degree": p.poly_degree,
                    "log_scale": p.log_scale,
                    "chain": list(p.chain),
                    "security": p.security_score,
                    "real_rows": inf.to_dict(),
                }
                for sec, p, inf in tool.results.recommended_params
            ],
        }, f, indent=2)
    logger.info("[%s] baseline done", name)

    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the CKKS parameter tool benchmarks.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=None, help="optional experiment config JSON")
    parser.add_argument("--no-fuller", action="store_true", help="skip the near-exhaustive search")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    for name in args.datasets:
        run_dataset(name, args.output, config=args.config, fuller=not args.no_fuller)


if __name__ == "__main__":
    main()

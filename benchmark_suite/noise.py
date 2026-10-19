import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ckks_param_tool import Experiment, DataDescriptor, client_server_split
from ckks_param_tool.config import load_experiment_config
from ckks_param_tool.data.generators import generate_test_vectors
from ckks_param_tool.evaluation import TenSEALParameterEvaluator
from ckks_param_tool.search import expert_baseline_params

from .datasets import DATASETS

'''
Measures how much the two objectives vary between repeated measurements of the same
configuration, which decides when two configurations should count as equally fast or equally
precise.

Every repetition generates new keys and new encryption noise (CKKS errors are random), and times
the inferences again. For each dataset the configurations the tool reports (knee and front
endpoints of every security level) and the expert baseline are measured `repeats` times on the
same test vectors.

    python -m benchmark_suite.noise --datasets iris diabetes --repeats 5

writes results/noise/<timestamp>/noise.csv (one row per measurement) and summary.json with the
spread per configuration and the suggested tolerances:
    runtime_tolerance        = 2 x the largest coefficient of variation of the median runtime
    precision_tolerance_bits = 2 x the largest standard deviation of the precision (bits)
'''

logger = logging.getLogger("noise")


def measure(name: str, repeats: int, config=None) -> pd.DataFrame:
    loader, trainer, _ = DATASETS[name]
    config = load_experiment_config(config)

    X, y = loader()
    X_client, X_server, _, y_server = client_server_split(X, y)
    model = trainer(X_server, y_server)
    descriptor = DataDescriptor.from_data(X_client)

    tool = Experiment(descriptor, model, config=config)
    tool.run()

    configs = {p for _, p, _ in tool.results.recommended_params}
    configs |= {p for _, _, p, _ in tool.results.front_points}
    configs.add(expert_baseline_params(model.multiplicative_depth))

    no_cost = config.to_dict()
    no_cost["evaluation"]["measure_cost"] = False
    evaluator = TenSEALParameterEvaluator(
        model, generate_test_vectors(descriptor, config.data), load_experiment_config(no_cost).evaluation
    )

    rows = []
    for params in sorted(configs, key=lambda p: (p.poly_degree, p.chain)):
        for rep in range(repeats):
            r = evaluator.evaluate(params)
            rows.append({
                "dataset": name,
                "poly_degree": params.poly_degree,
                "chain": params.chain,
                "repeat": rep,
                "status": r.status,
                "precision_bits": r.precision_bits,
                "runtime": r.runtime,
            })
        logger.info("[%s] %s measured %d times", name, params.chain, repeats)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ok = df[df["status"] == "success"]
    per_config = ok.groupby(["dataset", "poly_degree", "chain"]).agg(
        n=("precision_bits", "size"),
        precision_mean=("precision_bits", "mean"),
        precision_std=("precision_bits", "std"),
        runtime_median=("runtime", "median"),
        runtime_cv=("runtime", lambda r: float(np.std(r, ddof=1) / np.mean(r))),
    ).reset_index()

    suggested = {
        "runtime_tolerance": float(2 * per_config["runtime_cv"].max()),
        "precision_tolerance_bits": float(2 * per_config["precision_std"].max()),
        "max_runtime_cv": float(per_config["runtime_cv"].max()),
        "max_precision_std_bits": float(per_config["precision_std"].max()),
        "configurations": int(len(per_config)),
    }
    return per_config, suggested


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure measurement noise of the two objectives.")
    parser.add_argument("--datasets", nargs="+", default=["iris", "diabetes", "breast_cancer"], choices=list(DATASETS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results") / "noise")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    df = pd.concat([measure(name, args.repeats, args.config) for name in args.datasets], ignore_index=True)
    per_config, suggested = summarize(df)

    out = args.output / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True)
    df.to_csv(out / "noise.csv", index=False)
    per_config.to_csv(out / "per_config.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(suggested, f, indent=2)

    print(per_config.round(4).to_string(index=False))
    print(json.dumps(suggested, indent=2))


if __name__ == "__main__":
    main()

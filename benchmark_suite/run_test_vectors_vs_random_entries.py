from pathlib import Path
import time

from ckks_param_tool import Experiment
from ckks_param_tool.search import CKKSSearchPolicy

def run_test_vectors_vs_random_entries(
    X_client,
    model,
    dataset_name,
    output_dir="benchmark_suite/test_vectors_vs_random_entries_experiments"
):

    base = Path(output_dir) / dataset_name

    # --------------------
    # Random Search
    # --------------------

    random_dir = base / "Random"

    exp_random = Experiment(
        X_client,
        model,
        test_vector_policy="random_rows"
    )

    # exp_random.run(verbose=False)

    # exp_random.save(random_dir)

    # --------------------
    # Representative Test Vector Search
    # --------------------

    rep_dir = base / "Representative"

    exp_rep = Experiment(
        X_client,
        model,
        test_vector_policy="representative"
    )

    exp_rep.run(verbose=False)

    exp_rep.save(rep_dir)

    print(f"Finished {dataset_name}")


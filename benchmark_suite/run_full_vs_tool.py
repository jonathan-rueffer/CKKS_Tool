from pathlib import Path
import time

from ckks_param_tool import Experiment
from ckks_param_tool.search import CKKSSearchPolicy


def run_full_vs_tool(
    X_client,
    model,
    dataset_name,
    output_dir="benchmark_suite/full_vs_tool_experiments"
):

    base = Path(output_dir) / dataset_name

    # --------------------
    # Tool search
    # --------------------

    tool_dir = base / "Tool"

    exp_tool = Experiment(
        X_client,
        model,
        search_policy=CKKSSearchPolicy.default()
    )
    t_start = time.perf_counter()
    exp_tool.run(verbose=False)
    t_end = time.perf_counter() 

    runtime =  t_end - t_start

    print(f"Tool Runtime: {runtime:.6f} seconds")

    exp_tool.save(tool_dir)

    # --------------------
    # Fuller search
    # --------------------

    full_dir = base / "Fuller"

    exp_full = Experiment(
        X_client,
        model,
        search_policy=CKKSSearchPolicy.fuller()
    )

    t_start = time.perf_counter()
    exp_full.run(verbose=False)
    t_end = time.perf_counter() 

    runtime =  t_end - t_start

    print(f"Full Runtime: {runtime:.6f} seconds")

    exp_full.save(full_dir)

    print(f"Finished {dataset_name}")
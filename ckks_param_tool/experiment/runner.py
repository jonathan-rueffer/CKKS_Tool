#Config
from ckks_param_tool.config import (
    load_experiment_config,
    ExperimentConfig
)

#Test vectors:
from ckks_param_tool.data import (
    generate_representative_test_vectors,
    generate_test_vectors
)

#Bit Estimation:
from ckks_param_tool.data.generators import estimate_precision_bits

#Evaluation:
from ckks_param_tool.evaluation import (
    TenSEALParameterEvaluator,
    TenSEALInferenceEvaluator
)

#Search:
from ckks_param_tool.search import (
    CKKSParameterSearchSpace,
    CKKSParameterSelector,
    CKKSSearchPolicy
)

#Plotting:
from ckks_param_tool.plotting import (
    results_to_dataframe,
    plot_runtime_vs_precision_pareto,
    plot_single_security_level
)
import matplotlib.pyplot as plt

#Return object:
from .metrics import ExperimentResult

#Saving:
from pathlib import Path
from datetime import datetime
import json

from ckks_param_tool.search.pareto import pareto_mask

'''
This file contains the class that represents the core tool logic.
'''

class Experiment:

    # this is the object that the user loads
    def __init__(self, dataset, model, config=None, test_vector_policy="representative"):
        self.dataset = dataset # the client dataset
        self.model = model # the model
        self.config = load_experiment_config(config) # optional config file
        self.results = None 
        self.pareto_df = None 
        self.test_vector_policy = test_vector_policy

    # helper function for byte conversion for printing
    def _format_bytes(self, n: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} TB"

    # helper function for simple printing the tool results (if the rich library is not installed)
    def _print_summary(self):
        pareto = self.results.best_params

        print("\n=== Experiment Summary ===")
        print(f"Pareto-optimal configurations: {len(pareto)}\n")

        by_runtime = sorted(pareto, key=lambda x: x[1].runtime)
        by_error = sorted(pareto, key=lambda x: x[1].error)

        def print_entry(title, params, inf):
            print(f"--- {title} ---")
            print(f"Poly degree     : {params.poly_degree}")
            print(f"Scale           : {params.scale}")
            print(f"Chain           : {params.chain}")
            print(f"Security (bits) : {params.security_score}")
            print(f"Total memory    : {self._format_bytes(params.memory['total'])}")
            print(f"Inference error : {inf.error:.4e}")
            print(f"Inference time  : {inf.runtime:.3f}s\n")

        print_entry("Fastest Pareto Point", *by_runtime[0])
        print_entry("Most Accurate Pareto Point", *by_error[0])
    
    def _save_config(self, run_dir: Path):
        with open(run_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _save_param_search(self, run_dir: Path):
        df = results_to_dataframe(
            self.results.param_search_results
        )
        df.to_csv(run_dir / "param_search.csv", index=False)

    def _save_plots(self, run_dir: Path):
        plots_dir = run_dir / "plots"
        plots_dir.mkdir()
        self.plot(output_dir=plots_dir, show=False)

    def _save_recommended_params(self, run_dir: Path):
        pareto_summary = []

        for params, inference in self.results.recommended_params:
            pareto_summary.append({
                "security": params.security_score,
                "poly_degree": params.poly_degree,
                "scale": params.scale,
                "chain": list(params.chain),
                "memory": params.memory,
                "inference": {
                    "error": inference.error,
                    "runtime": inference.runtime
                }
            })

        with open(run_dir / "recommended_params.json", "w") as f:
            json.dump(pareto_summary, f, indent=2)

    # function that runs the tool on the user provided dataset and model input
    def run(self, verbose: bool = False) -> ExperimentResult:

        # generate the test vectors from the dataset
        test_vectors = generate_test_vectors(
            self.dataset, 
            self.config.data,
            test_vector_policy = self.test_vector_policy
        )

        # initialze the evaluator
        evaluator = TenSEALParameterEvaluator(
            model=self.model,
            test_vectors=test_vectors
        )

        # determine the estimated bits of precision from the dataset
        bits_int, bits_dec = estimate_precision_bits(self.dataset)

        # define search policy from config
        policy = CKKSSearchPolicy.from_config(self.config.search)

        search_space = CKKSParameterSearchSpace(
            self.model.multiplicative_depth,
            bits_int,
            bits_dec,
            policy=policy
        )

        # intialize the selector
        selector = CKKSParameterSelector(
            evaluator=evaluator,
            search_space=search_space
        )

        # generate the search space and evaluate it
        param_search_results = selector.select()

        # apply pareto logic
        selection = selector.get_pareto_optimal_params()
        df_pareto = selection["df"]
        self.pareto_df = df_pareto[df_pareto["status"] == "success"]
        pareto_params = selection["pareto"]

        # get recommended parameters
        recommended_params = selection["recommended"]

        # initialize inference evaluator to evaluate recommended parameters
        inference_evaluator = TenSEALInferenceEvaluator(
            model=self.model,
            X=self.dataset,
            config=self.config.evaluation
        )
        recommended_pareto_inference_results = []

        # real inference evaluation of recommended parameters
        for params,_ in recommended_params:
            inference_result = inference_evaluator.evaluate(params)
            recommended_pareto_inference_results.append((params, inference_result))

        # store all results
        self.results = ExperimentResult(
            param_search_results=param_search_results,
            pareto_params=pareto_params,
            recommended_params=recommended_pareto_inference_results
        )

        # optional printing of results to terminal, otherwise only in saved folder
        if verbose:
            try:
                from ckks_param_tool.experiment.output import print_experiment_summary
                print_experiment_summary(self.results)
            except ImportError:
                self._print_summary()

        return self.results

    # function that plots the results
    def plot(self, output_dir=None, show=False):
        if self.results is None:
            raise RuntimeError("Experiment not run.")

        df = results_to_dataframe(self.results.param_search_results)
        df = df[df["status"] == "success"].copy()

        figs = {}

        # combined Pareto plot
        if self.pareto_df is not None:
            figs["runtime_vs_precision_pareto"] = plot_runtime_vs_precision_pareto(
                self.pareto_df, 
                self.config.plotting
            )

        # individual security level plots
        df_for_sec = self.pareto_df if self.pareto_df is not None else df
        for sec in sorted(df_for_sec["security"].unique()):
            fig = plot_single_security_level(df_for_sec, sec, self.config.plotting)
            if fig is not None:
                figs[f"runtime_vs_precision_{int(sec)}bit"] = fig

        # save figures
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            for name, fig in figs.items():
                fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

        # optionally show figures
        if show:
            for fig in figs.values():
                fig.show()

        return figs

    # function that saves the results to a dedicated folder, must be called by user
    def save(self, output_dir: str | Path):

        if self.results is None:
            raise RuntimeError("Cannot save before running experiment.")

        output_dir = Path(output_dir) 

        # time stamp for files
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        # save the JSON configuration file used to run the tool
        self._save_config(run_dir)

        # save the evaluation parameter space into a CSV
        self._save_param_search(run_dir)

        # save the plots as PNGs in a spereate subfolder
        self._save_plots(run_dir)

        # save JSON of recommended parameters and their evaluation
        self._save_recommended_params(run_dir)

        return run_dir

    
    
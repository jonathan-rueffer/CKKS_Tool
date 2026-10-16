import json
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

#Config
from ckks_param_tool.config import load_experiment_config

#Test vectors and bit estimation:
from ckks_param_tool.data.generators import (
    generate_test_vectors,
    estimate_precision_bits,
    estimate_output_int_bits,
)

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

#Return object:
from .metrics import ExperimentResult
from .environment import environment_info

'''
This file contains the class that represents the core tool logic.
'''

logger = logging.getLogger(__name__)


class Experiment:

    # this is the object that the user loads
    def __init__(self, dataset, model, config=None, test_vector_policy="representative"):
        self.dataset = dataset # the client dataset
        self.model = model # the model
        self.config = load_experiment_config(config) # optional config (path, dict or object)
        self.results: ExperimentResult | None = None
        self.pareto_df = None
        self.test_vector_policy = test_vector_policy

    # helper function for simple printing the tool results (if the rich library is not installed)
    def _print_summary(self):
        print("\n=== Experiment Summary ===")
        print(f"Evaluated candidates : {len(self.results.param_search_results)}")
        print(f"Search time          : {self.results.search_seconds:.1f}s\n")

        for sec, params, inf in self.results.recommended_params:
            print(f"--- Recommended for >= {sec}-bit security ---")
            print(f"Poly degree     : {params.poly_degree}")
            print(f"Scale           : 2^{params.log_scale}")
            print(f"Chain           : {params.chain}")
            print(f"Security (bits) : {params.security_score}")
            print(f"Inference error : {inf.error:.4e} ({inf.precision_bits:.1f} bits)")
            print(f"Inference time  : {inf.runtime:.4f}s")
            if inf.prediction_agreement is not None:
                print(f"Prediction agreement : {inf.prediction_agreement:.1%}")
            print()

    def _save_config(self, run_dir: Path):
        with open(run_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _save_param_search(self, run_dir: Path):
        self.results_dataframe().to_csv(run_dir / "param_search.csv", index=False)

        infeasible = pd.DataFrame([
            {
                "poly_degree": p.poly_degree,
                "scale": p.scale,
                "chain": p.chain,
                "reason": reason,
            }
            for p, reason in self.results.infeasible
        ], columns=["poly_degree", "scale", "chain", "reason"])
        infeasible.to_csv(run_dir / "infeasible.csv", index=False)

    def _save_plots(self, run_dir: Path):
        plots_dir = run_dir / "plots"
        plots_dir.mkdir()
        self.plot(output_dir=plots_dir, show=False)

    def _save_recommended_params(self, run_dir: Path):
        summary = []

        for sec, params, inference in self.results.recommended_params:
            summary.append({
                "target_security": sec,
                "security": params.security_score,
                "poly_degree": params.poly_degree,
                "scale": params.scale,
                "log_scale": params.log_scale,
                "chain": list(params.chain),
                "inference": inference.to_dict(),
            })

        with open(run_dir / "recommended_params.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _save_run_info(self, run_dir: Path):
        r = self.results
        info = {
            "test_vector_policy": self.test_vector_policy,
            "model_depth": self.model.multiplicative_depth,
            "input_int_bits": r.input_int_bits,
            "decimal_bits": r.decimal_bits,
            "output_int_bits": r.output_int_bits,
            "num_candidates": len(r.param_search_results),
            "num_successful": sum(res.status == "success" for _, res in r.param_search_results),
            "num_infeasible": len(r.infeasible),
            "num_step_downs": r.step_downs,
            "search_seconds": r.search_seconds,
            "environment": environment_info(self.config.evaluation.n_threads),
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # function that runs the tool on the user provided dataset and model input
    def run(self, verbose: bool = False) -> ExperimentResult:

        # generate the test vectors from the dataset
        test_vectors = generate_test_vectors(
            self.dataset,
            self.config.data,
            test_vector_policy=self.test_vector_policy
        )

        # determine the estimated bits of precision from the dataset
        bits_int, bits_dec = estimate_precision_bits(
            self.dataset,
            max_decimal_bits=self.config.data.max_decimal_bits,
        )
        if self.config.data.target_precision_bits is not None:
            bits_dec = self.config.data.target_precision_bits

        # the last prime has to hold the model output, so size it from the plaintext outputs
        # of the test vectors
        output_int_bits = estimate_output_int_bits(self.model, test_vectors)

        logger.info(
            "precision estimate: input int bits=%d, decimal bits=%d, output int bits=%d",
            bits_int, bits_dec, output_int_bits,
        )

        # define search policy from config
        policy = CKKSSearchPolicy.from_config(self.config.search)

        search_space = CKKSParameterSearchSpace(
            depth=self.model.multiplicative_depth,
            bits_decimal=bits_dec,
            output_int_bits=output_int_bits,
            policy=policy
        )

        # initialze the evaluator
        evaluator = TenSEALParameterEvaluator(
            model=self.model,
            test_vectors=test_vectors,
            config=self.config.evaluation,
        )

        # intialize the selector
        selector = CKKSParameterSelector(
            evaluator=evaluator,
            search_space=search_space
        )

        # generate the search space and evaluate it
        start = time.perf_counter()
        param_search_results = selector.select()
        search_seconds = time.perf_counter() - start

        logger.info(
            "evaluated %d candidates in %.1fs (%d infeasible skipped, %d step-downs)",
            len(param_search_results), search_seconds, len(search_space.infeasible),
            selector.step_downs,
        )

        if not any(r.status == "success" for _, r in param_search_results):
            logger.warning(
                "no candidate succeeded (%d evaluated, %d infeasible). Try a broader search "
                "profile or set data.target_precision_bits to raise the candidate scales.",
                len(param_search_results), len(search_space.infeasible),
            )

        # apply pareto logic
        selection = selector.get_pareto_optimal_params()
        df_pareto = selection["df"]
        self.pareto_df = df_pareto[df_pareto["status"] == "success"]

        # initialize inference evaluator to evaluate recommended parameters on real entries
        inference_evaluator = TenSEALInferenceEvaluator(
            model=self.model,
            X=self.dataset,
            config=self.config.evaluation
        )

        recommended = []
        for sec, params, _ in selection["recommended"]:
            recommended.append((sec, params, inference_evaluator.evaluate(params)))

        # store all results
        self.results = ExperimentResult(
            param_search_results=param_search_results,
            pareto_params=selection["pareto"],
            recommended_params=recommended,
            infeasible=list(search_space.infeasible),
            search_seconds=search_seconds,
            step_downs=selector.step_downs,
            input_int_bits=bits_int,
            decimal_bits=bits_dec,
            output_int_bits=output_int_bits,
        )

        # optional printing of results to terminal, otherwise only in saved folder
        if verbose:
            try:
                from ckks_param_tool.experiment.output import print_experiment_summary
                print_experiment_summary(self.results)
            except ImportError:
                self._print_summary()

        return self.results

    # all evaluated candidates with the per-level Pareto and recommendation flags
    def results_dataframe(self) -> pd.DataFrame:
        if self.results is None:
            raise RuntimeError("Experiment not run.")
        if self.pareto_df is not None:
            df = results_to_dataframe(self.results.param_search_results)
            flags = [c for c in self.pareto_df.columns if c not in df.columns]
            return df.join(self.pareto_df[flags]).fillna({c: False for c in flags})
        return results_to_dataframe(self.results.param_search_results)

    # function that plots the results
    def plot(self, output_dir=None, show=False):
        if self.results is None:
            raise RuntimeError("Experiment not run.")

        figs = {}

        if self.pareto_df is None or self.pareto_df.empty:
            logger.warning("no successful candidates, nothing to plot")
            return figs

        # combined Pareto plot
        figs["runtime_vs_precision_pareto"] = plot_runtime_vs_precision_pareto(
            self.pareto_df,
            self.config.plotting
        )

        # individual security level plots (each shows configurations at least that secure)
        for sec in sorted(self.pareto_df["security"].unique()):
            if sec <= 0:
                continue
            fig = plot_single_security_level(self.pareto_df, int(sec), self.config.plotting)
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

    # function that saves the results to a dedicated folder, must be called by user. By default
    # a time-stamped subfolder of output_dir is created; with timestamped=False the results are
    # written to output_dir itself (which must not exist yet)
    def save(self, output_dir: str | Path, timestamped: bool = True):

        if self.results is None:
            raise RuntimeError("Cannot save before running experiment.")

        output_dir = Path(output_dir)

        # time stamp for files
        if timestamped:
            run_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            run_dir = output_dir
        run_dir.mkdir(parents=True, exist_ok=False)

        # save the JSON configuration file used to run the tool
        self._save_config(run_dir)

        # save the evaluated parameter space (and the skipped infeasible chains) into CSVs
        self._save_param_search(run_dir)

        # save the plots as PNGs in a seperate subfolder
        self._save_plots(run_dir)

        # save JSON of recommended parameters and their evaluation
        self._save_recommended_params(run_dir)

        # save the precision estimates, search cost and software/hardware environment
        self._save_run_info(run_dir)

        return run_dir

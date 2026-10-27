import json
from dataclasses import asdict
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Config
from ckks_param_tool.config import load_experiment_config

#Test vectors and bit estimation:
from ckks_param_tool.data.descriptor import DataDescriptor
from ckks_param_tool.data.generators import (
    generate_test_vectors,
    magnitude_inputs,
    estimate_output_int_bits,
    estimate_required_precision_bits,
)

#Evaluation:
from ckks_param_tool.evaluation import TenSEALParameterEvaluator

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

    # this is the object that the user loads. It is meant to be run by the client:
    #   descriptor : the client's DataDescriptor (per-feature statistics), computed locally with
    #                DataDescriptor.from_data. No record ever enters the tool.
    #   model      : the service's model as a black box. The tool only calls it in plaintext on
    #                synthetic vectors, and on ciphertexts (the service's encrypted inference),
    #                and reads its declared multiplicative_depth and task.
    def __init__(self, descriptor, model, config=None, test_vector_policy="representative"):
        if not isinstance(descriptor, DataDescriptor):
            raise TypeError(
                "Experiment takes a DataDescriptor, not raw data: compute "
                "DataDescriptor.from_data(X) on the client and pass that"
            )
        if test_vector_policy == "random_rows":
            raise ValueError("the random_rows policy needs raw data; it is only used in the benchmark")
        self.descriptor = descriptor
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
            print(f"Error           : {inf.error:.4e} ({inf.precision_bits:.1f} bits)")
            print(f"Median runtime  : {inf.runtime:.4f}s\n")

    def _save_config(self, run_dir: Path):
        with open(run_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        # the only information about the data the tool used
        self.descriptor.save(run_dir / "descriptor.json")

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

    # a reported configuration whose measured coefficient headroom is below the output margin:
    # it worked on the test vectors, but with less spare room than the margin was meant to leave
    # (information only: it is still reported, see README "What decides what")
    def _low_headroom(self, result) -> bool:
        h = result.coefficient_headroom_bits
        return bool(np.isfinite(h) and h < self.config.data.output_margin_bits)

    def _describe(self, params, result) -> dict:
        return {
            "security": params.security_score,
            "poly_degree": params.poly_degree,
            "scale": params.scale,
            "log_scale": params.log_scale,
            "chain": list(params.chain),
            # measured on the test vectors
            "error": result.error,
            "precision_bits": result.precision_bits,
            "relative_precision_bits": result.relative_precision_bits,
            "runtime": result.runtime,
            "runtime_std": result.runtime_std,
            # diagnostic only (see TenSEALBackend.max_abs_coefficient)
            "output_headroom_bits": params.output_headroom_bits(self.model.multiplicative_depth),
            "max_output": result.max_output,
            "max_coefficient": result.max_coefficient,
            "coefficient_headroom_bits": result.coefficient_headroom_bits,
            "coefficient_headroom_below_margin": self._low_headroom(result),
        }

    # per security level: the recommendation (knee by default) and the front's endpoints
    def _save_recommended_params(self, run_dir: Path):
        summary = []

        for sec, params, result in self.results.recommended_params:
            entry = {"target_security": sec, **self._describe(params, result)}
            for s, label, p, r in self.results.front_points:
                if s == sec:
                    entry[label] = self._describe(p, r)
            summary.append(entry)

        with open(run_dir / "recommended_params.json", "w") as f:
            json.dump(summary, f, indent=2)

    # what evaluating every candidate through a remote service would cost the client: the keys
    # uploaded once per candidate, and every encrypted query's input and output ciphertexts
    def _tuning_cost(self) -> dict:
        results = [r for _, r in self.results.param_search_results]
        measured = [r for r in results if np.isfinite(r.eval_key_bytes)]
        return {
            "candidates": len(results),
            "encrypted_queries": int(sum(r.encrypted_queries for r in results)),
            "keygen_seconds": float(np.nansum([r.keygen_seconds for r in results])),
            "measured_candidates": len(measured),
            "eval_key_bytes": float(sum(r.eval_key_bytes for r in measured)),
            "ciphertext_bytes": float(sum(
                r.encrypted_queries * (r.input_ciphertext_bytes + r.output_ciphertext_bytes) for r in measured
            )),
        }

    def _save_run_info(self, run_dir: Path):
        r = self.results
        info = {
            "test_vector_policy": self.test_vector_policy,
            "model_depth": self.model.multiplicative_depth,
            "model_task": getattr(self.model, "task", "regression"),
            "input_int_bits": r.input_int_bits,
            "estimated_required_bits": r.estimated_required_bits,
            "target_precision_bits": r.target_precision_bits,
            "scale_floor_bits": r.scale_floor_bits, # the lowest scale the search considered
            "output_int_bits": r.output_int_bits,
            "output_margin_bits": self.config.data.output_margin_bits,
            "num_candidates": len(r.param_search_results),
            "num_successful": sum(res.status == "success" for _, res in r.param_search_results),
            "num_infeasible": len(r.infeasible),
            "num_step_downs": r.step_downs,
            "search_seconds": r.search_seconds,
            "tuning_cost": self._tuning_cost(),
            # the resolved search policy (the profile's defaults with the config's overrides)
            "search_policy": asdict(CKKSSearchPolicy.from_config(self.config.search)),
            # how failures were detected and retried (changed after the final_v3 benchmark runs,
            # which used "error >= max |output|" only and re-sized the final modulus on retry)
            "safety_checks": {
                "catastrophic_deviation": "error >= max |output| or error >= 2^output_headroom_bits",
                "step_down": "same final modulus, levels and special prime; scale - step_down_bits",
            },
            "environment": environment_info(self.config.evaluation.n_threads),
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # function that runs the tool on the user provided dataset and model input
    def run(self, verbose: bool = False) -> ExperimentResult:

        data_cfg = self.config.data
        # the test vectors that are encrypted during the evaluation
        test_vectors = generate_test_vectors(self.descriptor, data_cfg, test_vector_policy=self.test_vector_policy)

        # plaintext-only inputs: many samples from the descriptor plus its stress vectors
        bounds_inputs = magnitude_inputs(self.descriptor, data_cfg)

        bits_int = self.descriptor.input_int_bits()

        # the last prime has to hold the model output, so size it from the largest plaintext
        # output on the bound inputs, plus a safety margin for inputs beyond them
        output_int_bits = estimate_output_int_bits(
            self.model, np.vstack([bounds_inputs, test_vectors]), margin_bits=data_cfg.output_margin_bits
        )

        # information only: for a classifier, the precision that keeps (1 - margin_quantile) of
        # the predictions unchanged (from its decision margins on the samples)
        estimated_required_bits = estimate_required_precision_bits(
            self.model, bounds_inputs, data_cfg.margin_quantile, data_cfg.max_decimal_bits
        )
        target_bits = data_cfg.target_precision_bits

        logger.info(
            "data: input int bits=%d | output int bits=%d (incl. %d safety) | "
            "estimated precision need=%s | target=%s",
            bits_int, output_int_bits, data_cfg.output_margin_bits,
            f"{estimated_required_bits} bits" if estimated_required_bits is not None else "n/a (regression)",
            f"{target_bits} bits" if target_bits is not None else "none",
        )

        # define search policy from config
        policy = CKKSSearchPolicy.from_config(self.config.search)

        search_space = CKKSParameterSearchSpace(
            depth=self.model.multiplicative_depth,
            output_int_bits=output_int_bits,
            policy=policy,
            target_bits=target_bits,
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
            search_space=search_space,
            recommendation_policy=self.config.evaluation.recommendation,
            target_bits=target_bits,
            runtime_tol=self.config.evaluation.runtime_tolerance,
            precision_tol_bits=self.config.evaluation.precision_tolerance_bits,
            min_precision_bits=self.config.evaluation.min_precision_bits,
            shuffle_seed=self.config.evaluation.random_state,
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

        recommended = selection["recommended"] # (security, params, test-vector result)
        front_points = selection["front_points"] # (security, label, params, test-vector result)

        reported = {(p, r.coefficient_headroom_bits) for _, p, r in recommended} | \
                   {(p, r.coefficient_headroom_bits) for _, _, p, r in front_points}
        for p, h in sorted(reported, key=str):
            if np.isfinite(h) and h < data_cfg.output_margin_bits:
                logger.warning("reported configuration N=%d %s keeps only %.1f bits of coefficient headroom "
                               "(margin %d bits) on the test vectors", p.poly_degree, p.chain, h,
                               data_cfg.output_margin_bits)

        if target_bits is not None:
            met = {sec for sec, label, _, _ in front_points if label == "fastest_meeting_target"}
            for sec, _, _ in recommended:
                if sec not in met:
                    logger.warning(">= %d-bit security: no configuration reached the %d-bit target", sec, target_bits)

        # store all results
        self.results = ExperimentResult(
            param_search_results=param_search_results,
            pareto_params=selection["pareto"],
            recommended_params=recommended,
            front_points=front_points,
            infeasible=list(search_space.infeasible),
            search_seconds=search_seconds,
            step_downs=selector.step_downs,
            input_int_bits=bits_int,
            estimated_required_bits=estimated_required_bits,
            target_precision_bits=target_bits,
            scale_floor_bits=policy.lower_scale_bits(target_bits),
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

    # precision drawn as a line in the plots: the target if set, else a classifier's estimated need
    def _line_bits(self):
        r = self.results
        return r.target_precision_bits if r.target_precision_bits is not None else r.estimated_required_bits

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
            self.config.plotting,
            required_bits=self._line_bits(),
        )

        # individual security level plots (each shows configurations at least that secure)
        for sec in sorted(self.pareto_df["security"].unique()):
            if sec <= 0:
                continue
            fig = plot_single_security_level(self.pareto_df, int(sec), self.config.plotting,
                                             required_bits=self._line_bits())
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

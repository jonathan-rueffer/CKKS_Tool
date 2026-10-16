from .selector import ParameterSelector
from ckks_param_tool.params import SECURITY_LEVELS
from ckks_param_tool.plotting import results_to_dataframe
from .pareto import (
    pareto_mask,
    select_representative_pareto
)

'''
This file contains the class that defines the searching and selection of parameters using TenSEAL
'''

class CKKSParameterSelector(ParameterSelector):
    def __init__(self, evaluator, search_space, recommendation_policy: str = "knee"):
        self.evaluator = evaluator # how to evaluate a parameter set
        self.search_space = search_space # how to generate candidates
        self.recommendation_policy = recommendation_policy
        self.results = None

    # this function generates the candidate parameters and evaluates them
    def select(self):
        results = []
        for params in self.search_space.generate():

            result = self.evaluator.evaluate(params)
            results.append((params, result))

        self.results = sorted(results, key=lambda x: x[1].error) # sort by error for the output

        return self.results

    # this function applies the Pareto logic to the results. For a target security level the
    # admissible configurations are all successful ones that are AT LEAST that secure, so a
    # 256-bit configuration also competes at 128 and 192 bits.
    def get_pareto_optimal_params(self):
        df = results_to_dataframe(self.results)

        df["is_pareto"] = False
        df["is_recommended"] = False

        recommended = []

        for sec in SECURITY_LEVELS:
            df[f"pareto_{sec}"] = False
            df[f"recommended_{sec}"] = False

            subset = df[(df["status"] == "success") & (df["security"] >= sec)]

            if subset.empty:
                continue

            # calculate the Pareto-optimal points across runtime and error
            mask = pareto_mask(
                subset,
                metrics=["runtime", "max_err"]
            )

            df.loc[subset.index, f"pareto_{sec}"] = mask

            pareto_df = subset[mask]

            # apply the knee point logic
            rep = select_representative_pareto(pareto_df, self.recommendation_policy)

            df.loc[rep.name, f"recommended_{sec}"] = True
            recommended.append((sec,) + self.results[rep.name])

        level_cols = [f"pareto_{sec}" for sec in SECURITY_LEVELS]
        df["is_pareto"] = df[level_cols].any(axis=1)
        df["is_recommended"] = df[[f"recommended_{sec}" for sec in SECURITY_LEVELS]].any(axis=1)

        pareto_indices = df.index[df["is_pareto"]].tolist()

        return {
            "pareto": [self.results[i] for i in pareto_indices],
            # list of (target security level, params, result)
            "recommended": recommended,
            "df": df
        }

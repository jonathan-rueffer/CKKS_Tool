from .selector import ParameterSelector
from ckks_param_tool.params import SECURITY_LEVELS
from ckks_param_tool.plotting import results_to_dataframe
from .pareto import (
    pareto_mask,
    select_representative_pareto,
    fastest_point,
    most_precise_point,
    fastest_meeting,
)

'''
This file contains the class that defines the searching and selection of parameters using TenSEAL
'''

class CKKSParameterSelector(ParameterSelector):
    # target_bits: an optional precision target; if set, the fastest configuration meeting it is
    # reported for every security level as well
    def __init__(self, evaluator, search_space, recommendation_policy: str = "knee",
                 target_bits: float | None = None):
        self.evaluator = evaluator # how to evaluate a parameter set
        self.search_space = search_space # how to generate candidates
        self.recommendation_policy = recommendation_policy
        self.target_bits = target_bits
        self.results = None
        self.step_downs = 0

    # this function generates the candidate parameters and evaluates them. A candidate that fails
    # (e.g. overflow) is retried with the same structure at a lower scale until one works; every
    # attempt is kept in the results.
    def select(self):
        results = []
        seen = set()
        self.step_downs = 0

        for params in self.search_space.generate():
            while params is not None and params not in seen:
                seen.add(params)
                result = self.evaluator.evaluate(params)
                results.append((params, result))

                if result.status == "success":
                    break
                params = self.search_space.step_down(params)
                if params is not None and params not in seen:
                    self.step_downs += 1

        self.results = sorted(results, key=lambda x: x[1].error) # sort by error for the output

        return self.results

    # this function applies the Pareto logic to the results. For a target security level the
    # admissible configurations are all successful ones that are AT LEAST that secure, so a
    # 256-bit configuration also competes at 128 and 192 bits.
    def get_pareto_optimal_params(self):
        df = results_to_dataframe(self.results)

        df["is_pareto"] = False
        df["is_recommended"] = False
        if self.target_bits is not None:
            df["meets_target"] = (df["status"] == "success") & (df["precision_bits"] >= self.target_bits)

        recommended = []
        front_points = [] # (security, label, params, result): the front's endpoints and target match

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

            points = {"fastest": fastest_point(pareto_df), "most_precise": most_precise_point(pareto_df)}
            if self.target_bits is not None:
                points["fastest_meeting_target"] = fastest_meeting(pareto_df, self.target_bits)
            for label, row in points.items():
                if row is not None:
                    front_points.append((sec, label) + self.results[row.name])

        level_cols = [f"pareto_{sec}" for sec in SECURITY_LEVELS]
        df["is_pareto"] = df[level_cols].any(axis=1)
        df["is_recommended"] = df[[f"recommended_{sec}" for sec in SECURITY_LEVELS]].any(axis=1)

        pareto_indices = df.index[df["is_pareto"]].tolist()

        return {
            "pareto": [self.results[i] for i in pareto_indices],
            # list of (target security level, params, result)
            "recommended": recommended,
            "front_points": front_points,
            "df": df
        }

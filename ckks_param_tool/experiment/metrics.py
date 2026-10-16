from dataclasses import dataclass, field
from ckks_param_tool.params import CKKSParams
from ckks_param_tool.evaluation import InferenceEvaluationResult, ParameterEvaluationResult

'''
This file contains the class that outlines the results of running the tool.
'''

@dataclass
class ExperimentResult:

    # a list of candidate parameters and their evaluated metrics
    param_search_results: list[tuple[CKKSParams, ParameterEvaluationResult]]

    # a list of (params, metrics) that are Pareto-optimal for at least one security level
    pareto_params: list[tuple[CKKSParams, ParameterEvaluationResult]]

    # (target security level, params, real inference evaluation) for each security level
    recommended_params: list[tuple[int, CKKSParams, InferenceEvaluationResult]]

    # candidates the policy produced but SEAL cannot instantiate, with the reason
    infeasible: list[tuple[CKKSParams, str]] = field(default_factory=list)

    # wall-clock time of the candidate evaluation stage (s)
    search_seconds: float = float("nan")

    # precision estimates used to build the search space
    input_int_bits: int | None = None
    decimal_bits: int | None = None
    output_int_bits: int | None = None

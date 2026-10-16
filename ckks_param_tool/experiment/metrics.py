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

    # candidates added because a candidate failed and was retried at a lower scale
    step_downs: int = 0

    # (target security level, label, params, real inference evaluation) for the front's endpoints
    # ("fastest", "most_precise") and, if a precision target is set, "fastest_meeting_target"
    front_points: list = field(default_factory=list)

    # what the tool derived from the data descriptor and the model
    input_int_bits: int | None = None
    resolution_bits: int | None = None # finest per-feature resolution of the data
    estimated_required_bits: int | None = None # classifiers: precision that keeps predictions (info)
    target_precision_bits: int | None = None # user-provided precision target, if any
    decimal_bits: int | None = None # lower bound on the scale passed to the search space
    output_int_bits: int | None = None # integer bits of the output, safety margin included

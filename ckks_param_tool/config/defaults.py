from .experiment import (
    ExperimentConfig, 
    DataConfig, 
    EvaluationConfig, 
    PlottingConfig, 
    SearchPolicyConfig
)

'''
This file contains the default configurations used when running the tool, can be overwritten 
with a sperarate config file of the same format.
'''
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    data=DataConfig(
        num_samples=10,
        method="gaussian",
        rounding=None,
        test_size=0.3,
        random_state=42,
    ),
    evaluation=EvaluationConfig(
        num_samples=10
    ),
    plotting=PlottingConfig(
        error_tol=None
    ),
    search=SearchPolicyConfig()
)
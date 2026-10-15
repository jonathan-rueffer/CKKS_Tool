from dataclasses import dataclass
from typing import Optional
from dataclasses import asdict

'''
This file contains the tool configurations that can be controlled by a user. 
'''

# configurations relating to the data input and procession stage.
@dataclass(frozen=True)
class DataConfig:
    num_samples: int # number of representative test vectors to use
    method: str # test vector generation strategy
    rounding: Optional[int] # optional rounding of the dataset entries (default to None)
    test_size: float # percent size of the dataset that represents the client, other half used for training
    random_state: Optional[int] # random state for reproducability

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):

        # verifying correct configuration values
        if self.num_samples <= 0:
            raise ValueError("data.num_samples must be > 0")

        if self.method not in {"truncated_normal", "gaussian", "uniform", "student_t"}:
            raise ValueError(
                f"data.method must be one of "
                f"['truncated_normal', 'gaussian', 'uniform', 'student_t'], got {self.method}"
            )

        if self.rounding is not None and self.rounding < 0:
            raise ValueError("data.rounding must be >= 0 or None")

        if not (0.0 < self.test_size < 1.0):
            raise ValueError("data.test_size must be in (0,1)")


@dataclass(frozen=True)
class EvaluationConfig:
    num_samples: int # number of real dataset entries to use to evaluate recommended parameters

    def to_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError("evaluation.num_samples must be > 0")

@dataclass(frozen=True)
class PlottingConfig:
    error_tol: Optional[float] # optional error tolerance for filtering

    def to_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.error_tol is not None and self.error_tol <= 0:
            raise ValueError("plotting.error_tol must be > 0 or None")


@dataclass(frozen=True)
class SearchPolicyConfig:

    profile: str = "default"

    num_scale_options: int = 3
    scale_step: int = 10

    base_scale_buffer: int = 5

    chain_length_offsets: tuple[int,...] = (2,3)

    q0_deltas: tuple[int,...] = (0,10,20)

    max_qL: int = 60

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):

        valid_profiles = {
            "default",
            "fast",
            "full",
            "fuller",
            "custom"
        }

        if self.profile not in valid_profiles:
            raise ValueError(
                f"search.profile must be one of {valid_profiles}"
            )

        if self.num_scale_options <= 0:
            raise ValueError(
                "search.num_scale_options must be >0"
            )

        if self.scale_step <=0:
            raise ValueError(
                "search.scale_step must be >0"
            )

        if self.max_qL <=0:
            raise ValueError(
                "search.max_qL must be >0"
            )

@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    evaluation: EvaluationConfig
    plotting: PlottingConfig
    search: SearchPolicyConfig

    def to_dict(self):
        return {
            "data": self.data.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "plotting": self.plotting.to_dict(),
            "search": self.search.to_dict()
        }

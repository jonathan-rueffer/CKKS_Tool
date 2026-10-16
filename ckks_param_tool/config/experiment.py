from dataclasses import dataclass, field, asdict
from typing import Optional

'''
This file contains the tool configurations that can be controlled by a user.
'''

SAMPLING_METHODS = {"truncated_normal", "gaussian", "uniform", "student_t"}
SEARCH_PROFILES = {"default", "fast", "full", "fuller", "custom"}


# configurations relating to the data input and procession stage.
@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 10 # number of representative test vectors to use
    method: str = "truncated_normal" # test vector generation strategy
    rounding: Optional[int] = None # optional rounding of the generated test vectors
    random_state: Optional[int] = 42 # random state for reproducibility
    max_decimal_bits: int = 40 # cap on the decimal precision estimated from the data
    target_precision_bits: Optional[int] = None # if set, used instead of the data estimate

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):

        # verifying correct configuration values
        if self.num_samples <= 0:
            raise ValueError("data.num_samples must be > 0")

        if self.method not in SAMPLING_METHODS:
            raise ValueError(
                f"data.method must be one of {sorted(SAMPLING_METHODS)}, got {self.method}"
            )

        if self.rounding is not None and self.rounding < 0:
            raise ValueError("data.rounding must be >= 0 or None")

        if self.max_decimal_bits <= 0:
            raise ValueError("data.max_decimal_bits must be > 0")

        if self.target_precision_bits is not None and self.target_precision_bits <= 0:
            raise ValueError("data.target_precision_bits must be > 0 or None")


@dataclass(frozen=True)
class EvaluationConfig:
    num_samples: int = 100 # number of real dataset entries used to evaluate recommended parameters
    warmup_runs: int = 1 # untimed encrypted inferences run before timing each candidate
    timing_repeats: int = 1 # timed passes over the test vectors per candidate
    n_threads: Optional[int] = None # TenSEAL worker threads (None = all cores)
    random_state: Optional[int] = 42 # random state for selecting the real dataset entries

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError("evaluation.num_samples must be > 0")
        if self.warmup_runs < 0:
            raise ValueError("evaluation.warmup_runs must be >= 0")
        if self.timing_repeats <= 0:
            raise ValueError("evaluation.timing_repeats must be > 0")
        if self.n_threads is not None and self.n_threads <= 0:
            raise ValueError("evaluation.n_threads must be > 0 or None")


@dataclass(frozen=True)
class PlottingConfig:
    error_tol: Optional[float] = None # optional error tolerance for filtering

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if self.error_tol is not None and self.error_tol <= 0:
            raise ValueError("plotting.error_tol must be > 0 or None")


# the profile selects a preset policy (see CKKSSearchPolicy); every other field, when set,
# overrides that preset. Fields left as None keep the preset's value.
@dataclass(frozen=True)
class SearchPolicyConfig:

    profile: str = "default"

    poly_degrees: Optional[tuple[int, ...]] = None
    num_scale_options: Optional[int] = None
    scale_step: Optional[int] = None
    base_scale_buffer: Optional[int] = None
    min_scale_bits: Optional[int] = None
    budget_scales: Optional[bool] = None
    budget_extra_levels: Optional[tuple[int, ...]] = None
    extra_levels: Optional[tuple[int, ...]] = None
    q0_deltas: Optional[tuple[int, ...]] = None
    max_qL: Optional[int] = None
    special_primes: Optional[str] = None
    step_down_bits: Optional[int] = None

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):

        if self.profile not in SEARCH_PROFILES:
            raise ValueError(f"search.profile must be one of {sorted(SEARCH_PROFILES)}")

        # JSON gives lists; store tuples so the config stays hashable
        for name in ("poly_degrees", "extra_levels", "budget_extra_levels", "q0_deltas"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))

        if self.num_scale_options is not None and self.num_scale_options < 0:
            raise ValueError("search.num_scale_options must be >= 0 (0 = no limit)")

        if self.scale_step is not None and self.scale_step <= 0:
            raise ValueError("search.scale_step must be > 0")

        if self.max_qL is not None and self.max_qL <= 0:
            raise ValueError("search.max_qL must be > 0")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    search: SearchPolicyConfig = field(default_factory=SearchPolicyConfig)

    def to_dict(self):
        return {
            "data": self.data.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "plotting": self.plotting.to_dict(),
            "search": self.search.to_dict()
        }

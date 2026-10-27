from dataclasses import dataclass, field, asdict
from typing import Optional

'''
This file contains the tool configurations that can be controlled by a user.
'''

SAMPLING_METHODS = {"truncated_normal", "gaussian", "uniform", "student_t"}
SEARCH_PROFILES = {"default", "fast", "full", "broad_grid", "custom"}


# configurations relating to the data input and procession stage.
@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 50 # representative test vectors encrypted per candidate (more = less noise)
    method: str = "truncated_normal" # test vector generation strategy
    rounding: Optional[int] = None # optional rounding of the generated test vectors
    random_state: Optional[int] = 42 # random state for reproducibility
    max_decimal_bits: int = 40 # cap on the precision estimated from decision margins
    target_precision_bits: Optional[int] = None # optional precision target in bits (see README)
    magnitude_samples: int = 10000 # plaintext-only samples used to bound outputs and margins
    output_margin_bits: int = 3 # safety bits added to the estimated output magnitude (sizes q0)
    margin_quantile: float = 0.01 # classifiers: estimated precision need keeps (1 - q) of predictions

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

        if self.magnitude_samples < 0:
            raise ValueError("data.magnitude_samples must be >= 0")

        if self.output_margin_bits < 0:
            raise ValueError("data.output_margin_bits must be >= 0")

        if not 0.0 <= self.margin_quantile < 1.0:
            raise ValueError("data.margin_quantile must be in [0, 1)")


@dataclass(frozen=True)
class EvaluationConfig:
    num_samples: int = 100 # benchmark validation only: real rows per configuration (0 = all)
    warmup_runs: int = 1 # untimed encrypted inferences run before timing each candidate
    timing_repeats: int = 1 # timed passes over the test vectors per candidate
    n_threads: Optional[int] = None # TenSEAL worker threads (None = all cores)
    random_state: Optional[int] = 42 # random state for selecting the real dataset entries
    # the recommended point per security level: "knee", "fastest" or "most_accurate"
    recommendation: str = "knee"
    # differences below these count as measurement noise when points are compared: relative
    # runtime, and precision in bits (see benchmark_suite/noise.py for how to measure them)
    runtime_tolerance: float = 0.05
    precision_tolerance_bits: float = 1.5 # 2x the largest std measured by benchmark_suite/noise.py
    # reported points (knee, endpoints) must reach this precision; error >= 1 is rarely usable
    min_precision_bits: float = 0.0
    # record the cost of evaluating a candidate remotely: key generation time, serialized size
    # of the evaluation keys and of an input / output ciphertext (serialization takes time)
    measure_cost: bool = True
    # record the largest plaintext coefficient of every result and the headroom left above it
    # (diagnostic only, outside the timed region; adds about 10% to the search time)
    measure_coefficients: bool = True

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if self.num_samples < 0:
            raise ValueError("evaluation.num_samples must be >= 0 (0 = all rows)")
        if self.runtime_tolerance < 0 or self.precision_tolerance_bits < 0:
            raise ValueError("evaluation tolerances must be >= 0")
        if self.recommendation not in ("knee", "fastest", "most_accurate"):
            raise ValueError("evaluation.recommendation must be knee, fastest or most_accurate")
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
    full_grid: Optional[bool] = None
    scale_step: Optional[int] = None
    base_scale_buffer: Optional[int] = None
    min_scale_bits: Optional[int] = None
    budget_scales: Optional[bool] = None
    budget_extra_levels: Optional[tuple[int, ...]] = None
    extra_levels: Optional[tuple[int, ...]] = None
    q0_deltas: Optional[tuple[int, ...]] = None
    q0_rounding: Optional[tuple[str, ...]] = None
    two_prime_q0: Optional[bool] = None
    max_qL: Optional[int] = None
    special_primes: Optional[str] = None
    step_down_bits: Optional[int] = None

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):

        if self.profile not in SEARCH_PROFILES:
            raise ValueError(f"search.profile must be one of {sorted(SEARCH_PROFILES)}")

        # JSON gives lists; store tuples so the config stays hashable
        for name in ("poly_degrees", "extra_levels", "budget_extra_levels", "q0_deltas", "q0_rounding"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))

        if self.num_scale_options is not None and self.num_scale_options < 0:
            raise ValueError("search.num_scale_options must be >= 0 (0 = no grid)")

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

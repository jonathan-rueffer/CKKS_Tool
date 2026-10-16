import numpy as np
from scipy.stats import truncnorm
from ckks_param_tool.config import DataConfig
import math

'''
This file contains the functions central to the data processing stage: generating the
representative test vectors, and analyzing the dataset and model to determine the estimated
integer and decimal bits of precision.
'''

def _compute_feature_descriptors(X):
    return {
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0),
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0),
    }

def _sample_truncated_normal(descriptors, n_samples, rng):
    mean = descriptors["mean"]
    std = np.clip(descriptors["std"], 1e-8, None)
    min_val = descriptors["min"]
    max_val = descriptors["max"]

    # constant features (min == max) have an empty truncation interval: use the constant
    samples = np.tile(np.asarray(mean, dtype=float), (n_samples, 1))
    varying = max_val > min_val
    if np.any(varying):
        a = (min_val[varying] - mean[varying]) / std[varying]
        b = (max_val[varying] - mean[varying]) / std[varying]
        samples[:, varying] = truncnorm.rvs(
            a, b,
            loc=mean[varying],
            scale=std[varying],
            size=(n_samples, int(np.sum(varying))),
            random_state=rng
        )

    return samples

def _sample_gaussian(descriptors, n_samples, rng):
    std = np.clip(descriptors["std"], 1e-8, None)
    return rng.normal(
        loc=descriptors["mean"],
        scale=std,
        size=(n_samples, len(descriptors["mean"]))
    )

def _sample_uniform(descriptors, n_samples, rng):
    return rng.uniform(
        low=descriptors["min"],
        high=descriptors["max"],
        size=(n_samples, len(descriptors["min"]))
    )

def _sample_student_t(descriptors, n_samples, rng, df=3):
    mean = descriptors["mean"]
    std = np.clip(descriptors["std"], 1e-8, None)

    n_features = len(mean)

    scale = std * np.sqrt((df - 2) / df)

    samples = rng.standard_t(
        df,
        size=(n_samples, n_features)
    )

    samples = samples * scale + mean

    return samples

_SAMPLERS = {
    "truncated_normal": _sample_truncated_normal,
    "gaussian": _sample_gaussian,
    "uniform": _sample_uniform,
    "student_t": _sample_student_t
}

TEST_VECTOR_POLICIES = ("representative", "random_rows", "naive")


def generate_test_vectors(
    X,
    config: DataConfig,
    test_vector_policy: str = "representative",
):

    if test_vector_policy == "representative":
        return generate_representative_test_vectors(X, config)

    if test_vector_policy == "random_rows":
        return generate_random_row_test_vectors(X, config)

    if test_vector_policy == "naive":
        return generate_naive_test_vectors(X, config)

    raise ValueError(f"Unknown test vector method: {test_vector_policy}")


# generates the representative test vectors
def generate_representative_test_vectors(
    X,
    config: DataConfig
):

    method = config.method
    num_samples = config.num_samples
    rounding = config.rounding

    if method not in _SAMPLERS:
        raise ValueError(f"Unknown sampling method '{method}'")

    rng = np.random.default_rng(config.random_state)
    descriptors = _compute_feature_descriptors(X)

    test_vectors = _SAMPLERS[method](descriptors, num_samples, rng)

    if rounding is not None:
        test_vectors = np.round(test_vectors, rounding)

    return test_vectors


# relative size below which a gap between two values is treated as floating-point noise
_FLOAT_NOISE_REL = 2.0 ** -40


# estimates the precision bits from the dataset, which are used throughout the tool.
#   integer bits : bits needed for the largest input magnitude
#   decimal bits : bits needed to resolve the finest step between distinct values of any
#                  single feature. Gaps are measured per feature (gaps between values of
#                  different features carry no meaning), gaps at the level of floating-point
#                  noise are ignored, and the result is capped at max_decimal_bits.
def estimate_precision_bits(
    X,
    max_decimal_bits: int = 40,
    verbose: bool = False,
):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # integer precision
    max_abs = float(np.max(np.abs(X)))
    int_bits = max(0, math.ceil(math.log2(max_abs))) if max_abs > 0 else 1

    # decimal precision
    dec_bits = 0
    finest_gap = None
    for col in X.T:
        values = np.unique(col)
        if len(values) < 2: # constant feature
            continue
        gaps = np.diff(values)
        noise = _FLOAT_NOISE_REL * max(1.0, float(np.max(np.abs(values))))
        gaps = gaps[gaps > noise]
        if len(gaps) == 0:
            continue
        gap = float(np.min(gaps))
        finest_gap = gap if finest_gap is None else min(finest_gap, gap)
        dec_bits = max(dec_bits, math.ceil(-math.log2(gap) - 1e-9)) # tolerance for gaps like 0.4999...

    dec_bits = min(max(dec_bits, 0), max_decimal_bits)

    # for testing
    if verbose:
        print("CKKS precision estimate:")
        print(f"  Max |x|                : {max_abs}")
        print(f"  Integer bits           : {int_bits}")
        print(f"  Finest per-feature gap : {finest_gap if finest_gap is not None else 'N/A'}")
        print(f"  Decimal bits           : {dec_bits}")

    return int_bits, dec_bits


# integer bits of the largest plaintext model output on the given inputs. The last prime q0
# has to hold the output at scale Delta, so this (not the input magnitude) sizes q0.
def estimate_output_int_bits(model, inputs) -> int:
    max_abs = max(float(np.max(np.abs(np.atleast_1d(model(np.asarray(x)))))) for x in inputs)
    return max(0, math.ceil(math.log2(max_abs))) if max_abs > 0 else 0


'''
Test vector policies used to validate the representative test vectors (not used by the tool
itself).
'''

# takes random dataset entries rather than representative test vectors
def generate_random_row_test_vectors(
    X,
    config: DataConfig
):
    num_samples = config.num_samples
    rounding = config.rounding

    rng = np.random.default_rng(config.random_state)

    if num_samples > len(X):
        raise ValueError("num_samples cannot exceed dataset size when sampling without replacement")

    # Randomly select rows from dataset
    indices = rng.choice(len(X), size=num_samples, replace=False)
    test_vectors = X[indices]

    if rounding is not None:
        test_vectors = np.round(test_vectors, rounding)

    return test_vectors


# dataset-agnostic baseline: uniform values in [-1, 1] with the right number of features
def generate_naive_test_vectors(
    X,
    config: DataConfig
):
    rng = np.random.default_rng(config.random_state)
    return rng.uniform(-1.0, 1.0, size=(config.num_samples, np.asarray(X).shape[1]))

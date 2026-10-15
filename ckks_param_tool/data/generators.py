import numpy as np
from scipy.stats import truncnorm
from ckks_param_tool.config import DataConfig
import math

'''
This file contains two functions central to the data processing stage: generating the representative
test vectors, and analyzing the dataset to determine the estimated integer and decimal bits of precision
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
    std = descriptors["std"]
    std = np.clip(descriptors["std"], 1e-8, None)
    min_val = descriptors["min"]
    max_val = descriptors["max"]

    a = (min_val - mean) / std
    b = (max_val - mean) / std

    samples = truncnorm.rvs(
        a, b,
        loc=mean,
        scale=std,
        size=(n_samples, len(mean)),
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
    std = descriptors["std"]
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


def generate_test_vectors(
    X,
    config: DataConfig,
    test_vector_policy
):

    if test_vector_policy == "representative":
        return generate_representative_test_vectors(X, config)

    if test_vector_policy == "random_rows":
        return generate_random_row_test_vectors(X, config)

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

# estimates the precision bits from the dataset, which are used throughout the tool   
def estimate_precision_bits(
    X,
    integer_buffer_bits: int = 5,
    verbose: bool = False,
):
    X = np.asarray(X)

    # integer precision
    max_abs = np.max(np.abs(X))
    if max_abs == 0:
        int_bits = 1
    else:
        int_bits = max(0, math.ceil(math.log2(max_abs))) # + integer_buffer_bits
    

    # decimal precision
    flat = np.sort(np.unique(X.flatten()))
    diffs = np.diff(flat)

    nonzero_diffs = diffs[diffs > 0]


    if len(nonzero_diffs) == 0: # accounts for all values the same
        dec_bits = 0
    else:
        delta_min = np.min(nonzero_diffs)
        dec_bits = math.ceil(-math.log2(delta_min))

    # for testing
    if verbose:
        print("CKKS precision estimate:")
        print(f"  Max |x|                : {max_abs}")
        print(f"  Integer bits (with buf): {int_bits}")
        print(f"  Min nonzero gap        : {nonzero_diffs.min() if len(nonzero_diffs) else 'N/A'}")
        print(f"  Decimal bits    : {dec_bits}")

    return int_bits, dec_bits


# testing function that takes random dataset entries rather than representative test vectors:

def generate_random_row_test_vectors(
    X,
    config: DataConfig
):
    """
    Generate random test vectors sampled directly from the dataset.

    Parameters
    ----------
    X : np.ndarray
        Client-side feature matrix.
    config : DataConfig
        Configuration containing num_samples and optional rounding.

    Returns
    -------
    np.ndarray
        Shape (num_samples, num_features)
    """

    num_samples = config.num_samples
    rounding = config.rounding

    rng = np.random.default_rng(42)

    if num_samples > len(X):
        raise ValueError("num_samples cannot exceed dataset size when sampling without replacement")

    # Randomly select rows from dataset
    indices = rng.choice(len(X), size=num_samples, replace=False)
    test_vectors = X[indices]

    if rounding is not None:
        test_vectors = np.round(test_vectors, rounding)

    return test_vectors

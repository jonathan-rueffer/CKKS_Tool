import numpy as np
from scipy.stats import truncnorm
from ckks_param_tool.config import DataConfig
from .descriptor import DataDescriptor
import math

'''
This file contains the functions central to the data processing stage. Everything is computed
from a DataDescriptor (per-feature statistics), so the raw data is not needed:
    - the representative test vectors (encrypted during the evaluation)
    - many more plaintext-only samples plus stress vectors, used to bound the model's outputs
      (which sizes q0) and its decision margins (the precision a classifier needs)
'''

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


def _descriptor(source) -> DataDescriptor:
    return source if isinstance(source, DataDescriptor) else DataDescriptor.from_data(source)


# generates the test vectors used for encrypted evaluation. `source` is either the raw data or
# a DataDescriptor; the random_rows policy (a validation baseline) needs the raw data.
def generate_test_vectors(
    source,
    config: DataConfig,
    test_vector_policy: str = "representative",
):

    if test_vector_policy == "representative":
        return generate_representative_test_vectors(source, config)

    if test_vector_policy == "random_rows":
        if isinstance(source, DataDescriptor):
            raise ValueError("the random_rows policy needs the raw data, not a descriptor")
        return generate_random_row_test_vectors(source, config)

    if test_vector_policy == "naive":
        return generate_naive_test_vectors(source, config)

    raise ValueError(f"Unknown test vector method: {test_vector_policy}")


def _sample(descriptor: DataDescriptor, method: str, n: int, seed) -> np.ndarray:
    if method not in _SAMPLERS:
        raise ValueError(f"Unknown sampling method '{method}'")
    rng = np.random.default_rng(seed)
    return _SAMPLERS[method](descriptor.arrays(), n, rng)


# generates the representative test vectors from the descriptor
def generate_representative_test_vectors(
    source,
    config: DataConfig
):
    test_vectors = _sample(_descriptor(source), config.method, config.num_samples, config.random_state)

    if config.rounding is not None:
        test_vectors = np.round(test_vectors, config.rounding)

    return test_vectors


# plaintext-only inputs used to bound the model's outputs and margins: many representative
# samples (cheap, since they are never encrypted) plus the stress vectors of the descriptor
def magnitude_inputs(source, config: DataConfig) -> np.ndarray:
    descriptor = _descriptor(source)
    seed = None if config.random_state is None else config.random_state + 1
    samples = _sample(descriptor, config.method, config.magnitude_samples, seed)
    return np.vstack([samples, descriptor.stress_vectors()])


# estimates the precision bits from the dataset: (input integer bits, finest per-feature
# resolution in bits, capped). Kept for convenience; the descriptor computes both.
def estimate_precision_bits(
    X,
    max_decimal_bits: int = 40,
    verbose: bool = False,
):
    descriptor = _descriptor(X)
    int_bits, dec_bits = descriptor.input_int_bits(), descriptor.resolution_bits(max_decimal_bits)
    if verbose:
        print(f"CKKS precision estimate: integer bits {int_bits}, decimal bits {dec_bits}")
    return int_bits, dec_bits


def _outputs(model, inputs) -> list[np.ndarray]:
    return [np.atleast_1d(np.asarray(model(np.asarray(x)), dtype=float)) for x in inputs]


# integer bits of the largest plaintext model output on the given inputs, plus margin_bits of
# safety. The last prime q0 has to hold the output at scale Delta, so this sizes q0.
def estimate_output_int_bits(model, inputs, margin_bits: int = 0) -> int:
    max_abs = max(float(np.max(np.abs(o))) for o in _outputs(model, inputs))
    bits = max(0, math.ceil(math.log2(max_abs))) if max_abs > 0 else 0
    return bits + margin_bits


# the precision (bits) a classification model needs so that the predicted class does not change
# on at least (1 - quantile) of the inputs. An absolute error e on every output flips
#   binary     : the class if e >= |logit|
#   multiclass : the argmax if 2e >= (largest - second largest logit)
# so the tolerated error of an input is its margin (|logit|, or half the gap between the two
# largest logits). Returns None for regression, which has no decision margin.
def estimate_required_precision_bits(model, inputs, quantile: float = 0.01, max_bits: int = 40) -> int | None:
    task = getattr(model, "task", "regression")
    if task == "regression":
        return None

    outputs = np.array(_outputs(model, inputs))
    if task == "binary":
        margins = np.abs(outputs[:, 0])
    else:
        top = np.sort(outputs, axis=1)
        margins = (top[:, -1] - top[:, -2]) / 2

    margin = float(np.quantile(margins, quantile))
    if margin <= 0:
        return max_bits
    return min(max(0, math.ceil(-math.log2(margin))), max_bits)


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
    test_vectors = np.asarray(X)[indices]

    if rounding is not None:
        test_vectors = np.round(test_vectors, rounding)

    return test_vectors


# dataset-agnostic baseline: uniform values in [-1, 1] with the right number of features
def generate_naive_test_vectors(
    source,
    config: DataConfig
):
    rng = np.random.default_rng(config.random_state)
    return rng.uniform(-1.0, 1.0, size=(config.num_samples, _descriptor(source).num_features))

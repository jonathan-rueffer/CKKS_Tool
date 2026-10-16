import numpy as np

from ckks_param_tool.data.generators import estimate_precision_bits, estimate_output_int_bits
from ckks_param_tool.evaluation.metrics import precision_bits
from ckks_param_tool.models import LinearModel


def test_gaps_are_per_feature():
    # each feature has resolution 0.5 (1 bit); values of different features are 1e-6 apart,
    # which the old flattened estimate would have turned into 20 bits
    X = np.array([[0.0, 1e-6], [0.5, 0.500001], [1.0, 1.000001]])
    assert estimate_precision_bits(X)[1] == 1


def test_float_noise_is_ignored():
    X = np.array([[0.1], [0.1 + 1e-17], [0.2]])
    assert estimate_precision_bits(X)[1] == 4 # gap 0.1 -> ceil(3.32)


def test_decimal_bits_are_capped():
    X = np.array([[0.0], [2.0**-50], [1.0]])
    assert estimate_precision_bits(X, max_decimal_bits=40)[1] <= 40


def test_integer_bits():
    X = np.array([[100.0, -300.0]])
    assert estimate_precision_bits(X)[0] == 9


def test_output_int_bits():
    model = LinearModel(np.array([10.0, 10.0]), 5.0)
    assert estimate_output_int_bits(model, [np.array([10.0, 10.0])]) == 8 # 205 -> 8 bits


def test_precision_bits():
    assert precision_bits(2**-10) == 10
    assert np.isnan(precision_bits(float("inf")))


def test_truncated_normal_handles_constant_features():
    from ckks_param_tool.config import DataConfig
    from ckks_param_tool.data.generators import generate_representative_test_vectors

    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    vectors = generate_representative_test_vectors(X, DataConfig(num_samples=5))
    assert np.all(vectors[:, 0] == 0.0)
    assert np.all((vectors[:, 1] >= 1.0) & (vectors[:, 1] <= 3.0))

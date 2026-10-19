import numpy as np

from ckks_param_tool import DataDescriptor
from ckks_param_tool.data.generators import estimate_output_int_bits
from ckks_param_tool.evaluation.metrics import precision_bits
from ckks_param_tool.models import LinearModel


def test_integer_bits():
    X = np.array([[100.0, -300.0]])
    assert DataDescriptor.from_data(X).input_int_bits() == 9


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

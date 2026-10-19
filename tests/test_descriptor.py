import numpy as np
import pandas as pd
import pytest

from ckks_param_tool import DataDescriptor
from ckks_param_tool.config import DataConfig
from ckks_param_tool.data import (
    generate_test_vectors,
    magnitude_inputs,
    estimate_output_int_bits,
    estimate_required_precision_bits,
)
from ckks_param_tool.models import LinearModel


X = np.array([[0.0, 10.0, 5.0], [0.5, 20.0, 5.0], [1.0, 40.0, 5.0], [0.5, 10.0, 5.0]])


def test_descriptor_from_data_and_roundtrip(tmp_path):
    d = DataDescriptor.from_data(X)
    assert d.num_features == 3
    assert d.min == (0.0, 10.0, 5.0) and d.max == (1.0, 40.0, 5.0)
    assert set(d.to_dict()) == {"mean", "std", "min", "max"} # nothing else is shared
    assert d.input_int_bits() == 6 # 40 -> 6 bits

    d.save(tmp_path / "d.json")
    assert DataDescriptor.load(tmp_path / "d.json") == d


def test_stress_vectors_cover_each_extreme():
    d = DataDescriptor.from_data(X)
    s = d.stress_vectors()
    assert s.shape == (6, 3)
    assert s[3, 1] == 40.0 and s[3, 0] == pytest.approx(0.5) # feature 1 at its max, others at mean


def test_test_vectors_need_only_the_descriptor():
    d = DataDescriptor.from_data(X)
    cfg = DataConfig(num_samples=4)
    np.testing.assert_allclose(generate_test_vectors(d, cfg), generate_test_vectors(X, cfg))
    with pytest.raises(ValueError):
        generate_test_vectors(d, cfg, test_vector_policy="random_rows")


def test_output_bits_cover_single_feature_outliers():
    # one rare, very large feature value: random samples miss it, the stress vectors do not
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1000, 2))
    data[0, 1] = 500.0
    model = LinearModel(np.array([1.0, 1.0]), 0.0)
    inputs = magnitude_inputs(DataDescriptor.from_data(data), DataConfig(magnitude_samples=100))
    real_max = np.max(np.abs(data @ model.W))
    assert 2 ** estimate_output_int_bits(model, inputs) >= real_max
    assert estimate_output_int_bits(model, inputs, margin_bits=3) == estimate_output_int_bits(model, inputs) + 3


def test_required_precision_from_margins():
    binary = LinearModel(np.array([1.0]), 0.0, task="binary")
    # margins |logit| = 0.25 ... 4: the 0-quantile margin 0.25 needs 2 bits
    inputs = np.array([[0.25], [-1.0], [4.0]])
    assert estimate_required_precision_bits(binary, inputs, quantile=0.0) == 2

    multi = LinearModel(np.eye(2), np.zeros(2), task="multiclass")
    # gap between the two logits 0.5 -> tolerated error 0.25 -> 2 bits
    assert estimate_required_precision_bits(multi, np.array([[1.0, 0.5]]), quantile=0.0) == 2

    regression = LinearModel(np.array([1.0]), 0.0)
    assert estimate_required_precision_bits(regression, inputs) is None


def test_front_endpoints_and_target_match():
    from ckks_param_tool.search.pareto import fastest_point, most_precise_point, fastest_meeting

    front = pd.DataFrame({"runtime": [1.0, 1.02, 3.0], "max_err": [2**-5, 2**-30, 2**-35]})
    # 1.0 and 1.02 are equally fast within 5%: the fastest endpoint is the more precise of them
    assert fastest_point(front).name == 1
    assert most_precise_point(front).name == 2
    assert fastest_meeting(front, 32).name == 2
    assert fastest_meeting(front, 40) is None


def test_private_descriptor():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(20000, 3))
    bounds = (np.full(3, -5.0), np.full(3, 15.0))

    d = DataDescriptor.private(data, epsilon=1.0, bounds=bounds, rng=1)
    assert d.min == (-5.0,) * 3 and d.max == (15.0,) * 3 # the public bounds are released
    np.testing.assert_allclose(d.mean, 5.0, atol=0.1)     # little noise for n = 20,000
    np.testing.assert_allclose(d.std, 2.0, atol=0.3)

    # smaller epsilon, more noise; same seed, reproducible
    assert DataDescriptor.private(data, 1.0, bounds, rng=1) == d
    tiny = DataDescriptor.private(data[:50], epsilon=0.01, bounds=bounds, rng=1)
    assert all(-5.0 <= m <= 15.0 for m in tiny.mean) # clipped into the bounds

    with pytest.raises(ValueError):
        DataDescriptor.private(data, epsilon=0.0, bounds=bounds)

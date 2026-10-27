import numpy as np
import pytest

from ckks_param_tool import DataDescriptor
from ckks_param_tool.config import DataConfig, EvaluationConfig
from ckks_param_tool.data.generators import estimate_output_int_bits, magnitude_inputs
from ckks_param_tool.models import LinearModel

'''
Known failure modes of the output bound, and which of them the tool catches. The output bound is
estimated from plaintext outputs on inputs drawn from the descriptor (independent per-feature
truncated normals plus one-feature-at-a-time stress vectors), plus a 3-bit margin.

    failure mode                                   caught?
    correlated / jointly extreme features          no: the bound is too small and the test
                                                   vectors (from the same descriptor) never
                                                   reach the joint extreme
    small output, large values in other slots      yes: the overflow happens on every input, so
    (a large dot product cancelled by a bias       the test vectors flag it and the retry with
    leaves partial sums + bias in the other slots) more headroom resolves it
    large intermediate values                      harmless: the arithmetic is modular, only the
                                                   decrypted result must fit (see
                                                   tests/test_coefficient_headroom.py)
    single feature at its descriptor min / max     yes: the stress vectors cover it
    inputs beyond the descriptor's range           only where such rows are evaluated (counted
                                                   as catastrophic on real rows), not prevented
'''

CONFIG = DataConfig(magnitude_samples=2000)


def _needed_bits(model, X):
    return int(np.ceil(np.log2(max(float(np.max(np.abs(model(x)))) for x in X))))


def test_jointly_extreme_correlated_features_are_underestimated():
    # 2048 perfectly correlated features (copies of one standard normal): the real output reaches
    # d * max(z), independent sampling only about 4 * sqrt(d), roughly 5 bits less
    d = 2048
    z = np.random.default_rng(0).normal(size=300)
    X = np.repeat(z[:, None], d, axis=1)
    model = LinearModel(np.ones(d), 0.0)
    desc = DataDescriptor.from_data(X)

    b_raw = estimate_output_int_bits(model, magnitude_inputs(desc, CONFIG))
    real_max = max(float(np.max(np.abs(model(x)))) for x in X)
    assert 2 ** (b_raw + 3) < real_max # short even with the 3-bit margin: outside the guarantee
    # the stress vectors move one feature at a time and cannot see the joint extreme
    stress = max(float(np.max(np.abs(model(x)))) for x in desc.stress_vectors())
    assert stress < real_max / 2 ** 6


def test_single_feature_extremes_are_covered_by_the_stress_vectors():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    X[0, 3] = 40.0 # one rare, extreme value
    model = LinearModel(np.array([0.1, 0.1, 0.1, 10.0, 0.1]), 0.0)
    desc = DataDescriptor.from_data(X)
    b_raw = estimate_output_int_bits(model, magnitude_inputs(desc, CONFIG))
    assert 2 ** b_raw >= max(float(np.max(np.abs(model(x)))) for x in X)


def test_inputs_beyond_the_descriptor_range_exceed_the_bound():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(1000, 4))
    model = LinearModel(np.full(4, 4.0), 0.0)
    desc = DataDescriptor.from_data(X)
    b_out = estimate_output_int_bits(model, magnitude_inputs(desc, CONFIG), margin_bits=3)
    shifted = X * 40 # a later batch far outside the client's range
    assert _needed_bits(model, shifted) > b_out


def _evaluate(model, inputs, params):
    pytest.importorskip("tenseal")
    from ckks_param_tool.evaluation import TenSEALParameterEvaluator
    return TenSEALParameterEvaluator(model, inputs, EvaluationConfig(warmup_runs=0, measure_cost=False)).evaluate(params)


def test_cancelling_terms_overflow_is_caught_and_resolved_by_the_retry():
    # the output is about 1, but the dot product (16) is computed before the bias (-15) is added:
    # TenSEAL broadcasts the bias to every slot and keeps partial sums in the others, so the
    # plaintext polynomial's coefficients reach about 2^3.9, above 3 bits of headroom
    from ckks_param_tool.params import CKKSParams
    from ckks_param_tool.search import CKKSParameterSearchSpace, CKKSParameterSelector, CKKSSearchPolicy
    from ckks_param_tool.evaluation import TenSEALParameterEvaluator

    model = LinearModel(np.full(4, 4.0), -15.0)
    inputs = np.ones((3, 4)) + np.random.default_rng(0).uniform(0, 1e-3, size=(3, 4))
    assert max(abs(float(model(x))) for x in inputs) < 2 # the output alone needs 1 bit

    lacking = _evaluate(model, inputs, CKKSParams(8192, 2**40, (44, 40, 44))) # headroom 3 bits
    assert lacking.status == "failure" and "catastrophic" in lacking.failure_reason

    enough = _evaluate(model, inputs, CKKSParams(8192, 2**40, (49, 40, 49))) # headroom 8 bits
    assert enough.status == "success" and enough.error < 1e-4
    # the measured coefficients, not the outputs, are what the headroom has to hold
    assert enough.max_coefficient > 4 * enough.max_output
    assert 0 < enough.coefficient_headroom_bits < 8 - 3

    # the tool's search with an output-based bound of 3 bits recovers through the retry
    policy = CKKSSearchPolicy(poly_degrees=(8192,), two_prime_q0=False, budget_extra_levels=())
    selector = CKKSParameterSelector(
        TenSEALParameterEvaluator(model, inputs, EvaluationConfig(warmup_runs=0, measure_cost=False)),
        CKKSParameterSearchSpace(1, output_int_bits=3, policy=policy))
    results = selector.select()
    assert selector.step_downs > 0
    assert any(r.status == "success" for _, r in results)
    assert all(r.coefficient_headroom_bits > 0 for _, r in results if r.status == "success")


@pytest.mark.parametrize("weight,bias", [
    (0.01, 2.0), (0.01, 6.0), (0.01, 12.0), (0.01, 40.0), # a broadcast bias fills every slot
    (2.0**10, 0.0), # one large output in one slot: spread over all coefficients
])
def test_coefficient_measurement_predicts_overflow(weight, bias):
    # the coefficients measured with ample headroom predict what happens with 3 bits of headroom:
    # an overflow exactly when the largest coefficient reaches 2^3, whatever the output size
    from ckks_param_tool.params import CKKSParams

    model = LinearModel(np.array([weight]), bias)
    inputs = np.array([[1.0]])
    coefficient_bits = np.log2(_evaluate(model, inputs, CKKSParams(8192, 2**40, (60, 40, 60))).max_coefficient)
    overflowed = _evaluate(model, inputs, CKKSParams(8192, 2**40, (44, 40, 44))).status == "failure"
    assert overflowed == (coefficient_bits >= 3)
    if weight > 1:
        # the output (2^10) far exceeds the nominal headroom, yet nothing overflows
        assert not overflowed and np.log2(abs(float(model(inputs[0])))) > 3

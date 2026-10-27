import numpy as np
import pytest

pytest.importorskip("tenseal")

from ckks_param_tool.config import EvaluationConfig
from ckks_param_tool.evaluation import TenSEALParameterEvaluator
from ckks_param_tool.models import LinearModel
from ckks_param_tool.models.wrapper import WrappedModel
from ckks_param_tool.params import CKKSParams

'''
What coefficient_headroom_bits measures, and when it predicts overflow.

CKKS additions, multiplications and rescalings preserve the message modulo the current modulus,
so a value may exceed the modulus in the middle of a computation without harm: an overflow can
only happen at decryption, when the coefficients of the decrypted plaintext (in units of the
scale) reach half the modulus left after the last rescale. The tool reads these coefficients
from every decoded slot (TenSEALBackend.max_abs_coefficient), so on the evaluated inputs the
measured headroom predicts overflow for any sequence of operations, under these assumptions:
the result did not overflow (a wrapped result shows wrapped coefficients), the slots' imaginary
parts are noise (TenSEAL encodes real vectors), and the nominal prime sizes and scale are used
(SEAL's primes lie just below 2^bits, so cases within ~0.1 bit of the threshold are skipped).
It says nothing about inputs that were not evaluated.

Covered below: broadcast and per-slot biases, single-slot dot products, matmuls, a two-prime
final modulus, a spare level, a depth-3 network with a square activation, intermediate values
far above the headroom, and the dependence on the inputs.
'''

CFG = EvaluationConfig(warmup_runs=0, measure_cost=False)
TIGHT = CKKSParams(8192, 2**40, (44, 40, 44)) # 3 bits of headroom


def _evaluate(model, inputs, params, config=CFG):
    return TenSEALParameterEvaluator(model, inputs, config).evaluate(params)


def _predicts(model, inputs, ample, tight):
    depth = model.multiplicative_depth
    measured = _evaluate(model, inputs, ample)
    assert measured.status == "success"
    coefficient_bits = np.log2(measured.max_coefficient)
    headroom = tight.output_headroom_bits(depth)
    if abs(coefficient_bits - headroom) < 0.1:
        pytest.skip("too close to the threshold to call")
    # the headroom measured with the ample parameters, transferred to the tight ones
    assert measured.coefficient_headroom_bits == pytest.approx(
        ample.output_headroom_bits(depth) - coefficient_bits)
    overflowed = _evaluate(model, inputs, tight).status == "failure"
    assert overflowed == (coefficient_bits >= headroom)
    return coefficient_bits


@pytest.mark.parametrize("bias", [2.0**21, 2.0**22.5, 2.0**23.5, 2.0**25])
def test_two_prime_final_modulus(bias):
    # final modulus (24, 40) after one rescale: 64 - 40 - 1 = 23 bits of headroom
    tight = CKKSParams(8192, 2**40, (24, 40, 40, 60))
    assert tight.output_headroom_bits(1) == 23
    _predicts(LinearModel(np.array([0.01]), bias), np.array([[1.0]]), CKKSParams(8192, 2**40, (60, 40, 40, 60)), tight)


def test_spare_level_is_headroom():
    # the spare prime is never consumed, so it stays in the final modulus
    params = CKKSParams(8192, 2**40, (44, 40, 40, 44))
    assert params.output_headroom_bits(1) == 43
    r = _evaluate(LinearModel(np.array([0.01]), 2.0**20), np.array([[1.0]]), params)
    assert r.status == "success" and r.coefficient_headroom_bits == pytest.approx(23, abs=0.1)


@pytest.mark.parametrize("bias", [-6.0, 6.0, 20.0])
def test_per_slot_bias_on_a_matmul(bias):
    model = LinearModel(np.full((4, 2), 0.5), np.array([bias, bias]))
    _predicts(model, np.ones((2, 4)), CKKSParams(8192, 2**40, (60, 40, 60)), TIGHT)


@pytest.mark.parametrize("k", [0, 2, 4, 6])
def test_depth_three_network_with_square_activation(k):
    rng = np.random.default_rng(0)
    W1, W2 = rng.normal(size=(4, 3)), rng.normal(size=(3, 2)) * 2.0**k

    def encrypted(x):
        h = x.matmul(W1)
        h.square_()
        return h.matmul(W2) + [0.5, 0.5]

    model = WrappedModel(lambda x: ((np.asarray(x) @ W1) ** 2) @ W2 + 0.5, depth=3,
                         encrypted_forward_fn=encrypted)
    inputs = rng.uniform(-1, 1, size=(3, 4))
    _predicts(model, inputs, CKKSParams(8192, 2**30, (60, 30, 30, 30, 60)),
              CKKSParams(8192, 2**30, (34, 30, 30, 30, 34)))


@pytest.mark.parametrize("c", [4.0, 2.0**10])
def test_intermediate_values_above_the_headroom_are_harmless(c):
    # x @ W * c reaches 4c (up to 2^12) before a broadcast bias cancels it to 1: the arithmetic is
    # modular, so only the decrypted result has to fit the 3 bits of headroom
    W = np.full((4, 2), c)
    model = WrappedModel(lambda x: np.asarray(x) @ W - 4 * c + 1.0, depth=1,
                         encrypted_forward_fn=lambda x: x.matmul(W) + (-4 * c + 1.0))
    r = _evaluate(model, np.ones((2, 4)), TIGHT)
    assert r.status == "success" and r.error < 1e-2
    assert r.coefficient_headroom_bits == pytest.approx(3.0, abs=0.1) # output 1 in every slot


def test_headroom_depends_on_the_evaluated_inputs():
    model = LinearModel(np.full((4, 2), 1.0), np.zeros(2))
    small = _evaluate(model, np.full((2, 4), 0.25), TIGHT)
    large = _evaluate(model, np.full((2, 4), 1.5), TIGHT)
    assert small.coefficient_headroom_bits == pytest.approx(large.coefficient_headroom_bits + np.log2(6), abs=0.1)
    # with inputs beyond the evaluated ones the same configuration overflows
    assert _evaluate(model, np.full((2, 4), 3.0), TIGHT).status == "failure"


def test_failed_and_unmeasured_results_carry_no_headroom():
    model = LinearModel(np.array([0.01]), 50.0) # overflows the 3 bits
    failed = _evaluate(model, np.array([[1.0]]), TIGHT)
    assert failed.status == "failure" and np.isnan(failed.coefficient_headroom_bits)
    off = _evaluate(model, np.array([[1.0]]), CKKSParams(8192, 2**40, (60, 40, 60)),
                    EvaluationConfig(warmup_runs=0, measure_cost=False, measure_coefficients=False))
    assert off.status == "success" and np.isnan(off.max_coefficient) and np.isnan(off.coefficient_headroom_bits)

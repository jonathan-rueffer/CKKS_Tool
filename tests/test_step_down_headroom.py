import numpy as np
import pytest

pytest.importorskip("tenseal")

from ckks_param_tool.config import EvaluationConfig
from ckks_param_tool.evaluation import TenSEALParameterEvaluator
from ckks_param_tool.models import LinearModel
from ckks_param_tool.params import CKKSParams
from ckks_param_tool.search import CKKSParameterSearchSpace, CKKSParameterSelector, CKKSSearchPolicy

'''
Regression test for the step-down retry: a candidate whose final modulus lacks headroom for the
output overflows (a catastrophic deviation), and the retry at a lower scale with the same final
modulus resolves it. The output bound is deliberately underestimated (3 bits for outputs of
about 50), as an unrepresentative descriptor would. The bias makes the overflow deterministic:
TenSEAL adds a scalar to every slot, so the nominal headroom is exact (see
tests/test_failure_modes.py).
'''

MODEL = LinearModel(np.full(4, 0.01), 50.0)
INPUTS = np.random.default_rng(0).uniform(0, 1, size=(5, 4))
CONFIG = EvaluationConfig(warmup_runs=0, measure_cost=False)


def _space():
    policy = CKKSSearchPolicy(poly_degrees=(8192,), two_prime_q0=False, budget_extra_levels=(),
                              special_primes="min_max")
    return CKKSParameterSearchSpace(1, output_int_bits=3, policy=policy)


def test_lacking_headroom_overflows_and_the_retry_resolves_it():
    evaluator = TenSEALParameterEvaluator(MODEL, INPUTS, CONFIG)
    space = _space()
    original = next(p for p in space.generate() if p.chain == (59, 55, 59)) # headroom 3 bits
    failed = evaluator.evaluate(original)
    assert failed.status == "failure" and "catastrophic" in failed.failure_reason

    retry = space.step_down(original)
    assert retry.chain == (59, 50, 59) # headroom 8 bits >= log2(50)
    ok = evaluator.evaluate(retry)
    assert ok.status == "success" and ok.error < 1e-3


def test_resizing_the_final_modulus_would_not_have_helped():
    # the previous retry rule (final modulus re-sized for the lower scale) keeps 3 bits of
    # headroom and overflows again
    evaluator = TenSEALParameterEvaluator(MODEL, INPUTS, CONFIG)
    assert evaluator.evaluate(CKKSParams(8192, 2**50, (54, 50, 54))).status == "failure"


def test_selector_ends_with_working_candidates_at_every_security_level():
    selector = CKKSParameterSelector(TenSEALParameterEvaluator(MODEL, INPUTS, CONFIG), _space())
    results = selector.select()
    assert selector.step_downs > 0
    failed = [p for p, r in results if r.status != "success"]
    working = [p for p, r in results if r.status == "success"]
    assert failed and working
    assert all(p.chain[0] - p.log_scale - 1 >= 6 for p in working) # log2(50) < 6
    assert {p.security_score for p in working} == {128, 192, 256}

import pytest

from ckks_param_tool.config import SearchPolicyConfig
from ckks_param_tool.params import ALLOWED_CHAIN_BITS, SECURITY_TABLES, CKKSParams
from ckks_param_tool.search import CKKSParameterSearchSpace, CKKSSearchPolicy
from ckks_param_tool.search.policy import ChainStructure


def _chains(policy, n, scale, depth, out):
    return [c for c, _ in policy.generate_chains(n, scale, depth, out)]


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_single_prime_chain_has_depth_plus_two_primes(depth):
    policy = CKKSSearchPolicy(two_prime_q0=False)
    for chain in _chains(policy, 16384, 30, depth, out=8):
        assert len(chain) == depth + 2
        assert all(q == 30 for q in chain[1:-1])


def test_exact_q0_holds_output_with_nothing_to_spare():
    policy = CKKSSearchPolicy()
    for chain in _chains(policy, 8192, 40, 1, out=9):
        assert chain[0] == 40 + 9 + 1 # exactly the minimum, not rounded up


def test_round10_q0():
    policy = CKKSSearchPolicy(q0_rounding=("round10",))
    for chain in _chains(policy, 8192, 40, 1, out=9):
        assert chain[0] == 50


def test_two_prime_final_modulus_lifts_the_one_prime_ceiling():
    policy = CKKSSearchPolicy()
    st = ChainStructure(two_prime_q0=True)
    # one prime cannot hold scale 55 plus 10 output bits plus sign (66 bits)
    assert policy.final_modulus(55, 10, ChainStructure()) is None
    assert policy.final_modulus(55, 10, st) == (20, 60) # 20 + 60 >= 66
    # where one prime suffices, no two-prime variant is generated
    assert policy.final_modulus(40, 10, st) is None
    chain = policy.chain(55, 1, 10, st)
    assert chain[:2] == (20, 60) and chain[2] == 55 and chain[-1] >= 60


def test_special_prime_is_largest():
    policy = CKKSSearchPolicy.full()
    for chain in _chains(policy, 16384, 30, 2, out=4):
        assert chain[-1] >= max(chain[:-1])


def test_special_prime_min_max():
    policy = CKKSSearchPolicy.default()
    assert policy.special_prime_sizes(50, 45) == [50, 60]
    assert policy.special_prime_sizes(60, 55) == [60]
    assert CKKSSearchPolicy.broad_grid().special_prime_sizes(41, 35) == [41, 50, 60]


def test_chains_fit_128_bit_budget():
    policy = CKKSSearchPolicy.broad_grid()
    for n in policy.poly_degrees:
        for chain in _chains(policy, n, 40, 3, out=2):
            assert sum(chain) <= ALLOWED_CHAIN_BITS[n]


def test_extra_levels():
    policy = CKKSSearchPolicy(extra_levels=(0, 2), two_prime_q0=False)
    lengths = {len(c) for c in _chains(policy, 16384, 30, 1, out=0)}
    assert lengths == {3, 5}


@pytest.mark.parametrize("out,depth,target", [(3, 1, None), (8, 1, None), (0, 1, None), (6, 5, None),
                                              (2, 2, None), (3, 1, 20), (9, 1, 30), (13, 5, None)])
def test_default_space_is_subset_of_broad_grid(out, depth, target):
    def key_set(policy):
        space = CKKSParameterSearchSpace(depth, out, policy, target_bits=target)
        return {(p.poly_degree, p.scale, p.chain) for p in space.generate()}

    assert key_set(CKKSSearchPolicy.default()) <= key_set(CKKSSearchPolicy.broad_grid())


def test_scales_do_not_depend_on_the_data():
    # without a precision target the candidates depend only on the model (depth, output bound)
    scales = CKKSSearchPolicy.default().candidate_scales(None, 8192, 1, 10)
    # the 256-bit budget scale at N=8192, (41, 30, 41) = 112 <= 118, is not cut off
    assert 30 in scales


def test_infeasible_candidates_are_set_aside():
    # a 2^10 grid scale has no NTT-friendly primes for N >= 8192
    space = CKKSParameterSearchSpace(1, output_int_bits=3,
                                     policy=CKKSSearchPolicy(num_scale_options=1, min_scale_bits=10))
    space.generate()
    assert space.infeasible


@pytest.mark.parametrize("profile", ["fast", "full", "broad_grid"])
def test_from_config_uses_preset(profile):
    policy = CKKSSearchPolicy.from_config(SearchPolicyConfig(profile=profile))
    assert policy == getattr(CKKSSearchPolicy, profile)()


def test_from_config_overrides():
    cfg = SearchPolicyConfig(profile="broad_grid", q0_deltas=[0, 10], max_qL=50)
    policy = CKKSSearchPolicy.from_config(cfg)
    assert policy.q0_deltas == (0, 10)
    assert policy.max_qL == 50
    assert policy.extra_levels == CKKSSearchPolicy.broad_grid().extra_levels


def test_budget_scales_are_the_largest_fitting_scale_per_level():
    policy = CKKSSearchPolicy(two_prime_q0=False, budget_extra_levels=())
    scales = policy.candidate_scales(None, 8192, 1, 3)
    for sec in (128, 192, 256):
        budget = SECURITY_TABLES[sec][8192]
        fits = [s for s in scales if sum(policy.chain(s, 1, 3, ChainStructure())) <= budget]
        best = max(fits)
        nxt = policy.chain(best + 5, 1, 3, ChainStructure())
        assert nxt is None or sum(nxt) > budget


def test_broad_grid_covers_every_5_bit_scale():
    policy = CKKSSearchPolicy.broad_grid()
    assert policy.candidate_scales(None, poly_degree=8192, depth=1, output_int_bits=3) == list(range(20, 61, 5))


def test_budget_scales_get_a_spare_level():
    space = CKKSParameterSearchSpace(1, output_int_bits=3)
    params = space.generate()
    assert any(space.structure_of[p].extra_levels == 1 for p in params)


def test_step_down_keeps_structure():
    space = CKKSParameterSearchSpace(1, output_int_bits=3)
    params = space.generate()
    single = next(p for p in params if p.poly_degree == 8192 and p.chain == (59, 55, 59))
    assert space.step_down(single).chain == (54, 50, 54) # exact q0, smallest special, same levels

    two = next(p for p in params if space.structure_of[p].two_prime_q0)
    lower = space.step_down(two)
    # a lower two-prime chain exists only while one prime still cannot hold the output
    assert lower is None or (lower.chain[1] == 60 and len(lower.chain) == len(two.chain))


def test_step_down_stops_at_the_floor():
    space = CKKSParameterSearchSpace(1, output_int_bits=3, target_bits=40) # floor round10(45) = 50
    params = space.generate()
    assert min(p.log_scale for p in params) >= 50
    assert space.step_down(CKKSParams(8192, 2**50, (54, 50, 54))) is None # not a generated candidate


def test_selector_steps_down_after_failure():
    from ckks_param_tool.evaluation import ParameterEvaluationResult
    from ckks_param_tool.search import CKKSParameterSelector

    class Evaluator:
        # pretend every scale above 2^50 overflows
        def evaluate(self, p):
            ok = p.log_scale <= 50
            return ParameterEvaluationResult(error=1e-9 if ok else float("inf"), runtime=1.0,
                                             security_bits=p.security_score,
                                             status="success" if ok else "failure")

    space = CKKSParameterSearchSpace(1, output_int_bits=3)
    selector = CKKSParameterSelector(Evaluator(), space)
    results = selector.select()
    evaluated = {p.chain for p, _ in results}
    assert (59, 55, 59) in evaluated and (54, 50, 54) in evaluated
    assert selector.step_downs > 0

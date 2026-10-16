import pytest

from ckks_param_tool.config import SearchPolicyConfig
from ckks_param_tool.params import ALLOWED_CHAIN_BITS
from ckks_param_tool.search import CKKSParameterSearchSpace, CKKSSearchPolicy


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_chain_has_depth_plus_two_primes(depth):
    policy = CKKSSearchPolicy.default()
    for chain in policy.generate_chains(16384, 30, depth, output_int_bits=8):
        assert len(chain) == depth + 2
        assert all(q == 30 for q in chain[1:-1])


def test_q0_holds_output():
    policy = CKKSSearchPolicy.default()
    for chain in policy.generate_chains(8192, 40, 1, output_int_bits=9):
        assert chain[0] >= 40 + 9 + 1


def test_special_prime_is_largest():
    policy = CKKSSearchPolicy.full()
    for chain in policy.generate_chains(16384, 30, 2, output_int_bits=4):
        assert chain[-1] >= max(chain[:-1])


def test_chains_fit_128_bit_budget():
    policy = CKKSSearchPolicy.fuller()
    for n in policy.poly_degrees:
        for chain in policy.generate_chains(n, 40, 3, output_int_bits=2):
            assert sum(chain) <= ALLOWED_CHAIN_BITS[n]


def test_extra_levels():
    policy = CKKSSearchPolicy(extra_levels=(0, 2))
    lengths = {len(c) for c in policy.generate_chains(16384, 30, 1, output_int_bits=0)}
    assert lengths == {3, 5}


@pytest.mark.parametrize("dec,out,depth", [(4, 3, 1), (13, 8, 1), (24, 0, 1), (8, 6, 5), (30, 2, 2)])
def test_default_space_is_subset_of_fuller(dec, out, depth):
    def key_set(policy):
        space = CKKSParameterSearchSpace(depth, dec, out, policy)
        return {(p.poly_degree, p.scale, p.chain) for p in space.generate()}

    assert key_set(CKKSSearchPolicy.default()) <= key_set(CKKSSearchPolicy.fuller())


def test_infeasible_candidates_are_set_aside():
    space = CKKSParameterSearchSpace(1, bits_decimal=4, output_int_bits=3)
    feasible = space.generate()
    assert space.infeasible
    assert all(p.log_scale > 10 for p in feasible)


@pytest.mark.parametrize("profile", ["fast", "full", "fuller"])
def test_from_config_uses_preset(profile):
    policy = CKKSSearchPolicy.from_config(SearchPolicyConfig(profile=profile))
    assert policy == getattr(CKKSSearchPolicy, profile)()


def test_from_config_overrides():
    cfg = SearchPolicyConfig(profile="fuller", q0_deltas=[0, 10], max_qL=50)
    policy = CKKSSearchPolicy.from_config(cfg)
    assert policy.q0_deltas == (0, 10)
    assert policy.max_qL == 50
    assert policy.extra_levels == CKKSSearchPolicy.fuller().extra_levels


def test_budget_scales_are_largest_fitting_scale_per_level():
    from ckks_param_tool.params import SECURITY_TABLES

    policy = CKKSSearchPolicy.default()
    scales = policy.candidate_scales(bits_decimal=4, poly_degree=8192, depth=1, output_int_bits=3)
    for sec in (128, 192, 256):
        budget = SECURITY_TABLES[sec][8192]
        fitting = [s for s in scales
                   if policy.base_q0(s, 3) + s + policy.base_q0(s, 3) <= budget]
        best = max(fitting)
        # the next multiple of 5 above must not fit (or exceed the q0 <= 60 limit)
        nxt = best + 5
        q0 = policy.base_q0(nxt, 3)
        assert q0 > 60 or q0 + nxt + q0 > budget


def test_scales_never_exceed_one_prime_for_q0():
    policy = CKKSSearchPolicy.fuller()
    for n in policy.poly_degrees:
        for s in policy.candidate_scales(bits_decimal=4, poly_degree=n, depth=1, output_int_bits=9):
            assert policy.base_q0(s, 9) <= 60


def test_fuller_scales_ignore_the_data():
    policy = CKKSSearchPolicy.fuller()
    a = policy.candidate_scales(bits_decimal=4, poly_degree=8192, depth=1, output_int_bits=3)
    b = policy.candidate_scales(bits_decimal=30, poly_degree=8192, depth=1, output_int_bits=3)
    assert a == b == list(range(20, 56, 5))


def test_budget_scales_get_a_spare_level():
    space = CKKSParameterSearchSpace(1, bits_decimal=20, output_int_bits=3)
    params = space.generate()
    spare = [p for p in params if p.levels == 2]
    assert spare, "budget scales should also be tried with one spare level"
    grid = set(CKKSSearchPolicy.default().scale_plan(20, 8192, 1, 3))
    assert all(p.log_scale in grid for p in spare)

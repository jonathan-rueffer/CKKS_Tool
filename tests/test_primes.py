from ckks_param_tool.params import chain_infeasibility_reason
from ckks_param_tool.params.primes import _is_prime


def test_miller_rabin_small():
    primes = [p for p in range(2, 200) if _is_prime(p)]
    assert primes == [p for p in range(2, 200) if all(p % d for d in range(2, p))]


def test_miller_rabin_large():
    assert _is_prime(2**61 - 1)
    assert not _is_prime(2**61 + 1)


def test_typical_chain_is_feasible():
    assert chain_infeasibility_reason(8192, (60, 40, 40, 60)) == ""


def test_prime_size_limits():
    assert "outside" in chain_infeasibility_reason(8192, (61, 40, 60))
    assert "outside" in chain_infeasibility_reason(8192, (1, 40, 60))


def test_too_small_primes_are_infeasible():
    # 10-bit primes cannot be 1 mod 2N = 16384
    assert "10-bit" in chain_infeasibility_reason(8192, (20, 10, 20))

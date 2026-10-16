from collections import Counter
from functools import lru_cache

'''
This file reproduces the prime generation rule of Microsoft SEAL (CoeffModulus::Create and
util::get_primes) so that a modulus chain can be checked for feasibility before any keys are
generated.

SEAL requires every prime q_i in the chain to:
    - have between 2 and 60 bits (SEAL_USER_MOD_BIT_COUNT_MIN/MAX),
    - satisfy q_i = 1 (mod 2N), so that the NTT exists,
    - be distinct from every other prime in the chain.
For each bit size b it searches downwards from 2^b for primes of the form k * 2N + 1 and fails
with "failed to find enough qualifying primes" when there are not enough of them.
'''

MIN_PRIME_BITS = 2
MAX_PRIME_BITS = 60

# deterministic Miller-Rabin bases for all 64-bit integers
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# returns how many qualifying primes of exactly bit_size bits exist, stopping at `needed`
@lru_cache(maxsize=None)
def _count_primes(poly_degree: int, bit_size: int, needed: int) -> int:
    factor = 2 * poly_degree
    value = ((1 << bit_size) - 1) // factor * factor + 1
    lower_bound = 1 << (bit_size - 1)

    found = 0
    while found < needed and value > lower_bound:
        if _is_prime(value):
            found += 1
        value -= factor
    return found


# returns an empty string if SEAL can build the chain, otherwise the reason it cannot
def chain_infeasibility_reason(poly_degree: int, chain: tuple[int, ...]) -> str:
    for bits in chain:
        if not MIN_PRIME_BITS <= bits <= MAX_PRIME_BITS:
            return f"prime size {bits} outside [{MIN_PRIME_BITS}, {MAX_PRIME_BITS}] bits"

    for bits, count in Counter(chain).items():
        if _count_primes(poly_degree, bits, count) < count:
            return f"not enough {bits}-bit primes congruent to 1 mod 2N"

    return ""

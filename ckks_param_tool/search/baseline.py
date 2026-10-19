from ckks_param_tool.params import CKKSParams, ALLOWED_CHAIN_BITS

'''
This file contains the two baselines the tool's configurations are compared with.

Tutorial defaults: what a non-expert would copy from the TenSEAL / Microsoft SEAL CKKS examples.

    - depth <= 2: N = 8192, chain [60, 40, 40, 60], scale 2^40 (the TenSEAL tutorial context)
    - depth  > 2: the SEAL example rule extended to the depth, [60] + [40] * depth + [60] with
                  scale 2^40, at the smallest N whose 128-bit budget fits the chain
'''

TUTORIAL_LOG_SCALE = 40


def tutorial_baseline_params(depth: int) -> CKKSParams:
    if depth <= 2:
        return CKKSParams(poly_degree=8192, scale=2**TUTORIAL_LOG_SCALE, chain=(60, 40, 40, 60))

    chain = (60,) + (TUTORIAL_LOG_SCALE,) * depth + (60,)
    for n in sorted(ALLOWED_CHAIN_BITS):
        if sum(chain) <= ALLOWED_CHAIN_BITS[n]:
            return CKKSParams(poly_degree=n, scale=2**TUTORIAL_LOG_SCALE, chain=chain)

    raise ValueError(f"no supported ring dimension fits a depth-{depth} tutorial chain")


'''
Expert defaults: the rule of thumb of OpenFHE's CKKS examples and SEAL's guidance for 64-bit
words: a 60-bit first prime, 50-bit scaling primes (one per level, no spare level), a 60-bit
special prime, and the smallest ring dimension whose 128-bit budget fits the chain.
'''

EXPERT_LOG_SCALE = 50
EXPERT_FIRST_PRIME = 60
EXPERT_SPECIAL_PRIME = 60


def expert_baseline_params(depth: int) -> CKKSParams:
    chain = (EXPERT_FIRST_PRIME,) + (EXPERT_LOG_SCALE,) * depth + (EXPERT_SPECIAL_PRIME,)
    for n in sorted(ALLOWED_CHAIN_BITS):
        if sum(chain) <= ALLOWED_CHAIN_BITS[n]:
            return CKKSParams(poly_degree=n, scale=2**EXPERT_LOG_SCALE, chain=chain)

    raise ValueError(f"no supported ring dimension fits a depth-{depth} expert chain")

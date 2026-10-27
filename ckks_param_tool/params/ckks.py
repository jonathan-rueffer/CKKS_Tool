from dataclasses import dataclass
from .security import security_level

'''
This file contains the class that outlines the CKKS parameter set structure used throughout the tool
'''
@dataclass(frozen=True)
class CKKSParams:
    poly_degree: int
    scale: int
    chain: tuple[int, ...]

    @property
    def log_scale(self) -> int:
        return self.scale.bit_length() - 1

    # number of rescales the chain supports: every prime except q0 and the special prime
    @property
    def levels(self) -> int:
        return len(self.chain) - 2

    @property
    def chain_sum(self) -> int:
        return sum(self.chain)
    
    @property
    def security_score(self) -> float:
        return security_level(self.poly_degree, self.chain_sum)

    # bits of output headroom after `depth` rescales: the modulus left (every data prime not
    # consumed, spare levels included) minus the scale and the sign bit. A value v at scale Delta
    # fits while |v| < 2^headroom (nominal prime sizes; SEAL's primes lie just below 2^bits).
    # The tool's own capacity rule, not a SEAL constraint: SEAL does not check message sizes.
    def output_headroom_bits(self, depth: int) -> int:
        remaining = self.chain[: len(self.chain) - 1 - depth]
        return sum(remaining) - self.log_scale - 1

    # the smallest error an overflow can cause: a value that exceeds the headroom wraps modulo
    # the final modulus, i.e. changes by a multiple of about 2^(headroom + 1). Half of that is
    # used as the threshold, far above any noise-level error.
    def overflow_error(self, depth: int) -> float:
        return 2.0 ** self.output_headroom_bits(depth)

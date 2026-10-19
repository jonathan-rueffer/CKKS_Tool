from __future__ import annotations
from dataclasses import dataclass, replace, fields
import math

from ckks_param_tool.params.security import ALLOWED_CHAIN_BITS, SECURITY_LEVELS, SECURITY_TABLES
from ckks_param_tool.params.primes import MAX_PRIME_BITS

'''
This file contains the class that outlines the rules that the search space generation stage
follows.

Chain structure (SEAL / TenSEAL convention):

    (q0, q1, ..., qL, special)

    - q1..qL : one prime per rescale. TenSEAL rescales after every multiplication, including
               ciphertext-plaintext multiplications, so a model of multiplicative depth L needs
               exactly L of these primes. They are set to the size of the scale so that the
               scale stays stable after each rescale.
    - q0     : the prime left after the last rescale. It must hold the output value at scale
               Delta, so log2(q0) >= log2(Delta) + (integer bits of the output) + 1 (sign).
               The bits of q0 above the scale are the "bits of precision before the decimal
               point" in the SEAL CKKS example.
    - special: the key-switching prime. SEAL recommends it to be at least as large as the
               largest other prime. It is counted in the security budget but holds no data. A
               larger special prime lowers key-switching noise, so by default two sizes are
               tried: the smallest valid one (max(q0, scale), cheapest in budget) and the
               largest (max_qL, least noise).

A model of depth L therefore needs a chain of length L + 2. Additional "spare" levels are
allowed but only add cost; they are kept as a configurable option for exploration.

Scales. In the RNS representation every prime occupies one 64-bit word (Cheon et al., SAC
2018), so the cost of encrypted inference depends on N and on the number of primes, but hardly
on their bit sizes: for a fixed N and chain length a larger scale gives more precision at the
same runtime. The candidate scales are therefore
    - the data-derived lower bound (the precision the data needs; optionally a grid above it),
    - for every security level, the largest scale whose smallest valid chain still fits that
      level's modulus budget at this N ("budget scales"); these are the most precise
      configurations each security level admits.
q0 beyond its minimum only adds headroom, not precision, so by default only the minimum is used.
Budget scales are also tried with spare levels (budget_extra_levels, default one): before the
first rescale a value carries scale Delta^2, so at the largest scales the top modulus can be
too small for the intermediate products (e.g. large raw inputs times weights). A spare level
adds that headroom. In addition, a candidate whose evaluation fails or overflows is retried with
the same structure step_down_bits lower (see CKKSParameterSearchSpace.step_down), so the search
finds the largest scale that works for the workload.
'''


@dataclass(frozen=True)
class CKKSSearchPolicy:

    # ring dimensions to consider (each is filtered by its 128-bit modulus budget)
    poly_degrees: tuple[int, ...] = (8192, 16384, 32768)

    # how many grid scales to include, starting at the lower bound (1 = only the lower bound,
    # 0 = every grid step up to the largest feasible scale)
    num_scale_options: int = 1

    # step size (bits) between grid scales
    scale_step: int = 10

    # buffer (bits) added to the decimal precision estimate before rounding
    base_scale_buffer: int = 5

    # if set, the grid starts at this scale instead of the data-derived lower bound
    min_scale_bits: int | None = None

    # add, for every security level, the largest scale that fits its budget (see above)
    budget_scales: bool = True

    # spare levels additionally tried at the budget scales (on top of extra_levels)
    budget_extra_levels: tuple[int, ...] = (1,)

    # unused levels added on top of the model depth (0 = exactly depth + 2 primes)
    extra_levels: tuple[int, ...] = (0,)

    # bits added to the minimum admissible q0
    q0_deltas: tuple[int, ...] = (0,)

    # max size of the special prime
    max_qL: int = MAX_PRIME_BITS

    # special prime sizes: "min_max" = {smallest valid, max_qL}; "all" = every 10 bits in between
    special_primes: str = "min_max"

    # when a candidate fails or overflows, retry it this many bits lower (0 = off)
    step_down_bits: int = 5

    def __post_init__(self):
        unknown = [n for n in self.poly_degrees if n not in ALLOWED_CHAIN_BITS]
        if unknown:
            raise ValueError(f"unsupported poly_degrees {unknown}, choose from {sorted(ALLOWED_CHAIN_BITS)}")
        if self.num_scale_options < 0:
            raise ValueError("num_scale_options must be >= 0")
        if self.min_scale_bits is not None and not 2 <= self.min_scale_bits <= MAX_PRIME_BITS:
            raise ValueError(f"min_scale_bits must be in [2, {MAX_PRIME_BITS}]")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be > 0")
        if any(e < 0 for e in self.extra_levels) or any(e < 0 for e in self.budget_extra_levels):
            raise ValueError("extra_levels and budget_extra_levels must be >= 0")
        if any(d < 0 for d in self.q0_deltas):
            raise ValueError("q0_deltas must be >= 0")
        if not 2 <= self.max_qL <= MAX_PRIME_BITS:
            raise ValueError(f"max_qL must be in [2, {MAX_PRIME_BITS}]")
        if self.special_primes not in ("min_max", "all"):
            raise ValueError("special_primes must be 'min_max' or 'all'")
        if self.step_down_bits < 0:
            raise ValueError("step_down_bits must be >= 0")

    # helper function to round a number up to the nearest 10
    @staticmethod
    def _round_to_10(n: int) -> int:
        return math.ceil(n / 10.0) * 10

    # smallest q0 (in bits) that can hold the output at the given scale
    @staticmethod
    def min_q0_bits(scale_bits: int, output_int_bits: int) -> int:
        return scale_bits + output_int_bits + 1

    # q0 as generated by the policy before the q0_deltas are added
    def base_q0(self, scale_bits: int, output_int_bits: int) -> int:
        return self._round_to_10(self.min_q0_bits(scale_bits, output_int_bits))

    # largest scale for which q0 still fits in one prime
    def max_scale_bits(self, output_int_bits: int) -> int:
        return MAX_PRIME_BITS - output_int_bits - 1

    # the data-derived lower bound on the scale (or min_scale_bits if set)
    def lower_scale_bits(self, bits_decimal: int) -> int:
        if self.min_scale_bits is not None:
            return self.min_scale_bits
        return self._round_to_10(bits_decimal + self.base_scale_buffer)

    # the largest scale (multiple of 5, >= lower) whose smallest chain fits the budget
    def _budget_scale(self, budget: int, levels: int, lower: int, output_int_bits: int) -> int | None:
        top = self.max_scale_bits(output_int_bits)
        for s in range(top - top % 5, lower - 1, -5):
            q0 = self.base_q0(s, output_int_bits)
            if q0 + levels * s + max(q0, s) <= budget:
                return s
        return None

    # maps every candidate scale (in bits) for one ring dimension to the spare levels it is
    # tried with: grid scales use extra_levels, budget scales additionally budget_extra_levels
    def scale_plan(
        self,
        bits_decimal: int,
        poly_degree: int,
        depth: int,
        output_int_bits: int,
    ) -> dict[int, set[int]]:
        lower = self.lower_scale_bits(bits_decimal)
        top = self.max_scale_bits(output_int_bits)
        base_extras = set(self.extra_levels)

        plan: dict[int, set[int]] = {}
        i = 0
        while lower + i * self.scale_step <= top and (self.num_scale_options == 0 or i < self.num_scale_options):
            plan.setdefault(lower + i * self.scale_step, set()).update(base_extras)
            i += 1

        if self.budget_scales:
            for sec in SECURITY_LEVELS:
                for extra in base_extras | set(self.budget_extra_levels):
                    s = self._budget_scale(SECURITY_TABLES[sec][poly_degree], depth + extra, lower, output_int_bits)
                    if s is not None:
                        plan.setdefault(s, set()).update(base_extras | {extra})

        return plan

    # the candidate scale sizes (in bits) for one ring dimension
    def candidate_scales(
        self,
        bits_decimal: int,
        poly_degree: int,
        depth: int,
        output_int_bits: int,
    ) -> list[int]:
        return sorted(self.scale_plan(bits_decimal, poly_degree, depth, output_int_bits))

    # the special prime should be at least as large as every other prime
    def special_prime_sizes(self, q0: int, scale_bits: int) -> list[int]:
        smallest = max(q0, scale_bits)
        if smallest > self.max_qL:
            return []
        if self.special_primes == "min_max":
            return sorted({smallest, self.max_qL})
        return list(range(smallest, self.max_qL + 1, 10))

    # generates candidate chains according to the policy constraints
    def generate_chains(
        self,
        poly_degree: int,
        scale_bits: int,
        depth: int,
        output_int_bits: int,
        extra_levels: set[int] | None = None,
    ) -> list[tuple[int, ...]]:

        budget = ALLOWED_CHAIN_BITS[poly_degree]
        extra_levels = self.extra_levels if extra_levels is None else extra_levels
        base_q0 = self.base_q0(scale_bits, output_int_bits)

        q0_candidates = sorted({base_q0 + d for d in self.q0_deltas if base_q0 + d <= MAX_PRIME_BITS})

        chains = []
        for extra in sorted(set(extra_levels)):
            internal = (scale_bits,) * (depth + extra)

            for q0 in q0_candidates:
                for special in self.special_prime_sizes(q0, scale_bits):
                    chain = (q0,) + internal + (special,)
                    if sum(chain) <= budget:
                        chains.append(chain)

        return chains

    # produces the default search policy
    @classmethod
    def default(cls) -> "CKKSSearchPolicy":
        return cls()

    # produces a smaller candidate set for quick experimentation
    @classmethod
    def fast(cls) -> "CKKSSearchPolicy":
        return cls(
            poly_degrees=(8192, 16384),
            budget_extra_levels=(),
        )

    # produces a more expansive parameter space policy
    @classmethod
    def full(cls) -> "CKKSSearchPolicy":
        return cls(
            num_scale_options=4,
            extra_levels=(0, 1),
            q0_deltas=(0, 10, 20, 30),
            special_primes="all",
        )

    # produces a broad-grid policy, used to compare the tool against a broad search. Its
    # scales do not depend on the data: every multiple of 5 bits from 2^20 up to the largest
    # feasible scale. Every candidate of `default` is also a candidate of `broad_grid` (see tests).
    @classmethod
    def broad_grid(cls) -> "CKKSSearchPolicy":
        return cls(
            num_scale_options=0,
            scale_step=5,
            min_scale_bits=20,
            budget_scales=False,
            extra_levels=(0, 1, 2, 3, 4),
            q0_deltas=(0, 10, 20, 30, 40, 50),
            special_primes="all",
            step_down_bits=0, # every scale is already evaluated
        )

    PROFILES = ("default", "fast", "full", "broad_grid", "custom")

    # builds a policy from a SearchPolicyConfig: the named profile is the starting point and
    # every field set in the config (not None) overrides it
    @classmethod
    def from_config(cls, config) -> "CKKSSearchPolicy":
        presets = {
            "default": cls.default,
            "custom": cls.default,
            "fast": cls.fast,
            "full": cls.full,
            "broad_grid": cls.broad_grid,
        }
        policy = presets[config.profile]()

        overrides = {}
        for f in fields(cls):
            value = getattr(config, f.name, None)
            if value is not None:
                overrides[f.name] = tuple(value) if isinstance(value, list) else value

        return replace(policy, **overrides)

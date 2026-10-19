from __future__ import annotations
from dataclasses import dataclass, replace, fields
import math

from ckks_param_tool.params.security import ALLOWED_CHAIN_BITS, SECURITY_LEVELS, SECURITY_TABLES
from ckks_param_tool.params.primes import MAX_PRIME_BITS

'''
This file contains the class that outlines the rules that the search space generation stage
follows.

Chain structure (SEAL / TenSEAL convention):

    (final modulus, q1, ..., qL, special)

    - q1..qL : one prime per rescale. TenSEAL rescales after every multiplication, including
               ciphertext-plaintext multiplications, so a model of multiplicative depth L needs
               exactly L of these primes. They are set to the size of the scale so that the
               scale stays stable after each rescale.
    - final  : the modulus left after the last rescale. It must hold the output value at scale
               Delta, so its size must be at least log2(Delta) + (integer bits of the output) + 1
               (sign); the bits above the scale are the "bits of precision before the decimal
               point" in the SEAL CKKS example. It is either one prime q0 sized exactly for that
               ("exact") or rounded up to a multiple of 10 ("round10"), or, when one prime of at
               most 60 bits cannot hold it, two primes (q0, 60): every prime is limited to 60
               bits, so with a single q0 the scale can never exceed 60 - output bits - 1 however
               much security budget is left. The second prime lifts that ceiling.
    - special: the key-switching prime. SEAL requires it to be at least as large as the largest
               other prime. It is counted in the security budget but holds no data. A larger
               special prime lowers key-switching noise, so by default two sizes are tried: the
               smallest valid one (cheapest in budget) and the largest (max_qL, least noise).

A model of depth L therefore needs L + 2 primes (L + 3 with a two-prime final modulus). Spare
levels are allowed but only add cost; they are kept as a configurable option for exploration.

Scales. In the RNS representation every prime occupies one 64-bit word (Cheon et al., SAC
2018), so the cost of encrypted inference depends on N and on the number of primes, but hardly
on their bit sizes: for a fixed N and chain length a larger scale gives more precision at the
same runtime. The candidate scales are therefore, for every ring dimension N, security level and
chain structure, the largest scale whose smallest valid chain still fits that level's modulus
budget at this N ("budget scales"): the most precise configuration each of them admits. They
cover both ends of the trade-off: the budget scale of a small N at a high security level is the
cheap, less precise option, that of a larger N the precise one. Scales are not derived from the
data: input resolution says nothing about the precision an output needs. A precision target set
by the user only raises the lowest scale considered (min_scale_bits otherwise). Scales are
multiples of 5 bits so that every candidate is also part of the benchmark's reference grid.

Extra bits in q0 beyond its minimum only add headroom, not precision, so by default only the
minimum is used, sized exactly. Budget scales are also tried with spare levels
(budget_extra_levels, default one): before the first rescale a value carries scale Delta^2, so at
the largest scales the top modulus can be too small for the intermediate products (e.g. large
raw inputs times weights). A spare level adds that headroom. In addition, a candidate whose
evaluation fails or overflows is retried with the same structure step_down_bits lower (see
CKKSParameterSearchSpace.step_down), so the search finds the largest scale that works.
'''

# grid of candidate scales (bits): budget scales and step-downs stay on multiples of this
SCALE_GRANULARITY = 5

# smallest first prime of a two-prime final modulus
MIN_TWO_PRIME_Q0 = 20


@dataclass(frozen=True)
class ChainStructure:
    rounding: str = "exact" # "exact" or "round10": how the first prime q0 is sized
    two_prime_q0: bool = False # final modulus (q0, 60) instead of q0
    extra_levels: int = 0 # spare levels beyond the model depth
    q0_delta: int = 0 # bits added to the minimum q0
    special: str = "min" # "min" (smallest valid special prime) or "max" (max_qL); "all" grids use sizes


@dataclass(frozen=True)
class CKKSSearchPolicy:

    # ring dimensions to consider (each is filtered by its 128-bit modulus budget)
    poly_degrees: tuple[int, ...] = (8192, 16384, 32768)

    # grid scales starting at the scale floor, scale_step apart (0 = no grid)
    num_scale_options: int = 0

    # if True the grid covers every step from the floor to the largest feasible scale
    full_grid: bool = False

    # step size (bits) between grid scales
    scale_step: int = 10

    # buffer (bits) added to a user precision target to get the scale floor
    base_scale_buffer: int = 5

    # the lowest scale considered by the budget scales, the step-down and the grid. Below about
    # 2^20 there are too few NTT-friendly primes for N >= 8192 and no useful precision.
    min_scale_bits: int = 20

    # add, for every security level and structure, the largest scale that fits its budget
    budget_scales: bool = True

    # spare levels additionally tried at the budget scales (on top of extra_levels)
    budget_extra_levels: tuple[int, ...] = (1,)

    # unused levels added on top of the model depth (0 = exactly depth + 2 primes)
    extra_levels: tuple[int, ...] = (0,)

    # bits added to the minimum admissible q0
    q0_deltas: tuple[int, ...] = (0,)

    # how q0 is sized: "exact" (just large enough) and/or "round10" (rounded up to 10 bits)
    q0_rounding: tuple[str, ...] = ("exact",)

    # also use a two-prime final modulus (q0, 60) where one prime cannot hold the output
    two_prime_q0: bool = True

    # max size of the special prime
    max_qL: int = MAX_PRIME_BITS

    # special prime sizes: "min_max" = {smallest valid, max_qL}; "all" = smallest valid and every
    # multiple of 10 above it
    special_primes: str = "min_max"

    # when a candidate fails or overflows, retry it this many bits lower (0 = off)
    step_down_bits: int = 5

    def __post_init__(self):
        unknown = [n for n in self.poly_degrees if n not in ALLOWED_CHAIN_BITS]
        if unknown:
            raise ValueError(f"unsupported poly_degrees {unknown}, choose from {sorted(ALLOWED_CHAIN_BITS)}")
        if self.num_scale_options < 0:
            raise ValueError("num_scale_options must be >= 0")
        if not 2 <= self.min_scale_bits <= MAX_PRIME_BITS:
            raise ValueError(f"min_scale_bits must be in [2, {MAX_PRIME_BITS}]")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be > 0")
        if any(e < 0 for e in self.extra_levels) or any(e < 0 for e in self.budget_extra_levels):
            raise ValueError("extra_levels and budget_extra_levels must be >= 0")
        if any(d < 0 for d in self.q0_deltas):
            raise ValueError("q0_deltas must be >= 0")
        if not self.q0_rounding or any(r not in ("exact", "round10") for r in self.q0_rounding):
            raise ValueError("q0_rounding must contain 'exact' and/or 'round10'")
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

    # smallest final modulus (in bits) that can hold the output at the given scale
    @staticmethod
    def min_q0_bits(scale_bits: int, output_int_bits: int) -> int:
        return scale_bits + output_int_bits + 1

    # the minimum q0 rounded up to 10 bits (the "round10" structure)
    def base_q0(self, scale_bits: int, output_int_bits: int) -> int:
        return self._round_to_10(self.min_q0_bits(scale_bits, output_int_bits))

    # the lowest scale considered: min_scale_bits, raised by a user precision target if given
    def lower_scale_bits(self, target_bits: int | None = None) -> int:
        if target_bits is None:
            return self.min_scale_bits
        return max(self.min_scale_bits, self._round_to_10(target_bits + self.base_scale_buffer))

    # the final modulus of a structure at a scale, or None if the structure cannot hold it
    def final_modulus(self, scale_bits: int, output_int_bits: int, structure: ChainStructure) -> tuple[int, ...] | None:
        needed = self.min_q0_bits(scale_bits, output_int_bits)
        if structure.rounding == "round10":
            needed = self._round_to_10(needed)
        needed += structure.q0_delta

        if not structure.two_prime_q0:
            return (needed,) if needed <= MAX_PRIME_BITS else None

        # two primes only where one cannot hold the output
        if needed <= MAX_PRIME_BITS:
            return None
        q0 = needed - MAX_PRIME_BITS
        if structure.rounding == "round10":
            q0 = self._round_to_10(q0)
        return (max(q0, MIN_TWO_PRIME_Q0), MAX_PRIME_BITS)

    # the special prime should be at least as large as every other prime
    def special_prime_sizes(self, largest_prime: int, scale_bits: int) -> list[int]:
        smallest = max(largest_prime, scale_bits)
        if smallest > self.max_qL:
            return []
        if self.special_primes == "min_max":
            return sorted({smallest, self.max_qL})
        return sorted({smallest} | set(range(self._round_to_10(smallest), self.max_qL + 1, 10)))

    def chain(self, scale_bits: int, depth: int, output_int_bits: int, structure: ChainStructure,
              special: int | None = None) -> tuple[int, ...] | None:
        final = self.final_modulus(scale_bits, output_int_bits, structure)
        if final is None:
            return None
        if special is None:
            special = max(max(final), scale_bits) if structure.special == "min" else self.max_qL
        if special < max(max(final), scale_bits) or special > MAX_PRIME_BITS:
            return None
        return final + (scale_bits,) * (depth + structure.extra_levels) + (special,)

    def structures(self, extra_levels) -> list[ChainStructure]:
        return [
            ChainStructure(rounding=r, two_prime_q0=two, extra_levels=e, q0_delta=d)
            for r in self.q0_rounding
            for two in ((False, True) if self.two_prime_q0 else (False,))
            for e in sorted(set(extra_levels))
            for d in sorted(set(self.q0_deltas))
        ]

    # the largest scale (multiple of SCALE_GRANULARITY, >= lower) whose smallest chain of this
    # structure fits the budget
    def _budget_scale(self, budget: int, depth: int, lower: int, output_int_bits: int,
                      structure: ChainStructure) -> int | None:
        top = MAX_PRIME_BITS - MAX_PRIME_BITS % SCALE_GRANULARITY
        for s in range(top, lower - 1, -SCALE_GRANULARITY):
            chain = self.chain(s, depth, output_int_bits, structure)
            if chain is not None and sum(chain) <= budget:
                return s
        return None

    # maps every candidate scale (in bits) for one ring dimension to the structures it is tried
    # with: grid scales (if any) every structure, budget scales the structure they came from
    def scale_plan(
        self,
        target_bits: int | None,
        poly_degree: int,
        depth: int,
        output_int_bits: int,
    ) -> dict[int, set[ChainStructure]]:
        lower = self.lower_scale_bits(target_bits)
        grid_structures = self.structures(self.extra_levels)

        plan: dict[int, set[ChainStructure]] = {}
        i = 0
        while lower + i * self.scale_step <= MAX_PRIME_BITS and (self.full_grid or i < self.num_scale_options):
            plan.setdefault(lower + i * self.scale_step, set()).update(grid_structures)
            i += 1

        if self.budget_scales:
            for sec in SECURITY_LEVELS:
                for structure in self.structures(set(self.extra_levels) | set(self.budget_extra_levels)):
                    s = self._budget_scale(SECURITY_TABLES[sec][poly_degree], depth, lower, output_int_bits, structure)
                    if s is not None:
                        plan.setdefault(s, set()).add(structure)

        return plan

    # the candidate scale sizes (in bits) for one ring dimension
    def candidate_scales(
        self,
        target_bits: int | None,
        poly_degree: int,
        depth: int,
        output_int_bits: int,
    ) -> list[int]:
        return sorted(self.scale_plan(target_bits, poly_degree, depth, output_int_bits))

    # generates the candidate chains of one scale, with the structure each chain belongs to
    def generate_chains(
        self,
        poly_degree: int,
        scale_bits: int,
        depth: int,
        output_int_bits: int,
        structures=None,
    ) -> list[tuple[tuple[int, ...], ChainStructure]]:

        budget = ALLOWED_CHAIN_BITS[poly_degree]
        structures = self.structures(self.extra_levels) if structures is None else structures

        chains = {}
        for structure in sorted(structures, key=lambda st: (st.rounding, st.two_prime_q0, st.extra_levels, st.q0_delta)):
            final = self.final_modulus(scale_bits, output_int_bits, structure)
            if final is None:
                continue
            for special in self.special_prime_sizes(max(final), scale_bits):
                chain = self.chain(scale_bits, depth, output_int_bits, structure, special=special)
                if chain is not None and sum(chain) <= budget and chain not in chains:
                    rule = "min" if special == max(max(final), scale_bits) else "max"
                    chains[chain] = replace(structure, special=rule)

        return list(chains.items())

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
            two_prime_q0=False,
        )

    # produces a more expansive parameter space policy
    @classmethod
    def full(cls) -> "CKKSSearchPolicy":
        return cls(
            num_scale_options=4, # a grid of 4 scales from the floor, in addition to the budget scales
            extra_levels=(0, 1),
            q0_deltas=(0, 10, 20, 30),
            q0_rounding=("exact", "round10"),
            special_primes="all",
        )

    # produces the benchmark's reference grid: every multiple of 5 bits from 2^20 to 2^60, q0
    # sized exactly and rounded to 10 bits with offsets up to +50, one- and two-prime final
    # moduli, every special prime size in steps of 10, and up to 4 spare levels. Every candidate
    # of `default` is also a candidate of `broad_grid` (see tests).
    @classmethod
    def broad_grid(cls) -> "CKKSSearchPolicy":
        return cls(
            full_grid=True,
            scale_step=SCALE_GRANULARITY,
            min_scale_bits=20,
            budget_scales=False,
            extra_levels=(0, 1, 2, 3, 4),
            q0_deltas=(0, 10, 20, 30, 40, 50),
            q0_rounding=("exact", "round10"),
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

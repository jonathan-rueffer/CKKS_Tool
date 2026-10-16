from ckks_param_tool.params import CKKSParams, ALLOWED_CHAIN_BITS, chain_infeasibility_reason
from .policy import CKKSSearchPolicy

'''
This file contains the class that defines the search space generation functionality
'''


class CKKSParameterSearchSpace:
    def __init__(
        self,
        depth: int,
        bits_decimal: int,
        output_int_bits: int,
        policy: CKKSSearchPolicy | None = None,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.bits_decimal = bits_decimal
        self.output_int_bits = output_int_bits
        self.policy = policy or CKKSSearchPolicy.default()

        # candidates that the policy produced but SEAL cannot instantiate, with the reason
        self.infeasible: list[tuple[CKKSParams, str]] = []

    # generates the candidate parameter search space according to the policy. Chains that
    # SEAL cannot build (prime sizes or not enough NTT-friendly primes) are kept aside in
    # self.infeasible instead of being evaluated.
    def generate(self) -> list[CKKSParams]:
        params = []
        self.infeasible = []

        for deg in self.policy.poly_degrees:
            plan = self.policy.scale_plan(
                self.bits_decimal, deg, self.depth, self.output_int_bits
            )
            for scale_bits, extra_levels in sorted(plan.items()):
                chains = self.policy.generate_chains(
                    poly_degree=deg,
                    scale_bits=scale_bits,
                    depth=self.depth,
                    output_int_bits=self.output_int_bits,
                    extra_levels=extra_levels,
                )
                for chain in chains:
                    candidate = CKKSParams(poly_degree=deg, scale=2**scale_bits, chain=chain)
                    reason = chain_infeasibility_reason(deg, chain)
                    if reason:
                        self.infeasible.append((candidate, reason))
                    else:
                        params.append(candidate)

        return params

    # the same structure (N, number of levels, q0 offset, special prime rule) at the next lower
    # scale that SEAL can build, or None once the data-derived lower bound is passed. Used when
    # a candidate fails or overflows: the intermediate values were too large for its modulus.
    def step_down(self, params: CKKSParams) -> CKKSParams | None:
        step = self.policy.step_down_bits
        if step <= 0:
            return None

        out = self.output_int_bits
        lower = max(self.policy.lower_scale_bits(self.bits_decimal), 2)
        old_scale = params.log_scale
        q0_offset = params.chain[0] - self.policy.base_q0(old_scale, out)
        special_is_min = params.chain[-1] == max(params.chain[0], old_scale)

        scale = old_scale - step
        while scale >= lower:
            q0 = self.policy.base_q0(scale, out) + q0_offset
            special = max(q0, scale) if special_is_min else max(params.chain[-1], q0)
            chain = (q0,) + (scale,) * params.levels + (special,)
            candidate = CKKSParams(poly_degree=params.poly_degree, scale=2**scale, chain=chain)

            if sum(chain) <= ALLOWED_CHAIN_BITS[params.poly_degree]:
                reason = chain_infeasibility_reason(params.poly_degree, chain)
                if not reason:
                    return candidate
                self.infeasible.append((candidate, reason))
            scale -= step

        return None

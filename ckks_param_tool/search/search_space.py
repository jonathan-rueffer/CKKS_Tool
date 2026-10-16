from ckks_param_tool.params import CKKSParams, chain_infeasibility_reason
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

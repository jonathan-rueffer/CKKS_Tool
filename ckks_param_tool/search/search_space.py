from ckks_param_tool.params import CKKSParams, chain_infeasibility_reason
from .policy import CKKSSearchPolicy, ChainStructure

'''
This file contains the class that defines the search space generation functionality
'''


class CKKSParameterSearchSpace:
    # target_bits: an optional precision target that raises the lowest scale considered
    def __init__(
        self,
        depth: int,
        output_int_bits: int,
        policy: CKKSSearchPolicy | None = None,
        target_bits: int | None = None,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.target_bits = target_bits
        self.output_int_bits = output_int_bits
        self.policy = policy or CKKSSearchPolicy.default()

        # candidates that the policy produced but SEAL cannot instantiate, with the reason
        self.infeasible: list[tuple[CKKSParams, str]] = []

        # the chain structure every generated candidate belongs to (used by the step-down)
        self.structure_of: dict[CKKSParams, ChainStructure] = {}

    # generates the candidate parameter search space according to the policy. Chains that
    # SEAL cannot build (prime sizes or not enough NTT-friendly primes) are kept aside in
    # self.infeasible instead of being evaluated.
    def generate(self) -> list[CKKSParams]:
        params = []
        self.infeasible = []
        self.structure_of = {}

        for deg in self.policy.poly_degrees:
            plan = self.policy.scale_plan(self.target_bits, deg, self.depth, self.output_int_bits)
            for scale_bits, structures in sorted(plan.items()):
                for chain, structure in self.policy.generate_chains(
                    poly_degree=deg,
                    scale_bits=scale_bits,
                    depth=self.depth,
                    output_int_bits=self.output_int_bits,
                    structures=structures,
                ):
                    candidate = CKKSParams(poly_degree=deg, scale=2**scale_bits, chain=chain)
                    if candidate in self.structure_of:
                        continue
                    reason = chain_infeasibility_reason(deg, chain)
                    if reason:
                        self.infeasible.append((candidate, reason))
                    else:
                        self.structure_of[candidate] = structure
                        params.append(candidate)

        return params

    # the retry for a candidate that fails or deviates catastrophically (typically an overflow):
    # the same N, final modulus, number of levels and special prime at the next lower scale that
    # SEAL can build, or None once the lowest scale considered is passed.
    #
    # The final modulus is KEPT, so each step adds step_down_bits of output headroom
    # (final modulus bits - log2(Delta) - 1); re-sizing the final modulus for the lower scale
    # would leave the headroom unchanged and could never fix an overflow. Overflow is only decided
    # at decryption (the arithmetic is modular), so the final modulus is all that matters. A
    # step-down also adds noise relative to the scale (5 bits less precision); it does not
    # guarantee that the retried candidate is safe on other inputs. Only the scaling primes
    # shrink, so the total modulus shrinks and the
    # candidate stays at least as secure as the original. A retried candidate keeps the
    # structure of its original (structure_of), but its final modulus is larger than the
    # structure's minimum for its scale, so it is generally not a candidate of broad_grid.
    def step_down(self, params: CKKSParams) -> CKKSParams | None:
        step = self.policy.step_down_bits
        structure = self.structure_of.get(params)
        if step <= 0 or structure is None:
            return None

        final_primes = 2 if structure.two_prime_q0 else 1
        final, special = params.chain[:final_primes], params.chain[-1]
        levels = len(params.chain) - final_primes - 1

        lower = max(self.policy.lower_scale_bits(self.target_bits), 2)
        scale = params.log_scale - step
        while scale >= lower:
            chain = final + (scale,) * levels + (special,)
            candidate = CKKSParams(poly_degree=params.poly_degree, scale=2**scale, chain=chain)
            reason = chain_infeasibility_reason(params.poly_degree, chain)
            if not reason:
                self.structure_of.setdefault(candidate, structure)
                return candidate
            self.infeasible.append((candidate, reason))
            scale -= step

        return None

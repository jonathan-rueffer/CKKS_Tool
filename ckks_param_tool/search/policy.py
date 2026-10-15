from __future__ import annotations
from dataclasses import dataclass
import math

'''
This file contains the class that outlines the hardcoded rules that the search space generation
stage follows.
'''

@dataclass
class CKKSSearchPolicy:

    # what modulus bit sizes can be included in the chain
    allowed_chain_bits: dict[int, int]

    # how many scale candidates to include
    num_scale_options: int = 3

    # step size between candidate scales
    scale_step: int = 10

    # base buffer added to decimal precision
    base_scale_buffer: int = 5

    # extra buffer that grows with depth, not included due to testing results
    depth_buffer_factor: int = 10

    # added chain lengths to include
    chain_length_offsets: tuple[int, int] = (2, 3)

    # what sizes to be added to first modulus q0
    q0_deltas: tuple[int, int, int] = (0, 10, 20)

    # max size of the final modulus qL
    max_qL: int = 60

    # helper function to round a number to the nearest 10
    def _round_to_10(self, n: int) -> int:
        return math.ceil(n / 10.0) * 10

    # defines a buffer to add to the scale based on depth, not used
    def buffer_from_depth(self, depth: int) -> int:
        if depth <= 1:
            return self.depth_buffer_factor
        return self.depth_buffer_factor * depth

    # returns the polynomial degrees that can support the required depth of the model
    def valid_degrees(self, depth: int) -> list[int]:
        degrees = []

        if depth <= 7:
            degrees.append(8192)

        if depth <= 14:
            degrees.append(16384)

        if depth <= 29:
            degrees.append(32768)

        return degrees

    # generates the candidate scale sizes based on the decimal precision infered from the dataset
    def candidate_scales(self, bits_decimal: int, depth: int) -> list[int]:

        buffer_bits = self.base_scale_buffer #+ self.buffer_from_depth(depth) 

        base = self._round_to_10(bits_decimal + buffer_bits)

        return [
            base + i * self.scale_step
            for i in range(self.num_scale_options)
        ]
    
    # generates candidate chains according to the policy constraints
    def generate_chains(
        self,
        poly_degree: int,
        scale_bits: int,
        depth: int,
        bits_integer: int,
        bits_decimal: int,
    ) -> list[tuple[int, ...]]:

        max_bits = self.allowed_chain_bits[poly_degree]

        base_q0 = self._round_to_10(bits_integer + bits_decimal)

        # constructing candidate q0
        q0_candidates = []
        for delta in self.q0_deltas:
            q0 = max(base_q0 + delta, scale_bits + delta)
            if q0 not in q0_candidates:
                q0_candidates.append(q0)

        # constructing candidate Q for each q0
        chains = []
        for extra in self.chain_length_offsets:
            chain_length = max(3, depth + extra)

            internal = (scale_bits,) * (chain_length - 2)

            for q0 in q0_candidates:
                for qL in range(q0, self.max_qL + 1, 10):

                    chain = (q0,) + internal + (qL,)

                    if sum(chain) <= max_bits:
                        chains.append(chain)

        return chains

    # produces the default search policy
    @classmethod
    def default(cls) -> "CKKSSearchPolicy":
        return cls(
            allowed_chain_bits={
                8192: 218,
                16384: 438,
                32768: 881,
            }
        )
    
    # produces a smaller candidate size for quick experimentation
    @classmethod
    def fast(cls) -> "CKKSSearchPolicy":
        return cls(
            allowed_chain_bits={
                8192: 218,
                16384: 438,
            },
            num_scale_options=1,             
            chain_length_offsets=(2,),       
            q0_deltas=(0,),                  
            max_qL=50                        
        )

    # produces a more expansive parameter space policy
    @classmethod
    def full(cls) -> "CKKSSearchPolicy":
        return cls(
            allowed_chain_bits={
                8192: 218,
                16384: 438,
                32768: 881,
            },
            num_scale_options=4,             
            chain_length_offsets=(2, 3, 4),  
            q0_deltas=(0, 10, 20, 30),       
            max_qL=60
        )

    # produces an even more broad parameter space polciy, used to compare tool vs near-exhaustive
    @classmethod
    def fuller(cls) -> "CKKSSearchPolicy":
        return cls(
            allowed_chain_bits={
                8192: 218,
                16384: 438,
                32768: 881,
            },

            num_scale_options=8,          
            scale_step=5,

            chain_length_offsets=(1, 2, 3, 4, 5, 6),

            q0_deltas=(0, 10, 20, 30, 40, 50),

            max_qL=60,

            base_scale_buffer=0
        )

    @classmethod
    def from_config(
        cls,
        config
    ):

        presets = {

            "default": cls.default,

            "fast": cls.fast,

            "full": cls.full,

            "fuller": cls.fuller,
        }

        if config.profile != "custom":

            policy = presets[
                config.profile
            ]()

        else:

            policy = cls.default()

        return cls(

            allowed_chain_bits=policy.allowed_chain_bits,

            num_scale_options=config.num_scale_options,

            scale_step=config.scale_step,

            base_scale_buffer=config.base_scale_buffer,

            chain_length_offsets=config.chain_length_offsets,

            q0_deltas=config.q0_deltas,

            max_qL=config.max_qL,
        )
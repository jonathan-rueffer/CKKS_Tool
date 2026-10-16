from dataclasses import dataclass
from .security import security_level
from .memory import estimate_ckks_memory

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
    
    @property
    def memory(self) -> dict[str, int]:
        return estimate_ckks_memory(self.poly_degree, self.chain)
    
   
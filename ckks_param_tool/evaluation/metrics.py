from dataclasses import dataclass, asdict
import math

'''
This file contains the classes that represent the metrics obtained during parameter evaluation
and inference evaluation.

error   : maximum absolute difference between the decrypted output and the plaintext output,
          over every output slot and every input (worst case)
runtime : median wall-clock time of one encrypted inference (encrypt + evaluate + decrypt)
'''

# bits of precision of an absolute error, -log2(error), the usual way CKKS precision is reported
def precision_bits(error: float) -> float:
    if error is None or not math.isfinite(error):
        return float("nan")
    if error <= 0:
        return float("inf")
    return -math.log2(error)


@dataclass
class ParameterEvaluationResult:
    error: float # worst-case plaintext-ciphertext error over the test vectors
    runtime: float # median encrypted inference runtime across test vectors (s)
    security_bits: float # security level of the chosen parameters
    status: str # was backend encryption success or failure
    failure_reason: str = "" # if failure, what was the reason
    runtime_std: float = float("nan") # standard deviation of the per-inference runtimes (s)
    eval_seconds: float = float("nan") # total time spent evaluating this candidate, keygen included (s)

    @property
    def precision_bits(self) -> float:
        return precision_bits(self.error)


@dataclass
class InferenceEvaluationResult:
    error: float # worst-case plaintext-ciphertext error over the real dataset entries
    runtime: float # median encrypted inference runtime (s)
    runtime_std: float = float("nan")
    status: str = "success"
    failure_reason: str = ""

    @property
    def precision_bits(self) -> float:
        return precision_bits(self.error)

    def to_dict(self): # used for output
        d = asdict(self)
        d["precision_bits"] = self.precision_bits
        return d

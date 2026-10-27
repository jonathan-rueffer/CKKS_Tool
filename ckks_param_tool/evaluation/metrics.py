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


# headroom bits left above the largest plaintext coefficient (nan if unknown): negative means
# the evaluated inputs overflowed or came within noise of it
def coefficient_headroom_bits(headroom_bits: int, max_coefficient: float) -> float:
    if max_coefficient is None or not math.isfinite(max_coefficient):
        return float("nan")
    return headroom_bits - math.log2(max_coefficient) if max_coefficient > 0 else float("inf")


# bits of precision relative to the output magnitude, comparable across workloads whose outputs
# have different sizes: precision_bits(error) + log2(max |output|)
def relative_precision_bits(error: float, max_output: float) -> float:
    if max_output is None or not math.isfinite(max_output) or max_output <= 0:
        return float("nan")
    return precision_bits(error) + math.log2(max_output)


@dataclass
class ParameterEvaluationResult:
    error: float # worst-case plaintext-ciphertext error over the test vectors
    runtime: float # median encrypted inference runtime across test vectors (s)
    security_bits: float # security level of the chosen parameters
    status: str # was backend encryption success or failure
    failure_reason: str = "" # if failure, what was the reason
    runtime_std: float = float("nan") # standard deviation of the per-inference runtimes (s)
    eval_seconds: float = float("nan") # total time spent evaluating this candidate, keygen included (s)
    max_output: float = float("nan") # largest |plaintext output| over the test vectors
    # largest |coefficient| of the decrypted results in units of the scale (it mixes the outputs
    # with rotation partial sums and broadcast biases in the other slots) and the headroom bits
    # left above it: what an overflow depends on (see TenSEALBackend.max_abs_coefficient)
    max_coefficient: float = float("nan")
    coefficient_headroom_bits: float = float("nan")
    # cost of evaluating this candidate through a remote service (if measured)
    keygen_seconds: float = float("nan") # context, relinearization and Galois key generation
    eval_key_bytes: float = float("nan") # serialized public context and evaluation keys
    input_ciphertext_bytes: float = float("nan") # one encrypted input
    output_ciphertext_bytes: float = float("nan") # one encrypted result
    encrypted_queries: int = 0 # encrypted inferences run (warm-up included)

    @property
    def precision_bits(self) -> float:
        return precision_bits(self.error)

    # precision relative to the output magnitude: -log2(error / max |output|)
    @property
    def relative_precision_bits(self) -> float:
        return relative_precision_bits(self.error, self.max_output)


@dataclass
class InferenceEvaluationResult:
    error: float # worst-case plaintext-ciphertext error over the real dataset entries
    runtime: float # median encrypted inference runtime (s)
    runtime_std: float = float("nan")
    status: str = "success"
    failure_reason: str = ""
    max_relative_error: float = float("nan") # worst error relative to the output magnitude
    prediction_agreement: float | None = None # fraction of rows with the same predicted class
    catastrophic_rate: float = 0.0 # fraction of rows with a catastrophic deviation (see task_metrics)
    num_rows: int = 0 # number of real rows evaluated
    max_output: float = float("nan") # largest |plaintext output| over the rows
    max_coefficient: float = float("nan") # largest |coefficient| of the decrypted results (scale units)
    coefficient_headroom_bits: float = float("nan") # headroom bits left above it

    @property
    def precision_bits(self) -> float:
        return precision_bits(self.error)

    @property
    def relative_precision_bits(self) -> float:
        return relative_precision_bits(self.error, self.max_output)

    def to_dict(self): # used for output
        d = asdict(self)
        d["precision_bits"] = self.precision_bits
        d["relative_precision_bits"] = self.relative_precision_bits
        return d

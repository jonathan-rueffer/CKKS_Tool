import time
import numpy as np


# raised when a result deviates catastrophically from the plaintext model. Two criteria:
#   - an error at least as large as the outputs themselves leaves no correct bit (a scale too
#     small for the noise, or an overflow);
#   - an error of at least the overflow threshold of the parameters (params.overflow_error): an
#     overflow changes a value by a multiple of about 2^(headroom + 1), which noise never does.
#     The first criterion alone misses overflows whose wrapped value keeps its sign: an output
#     of 50 with 3 bits of headroom wraps to 2 (error 48 < 50).
# The check only sees the evaluated inputs and does not prove the cause.
class CatastrophicDeviation(ArithmeticError):
    pass

'''
This file contains the shared measurement loop used by both evaluators.
'''

# runs encrypted inference on every input and returns (errors, runtimes, outputs), where outputs
# holds one (decrypted output, plaintext output) pair per timed inference. If coefficient_maxima
# is a list, the largest |coefficient| of each decrypted result is appended to it (outside the
# timed region; needs a backend with max_abs_coefficient).
#   - warmup_runs untimed inferences are run first so one-off costs (caches, lazy allocation,
#     thread start-up) are not attributed to the first timed input
#   - every input is timed timing_repeats times with a monotonic high-resolution clock
#   - the error of an input is the maximum absolute error over its output slots
def run_encrypted_inference(backend, model, inputs, warmup_runs: int = 1, timing_repeats: int = 1,
                            raise_on_catastrophic: bool = True, coefficient_maxima: list | None = None):
    inputs = [np.asarray(x) for x in inputs]
    overflow = overflow_error(backend, model)

    for i in range(warmup_runs):
        backend.decrypt(model.forward_encrypted(backend.encrypt(inputs[i % len(inputs)])))

    errors = []
    runtimes = []
    outputs = []
    max_output = 0.0

    for _ in range(timing_repeats):
        for x in inputs:
            start = time.perf_counter()
            enc_x = backend.encrypt(x)
            enc_y = model.forward_encrypted(enc_x)
            dec_y = backend.decrypt(enc_y)
            runtimes.append(time.perf_counter() - start)
            if coefficient_maxima is not None:
                coefficient_maxima.append(backend.max_abs_coefficient(enc_y))

            plain_y = np.atleast_1d(np.asarray(model(x), dtype=float))
            dec_y = np.atleast_1d(dec_y)[: plain_y.size]
            errors.append(float(np.max(np.abs(dec_y - plain_y))))
            outputs.append((dec_y, plain_y))
            max_output = max(max_output, float(np.max(np.abs(plain_y))))

    # see CatastrophicDeviation
    if raise_on_catastrophic:
        if max(errors) >= overflow:
            raise CatastrophicDeviation(
                f"catastrophic deviation (overflow): error {max(errors):.3g} >= overflow threshold "
                f"{overflow:.3g}"
            )
        if max_output > 0 and max(errors) >= max_output:
            raise CatastrophicDeviation(
                f"catastrophic deviation (likely overflow, or scale too small): error {max(errors):.3g} "
                f">= max |output| {max_output:.3g}"
            )

    return errors, runtimes, outputs


# the overflow threshold of the backend's parameters for the model's depth (inf if unknown)
def overflow_error(backend, model) -> float:
    params = getattr(backend, "params", None)
    depth = getattr(model, "multiplicative_depth", None)
    if params is None or depth is None:
        return float("inf")
    return params.overflow_error(depth)

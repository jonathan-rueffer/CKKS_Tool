import time
import numpy as np

'''
This file contains the shared measurement loop used by both evaluators.
'''

# runs encrypted inference on every input and returns (errors, runtimes).
#   - warmup_runs untimed inferences are run first so one-off costs (caches, lazy allocation,
#     thread start-up) are not attributed to the first timed input
#   - every input is timed timing_repeats times with a monotonic high-resolution clock
#   - the error of an input is the maximum absolute error over its output slots
def run_encrypted_inference(backend, model, inputs, warmup_runs: int = 1, timing_repeats: int = 1):
    inputs = [np.asarray(x) for x in inputs]

    for i in range(warmup_runs):
        backend.decrypt(model.forward_encrypted(backend.encrypt(inputs[i % len(inputs)])))

    errors = []
    runtimes = []
    max_output = 0.0

    for _ in range(timing_repeats):
        for x in inputs:
            start = time.perf_counter()
            enc_x = backend.encrypt(x)
            enc_y = model.forward_encrypted(enc_x)
            dec_y = backend.decrypt(enc_y)
            runtimes.append(time.perf_counter() - start)

            plain_y = np.atleast_1d(np.asarray(model(x), dtype=float))
            dec_y = np.atleast_1d(dec_y)[: plain_y.size]
            errors.append(float(np.max(np.abs(dec_y - plain_y))))
            max_output = max(max_output, float(np.max(np.abs(plain_y))))

    # an error at least as large as the outputs themselves means the result overflowed the
    # modulus (e.g. intermediate products too large for the top modulus) and is meaningless
    if max_output > 0 and max(errors) >= max_output:
        raise OverflowError(
            f"result overflowed: error {max(errors):.3g} >= max |output| {max_output:.3g}"
        )

    return errors, runtimes

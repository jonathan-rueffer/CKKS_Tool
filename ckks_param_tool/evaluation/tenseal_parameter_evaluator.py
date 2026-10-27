import time
import numpy as np

from .metrics import ParameterEvaluationResult, coefficient_headroom_bits
from .parameter_evaluator import ParameterEvaluator
from .encrypted_run import run_encrypted_inference
from ckks_param_tool.params import CKKSParams
from ckks_param_tool.encryption import TenSEALBackend
from ckks_param_tool.config import EvaluationConfig

'''
This file contains the parameter evalation class that outlines the process of evaluating
candidate parameters using TenSEAL.
'''

class TenSEALParameterEvaluator(ParameterEvaluator):

    def __init__(self, model, test_vectors, config: EvaluationConfig | None = None):

        self.backend = TenSEALBackend
        self.model = model
        self.test_vectors = test_vectors
        self.config = config or EvaluationConfig()

    # given a parameter set, return a ParameterEvaluationResult object
    def evaluate(self, params: CKKSParams) -> ParameterEvaluationResult:
        start = time.perf_counter()
        try:

            # initialize TenSEAL backend using parameters (includes key generation)
            keygen_start = time.perf_counter()
            backend = self.backend(params, n_threads=self.config.n_threads)
            cost = {"keygen_seconds": time.perf_counter() - keygen_start}

            if self.config.measure_cost:
                enc_x = backend.encrypt(np.asarray(self.test_vectors[0]))
                cost.update({
                    "eval_key_bytes": backend.evaluation_key_bytes(),
                    "input_ciphertext_bytes": backend.ciphertext_bytes(enc_x),
                    "output_ciphertext_bytes": backend.ciphertext_bytes(self.model.forward_encrypted(enc_x)),
                })

            coefficients = []
            errors, runtimes, outputs = run_encrypted_inference(
                backend,
                self.model,
                self.test_vectors,
                warmup_runs=self.config.warmup_runs,
                timing_repeats=self.config.timing_repeats,
                coefficient_maxima=coefficients if self.config.measure_coefficients else None,
            )

            return ParameterEvaluationResult(
                error=max(errors),
                runtime=float(np.median(runtimes)),
                runtime_std=float(np.std(runtimes)),
                security_bits=params.security_score,
                status="success",
                failure_reason="",
                eval_seconds=time.perf_counter() - start,
                max_output=max(float(np.max(np.abs(plain))) for _, plain in outputs),
                max_coefficient=max(coefficients, default=float("nan")),
                coefficient_headroom_bits=coefficient_headroom_bits(
                    params.output_headroom_bits(self.model.multiplicative_depth),
                    max(coefficients, default=float("nan"))),
                encrypted_queries=self.config.warmup_runs + len(runtimes),
                **cost,
            )

        # handle case when encrypted inference fails, stores failure reason
        except Exception as e:
            return ParameterEvaluationResult(
                error=np.inf,
                runtime=np.inf,
                security_bits=params.security_score,
                status="failure",
                failure_reason=str(e),
                eval_seconds=time.perf_counter() - start,
            )

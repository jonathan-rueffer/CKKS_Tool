import numpy as np

from .metrics import InferenceEvaluationResult
from .inference_evaluator import InferenceEvaluator
from .encrypted_run import run_encrypted_inference
from ckks_param_tool.params import CKKSParams
from ckks_param_tool.encryption import TenSEALBackend
from ckks_param_tool.config import EvaluationConfig
from ckks_param_tool.models.base import predict

'''
This file contains the TenSEAL inference evaluation class, which defines how a given
recommended parameter set will be evaluated on real dataset entries.
'''
# task-level comparison of encrypted and plaintext outputs on real rows:
#   max_relative_error   : worst |decrypted - plaintext| relative to the largest plaintext output
#                          of that row (most meaningful for regression)
#   prediction_agreement : fraction of rows whose predicted class is the same encrypted and in
#                          plaintext (classification only)
def task_metrics(task: str, outputs) -> dict:
    rel = [
        float(np.max(np.abs(dec - plain)) / max(float(np.max(np.abs(plain))), 1e-300))
        for dec, plain in outputs
    ]
    agreement = None
    if task in ("binary", "multiclass"):
        agreement = float(np.mean([
            np.array_equal(predict(task, dec), predict(task, plain)) for dec, plain in outputs
        ]))
    return {"max_relative_error": max(rel), "prediction_agreement": agreement}


class TenSEALInferenceEvaluator(InferenceEvaluator):

    def __init__(self, model, X, config: EvaluationConfig | None = None):

        self.backend = TenSEALBackend
        self.model = model
        self.X = np.asarray(X)
        self.config = config or EvaluationConfig()

        # the same real entries are used for every parameter set so results are comparable
        rng = np.random.default_rng(self.config.random_state)
        size = min(self.config.num_samples, self.X.shape[0])
        self.sample_idx = rng.choice(self.X.shape[0], size=size, replace=False)

    # Given a parameter set, returns an InferenceEvaluationResult object
    def evaluate(self, params: CKKSParams) -> InferenceEvaluationResult:

        try:
            backend = self.backend(params, n_threads=self.config.n_threads)

            errors, runtimes, outputs = run_encrypted_inference(
                backend,
                self.model,
                self.X[self.sample_idx, :],
                warmup_runs=self.config.warmup_runs,
                timing_repeats=self.config.timing_repeats,
            )

            return InferenceEvaluationResult(
                error=max(errors), # maximum error across dataset samples
                runtime=float(np.median(runtimes)), # median runtime across dataset samples
                runtime_std=float(np.std(runtimes)),
                **task_metrics(getattr(self.model, "task", "regression"), outputs),
            )

        # handle case of encrypted inference fails
        except Exception as e:
            return InferenceEvaluationResult(
                error=np.inf,
                runtime=np.inf,
                status="failure",
                failure_reason=str(e),
            )

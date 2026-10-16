import numpy as np
import pytest

from ckks_param_tool.evaluation.encrypted_run import run_encrypted_inference


class _IdentityBackend:
    def __init__(self, noise):
        self.noise = noise

    def encrypt(self, x):
        return np.asarray(x, dtype=float)

    def decrypt(self, y):
        return np.atleast_1d(y) + self.noise


class _SumModel:
    def __call__(self, x):
        return np.sum(x)

    def forward_encrypted(self, x):
        return np.sum(x)


def test_small_error_passes():
    errors, runtimes, outputs = run_encrypted_inference(_IdentityBackend(1e-6), _SumModel(), [np.ones(3)])
    assert errors[0] == pytest.approx(1e-6)
    assert len(runtimes) == 1


def test_overflow_is_reported():
    with pytest.raises(OverflowError):
        run_encrypted_inference(_IdentityBackend(32.0), _SumModel(), [np.ones(3)])


def test_predict_and_task_metrics():
    from ckks_param_tool.models.base import predict
    from ckks_param_tool.evaluation.tenseal_inference_evaluator import task_metrics

    assert predict("binary", [0.3]).tolist() == [1]
    assert predict("multiclass", [0.1, 2.0, -1.0]).tolist() == [1]

    outputs = [(np.array([0.1, 2.0]), np.array([0.0, 2.0])),   # same class
               (np.array([1.1, 1.0]), np.array([0.9, 1.0]))]   # class flips
    m = task_metrics("multiclass", outputs)
    assert m["prediction_agreement"] == 0.5
    assert m["max_relative_error"] == pytest.approx(0.2)
    assert task_metrics("regression", outputs)["prediction_agreement"] is None


def test_encrypted_multiclass_end_to_end():
    pytest.importorskip("tenseal")
    from ckks_param_tool.models import LinearModel
    from ckks_param_tool.params import CKKSParams
    from ckks_param_tool.encryption import TenSEALBackend

    rng = np.random.default_rng(0)
    model = LinearModel(rng.normal(size=(5, 3)), rng.normal(size=3), task="multiclass")
    backend = TenSEALBackend(CKKSParams(8192, 2**40, (60, 40, 60)))
    errors, _, _ = run_encrypted_inference(backend, model, rng.normal(size=(3, 5)), warmup_runs=0)
    assert max(errors) < 1e-5

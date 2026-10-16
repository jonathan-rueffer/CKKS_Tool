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
    errors, runtimes = run_encrypted_inference(_IdentityBackend(1e-6), _SumModel(), [np.ones(3)])
    assert errors[0] == pytest.approx(1e-6)
    assert len(runtimes) == 1


def test_overflow_is_reported():
    with pytest.raises(OverflowError):
        run_encrypted_inference(_IdentityBackend(32.0), _SumModel(), [np.ones(3)])

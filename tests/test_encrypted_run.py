import numpy as np
import pytest

from ckks_param_tool.evaluation.encrypted_run import run_encrypted_inference, CatastrophicDeviation


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


def test_catastrophic_deviation_is_reported():
    with pytest.raises(CatastrophicDeviation):
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


def test_catastrophic_rate_is_per_row():
    from ckks_param_tool.evaluation.tenseal_inference_evaluator import task_metrics

    outputs = [(np.array([2.0 + 32.0]), np.array([2.0])),      # garbage row (error 32 > output 2)
               (np.array([100.0 + 1e-9]), np.array([100.0])),  # fine
               (np.array([1.0 + 1e-9]), np.array([1.0]))]      # fine
    assert task_metrics("regression", outputs)["catastrophic_rate"] == 1 / 3


def test_tiny_outputs_are_not_catastrophic():
    from ckks_param_tool.evaluation.tenseal_inference_evaluator import task_metrics

    outputs = [(np.array([1e-8 + 1e-6]), np.array([1e-8]))] # logit near 0, small error
    assert task_metrics("binary", outputs)["catastrophic_rate"] == 0.0


def test_mnist_forward_handles_bias_before_matmul():
    # a plaintext bias added before a further matmul corrupts the result in TenSEAL; the MNIST
    # forward pass adds those biases as encrypted vectors
    pytest.importorskip("tenseal")
    torch = pytest.importorskip("torch")
    from ckks_param_tool.models.mnist_square import SquareNet, build_wrapped_mnist_model
    from ckks_param_tool.params import CKKSParams
    from ckks_param_tool.encryption import TenSEALBackend

    torch.manual_seed(0)
    net = SquareNet()
    with torch.no_grad():
        for layer in (net.fc1, net.fc2, net.fc3):
            layer.bias.uniform_(0.5, 1.0) # clearly non-zero biases
    model = build_wrapped_mnist_model(net)

    x = np.random.default_rng(0).uniform(size=(1, 784))
    backend = TenSEALBackend(CKKSParams(16384, 2**45, (60,) + (45,) * 5 + (60,)))
    errors, _, _ = run_encrypted_inference(backend, model, x, warmup_runs=0)
    assert max(errors) < 1e-2


class _WrappingBackend:
    # simulates an overflow: values wrap modulo 2^(headroom + 1) into [-2^headroom, 2^headroom)
    def __init__(self, params, depth):
        self.params = params
        self.wrap = 2.0 ** (params.output_headroom_bits(depth) + 1)

    def encrypt(self, x):
        return np.asarray(x, dtype=float)

    def decrypt(self, y):
        y = np.atleast_1d(y)
        return (y + self.wrap / 2) % self.wrap - self.wrap / 2


class _DepthOneSum(_SumModel):
    multiplicative_depth = 1


def test_overflow_that_keeps_its_sign_is_caught():
    # output 50 with 3 bits of headroom wraps to 2: the error 48 is below the output, so the
    # "error >= output" criterion alone would miss it
    from ckks_param_tool.params import CKKSParams

    params = CKKSParams(8192, 2**55, (59, 55, 59))
    assert params.output_headroom_bits(1) == 3
    backend = _WrappingBackend(params, depth=1)
    assert backend.decrypt(50.0)[0] == pytest.approx(2.0)
    with pytest.raises(CatastrophicDeviation, match="overflow"):
        run_encrypted_inference(backend, _DepthOneSum(), [np.array([50.0])], warmup_runs=0)
    # within the headroom nothing is flagged
    run_encrypted_inference(backend, _DepthOneSum(), [np.array([7.0])], warmup_runs=0)


def test_real_row_overflow_that_keeps_its_sign_is_counted():
    from ckks_param_tool.evaluation.tenseal_inference_evaluator import task_metrics

    outputs = [(np.array([2.0]), np.array([50.0])), (np.array([7.0]), np.array([7.0]))]
    assert task_metrics("regression", outputs)["catastrophic_rate"] == 0.0 # the old criterion
    assert task_metrics("regression", outputs, overflow_error=2.0 ** 3)["catastrophic_rate"] == 0.5


def test_headroom_counts_every_unconsumed_prime():
    from ckks_param_tool.params import CKKSParams

    assert CKKSParams(8192, 2**40, (49, 40, 49)).output_headroom_bits(1) == 8
    assert CKKSParams(8192, 2**60, (20, 60, 60, 60)).output_headroom_bits(1) == 19 # two-prime
    assert CKKSParams(8192, 2**40, (49, 40, 40, 49)).output_headroom_bits(1) == 48 # spare level

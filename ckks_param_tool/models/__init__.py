from .base import BaseModel
from .linear import LinearModel
from .polynomial import PolynomialModel
from .wrapper import WrappedModel
from .training import (
    train_logistic_linear_regression_model,
    train_linear_regression_model,
)


# the MNIST network needs torch (optional extra "mnist"), so it is imported only when used
def train_mnist_model(*args, **kwargs):
    from .mnist_square import train_mnist_model as train
    return train(*args, **kwargs)


__all__ = [
    "BaseModel",
    "LinearModel",
    "PolynomialModel",
    "WrappedModel",
    "train_logistic_linear_regression_model",
    "train_linear_regression_model",
    "train_mnist_model"
]
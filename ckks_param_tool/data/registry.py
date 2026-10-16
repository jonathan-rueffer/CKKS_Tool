from .loaders import (
    load_diabetes_data,
    load_breast_cancer_data,
    load_wine_data,
    load_iris_data,
    load_california_housing_data,
    load_concrete_data,
    load_ionosphere_data,
    load_heart_disease_data,
    load_mnist_data,
)

'''
The built-in workloads: dataset name -> (loader, trainer, task). Trainers are imported lazily so
that torch is only needed for MNIST.
'''

def _linear(X, y):
    from ckks_param_tool.models import train_linear_regression_model
    return train_linear_regression_model(X, y)

def _logistic(X, y):
    from ckks_param_tool.models import train_logistic_linear_regression_model
    return train_logistic_linear_regression_model(X, y)

def _mnist(X, y):
    from ckks_param_tool.models import train_mnist_model
    return train_mnist_model(X, y)

DATASETS = {
    "diabetes": (load_diabetes_data, _linear, "regression"),
    "california_housing": (load_california_housing_data, _linear, "regression"),
    "concrete": (load_concrete_data, _linear, "regression"),
    "breast_cancer": (load_breast_cancer_data, _logistic, "classification"),
    "heart_disease": (load_heart_disease_data, _logistic, "classification"),
    "wine": (load_wine_data, _logistic, "classification"),
    "ionosphere": (load_ionosphere_data, _logistic, "classification"),
    "iris": (load_iris_data, _logistic, "classification"),
    "mnist": (load_mnist_data, _mnist, "classification"),
}

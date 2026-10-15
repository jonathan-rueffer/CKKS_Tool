from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    load_iris,
    fetch_california_housing
)
from sklearn.datasets import fetch_openml
import numpy as np

'''
Dataset loading utilities
'''

def load_breast_cancer_data():
    data = load_breast_cancer()
    return data.data, data.target


def load_diabetes_data():
    data = load_diabetes()
    return data.data, data.target


def load_wine_data():
    data = load_wine()
    return data.data, data.target


def load_iris_data():
    data = load_iris()
    return data.data, data.target


def load_california_housing_data():
    data = fetch_california_housing()
    return data.data, data.target

def load_concrete_data():
    data = fetch_openml(data_id=44959, as_frame=False)
    return data.data, data.target


def load_heart_disease_data():
    data = fetch_openml(
        data_id=43672,
        as_frame=False
    )

    X = data.data
    y = data.target

    return X, y

def load_ionosphere_data():
    data = fetch_openml(
        data_id=59,
        as_frame=False
    )

    X = data.data
    y = data.target

    return X, y

def load_mnist_data():

    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False
    )

    X = X.astype(np.float64)

    y = y.astype(int)

    X = X / 255.0

    return X, y

def load_cifar10_data():
    cifar = fetch_openml("CIFAR_10_small", version=1, as_frame=False)

    X = cifar.data.astype(np.float64)
    y = cifar.target.astype(int)

    # normalize
    X = X / 255.0

    return X, y
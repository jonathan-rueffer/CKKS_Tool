import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ckks_param_tool.models import train_linear_regression_model


def test_scaler_is_folded_into_weights():
    X, y = load_diabetes(return_X_y=True)
    model = train_linear_regression_model(X, y)

    scaler = StandardScaler().fit(X)
    reference = LinearRegression().fit(scaler.transform(X), y)

    np.testing.assert_allclose(model(X), reference.predict(scaler.transform(X)), rtol=1e-9, atol=1e-9)


def test_multiclass_logistic_is_a_full_classifier():
    from sklearn.datasets import load_wine
    from sklearn.linear_model import LogisticRegression
    from ckks_param_tool.models import train_logistic_linear_regression_model

    X, y = load_wine(return_X_y=True)
    model = train_logistic_linear_regression_model(X, y)
    assert model.task == "multiclass"
    assert model.W.shape == (X.shape[1], 3)

    scaler = StandardScaler().fit(X)
    reference = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    np.testing.assert_allclose(model(X), reference.decision_function(scaler.transform(X)), atol=1e-8)
    assert np.array_equal(np.argmax(model(X), axis=1), reference.predict(scaler.transform(X)))


def test_binary_logistic_task():
    from sklearn.datasets import load_breast_cancer
    from ckks_param_tool.models import train_logistic_linear_regression_model

    X, y = load_breast_cancer(return_X_y=True)
    model = train_logistic_linear_regression_model(X, y)
    assert model.task == "binary" and model.W.ndim == 1


def test_numpy_input_uses_plain_path():
    from ckks_param_tool.models import LinearModel

    model = LinearModel(np.ones((2, 3)), np.zeros(3), task="multiclass")
    assert model(np.array([1.0, 2.0])).shape == (3,)

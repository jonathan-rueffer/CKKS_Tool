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

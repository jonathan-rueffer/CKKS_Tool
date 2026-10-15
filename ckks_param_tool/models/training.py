import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from .linear import LinearModel
from .wrapper import WrappedModel
from sklearn.neural_network import MLPClassifier

'''
This file contains helper functions to train simply models on a dataset that are compatible 
with the structure of the tool.
'''

# this function takes the dataset and targets to train a logistic linear regression model, it
# returns a LinearModel object that contains the specific learned weights. Notably, there is
# no activation function for simplicity.
def train_logistic_linear_regression_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    return LinearModel(
        weights=clf.coef_[0],
        bias=clf.intercept_[0],
    )

# this function takes the dataset and targets to train a linear regression model, it
# returns a LinearModel object that contains the specific learned weights.
def train_linear_regression_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LinearRegression()
    clf.fit(X_scaled, y)

    return LinearModel(
        weights=clf.coef_,
        bias=clf.intercept_,
    )


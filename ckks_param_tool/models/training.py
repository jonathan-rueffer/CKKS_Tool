import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from .linear import LinearModel

'''
This file contains helper functions to train simple models on a dataset that are compatible
with the structure of the tool.

The models are trained on standardized features, but the client encrypts raw features. The
standardization is therefore folded into the weights:

    w . ((x - mu) / sigma) + b  =  (w / sigma) . x  +  (b - sum(w * mu / sigma))

so the encrypted computation stays a single dot product (or vector-matrix product) plus a bias
(depth 1) and returns the same prediction as the trained model.
'''

def _fold_scaler(scaler: StandardScaler, weights, bias):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        folded_weights = weights / scaler.scale_
        folded_bias = float(bias - np.sum(weights * scaler.mean_ / scaler.scale_))
    else: # one column per output
        folded_weights = weights / scaler.scale_[:, None]
        folded_bias = np.asarray(bias, dtype=float) - (scaler.mean_ / scaler.scale_) @ weights
    return folded_weights, folded_bias

# this function takes the dataset and targets to train a logistic regression model. It returns
# a LinearModel that computes the logits (no sigmoid / softmax, which do not change the
# predicted class): one logit for a binary target, one per class for a multi-class target.
def train_logistic_linear_regression_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    if len(clf.classes_) == 2:
        weights, bias = _fold_scaler(scaler, clf.coef_[0], clf.intercept_[0])
        return LinearModel(weights=weights, bias=bias, task="binary")

    weights, bias = _fold_scaler(scaler, clf.coef_.T, clf.intercept_)
    return LinearModel(weights=weights, bias=bias, task="multiclass")

# this function takes the dataset and targets to train a linear regression model, it
# returns a LinearModel object that contains the specific learned weights.
def train_linear_regression_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LinearRegression()
    clf.fit(X_scaled, y)

    weights, bias = _fold_scaler(scaler, clf.coef_, clf.intercept_)
    return LinearModel(weights=weights, bias=bias, task="regression")

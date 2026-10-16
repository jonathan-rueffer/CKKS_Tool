import numpy as np

from .base import BaseModel, TASKS

'''
This file contains the class for a linear machine learning model that can be used by the tool.
Notably, it only includes multiplications and additions, without an approximated activation
function. The weights are either a vector (one output: a regression value or a binary logit)
or a matrix with one column per output (e.g. one logit per class).
'''
class LinearModel(BaseModel):
    def __init__(self, weights, bias, task: str = "regression"):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        self.W = np.asarray(weights, dtype=float)
        self.b = bias if np.isscalar(bias) else np.asarray(bias, dtype=float)
        self.task = task

    def forward_plain(self, x):
        return x @ self.W + self.b

    def forward_encrypted(self, x_enc):
        if self.W.ndim == 1:
            return x_enc.dot(self.W) + self.b # one output: dot product
        return x_enc.matmul(self.W) + self.b.tolist() # one output per column: vector-matrix product

    # one ciphertext-plaintext multiplication (dot product or vector-matrix product)
    @property
    def multiplicative_depth(self):
        return 1

import numpy as np

'''
This file contains the abstract class that defines the machine learning model API used by the tool
'''

TASKS = ("regression", "binary", "multiclass")


class BaseModel:

    # the prediction task, used to turn raw outputs into predictions when the encrypted and
    # plaintext predictions are compared:
    #   regression : the output itself
    #   binary     : class 1 if the (single) logit is positive
    #   multiclass : the index of the largest output (argmax)
    task: str = "regression"

    def forward_plain(self, x):
        raise NotImplementedError

    def forward_encrypted(self, x_enc):
        raise NotImplementedError

    # the number of rescales the encrypted forward pass performs, i.e. the number of levels it
    # consumes. TenSEAL rescales after every multiplication, so ciphertext-plaintext
    # multiplications (dot, matmul, scalar mul) count as well as ciphertext-ciphertext ones.
    @property
    def multiplicative_depth(self):
        raise NotImplementedError

    # plaintext inputs are numpy arrays (or lists); anything else is treated as a ciphertext
    def __call__(self, x):
        if isinstance(x, (np.ndarray, list, tuple)):
            return self.forward_plain(np.asarray(x))
        return self.forward_encrypted(x)


def predict(task: str, outputs) -> np.ndarray:
    outputs = np.atleast_1d(np.asarray(outputs, dtype=float))
    if task == "regression":
        return outputs
    if task == "binary":
        return (outputs[:1] > 0).astype(int)
    if task == "multiclass":
        return np.array([int(np.argmax(outputs))])
    raise ValueError(f"unknown task '{task}', choose from {TASKS}")

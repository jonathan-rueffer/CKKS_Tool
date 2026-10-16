from .base import BaseModel, TASKS

'''
This file contains the class that defines how a pre-existing model can be wrapped to fit the
format of the tool.
'''


class WrappedModel(BaseModel):

    # it takes the original model, its predetermined depth (number of rescales), the specific
    # encrypted forward pass function (which is compatible with homomorphic operations), and
    # the prediction task
    def __init__(self, model, depth, encrypted_forward_fn, task: str = "regression"):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        self.model = model
        self._depth = depth
        self._encrypted_forward_fn = encrypted_forward_fn
        self.task = task

    def forward_plain(self, x):
        return self.model(x)

    def forward_encrypted(self, x_enc):
        return self._encrypted_forward_fn(x_enc)

    @property
    def multiplicative_depth(self):
        return self._depth

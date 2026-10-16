from .base import BaseModel

'''
This file contains a toy polynomial model that artificially multiplies weights repeatedly, 
and can easily be modified to reflect a specific multiplicative depth.
'''

class PolynomialModel(BaseModel):

    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        # self.W3 = W3
        # self.b3 = b3

    def forward_plain(self, x):
        h1 = x @ self.W1 + self.b1
        h2 = x @ self.W2 + self.b2
        # h3 = x @ self.W3 + self.b3
        return h1 * h2 #* h3

    def forward_encrypted(self, x_enc):
        h1 = x_enc.dot(self.W1) + self.b1
        h2 = x_enc.dot(self.W2) + self.b2
        #h3 = x_enc.dot(self.W3) + self.b3
        return h1 * h2 #* h3

    @property
    def multiplicative_depth(self):
        return 2

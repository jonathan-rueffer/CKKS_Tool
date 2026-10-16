'''
This file contains the abstract class that defines the machine learning model API used by the tool
'''

class BaseModel:
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

import tenseal as ts
import numpy as np
from ckks_param_tool.encryption import CKKSBackend
from ckks_param_tool.params import CKKSParams

'''
This file contains the encryption backend class used by the tool. It fully defines initialization,
and the operations of encryption and decryption using TenSEAL.
'''
class TenSEALBackend(CKKSBackend):
    ciphertext_type = ts.CKKSVector

    def __init__(self, params: CKKSParams, n_threads: int | None = None):
        self.params = params
        self.n_threads = n_threads
        self.context = self._create_context()

    def _create_context(self):
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.params.poly_degree,
            coeff_mod_bit_sizes=list(self.params.chain),
            n_threads=self.n_threads,
        )
        ctx.global_scale = self.params.scale
        ctx.generate_galois_keys() # rotations are needed for dot products and matmuls
        return ctx

    def encrypt(self, vector):
        return ts.ckks_vector(self.context, vector)

    def decrypt(self, enc_vector):
        return np.array(enc_vector.decrypt())

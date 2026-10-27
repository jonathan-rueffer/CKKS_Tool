import tenseal as ts
import tenseal.sealapi as sealapi
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

    # bytes the client sends the service so it can evaluate: the public context with the public,
    # relinearization and Galois keys, never the secret key
    def evaluation_key_bytes(self) -> int:
        return len(self.context.serialize(
            save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True
        ))

    @staticmethod
    def ciphertext_bytes(enc_vector) -> int:
        return len(enc_vector.serialize())

    def encrypt(self, vector):
        return ts.ckks_vector(self.context, vector)

    def decrypt(self, enc_vector):
        return np.array(enc_vector.decrypt())

    # the largest |coefficient| of the decrypted plaintext polynomial, in units of the scale: what
    # an overflow actually depends on (a value wraps once a coefficient times Delta reaches half
    # the final modulus). It is not the largest output: TenSEAL leaves rotation partial sums and
    # broadcast scalars (e.g. a bias) in the slots beyond the result, and the coefficients mix
    # all slots. Computed from every decoded slot by inverting the canonical embedding: slot k of
    # SEAL's CKKS encoder holds the polynomial's value at the root exp(i pi g / N), g = 5^k mod 2N,
    # and the conjugate root holds its conjugate. Exact up to the noise and the (noise-only)
    # imaginary parts; only meaningful if nothing overflowed. Client side: needs the secret key.
    def max_abs_coefficient(self, enc_vector) -> float:
        n = self.params.poly_degree
        if not hasattr(self, "_decryptor"):
            seal_context = self.context.seal_context().data
            self._decryptor = sealapi.Decryptor(seal_context, self.context.secret_key().data)
            self._encoder = sealapi.CKKSEncoder(seal_context)
            self._roots = np.array([pow(5, k, 2 * n) for k in range(n // 2)])
        plain = sealapi.Plaintext()
        self._decryptor.decrypt(enc_vector.ciphertext()[0], plain)
        slots = np.asarray(self._encoder.decode_double(plain), dtype=float)
        spectrum = np.zeros(2 * n)
        spectrum[self._roots] = slots
        spectrum[2 * n - self._roots] = slots
        coefficients = np.fft.fft(spectrum).real[:n] / n
        return float(np.max(np.abs(coefficients)))

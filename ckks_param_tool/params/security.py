'''
This file contains the security tables outlined by the Homomorphic Encryption Standard
(Albrecht et al., 2018), as implemented in Microsoft SEAL (native/src/seal/util/hestdparms.h).

Each entry gives the maximum total coefficient modulus size log2(Q) (in bits, summed over
every prime in the chain, including the special prime) that achieves the given security level
for a ring dimension N.

Assumptions behind these values (these are the defaults used by SEAL and therefore TenSEAL):
    - secret key sampled from the uniform ternary distribution {-1, 0, 1}
    - error sampled from a discrete Gaussian with standard deviation 3.2
    - classical (not quantum) attacker cost model

Note that TenSEAL itself only enforces the 128-bit table: a chain exceeding the 128-bit budget
is rejected when the context is created. The 192-bit and 256-bit labels are assigned by this
tool. For updated guidance see Bossuat et al., "Security Guidelines for Implementing
Homomorphic Encryption", IACR CiC 1(4), 2024 (ePrint 2024/463).
'''

SECURITY_LEVELS = (128, 192, 256)

SECURITY_TABLES = {
    128: {
        4096: 109,
        8192: 218,
        16384: 438,
        32768: 881,
    },
    192: {
        4096: 75,
        8192: 152,
        16384: 305,
        32768: 611,
    },
    256: {
        4096: 58,
        8192: 118,
        16384: 237,
        32768: 476,
    },
}

# the largest modulus budget allowed for each ring dimension (128-bit security)
ALLOWED_CHAIN_BITS = dict(SECURITY_TABLES[128])


# this function returns the highest security level satisfied by (N, sum(qi)) using the tables
def security_level(poly_degree: int, total_modulus_bits: int) -> int:
    for sec in sorted(SECURITY_TABLES, reverse=True):
        if total_modulus_bits <= SECURITY_TABLES[sec][poly_degree]:
            return sec
    return 0

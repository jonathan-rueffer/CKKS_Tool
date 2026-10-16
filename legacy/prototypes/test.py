import tenseal as ts

# Create context for CKKS scheme
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

# Set the global scale (precision control)
ctx.global_scale = 2 ** 40

# Generate rotation keys (for vector operations like add/mul)
ctx.generate_galois_keys()

# Encrypt and operate on a vector
vec = [1.1, 2.2, 3.3]
enc_vec = ts.ckks_vector(ctx, vec)
enc_vec.square_()

# Decrypt to check result
dec = enc_vec.decrypt()
print(dec)

# asl_ckks_test.py
import tenseal as ts
import numpy as np
import time
from asl_sampler import sample_uniform, sample_gaussian_clipped, sample_boundary_corners

# assume `model` is a trained linear model: weights (w) and bias (b)
# re-use or load your model; for demo we'll create a random w/b or load sklearn model

def build_ckks_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60], scale=2**40):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.global_scale = scale
    ctx.generate_galois_keys()
    # keep secret key for decrypt in ctx (in TenSEAL, local decrypt ok)
    return ctx

def run_encrypted_inference_on_samples(ctx, weights, bias, samples):
    results = []
    runtimes = []
    decrypt_ok = True
    for x in samples:
        try:
            t0 = time.perf_counter()
            enc = ts.ckks_vector(ctx, x.tolist())
            enc_res = enc.dot(weights.tolist()) + float(bias)
            # optionally relinearize/rescale is automatic in TenSEAL for simple ops
            dec = enc_res.decrypt()
            t1 = time.perf_counter()
            runtimes.append(t1 - t0)
            # dec may be scalar or list
            val = float(dec) if np.isscalar(dec) else float(dec[0])
            results.append(val)
        except Exception as e:
            decrypt_ok = False
            print("Exception during encrypted eval:", e)
            break
    return np.array(results), np.array(runtimes), decrypt_ok

# Example usage
if __name__ == "__main__":
    # load or create model weights
    d = 10
    np.random.seed(0)
    w = np.random.randn(d)
    b = 0.1

    # Example ASL for d features (replace with your real ASL)
    asl = [{"name": f"x{i}", "min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.3} for i in range(d)]

    # generate samples
    corners = sample_boundary_corners(asl)
    gaussian = sample_gaussian_clipped(asl, n=100)
    uniform = sample_uniform(asl, n=100)

    # pick context & scale; you will loop over scale / moduli in experiments
    ctx = build_ckks_context(scale=2**30)  # adjust scale

    # evaluate sets
    for name, S in [("corners", corners), ("gaussian", gaussian), ("uniform", uniform)]:
        # plaintext preds
        plaintext_preds = (S.dot(w) + b)
        enc_preds, times, ok = run_encrypted_inference_on_samples(ctx, w, b, S)
        if not ok:
            print(f"{name}: decryption or eval failed.")
            continue
        # compute metrics
        abs_err = np.abs(enc_preds - plaintext_preds)
        print(f"Set {name}: mean_err={abs_err.mean():.3e}, max_err={abs_err.max():.3e}, avg_time_ms={times.mean()*1000:.1f}")


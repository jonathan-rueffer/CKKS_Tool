# sweep_ckks_scale.py
# Requires: pip install tenseal scikit-learn numpy matplotlib
import time
import numpy as np
import tenseal as ts
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import statistics
import matplotlib.pyplot as plt

# --- 1) Prepare data & model (same as before) ---
X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = y / np.max(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
weights = model.coef_.astype(np.float64)
bias = float(model.intercept_)

# We'll test on a small subset for speed & reliability:
test_samples = X_test[:50]
plaintext_preds = np.dot(test_samples, weights) + bias

# --- 2) Parameter setup common across runs ---
poly_modulus_degree = 8192
# A modest coeff mod chain suitable for a shallow circuit (dot-product).
# You may need to increase this if you try larger scales or deeper circuits.
coeff_mod_bit_sizes = [60, 40, 40, 60]

# List of scales to test (powers of two)
scale_exponents = list(range(20, 49, 3))  # 2^20 .. 2^48 in steps
scales = [2 ** e for e in scale_exponents]

results = []

# Helper: run a single evaluation for given scale
def eval_scale(scale, verbose=False):
    # create fresh context per scale (TenSEAL requires consistent context for decrypting)
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    ctx.global_scale = scale
    ctx.generate_galois_keys()
    # Note: secret key is kept in context for decrypt; TenSEAL uses same context for encrypt/decrypt locally.

    abs_errors = []
    runtimes = []
    decrypt_fail = False

    for x_vec, plain in zip(test_samples, plaintext_preds):
        t0 = time.perf_counter()
        try:
            enc_x = ts.ckks_vector(ctx, x_vec.tolist())
            # encrypted dot product with plaintext weights + bias
            enc_res = enc_x.dot(weights.tolist()) + bias
            dec = enc_res.decrypt()
            t1 = time.perf_counter()
            runtimes.append(t1 - t0)
            # decrypted value is a 1-element list (or scalar)
            dec_val = float(dec) if np.isscalar(dec) else float(dec[0])
            abs_errors.append(abs(dec_val - float(plain)))
        except Exception as e:
            # Capture failure (e.g., overflow or decryption error)
            decrypt_fail = True
            if verbose:
                print(f"[scale={scale}] exception: {e}")
            break

    if len(abs_errors) == 0:
        mean_error = float("inf")
        median_error = float("inf")
        avg_time = float("inf")
    else:
        mean_error = float(np.mean(abs_errors))
        median_error = float(np.median(abs_errors))
        avg_time = float(np.mean(runtimes))

    return {
        "scale": scale,
        "scale_exp": int(math.log2(scale)),
        "mean_abs_error": mean_error,
        "median_abs_error": median_error,
        "avg_runtime_s": avg_time,
        "decrypt_fail": decrypt_fail,
        "num_samples": len(abs_errors)
    }

# --- 3) Sweep scales ---
print("Starting scale sweep. This may take a few minutes depending on your machine...")
for s in scales:
    print(f"Testing scale 2^{int(math.log2(s))} ...", end=" ", flush=True)
    r = eval_scale(s, verbose=False)
    results.append(r)
    if r["decrypt_fail"]:
        print("FAIL")
    else:
        print(f"ok (mean_err={r['mean_abs_error']:.3e}, time={r['avg_runtime_s']*1000:.1f} ms/sample)")

# --- 4) Summarize & plot ---
# Filter out failing runs for plotting
valid = [r for r in results if not r["decrypt_fail"] and math.isfinite(r["mean_abs_error"])]

if len(valid) == 0:
    print("All runs failed. Consider lowering target scale or increase coeff_mod_bit_sizes.")
else:
    exps = [r["scale_exp"] for r in valid]
    mean_errs = [r["mean_abs_error"] for r in valid]
    times_ms = [r["avg_runtime_s"] * 1000 for r in valid]

    print("\nSweep results (valid runs):")
    for r in valid:
        print(f"2^{r['scale_exp']}: mean_err={r['mean_abs_error']:.3e}, median_err={r['median_abs_error']:.3e}, avg_time={r['avg_runtime_s']*1000:.1f}ms")

    # Plot error vs scale exponent
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(exps, mean_errs, marker='o')
    plt.xlabel("scale exponent (log2 Δ)")
    plt.ylabel("mean absolute error")
    plt.title("Precision vs scale")
    plt.grid(True)

    # Plot runtime vs scale exponent
    plt.subplot(1,2,2)
    plt.plot(exps, times_ms, marker='o')
    plt.xlabel("scale exponent (log2 Δ)")
    plt.ylabel("avg runtime (ms / sample)")
    plt.title("Runtime vs scale")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Recommend: choose smallest scale with mean_err below threshold
    # Example threshold (you can pick your own)
    threshold = 1e-3
    feasible = [r for r in valid if r["mean_abs_error"] <= threshold]
    if feasible:
        best = min(feasible, key=lambda r: r["scale_exp"])  # smallest exponent meeting threshold
        print(f"\nRecommendation: choose scale 2^{best['scale_exp']} (mean_err={best['mean_abs_error']:.3e})")
    else:
        best = min(valid, key=lambda r: r["mean_abs_error"])
        print(f"\nNo scale reached mean error <= {threshold:.1e}. Best observed: 2^{best['scale_exp']} mean_err={best['mean_abs_error']:.3e}")


import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tenseal as ts
import time

np.random.seed(0)

# === 1️⃣ Load Dataset ===
data = load_diabetes()
X = data.data
y = data.target
d = X.shape[1]


X_train, X_test_plain, y_train, y_test_plain = train_test_split(
    X, y, test_size=0.2, random_state=0
)
model = LinearRegression()
model.fit(X_train, y_train)

weights = model.coef_.astype(float)
bias = float(model.intercept_)

print("Trained model:")
print("Weights:", weights)
print("Bias:", bias)

# === 2️⃣ Compute feature descriptors ===
feature_descriptors = []
for i in range(X.shape[1]):
    feature = X[:, i]
    feature_descriptors.append({
        "name": data.feature_names[i],
        "min": np.min(feature),
        "max": np.max(feature),
        "mean": np.mean(feature),
        "std": np.std(feature),
        "var": np.var(feature),
        "skew": skew(feature),
        "kurtosis": kurtosis(feature)
    })

corr = np.corrcoef(X, rowvar=False)
stds = np.array([fd["std"] for fd in feature_descriptors])
cov = np.outer(stds, stds) * corr
# make PSD (tiny regularization)
cov = (cov + cov.T) * 0.5 + np.eye(d) * 1e-8
# Also compute the correlation matrix to guide packing decisions
corr_matrix = np.corrcoef(X, rowvar=False)


# === 4️⃣ Estimate dynamic range across all features ===
min_val = np.min(X)
max_val = np.max(X)
dynamic_range = max_val - min_val

# === 5️⃣ Data-aware scaling factor heuristic ===
# Choose scaling so the largest value fits within a 60-bit modulus
target_modulus_bits = 60
max_cipher_val = 2 ** target_modulus_bits
# allocate about 1/8th of range for noise margin
scale_est = max_cipher_val / (8 * dynamic_range)

# Round to nearest power of 2 for CKKS compatibility
def nearest_power_of_two(x):
    return 2 ** int(np.round(np.log2(x)))


def generate_test_vectors(descriptors, n=128, mode="uniform", cov_mat=None):
    m = len(descriptors)
    means = np.array([desc["mean"] for desc in descriptors])
    stds = np.array([desc["std"] for desc in descriptors])
    mins = np.array([desc["min"] for desc in descriptors])
    maxs = np.array([desc["max"] for desc in descriptors])

    if mode == "uniform":
        samples = np.random.uniform(mins, maxs, size=(n, m))
    elif mode == "gaussian":
        samples = np.random.normal(means, stds, size=(n, m))
        # clip to observed ranges
        for i in range(m):
            samples[:, i] = np.clip(samples[:, i], mins[i], maxs[i])
    elif mode == "corner":
        # random corners: each feature min or max for each sample
        samples = np.vstack([np.random.choice([mins[i], maxs[i]], size=n) for i in range(m)]).T
    elif mode == "data-aware":
        if cov_mat is None:
            cov_mat = np.diag(stds**2)
        cov_mat = (cov_mat + cov_mat.T) * 0.5 + np.eye(m) * 1e-8
        samples = np.random.multivariate_normal(means, cov_mat, size=n)
        for i in range(m):
            samples[:, i] = np.clip(samples[:, i], mins[i], maxs[i])
    else:
        raise ValueError("Unknown mode")
    return samples


n_test = 128
sets = {
    "corner": generate_test_vectors(feature_descriptors, n=n_test, mode="corner"),
    "uniform": generate_test_vectors(feature_descriptors, n=n_test, mode="uniform"),
    "gaussian": generate_test_vectors(feature_descriptors, n=n_test, mode="gaussian"),
    "data-aware": generate_test_vectors(feature_descriptors, n=n_test, mode="data-aware", cov_mat=cov)
}

poly_mod_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]  # adjust if needed
scale = 2**30
#nearest_power_of_two(scale_est)

# Use positional args for compatibility with TenSEAL versions
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, coeff_mod_bit_sizes)
ctx.global_scale = scale
ctx.generate_galois_keys()

def encrypted_linear_predict(ctx, weights, bias, X_batch):
    # X_batch: (n_samples, d)
    d = X_batch.shape[1]
    # encrypt each feature column as a ckks_vector (length = n_samples)
    enc_cols = [ts.ckks_vector(ctx, X_batch[:, j].tolist()) for j in range(d)]
    # compute encrypted linear combination
    enc_y = None
    for w, enc in zip(weights, enc_cols):
        term = enc * float(w)
        enc_y = term if enc_y is None else (enc_y + term)
    if enc_y is None:
        enc_y = ts.ckks_vector(ctx, [0.0] * X_batch.shape[0])
    enc_y += float(bias)
    return enc_y

results = []
for name, X_test in sets.items():
    y_plain = X_test @ weights + bias

    t0 = time.perf_counter()
    enc_y = encrypted_linear_predict(ctx, weights, bias, X_test)
    t1 = time.perf_counter()

    dec = np.array(enc_y.decrypt())
    t2 = time.perf_counter()

    mean_err = float(np.mean(np.abs(dec - y_plain)))
    max_err = float(np.max(np.abs(dec - y_plain)))
    avg_time_ms = ((t1 - t0) + (t2 - t1)) / n_test * 1000.0

    results.append({
        "set": name,
        "mean_err": mean_err,
        "max_err": max_err,
        "avg_time_ms": avg_time_ms
    })

# display
print("\n=== Comparison results (trivial fixed linear model) ===")
for r in results:
    print(f"{r['set']:10s} | mean_err={r['mean_err']:.4e} | max_err={r['max_err']:.4e} | avg_time_ms={r['avg_time_ms']:.3f}")

best = min(results, key=lambda r: r["mean_err"])
print(f"\nBest set by mean_error: {best['set']} (mean_err={best['mean_err']:.4e})")
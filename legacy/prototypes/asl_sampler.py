# asl_sampler.py
import numpy as np
from scipy.stats import truncnorm

def sample_uniform(asl, n):
    # asl: list of dicts { "name":..., "min":a, "max":b, "mean":opt, "std":opt }
    samples = np.zeros((n, len(asl)), dtype=float)
    for j, feat in enumerate(asl):
        a, b = feat["min"], feat["max"]
        samples[:, j] = np.random.uniform(a, b, size=n)
    return samples

def sample_gaussian_clipped(asl, n):
    samples = np.zeros((n, len(asl)), dtype=float)
    for j, feat in enumerate(asl):
        a, b = feat["min"], feat["max"]
        mu = feat.get("mean", (a + b) / 2.0)
        sigma = feat.get("std", (b - a) / 6.0)  # default: ~99.7% inside range
        # truncated normal
        low, high = (a - mu) / sigma, (b - mu) / sigma
        rv = truncnorm.rvs(low, high, loc=mu, scale=sigma, size=n)
        samples[:, j] = rv
    return samples

def sample_boundary_corners(asl):
    # returns a small set of corner vectors: all min, all max, and alternating
    d = len(asl)
    v_min = np.array([f["min"] for f in asl], dtype=float)
    v_max = np.array([f["max"] for f in asl], dtype=float)
    v_alt = np.array([f["min"] if i % 2 == 0 else f["max"] for i, f in enumerate(asl)], dtype=float)
    return np.stack([v_min, v_max, v_alt], axis=0)

# Example ASL
if __name__ == "__main__":
    asl = [
        {"name": "age", "min": 20, "max": 80, "mean": 50, "std": 10},
        {"name": "bmi", "min": 15, "max": 40, "mean": 26, "std": 4},
        # add more features...
    ]
    print(sample_uniform(asl, n=5))
    print(sample_gaussian_clipped(asl, n=5))
    print(sample_boundary_corners(asl))

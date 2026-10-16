import tenseal as ts
import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import time
import itertools
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ALLOWED_CHAIN_BITS = {8192: 218, 16384: 438, 32768: 881}
VALID_CHAIN_PRIME_SIZES = [40, 60]

def generate_feature_descriptors(X):

    num_features = X.shape[1]
    feature_descriptors = []
    for i in range(num_features):
        col = X[:, i]
        stats = {
            "mean": float(np.mean(col)),
            "std_dev": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col))
        }
        feature_descriptors.append(stats)
    
    return feature_descriptors

def get_feature_samples(feature_descriptors, num_samples):

    mean = feature_descriptors["mean"]
    std_dev = feature_descriptors["std_dev"]
    min_val = feature_descriptors["min"]
    max_val = feature_descriptors["max"]

    # Convert min/max to standard-normal scale
    a, b = (min_val - mean) / std_dev, (max_val - mean) / std_dev 

    samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=num_samples)

    samples = np.round(samples, 2)

    return samples

def generate_test_vectors(feature_samples, num_samples):

    test_vectors = []

    for i in range(num_samples):

        test_vector = []
        for feature_sample in feature_samples:
            test_vector.append(float(feature_sample[i]))
        
        test_vectors.append(test_vector)
    
    return test_vectors

def max_bits_for_degree(poly_deg):
    match poly_deg:
        case 8192: return 218
        case 16384: return 438
        case 32768: return 881
        case _: return None

def find_min_chain_length(test_vectors, initial_chain, poly_degree, scale, max_total_bits, prime_size, model):

    results = []
    chain = initial_chain.copy()

    while sum(chain) + prime_size <= max_total_bits:
        try:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_degree,
                coeff_mod_bit_sizes=chain
            )
            ctx.global_scale = scale 
            ctx.generate_galois_keys()
            
            for vec in test_vectors:
                enc_x = ts.ckks_vector(ctx, vec)
                enc_y = model(enc_x)
                plain_y = enc_y.decrypt()
            
            return chain 
            
        except Exception as e:
            print(e)
            chain.append(prime_size)

    print("Error: Unable to add modulus (128-bit security constraint violation)")
    return None

def find_min_poly_degree(test_vectors, chain, scale, model):

    degrees = {8192, 16384, 32768}
    
    for deg in degrees:

        max_total_bits = max_bits_for_degree(deg)

        if sum(chain) < max_total_bits: 
            try:
                ctx = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=poly_degree,
                    coeff_mod_bit_sizes=chain
                )
                ctx.global_scale = scale 
                ctx.generate_galois_keys()
                
                for vec in test_vectors:
                    enc_x = ts.ckks_vector(ctx, vec)
                    enc_y = model(enc_x)
                    plain_y = enc_y.decrypt()
                
                return deg 
                
            except Exception as e:
                print(e)
    
    print("Error: ciphertext too large (128-bit security constraint violation)")
    return None

def find_scale(test_vectors, model, poly_degree, chain):

    scales = [2**10, 2**20, 2**30, 2**40, 2**50, 2**60]

    for scale in scales:
        try:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_degree,
                coeff_mod_bit_sizes=chain
            )
            ctx.global_scale = scale 
            ctx.generate_galois_keys()

            output = []

            for vec in test_vectors:
                enc_x = ts.ckks_vector(ctx, vec)
                enc_y = model(enc_x)
                dec_y = enc_y.decrypt()
                plain_y = model(np.array(vec))
    
                # print(scale,":")
               
                # print(vec)
                # print(plain_y)
                # print(dec_y)

                err = float(np.mean(np.abs(np.array(dec_y) - plain_y)))

                output.append(err)

            print(scale, ":", output)

        except Exception as e:
                print(e)
   
def get_security_score(deg, chain_sum):

    # Compute relative score normalized to 128-bit baseline
    base = ALLOWED_CHAIN_BITS[deg]  
    
    relative = base / chain_sum  
    # Weight slightly by degree
    weight = np.log2(deg / 8192 + 1)
    return relative * weight

def plot_precision_security(df, err_TOL=None):
    df_success = df[df["status"] == "success"]
    
    if err_TOL:
        df_success = df_success[df_success["max_err"] < err_TOL]
        
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df_success["max_err"], 
                df_success["security"], 
                alpha=0.7,
                s=50)

    plt.xscale("log")   
    plt.xlabel("Max Absolute Error")
    plt.ylabel("Estimated security")
    plt.title("Security vs Precision")
    plt.grid(True, which="both", ls="--", lw=0.5)

def plot_precision_runtime(df, err_TOL=None):
    df_success = df[df["status"] == "success"]
    
    if err_TOL:
        df_success = df_success[df_success["max_err"] < err_TOL]
        
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df_success["max_err"], 
                df_success["runtime"], 
                alpha=0.7,
                s=50)

    plt.xscale("log")    
    plt.xlabel("Max Absolute Error")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Precision")
    plt.grid(True, which="both", ls="--", lw=0.5)

def plot_runtime_security(df, err_TOL=None):
    df_success = df[df["status"] == "success"]
    
    if err_TOL:
        df_success = df_success[df_success["max_err"] < err_TOL]
        
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df_success["runtime"], 
                df_success["security"], 
                alpha=0.7,
                s=50)
  
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Estimated security")
    plt.title("Security vs Runtime")
    plt.grid(True, which="both", ls="--", lw=0.5)

def find_params_with_depth_128_bit(test_vectors, model, h_mult_depth):
    
    valid_degrees = []
    if h_mult_depth <= 7:
        valid_degrees.append(8192)   
    if h_mult_depth <= 14:
        valid_degrees.append(16384)
    if h_mult_depth <= 29:
        valid_degrees.append(32768)
    else:
        print("depth of model too large")
        return None

    valid_chains = []

    for deg in valid_degrees:
        chains = []
        for L in range(h_mult_depth + 1, h_mult_depth + 3):
             for tail in itertools.product(VALID_CHAIN_PRIME_SIZES, repeat=L-1):
                chain = (60,) + tail
                if sum(chain) <= ALLOWED_CHAIN_BITS[deg]:
                    chains.append(chain)

        valid_chains.append(chains)
    
    scales = [2**40,2**50,2**60]
    results = []

    for n, deg in enumerate(valid_degrees):
        for scale in scales:
            for chain in valid_chains[n]:
                try:
                    ctx = ts.context(
                        ts.SCHEME_TYPE.CKKS,
                        poly_modulus_degree=deg,
                        coeff_mod_bit_sizes=chain
                    )
                    ctx.global_scale = scale
                    ctx.generate_galois_keys()

                    output = []
                    start = time.time()
                    for vec in test_vectors:
                        enc_x = ts.ckks_vector(ctx, vec)
                        enc_y = model(enc_x)
                        dec_y = np.array(enc_y.decrypt())
                        plain_y = np.array(model(np.array(vec)))
                        err = np.mean(np.abs(dec_y - plain_y))
                        output.append(err)
                    max_err = np.max(output)
                    runtime = time.time() - start

                    results.append({
                        "poly_degree": deg,
                        "scale": scale,
                        "chain": chain,
                        "chain_sum": sum(chain),
                        "security": get_security_score(deg, sum(chain)),
                        "max_err": max_err,
                        "runtime": runtime,
                        "status": "success"
                    })

                    print(f"✓ degree={deg}, scale={scale:.1e}, chain={chain}, err={max_err:.2e}, time={runtime:.3f}s")

                except Exception as e:
                    print(f"⚠️ degree={deg}, scale={scale:.1e}, chain={chain} → {e}")
                    results.append({
                        "poly_degree": deg,
                        "scale": scale,
                        "chain": chain,
                        "chain_sum": sum(chain),
                        "security": get_security_score(deg, sum(chain)),
                        "max_err": np.nan,
                        "runtime": np.nan,
                        "status": f"fail: {type(e).__name__}"
                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    idx_min = df["max_err"].idxmin()
    deg = int(df.loc[idx_min, "poly_degree"])
    scale = int(df.loc[idx_min, "scale"])
    chain = df.loc[idx_min, "chain"]

    print("Lowest Error Parameters:", deg, scale, chain)

    min_ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=deg,
        coeff_mod_bit_sizes=chain
    )
    min_ctx.global_scale=scale
    #min_err_parameters = {"ctx": min_ctx, "scale": scale}

    plot_precision_security(df)
    plot_precision_runtime(df)
    plot_runtime_security(df)

    plt.show()

    return df, min_ctx

def enc_vs_plain_inference(X, num_samples, ctx, model):
    
    idx = np.random.choice(X.shape[0], size=num_samples, replace=False)
    X_sample = X[idx, :]

    results = []

    ctx.generate_galois_keys()

    for sample in X_sample:
        enc_x = ts.ckks_vector(ctx, sample)
        enc_y = model(enc_x)
        dec_y = np.array(enc_y.decrypt())
        plain_y = np.array(model(np.array(sample)))
        err = np.mean(np.abs(dec_y - plain_y))
        results.append(err)

    return max(results)
        
def iterate_params(test_vectors, model):
    # Parameters to sweep
    scales = [2**40]
    chains = [[60,40,40],[60,40,40,60],[60,40,60,40]]
    degrees = [8192, 16384, 32768]

    results = []

    # Parameter sweep
    for deg in degrees:
        for scale in scales:
            for chain in chains:
                try:
                    ctx = ts.context(
                        ts.SCHEME_TYPE.CKKS,
                        poly_modulus_degree=deg,
                        coeff_mod_bit_sizes=chain
                    )
                    ctx.global_scale = scale
                    ctx.generate_galois_keys()

                    output = []
                    start = time.time()
                    for vec in test_vectors:
                        enc_x = ts.ckks_vector(ctx, vec)
                        enc_y = model(enc_x)
                        dec_y = np.array(enc_y.decrypt())
                        plain_y = np.array(model(np.array(vec)))
                        err = np.mean(np.abs(dec_y - plain_y))
                        output.append(err)
                    avg_err = np.mean(output)
                    runtime = time.time() - start

                    results.append({
                        "poly_degree": deg,
                        "scale": scale,
                        "chain": str(chain),
                        "chain_len": len(chain),
                        "avg_err": avg_err,
                        "runtime": runtime,
                        "status": "success"
                    })

                    print(f"✓ degree={deg}, scale={scale:.1e}, chain={chain}, err={avg_err:.2e}, time={runtime:.3f}s")

                except Exception as e:
                    print(f"⚠️ degree={deg}, scale={scale:.1e}, chain={chain} → {e}")
                    results.append({
                        "poly_degree": deg,
                        "scale": scale,
                        "chain": str(chain),
                        "chain_len": len(chain),
                        "avg_err": np.nan,
                        "runtime": np.nan,
                        "status": f"fail: {type(e).__name__}"
                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # --- Plot dual heatmaps ---
    for deg in degrees:
        subset = df[df["poly_degree"] == deg].copy()
        if subset.empty:
            continue

        # Pivot tables for heatmaps
        pivot_err = subset.pivot_table(index="chain_len", columns="scale", values="avg_err")
        pivot_time = subset.pivot_table(index="chain_len", columns="scale", values="runtime")

        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        sns.heatmap(np.log10(pivot_err), annot=True, fmt=".2f", cmap="mako", ax=axes[0])
        sns.heatmap(np.log10(pivot_time), annot=True, fmt=".2f", cmap="crest", ax=axes[1])

        axes[0].set_title(f"log₁₀(Avg Error) — poly_degree={deg}")
        axes[1].set_title(f"log₁₀(Runtime [s]) — poly_degree={deg}")

        for ax in axes:
            ax.set_xlabel("Scale")
            ax.set_ylabel("Chain Length")

        plt.tight_layout()
        plt.show()

    return df

def blackbox_model_fhe(enc_x):

    result = enc_x**5
    return result

def blackbox_model_breast_cancer(enc_x, weights, bias):
    
    lin_comb = enc_x.dot(weights) + bias
    #sigmoid_approx = 0.5 + 0.25 * lin_comb - 0.015 * (lin_comb ** 3)
    return lin_comb

if __name__ == "__main__":

    # num_samples = 10
    # # Create 3 fake features
    # fake_features = [
    #     {"mean": 10, "std_dev": 2, "min": 5, "max": 15},
    #     {"mean": 0, "std_dev": 1, "min": -2, "max": 2},
    #     {"mean": 100, "std_dev": 10, "min": 80, "max": 120},
    # ]

    # # Generate samples for each feature
    # feature_samples = [get_feature_samples(fd, num_samples) for fd in fake_features]
    # test_vectors = generate_test_vectors(feature_samples, num_samples)

    # poly_degree = 8192
    # initial_scale = 2**20
    # initial_chain = [60, 40]  
    # max_total_bits = max_bits_for_degree(poly_degree)
    # prime_size = 60

    # chain = find_min_chain_length(test_vectors, initial_chain, poly_degree, initial_scale, max_total_bits, prime_size, blackbox_model_fhe)
    # print(chain)

    # if chain is not None: 
    #     deg = find_min_poly_degree(test_vectors, chain, initial_scale, blackbox_model_fhe)
    #     print(deg)

    #iterate_params(test_vectors, blackbox_model_fhe)

    #find_params_with_depth_128_bit(test_vectors, blackbox_model_logistic, 3)

        
    X, y = load_breast_cancer(return_X_y=True)

    # Split: model (train) vs. client (test/private)
    X_model, X_client, y_model, y_client = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model side: train a simple logistic regression
    scaler = StandardScaler()
    X_model_scaled = scaler.fit_transform(X_model)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_model_scaled, y_model)

    # Extract model weights
    weights = clf.coef_[0]
    bias = clf.intercept_[0]

    num_samples = 50    

    feature_descriptors = generate_feature_descriptors(X_client)
    feature_samples = [get_feature_samples(fd, num_samples) for fd in feature_descriptors]
    test_vectors = generate_test_vectors(feature_samples, num_samples)

    model = lambda enc_x: blackbox_model_breast_cancer(enc_x, weights, bias)

    _, best_params = find_params_with_depth_128_bit(test_vectors, model, 1)

    max_err = enc_vs_plain_inference(X_client, 5, best_params, model)

    print("max abs inference error:", max_err)




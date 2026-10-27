import pandas as pd

COLUMNS = [
    "poly_degree", "scale", "log_scale", "chain", "chain_sum", "chain_length", "levels",
    "max_err", "precision_bits", "relative_precision_bits", "runtime", "runtime_std", "eval_seconds", "security",
    "keygen_seconds", "eval_key_bytes", "input_ciphertext_bytes", "output_ciphertext_bytes",
    "encrypted_queries", "max_output", "max_coefficient", "coefficient_headroom_bits", "status", "failure_reason",
]

'''
This file contains a helper function for the plotting functionality. It turns the list of
(params, result) pairs into a structured dataframe.
'''
def results_to_dataframe(results):
    rows = []

    for result in results:
        params = result[0]
        metrics = result[1]

        rows.append({
            "poly_degree": params.poly_degree,
            "scale": params.scale,
            "log_scale": params.log_scale,
            "chain": params.chain,
            "chain_sum": sum(params.chain),
            "chain_length": len(params.chain),
            "levels": params.levels,
            "max_err": metrics.error,
            "precision_bits": metrics.precision_bits,
            "relative_precision_bits": metrics.relative_precision_bits,
            "runtime": metrics.runtime,
            "runtime_std": metrics.runtime_std,
            "eval_seconds": metrics.eval_seconds,
            "security": metrics.security_bits,
            "keygen_seconds": metrics.keygen_seconds,
            "eval_key_bytes": metrics.eval_key_bytes,
            "input_ciphertext_bytes": metrics.input_ciphertext_bytes,
            "output_ciphertext_bytes": metrics.output_ciphertext_bytes,
            "encrypted_queries": metrics.encrypted_queries,
            "max_output": metrics.max_output,
            "max_coefficient": metrics.max_coefficient,
            "coefficient_headroom_bits": metrics.coefficient_headroom_bits,
            "status": metrics.status,
            "failure_reason": metrics.failure_reason
        })

    return pd.DataFrame(rows, columns=COLUMNS)

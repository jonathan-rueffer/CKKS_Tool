import pandas as pd

COLUMNS = [
    "poly_degree", "scale", "log_scale", "chain", "chain_sum", "chain_length", "levels",
    "max_err", "precision_bits", "runtime", "runtime_std", "eval_seconds", "security",
    "total_memory", "status", "failure_reason",
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
            "runtime": metrics.runtime,
            "runtime_std": metrics.runtime_std,
            "eval_seconds": metrics.eval_seconds,
            "security": metrics.security_bits,
            "total_memory": params.memory["total"],
            "status": metrics.status,
            "failure_reason": metrics.failure_reason
        })

    return pd.DataFrame(rows, columns=COLUMNS)

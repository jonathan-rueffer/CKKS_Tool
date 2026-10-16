from ckks_param_tool.data.registry import DATASETS

'''
The benchmark workloads (see ckks_param_tool.data.registry).
'''

__all__ = ["DATASETS", "NO_FULLER"]

# workloads that are too expensive for the near-exhaustive comparison
NO_FULLER = {"mnist"}

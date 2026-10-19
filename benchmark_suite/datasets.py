from ckks_param_tool.data.registry import DATASETS, HELD_OUT

'''
The benchmark workloads (see ckks_param_tool.data.registry).
'''

__all__ = ["DATASETS", "HELD_OUT", "NO_BROAD_GRID"]

# workloads that are too expensive for the near-exhaustive comparison
NO_BROAD_GRID = {"mnist"}

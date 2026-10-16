import os
import platform
import sys
from importlib import metadata

'''
This file records the software and hardware environment of a run, since encrypted inference
runtimes are only comparable on the same machine, thread count and library versions.
'''

_PACKAGES = ("tenseal", "numpy", "scipy", "scikit-learn", "pandas", "torch")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment_info(n_threads: int | None = None) -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tenseal_threads": n_threads if n_threads is not None else os.cpu_count(),
        "packages": {p: _version(p) for p in _PACKAGES},
    }

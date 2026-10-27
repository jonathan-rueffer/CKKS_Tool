import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

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


# the commit of the tool's source tree and whether it had uncommitted changes
def _git_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "ckks_param_tool", "benchmark_suite"],
                               cwd=root, capture_output=True, text=True, check=True).stdout.strip() != ""
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


# the Microsoft SEAL version TenSEAL was built with (major.minor), read from the header SEAL
# writes in front of every serialized object (magic 0xA15E, header size, major, minor); TenSEAL
# reports only its own version
def _seal_version() -> str | None:
    try:
        import tempfile
        import tenseal as ts

        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40])
        ctx.global_scale = 2**20
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ct")
            ts.ckks_vector(ctx, [0.0]).ciphertext()[0].save(path)
            header = Path(path).read_bytes()[:5]
        if header[:2] != bytes((0x5E, 0xA1)):
            return None
        return f"{header[3]}.{header[4]}"
    except Exception:
        return None


def environment_info(n_threads: int | None = None) -> dict:
    return {
        "git": _git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tenseal_threads": n_threads if n_threads is not None else os.cpu_count(),
        "packages": {p: _version(p) for p in _PACKAGES},
        "seal": _seal_version(),
    }

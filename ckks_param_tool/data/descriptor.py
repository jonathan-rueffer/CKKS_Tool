import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

'''
This file contains the data descriptor: the only information about the client's data that the
tool needs. In the intended deployment the model owner (server) runs the tool, and the client
computes the descriptor locally and shares it instead of any record:

    per feature: mean, standard deviation, minimum, maximum, and resolution (the smallest step
    between two distinct values of that feature)

Everything the tool derives from the data (test vectors, output magnitude, precision
requirement) is computed from the descriptor.
'''

# relative size below which a gap between two values is treated as floating-point noise
_FLOAT_NOISE_REL = 2.0 ** -40


def _feature_resolution(col: np.ndarray) -> float | None:
    values = np.unique(col)
    if len(values) < 2: # constant feature
        return None
    gaps = np.diff(values)
    noise = _FLOAT_NOISE_REL * max(1.0, float(np.max(np.abs(values))))
    gaps = gaps[gaps > noise]
    return float(np.min(gaps)) if len(gaps) else None


@dataclass(frozen=True)
class DataDescriptor:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    min: tuple[float, ...]
    max: tuple[float, ...]
    resolution: tuple[float | None, ...] # None for constant features

    def __post_init__(self):
        n = len(self.mean)
        if not all(len(v) == n for v in (self.std, self.min, self.max, self.resolution)):
            raise ValueError("all descriptor fields need one value per feature")
        if any(lo > hi for lo, hi in zip(self.min, self.max)):
            raise ValueError("feature minimum larger than maximum")

    # computed by the client on its own data
    @classmethod
    def from_data(cls, X) -> "DataDescriptor":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return cls(
            mean=tuple(float(v) for v in X.mean(axis=0)),
            std=tuple(float(v) for v in X.std(axis=0)),
            min=tuple(float(v) for v in X.min(axis=0)),
            max=tuple(float(v) for v in X.max(axis=0)),
            resolution=tuple(_feature_resolution(col) for col in X.T),
        )

    @property
    def num_features(self) -> int:
        return len(self.mean)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": np.asarray(self.mean, dtype=float),
            "std": np.asarray(self.std, dtype=float),
            "min": np.asarray(self.min, dtype=float),
            "max": np.asarray(self.max, dtype=float),
        }

    # bits needed for the largest input magnitude
    def input_int_bits(self) -> int:
        max_abs = max(max(abs(v) for v in self.min), max(abs(v) for v in self.max))
        return max(0, math.ceil(math.log2(max_abs))) if max_abs > 0 else 1

    # bits needed to resolve the finest step of any single feature, capped
    def resolution_bits(self, max_bits: int = 40) -> int:
        bits = 0
        for gap in self.resolution:
            if gap is not None:
                bits = max(bits, math.ceil(-math.log2(gap) - 1e-9)) # tolerance for gaps like 0.4999...
        return min(max(bits, 0), max_bits)

    # one vector per feature extreme: that feature at its minimum or maximum, the others at their
    # mean. These cover single-feature outliers, which random samples almost never reach.
    def stress_vectors(self) -> np.ndarray:
        a = self.arrays()
        rows = []
        for i in range(self.num_features):
            for value in (a["min"][i], a["max"][i]):
                row = a["mean"].copy()
                row[i] = value
                rows.append(row)
        return np.array(rows)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataDescriptor":
        return cls(**{k: tuple(v) for k, v in d.items()})

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "DataDescriptor":
        return cls.from_dict(json.loads(Path(path).read_text()))

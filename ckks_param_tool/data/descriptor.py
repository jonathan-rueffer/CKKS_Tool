import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

'''
This file contains the data descriptor: the only information about the client's data that the
tool uses. The client runs the tool and computes the descriptor locally, so no record enters
the tool:

    per feature: mean, standard deviation, minimum and maximum

Everything the tool derives from the data (test vectors, output bound, decision margins) is
computed from the descriptor. Synthetic vectors drawn from it are sent to the service in
plaintext, so the service learns the descriptor but no record.
'''

@dataclass(frozen=True)
class DataDescriptor:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    min: tuple[float, ...]
    max: tuple[float, ...]

    def __post_init__(self):
        n = len(self.mean)
        if not all(len(v) == n for v in (self.std, self.min, self.max)):
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

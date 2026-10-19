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

DataDescriptor.private releases an epsilon-differentially private descriptor instead, which
bounds what the service can learn about any single record from the descriptor (see there).
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

    # epsilon-differentially private descriptor (Laplace mechanism). Neighbouring datasets differ
    # in one replaced row; the number of rows n is public.
    #   bounds : public per-feature ranges (lo, hi), e.g. from domain knowledge. The data is
    #            clipped into them and they are released as the descriptor's min / max, so the
    #            range costs no privacy budget. They must not be computed from the private data.
    #   epsilon: total budget, split evenly over the mean and the second moment of every
    #            feature (sequential composition over 2 * d releases)
    # On clipped data the mean has sensitivity (hi - lo) / n and the second moment
    # (max x^2 - min x^2) / n; the standard deviation is derived from the two noisy moments.
    @classmethod
    def private(cls, X, epsilon: float, bounds, rng=None) -> "DataDescriptor":
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        lo, hi = (np.asarray(b, dtype=float) for b in bounds)
        if lo.shape != (X.shape[1],) or hi.shape != (X.shape[1],) or np.any(lo > hi):
            raise ValueError("bounds must give one (lo, hi) pair per feature with lo <= hi")

        rng = np.random.default_rng(rng)
        n, d = X.shape
        eps_each = epsilon / (2 * d)
        Xc = np.clip(X, lo, hi)

        sq_lo = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(lo**2, hi**2))
        sq_hi = np.maximum(lo**2, hi**2)

        mean = Xc.mean(axis=0) + rng.laplace(scale=(hi - lo) / (n * eps_each))
        second = (Xc**2).mean(axis=0) + rng.laplace(scale=(sq_hi - sq_lo) / (n * eps_each))

        mean = np.clip(mean, lo, hi)
        std = np.sqrt(np.clip(second - mean**2, 0.0, ((hi - lo) / 2) ** 2))

        return cls(
            mean=tuple(float(v) for v in mean),
            std=tuple(float(v) for v in std),
            min=tuple(float(v) for v in lo),
            max=tuple(float(v) for v in hi),
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

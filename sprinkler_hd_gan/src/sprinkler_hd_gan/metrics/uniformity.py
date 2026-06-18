from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UniformityReport:
    mean_nn_spacing_px: float
    std_nn_spacing_px: float
    coeff_of_var: float


def nearest_neighbor_spacing_stats(heads_xy: list[tuple[int, int]]) -> UniformityReport:
    """Paper §3.3 notes spacing uniformity as a quality axis; cheap scalar summary."""
    if len(heads_xy) < 2:
        return UniformityReport(0.0, 0.0, 0.0)
    pts = np.asarray(heads_xy, dtype=np.float32)
    dists: list[float] = []
    for i, p in enumerate(pts):
        q = np.delete(pts, i, axis=0)
        d = np.linalg.norm(q - p, axis=1)
        dists.append(float(np.min(d)))
    arr = np.asarray(dists, dtype=np.float64)
    mu = float(arr.mean())
    sigma = float(arr.std())
    cv = float(sigma / mu) if mu > 1e-6 else 0.0
    return UniformityReport(mu, sigma, cv)

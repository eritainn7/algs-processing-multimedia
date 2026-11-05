from __future__ import annotations

import numpy as np


def fit_line_ransac(points: np.ndarray, tolerance: float = 3, attempts: int = 100):
    """Return a normalized line model and its supporting point indices."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return None, np.array([], dtype=np.int64)

    rng = np.random.default_rng(42)
    best_model = None
    best_inliers = np.array([], dtype=np.int64)
    for _ in range(attempts):
        first, second = rng.choice(len(points), size=2, replace=False)
        p1, p2 = points[first], points[second]
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        norm = np.hypot(a, b)
        if norm == 0:
            continue
        a, b = a / norm, b / norm
        c = -(a * p1[0] + b * p1[1])
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        inliers = np.flatnonzero(distances <= tolerance)
        if len(inliers) > len(best_inliers):
            best_model = np.array([a, b, c], dtype=np.float64)
            best_inliers = inliers
    return best_model, best_inliers

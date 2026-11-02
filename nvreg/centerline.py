"""Vessel centrelines from a label map, and their projection into the 2-D view.

The point of extracting a centreline here is to have a planning primitive of
the kind that gets overlaid on a live angiographic image, so that registration
accuracy can be reported as the error of that overlay rather than only as a
pose error.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize

from .geometry import CArmGeometry, Pose


def centerline_points(
    label: np.ndarray,
    spacing: tuple[float, float, float],
    max_points: int | None = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Skeletonize a binary vessel mask and return centreline points in volume mm.

    ``label`` is indexed ``[k, j, i]`` = ``[z, y, x]``, matching the renderer.
    Returns an (n, 3) array of (x, y, z) coordinates in volume-space mm.
    """
    skeleton = skeletonize(np.asarray(label, dtype=bool))
    kk, jj, ii = np.nonzero(skeleton)
    if ii.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    nk, nj, ni = label.shape
    sx, sy, sz = spacing
    pts = np.stack((
        (ii - (ni - 1) / 2) * sx,
        (jj - (nj - 1) / 2) * sy,
        (kk - (nk - 1) / 2) * sz,
    ), axis=-1).astype(np.float32)

    if max_points is not None and pts.shape[0] > max_points:
        rng = rng or np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    return pts


def project_centerline(
    points_vol: torch.Tensor,
    pose: Pose,
    geometry: CArmGeometry,
) -> torch.Tensor:
    """Project centreline points to detector pixel coordinates (col, row)."""
    uv_mm = geometry.project(pose.apply(points_vol))
    col = uv_mm[:, 0] / geometry.pixel_mm + (geometry.width - 1) / 2
    row = uv_mm[:, 1] / geometry.pixel_mm + (geometry.height - 1) / 2
    return torch.stack((col, row), dim=-1)

"""Synthetic volumes with analytically known projections, used by the tests.

The tests must run in CI without the CTA data, and a phantom whose line
integral can be worked out by hand is the only way to check that the renderer
is right rather than merely self-consistent.
"""

from __future__ import annotations

import numpy as np
import torch


def _grid_mm(shape_xyz, spacing):
    ni, nj, nk = shape_xyz
    sx, sy, sz = spacing
    x = (np.arange(ni) - (ni - 1) / 2) * sx
    y = (np.arange(nj) - (nj - 1) / 2) * sy
    z = (np.arange(nk) - (nk - 1) / 2) * sz
    # Index order is [k, j, i] = [z, y, x] to match the renderer.
    return np.meshgrid(z, y, x, indexing="ij")


def sphere(radius_mm=40.0, mu=0.02, shape_xyz=(96, 96, 96), spacing=(1.0, 1.0, 1.0),
           dtype=torch.float32):
    """Uniform sphere centred in the volume.

    A ray through the centre has path length ``2 * radius_mm``, so the central
    DRR pixel should read ``mu * 2 * radius_mm``.
    """
    zz, yy, xx = _grid_mm(shape_xyz, spacing)
    inside = (xx ** 2 + yy ** 2 + zz ** 2) <= radius_mm ** 2
    return torch.tensor(inside * mu, dtype=dtype)


def slab(thickness_mm=30.0, mu=0.02, shape_xyz=(96, 96, 96), spacing=(1.0, 1.0, 1.0),
         dtype=torch.float32):
    """Slab of given thickness normal to z, centred in the volume."""
    zz, _, _ = _grid_mm(shape_xyz, spacing)
    inside = np.abs(zz) <= thickness_mm / 2
    return torch.tensor(inside * mu, dtype=dtype)


def vessel_tree(shape_xyz=(96, 96, 96), spacing=(1.0, 1.0, 1.0), mu_vessel=0.05,
                mu_background=0.004, radius_mm=3.0, dtype=torch.float32):
    """A bent tube in a soft background, as a stand-in for a contrast-filled vessel.

    Returns ``(mu_volume, label_volume)``. The tube follows a smooth arc so
    that centreline extraction has something with a known, simple topology.
    """
    zz, yy, xx = _grid_mm(shape_xyz, spacing)
    # Arc: x traces a parabola in z, y stays near zero.
    axis_x = 0.002 * zz ** 2 - 20.0
    dist = np.sqrt((xx - axis_x) ** 2 + yy ** 2)
    label = (dist <= radius_mm) & (np.abs(zz) <= 0.35 * shape_xyz[2] * spacing[2])
    head = (xx ** 2 + yy ** 2 + zz ** 2) <= (0.45 * min(shape_xyz) * min(spacing)) ** 2
    mu = np.where(head, mu_background, 0.0)
    mu = np.where(label, mu_vessel, mu)
    return torch.tensor(mu, dtype=dtype), torch.tensor(label, dtype=torch.bool)

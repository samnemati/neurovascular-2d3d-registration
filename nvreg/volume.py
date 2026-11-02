"""Loading and preparing head CTA volumes for projection.

Volumes arrive as NIfTI in whatever orientation and spacing the scanner wrote.
They are reoriented to a canonical axis order, resampled to isotropic voxels,
and cropped to a field of view around the vessels of interest, so that every
case is rendered under the same geometry and the error numbers from different
cases mean the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


@dataclass
class HeadVolume:
    """A prepared volume, indexed ``[k, j, i]`` = ``[z, y, x]``."""

    hu: np.ndarray
    spacing: tuple[float, float, float]
    label: np.ndarray | None = None
    case_id: str = ""

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        nk, nj, ni = self.hu.shape
        return (ni * self.spacing[0], nj * self.spacing[1], nk * self.spacing[2])


def _load_canonical(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI in closest-canonical (RAS) orientation as [k, j, i]."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(img.dataobj, dtype=np.float32)  # (i, j, k) = (x, y, z)
    zooms = img.header.get_zooms()[:3]
    # Transpose to [k, j, i] so the fastest axis is x, matching the renderer.
    return np.ascontiguousarray(data.transpose(2, 1, 0)), tuple(float(z) for z in zooms)


def load_case(
    image_path: str | Path,
    label_path: str | Path | None = None,
    target_spacing: float = 1.0,
    fov_mm: float = 180.0,
    case_id: str = "",
) -> HeadVolume:
    """Load one CTA case, resample to isotropic, and crop to a cube of ``fov_mm``.

    The crop is centred on the centroid of the vessel label when one is given,
    which puts the circle of Willis near the isocenter, and on the image centre
    otherwise.
    """
    image_path = Path(image_path)
    hu, spacing = _load_canonical(image_path)

    label = None
    if label_path is not None:
        label_arr, label_spacing = _load_canonical(Path(label_path))
        if label_arr.shape != hu.shape:
            raise ValueError(
                f"label shape {label_arr.shape} does not match image {hu.shape} "
                f"for {image_path.name}"
            )
        label = label_arr > 0.5

    # Resample to isotropic voxels. Order 1 for the image, nearest for labels.
    factors = (spacing[2] / target_spacing, spacing[1] / target_spacing,
               spacing[0] / target_spacing)  # [k, j, i] order
    if not np.allclose(factors, 1.0, atol=1e-3):
        hu = ndimage.zoom(hu, factors, order=1)
        if label is not None:
            label = ndimage.zoom(label.astype(np.uint8), factors, order=0) > 0
    spacing = (target_spacing,) * 3

    centre = _crop_centre(hu, label)
    hu, label = _crop_cube(hu, label, centre, fov_mm, target_spacing)

    return HeadVolume(hu=np.ascontiguousarray(hu), spacing=spacing,
                      label=None if label is None else np.ascontiguousarray(label),
                      case_id=case_id or image_path.name)


def _crop_centre(hu: np.ndarray, label: np.ndarray | None) -> np.ndarray:
    if label is not None and label.any():
        return np.array(ndimage.center_of_mass(label))
    return np.array(hu.shape) / 2.0


def _crop_cube(hu, label, centre, fov_mm, spacing):
    half = int(round(fov_mm / spacing / 2))
    slices = []
    for axis, c in enumerate(centre):
        lo = int(round(c)) - half
        hi = lo + 2 * half
        # Shift the window to stay inside the array rather than padding, so the
        # rendered volume never contains fabricated air.
        if lo < 0:
            lo, hi = 0, min(2 * half, hu.shape[axis])
        if hi > hu.shape[axis]:
            hi = hu.shape[axis]
            lo = max(0, hi - 2 * half)
        slices.append(slice(lo, hi))
    sl = tuple(slices)
    return hu[sl], (None if label is None else label[sl])


def air_fill(hu: np.ndarray, value: float = -1000.0) -> np.ndarray:
    """Replace non-finite voxels with air, which some reconstructions leave as NaN."""
    return np.where(np.isfinite(hu), hu, value).astype(np.float32)

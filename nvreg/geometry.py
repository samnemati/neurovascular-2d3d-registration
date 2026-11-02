"""C-arm projection geometry and rigid pose parameterization.

Coordinate conventions
----------------------
World space is millimetres, right-handed, with the isocenter at the origin.
The X-ray source sits at ``(0, 0, -sad)`` and the flat detector lies in the
plane ``z = sdd - sad``, so that rays travel in the +z direction and the
isocenter is ``sad`` mm from the source. Detector columns run along +x (u) and
rows along +y (v).

Volume space is also millimetres, centred on the volume: voxel index
``(i, j, k)`` sits at ``((i - (ni-1)/2) * sx, (j - (nj-1)/2) * sy,
(k - (nk-1)/2) * sz)``.

A pose maps volume space into world space as ``p_world = R @ p_vol + t``. The
renderer never resamples the volume; it maps each ray into volume space with
the inverse pose, which is both cheaper and keeps the volume exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class CArmGeometry:
    """Cone-beam geometry of a C-arm, in millimetres and pixels.

    Defaults are in the range quoted for interventional angiography systems:
    roughly 1200 mm source-to-detector, 750 mm source-to-isocenter, and a flat
    detector about 30 cm across. At 160 pixels the pitch is 1.875 mm, which is
    far coarser than a clinical detector; it is the resolution these
    experiments run at, and the physical size is what decides how much anatomy
    is in the field of view (here 188 mm at the isocenter).
    """

    sdd: float = 1200.0       # source-to-detector distance
    sad: float = 750.0        # source-to-axis (isocenter) distance
    height: int = 160         # detector rows (v, along +y)
    width: int = 160          # detector columns (u, along +x)
    pixel_mm: float = 1.875   # detector pixel pitch (160 px -> 300 mm)

    @property
    def source(self) -> tuple[float, float, float]:
        return (0.0, 0.0, -self.sad)

    @property
    def detector_z(self) -> float:
        return self.sdd - self.sad

    @property
    def magnification(self) -> float:
        """Isocenter magnification: detector mm per isocenter mm."""
        return self.sdd / self.sad

    def rescale(self, factor: float) -> "CArmGeometry":
        """Same physical geometry at a coarser detector sampling.

        Used by the multi-resolution pyramid: halving the pixel count while
        doubling the pitch keeps the detector the same physical size.
        """
        return CArmGeometry(
            sdd=self.sdd,
            sad=self.sad,
            height=max(1, int(round(self.height * factor))),
            width=max(1, int(round(self.width * factor))),
            pixel_mm=self.pixel_mm / factor,
        )

    def detector_points(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """World-space coordinates of every detector pixel centre, (H, W, 3)."""
        u = (torch.arange(self.width, device=device, dtype=dtype)
             - (self.width - 1) / 2) * self.pixel_mm
        v = (torch.arange(self.height, device=device, dtype=dtype)
             - (self.height - 1) / 2) * self.pixel_mm
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        zz = torch.full_like(uu, self.detector_z)
        return torch.stack((uu, vv, zz), dim=-1)

    def project(self, points_world: torch.Tensor) -> torch.Tensor:
        """Perspective-project world points onto the detector, in mm (u, v).

        Points are returned in detector millimetres, not pixels, so that an
        error can be quoted at the detector without reference to pixel pitch.
        """
        x, y, z = points_world.unbind(dim=-1)
        # Ray from the source through the point, extended to the detector plane.
        scale = (self.detector_z - (-self.sad)) / (z + self.sad)
        return torch.stack((x * scale, y * scale), dim=-1)


def rotation_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: axis-angle vector (radians) to a 3x3 rotation.

    Differentiable, including at zero rotation, where the Taylor expansion of
    ``sin(a)/a`` and ``(1-cos(a))/a^2`` is used to keep the gradient finite.
    """
    angle = torch.linalg.norm(rotvec)
    eps = 1e-6
    if angle < eps:
        # Second-order expansion; exact to O(angle^4) and safe to differentiate.
        sin_over = 1.0 - angle ** 2 / 6.0
        one_minus_cos_over = 0.5 - angle ** 2 / 24.0
    else:
        sin_over = torch.sin(angle) / angle
        one_minus_cos_over = (1.0 - torch.cos(angle)) / angle ** 2

    wx, wy, wz = rotvec.unbind()
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    skew = torch.stack((
        torch.stack((zero, -wz, wy)),
        torch.stack((wz, zero, -wx)),
        torch.stack((-wy, wx, zero)),
    ))
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + sin_over * skew + one_minus_cos_over * (skew @ skew)


@dataclass
class Pose:
    """Rigid pose of the volume in world space: ``p_world = R @ p_vol + t``."""

    rotvec: torch.Tensor   # (3,) axis-angle, radians
    translation: torch.Tensor  # (3,) millimetres

    @staticmethod
    def identity(device=None, dtype=torch.float32) -> "Pose":
        z = torch.zeros(3, device=device, dtype=dtype)
        return Pose(z.clone(), z.clone())

    @staticmethod
    def from_vector(vec: torch.Tensor) -> "Pose":
        return Pose(vec[:3], vec[3:])

    def to(self, device=None, dtype=None) -> "Pose":
        """Move the pose onto a device, so it can meet a volume that lives there."""
        return Pose(self.rotvec.to(device=device, dtype=dtype),
                    self.translation.to(device=device, dtype=dtype))

    def to_vector(self) -> torch.Tensor:
        return torch.cat((self.rotvec, self.translation))

    def matrix(self) -> torch.Tensor:
        return rotation_matrix(self.rotvec)

    def apply(self, points_vol: torch.Tensor) -> torch.Tensor:
        """Map volume-space points (..., 3) into world space."""
        return points_vol @ self.matrix().T + self.translation

    def inverse_apply(self, points_world: torch.Tensor) -> torch.Tensor:
        """Map world-space points (..., 3) into volume space."""
        return (points_world - self.translation) @ self.matrix()

    def as_transform(self) -> "RigidTransform":
        """Materialize the rotation matrix once, for composing with a view."""
        return RigidTransform(self.matrix(), self.translation)


@dataclass
class RigidTransform:
    """A rigid transform that stores its rotation as a matrix.

    A view rotation composed onto an object pose has no reason to be turned
    back into an axis-angle vector, and doing so is numerically awkward near a
    half turn and awkward to differentiate at zero. Composition therefore
    happens on matrices, and this type is what the renderer consumes.
    """

    rotation: torch.Tensor      # (3, 3)
    translation: torch.Tensor   # (3,)

    def matrix(self) -> torch.Tensor:
        return self.rotation

    def apply(self, points_vol: torch.Tensor) -> torch.Tensor:
        return points_vol @ self.rotation.T + self.translation

    def inverse_apply(self, points_world: torch.Tensor) -> torch.Tensor:
        return (points_world - self.translation) @ self.rotation

    def compose(self, inner: "RigidTransform") -> "RigidTransform":
        """Applying ``inner`` then ``self``."""
        return RigidTransform(
            self.rotation @ inner.rotation,
            inner.translation @ self.rotation.T + self.translation,
        )


@dataclass(frozen=True)
class View:
    """One C-arm plane: a detector geometry plus where that plane sits.

    ``rotation`` is an axis-angle vector giving the rotation of the imaging
    chain about the isocenter. A biplane system is two views whose rotations
    differ by about 90 degrees, which is what makes the depth direction of one
    plane the in-plane direction of the other.
    """

    geometry: CArmGeometry = field(default_factory=CArmGeometry)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "frontal"

    def rotation_matrix(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return rotation_matrix(torch.tensor(self.rotation, device=device, dtype=dtype))

    def effective_pose(self, object_pose):
        """The object pose as this imaging plane sees it."""
        if all(r == 0.0 for r in self.rotation):
            return object_pose
        r_view = self.rotation_matrix(
            object_pose.translation.device, object_pose.translation.dtype)
        inner = (object_pose.as_transform() if isinstance(object_pose, Pose)
                 else object_pose)
        return RigidTransform(r_view, torch.zeros_like(inner.translation)).compose(inner)

    def rescale(self, factor: float) -> "View":
        return View(self.geometry.rescale(factor), self.rotation, self.name)


def biplane(geometry: CArmGeometry | None = None, separation_deg: float = 90.0):
    """The standard pair of planes: frontal, and a lateral rotated about y."""
    geom = geometry or CArmGeometry()
    angle = float(np.deg2rad(separation_deg))
    return (
        View(geom, (0.0, 0.0, 0.0), "frontal"),
        View(geom, (0.0, angle, 0.0), "lateral"),
    )

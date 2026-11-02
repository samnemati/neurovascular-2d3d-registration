"""Similarity measures for registration, and error measures for validating it.

The error measures follow the standardized evaluation methodology for 2D/3D
registration (van de Kraats et al., 2005): accuracy is reported as mean target
registration error over points distributed through the volume of interest, and
success is judged against a millimetre threshold rather than against the
similarity value, which says nothing about whether the pose is right.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import CArmGeometry, Pose


def ncc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized cross-correlation of two images, in [-1, 1]."""
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return (a * b).sum() / (denom + eps)


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.T.contiguous()


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of a 2-D image, same shape as the input."""
    kx = _SOBEL_X.to(image).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(image).view(1, 1, 3, 3)
    img = F.pad(image[None, None], (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(img, kx)
    gy = F.conv2d(img, ky)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)[0, 0]


def gradient_ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """NCC of Sobel gradient magnitudes.

    Preferred over plain NCC when the two images do not share an intensity
    model, which is the case as soon as the target is anything other than the
    renderer's own output.
    """
    return ncc(gradient_magnitude(a), gradient_magnitude(b))


SIMILARITY = {"ncc": ncc, "gncc": gradient_ncc}


def mtre(points_vol: torch.Tensor, pose_true: Pose, pose_est: Pose) -> float:
    """Mean target registration error in mm, over volume-space target points."""
    with torch.no_grad():
        p_true = pose_true.apply(points_vol)
        p_est = pose_est.apply(points_vol)
        return float(torch.linalg.norm(p_true - p_est, dim=-1).mean())


def projection_error(
    points_vol: torch.Tensor,
    pose_true: Pose,
    pose_est: Pose,
    geometry: CArmGeometry,
) -> float:
    """Mean 2-D error at the detector, in detector mm.

    This is what an overlay error looks like to the operator: how far the
    projected centreline lands from where it should, in the plane of the live
    image, rather than how far the pose is wrong in space.
    """
    with torch.no_grad():
        uv_true = geometry.project(pose_true.apply(points_vol))
        uv_est = geometry.project(pose_est.apply(points_vol))
        return float(torch.linalg.norm(uv_true - uv_est, dim=-1).mean())


def error_components(points_vol: torch.Tensor, pose_true, pose_est) -> tuple[float, float]:
    """Split the target error into in-plane and depth parts, in mm (RMS).

    "Depth" is the world z axis, which runs along the beam of the frontal
    plane. The split matters because a single projection constrains the two
    in-plane directions far better than it constrains depth, so a single mTRE
    number hides where the error actually is.
    """
    with torch.no_grad():
        delta = pose_true.apply(points_vol) - pose_est.apply(points_vol)
        inplane = torch.sqrt((delta[:, :2] ** 2).sum(dim=-1).mean())
        depth = torch.sqrt((delta[:, 2] ** 2).mean())
        return float(inplane), float(depth)


def pose_error(pose_true: Pose, pose_est: Pose) -> tuple[float, float]:
    """Rotation error in degrees and translation error in mm.

    The rotation error is the angle of the relative rotation, which is the
    only rotation error that does not depend on the parameterization.
    """
    with torch.no_grad():
        r_rel = pose_est.matrix().T @ pose_true.matrix()
        cos = torch.clamp((torch.trace(r_rel) - 1) / 2, -1.0, 1.0)
        angle_deg = float(torch.rad2deg(torch.arccos(cos)))
        trans_mm = float(torch.linalg.norm(pose_true.translation - pose_est.translation))
        return angle_deg, trans_mm

"""Differentiable digitally reconstructed radiograph (DRR) renderer.

The forward model is a single-energy line integral of linear attenuation:
for a ray from the source to a detector pixel, the rendered value is
``sum(mu(p) * dt)`` along the ray. Rays are mapped into volume space with the
inverse pose and sampled with ``grid_sample``, so the DRR is differentiable
with respect to the six pose parameters and the pose can be optimized by
gradient descent through the renderer.

What this model leaves out is listed in the repository README: there is no
X-ray energy spectrum and so no beam hardening, no scatter transport, no
detector response, and no soft-tissue motion. ``degrade`` adds a coarse
stand-in for the first-order effects (quantum noise, detector blur, a smooth
scatter floor) so that registration can at least be tested against something
other than its own forward model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import CArmGeometry, Pose, View

MU_WATER_PER_MM = 0.019
"""Linear attenuation of water near 70 keV, the effective energy assumed here."""


def hu_to_mu(hu: torch.Tensor, mu_water: float = MU_WATER_PER_MM) -> torch.Tensor:
    """Hounsfield units to linear attenuation coefficient, clamped at air."""
    return torch.clamp(mu_water * (1.0 + hu / 1000.0), min=0.0)


class Renderer:
    """Renders DRRs of one volume under arbitrary poses.

    Parameters
    ----------
    mu : (nk, nj, ni) tensor
        Linear attenuation per mm, indexed ``[k, j, i]`` = ``[z, y, x]``.
    spacing : (sx, sy, sz)
        Voxel size in mm along x, y, z.
    geometry : CArmGeometry
    n_samples : int
        Samples per ray. The step size adapts to each ray's chord through the
        volume's bounding sphere, so this sets accuracy, not extent.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        spacing: tuple[float, float, float],
        geometry: CArmGeometry | None = None,
        n_samples: int = 256,
    ) -> None:
        if mu.ndim != 3:
            raise ValueError(f"mu must be 3-D (nk, nj, ni), got shape {tuple(mu.shape)}")
        self.mu = mu
        self.spacing = tuple(float(s) for s in spacing)
        self.geometry = geometry or CArmGeometry()
        self.n_samples = int(n_samples)

        nk, nj, ni = mu.shape
        self.shape_xyz = (ni, nj, nk)
        self._volume = mu[None, None]  # (1, 1, D=nk, H=nj, W=ni)
        # Half-extent of the volume in mm, and the radius of its bounding sphere.
        self.half_extent = torch.tensor(
            [ni * self.spacing[0] / 2, nj * self.spacing[1] / 2, nk * self.spacing[2] / 2],
            dtype=mu.dtype, device=mu.device,
        )
        self.bounding_radius = float(torch.linalg.norm(self.half_extent))

    # -- coordinate helpers -------------------------------------------------

    def _to_grid(self, points_vol: torch.Tensor) -> torch.Tensor:
        """Volume-space mm to ``grid_sample`` coordinates in [-1, 1], (x, y, z)."""
        ni, nj, nk = self.shape_xyz
        sx, sy, sz = self.spacing
        x, y, z = points_vol.unbind(dim=-1)
        # Physical to fractional index, then to normalized coordinates. With
        # align_corners=True, index 0 maps to -1 and index n-1 maps to +1.
        gx = (x / sx + (ni - 1) / 2) / ((ni - 1) / 2) - 1.0
        gy = (y / sy + (nj - 1) / 2) / ((nj - 1) / 2) - 1.0
        gz = (z / sz + (nk - 1) / 2) / ((nk - 1) / 2) - 1.0
        return torch.stack((gx, gy, gz), dim=-1)

    # -- forward model ------------------------------------------------------

    def render(
        self,
        pose: Pose,
        geometry: CArmGeometry | None = None,
        view: View | None = None,
    ) -> torch.Tensor:
        """Render a DRR of shape (H, W) with the volume at ``pose``.

        If ``view`` is given, the pose is first mapped into that imaging
        plane's frame, and ``view.geometry`` is used unless ``geometry``
        overrides it (which the registration pyramid does).
        """
        if view is not None:
            pose = view.effective_pose(pose)
            geom = geometry or view.geometry
        else:
            geom = geometry or self.geometry
        device, dtype = self.mu.device, self.mu.dtype

        det = geom.detector_points(device=device, dtype=dtype).reshape(-1, 3)
        src = torch.tensor(geom.source, device=device, dtype=dtype)

        # Rays in world space, then mapped into volume space by the inverse pose.
        src_vol = pose.inverse_apply(src)
        dirs_world = det - src
        dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True)
        dirs_vol = dirs_world @ pose.matrix()

        # Intersect each ray with the volume's bounding sphere so the samples
        # land where there is something to integrate, then clip the segment to
        # the physical path between the source and the detector: anything
        # behind the detector is not in the imaged beam. Rays that miss, or
        # that only graze outside that path, get a zero-length span and
        # contribute nothing.
        t_centre = -(src_vol * dirs_vol).sum(dim=-1)
        closest_sq = (src_vol * src_vol).sum() - t_centre ** 2
        half_chord = torch.sqrt(
            torch.clamp(self.bounding_radius ** 2 - closest_sq, min=0.0)
        )
        ray_length = torch.linalg.norm(det - src, dim=-1)
        t_lo = torch.clamp(t_centre - half_chord, min=0.0)
        t_hi = torch.minimum(t_centre + half_chord, ray_length)
        span = torch.clamp(t_hi - t_lo, min=0.0)

        steps = torch.linspace(0.0, 1.0, self.n_samples, device=device, dtype=dtype)
        t = t_lo[:, None] + steps[None, :] * span[:, None]
        points = src_vol[None, None, :] + t[..., None] * dirs_vol[:, None, :]

        grid = self._to_grid(points).reshape(1, 1, -1, self.n_samples, 3)
        sampled = F.grid_sample(
            self._volume, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).reshape(-1, self.n_samples)

        dt = span / (self.n_samples - 1)
        integral = sampled.sum(dim=-1) * dt
        return integral.reshape(geom.height, geom.width)


def degrade(
    drr: torch.Tensor,
    photons: float = 2.0e4,
    blur_sigma_px: float = 1.0,
    scatter_fraction: float = 0.15,
    scatter_sigma_px: float = 24.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add a coarse stand-in for real projection physics to a clean DRR.

    Applies, in order: Beer-Lambert conversion to transmitted intensity,
    a wide-kernel scatter floor, detector blur, and Poisson quantum noise,
    then converts back to a line-integral image. This is not a scatter
    simulation; it is enough to stop the registration being handed back
    exactly the image its own forward model produces.
    """
    intensity = torch.exp(-drr)
    if scatter_fraction > 0:
        scatter = _gaussian_blur(intensity, scatter_sigma_px)
        intensity = (1 - scatter_fraction) * intensity + scatter_fraction * scatter
    if blur_sigma_px > 0:
        intensity = _gaussian_blur(intensity, blur_sigma_px)
    if photons and photons > 0:
        counts = torch.poisson(intensity * photons, generator=generator)
        intensity = torch.clamp(counts / photons, min=1.0 / photons)
    return -torch.log(intensity)


def _gaussian_blur(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding, for 2-D (H, W) input.

    The kernel radius is capped at one less than the image dimension, because
    reflect padding cannot exceed it. The scatter kernel is deliberately wide
    and hits that cap on small images.
    """
    height, width = image.shape
    img = image[None, None]
    for axis_len, pad_spec, kernel_shape in (
        (width, lambda r: (r, r, 0, 0), lambda k: k.view(1, 1, 1, -1)),
        (height, lambda r: (0, 0, r, r), lambda k: k.view(1, 1, -1, 1)),
    ):
        radius = min(max(1, int(round(3 * sigma_px))), axis_len - 1)
        x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-0.5 * (x / sigma_px) ** 2)
        kernel = kernel / kernel.sum()
        img = F.pad(img, pad_spec(radius), mode="reflect")
        img = F.conv2d(img, kernel_shape(kernel))
    return img[0, 0]

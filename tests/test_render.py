"""The renderer is checked against path lengths that can be worked out by hand.

A DRR pipeline that is merely self-consistent will register beautifully to its
own output and be wrong about the world, so these tests compare against
analytic line integrals rather than against stored values.
"""

import math

import pytest
import torch

from nvreg import phantom
from nvreg.geometry import CArmGeometry, Pose
from nvreg.render import Renderer, degrade, hu_to_mu


def _centre(image):
    h, w = image.shape
    return image[h // 2, w // 2].item()


def test_sphere_central_ray_equals_mu_times_diameter():
    radius, mu = 40.0, 0.02
    geom = CArmGeometry(height=65, width=65, pixel_mm=1.0)
    r = Renderer(phantom.sphere(radius_mm=radius, mu=mu), (1.0, 1.0, 1.0), geom,
                 n_samples=1024)
    drr = r.render(Pose.identity())
    assert _centre(drr) == pytest.approx(mu * 2 * radius, rel=1e-3)


def test_sphere_offaxis_ray_equals_chord_length():
    """A ray missing the centre should integrate the shorter chord."""
    radius, mu = 40.0, 0.02
    geom = CArmGeometry(height=65, width=65, pixel_mm=1.0)
    r = Renderer(phantom.sphere(radius_mm=radius, mu=mu), (1.0, 1.0, 1.0), geom,
                 n_samples=1024)
    drr = r.render(Pose.identity())
    col_offset = 16
    # Perpendicular distance from the sphere centre to this ray, at the isocenter.
    u_mm = col_offset * geom.pixel_mm
    offset_at_iso = u_mm / geom.magnification
    theta = math.atan2(u_mm, geom.sdd)
    chord = 2 * math.sqrt(radius ** 2 - offset_at_iso ** 2)
    expected = mu * chord / math.cos(theta)
    got = drr[geom.height // 2, geom.width // 2 + col_offset].item()
    assert got == pytest.approx(expected, rel=2e-2)


def test_slab_central_ray_equals_mu_times_thickness():
    thickness, mu = 30.0, 0.02
    geom = CArmGeometry(height=33, width=33, pixel_mm=1.0)
    r = Renderer(phantom.slab(thickness_mm=thickness, mu=mu), (1.0, 1.0, 1.0), geom,
                 n_samples=1024)
    assert _centre(r.render(Pose.identity())) == pytest.approx(mu * thickness, rel=1e-3)


def test_slab_oblique_ray_is_longer_by_one_over_cos():
    thickness, mu = 30.0, 0.02
    geom = CArmGeometry(height=81, width=81, pixel_mm=1.0)
    r = Renderer(phantom.slab(thickness_mm=thickness, mu=mu), (1.0, 1.0, 1.0), geom,
                 n_samples=1024)
    drr = r.render(Pose.identity())
    col_offset = 30
    theta = math.atan2(col_offset * geom.pixel_mm, geom.sdd)
    expected = mu * thickness / math.cos(theta)
    got = drr[geom.height // 2, geom.width // 2 + col_offset].item()
    assert got == pytest.approx(expected, rel=1e-2)


def test_translating_an_empty_volume_out_of_the_beam_gives_zero():
    geom = CArmGeometry(height=33, width=33, pixel_mm=1.0)
    r = Renderer(phantom.sphere(radius_mm=10.0, mu=0.02), (1.0, 1.0, 1.0), geom)
    far = Pose(torch.zeros(3), torch.tensor([0.0, 0.0, 5000.0]))
    assert r.render(far).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_translation_along_x_shifts_the_projection():
    geom = CArmGeometry(height=65, width=65, pixel_mm=1.0)
    r = Renderer(phantom.sphere(radius_mm=15.0, mu=0.02), (1.0, 1.0, 1.0), geom)
    base = r.render(Pose.identity())
    shifted = r.render(Pose(torch.zeros(3), torch.tensor([10.0, 0.0, 0.0])))
    # Centroid of the shifted image should move by 10 mm * magnification.
    def centroid_col(img):
        w = img / img.sum()
        cols = torch.arange(img.shape[1], dtype=img.dtype)
        return float((w.sum(dim=0) * cols).sum())
    moved = centroid_col(shifted) - centroid_col(base)
    assert moved == pytest.approx(10.0 * geom.magnification, rel=0.1)


def test_render_is_differentiable_in_the_pose():
    geom = CArmGeometry(height=33, width=33, pixel_mm=2.0)
    r = Renderer(phantom.sphere(radius_mm=20.0, mu=0.02), (1.0, 1.0, 1.0), geom)
    pose = Pose(torch.zeros(3, requires_grad=True),
                torch.tensor([2.0, 0.0, 0.0], requires_grad=True))
    r.render(pose).sum().backward()
    assert torch.isfinite(pose.rotvec.grad).all()
    assert torch.isfinite(pose.translation.grad).all()
    assert pose.translation.grad.abs().sum() > 0


def test_hu_to_mu_anchors_water_and_clamps_air():
    assert hu_to_mu(torch.tensor(0.0)).item() == pytest.approx(0.019)
    assert hu_to_mu(torch.tensor(-1000.0)).item() == pytest.approx(0.0)
    assert hu_to_mu(torch.tensor(-3000.0)).item() == 0.0  # clamped, not negative


def test_renderer_rejects_non_3d_volume():
    with pytest.raises(ValueError, match="must be 3-D"):
        Renderer(torch.zeros(4, 4), (1.0, 1.0, 1.0))


def test_degrade_changes_the_image_but_keeps_its_structure():
    geom = CArmGeometry(height=64, width=64, pixel_mm=2.0)
    mu, _ = phantom.vessel_tree(shape_xyz=(64, 64, 64))
    clean = Renderer(mu, (1.0, 1.0, 1.0), geom).render(Pose.identity())
    torch.manual_seed(0)
    noisy = degrade(clean)
    assert not torch.allclose(clean, noisy)
    a = (clean - clean.mean()).flatten()
    b = (noisy - noisy.mean()).flatten()
    correlation = float((a @ b) / (a.norm() * b.norm()))
    assert correlation > 0.8

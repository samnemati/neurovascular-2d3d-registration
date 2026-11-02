import math

import pytest
import torch

from nvreg.geometry import CArmGeometry, Pose, rotation_matrix


def test_rotation_identity_at_zero():
    assert torch.allclose(rotation_matrix(torch.zeros(3)), torch.eye(3), atol=1e-7)


def test_rotation_is_orthonormal_and_right_handed():
    for rv in ([0.3, -0.2, 0.7], [math.pi / 2, 0, 0], [0, 0, math.pi]):
        r = rotation_matrix(torch.tensor(rv))
        assert torch.allclose(r.T @ r, torch.eye(3), atol=1e-5)
        assert torch.det(r).item() == pytest.approx(1.0, abs=1e-5)


def test_rotation_90_degrees_about_z_maps_x_to_y():
    r = rotation_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    got = r @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(got, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_rotation_gradient_is_finite_at_zero():
    rv = torch.zeros(3, requires_grad=True)
    rotation_matrix(rv).sum().backward()
    assert torch.isfinite(rv.grad).all()


def test_isocenter_projects_to_detector_centre():
    geom = CArmGeometry()
    uv = geom.project(torch.zeros(3))
    assert torch.allclose(uv, torch.zeros(2), atol=1e-6)


def test_offaxis_point_is_magnified_by_sdd_over_sad():
    geom = CArmGeometry(sdd=1200.0, sad=750.0)
    uv = geom.project(torch.tensor([10.0, 0.0, 0.0]))
    assert uv[0].item() == pytest.approx(10.0 * geom.magnification, rel=1e-6)


def test_detector_grid_is_centred_and_sized_in_mm():
    geom = CArmGeometry(height=5, width=7, pixel_mm=2.0)
    pts = geom.detector_points()
    assert pts.shape == (5, 7, 3)
    assert pts[2, 3, 0].item() == pytest.approx(0.0)
    assert pts[0, 0, 0].item() == pytest.approx(-6.0)
    assert pts[0, 0, 1].item() == pytest.approx(-4.0)
    assert torch.allclose(pts[..., 2], torch.tensor(geom.detector_z))


def test_rescale_keeps_physical_detector_size():
    geom = CArmGeometry(height=192, width=192, pixel_mm=1.0)
    coarse = geom.rescale(0.25)
    assert coarse.height == 48 and coarse.width == 48
    assert coarse.height * coarse.pixel_mm == pytest.approx(geom.height * geom.pixel_mm)


def test_pose_round_trip():
    pose = Pose(torch.tensor([0.1, -0.2, 0.3]), torch.tensor([4.0, -5.0, 6.0]))
    pts = torch.randn(17, 3) * 20
    assert torch.allclose(pose.inverse_apply(pose.apply(pts)), pts, atol=1e-4)


def test_pose_vector_round_trip():
    vec = torch.tensor([0.1, -0.2, 0.3, 4.0, -5.0, 6.0])
    assert torch.allclose(Pose.from_vector(vec).to_vector(), vec)


def test_biplane_views_are_ninety_degrees_apart():
    from nvreg.geometry import biplane
    frontal, lateral = biplane()
    assert frontal.name == "frontal" and lateral.name == "lateral"
    assert frontal.rotation == (0.0, 0.0, 0.0)
    assert lateral.rotation[1] == pytest.approx(math.pi / 2)


def test_frontal_view_leaves_the_pose_untouched():
    from nvreg.geometry import biplane
    frontal, _ = biplane()
    pose = Pose(torch.tensor([0.1, 0.0, 0.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert frontal.effective_pose(pose) is pose


def test_lateral_view_rotates_the_object_pose():
    from nvreg.geometry import biplane
    _, lateral = biplane()
    pose = Pose(torch.tensor([0.1, -0.2, 0.3]), torch.tensor([1.0, 2.0, 3.0]))
    pts = torch.randn(13, 3) * 20
    expected = pose.apply(pts) @ lateral.rotation_matrix().T
    assert torch.allclose(lateral.effective_pose(pose).apply(pts), expected, atol=1e-4)


def test_view_effective_pose_is_differentiable():
    from nvreg.geometry import biplane
    _, lateral = biplane()
    pose = Pose(torch.zeros(3, requires_grad=True), torch.zeros(3, requires_grad=True))
    lateral.effective_pose(pose).apply(torch.randn(5, 3)).sum().backward()
    assert torch.isfinite(pose.rotvec.grad).all()
    assert torch.isfinite(pose.translation.grad).all()


def test_rigid_transform_round_trip_and_composition():
    from nvreg.geometry import RigidTransform
    a = Pose(torch.tensor([0.1, -0.2, 0.3]), torch.tensor([1.0, 2.0, 3.0])).as_transform()
    b = Pose(torch.tensor([0.05, 0.4, -0.1]), torch.tensor([-2.0, 1.0, 0.5])).as_transform()
    pts = torch.randn(11, 3) * 25
    assert torch.allclose(a.inverse_apply(a.apply(pts)), pts, atol=1e-4)
    assert torch.allclose(a.compose(b).apply(pts), a.apply(b.apply(pts)), atol=1e-4)
    assert isinstance(a.compose(b), RigidTransform)

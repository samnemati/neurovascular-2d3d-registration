import math

import pytest
import torch

from nvreg.geometry import CArmGeometry, Pose
from nvreg.metrics import gradient_ncc, mtre, ncc, pose_error, projection_error


def test_ncc_of_identical_images_is_one():
    img = torch.randn(16, 16)
    assert ncc(img, img).item() == pytest.approx(1.0, abs=1e-5)


def test_ncc_of_negated_image_is_minus_one():
    img = torch.randn(16, 16)
    assert ncc(img, -img).item() == pytest.approx(-1.0, abs=1e-5)


def test_ncc_is_invariant_to_brightness_and_contrast():
    img = torch.randn(16, 16)
    assert ncc(img, 3.5 * img + 2.0).item() == pytest.approx(1.0, abs=1e-5)


def test_gradient_ncc_of_identical_images_is_one():
    img = torch.randn(16, 16)
    assert gradient_ncc(img, img).item() == pytest.approx(1.0, abs=1e-5)


def test_mtre_is_zero_for_equal_poses():
    pts = torch.randn(10, 3) * 30
    pose = Pose(torch.tensor([0.1, 0.0, 0.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert mtre(pts, pose, pose) == pytest.approx(0.0, abs=1e-5)


def test_mtre_equals_the_offset_for_a_pure_translation():
    pts = torch.randn(10, 3) * 30
    a = Pose.identity()
    b = Pose(torch.zeros(3), torch.tensor([3.0, 4.0, 0.0]))
    assert mtre(pts, a, b) == pytest.approx(5.0, abs=1e-4)


def test_pose_error_recovers_a_known_rotation_angle():
    a = Pose.identity()
    b = Pose(torch.tensor([0.0, 0.0, math.radians(7.0)]), torch.zeros(3))
    rot_deg, trans_mm = pose_error(a, b)
    assert rot_deg == pytest.approx(7.0, abs=1e-3)
    assert trans_mm == pytest.approx(0.0, abs=1e-6)


def test_projection_error_is_magnified_for_an_in_plane_shift():
    geom = CArmGeometry()
    pts = torch.zeros(1, 3)
    a = Pose.identity()
    b = Pose(torch.zeros(3), torch.tensor([2.0, 0.0, 0.0]))
    err = projection_error(pts, a, b, geom)
    assert err == pytest.approx(2.0 * geom.magnification, rel=1e-4)

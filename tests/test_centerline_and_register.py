import numpy as np
import pytest
import torch

from nvreg import phantom
from nvreg.centerline import centerline_points, project_centerline
from nvreg.geometry import CArmGeometry, Pose, biplane
from nvreg.metrics import error_components, mtre
from nvreg.register import perturb_pose, register
from nvreg.render import Renderer


def test_centerline_points_lie_on_the_tube_axis():
    spacing = (1.0, 1.0, 1.0)
    radius = 3.0
    _, label = phantom.vessel_tree(shape_xyz=(80, 80, 80), spacing=spacing,
                                   radius_mm=radius)
    pts = centerline_points(label.numpy(), spacing)
    assert len(pts) > 10
    # The phantom's axis is x = 0.002 z^2 - 20, y = 0.
    axis_x = 0.002 * pts[:, 2] ** 2 - 20.0
    offset = np.sqrt((pts[:, 0] - axis_x) ** 2 + pts[:, 1] ** 2)
    assert offset.max() <= radius


def test_centerline_of_an_empty_mask_is_empty():
    pts = centerline_points(np.zeros((8, 8, 8), dtype=bool), (1.0, 1.0, 1.0))
    assert pts.shape == (0, 3)


def test_centerline_respects_max_points():
    _, label = phantom.vessel_tree(shape_xyz=(80, 80, 80))
    pts = centerline_points(label.numpy(), (1.0, 1.0, 1.0), max_points=5)
    assert pts.shape == (5, 3)


def test_projected_centerline_lands_inside_the_detector():
    geom = CArmGeometry(height=96, width=96, pixel_mm=2.0)
    _, label = phantom.vessel_tree(shape_xyz=(80, 80, 80))
    pts = torch.tensor(centerline_points(label.numpy(), (1.0, 1.0, 1.0)))
    cols_rows = project_centerline(pts, Pose.identity(), geom)
    assert cols_rows.shape[1] == 2
    assert cols_rows.min() >= 0 and cols_rows[:, 0].max() < geom.width


def test_perturb_pose_hits_the_requested_mtre():
    pts = torch.randn(50, 3) * 30
    pose = Pose.identity()
    rng = np.random.default_rng(1)
    for requested in (2.0, 10.0, 25.0):
        moved = perturb_pose(pose, pts, requested, rng)
        assert mtre(pts, pose, moved) == pytest.approx(requested, rel=0.1)


def _phantom_setup(detector=80, pixel_mm=2.0, n_samples=160):
    mu, label = phantom.vessel_tree(shape_xyz=(80, 80, 80))
    geom = CArmGeometry(height=detector, width=detector, pixel_mm=pixel_mm)
    renderer = Renderer(mu, (1.0, 1.0, 1.0), geom, n_samples=n_samples)
    pts = torch.tensor(centerline_points(label.numpy(), (1.0, 1.0, 1.0), max_points=120))
    return renderer, geom, pts


@pytest.mark.parametrize("method", ["adam", "powell"])
def test_biplane_registration_recovers_a_known_pose(method):
    """Two planes constrain all six parameters, so the full error should be small."""
    torch.manual_seed(0)
    renderer, geom, pts = _phantom_setup()
    views = biplane(geom)

    truth = Pose(torch.tensor([0.03, -0.02, 0.01]), torch.tensor([2.0, -1.5, 1.0]))
    targets = [renderer.render(truth, v.geometry, v) for v in views]
    start = perturb_pose(truth, pts, 6.0, np.random.default_rng(0))
    assert mtre(pts, truth, start) > 4.0

    result = register(renderer, targets, start, views=views, similarity="gncc",
                      method=method, schedule=((0.5, 60), (1.0, 40)))
    assert mtre(pts, truth, result.pose) < 1.5
    assert result.n_renders > 0 and result.seconds > 0


def test_single_plane_registration_recovers_the_in_plane_pose():
    """One projection pins down the two in-plane directions, not depth.

    Asserting a small total error here would be asserting something the
    geometry does not provide: moving the volume along the beam barely changes
    the image, so depth stays loose. The in-plane error is the part a single
    view is entitled to get right.
    """
    torch.manual_seed(0)
    renderer, geom, pts = _phantom_setup()
    truth = Pose(torch.tensor([0.03, -0.02, 0.01]), torch.tensor([2.0, -1.5, 1.0]))
    target = renderer.render(truth)
    start = perturb_pose(truth, pts, 6.0, np.random.default_rng(0))

    result = register(renderer, target, start, similarity="gncc", method="adam",
                      schedule=((0.5, 60), (1.0, 40)))
    inplane, depth = error_components(pts, truth, result.pose)
    assert inplane < 1.5
    assert depth >= 0.0  # recorded, deliberately not bounded


def test_register_rejects_unknown_similarity_and_method():
    renderer = Renderer(phantom.sphere(radius_mm=10.0), (1.0, 1.0, 1.0),
                        CArmGeometry(height=16, width=16, pixel_mm=4.0))
    target = renderer.render(Pose.identity())
    with pytest.raises(ValueError, match="unknown similarity"):
        register(renderer, target, Pose.identity(), similarity="nope")
    with pytest.raises(ValueError, match="unknown method"):
        register(renderer, target, Pose.identity(), method="nope",
                 schedule=((1.0, 1),))


def test_register_rejects_mismatched_views_and_targets():
    renderer, geom, _ = _phantom_setup(detector=16, pixel_mm=10.0, n_samples=32)
    target = renderer.render(Pose.identity())
    with pytest.raises(ValueError, match="2 views but 1 targets"):
        register(renderer, [target], Pose.identity(), views=biplane(geom),
                 schedule=((1.0, 1),))

"""Multi-resolution 2D/3D rigid registration of a volume to a projection image.

Two optimizers are provided over the same loss, because the interesting
question is not whether registration works at all but how far wrong the
initial pose can be before it stops working, and the two behave differently:

``adam``
    Gradient descent through the differentiable renderer. Cheap per step and
    precise near the solution.
``powell``
    Derivative-free coordinate search (SciPy). Each step costs several
    renders, but it does not rely on the local gradient pointing anywhere
    useful, which matters when the initial pose is far out.

Both run on a coarse-to-fine pyramid. Coarse levels smooth the similarity
landscape, which is what makes a wrong-but-nearby pose recoverable at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from .geometry import Pose, View
from .metrics import SIMILARITY
from .render import Renderer

ROT_SCALE = 0.1   # radians per unit of the scaled rotation parameter (5.7 deg)
TRANS_SCALE = 10.0  # mm per unit of the scaled translation parameter

DEFAULT_SCHEDULE = ((0.25, 120), (0.5, 80), (1.0, 40))
"""(detector scale, iterations) per pyramid level, coarse to fine."""


@dataclass
class RegistrationResult:
    pose: Pose
    loss: float
    n_renders: int
    seconds: float
    loss_history: list[float] = field(default_factory=list)


def _to_scaled(pose: Pose) -> np.ndarray:
    v = pose.to_vector().detach().cpu().numpy()
    return np.concatenate((v[:3] / ROT_SCALE, v[3:] / TRANS_SCALE))


def _from_scaled(q, device=None, dtype=torch.float32, requires_grad=False) -> Pose:
    q = torch.as_tensor(np.asarray(q), device=device, dtype=dtype)
    rot = (q[:3] * ROT_SCALE).clone().requires_grad_(requires_grad)
    trans = (q[3:] * TRANS_SCALE).clone().requires_grad_(requires_grad)
    return Pose(rot, trans)


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if image.shape == (height, width):
        return image
    return F.interpolate(
        image[None, None], size=(height, width), mode="area"
    )[0, 0]


def register(
    renderer: Renderer,
    targets,
    pose_init: Pose,
    views=None,
    similarity: str = "gncc",
    schedule=DEFAULT_SCHEDULE,
    method: str = "adam",
    lr: float = 0.05,
) -> RegistrationResult:
    """Recover the pose of ``renderer``'s volume that reproduces ``targets``.

    ``targets`` is one projection image, or one per view, at the native
    detector resolution. ``views`` defaults to a single frontal plane built
    from the renderer's geometry; pass ``biplane()`` for two planes, in which
    case the loss is averaged over them. The returned pose is in the same
    convention as ``pose_init``.
    """
    if similarity not in SIMILARITY:
        raise ValueError(f"unknown similarity {similarity!r}; have {sorted(SIMILARITY)}")
    if method not in ("adam", "powell"):
        raise ValueError(f"unknown method {method!r}; use 'adam' or 'powell'")
    sim_fn = SIMILARITY[similarity]
    device, dtype = renderer.mu.device, renderer.mu.dtype

    if views is None:
        views = (View(renderer.geometry, (0.0, 0.0, 0.0), "frontal"),)
    if isinstance(targets, torch.Tensor):
        targets = (targets,)
    views, targets = tuple(views), tuple(targets)
    if len(views) != len(targets):
        raise ValueError(f"got {len(views)} views but {len(targets)} targets")

    q = _to_scaled(pose_init)
    history: list[float] = []
    renders = 0
    start = time.perf_counter()

    def total_loss(pose, level_views, fixed_images):
        return sum(
            1.0 - sim_fn(renderer.render(pose, view.geometry, view), fixed)
            for view, fixed in zip(level_views, fixed_images)
        ) / len(level_views)

    for scale, iterations in schedule:
        level_views = [v.rescale(scale) if scale != 1.0 else v for v in views]
        fixed_images = [
            _resize(t, v.geometry.height, v.geometry.width)
            for v, t in zip(level_views, targets)
        ]

        if method == "adam":
            pose = _from_scaled(q, device, dtype, requires_grad=True)
            # Rotation and translation are stepped in the same scaled units, so
            # each gets the learning rate that corresponds to that unit.
            opt = torch.optim.Adam([
                {"params": [pose.rotvec], "lr": lr * ROT_SCALE},
                {"params": [pose.translation], "lr": lr * TRANS_SCALE},
            ])
            for _ in range(iterations):
                opt.zero_grad(set_to_none=True)
                loss = total_loss(pose, level_views, fixed_images)
                loss.backward()
                opt.step()
                renders += len(level_views)
                history.append(float(loss.detach()))
            q = _to_scaled(Pose(pose.rotvec.detach(), pose.translation.detach()))

        else:
            def objective(qv):
                nonlocal renders
                with torch.no_grad():
                    loss = total_loss(_from_scaled(qv, device, dtype),
                                      level_views, fixed_images)
                renders += len(level_views)
                history.append(float(loss))
                return float(loss)

            res = minimize(
                objective, q, method="Powell",
                options={"maxfev": iterations * 6, "xtol": 1e-3, "ftol": 1e-4},
            )
            q = np.asarray(res.x, dtype=float)

    final_pose = _from_scaled(q, device, dtype)
    with torch.no_grad():
        final_loss = float(total_loss(final_pose, views, targets))
    return RegistrationResult(
        pose=final_pose,
        loss=final_loss,
        n_renders=renders,
        seconds=time.perf_counter() - start,
        loss_history=history,
    )


def perturb_pose(
    pose: Pose,
    target_points: torch.Tensor,
    mtre_mm: float,
    rng: np.random.Generator,
    rot_fraction: float = 0.5,
    tol: float = 0.05,
    max_tries: int = 60,
) -> Pose:
    """Perturb a pose so that its mean target registration error is ``mtre_mm``.

    Offsets are drawn in a random direction, with roughly ``rot_fraction`` of
    the displacement contributed by rotation, then scaled by bisection until
    the resulting mTRE hits the requested value. Sampling by displacement
    rather than by parameter value is what makes a capture-range curve
    comparable across cases of different size.
    """
    from .metrics import mtre as _mtre

    device, dtype = pose.rotvec.device, pose.rotvec.dtype
    rot_dir = rng.normal(size=3)
    rot_dir /= np.linalg.norm(rot_dir)
    trans_dir = rng.normal(size=3)
    trans_dir /= np.linalg.norm(trans_dir)

    def build(magnitude: float) -> Pose:
        rot = pose.rotvec + torch.as_tensor(
            rot_dir * magnitude * rot_fraction * 0.02, device=device, dtype=dtype)
        trans = pose.translation + torch.as_tensor(
            trans_dir * magnitude * (1 - rot_fraction), device=device, dtype=dtype)
        return Pose(rot, trans)

    lo, hi = 0.0, max(mtre_mm, 1.0) * 4
    for _ in range(max_tries):
        mid = 0.5 * (lo + hi)
        candidate = build(mid)
        achieved = _mtre(target_points, pose, candidate)
        if abs(achieved - mtre_mm) <= tol * mtre_mm:
            return candidate
        if achieved < mtre_mm:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))

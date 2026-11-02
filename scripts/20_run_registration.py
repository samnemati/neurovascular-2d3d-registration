#!/usr/bin/env python3
"""Measure 2D/3D registration accuracy against a known pose.

For each case a target projection is rendered at a known pose and degraded,
the pose is then thrown off by a controlled amount, and registration is asked
to find its way back. Because the target was rendered from the same volume,
the true pose is known exactly and the error can be quoted in millimetres
rather than as a similarity score.

The sweep crosses four initial-offset sizes with two optimizers and one or two
imaging planes. Protocol follows the standardized evaluation of 2D/3D
registration (van de Kraats et al., 2005): offsets are sampled by the
displacement they cause at the target, accuracy is mean target registration
error, and success is a 2 mm threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nvreg.geometry import CArmGeometry, Pose, View, biplane
from nvreg.metrics import error_components, mtre, pose_error, projection_error
from nvreg.register import perturb_pose, register
from nvreg.render import Renderer, degrade, hu_to_mu
from nvreg.runtime import pick_device, sync

SUCCESS_MM = 2.0
"""A registration counts as successful below this mean target error."""

ADAM_SCHEDULE = ((0.25, 120), (0.5, 80), (1.0, 40))
POWELL_SCHEDULE = ((0.25, 60), (0.5, 40), (1.0, 20))

CONFIGS = (
    ("adam", 1), ("adam", 2),
    ("powell", 1), ("powell", 2),
)


def case_true_pose(case_index: int, device, dtype=torch.float32) -> Pose:
    """A plausible, case-specific starting pose.

    Deliberately not the identity: zero is a special point for an optimizer
    initialized near it, and a pose of exactly zero would flatter the result.
    """
    rng = np.random.default_rng(1000 + case_index)
    rot = rng.uniform(-np.deg2rad(8), np.deg2rad(8), size=3)
    trans = rng.uniform(-10.0, 10.0, size=3)
    return Pose(torch.tensor(rot, device=device, dtype=dtype),
                torch.tensor(trans, device=device, dtype=dtype))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    ap.add_argument("--cases", type=int, default=10, help="number of cases to use")
    ap.add_argument("--offsets", type=float, nargs="+",
                    default=[2.5, 5.0, 10.0, 20.0, 30.0, 40.0],
                    help="initial mean target registration error, mm")
    ap.add_argument("--repeats", type=int, default=3, help="random restarts per cell")
    ap.add_argument("--detector", type=int, default=160, help="detector pixels per side")
    ap.add_argument("--detector-mm", type=float, default=300.0, help="detector side, mm")
    ap.add_argument("--samples", type=int, default=192, help="samples per ray")
    ap.add_argument("--similarity", default="gncc", choices=["gncc", "ncc"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/registrations.csv"))
    ap.add_argument("--append", action="store_true",
                    help="add to an existing results file instead of replacing it, "
                         "so a sweep can be extended to further offsets later")
    args = ap.parse_args()

    device = pick_device(args.device)
    files = sorted(args.prepared.glob("topcow_ct_*.npz"))[: args.cases]
    if not files:
        raise SystemExit(f"no prepared cases in {args.prepared}; run 10_prepare_cases.py")

    geometry = CArmGeometry(height=args.detector, width=args.detector,
                            pixel_mm=args.detector_mm / args.detector)
    views_two = biplane(geometry)
    views_one = (views_two[0],)
    print(f"device {device}, detector {args.detector}px at "
          f"{geometry.pixel_mm:.2f} mm, {args.samples} samples/ray")

    rows = []
    started = time.perf_counter()
    for case_index, path in enumerate(files):
        cached = np.load(path)
        mu = hu_to_mu(torch.tensor(cached["hu"])).to(device)
        spacing = tuple(float(s) for s in cached["spacing"])
        points = torch.tensor(cached["centerline"], device=device)
        renderer = Renderer(mu, spacing, geometry, n_samples=args.samples)
        case_id = path.stem

        truth = case_true_pose(case_index, device)
        # One target per plane, degraded so that registration is not handed
        # back exactly what its own forward model produces.
        torch.manual_seed(case_index)
        targets_two = tuple(
            degrade(renderer.render(truth, view.geometry, view))
            for view in views_two
        )
        targets_one = (targets_two[0],)

        for offset in args.offsets:
            for repeat in range(args.repeats):
                rng = np.random.default_rng(10_000 * case_index + 100 * int(offset * 10) + repeat)
                start = perturb_pose(truth, points, offset, rng)
                initial = mtre(points, truth, start)

                for method, n_planes in CONFIGS:
                    views = views_two if n_planes == 2 else views_one
                    targets = targets_two if n_planes == 2 else targets_one
                    schedule = ADAM_SCHEDULE if method == "adam" else POWELL_SCHEDULE
                    result = register(renderer, targets, start, views=views,
                                      similarity=args.similarity,
                                      schedule=schedule, method=method)
                    sync(device)
                    final = mtre(points, truth, result.pose)
                    rot_deg, trans_mm = pose_error(truth, result.pose)
                    inplane, depth = error_components(points, truth, result.pose)
                    rows.append({
                        "case_id": case_id,
                        "offset_mm": offset,
                        "repeat": repeat,
                        "method": method,
                        "planes": n_planes,
                        "initial_mtre_mm": initial,
                        "final_mtre_mm": final,
                        "inplane_rms_mm": inplane,
                        "depth_rms_mm": depth,
                        "rot_err_deg": rot_deg,
                        "trans_err_mm": trans_mm,
                        "overlay_err_mm": projection_error(
                            points, truth, result.pose, geometry),
                        "success": final < SUCCESS_MM,
                        "n_renders": result.n_renders,
                        "seconds": result.seconds,
                        "final_loss": result.loss,
                    })
            done = len(rows)
            print(f"  {case_id} offset {offset:5.1f} mm -> "
                  f"{done} registrations, {time.perf_counter() - started:.0f}s elapsed",
                  flush=True)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path("outputs/registration_config.json")
    if args.append and args.out.exists():
        previous = pd.read_csv(args.out)
        table = pd.concat([previous, table], ignore_index=True)
        table = table.drop_duplicates(
            subset=["case_id", "offset_mm", "repeat", "method", "planes"], keep="last")
    table = table.sort_values(["case_id", "offset_mm", "repeat", "planes", "method"])
    table.to_csv(args.out, index=False)

    config = {
        "device": str(device), "n_cases": len(files),
        "offsets_mm": args.offsets, "repeats": args.repeats,
        "detector_px": args.detector, "detector_mm": args.detector_mm,
        "pixel_mm": geometry.pixel_mm, "samples_per_ray": args.samples,
        "similarity": args.similarity, "success_threshold_mm": SUCCESS_MM,
        "adam_schedule": ADAM_SCHEDULE, "powell_schedule": POWELL_SCHEDULE,
        "sdd_mm": geometry.sdd, "sad_mm": geometry.sad,
        "total_seconds": time.perf_counter() - started,
    }
    if args.append and config_path.exists():
        # Keep the record of the whole sweep, not only of this invocation.
        earlier = json.loads(config_path.read_text())
        config["offsets_mm"] = sorted(set(earlier.get("offsets_mm", [])) | set(args.offsets))
        config["total_seconds"] = round(
            earlier.get("total_seconds", 0.0) + config["total_seconds"], 1)
        config["invocations"] = earlier.get("invocations", 1) + 1
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\n{len(table)} registrations in {config['total_seconds']:.0f}s -> {args.out}")
    print(table.groupby(["planes", "method"])["success"].mean().round(3).to_string())


if __name__ == "__main__":
    main()

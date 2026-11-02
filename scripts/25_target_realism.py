#!/usr/bin/env python3
"""How much of the accuracy survives when the target is not a clean DRR?

Registering a volume to a projection rendered by the same forward model is the
easy case, and reporting only that number would overstate what the method
does. Here the same registration runs against three targets: the clean DRR,
a degraded one at the level used in the main sweep, and a harder degradation
with a quarter of the photons and more scatter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nvreg.geometry import CArmGeometry, biplane
from nvreg.metrics import error_components, mtre, projection_error
from nvreg.register import perturb_pose, register
from nvreg.render import Renderer, degrade, hu_to_mu
from nvreg.runtime import pick_device, sync

TARGETS = {
    "clean":          None,
    "degraded":       dict(photons=2.0e4, blur_sigma_px=1.0, scatter_fraction=0.15),
    "degraded_hard":  dict(photons=5.0e3, blur_sigma_px=1.5, scatter_fraction=0.30),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--offset", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--detector", type=int, default=160)
    ap.add_argument("--detector-mm", type=float, default=300.0)
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/realism.csv"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    case_true_pose = import_module("20_run_registration").case_true_pose

    device = pick_device(args.device)
    files = sorted(args.prepared.glob("topcow_ct_*.npz"))[: args.cases]
    if not files:
        raise SystemExit(f"no prepared cases in {args.prepared}; run 10_prepare_cases.py")

    geometry = CArmGeometry(height=args.detector, width=args.detector,
                            pixel_mm=args.detector_mm / args.detector)
    views = biplane(geometry)
    rows = []

    for case_index, path in enumerate(files):
        cached = np.load(path)
        mu = hu_to_mu(torch.tensor(cached["hu"])).to(device)
        spacing = tuple(float(s) for s in cached["spacing"])
        points = torch.tensor(cached["centerline"], device=device)
        renderer = Renderer(mu, spacing, geometry, n_samples=args.samples)
        truth = case_true_pose(case_index, device)
        clean = [renderer.render(truth, v.geometry, v) for v in views]

        for name, settings in TARGETS.items():
            torch.manual_seed(case_index)
            targets = clean if settings is None else [degrade(c, **settings) for c in clean]
            for repeat in range(args.repeats):
                rng = np.random.default_rng(500 * case_index + repeat)
                start = perturb_pose(truth, points, args.offset, rng)
                result = register(renderer, targets, start, views=views,
                                  similarity="gncc", method="adam")
                sync(device)
                final = mtre(points, truth, result.pose)
                inplane, depth = error_components(points, truth, result.pose)
                rows.append({
                    "case_id": path.stem, "target": name, "repeat": repeat,
                    "initial_mtre_mm": mtre(points, truth, start),
                    "final_mtre_mm": final,
                    "inplane_rms_mm": inplane, "depth_rms_mm": depth,
                    "overlay_err_mm": projection_error(points, truth, result.pose, geometry),
                    "success": final < 2.0,
                })
        print(f"  {path.stem}: done ({len(rows)} rows)", flush=True)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\n{len(table)} registrations -> {args.out}")
    print(table.groupby("target")[["final_mtre_mm", "success"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

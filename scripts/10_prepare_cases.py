#!/usr/bin/env python3
"""Prepare the TopBrain CT angiography cases for projection.

Each case is reoriented, resampled to 1 mm isotropic voxels, cropped to a cube
around the circle of Willis, and cached as an .npz alongside its vessel
centreline. Caching matters because the experiment registers every case many
times and resampling it each time would dominate the runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from nvreg.centerline import centerline_points
from nvreg.volume import load_case

DATA_ROOT = Path("data/topbrain/extracted/TopBrain_Data_Release_Batches1n2nTA36_081726")
PREPARED = Path("data/prepared")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=PREPARED)
    ap.add_argument("--spacing", type=float, default=1.0, help="isotropic voxel size, mm")
    ap.add_argument("--fov", type=float, default=180.0, help="cropped cube side, mm")
    ap.add_argument("--max-centerline", type=int, default=300)
    args = ap.parse_args()

    images = sorted((args.root / "imagesTr_topbrain").glob("topcow_ct_*_0000.nii.gz"))
    if not images:
        raise SystemExit(f"no CT images under {args.root}; see README for the download")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in images:
        case_id = image_path.name.replace("_0000.nii.gz", "")
        label_path = args.root / "labelsTr_topbrain_v1_ct" / f"{case_id}.nii.gz"
        if not label_path.exists():
            print(f"  {case_id}: no label, skipped")
            continue

        volume = load_case(image_path, label_path, target_spacing=args.spacing,
                           fov_mm=args.fov, case_id=case_id)
        points = centerline_points(volume.label, volume.spacing,
                                   max_points=args.max_centerline,
                                   rng=np.random.default_rng(0))
        if len(points) < 20:
            print(f"  {case_id}: only {len(points)} centreline points, skipped")
            continue

        np.savez_compressed(
            args.out / f"{case_id}.npz",
            hu=volume.hu.astype(np.float32),
            label=volume.label,
            centerline=points.astype(np.float32),
            spacing=np.array(volume.spacing, dtype=np.float32),
        )
        rows.append({
            "case_id": case_id,
            "shape_kji": "x".join(str(n) for n in volume.hu.shape),
            "extent_mm": "x".join(f"{e:.0f}" for e in volume.extent_mm),
            "vessel_voxels": int(volume.label.sum()),
            "centerline_points": int(len(points)),
            "hu_p99": float(np.percentile(volume.hu, 99)),
        })
        print(f"  {case_id}: {volume.hu.shape}, {len(points)} centreline points")

    table = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    table.to_csv("outputs/cases.csv", index=False)
    (Path("outputs") / "prepare_config.json").write_text(json.dumps({
        "spacing_mm": args.spacing, "fov_mm": args.fov,
        "max_centerline_points": args.max_centerline,
        "n_cases": len(table),
    }, indent=2) + "\n")
    print(f"\nprepared {len(table)} cases -> {args.out}  (table: outputs/cases.csv)")


if __name__ == "__main__":
    main()

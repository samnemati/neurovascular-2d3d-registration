#!/usr/bin/env python3
"""Tables and figures from the registration sweep.

Produces the capture-range curve, the in-plane versus depth error split, an
overlay figure showing what the recovered pose looks like in the projection,
and the headline numbers in outputs/summary.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module

from nvreg.centerline import project_centerline
from nvreg.geometry import CArmGeometry, biplane
from nvreg.metrics import error_components, mtre, projection_error
from nvreg.register import perturb_pose, register
from nvreg.render import Renderer, degrade, hu_to_mu
from nvreg.runtime import pick_device

OUT = Path("outputs")
SUCCESS_MM = 2.0
LABELS = {(1, "adam"): "1 plane, gradient", (2, "adam"): "2 planes, gradient",
          (1, "powell"): "1 plane, Powell", (2, "powell"): "2 planes, Powell"}
COLORS = {(1, "adam"): "tab:red", (2, "adam"): "tab:blue",
          (1, "powell"): "tab:orange", (2, "powell"): "tab:green"}


def _show(ax, image: torch.Tensor) -> None:
    """Display a projection the way a radiograph is read: dense tissue bright.

    The renderer returns a line integral of attenuation, so bone is the high
    value. Percentile limits keep a few very dense voxels from flattening the
    rest of the image.
    """
    array = image.detach().cpu().numpy()
    lo, hi = np.percentile(array, [1, 99.5])
    ax.imshow(array, cmap="gray", vmin=lo, vmax=hi)


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(["planes", "method", "offset_mm"])
    table = grouped.agg(
        n=("final_mtre_mm", "size"),
        median_mtre_mm=("final_mtre_mm", "median"),
        mean_mtre_mm=("final_mtre_mm", "mean"),
        median_inplane_mm=("inplane_rms_mm", "median"),
        median_depth_mm=("depth_rms_mm", "median"),
        median_rot_deg=("rot_err_deg", "median"),
        median_overlay_mm=("overlay_err_mm", "median"),
        success_rate=("success", "mean"),
        mean_renders=("n_renders", "mean"),
        mean_seconds=("seconds", "mean"),
    ).reset_index()
    return table.round(4)


def figure_capture_range(runs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    offsets = sorted(runs.offset_mm.unique())
    for key, label in LABELS.items():
        planes, method = key
        sub = runs[(runs.planes == planes) & (runs.method == method)]
        if sub.empty:
            continue
        rate = [sub[sub.offset_mm == o].success.mean() * 100 for o in offsets]
        median = [sub[sub.offset_mm == o].final_mtre_mm.median() for o in offsets]
        axes[0].plot(offsets, rate, "o-", color=COLORS[key], label=label)
        axes[1].plot(offsets, median, "o-", color=COLORS[key], label=label)

    axes[0].set_ylabel(f"successful registrations (%)\nmTRE < {SUCCESS_MM:g} mm, 30 per point")
    axes[0].set_ylim(-5, 105)
    axes[1].set_ylabel("median final mTRE (mm)")
    axes[1].set_yscale("log")
    axes[1].axhline(SUCCESS_MM, color="gray", ls=":", lw=1)
    axes[1].text(offsets[0], SUCCESS_MM * 1.1, f"{SUCCESS_MM:g} mm", color="gray", fontsize=8)
    for ax in axes:
        ax.set_xlabel("initial mTRE (mm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Success rate and accuracy against how far the pose started out", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "f_capture_range.png", dpi=150)
    plt.close(fig)


def figure_error_split(runs: pd.DataFrame) -> None:
    """Where the residual error lives, in-plane versus along the beam."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    keys = [k for k in LABELS if not runs[(runs.planes == k[0]) & (runs.method == k[1])].empty]
    width = 0.36
    x = np.arange(len(keys))
    inplane = [runs[(runs.planes == p) & (runs.method == m)].inplane_rms_mm.median()
               for p, m in keys]
    depth = [runs[(runs.planes == p) & (runs.method == m)].depth_rms_mm.median()
             for p, m in keys]
    bars_in = ax.bar(x - width / 2, inplane, width, label="in-plane (x, y)", color="tab:blue")
    bars_depth = ax.bar(x + width / 2, depth, width, label="depth (along the beam)",
                        color="tab:red")
    for bars in (bars_in, bars_depth):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=9)
    ax.set_ylabel("median residual error (mm, RMS)")
    ax.set_ylim(0, max(inplane + depth) * 1.35)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.set_title("A single projection constrains the image plane, not depth", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "f_error_decomposition.png", dpi=150)
    plt.close(fig)


def figure_realism(realism: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = [t for t in ("clean", "degraded", "degraded_hard") if t in set(realism.target)]
    data = [realism[realism.target == t].final_mtre_mm.values for t in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    for i, values in enumerate(data, start=1):
        ax.plot(np.full(len(values), i) + np.random.uniform(-0.08, 0.08, len(values)),
                values, ".", color="tab:blue", alpha=0.5, ms=5)
    ax.axhline(SUCCESS_MM, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("final mTRE (mm)")
    ax.set_title("Cost of a target that the forward model did not produce\n"
                 "(two planes, gradient optimizer, 10 mm initial offset)", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "f_target_realism.png", dpi=150)
    plt.close(fig)


def figure_overlay(prepared: Path) -> dict:
    """Project the vessel centreline under the true and the recovered pose."""
    case_true_pose = import_module("20_run_registration").case_true_pose
    device = pick_device()
    path = sorted(prepared.glob("topcow_ct_*.npz"))[0]
    cached = np.load(path)
    mu = hu_to_mu(torch.tensor(cached["hu"])).to(device)
    spacing = tuple(float(s) for s in cached["spacing"])
    points = torch.tensor(cached["centerline"], device=device)

    geometry = CArmGeometry(height=160, width=160, pixel_mm=300.0 / 160)
    views = biplane(geometry)
    renderer = Renderer(mu, spacing, geometry, n_samples=192)
    truth = case_true_pose(0, device)
    torch.manual_seed(0)
    targets = [degrade(renderer.render(truth, v.geometry, v)) for v in views]
    start = perturb_pose(truth, points, 10.0, np.random.default_rng(0))

    single = register(renderer, targets[:1], start, views=views[:1],
                      similarity="gncc", method="adam")
    both = register(renderer, targets, start, views=views,
                    similarity="gncc", method="adam")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    panels = [
        (axes[0], views[0], targets[0], single.pose, "registered on this plane only"),
        (axes[1], views[1], targets[1], single.pose, "the plane it never saw"),
        (axes[2], views[1], targets[1], both.pose, "registered on both planes"),
    ]
    for ax, view, target, estimate, caption in panels:
        _show(ax, target)
        truth_px = project_centerline(points, view.effective_pose(truth), geometry)
        est_px = project_centerline(points, view.effective_pose(estimate), geometry)
        ax.plot(truth_px[:, 0].cpu(), truth_px[:, 1].cpu(), ".",
                ms=2.5, color="tab:green", label="true centreline")
        ax.plot(est_px[:, 0].cpu(), est_px[:, 1].cpu(), ".",
                ms=2.5, color="tab:red", label="recovered")
        err = projection_error(points, view.effective_pose(truth),
                               view.effective_pose(estimate), geometry)
        ax.set_title(f"{view.name}: {caption}\noverlay error {err:.2f} mm", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="lower right", markerscale=3)
    fig.suptitle(f"Vessel centreline projected into the live view, {path.stem}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "f_projection_overlay.png", dpi=150)
    plt.close(fig)

    s_in, s_depth = error_components(points, truth, single.pose)
    b_in, b_depth = error_components(points, truth, both.pose)
    return {
        "case": path.stem,
        "initial_mtre_mm": round(mtre(points, truth, start), 3),
        "single_plane": {"mtre_mm": round(mtre(points, truth, single.pose), 3),
                         "inplane_mm": round(s_in, 3), "depth_mm": round(s_depth, 3)},
        "two_planes": {"mtre_mm": round(mtre(points, truth, both.pose), 3),
                       "inplane_mm": round(b_in, 3), "depth_mm": round(b_depth, 3)},
    }


def figure_example_projections(prepared: Path) -> None:
    """One clean DRR and its degraded counterpart, so the input is visible."""
    device = pick_device()
    path = sorted(prepared.glob("topcow_ct_*.npz"))[0]
    cached = np.load(path)
    mu = hu_to_mu(torch.tensor(cached["hu"])).to(device)
    spacing = tuple(float(s) for s in cached["spacing"])
    geometry = CArmGeometry(height=320, width=320, pixel_mm=300.0 / 320)
    views = biplane(geometry)
    renderer = Renderer(mu, spacing, geometry, n_samples=320)
    from nvreg.geometry import Pose
    pose = Pose.identity().to(device)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    torch.manual_seed(0)
    for i, view in enumerate(views):
        clean = renderer.render(pose, view.geometry, view)
        rough = degrade(clean)
        for j, (img, name) in enumerate(((clean, "clean DRR"), (rough, "degraded"))):
            ax = axes[2 * i + j]
            _show(ax, img)
            ax.set_title(f"{view.name}, {name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Simulated projections of {path.stem} "
                 f"(1200 mm source-to-detector, 750 mm to isocenter, 300 mm detector)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "f_example_projections.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = pd.read_csv(OUT / "registrations.csv")
    table = summary_table(runs)
    table.to_csv(OUT / "t_registration_summary.csv", index=False)

    figure_capture_range(runs)
    figure_error_split(runs)
    overlay = figure_overlay(Path("data/prepared"))
    figure_example_projections(Path("data/prepared"))

    summary = {
        "n_registrations": int(len(runs)),
        "n_cases": int(runs.case_id.nunique()),
        "success_threshold_mm": SUCCESS_MM,
        "by_configuration": {},
        "overlay_example": overlay,
    }
    for (planes, method), sub in runs.groupby(["planes", "method"]):
        summary["by_configuration"][f"{planes}plane_{method}"] = {
            "success_rate": round(float(sub.success.mean()), 3),
            "median_mtre_mm": round(float(sub.final_mtre_mm.median()), 3),
            "median_inplane_mm": round(float(sub.inplane_rms_mm.median()), 3),
            "median_depth_mm": round(float(sub.depth_rms_mm.median()), 3),
            "median_overlay_mm": round(float(sub.overlay_err_mm.median()), 3),
            "median_rot_err_deg": round(float(sub.rot_err_deg.median()), 3),
            "mean_renders": int(sub.n_renders.mean()),
            "mean_seconds": round(float(sub.seconds.mean()), 2),
        }
    success_by_offset = {}
    for (planes, method), sub in runs.groupby(["planes", "method"]):
        success_by_offset[f"{planes}plane_{method}"] = {
            f"{o:g}mm": round(float(s.success.mean()), 3)
            for o, s in sub.groupby("offset_mm")
        }
    summary["success_by_initial_offset"] = success_by_offset

    realism_path = OUT / "realism.csv"
    if realism_path.exists():
        realism = pd.read_csv(realism_path)
        figure_realism(realism)
        summary["target_realism"] = {
            name: {"median_mtre_mm": round(float(sub.final_mtre_mm.median()), 3),
                   "success_rate": round(float(sub.success.mean()), 3),
                   "median_overlay_mm": round(float(sub.overlay_err_mm.median()), 3)}
            for name, sub in realism.groupby("target")
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table.to_string(index=False))
    print(f"\nwrote {OUT}/summary.json and figures")


if __name__ == "__main__":
    main()

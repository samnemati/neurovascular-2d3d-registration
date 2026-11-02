# Provenance

## Source data

**TopBrain Challenge Data Release** (batches 1 and 2, plus the 36-class aneurysm label set),
Zenodo record 21972006, released 17 August 2026. DOI:
[10.5281/zenodo.21972006](https://doi.org/10.5281/zenodo.21972006).
Archive `TopBrain_Data_Release_Batches1n2nTA36_081726.zip`, 1.8 GB,
MD5 `aab94bed377ad71bfbde3f7e26d652c5` (verified against the Zenodo record on download).

The images derive from the TopCoW challenge release: 25 head CT angiography and 25 magnetic
resonance angiography scans, with voxel-level labels of the vessels of the circle of Willis.

**Data owner:** University Hospital of Zurich, Department of Neurology.

**Terms of use** (from the archive's `License.txt`): open use, source must be provided,
non-commercial use permitted, commercial use requires prior permission from the data owner. The
data is therefore **not redistributed in this repository**; `data/` is gitignored and the README
gives the download command. Nothing derived from it that could reconstruct an image is committed
either: the only committed outputs are error tables, summary statistics, and rendered figures of
one case.

## What this repository uses

Only the **25 CT angiography** cases and the `labelsTr_topbrain_v1_ct` vessel labels. The MRA
cases are not used: this is a study of X-ray projection geometry, and the forward model converts
Hounsfield units to attenuation, which has no meaning for MR intensity.

Each case is prepared by `scripts/10_prepare_cases.py`:

1. Reoriented to closest-canonical (RAS) axis order and transposed to `[z, y, x]`.
2. Resampled to 1.0 mm isotropic voxels (linear for the image, nearest for the labels).
3. Cropped to a 180 mm cube centred on the centroid of the vessel label, which puts the circle of
   Willis near the isocenter. Where the scan does not cover the full cube the window is shifted
   inward rather than padded, so no fabricated air enters the volume. Prepared extents therefore
   vary between about 130 and 180 mm per axis; `outputs/cases.csv` records each one.
4. The vessel label is skeletonized and up to 300 centreline points are sampled with a fixed seed.
   These points are the targets for every error measure in the study.

The main sweep (`scripts/20_run_registration.py`) uses the **first 10 cases by case ID**
(`topcow_ct_001` through `topcow_ct_010`), not a random sample, so the selection is reproducible
and was fixed before any result was looked at.

## Ground truth

No annotation supplies the ground-truth pose. Each target projection is rendered from the same
volume at a pose drawn from a fixed per-case seed (`case_true_pose`), so the pose that produced it
is known exactly and the error can be measured in millimetres. The consequence, stated in the
README's limitations, is that this measures registration against a simulated projection and not
against a real fluoroscopic image.

## Derived files (committed)

| File | What it holds |
| --- | --- |
| `outputs/cases.csv` | per-case prepared shape, extent, vessel voxel count, centreline point count |
| `outputs/prepare_config.json` | spacing, field of view, centreline cap used for preparation |
| `outputs/registrations.csv` | one row per registration in the main sweep |
| `outputs/registration_config.json` | geometry, schedules, seeds, and runtime of the sweep |
| `outputs/realism.csv` | one row per registration in the clean/degraded/hard comparison |
| `outputs/t_registration_summary.csv` | the sweep aggregated by planes, optimizer, and initial offset |
| `outputs/summary.json` | the headline numbers quoted in the README |
| `outputs/f_*.png` | the figures in the README |

## Software

Pinned in `requirements.txt`. Figures and tables in this repository were produced on macOS with
PyTorch on the Apple MPS backend. The renderer agrees between MPS and CPU to within float32
rounding, about 1e-5 relative on a real case; CPU is what continuous integration runs. Note that
the Poisson noise in `degrade` is drawn from the device's own generator, so the degraded targets
are reproducible on a given device but not identical across devices. The tests use analytic phantoms only and need none of the data.

## References

TopBrain and TopCoW challenge data (University Hospital of Zurich, Department of Neurology).
TopBrain Challenge Data Release [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21972006

van de Kraats, E. B., Penney, G. P., Tomazevic, D., van Walsum, T., & Niessen, W. J. (2005).
Standardized evaluation methodology for 2-D-3-D registration. *IEEE Transactions on Medical
Imaging, 24*(9), 1177-1189. https://doi.org/10.1109/TMI.2005.853240

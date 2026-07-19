#!/usr/bin/env python3
"""Create the MICCAI-paper representative subject output figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
# Median-volume ARC case used in the pre-existing small/medium/large figure.
# Its inpainting passed visual QC and is preferable to the former large-lesion
# default, whose generated anatomy was visibly implausible.
DEFAULT_CASE = "sub-M2278_ses-388_acq-tfl3p2_run-4_T1w"


def first_directory(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_dir():
            return path
    return paths[0]


def parse_args() -> argparse.Namespace:
    arc_root = Path(os.environ.get("ARC_ROOT", "/project2/ajoshi_1183/data/ARC"))
    inpainting_root = first_directory(
        [
            arc_root / "derivatives" / "stroke_inpainting",
            Path("/home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/stroke_inpainting"),
        ]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument("--inpainting-root", type=Path, default=inpainting_root)
    parser.add_argument(
        "--uncertainty-root",
        type=Path,
        default=HERE / "outputs" / "lesion_uncertainty_local",
    )
    parser.add_argument(
        "--mass-effect-root",
        type=Path,
        default=HERE / "outputs" / "mass_effect_local",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE
        / "Inpainting_Code"
        / "analysis"
        / "miccai_workshop"
        / "figures"
        / "representative_subject_outputs.pdf",
    )
    return parser.parse_args()


def entities(case_id: str) -> tuple[str, str]:
    parts = case_id.split("_")
    if len(parts) < 2 or not parts[0].startswith("sub-") or not parts[1].startswith("ses-"):
        raise ValueError(f"Cannot parse subject/session from {case_id}")
    return parts[0], parts[1]


def load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def rotated(data: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(data[:, :, index])


def intensity_limits(data: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    selected = data[np.isfinite(data) & (data > 0)] if mask is None else data[mask & np.isfinite(data)]
    return tuple(float(value) for value in np.percentile(selected, [1, 99])) if selected.size else (0, 1)


def contour(axis, mask: np.ndarray, index: int, color: str, width: float) -> None:
    image = rotated(mask, index)
    if np.any(image):
        axis.contour(image, levels=[0.5], colors=[color], linewidths=width)


def anatomical_panel(axis, image: np.ndarray, index: int, title: str) -> None:
    vmin, vmax = intensity_limits(image)
    axis.imshow(rotated(image, index), cmap="gray", vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)


def overlay_panel(
    figure,
    axis,
    anatomy: np.ndarray,
    overlay: np.ndarray,
    index: int,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    threshold: float,
) -> None:
    anatomical_panel(axis, anatomy, index, title)
    values = rotated(overlay, index)
    masked = np.ma.masked_where(values < threshold, values)
    image = axis.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.82)
    figure.colorbar(image, ax=axis, fraction=0.043, pad=0.02)


def field_panel(
    figure,
    axis,
    values: np.ndarray,
    valid: np.ndarray,
    index: int,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    image_values = rotated(values, index)
    image_valid = rotated(valid, index)
    image = axis.imshow(
        np.ma.masked_where(~image_valid, image_values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_facecolor("#eeeeee")
    axis.set_title(title, fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.043, pad=0.02)


def main() -> int:
    args = parse_args()
    subject, session = entities(args.case_id)
    inpainting = args.inpainting_root / subject / session / args.case_id
    uncertainty = args.uncertainty_root / subject / session / args.case_id
    mass = args.mass_effect_root / subject / session / args.case_id
    prefix = args.case_id

    observed = load(inpainting / f"{prefix}_mni_1mm.nii.gz")
    inpainted = load(inpainting / f"{prefix}_inpainted_mni_1mm.nii.gz")
    lesion = load(inpainting / f"{prefix}_stroke_mask_mni_1mm.nii.gz") > 0.5
    target = load(inpainting / f"{prefix}_stroke_mask_dilated_3mm_mni_1mm.nii.gz") > 0.5
    probability = load(uncertainty / f"{prefix}_lesion_probability_mni_1mm.nii.gz")
    entropy = load(uncertainty / f"{prefix}_lesion_uncertainty_entropy_mni_1mm.nii.gz")
    magnitude = load(mass / f"{prefix}_mass_effect_magnitude_atlas_mm.nii.gz")
    radial = load(mass / f"{prefix}_mass_effect_radial_atlas_mm.nii.gz")
    log_jacobian = load(mass / f"{prefix}_mass_effect_log_jacobian_asymmetry.nii.gz")
    valid = load(mass / f"{prefix}_mass_effect_valid_mask.nii.gz") > 0.5
    lesion_atlas = load(mass / f"{prefix}_stroke_mask_atlas.nii.gz") > 0.5
    target_atlas = load(mass / f"{prefix}_inpainting_target_atlas.nii.gz") > 0.5
    mapped_t1_path = next(
        (
            path
            for path in [
                mass / f"{prefix}_mapped_inpainted_t1_atlas.nii.gz",
                Path(str(json.loads((mass / "mass_effect_metrics.json").read_text())["source_inverse_map"]).replace(".svreg.inv.map.nii.gz", ".bfc.nii.gz")),
            ]
            if path.is_file()
        ),
        None,
    )
    # The BrainSuite BFC image is on the subject grid, whereas field outputs are
    # on the atlas grid. Recreate the mapped anatomical underlay from the inverse
    # coordinate map used by the extractor.
    metrics = json.loads((mass / "mass_effect_metrics.json").read_text())
    inverse_map = load(Path(metrics["source_inverse_map"]))
    subject_bfc = load(Path(str(metrics["source_inverse_map"]).replace(".svreg.inv.map.nii.gz", ".bfc.nii.gz")))
    from scipy.ndimage import map_coordinates

    mapped_t1 = map_coordinates(
        subject_bfc,
        [inverse_map[..., component] for component in range(3)],
        order=1,
        mode="constant",
        cval=0.0,
    )
    del mapped_t1_path

    source_index = int(np.argmax(np.sum(lesion, axis=(0, 1))))
    atlas_index = int(np.argmax(np.sum(lesion_atlas, axis=(0, 1))))
    uncertainty_metrics = json.loads((uncertainty / "uncertainty_metrics.json").read_text())

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tr2strokeseg-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(13.2, 7.1), constrained_layout=True)
    anatomical_panel(axes[0, 0], observed, source_index, "A  Observed T1 + lesion")
    contour(axes[0, 0], lesion, source_index, "#ff2d2d", 1.4)
    overlay_panel(
        figure,
        axes[0, 1],
        observed,
        probability,
        source_index,
        r"B  Lesion score $p_{lesion}$",
        "magma",
        0,
        1,
        0.05,
    )
    overlay_panel(
        figure,
        axes[0, 2],
        observed,
        entropy,
        source_index,
        "C  Predictive entropy",
        "plasma",
        0,
        1,
        0.02,
    )
    anatomical_panel(axes[0, 3], inpainted, source_index, "D  Inpainted T1 + target")
    contour(axes[0, 3], lesion, source_index, "#ff2d2d", 1.2)
    contour(axes[0, 3], target, source_index, "#ffe600", 0.8)

    anatomical_panel(axes[1, 0], mapped_t1, atlas_index, "E  Atlas-mapped inpainted T1")
    contour(axes[1, 0], lesion_atlas, atlas_index, "#ff2d2d", 1.2)
    contour(axes[1, 0], target_atlas, atlas_index, "#ffe600", 0.8)
    magnitude_limit = max(1.0, float(np.percentile(magnitude[valid], 99)))
    radial_limit = max(1.0, float(np.percentile(np.abs(radial[valid]), 99)))
    logj_limit = max(0.1, float(np.percentile(np.abs(log_jacobian[valid]), 99)))
    field_panel(
        figure,
        axes[1, 1],
        magnitude,
        valid,
        atlas_index,
        "F  Deformation magnitude (mm)",
        "viridis",
        0,
        magnitude_limit,
    )
    field_panel(
        figure,
        axes[1, 2],
        radial,
        valid,
        atlas_index,
        "G  Radial: inward $-$ / outward $+$",
        "coolwarm",
        -radial_limit,
        radial_limit,
    )
    field_panel(
        figure,
        axes[1, 3],
        log_jacobian,
        valid,
        atlas_index,
        "H  Log-Jacobian asymmetry",
        "PuOr_r",
        -logj_limit,
        logj_limit,
    )
    for axis in axes.flat:
        axis.axis("off")
    summary = (
        f"{prefix}: expected lesion {uncertainty_metrics['expected_lesion_volume_ml']:.1f} mL; "
        f"entropy mass {uncertainty_metrics['entropy_mass_ml']:.1f} mL; "
        f"3--20 mm median deformation {metrics['mass_effect_3_20mm_magnitude_mm_median']:.2f} mm; "
        f"median radial {metrics['mass_effect_3_20mm_radial_mm_median']:+.2f} mm"
    )
    figure.suptitle(summary, fontsize=10.5, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, facecolor="white", bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

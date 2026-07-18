#!/usr/bin/env python3
"""Estimate lesion-associated deformation from completed ARC SVReg outputs.

This is a cross-sectional proxy, not an identifiable physical ground-truth
mass-effect field.  The pipeline uses the inpainted atlas-to-subject SVReg
coordinate map, fits its global affine component on the contralateral
hemisphere, converts the nonlinear residual to atlas-aligned millimetres, and
subtracts the mirrored contralateral residual.  Values inside the lesion and
the synthetic inpainting target are excluded from every biological summary.

Per case, the script stores an atlas-space vector field, displacement
magnitude, radial displacement relative to the lesion, log-Jacobian asymmetry,
valid-analysis mask, shell-wise metrics, and optional registration-sensitivity
metrics from the direct (non-inpainted) BrainSuite branch. Subject-level QC
figures are written by default for audit, while the intended inference is
cohort-level analysis and WAB Aphasia Quotient prediction.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


HERE = Path(__file__).resolve().parent
CASE_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_.+_T1w$")
INPAINTED_PREFIX_SUFFIX = "_inpainted_mni_1mm"
INV_MAP_SUFFIX = ".svreg.inv.map.nii.gz"
SHELLS_MM = ((3.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0))
METHOD_VERSION = "contralateral_affine_residual_v1"


@dataclass(frozen=True)
class Case:
    case_id: str
    subject: str
    session: str
    brainsuite_prefix: Path
    inverse_map: Path
    subject_bfc: Path
    subject_mask: Path
    lesion_mask: Path
    inpainting_target: Path
    raw_prefix: Path | None


@dataclass(frozen=True)
class Outputs:
    case_dir: Path
    vector_field: Path
    magnitude: Path
    radial: Path
    log_jacobian_asymmetry: Path
    valid_mask: Path
    lesion_atlas: Path
    target_atlas: Path
    raw_sensitivity: Path
    metrics_json: Path
    qc_png: Path
    error_json: Path


@dataclass
class NormalizedField:
    displacement_mm: np.ndarray
    valid: np.ndarray
    affine_coefficients: np.ndarray
    affine_fit_rmse_mm: float
    affine_fit_points: int
    affine_fit_inlier_fraction: float
    jacobian: np.ndarray
    folding_fraction: float


def environment_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def default_arc_root() -> Path:
    return first_existing(
        [
            environment_path("ARC_ROOT"),
            Path("/project2/ajoshi_1183/data/ARC"),
            Path("/home/ajoshi/project2_ajoshi_1183/data/ARC"),
        ]
    ) or Path("/project2/ajoshi_1183/data/ARC")


def default_brainsuite_home() -> Path | None:
    return first_existing(
        [
            environment_path("BRAINSUITE_HOME"),
            Path("/opt/BrainSuite23a"),
            Path("/project2/ajoshi_27/BrainSuite23a"),
            Path("/home/ajoshi/Software/BrainSuite23a"),
        ]
    )


def parse_args() -> argparse.Namespace:
    arc_root = default_arc_root()
    brainsuite_home = default_brainsuite_home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-root", type=Path, default=arc_root)
    parser.add_argument(
        "--inpainted-brainsuite-root",
        type=Path,
        default=arc_root / "derivatives" / "brainsuite_anatomical_bidsapp",
    )
    parser.add_argument(
        "--raw-brainsuite-root",
        type=Path,
        default=arc_root / "derivatives" / "brainsuite_anatomical_raw",
    )
    parser.add_argument(
        "--inpainting-root",
        type=Path,
        default=arc_root / "derivatives" / "stroke_inpainting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=arc_root / "derivatives" / "lesion_mass_effect",
    )
    parser.add_argument(
        "--atlas-bfc",
        type=Path,
        default=(brainsuite_home / "svreg" / "BrainSuiteAtlas1" / "mri.bfc.nii.gz")
        if brainsuite_home
        else None,
    )
    parser.add_argument(
        "--atlas-mask",
        type=Path,
        default=(brainsuite_home / "svreg" / "BrainSuiteAtlas1" / "mri.mask.nii.gz")
        if brainsuite_home
        else None,
    )
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--smoothing-mm",
        type=float,
        default=2.0,
        help="Gaussian smoothing of normalized displacement before derivatives",
    )
    parser.add_argument(
        "--minimum-laterality",
        type=float,
        default=0.80,
        help="Flag below this value as unsupported by contralateral normalization",
    )
    parser.add_argument(
        "--fit-subsample",
        type=int,
        default=8,
        help="Use every Nth valid contralateral voxel for robust affine fitting",
    )
    parser.add_argument(
        "--raw-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare normalized fields from raw and inpainted BrainSuite branches",
    )
    parser.add_argument(
        "--make-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write subject-level QC PNGs (enabled by default)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def case_entities(case_id: str) -> tuple[str, str]:
    match = CASE_RE.match(case_id)
    if match is None:
        raise ValueError(f"Cannot parse BIDS entities from {case_id}")
    return match.group(1), match.group(2)


def raw_prefix_for_case(root: Path, subject: str, session: str, case_id: str) -> Path | None:
    search_root = root / subject / session
    if not search_root.is_dir():
        return None
    matches = sorted(search_root.glob(f"*/{case_id}{INV_MAP_SUFFIX}"))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple raw SVReg maps found for {case_id}: {matches}")
    if not matches:
        return None
    return Path(str(matches[0])[: -len(INV_MAP_SUFFIX)])


def discover_cases(args: argparse.Namespace) -> list[Case]:
    selected_subjects = set(args.subject)
    selected_cases = set(args.case)
    search_roots = (
        [args.inpainted_brainsuite_root / subject for subject in sorted(selected_subjects)]
        if selected_subjects
        else [args.inpainted_brainsuite_root]
    )
    inverse_maps: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            inverse_maps.extend(root.rglob(f"*{INPAINTED_PREFIX_SUFFIX}{INV_MAP_SUFFIX}"))

    cases: dict[str, Case] = {}
    for inverse_map in sorted(inverse_maps):
        prefix_text = str(inverse_map)[: -len(INV_MAP_SUFFIX)]
        prefix = Path(prefix_text)
        prefix_name = prefix.name
        if not prefix_name.endswith(INPAINTED_PREFIX_SUFFIX):
            continue
        case_id = prefix_name[: -len(INPAINTED_PREFIX_SUFFIX)]
        subject, session = case_entities(case_id)
        if selected_subjects and subject not in selected_subjects:
            continue
        if selected_cases and case_id not in selected_cases:
            continue
        if case_id in cases:
            raise RuntimeError(f"Duplicate inpainted SVReg case: {case_id}")
        case_dir = args.inpainting_root / subject / session / case_id
        raw_prefix = (
            raw_prefix_for_case(args.raw_brainsuite_root, subject, session, case_id)
            if args.raw_sensitivity
            else None
        )
        cases[case_id] = Case(
            case_id=case_id,
            subject=subject,
            session=session,
            brainsuite_prefix=prefix,
            inverse_map=inverse_map,
            subject_bfc=Path(prefix_text + ".bfc.nii.gz"),
            subject_mask=Path(prefix_text + ".mask.nii.gz"),
            lesion_mask=case_dir / f"{case_id}_stroke_mask_mni_1mm.nii.gz",
            inpainting_target=case_dir / f"{case_id}_stroke_mask_dilated_3mm_mni_1mm.nii.gz",
            raw_prefix=raw_prefix,
        )
    result = [cases[key] for key in sorted(cases)]
    if args.limit is not None:
        result = result[: args.limit]
    missing_subjects = selected_subjects - {case.subject for case in result}
    missing_cases = selected_cases - {case.case_id for case in result}
    if missing_subjects:
        raise ValueError(f"Requested subjects not found: {', '.join(sorted(missing_subjects))}")
    if missing_cases:
        raise ValueError(f"Requested cases not found: {', '.join(sorted(missing_cases))}")
    return result


def outputs_for_case(case: Case, output_dir: Path) -> Outputs:
    case_dir = output_dir / case.subject / case.session / case.case_id
    stem = case.case_id
    return Outputs(
        case_dir=case_dir,
        vector_field=case_dir / f"{stem}_mass_effect_displacement_atlas_mm.nii.gz",
        magnitude=case_dir / f"{stem}_mass_effect_magnitude_atlas_mm.nii.gz",
        radial=case_dir / f"{stem}_mass_effect_radial_atlas_mm.nii.gz",
        log_jacobian_asymmetry=case_dir / f"{stem}_mass_effect_log_jacobian_asymmetry.nii.gz",
        valid_mask=case_dir / f"{stem}_mass_effect_valid_mask.nii.gz",
        lesion_atlas=case_dir / f"{stem}_stroke_mask_atlas.nii.gz",
        target_atlas=case_dir / f"{stem}_inpainting_target_atlas.nii.gz",
        raw_sensitivity=case_dir / f"{stem}_registration_sensitivity_raw_vs_inpainted_mm.nii.gz",
        metrics_json=case_dir / "mass_effect_metrics.json",
        qc_png=case_dir / f"{stem}_mass_effect_qc.png",
        error_json=case_dir / "processing_error.json",
    )


def outputs_complete(outputs: Outputs, make_qc: bool) -> bool:
    required = [
        outputs.vector_field,
        outputs.magnitude,
        outputs.radial,
        outputs.log_jacobian_asymmetry,
        outputs.valid_mask,
        outputs.lesion_atlas,
        outputs.target_atlas,
        outputs.metrics_json,
    ]
    if make_qc:
        required.append(outputs.qc_png)
    return all(path.is_file() for path in required)


def validate_args(args: argparse.Namespace) -> None:
    required_dirs = {
        "--inpainted-brainsuite-root": args.inpainted_brainsuite_root,
        "--inpainting-root": args.inpainting_root,
    }
    for label, path in required_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for label, path in {"--atlas-bfc": args.atlas_bfc, "--atlas-mask": args.atlas_mask}.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.raw_sensitivity and not args.raw_brainsuite_root.is_dir():
        raise FileNotFoundError(f"--raw-brainsuite-root does not exist: {args.raw_brainsuite_root}")
    if args.smoothing_mm < 0:
        raise ValueError("--smoothing-mm must be nonnegative")
    if not 0 <= args.minimum_laterality <= 1:
        raise ValueError("--minimum-laterality must be in [0, 1]")
    if args.fit_subsample <= 0:
        raise ValueError("--fit-subsample must be positive")


def same_geometry(first: nib.spatialimages.SpatialImage, second: nib.spatialimages.SpatialImage) -> bool:
    return first.shape[:3] == second.shape[:3] and np.allclose(
        first.affine, second.affine, atol=1e-4, rtol=1e-5
    )


def sample_subject_image(data: np.ndarray, inverse_map: np.ndarray, order: int) -> np.ndarray:
    coordinates = [inverse_map[..., axis] for axis in range(3)]
    return map_coordinates(
        data,
        coordinates,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def inverse_map_validity(
    inverse_map: np.ndarray,
    subject_shape: tuple[int, int, int],
    atlas_mask: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    valid = atlas_mask & np.isfinite(inverse_map).all(axis=-1)
    for axis, size in enumerate(subject_shape):
        valid &= inverse_map[..., axis] >= 0
        valid &= inverse_map[..., axis] <= size - 1
    mapped_subject_mask = sample_subject_image(
        np.asarray(subject_mask, dtype=np.float32), inverse_map, order=0
    )
    valid &= mapped_subject_mask > 0.5
    return valid


def lesion_laterality(lesion: np.ndarray) -> tuple[str, float, int, int]:
    midline = (lesion.shape[0] - 1) / 2.0
    x = np.arange(lesion.shape[0], dtype=float)[:, None, None]
    left = int(np.count_nonzero(lesion & (x < midline)))
    right = int(np.count_nonzero(lesion & (x > midline)))
    total = left + right
    if total == 0:
        raise ValueError("Lesion is empty after atlas mapping")
    laterality = abs(left - right) / total
    side = "left" if left > right else "right" if right > left else "bilateral"
    return side, float(laterality), left, right


def contralateral_mask(shape: tuple[int, int, int], lesion_side: str) -> np.ndarray:
    midline = (shape[0] - 1) / 2.0
    x = np.arange(shape[0], dtype=float)[:, None, None]
    if lesion_side == "left":
        return np.broadcast_to(x > midline + 2.0, shape)
    if lesion_side == "right":
        return np.broadcast_to(x < midline - 2.0, shape)
    raise ValueError("A dominant lesion side is required")


def fit_robust_affine(
    inverse_map: np.ndarray,
    fit_mask: np.ndarray,
    subsample: int,
) -> tuple[np.ndarray, float, int, float]:
    coordinates = np.argwhere(fit_mask)
    if coordinates.shape[0] < 10_000:
        raise ValueError(f"Too few contralateral affine-fit voxels: {coordinates.shape[0]}")
    coordinates = coordinates[::subsample]
    target = inverse_map[tuple(coordinates.T)].astype(np.float64)
    design = np.column_stack([coordinates.astype(np.float64), np.ones(len(coordinates))])
    keep = np.ones(len(coordinates), dtype=bool)
    initial_count = len(coordinates)
    coefficients = np.zeros((4, 3), dtype=np.float64)
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        residual = np.linalg.norm(target - design @ coefficients, axis=1)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        scale = max(1e-6, 1.4826 * mad)
        new_keep = residual <= center + 4.5 * scale
        if np.array_equal(new_keep, keep):
            break
        if int(new_keep.sum()) < 10_000 / subsample:
            break
        keep = new_keep
    residual = np.linalg.norm(target[keep] - design[keep] @ coefficients, axis=1)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return coefficients, rmse, int(keep.sum()), float(keep.sum() / initial_count)


def affine_prediction(shape: tuple[int, int, int], coefficients: np.ndarray) -> np.ndarray:
    x, y, z = np.indices(shape, dtype=np.float32, sparse=True)
    prediction = np.empty((*shape, 3), dtype=np.float32)
    for component in range(3):
        prediction[..., component] = (
            coefficients[0, component] * x
            + coefficients[1, component] * y
            + coefficients[2, component] * z
            + coefficients[3, component]
        )
    return prediction


def smooth_vector_masked(field: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        result = field.copy()
        result[~valid] = 0
        return result
    weight = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="constant")
    result = np.zeros_like(field, dtype=np.float32)
    stable = weight > 0.25
    for component in range(3):
        numerator = gaussian_filter(
            np.where(valid, field[..., component], 0.0), sigma=sigma, mode="constant"
        )
        result[..., component][stable] = numerator[stable] / weight[stable]
    result[~valid] = 0
    return result


def jacobian_determinant(displacement: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    gradients: list[list[np.ndarray]] = []
    for component in range(3):
        component_gradients = np.gradient(
            displacement[..., component], *spacing, edge_order=1
        )
        gradients.append([np.asarray(value, dtype=np.float32) for value in component_gradients])
    f00 = 1.0 + gradients[0][0]
    f01 = gradients[0][1]
    f02 = gradients[0][2]
    f10 = gradients[1][0]
    f11 = 1.0 + gradients[1][1]
    f12 = gradients[1][2]
    f20 = gradients[2][0]
    f21 = gradients[2][1]
    f22 = 1.0 + gradients[2][2]
    determinant = (
        f00 * (f11 * f22 - f12 * f21)
        - f01 * (f10 * f22 - f12 * f20)
        + f02 * (f10 * f21 - f11 * f20)
    )
    return np.asarray(determinant, dtype=np.float32)


def compute_normalized_field(
    inverse_map_path: Path,
    subject_bfc_path: Path,
    subject_mask_path: Path,
    atlas_mask: np.ndarray,
    lesion_side: str,
    smoothing_mm: float,
    fit_subsample: int,
    atlas_spacing: tuple[float, float, float],
) -> NormalizedField:
    inverse_image = nib.load(inverse_map_path)
    inverse_map = np.asarray(inverse_image.dataobj, dtype=np.float32)
    subject_image = nib.load(subject_bfc_path)
    mask_image = nib.load(subject_mask_path)
    if inverse_map.shape != (*atlas_mask.shape, 3):
        raise ValueError(f"Unexpected inverse-map shape: {inverse_map.shape}")
    if subject_image.shape != mask_image.shape or not same_geometry(subject_image, mask_image):
        raise ValueError("Subject BFC and brain mask geometry differ")
    subject_mask = np.asarray(mask_image.dataobj) > 0
    valid = inverse_map_validity(inverse_map, subject_image.shape, atlas_mask, subject_mask)
    fit_mask = valid & contralateral_mask(atlas_mask.shape, lesion_side)
    coefficients, rmse, fit_points, inlier_fraction = fit_robust_affine(
        inverse_map, fit_mask, fit_subsample
    )
    prediction = affine_prediction(atlas_mask.shape, coefficients)
    residual_subject = inverse_map - prediction
    linear = coefficients[:3, :].T
    condition_number = float(np.linalg.cond(linear))
    if not np.isfinite(condition_number) or condition_number > 100:
        raise ValueError(f"Ill-conditioned fitted affine: condition={condition_number:.2f}")
    inverse_linear = np.linalg.inv(linear)
    displacement = residual_subject @ inverse_linear.T
    displacement = np.asarray(displacement, dtype=np.float32)
    displacement[~valid] = 0
    sigma_voxels = tuple(smoothing_mm / value for value in atlas_spacing)
    # Atlas is isotropic in the current pipeline; use the mean for scipy's
    # normalized masked smoothing to keep a common physical kernel.
    displacement = smooth_vector_masked(displacement, valid, float(np.mean(sigma_voxels)))
    jacobian = jacobian_determinant(displacement, atlas_spacing)
    folding_fraction = float(np.mean(jacobian[valid] <= 0))
    return NormalizedField(
        displacement_mm=displacement,
        valid=valid,
        affine_coefficients=coefficients,
        affine_fit_rmse_mm=rmse,
        affine_fit_points=fit_points,
        affine_fit_inlier_fraction=inlier_fraction,
        jacobian=jacobian,
        folding_fraction=folding_fraction,
    )


def mirrored_vector(field: np.ndarray) -> np.ndarray:
    mirrored = np.flip(field, axis=0).copy()
    mirrored[..., 0] *= -1
    return mirrored


def mass_effect_proxy(
    field: NormalizedField, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paired_valid = field.valid & np.flip(field.valid, axis=0)
    # Both members of a left/right pair must avoid synthetic inpainting data.
    # Excluding only the ipsilateral target would leave its mirror marked valid
    # on the contralateral half of the stored audit field.
    paired_valid &= ~target & ~np.flip(target, axis=0)
    asymmetry = field.displacement_mm - mirrored_vector(field.displacement_mm)
    asymmetry[~paired_valid] = 0
    positive_jacobian = (field.jacobian > 0) & field.valid
    log_jacobian = np.zeros(field.valid.shape, dtype=np.float32)
    log_jacobian[positive_jacobian] = np.log(field.jacobian[positive_jacobian])
    paired_jacobian = (
        positive_jacobian
        & np.flip(positive_jacobian, axis=0)
        & ~target
        & ~np.flip(target, axis=0)
    )
    log_jacobian_asymmetry = log_jacobian - np.flip(log_jacobian, axis=0)
    log_jacobian_asymmetry[~paired_jacobian] = 0
    valid = paired_valid & paired_jacobian
    asymmetry[~valid] = 0
    return asymmetry, log_jacobian_asymmetry, valid


def lesion_distance_and_normals(
    lesion: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    distance = distance_transform_edt(~lesion, sampling=spacing).astype(np.float32)
    gradients = np.gradient(distance, *spacing, edge_order=1)
    normal = np.stack(gradients, axis=-1).astype(np.float32)
    norm = np.linalg.norm(normal, axis=-1)
    stable = norm > 1e-6
    normal[stable] /= norm[stable, None]
    normal[~stable] = 0
    return distance, normal


def safe_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_n_voxels": 0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p25": math.nan,
            f"{prefix}_p75": math.nan,
            f"{prefix}_p95": math.nan,
        }
    p25, median, p75, p95 = np.percentile(finite, [25, 50, 75, 95])
    return {
        f"{prefix}_n_voxels": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(median),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p95": float(p95),
    }


def shell_key(lower: float, upper: float) -> str:
    return f"shell_{lower:g}_{upper:g}mm".replace(".", "p")


def calculate_metrics(
    case: Case,
    atlas_image: nib.spatialimages.SpatialImage,
    lesion: np.ndarray,
    target: np.ndarray,
    lesion_side: str,
    laterality: float,
    lesion_left_voxels: int,
    lesion_right_voxels: int,
    normalized: NormalizedField,
    asymmetry: np.ndarray,
    log_jacobian_asymmetry: np.ndarray,
    valid: np.ndarray,
    distance: np.ndarray,
    radial: np.ndarray,
    magnitude: np.ndarray,
    raw_sensitivity: np.ndarray | None,
    minimum_laterality: float,
    smoothing_mm: float,
) -> dict[str, object]:
    voxel_volume_mm3 = float(abs(np.linalg.det(atlas_image.affine[:3, :3])))
    midline = (lesion.shape[0] - 1) / 2.0
    x = np.arange(lesion.shape[0], dtype=float)[:, None, None]
    ipsilateral = x < midline if lesion_side == "left" else x > midline
    ipsilateral = np.broadcast_to(ipsilateral, lesion.shape)
    analysis = valid & ipsilateral
    metrics: dict[str, object] = {
        "case_id": case.case_id,
        "subject": case.subject,
        "session": case.session,
        "method_version": METHOD_VERSION,
        "interpretation": "contralateral-normalized lesion-associated deformation proxy",
        "not_physical_ground_truth": True,
        "lesion_side": lesion_side,
        "lesion_laterality_index": laterality,
        "minimum_supported_laterality": minimum_laterality,
        "laterality_supported": bool(laterality >= minimum_laterality),
        "lesion_left_voxels_atlas": lesion_left_voxels,
        "lesion_right_voxels_atlas": lesion_right_voxels,
        "lesion_volume_atlas_ml": float(lesion.sum() * voxel_volume_mm3 / 1000.0),
        "inpainting_target_volume_atlas_ml": float(target.sum() * voxel_volume_mm3 / 1000.0),
        "valid_analysis_volume_ml": float(analysis.sum() * voxel_volume_mm3 / 1000.0),
        "smoothing_mm": smoothing_mm,
        "contralateral_affine_fit_rmse_mm": normalized.affine_fit_rmse_mm,
        "contralateral_affine_fit_points": normalized.affine_fit_points,
        "contralateral_affine_fit_inlier_fraction": normalized.affine_fit_inlier_fraction,
        "normalized_field_folding_fraction": normalized.folding_fraction,
        "affine_coefficients_atlas_to_subject_voxels": normalized.affine_coefficients.tolist(),
        "source_inverse_map": str(case.inverse_map),
        "source_lesion_mask": str(case.lesion_mask),
        "source_inpainting_target": str(case.inpainting_target),
    }
    metrics.update(safe_summary(magnitude[analysis], "ipsilateral_magnitude_mm"))
    metrics.update(safe_summary(radial[analysis], "ipsilateral_radial_mm"))
    metrics.update(
        safe_summary(log_jacobian_asymmetry[analysis], "ipsilateral_log_jacobian_asymmetry")
    )

    for lower, upper in SHELLS_MM:
        region = analysis & (distance >= lower) & (distance < upper)
        key = shell_key(lower, upper)
        metrics.update(safe_summary(magnitude[region], f"{key}_magnitude_mm"))
        metrics.update(safe_summary(radial[region], f"{key}_radial_mm"))
        metrics.update(
            safe_summary(log_jacobian_asymmetry[region], f"{key}_log_jacobian_asymmetry")
        )
        count = int(region.sum())
        metrics[f"{key}_outward_fraction"] = (
            float(np.mean(radial[region] > 0)) if count else math.nan
        )
        metrics[f"{key}_inward_fraction"] = (
            float(np.mean(radial[region] < 0)) if count else math.nan
        )

    near = analysis & (distance >= 3.0) & (distance < 20.0)
    near_radial = radial[near]
    near_magnitude = magnitude[near]
    near_logj = log_jacobian_asymmetry[near]
    metrics.update(safe_summary(near_magnitude, "mass_effect_3_20mm_magnitude_mm"))
    metrics.update(safe_summary(near_radial, "mass_effect_3_20mm_radial_mm"))
    metrics.update(safe_summary(near_logj, "mass_effect_3_20mm_log_jacobian_asymmetry"))
    metrics["mass_effect_3_20mm_mean_absolute_radial_mm"] = (
        float(np.mean(np.abs(near_radial))) if near_radial.size else math.nan
    )
    metrics["mass_effect_3_20mm_outward_integral_ml_mm"] = float(
        np.sum(np.clip(near_radial, 0, None), dtype=np.float64) * voxel_volume_mm3 / 1000.0
    )
    metrics["mass_effect_3_20mm_inward_integral_ml_mm"] = float(
        np.sum(np.clip(-near_radial, 0, None), dtype=np.float64) * voxel_volume_mm3 / 1000.0
    )
    metrics["mass_effect_3_20mm_magnitude_integral_ml_mm"] = float(
        np.sum(near_magnitude, dtype=np.float64) * voxel_volume_mm3 / 1000.0
    )
    metrics["mass_effect_3_20mm_logjac_expansion_integral_ml"] = float(
        np.sum(np.clip(near_logj, 0, None), dtype=np.float64) * voxel_volume_mm3 / 1000.0
    )
    metrics["mass_effect_3_20mm_logjac_compression_integral_ml"] = float(
        np.sum(np.clip(-near_logj, 0, None), dtype=np.float64) * voxel_volume_mm3 / 1000.0
    )
    if raw_sensitivity is not None:
        sensitivity_values = raw_sensitivity[near]
        metrics.update(
            safe_summary(sensitivity_values, "registration_sensitivity_3_20mm_mm")
        )
        metrics["raw_brainsuite_prefix"] = str(case.raw_prefix)
    else:
        metrics["registration_sensitivity_3_20mm_mm_n_voxels"] = 0
        metrics["raw_brainsuite_prefix"] = ""

    signal_median = metrics.get("mass_effect_3_20mm_magnitude_mm_median", math.nan)
    sensitivity_median = metrics.get(
        "registration_sensitivity_3_20mm_mm_median", math.nan
    )
    metrics["mass_effect_to_registration_sensitivity_ratio"] = (
        float(signal_median / sensitivity_median)
        if isinstance(signal_median, (int, float))
        and isinstance(sensitivity_median, (int, float))
        and np.isfinite(signal_median)
        and np.isfinite(sensitivity_median)
        and sensitivity_median > 0
        else math.nan
    )
    metrics["deformation_qc_pass"] = bool(
        metrics["laterality_supported"]
        and normalized.folding_fraction <= 0.05
        and near_magnitude.size >= 1000
    )
    metrics["deformation_qc_criteria"] = {
        "laterality_fraction_minimum": float(minimum_laterality),
        "normalized_field_folding_fraction_maximum": 0.05,
        "mass_effect_3_20mm_valid_voxels_minimum": 1000,
    }
    return metrics


def save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
    description: str,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header["descrip"] = description[:79]
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)
    nib.save(image, path)


def write_json_atomic(path: Path, record: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=True)
        handle.write("\n")
    os.replace(temporary, path)


def make_qc(
    case: Case,
    mapped_t1: np.ndarray,
    lesion: np.ndarray,
    target: np.ndarray,
    magnitude: np.ndarray,
    radial: np.ndarray,
    log_jacobian_asymmetry: np.ndarray,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tr2strokeseg-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slice_index = int(np.argmax(np.sum(lesion, axis=(0, 1))))
    arrays = [mapped_t1, magnitude, radial, log_jacobian_asymmetry]
    slices = [np.rot90(array[:, :, slice_index]) for array in arrays]
    lesion_slice = np.rot90(lesion[:, :, slice_index])
    target_slice = np.rot90(target[:, :, slice_index])
    anatomy_values = slices[0][np.isfinite(slices[0]) & (slices[0] > 0)]
    vmin, vmax = np.percentile(anatomy_values, [1, 99]) if anatomy_values.size else (0, 1)
    magnitude_vmax = max(1.0, float(np.percentile(magnitude[magnitude > 0], 99))) if np.any(magnitude > 0) else 1.0
    radial_vmax = max(1.0, float(np.percentile(np.abs(radial[radial != 0]), 99))) if np.any(radial != 0) else 1.0
    logj_vmax = max(0.1, float(np.percentile(np.abs(log_jacobian_asymmetry[log_jacobian_asymmetry != 0]), 99))) if np.any(log_jacobian_asymmetry != 0) else 0.1

    figure, axes = plt.subplots(1, 4, figsize=(17, 4.3), constrained_layout=True)
    axes[0].imshow(slices[0], cmap="gray", vmin=vmin, vmax=vmax)
    if lesion_slice.any():
        axes[0].contour(lesion_slice, levels=[0.5], colors="red", linewidths=1.5)
    if target_slice.any():
        axes[0].contour(target_slice, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[0].set_title("Mapped T1; lesion/target")
    image = axes[1].imshow(slices[1], cmap="viridis", vmin=0, vmax=magnitude_vmax)
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    axes[1].set_title("Asymmetric displacement (mm)")
    image = axes[2].imshow(slices[2], cmap="coolwarm", vmin=-radial_vmax, vmax=radial_vmax)
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    axes[2].set_title("Radial: inward − / outward +")
    image = axes[3].imshow(slices[3], cmap="PuOr_r", vmin=-logj_vmax, vmax=logj_vmax)
    figure.colorbar(image, ax=axes[3], fraction=0.046)
    axes[3].set_title("Log-Jacobian asymmetry")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{case.case_id} | deformation proxy QC", fontweight="bold")
    figure.savefig(output_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def validate_case_inputs(case: Case) -> None:
    required = [
        case.inverse_map,
        case.subject_bfc,
        case.subject_mask,
        case.lesion_mask,
        case.inpainting_target,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing case inputs: {missing}")


def process_case(
    case: Case,
    args: argparse.Namespace,
    atlas_image: nib.spatialimages.SpatialImage,
    atlas_mask: np.ndarray,
) -> dict[str, object]:
    validate_case_inputs(case)
    outputs = outputs_for_case(case, args.output_dir)
    outputs.case_dir.mkdir(parents=True, exist_ok=True)
    inverse_map = np.asarray(nib.load(case.inverse_map).dataobj, dtype=np.float32)
    lesion_image = nib.load(case.lesion_mask)
    target_image = nib.load(case.inpainting_target)
    subject_image = nib.load(case.subject_bfc)
    if not same_geometry(lesion_image, target_image) or not same_geometry(lesion_image, subject_image):
        raise ValueError("Lesion, target, and inpainted subject geometry differ")
    lesion_subject = np.asarray(lesion_image.dataobj) > 0
    target_subject = np.asarray(target_image.dataobj) > 0
    lesion = sample_subject_image(lesion_subject.astype(np.float32), inverse_map, order=0) > 0.5
    target = sample_subject_image(target_subject.astype(np.float32), inverse_map, order=0) > 0.5
    target |= lesion
    lesion_side, laterality, left_voxels, right_voxels = lesion_laterality(lesion)
    spacing = tuple(float(value) for value in atlas_image.header.get_zooms()[:3])

    normalized = compute_normalized_field(
        case.inverse_map,
        case.subject_bfc,
        case.subject_mask,
        atlas_mask,
        lesion_side,
        args.smoothing_mm,
        args.fit_subsample,
        spacing,
    )
    asymmetry, logj_asymmetry, valid = mass_effect_proxy(normalized, target)
    distance, normal = lesion_distance_and_normals(lesion, spacing)
    radial = np.sum(asymmetry * normal, axis=-1, dtype=np.float32)
    magnitude = np.linalg.norm(asymmetry, axis=-1).astype(np.float32)
    radial[~valid] = 0
    magnitude[~valid] = 0

    raw_sensitivity: np.ndarray | None = None
    if args.raw_sensitivity and case.raw_prefix is not None:
        raw_inverse = Path(str(case.raw_prefix) + INV_MAP_SUFFIX)
        raw_bfc = Path(str(case.raw_prefix) + ".bfc.nii.gz")
        raw_mask = Path(str(case.raw_prefix) + ".mask.nii.gz")
        if raw_inverse.is_file() and raw_bfc.is_file() and raw_mask.is_file():
            raw_normalized = compute_normalized_field(
                raw_inverse,
                raw_bfc,
                raw_mask,
                atlas_mask,
                lesion_side,
                args.smoothing_mm,
                args.fit_subsample,
                spacing,
            )
            raw_asymmetry, _, raw_valid = mass_effect_proxy(raw_normalized, target)
            sensitivity_valid = valid & raw_valid
            raw_sensitivity = np.linalg.norm(asymmetry - raw_asymmetry, axis=-1).astype(np.float32)
            raw_sensitivity[~sensitivity_valid] = 0
            save_nifti(
                raw_sensitivity,
                atlas_image,
                outputs.raw_sensitivity,
                np.float32,
                "Registration sensitivity: inpainted minus raw field magnitude, mm",
            )

    metrics = calculate_metrics(
        case,
        atlas_image,
        lesion,
        target,
        lesion_side,
        laterality,
        left_voxels,
        right_voxels,
        normalized,
        asymmetry,
        logj_asymmetry,
        valid,
        distance,
        radial,
        magnitude,
        raw_sensitivity,
        args.minimum_laterality,
        args.smoothing_mm,
    )
    metrics.update(
        {
            "mass_effect_vector_path": str(outputs.vector_field),
            "mass_effect_magnitude_path": str(outputs.magnitude),
            "mass_effect_radial_path": str(outputs.radial),
            "log_jacobian_asymmetry_path": str(outputs.log_jacobian_asymmetry),
            "valid_mask_path": str(outputs.valid_mask),
            "subject_qc_png": str(outputs.qc_png) if args.make_qc else "",
        }
    )

    save_nifti(
        asymmetry,
        atlas_image,
        outputs.vector_field,
        np.float32,
        "Contralateral-normalized lesion deformation proxy, atlas mm",
    )
    save_nifti(magnitude, atlas_image, outputs.magnitude, np.float32, "Mass-effect proxy magnitude, mm")
    save_nifti(radial, atlas_image, outputs.radial, np.float32, "Radial deformation: inward negative, outward positive, mm")
    save_nifti(
        logj_asymmetry,
        atlas_image,
        outputs.log_jacobian_asymmetry,
        np.float32,
        "Ipsilateral-minus-mirrored log-Jacobian of normalized field",
    )
    save_nifti(valid, atlas_image, outputs.valid_mask, np.uint8, "Valid deformation proxy voxels outside inpainting target")
    save_nifti(lesion, atlas_image, outputs.lesion_atlas, np.uint8, "Stroke lesion mapped to BrainSuite atlas grid")
    save_nifti(target, atlas_image, outputs.target_atlas, np.uint8, "Synthetic inpainting target mapped to BrainSuite atlas grid")
    if args.make_qc:
        mapped_t1 = sample_subject_image(
            np.asarray(subject_image.dataobj, dtype=np.float32), inverse_map, order=1
        )
        make_qc(
            case,
            mapped_t1,
            lesion,
            target,
            magnitude,
            radial,
            logj_asymmetry,
            outputs.qc_png,
        )
    write_json_atomic(outputs.metrics_json, metrics)
    if outputs.error_json.is_file():
        outputs.error_json.unlink()
    return metrics


def write_error(case: Case, outputs: Outputs, error: Exception) -> None:
    outputs.case_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        outputs.error_json,
        {
            "case_id": case.case_id,
            "subject": case.subject,
            "session": case.session,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


@contextmanager
def manifest_lock(output_dir: Path) -> Iterator[None]:
    with (output_dir / ".manifest.lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rebuild_manifests(output_dir: Path) -> None:
    with manifest_lock(output_dir):
        metrics = []
        for path in sorted(output_dir.rglob("mass_effect_metrics.json")):
            with path.open(encoding="utf-8") as handle:
                metrics.append(json.load(handle))
        atomic_csv(output_dir / "mass_effect_manifest.csv", metrics)
        failures = []
        for path in sorted(output_dir.rglob("processing_error.json")):
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            record.pop("traceback", None)
            failures.append(record)
        failure_path = output_dir / "failures.csv"
        if failures:
            atomic_csv(failure_path, failures)
        elif failure_path.is_file():
            failure_path.unlink()


def mark_complete(cases: list[Case], args: argparse.Namespace) -> None:
    for subject in sorted({case.subject for case in cases}):
        subject_cases = [case for case in cases if case.subject == subject]
        if all(outputs_complete(outputs_for_case(case, args.output_dir), args.make_qc) for case in subject_cases):
            marker = args.output_dir / subject / "mass_effect_complete"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()


def main() -> int:
    args = parse_args()
    for name in (
        "arc_root",
        "inpainted_brainsuite_root",
        "raw_brainsuite_root",
        "inpainting_root",
        "output_dir",
        "atlas_bfc",
        "atlas_mask",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    validate_args(args)
    cases = discover_cases(args)
    if not cases:
        print("No matched inpainted SVReg cases were found", file=os.sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        case
        for case in cases
        if args.overwrite or not outputs_complete(outputs_for_case(case, args.output_dir), args.make_qc)
    ]
    print(
        f"Found {len(cases)} case(s) across {len({case.subject for case in cases})} subject(s); "
        f"{len(pending)} pending",
        flush=True,
    )
    print(f"Output root: {args.output_dir}", flush=True)
    if args.dry_run:
        for case in pending:
            print(f"Would calculate {case.case_id} -> {outputs_for_case(case, args.output_dir).case_dir}")
        return 0
    if not pending:
        rebuild_manifests(args.output_dir)
        mark_complete(cases, args)
        print("All selected cases are complete")
        return 0

    atlas_image = nib.load(args.atlas_bfc)
    atlas_mask_image = nib.load(args.atlas_mask)
    if not same_geometry(atlas_image, atlas_mask_image):
        raise ValueError("Atlas BFC and mask geometry differ")
    atlas_mask = np.asarray(atlas_mask_image.dataobj) > 0
    failures = 0
    for index, case in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {case.case_id}", flush=True)
        outputs = outputs_for_case(case, args.output_dir)
        try:
            metrics = process_case(case, args, atlas_image, atlas_mask)
            print(
                f"  side={metrics['lesion_side']} laterality={metrics['lesion_laterality_index']:.3f} "
                f"3-20mm |radial|={metrics['mass_effect_3_20mm_mean_absolute_radial_mm']:.3f} mm",
                flush=True,
            )
        except Exception as error:
            failures += 1
            write_error(case, outputs, error)
            print(
                f"  FAILED: {type(error).__name__}: {error}", file=os.sys.stderr, flush=True
            )
            if args.fail_fast:
                raise
        rebuild_manifests(args.output_dir)
    mark_complete(cases, args)
    print(f"Manifest: {args.output_dir / 'mass_effect_manifest.csv'}")
    if failures:
        print(f"{failures} case(s) failed", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired validation of the ARC lesion-inpainting/BrainSuite pipeline.

The script compares each T1 acquisition with itself under two processing
conditions:

1. direct BrainSuite processing of the original T1 (``raw``), and
2. BrainSuite processing after lesion inpainting (``inpainted``).

Because the ARC archive does not contain pre-stroke T1 images or expert
BrainSuite labels, the analysis deliberately avoids treating either branch as
ground truth. More importantly, generated tissue is only a computational
scaffold: no intensity or morphometric measurement from an ROI touching the
dilated inpainting target is used as a biological endpoint. Homologous
left/right BrainSuite regions entirely outside that target provide the internal
anatomical control.

Outputs are CSV/JSON tables, a publication-ready figure, and a small LaTeX
file containing the exact values used by the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy import stats


DEFAULT_ARC_ROOT = Path("/home/ajoshi/project2_ajoshi_1183/data/ARC")
STATS_SUFFIX = ".roiwise.stats.txt"
INPAINTED_SUFFIX = "_inpainted_mni_1mm"
CASE_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_.+_T1w$")

# A priori bilateral perisylvian language-network regions. BrainSuite uses an
# even ROI ID for the right hemisphere and the following odd ID for the left.
# The volumetric label image encodes cortical GM as 1000 + ROI_ID and cortical
# WM as 2000 + ROI_ID; ``atlas_roi_id`` below removes that tissue prefix.
LANGUAGE_RIGHT_ROIS = frozenset({142, 144, 146, 224, 226, 322, 324, 326, 500})
LANGUAGE_LEFT_ROIS = frozenset(roi_id + 1 for roi_id in LANGUAGE_RIGHT_ROIS)


@dataclass(frozen=True)
class Endpoint:
    key: str
    before_column: str
    after_column: str
    label: str
    unit: str
    manuscript_prefix: str


ENDPOINTS = (
    Endpoint(
        "spared_volume_asymmetry",
        "spared_volume_asymmetry_raw_pct",
        "spared_volume_asymmetry_inpainted_pct",
        "Spared-ROI tissue-volume asymmetry",
        "%",
        "ArcSparedVolume",
    ),
    Endpoint(
        "spared_area_asymmetry",
        "spared_area_asymmetry_raw_pct",
        "spared_area_asymmetry_inpainted_pct",
        "Spared-ROI cortical-area asymmetry",
        "%",
        "ArcSparedArea",
    ),
    Endpoint(
        "spared_thickness_asymmetry",
        "spared_thickness_asymmetry_raw_pct",
        "spared_thickness_asymmetry_inpainted_pct",
        "Spared-ROI cortical-thickness asymmetry",
        "%",
        "ArcSparedThickness",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-root", type=Path, default=DEFAULT_ARC_ROOT)
    parser.add_argument(
        "--raw-derivative",
        type=Path,
        help="Default: ARC_ROOT/derivatives/brainsuite_anatomical_raw",
    )
    parser.add_argument(
        "--inpainted-derivative",
        type=Path,
        help="Default: ARC_ROOT/derivatives/brainsuite_anatomical_bidsapp",
    )
    parser.add_argument(
        "--inpainting-derivative",
        type=Path,
        help="Default: ARC_ROOT/derivatives/stroke_inpainting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/arc_inpainting"),
    )
    parser.add_argument(
        "--minimum-roi-overlap-mm3",
        type=int,
        default=50,
        help="Minimum lesion/ROI intersection used to mark a homologous pair affected.",
    )
    parser.add_argument(
        "--unilateral-threshold",
        type=float,
        default=0.80,
        help="Minimum lesion laterality index for the mirror-image endpoint.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def case_id_from_stats(path: Path) -> str:
    name = path.name
    if not name.endswith(STATS_SUFFIX):
        raise ValueError(f"Unexpected BrainSuite statistics name: {path}")
    case_id = name[: -len(STATS_SUFFIX)]
    if case_id.endswith(INPAINTED_SUFFIX):
        case_id = case_id[: -len(INPAINTED_SUFFIX)]
    return case_id


def discover_stats(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob("sub-*/ses-*/*/*.roiwise.stats.txt")):
        case_id = case_id_from_stats(path)
        if case_id in result:
            raise RuntimeError(
                f"More than one statistics file maps to {case_id}: "
                f"{result[case_id]} and {path}"
            )
        result[case_id] = path
    return result


def read_roi_stats(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", na_values=["NaN", "nan"])
    if "ROI_ID" not in frame:
        raise ValueError(f"ROI_ID column is absent from {path}")
    frame["ROI_ID"] = frame["ROI_ID"].astype(int)
    return frame.set_index("ROI_ID", verify_integrity=True)


def finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def percent_asymmetry(right: float, left: float) -> float:
    denominator = right + left
    if denominator <= 0:
        return math.nan
    return 200.0 * abs(right - left) / denominator


def subject_session(case_id: str) -> tuple[str, str]:
    match = CASE_RE.match(case_id)
    if not match:
        raise ValueError(f"Cannot parse subject/session from {case_id}")
    return match.group(1), match.group(2)


def inpainting_case_dir(root: Path, case_id: str) -> Path:
    subject, session = subject_session(case_id)
    return root / subject / session / case_id


def load_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    return image, np.asanyarray(image.dataobj) > 0


def same_geometry(first: nib.spatialimages.SpatialImage, second: nib.spatialimages.SpatialImage) -> bool:
    return first.shape == second.shape and np.allclose(first.affine, second.affine, atol=1e-4)


def lesion_characteristics(
    lesion_image: nib.Nifti1Image, lesion: np.ndarray
) -> tuple[float, int, int, float, str]:
    indices = np.argwhere(lesion)
    voxel_volume = float(abs(np.linalg.det(lesion_image.affine[:3, :3])))
    volume_ml = float(indices.shape[0] * voxel_volume / 1000.0)
    if indices.size == 0:
        return volume_ml, 0, 0, math.nan, "none"
    x_world = nib.affines.apply_affine(lesion_image.affine, indices)[:, 0]
    left = int(np.count_nonzero(x_world < 0))
    right = int(np.count_nonzero(x_world > 0))
    lateral_total = left + right
    laterality = abs(left - right) / lateral_total if lateral_total else math.nan
    side = "left" if left > right else "right" if right > left else "bilateral"
    return volume_ml, left, right, float(laterality), side


def mirror_axis_is_symmetric(image: nib.Nifti1Image) -> bool:
    """Return True when array-axis 0 can be mirrored across world x=0."""
    n = image.shape[0]
    points = np.array([[0, 0, 0], [n - 1, 0, 0], [(n - 1) / 2, 0, 0]])
    world = nib.affines.apply_affine(image.affine, points)
    changes_only_in_x = np.allclose(world[0, 1:], world[1, 1:], atol=1e-4)
    return bool(
        changes_only_in_x
        and np.isclose(world[0, 0], -world[1, 0], atol=1e-4)
        and np.isclose(world[2, 0], 0, atol=1e-4)
    )


def atlas_roi_id(label_id: int) -> int | None:
    """Map a BrainSuite volumetric tissue label to its ROI-wise statistics ID."""
    label_id = int(label_id)
    if 1100 <= label_id < 1600:
        return label_id - 1000
    if 2100 <= label_id < 2600:
        return label_id - 2000
    if 600 <= label_id < 700:
        return label_id
    return None


def roi_voxel_counts(labels: np.ndarray, mask: np.ndarray | None = None) -> dict[int, int]:
    values = labels if mask is None else labels[mask]
    label_ids, counts = np.unique(values, return_counts=True)
    result: dict[int, int] = {}
    for label_id, count in zip(label_ids, counts, strict=True):
        roi_id = atlas_roi_id(int(label_id))
        if roi_id is not None:
            result[roi_id] = result.get(roi_id, 0) + int(count)
    return result


def language_network_burden(
    labels: np.ndarray,
    lesion: np.ndarray,
) -> dict[str, float]:
    lesion_counts = roi_voxel_counts(labels, lesion)

    def burden(roi_ids: frozenset[int]) -> float:
        lesion_voxels = sum(lesion_counts.get(roi_id, 0) for roi_id in roi_ids)
        return lesion_voxels / 1000.0

    left_ml = burden(LANGUAGE_LEFT_ROIS)
    right_ml = burden(LANGUAGE_RIGHT_ROIS)
    return {
        "left_language_lesion_ml": left_ml,
        "right_language_lesion_ml": right_ml,
        "lateralized_language_lesion_ml": left_ml - right_ml,
    }


def image_symmetry_metrics(
    case_dir: Path,
    case_id: str,
    lesion_image: nib.Nifti1Image,
    lesion: np.ndarray,
    laterality: float,
    dominant_side: str,
    unilateral_threshold: float,
) -> tuple[float, float, int]:
    if not np.isfinite(laterality) or laterality < unilateral_threshold:
        return math.nan, math.nan, 0
    if dominant_side not in {"left", "right"} or not mirror_axis_is_symmetric(lesion_image):
        return math.nan, math.nan, 0

    raw_path = case_dir / f"{case_id}_brain_mni_1mm.nii.gz"
    inpainted_path = case_dir / f"{case_id}_brain_inpainted_mni_1mm.nii.gz"
    brain_mask_path = case_dir / f"{case_id}_skullstrip_mask_mni_1mm.nii.gz"
    if not (raw_path.is_file() and inpainted_path.is_file() and brain_mask_path.is_file()):
        return math.nan, math.nan, 0

    raw_image = nib.load(raw_path)
    inpainted_image = nib.load(inpainted_path)
    brain_image, brain = load_mask(brain_mask_path)
    if not (
        same_geometry(lesion_image, raw_image)
        and same_geometry(lesion_image, inpainted_image)
        and same_geometry(lesion_image, brain_image)
    ):
        return math.nan, math.nan, 0

    raw = np.asanyarray(raw_image.dataobj).astype(np.float32, copy=False)
    inpainted = np.asanyarray(inpainted_image.dataobj).astype(np.float32, copy=False)
    x_mid = (raw.shape[0] - 1) / 2.0
    x_indices = np.arange(raw.shape[0], dtype=float)[:, None, None]
    side_mask = x_indices < x_mid if dominant_side == "left" else x_indices > x_mid

    # The contralateral reference voxel must be brain and must not itself lie
    # in a predicted lesion. This protects the internal control in multifocal
    # and weakly bilateral cases.
    target = lesion & side_mask & brain & np.flip(brain, axis=0) & ~np.flip(lesion, axis=0)
    coords = np.where(target)
    valid_count = int(coords[0].size)
    if valid_count < 100:
        return math.nan, math.nan, valid_count
    mirror_coords = (raw.shape[0] - 1 - coords[0], coords[1], coords[2])

    brain_values = raw[brain & np.isfinite(raw)]
    if brain_values.size == 0:
        return math.nan, math.nan, valid_count
    p05, p95 = np.percentile(brain_values, [5, 95])
    robust_range = float(p95 - p05)
    if robust_range <= 0:
        return math.nan, math.nan, valid_count

    raw_error = np.median(np.abs(raw[coords] - raw[mirror_coords]))
    inpainted_error = np.median(
        np.abs(inpainted[coords] - inpainted[mirror_coords])
    )
    return (
        float(100.0 * raw_error / robust_range),
        float(100.0 * inpainted_error / robust_range),
        valid_count,
    )


def roi_pair_metrics(
    raw_stats: pd.DataFrame,
    inpainted_stats: pd.DataFrame,
    labels: np.ndarray,
    dilated_lesion: np.ndarray,
    minimum_overlap: int,
) -> list[dict[str, object]]:
    overlaps = roi_voxel_counts(labels, dilated_lesion)

    available = set(raw_stats.index) & set(inpainted_stats.index)
    paired_ids: dict[int, tuple[int, int]] = {}
    for roi_id in available:
        right_id = int(roi_id) if int(roi_id) % 2 == 0 else int(roi_id) - 1
        left_id = right_id + 1
        if right_id >= 100 and right_id in available and left_id in available:
            paired_ids[right_id] = (right_id, left_id)

    records: list[dict[str, object]] = []
    for pair_id, (right_id, left_id) in sorted(paired_ids.items()):
        right_overlap = overlaps.get(right_id, 0)
        left_overlap = overlaps.get(left_id, 0)
        total_overlap = right_overlap + left_overlap
        affected_side = "right" if right_overlap >= left_overlap else "left"
        record: dict[str, object] = {
            "roi_pair_id": pair_id,
            "right_roi_id": right_id,
            "left_roi_id": left_id,
            "affected_side": affected_side,
            "right_overlap_mm3": right_overlap,
            "left_overlap_mm3": left_overlap,
            "pair_overlap_mm3": total_overlap,
            "region_class": (
                "spared"
                if total_overlap == 0
                else "target_overlap"
                if total_overlap >= minimum_overlap
                else "target_margin"
            ),
        }
        for short_name, column in (
            ("volume", "Total_Volume(GM+WM)(mm^3)"),
            ("thickness", "Mean_Thickness(mm)"),
            ("area", "Cortical_Area_mid(mm^2)"),
        ):
            for condition, frame in (("raw", raw_stats), ("inpainted", inpainted_stats)):
                right_value = frame.at[right_id, column]
                left_value = frame.at[left_id, column]
                if finite_positive(right_value) and finite_positive(left_value):
                    asymmetry = percent_asymmetry(float(right_value), float(left_value))
                else:
                    asymmetry = math.nan
                record[f"{short_name}_asymmetry_{condition}_pct"] = asymmetry
                record[f"{short_name}_right_{condition}"] = float(right_value)
                record[f"{short_name}_left_{condition}"] = float(left_value)
        records.append(record)
    return records


def iqr(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return float(median), float(q1), float(q3)


def bootstrap_median_difference(
    before: np.ndarray,
    after: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    differences = before - after
    count = differences.size
    if count == 0:
        return math.nan, math.nan
    bootstrap = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = rng.integers(0, count, size=count)
        bootstrap[iteration] = np.median(differences[sample])
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(low), float(high)


def rank_biserial_improvement(before: np.ndarray, after: np.ndarray) -> float:
    differences = before - after
    differences = differences[~np.isclose(differences, 0)]
    if differences.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_endpoint_summary(
    cases: pd.DataFrame,
    endpoint: Endpoint,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    paired = cases[[endpoint.before_column, endpoint.after_column]].dropna()
    before = paired[endpoint.before_column].to_numpy(dtype=float)
    after = paired[endpoint.after_column].to_numpy(dtype=float)
    before_median, before_q1, before_q3 = iqr(before)
    after_median, after_q1, after_q3 = iqr(after)
    difference = before - after
    difference_median, difference_q1, difference_q3 = iqr(difference)
    ci_low, ci_high = bootstrap_median_difference(before, after, iterations, rng)
    if before.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wilcoxon = stats.wilcoxon(
                before,
                after,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
        improved = int(np.count_nonzero(after < before))
        worsened = int(np.count_nonzero(after > before))
        ties = int(before.size - improved - worsened)
        non_ties = improved + worsened
        sign_p = (
            float(stats.binomtest(improved, non_ties, 0.5, alternative="two-sided").pvalue)
            if non_ties
            else 1.0
        )
        wilcoxon_statistic = float(wilcoxon.statistic)
        p_value = float(wilcoxon.pvalue)
    else:
        improved = worsened = ties = 0
        sign_p = wilcoxon_statistic = p_value = math.nan
    return {
        "endpoint": endpoint.key,
        "label": endpoint.label,
        "unit": endpoint.unit,
        "n": int(before.size),
        "before_median": before_median,
        "before_q1": before_q1,
        "before_q3": before_q3,
        "after_median": after_median,
        "after_q1": after_q1,
        "after_q3": after_q3,
        "paired_improvement_median": difference_median,
        "paired_improvement_q1": difference_q1,
        "paired_improvement_q3": difference_q3,
        "paired_improvement_ci95_low": ci_low,
        "paired_improvement_ci95_high": ci_high,
        "improved_n": improved,
        "worsened_n": worsened,
        "ties_n": ties,
        "improved_pct": 100.0 * improved / before.size if before.size else math.nan,
        "wilcoxon_statistic": wilcoxon_statistic,
        "p_value": p_value,
        "rank_biserial_improvement": rank_biserial_improvement(before, after),
        "sign_test_p_value": sign_p,
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p))
    if finite_indices.size == 0:
        return adjusted
    order = finite_indices[np.argsort(p[finite_indices])]
    running_max = 0.0
    count = order.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def partial_spearman(
    predictor: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
) -> tuple[float, float]:
    """Partial Spearman correlation using residualized ranks."""
    predictor_rank = stats.rankdata(predictor, method="average")
    outcome_rank = stats.rankdata(outcome, method="average")
    covariate_ranks = np.column_stack(
        [stats.rankdata(covariates[:, column], method="average") for column in range(covariates.shape[1])]
    )
    design = np.column_stack([np.ones(predictor.size), covariate_ranks])
    predictor_residual = predictor_rank - design @ np.linalg.lstsq(
        design, predictor_rank, rcond=None
    )[0]
    outcome_residual = outcome_rank - design @ np.linalg.lstsq(
        design, outcome_rank, rcond=None
    )[0]
    if np.isclose(np.std(predictor_residual), 0) or np.isclose(np.std(outcome_residual), 0):
        return math.nan, math.nan
    rho = float(np.corrcoef(predictor_residual, outcome_residual)[0, 1])
    degrees_freedom = predictor.size - covariates.shape[1] - 2
    if degrees_freedom <= 0 or abs(rho) >= 1:
        p_value = 0.0 if abs(rho) >= 1 else math.nan
    else:
        statistic = rho * math.sqrt(degrees_freedom / max(1e-15, 1 - rho**2))
        p_value = float(2 * stats.t.sf(abs(statistic), degrees_freedom))
    return rho, p_value


def clinical_association_summary(
    cases: pd.DataFrame,
    predictor: str,
    label: str,
    adjust_for_total_lesion_volume: bool,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    columns = [predictor, "wab_aq", "age_at_stroke", "wab_days"]
    if adjust_for_total_lesion_volume and predictor != "lesion_volume_ml":
        columns.append("lesion_volume_ml")
    frame = cases[columns].replace([np.inf, -np.inf], np.nan).dropna()
    x = frame[predictor].to_numpy(dtype=float)
    y = frame["wab_aq"].to_numpy(dtype=float)
    covariate_values = [
        frame["age_at_stroke"].to_numpy(dtype=float),
        np.log1p(frame["wab_days"].to_numpy(dtype=float)),
    ]
    if adjust_for_total_lesion_volume and predictor != "lesion_volume_ml":
        covariate_values.append(frame["lesion_volume_ml"].to_numpy(dtype=float))
    covariates = np.column_stack(covariate_values)
    rho, p_value = partial_spearman(x, y, covariates)
    bootstrap = []
    for _ in range(iterations):
        sample = rng.integers(0, x.size, size=x.size)
        sample_rho, _ = partial_spearman(x[sample], y[sample], covariates[sample])
        if np.isfinite(sample_rho):
            bootstrap.append(sample_rho)
    ci_low, ci_high = (
        np.percentile(bootstrap, [2.5, 97.5]) if bootstrap else (math.nan, math.nan)
    )
    return {
        "predictor": predictor,
        "label": label,
        "adjusted_for": (
            "age_at_stroke, log1p(wab_days), total_lesion_volume"
            if adjust_for_total_lesion_volume and predictor != "lesion_volume_ml"
            else "age_at_stroke, log1p(wab_days)"
        ),
        "n": int(x.size),
        "partial_spearman_rho": rho,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
    }


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return "$<0.001$" if value < 0.001 else f"${value:.3f}$"


def number(value: float, digits: int = 1) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def latex_macros(
    cases: pd.DataFrame,
    summaries: pd.DataFrame,
    matched_count: int,
    output_path: Path,
    unilateral_threshold: float,
    minimum_overlap: int,
    clinical: pd.DataFrame,
) -> None:
    lesion_median, lesion_q1, lesion_q3 = iqr(cases["lesion_volume_ml"].to_numpy())
    lesion_sides = cases["dominant_lesion_side"].value_counts()
    lines = [
        "% Generated by analyze_arc_inpainting.py; do not edit by hand.",
        f"\\newcommand{{\\ArcNMatched}}{{{matched_count}}}",
        f"\\newcommand{{\\ArcNAnalyzed}}{{{len(cases)}}}",
        f"\\newcommand{{\\ArcLesionMedian}}{{{number(lesion_median)}}}",
        f"\\newcommand{{\\ArcLesionQOne}}{{{number(lesion_q1)}}}",
        f"\\newcommand{{\\ArcLesionQThree}}{{{number(lesion_q3)}}}",
        f"\\newcommand{{\\ArcLeftDominant}}{{{int(lesion_sides.get('left', 0))}}}",
        f"\\newcommand{{\\ArcRightDominant}}{{{int(lesion_sides.get('right', 0))}}}",
        f"\\newcommand{{\\ArcUnilateralThreshold}}{{{unilateral_threshold:.2f}}}",
        f"\\newcommand{{\\ArcAuditOverlapThreshold}}{{{minimum_overlap}}}",
    ]
    summary_by_key = summaries.set_index("endpoint")
    for endpoint in ENDPOINTS:
        row = summary_by_key.loc[endpoint.key]
        prefix = endpoint.manuscript_prefix
        lines.extend(
            [
                f"\\newcommand{{\\{prefix}N}}{{{int(row['n'])}}}",
                f"\\newcommand{{\\{prefix}BeforeMedian}}{{{number(row['before_median'])}}}",
                f"\\newcommand{{\\{prefix}BeforeQOne}}{{{number(row['before_q1'])}}}",
                f"\\newcommand{{\\{prefix}BeforeQThree}}{{{number(row['before_q3'])}}}",
                f"\\newcommand{{\\{prefix}AfterMedian}}{{{number(row['after_median'])}}}",
                f"\\newcommand{{\\{prefix}AfterQOne}}{{{number(row['after_q1'])}}}",
                f"\\newcommand{{\\{prefix}AfterQThree}}{{{number(row['after_q3'])}}}",
                f"\\newcommand{{\\{prefix}DeltaMedian}}{{{number(row['paired_improvement_median'])}}}",
                f"\\newcommand{{\\{prefix}DeltaCILow}}{{{number(row['paired_improvement_ci95_low'], 2)}}}",
                f"\\newcommand{{\\{prefix}DeltaCIHigh}}{{{number(row['paired_improvement_ci95_high'], 2)}}}",
                f"\\newcommand{{\\{prefix}ImprovedN}}{{{int(row['improved_n'])}}}",
                f"\\newcommand{{\\{prefix}ImprovedPct}}{{{number(row['improved_pct'])}}}",
                f"\\newcommand{{\\{prefix}Effect}}{{{number(row['rank_biserial_improvement'], 2)}}}",
                f"\\newcommand{{\\{prefix}P}}{{{p_text(float(row['p_holm']))}}}",
            ]
        )
    clinical_by_predictor = clinical.set_index("predictor")
    for predictor, prefix in (
        ("left_language_lesion_ml", "ArcClinicalLeft"),
        ("right_language_lesion_ml", "ArcClinicalRight"),
        ("lesion_volume_ml", "ArcClinicalVolume"),
    ):
        row = clinical_by_predictor.loc[predictor]
        lines.extend(
            [
                f"\\newcommand{{\\{prefix}N}}{{{int(row['n'])}}}",
                f"\\newcommand{{\\{prefix}Rho}}{{{number(row['partial_spearman_rho'], 2)}}}",
                f"\\newcommand{{\\{prefix}CILow}}{{{number(row['ci95_low'], 2)}}}",
                f"\\newcommand{{\\{prefix}CIHigh}}{{{number(row['ci95_high'], 2)}}}",
                f"\\newcommand{{\\{prefix}P}}{{{p_text(float(row['p_holm']))}}}",
            ]
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def add_identity_scatter(
    axis: plt.Axes,
    before: np.ndarray,
    after: np.ndarray,
    title: str,
    unit: str,
    summary: pd.Series,
    before_label: str = "Direct/original",
    after_label: str = "Inpainted",
) -> None:
    finite = np.isfinite(before) & np.isfinite(after)
    before = before[finite]
    after = after[finite]
    if before.size == 0:
        axis.text(0.5, 0.5, "No valid pairs", ha="center", va="center")
        return
    combined = np.concatenate([before, after])
    low = min(0.0, float(np.percentile(combined, 0.5)))
    high = float(np.percentile(combined, 99.5))
    if high <= low:
        high = low + 1
    padding = 0.05 * (high - low)
    limit = (low - padding, high + padding)
    axis.scatter(
        before,
        after,
        s=16,
        alpha=0.55,
        color="#1874A5",
        edgecolors="white",
        linewidths=0.25,
    )
    axis.plot(limit, limit, linestyle="--", color="#555555", linewidth=1)
    axis.set(xlim=limit, ylim=limit, xlabel=f"{before_label} ({unit})", ylabel=f"{after_label} ({unit})")
    axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
    axis.text(
        0.04,
        0.96,
        f"n={int(summary['n'])}; improved={summary['improved_pct']:.0f}%\n"
        f"Holm p={summary['p_holm']:.3g}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    axis.grid(color="#dddddd", linewidth=0.5, alpha=0.8)
    axis.set_aspect("equal", adjustable="box")


def add_clinical_scatter(
    axis: plt.Axes,
    cases: pd.DataFrame,
    clinical: pd.DataFrame,
) -> None:
    frame = cases[["left_language_lesion_ml", "wab_aq"]].dropna()
    x = frame["left_language_lesion_ml"].to_numpy(dtype=float)
    y = frame["wab_aq"].to_numpy(dtype=float)
    axis.scatter(
        x,
        y,
        s=18,
        alpha=0.6,
        color="#9B3A4D",
        edgecolors="white",
        linewidths=0.25,
    )
    if np.unique(x).size > 1:
        coefficients = np.polyfit(x, y, 1)
        grid = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(grid, np.polyval(coefficients, grid), color="#6A2030", linewidth=1.5)
    row = clinical.set_index("predictor").loc["left_language_lesion_ml"]
    axis.set_title("D  Aphasia construct validity", loc="left", fontweight="bold", fontsize=10)
    axis.set_xlabel("Left language-network lesion volume (mL)")
    axis.set_ylabel("WAB Aphasia Quotient")
    axis.text(
        0.96,
        0.96,
        f"n={int(row['n'])}; partial $\\rho$={row['partial_spearman_rho']:.2f}\n"
        f"Holm p={row['p_holm']:.3g}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    axis.grid(color="#dddddd", linewidth=0.5, alpha=0.8)


def make_figure(
    cases: pd.DataFrame,
    summaries: pd.DataFrame,
    clinical: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary_by_key = summaries.set_index("endpoint")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), constrained_layout=True)
    for letter, axis, endpoint in zip("ABC", axes.flat[:3], ENDPOINTS, strict=True):
        short_label = endpoint.label.removeprefix("Spared-ROI ")
        add_identity_scatter(
            axis,
            cases[endpoint.before_column].to_numpy(dtype=float),
            cases[endpoint.after_column].to_numpy(dtype=float),
            f"{letter}  {short_label}\n     (spared ROI pairs)",
            endpoint.unit,
            summary_by_key.loc[endpoint.key],
            "Direct/original",
            "Inpainted workflow",
        )
    add_clinical_scatter(axes.flat[3], cases, clinical)
    figure.suptitle(
        "Spared-region ARC validation (target-overlapping ROIs excluded)",
        fontsize=11,
        fontweight="bold",
    )
    figure.savefig(output_dir / "arc_inpainting_validation.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "arc_inpainting_validation.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    derivatives = args.arc_root / "derivatives"
    raw_root = args.raw_derivative or derivatives / "brainsuite_anatomical_raw"
    inpainted_root = (
        args.inpainted_derivative or derivatives / "brainsuite_anatomical_bidsapp"
    )
    inpainting_root = args.inpainting_derivative or derivatives / "stroke_inpainting"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (raw_root, inpainted_root, inpainting_root):
        if not required.is_dir():
            raise FileNotFoundError(required)

    raw_files = discover_stats(raw_root)
    inpainted_files = discover_stats(inpainted_root)
    matched_ids = sorted(set(raw_files) & set(inpainted_files))
    if not matched_ids:
        raise RuntimeError("No matched raw/inpainted BrainSuite statistics were found")

    print(
        f"Discovered {len(raw_files)} direct and {len(inpainted_files)} inpainted "
        f"BrainSuite results; {len(matched_ids)} matched acquisitions."
    )
    case_records: list[dict[str, object]] = []
    roi_records: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for index, case_id in enumerate(matched_ids, start=1):
        if index == 1 or index % 25 == 0 or index == len(matched_ids):
            print(f"[{index:3d}/{len(matched_ids)}] {case_id}", flush=True)
        case_dir = inpainting_case_dir(inpainting_root, case_id)
        lesion_path = case_dir / f"{case_id}_stroke_mask_mni_1mm.nii.gz"
        dilated_path = case_dir / f"{case_id}_stroke_mask_dilated_3mm_mni_1mm.nii.gz"
        metadata_path = case_dir / "processing_metadata.json"
        inpainted_prefix = inpainted_files[case_id].name[: -len(STATS_SUFFIX)]
        label_path = inpainted_files[case_id].with_name(
            f"{inpainted_prefix}.svreg.label.nii.gz"
        )
        required = (lesion_path, dilated_path, metadata_path, label_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            skipped.append({"case_id": case_id, "reason": f"missing: {'; '.join(missing)}"})
            continue
        try:
            lesion_image, lesion = load_mask(lesion_path)
            dilated_image, dilated_lesion = load_mask(dilated_path)
            if not same_geometry(lesion_image, dilated_image):
                raise ValueError("source and dilated lesion masks have different geometry")
            volume_ml, left_voxels, right_voxels, laterality, side = lesion_characteristics(
                lesion_image, lesion
            )
            raw_stats = read_roi_stats(raw_files[case_id])
            inpainted_stats = read_roi_stats(inpainted_files[case_id])
            label_image = nib.load(label_path)
            if not same_geometry(lesion_image, label_image):
                raise ValueError("lesion and inpainted BrainSuite labels have different geometry")
            labels = np.asanyarray(label_image.dataobj).astype(np.int32, copy=False)
            language_burden = language_network_burden(labels, lesion)
            pairs = roi_pair_metrics(
                raw_stats,
                inpainted_stats,
                labels,
                dilated_lesion,
                args.minimum_roi_overlap_mm3,
            )
            for record in pairs:
                record.update({"case_id": case_id, "subject": subject_session(case_id)[0]})
            roi_records.extend(pairs)
            roi_frame = pd.DataFrame(pairs)
            spared_frame = roi_frame.loc[roi_frame["region_class"] == "spared"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            inpainting = metadata.get("inpainting", {})

            case_record: dict[str, object] = {
                "case_id": case_id,
                "subject": subject_session(case_id)[0],
                "session": subject_session(case_id)[1],
                "lesion_volume_ml": volume_ml,
                "lesion_left_voxels": left_voxels,
                "lesion_right_voxels": right_voxels,
                "lesion_laterality_index": laterality,
                "dominant_lesion_side": side,
                "spared_roi_pairs": int((roi_frame["region_class"] == "spared").sum()),
                "target_overlapping_roi_pairs": int(
                    (roi_frame["region_class"] == "target_overlap").sum()
                ),
                "boundary_error_before": inpainting.get(
                    "full_boundary_median_abs_error_before", math.nan
                ),
                "boundary_error_after": inpainting.get(
                    "full_boundary_median_abs_error_after", math.nan
                ),
                "outside_lesion_max_abs_change": inpainting.get(
                    "full_outside_lesion_max_abs_change", math.nan
                ),
            }
            case_record.update(language_burden)
            for metric in ("volume", "thickness", "area"):
                for condition in ("raw", "inpainted"):
                    column = f"{metric}_asymmetry_{condition}_pct"
                    values = (
                        pd.to_numeric(spared_frame[column], errors="coerce").dropna()
                        if column in spared_frame
                        else pd.Series(dtype=float)
                    )
                    case_record[f"spared_{column}"] = (
                        float(values.median()) if not values.empty else math.nan
                    )
            case_records.append(case_record)
        except Exception as error:  # preserve a complete audit trail
            skipped.append({"case_id": case_id, "reason": f"{type(error).__name__}: {error}"})

    cases = pd.DataFrame(case_records).sort_values("case_id").reset_index(drop=True)
    rois = pd.DataFrame(roi_records)
    if cases.empty:
        raise RuntimeError("Every matched case was skipped; inspect skipped_cases.csv")
    cases.to_csv(output_dir / "case_metrics.csv", index=False)
    rois.to_csv(output_dir / "roi_pair_metrics.csv", index=False)
    rois.loc[rois["region_class"] == "spared"].to_csv(
        output_dir / "spared_roi_metrics.csv", index=False
    )
    # Retain an explicit audit of excluded target-overlapping pairs so older
    # output directories cannot be mistaken for the analysis population.
    rois.loc[rois["region_class"] != "spared"].to_csv(
        output_dir / "affected_roi_metrics.csv", index=False
    )
    pd.DataFrame(skipped, columns=["case_id", "reason"]).to_csv(
        output_dir / "skipped_cases.csv", index=False
    )

    rng = np.random.default_rng(args.seed)
    summaries = pd.DataFrame(
        [
            paired_endpoint_summary(
                cases, endpoint, args.bootstrap_iterations, rng
            )
            for endpoint in ENDPOINTS
        ]
    )
    summaries["p_holm"] = holm_adjust(summaries["p_value"])
    summaries.to_csv(output_dir / "summary_statistics.csv", index=False)

    participants_path = args.arc_root / "participants.tsv"
    if not participants_path.is_file():
        raise FileNotFoundError(participants_path)
    participants = pd.read_csv(participants_path, sep="\t").rename(
        columns={"participant_id": "subject"}
    )
    if participants["subject"].duplicated().any():
        raise ValueError("participants.tsv contains duplicate participant_id values")
    cases = cases.drop(columns=[column for column in participants.columns if column in cases and column != "subject"])
    cases = cases.merge(participants, on="subject", how="left", validate="many_to_one")
    cases.to_csv(output_dir / "case_metrics.csv", index=False)
    clinical_specs = (
        (
            "left_language_lesion_ml",
            "Left perisylvian language-network lesion volume",
            True,
        ),
        (
            "right_language_lesion_ml",
            "Right homolog lesion volume (negative control)",
            True,
        ),
        ("lesion_volume_ml", "Total lesion volume", False),
    )
    clinical = pd.DataFrame(
        [
            clinical_association_summary(
                cases,
                predictor,
                label,
                adjust_for_total,
                args.bootstrap_iterations,
                rng,
            )
            for predictor, label, adjust_for_total in clinical_specs
        ]
    )
    clinical["p_holm"] = holm_adjust(clinical["p_value"])
    clinical.to_csv(output_dir / "clinical_associations.csv", index=False)

    outside = pd.to_numeric(cases["outside_lesion_max_abs_change"], errors="coerce")
    audit = {
        "arc_root": str(args.arc_root.resolve()),
        "raw_derivative": str(raw_root.resolve()),
        "inpainted_derivative": str(inpainted_root.resolve()),
        "inpainting_derivative": str(inpainting_root.resolve()),
        "raw_stats_count": len(raw_files),
        "inpainted_stats_count": len(inpainted_files),
        "matched_count": len(matched_ids),
        "analyzed_count": len(cases),
        "skipped_count": len(skipped),
        # This threshold only subdivides the excluded audit rows. The analysis
        # definition of "spared" is exact: pair overlap must equal zero.
        "target_overlap_audit_threshold_mm3": args.minimum_roi_overlap_mm3,
        "unilateral_threshold": args.unilateral_threshold,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "outside_target_exactly_unchanged_count": int(np.count_nonzero(outside == 0)),
        "outside_target_available_count": int(outside.notna().sum()),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
        },
        "endpoints": summaries.to_dict(orient="records"),
        "clinical_associations": clinical.to_dict(orient="records"),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(audit, indent=2, allow_nan=True), encoding="utf-8"
    )
    latex_macros(
        cases,
        summaries,
        len(matched_ids),
        output_dir / "paper_results.tex",
        args.unilateral_threshold,
        args.minimum_roi_overlap_mm3,
        clinical,
    )
    make_figure(cases, summaries, clinical, output_dir)

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\nPaired endpoint summary (positive effect favors inpainting):")
        print(
            summaries[
                [
                    "endpoint",
                    "n",
                    "before_median",
                    "after_median",
                    "paired_improvement_median",
                    "paired_improvement_ci95_low",
                    "paired_improvement_ci95_high",
                    "improved_pct",
                    "rank_biserial_improvement",
                    "p_value",
                    "p_holm",
                ]
            ].to_string(index=False)
        )
        print("\nAphasia construct validity (partial Spearman; age and log-days adjusted):")
        print(clinical.to_string(index=False))
    print(f"\nWrote analysis products to {output_dir}")


if __name__ == "__main__":
    main()

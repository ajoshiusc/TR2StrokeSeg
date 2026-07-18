#!/usr/bin/env python3
"""Export ARC stroke probabilities and predictive uncertainty from nnU-Net.

The input is the 1 mm MNI T1 derivative produced by the ARC stroke/inpainting
pipeline.  For each acquisition this program writes:

* the lesion probability ``p_lesion`` (not a statistical hypothesis-test
  p-value);
* normalized binary predictive entropy, in [0, 1];
* the thresholded lesion mask;
* scalar probability/uncertainty features for downstream WAB Aphasia Quotient
  prediction; and
* an optional three-panel quality-control PNG.

The program is safe to run concurrently for different subjects.  Each case has
its own output directory and updates to the global CSV manifest are protected
by an advisory file lock.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CASE_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_.+_T1w$")
MNI_SUFFIX = "_mni_1mm.nii.gz"


@dataclass(frozen=True)
class Case:
    case_id: str
    subject: str
    session: str
    mni_t1: Path


@dataclass(frozen=True)
class CaseOutputs:
    case_dir: Path
    lesion_probability: Path
    predictive_entropy: Path
    lesion_mask: Path
    metrics_json: Path
    qc_png: Path
    error_json: Path


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def environment_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def default_mni_root() -> Path:
    arc_root = environment_path("ARC_ROOT")
    configured = environment_path("STROKE_INPAINT_OUTPUT_DIR")
    return first_existing(
        [
            configured,
            arc_root / "derivatives" / "stroke_inpainting" if arc_root else None,
            Path("/project2/ajoshi_1183/data/ARC/derivatives/stroke_inpainting"),
            Path("/home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/stroke_inpainting"),
            HERE / "outputs" / "subjectwise_stroke_inpainting.old",
        ]
    ) or Path("/project2/ajoshi_1183/data/ARC/derivatives/stroke_inpainting")


def default_nnunet_results() -> Path | None:
    return first_existing(
        [
            environment_path("nnUNet_results", "NNUNET_RESULTS"),
            Path("/project2/ajoshi_1183/data/TR2/nnUNet_results"),
            Path("/home/ajoshi/project2_ajoshi_1183/data/TR2/nnUNet_results"),
            Path("/home/ajoshi/Projects/TR2preproc/supp_data/models/nnUNet_results"),
        ]
    )


def default_nnunet_predict() -> Path | None:
    configured = environment_path("NNUNET_PREDICT")
    if configured and configured.is_file():
        return configured
    local = REPO_ROOT / ".venv" / "bin" / "nnUNetv2_predict"
    if local.is_file():
        return local
    executable = shutil.which("nnUNetv2_predict")
    return Path(executable) if executable else None


def parse_args() -> argparse.Namespace:
    default_input = default_mni_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mni-root",
        type=Path,
        default=default_input,
        help="Root containing <case-id>/<case-id>_mni_1mm.nii.gz derivatives",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: sibling ARC derivative named lesion_uncertainty",
    )
    parser.add_argument(
        "--subject", action="append", default=[], help="Select a sub-* ID; repeatable"
    )
    parser.add_argument(
        "--case", action="append", default=[], help="Select a complete case ID; repeatable"
    )
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--nnunet-results", type=Path, default=default_nnunet_results())
    parser.add_argument("--nnunet-predict", type=Path, default=default_nnunet_predict())
    parser.add_argument("--dataset-id", default="Dataset001_Atlas2")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--folds", nargs="+", default=["0"])
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument(
        "--lesion-class-index",
        type=int,
        default=1,
        help="Class channel in nnU-Net probabilities (ATLAS2 stroke lesion: 1)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Probability threshold for hard mask"
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.05,
        help="Minimum p_lesion included in regional uncertainty summaries/QC",
    )
    parser.add_argument(
        "--make-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a three-panel probability/uncertainty PNG",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def case_entities(case_id: str) -> tuple[str, str]:
    match = CASE_RE.match(case_id)
    if match is None:
        raise ValueError(f"Cannot parse BIDS subject/session from case ID: {case_id}")
    return match.group(1), match.group(2)


def discover_cases(args: argparse.Namespace) -> list[Case]:
    selected_subjects = set(args.subject)
    selected_cases = set(args.case)
    discovered: dict[str, Case] = {}
    for path in sorted(args.mni_root.rglob(f"*{MNI_SUFFIX}")):
        case_id = path.name[: -len(MNI_SUFFIX)]
        # The full-head MNI image lives in a directory named after the case.
        # This excludes brain, mask, inpainted, and other *_mni_1mm derivatives.
        if path.parent.name != case_id:
            continue
        subject, session = case_entities(case_id)
        if selected_subjects and subject not in selected_subjects:
            continue
        if selected_cases and case_id not in selected_cases:
            continue
        if case_id in discovered:
            raise RuntimeError(
                f"Duplicate MNI input for {case_id}: {discovered[case_id].mni_t1} and {path}"
            )
        discovered[case_id] = Case(case_id, subject, session, path.resolve())

    cases = [discovered[key] for key in sorted(discovered)]
    if args.limit is not None:
        cases = cases[: args.limit]
    missing_subjects = selected_subjects - {case.subject for case in cases}
    missing_cases = selected_cases - {case.case_id for case in cases}
    if missing_subjects:
        raise ValueError(f"Requested subjects not found: {', '.join(sorted(missing_subjects))}")
    if missing_cases:
        raise ValueError(f"Requested cases not found: {', '.join(sorted(missing_cases))}")
    return cases


def outputs_for_case(case: Case, output_dir: Path) -> CaseOutputs:
    case_dir = output_dir / case.subject / case.session / case.case_id
    return CaseOutputs(
        case_dir=case_dir,
        lesion_probability=case_dir / f"{case.case_id}_lesion_probability_mni_1mm.nii.gz",
        predictive_entropy=case_dir / f"{case.case_id}_lesion_uncertainty_entropy_mni_1mm.nii.gz",
        lesion_mask=case_dir / f"{case.case_id}_stroke_mask_p50_mni_1mm.nii.gz",
        metrics_json=case_dir / "uncertainty_metrics.json",
        qc_png=case_dir / f"{case.case_id}_lesion_uncertainty_qc.png",
        error_json=case_dir / "processing_error.json",
    )


def outputs_complete(outputs: CaseOutputs, make_qc: bool) -> bool:
    required = [
        outputs.lesion_probability,
        outputs.predictive_entropy,
        outputs.lesion_mask,
        outputs.metrics_json,
    ]
    if make_qc:
        required.append(outputs.qc_png)
    return all(path.is_file() for path in required)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
            if total_gib >= 10:
                return "cuda"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def validate_args(args: argparse.Namespace) -> None:
    if not args.mni_root.is_dir():
        raise FileNotFoundError(f"--mni-root does not exist: {args.mni_root}")
    if args.nnunet_results is None or not args.nnunet_results.is_dir():
        raise FileNotFoundError(f"--nnunet-results does not exist: {args.nnunet_results}")
    if args.nnunet_predict is None or not args.nnunet_predict.is_file():
        raise FileNotFoundError(f"--nnunet-predict does not exist: {args.nnunet_predict}")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be strictly between 0 and 1")
    if not 0 <= args.candidate_threshold < args.threshold:
        raise ValueError("--candidate-threshold must be nonnegative and below --threshold")
    if args.lesion_class_index < 0:
        raise ValueError("--lesion-class-index must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


def build_env(args: argparse.Namespace, cache_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_results"] = str(args.nnunet_results)
    env["nnUNet_raw"] = str(cache_dir / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(cache_dir / "nnUNet_preprocessed")
    env["PATH"] = f"{args.nnunet_predict.parent}{os.pathsep}{env.get('PATH', '')}"
    for key in ("nnUNet_raw", "nnUNet_preprocessed"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_nnunet(
    cases: list[Case], args: argparse.Namespace, device: str, work_dir: Path
) -> Path:
    input_dir = work_dir / "input"
    prediction_dir = work_dir / "predictions"
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        destination = input_dir / f"{case.case_id}_0000.nii.gz"
        try:
            destination.symlink_to(case.mni_t1)
        except OSError:
            shutil.copy2(case.mni_t1, destination)

    command = [
        str(args.nnunet_predict),
        "-i",
        str(input_dir),
        "-o",
        str(prediction_dir),
        "-d",
        args.dataset_id,
        "-c",
        args.configuration,
        "-tr",
        args.trainer,
        "-f",
        *args.folds,
        "-chk",
        args.checkpoint,
        "--save_probabilities",
        "-device",
        device,
    ]
    if args.disable_tta or device == "cpu":
        command.append("--disable_tta")
    if args.disable_progress_bar:
        command.append("--disable_progress_bar")
    if device == "cpu":
        command.extend(["-npp", "1", "-nps", "1"])

    print(f"Running: {shlex.join(command)}", flush=True)
    if not args.dry_run:
        env = build_env(args, work_dir / "cache")
        subprocess.run(command, check=True, env=env)
    return prediction_dir


def align_probabilities(probabilities: np.ndarray, reference_shape: tuple[int, ...]) -> np.ndarray:
    if probabilities.ndim != 4:
        raise ValueError(f"Expected class-by-3D probabilities, got {probabilities.shape}")
    if probabilities.shape[1:] == reference_shape:
        return probabilities
    if probabilities.shape[1:] == tuple(reversed(reference_shape)):
        return probabilities.transpose(0, 3, 2, 1)
    raise ValueError(
        f"Probability shape {probabilities.shape[1:]} does not match reference {reference_shape}"
    )


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
    image.set_qform(reference.affine, int(reference.header["qform_code"]))
    image.set_sform(reference.affine, int(reference.header["sform_code"]))
    nib.save(image, path)


def write_json_atomic(path: Path, record: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def entropy_from_probability(probability: np.ndarray) -> np.ndarray:
    probability64 = np.asarray(probability, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probability64, eps, 1.0 - eps)
    entropy = -(
        clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped)
    )
    entropy[(probability64 <= 0.0) | (probability64 >= 1.0)] = 0.0
    return np.asarray(entropy, dtype=np.float32)


def volume_ml(mask_or_weight: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(np.sum(mask_or_weight, dtype=np.float64) * voxel_volume_mm3 / 1000.0)


def calculate_metrics(
    case: Case,
    reference: nib.spatialimages.SpatialImage,
    probability: np.ndarray,
    entropy: np.ndarray,
    threshold: float,
    candidate_threshold: float,
    outputs: CaseOutputs,
) -> dict[str, object]:
    voxel_volume_mm3 = float(abs(np.linalg.det(reference.affine[:3, :3])))
    hard = probability >= threshold
    candidate = probability >= candidate_threshold
    boundary = (probability >= 0.25) & (probability <= 0.75)
    indices = np.indices(reference.shape, sparse=True)[0]
    world_x = reference.affine[0, 0] * indices + reference.affine[0, 3]
    # MNI inputs are axis-aligned by construction. Refuse a silently incorrect
    # laterality feature if an unexpected oblique affine reaches this point.
    if not np.allclose(reference.affine[0, 1:3], 0.0, atol=1e-5):
        raise ValueError(f"{case.case_id}: oblique x-axis prevents laterality metrics")
    left = np.broadcast_to(world_x < 0, reference.shape)
    right = np.broadcast_to(world_x > 0, reference.shape)

    def mean_or_zero(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0

    metrics: dict[str, object] = {
        "case_id": case.case_id,
        "subject": case.subject,
        "session": case.session,
        "mni_t1": str(case.mni_t1),
        "shape": "x".join(str(value) for value in reference.shape),
        "orientation": "".join(nib.aff2axcodes(reference.affine)),
        "voxel_volume_mm3": voxel_volume_mm3,
        "probability_threshold": threshold,
        "candidate_probability_threshold": candidate_threshold,
        "maximum_lesion_probability": float(np.max(probability)),
        "expected_lesion_volume_ml": volume_ml(probability, voxel_volume_mm3),
        "expected_left_lesion_volume_ml": volume_ml(probability * left, voxel_volume_mm3),
        "expected_right_lesion_volume_ml": volume_ml(probability * right, voxel_volume_mm3),
        "hard_lesion_volume_ml": volume_ml(hard, voxel_volume_mm3),
        "hard_left_lesion_volume_ml": volume_ml(hard & left, voxel_volume_mm3),
        "hard_right_lesion_volume_ml": volume_ml(hard & right, voxel_volume_mm3),
        "entropy_mass_ml": volume_ml(entropy, voxel_volume_mm3),
        "left_entropy_mass_ml": volume_ml(entropy * left, voxel_volume_mm3),
        "right_entropy_mass_ml": volume_ml(entropy * right, voxel_volume_mm3),
        "candidate_mean_entropy": mean_or_zero(entropy[candidate]),
        "boundary_mean_entropy": mean_or_zero(entropy[boundary]),
        "high_uncertainty_volume_ml": volume_ml(entropy >= 0.8, voxel_volume_mm3),
        "mean_probability_in_hard_lesion": mean_or_zero(probability[hard]),
        "volume_p10_ml": volume_ml(probability >= 0.10, voxel_volume_mm3),
        "volume_p25_ml": volume_ml(probability >= 0.25, voxel_volume_mm3),
        "volume_p50_ml": volume_ml(probability >= 0.50, voxel_volume_mm3),
        "volume_p75_ml": volume_ml(probability >= 0.75, voxel_volume_mm3),
        "volume_p90_ml": volume_ml(probability >= 0.90, voxel_volume_mm3),
        "lesion_probability_path": str(outputs.lesion_probability),
        "predictive_entropy_path": str(outputs.predictive_entropy),
        "lesion_mask_path": str(outputs.lesion_mask),
        "qc_png": str(outputs.qc_png) if outputs.qc_png.is_file() else "",
        "uncertainty_definition": "normalized_binary_predictive_entropy_bits",
        "probability_is_calibrated": False,
    }
    return metrics


def robust_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    values = positive if positive.size else finite
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(values, [1, 99])
    if not np.isfinite((lower, upper)).all() or upper <= lower:
        return float(np.min(values)), float(np.max(values) + 1.0)
    return float(lower), float(upper)


def make_qc_figure(
    case: Case,
    reference: nib.spatialimages.SpatialImage,
    probability: np.ndarray,
    entropy: np.ndarray,
    threshold: float,
    candidate_threshold: float,
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tr2strokeseg-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t1 = np.asarray(reference.dataobj, dtype=np.float32)
    if float(np.max(probability)) > 0:
        slice_index = int(np.argmax(np.sum(probability, axis=(0, 1))))
    else:
        slice_index = reference.shape[2] // 2
    anatomy = np.rot90(t1[:, :, slice_index])
    p_slice = np.rot90(probability[:, :, slice_index])
    entropy_slice = np.rot90(entropy[:, :, slice_index])
    hard_slice = p_slice >= threshold
    vmin, vmax = robust_limits(anatomy)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis in axes:
        axis.imshow(anatomy, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.axis("off")

    probability_overlay = np.ma.masked_where(p_slice < candidate_threshold, p_slice)
    image_p = axes[0].imshow(
        probability_overlay, cmap="turbo", vmin=0, vmax=1, alpha=0.72, interpolation="nearest"
    )
    if p_slice.min() <= threshold <= p_slice.max():
        axes[0].contour(p_slice, levels=[threshold], colors="white", linewidths=1.2)
    axes[0].set_title(r"Lesion probability $p_{lesion}$")
    figure.colorbar(image_p, ax=axes[0], fraction=0.046, pad=0.02)

    uncertainty_overlay = np.ma.masked_where(p_slice < candidate_threshold, entropy_slice)
    image_u = axes[1].imshow(
        uncertainty_overlay, cmap="magma", vmin=0, vmax=1, alpha=0.78, interpolation="nearest"
    )
    axes[1].set_title("Predictive entropy (0 certain, 1 uncertain)")
    figure.colorbar(image_u, ax=axes[1], fraction=0.046, pad=0.02)

    if hard_slice.any() and not hard_slice.all():
        axes[2].contour(hard_slice.astype(float), levels=[0.5], colors="#ff3030", linewidths=2)
    probability_bands = [level for level in (0.25, 0.5, 0.75) if p_slice.min() <= level <= p_slice.max()]
    if probability_bands:
        axes[2].contour(
            p_slice,
            levels=probability_bands,
            colors=["#ffe082", "#ff3030", "#7cff6b"][: len(probability_bands)],
            linewidths=1.2,
        )
    axes[2].set_title(
        f"p≥{threshold:g} mask: {metrics['hard_lesion_volume_ml']:.1f} mL\n"
        f"Expected volume: {metrics['expected_lesion_volume_ml']:.1f} mL"
    )

    figure.suptitle(
        f"{case.case_id} | axial slice {slice_index} | max p={metrics['maximum_lesion_probability']:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def postprocess_case(
    case: Case, prediction_dir: Path, args: argparse.Namespace
) -> dict[str, object]:
    outputs = outputs_for_case(case, args.output_dir)
    outputs.case_dir.mkdir(parents=True, exist_ok=True)
    npz_path = prediction_dir / f"{case.case_id}.npz"
    segmentation_path = prediction_dir / f"{case.case_id}.nii.gz"
    if not npz_path.is_file() or not segmentation_path.is_file():
        raise FileNotFoundError(
            f"nnU-Net outputs missing for {case.case_id}: {npz_path}, {segmentation_path}"
        )

    reference = nib.load(case.mni_t1)
    with np.load(npz_path) as archive:
        if "probabilities" not in archive:
            raise KeyError(f"probabilities array is absent from {npz_path}")
        probabilities = align_probabilities(archive["probabilities"], reference.shape)
    if args.lesion_class_index >= probabilities.shape[0]:
        raise IndexError(
            f"Lesion class {args.lesion_class_index} absent from {probabilities.shape[0]} channels"
        )
    if not np.isfinite(probabilities).all():
        raise FloatingPointError(f"{case.case_id}: non-finite nnU-Net probabilities")
    probability = np.asarray(probabilities[args.lesion_class_index], dtype=np.float32)
    if float(probability.min()) < -1e-4 or float(probability.max()) > 1.0001:
        raise ValueError(f"{case.case_id}: probability outside [0, 1]")
    probability = np.clip(probability, 0.0, 1.0)
    entropy = entropy_from_probability(probability)
    hard = probability >= args.threshold

    save_nifti(
        probability,
        reference,
        outputs.lesion_probability,
        np.float32,
        "nnU-Net lesion probability; uncalibrated p_lesion",
    )
    save_nifti(
        entropy,
        reference,
        outputs.predictive_entropy,
        np.float32,
        "Normalized binary predictive entropy of p_lesion",
    )
    save_nifti(
        hard,
        reference,
        outputs.lesion_mask,
        np.uint8,
        f"Lesion mask from p_lesion >= {args.threshold:g}",
    )
    metrics = calculate_metrics(
        case,
        reference,
        probability,
        entropy,
        args.threshold,
        args.candidate_threshold,
        outputs,
    )
    if args.make_qc:
        make_qc_figure(
            case,
            reference,
            probability,
            entropy,
            args.threshold,
            args.candidate_threshold,
            metrics,
            outputs.qc_png,
        )
        metrics["qc_png"] = str(outputs.qc_png)
    write_json_atomic(outputs.metrics_json, metrics)
    if outputs.error_json.is_file():
        outputs.error_json.unlink()
    return metrics


def write_case_error(case: Case, outputs: CaseOutputs, error: Exception) -> None:
    outputs.case_dir.mkdir(parents=True, exist_ok=True)
    record = {
        **asdict(case),
        "mni_t1": str(case.mni_t1),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    write_json_atomic(outputs.error_json, record)


@contextmanager
def manifest_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / ".manifest.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def rebuild_manifests(output_dir: Path) -> None:
    with manifest_lock(output_dir):
        metrics: list[dict[str, object]] = []
        for path in sorted(output_dir.rglob("uncertainty_metrics.json")):
            with path.open(encoding="utf-8") as handle:
                metrics.append(json.load(handle))
        if metrics:
            fieldnames: list[str] = []
            for row in metrics:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            manifest = output_dir / "uncertainty_manifest.csv"
            temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics)
            os.replace(temporary, manifest)

        failures: list[dict[str, object]] = []
        for path in sorted(output_dir.rglob("processing_error.json")):
            with path.open(encoding="utf-8") as handle:
                failure = json.load(handle)
            failure.pop("traceback", None)
            failures.append(failure)
        failure_manifest = output_dir / "failures.csv"
        if failures:
            fieldnames = []
            for row in failures:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            temporary = failure_manifest.with_name(
                f".{failure_manifest.name}.{os.getpid()}.tmp"
            )
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failures)
            os.replace(temporary, failure_manifest)
        elif failure_manifest.is_file():
            failure_manifest.unlink()


def mark_completed_subjects(cases: list[Case], args: argparse.Namespace) -> None:
    for subject in sorted({case.subject for case in cases}):
        subject_cases = [case for case in cases if case.subject == subject]
        if all(
            outputs_complete(outputs_for_case(case, args.output_dir), args.make_qc)
            for case in subject_cases
        ):
            marker = args.output_dir / subject / "lesion_uncertainty_complete"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()


def main() -> int:
    args = parse_args()
    args.mni_root = args.mni_root.resolve()
    if args.output_dir is None:
        args.output_dir = args.mni_root.parent / "lesion_uncertainty"
    args.output_dir = args.output_dir.resolve()
    args.nnunet_results = args.nnunet_results.resolve() if args.nnunet_results else None
    args.nnunet_predict = args.nnunet_predict.resolve() if args.nnunet_predict else None
    validate_args(args)
    cases = discover_cases(args)
    if not cases:
        print(f"No eligible full-head MNI T1 images found under {args.mni_root}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        case
        for case in cases
        if args.overwrite
        or not outputs_complete(outputs_for_case(case, args.output_dir), args.make_qc)
    ]
    print(
        f"Found {len(cases)} case(s) across {len({case.subject for case in cases})} subject(s); "
        f"{len(pending)} pending",
        flush=True,
    )
    print(f"Input MNI root: {args.mni_root}", flush=True)
    print(f"Output root: {args.output_dir}", flush=True)
    if not pending:
        rebuild_manifests(args.output_dir)
        mark_completed_subjects(cases, args)
        print("All selected outputs are complete; nothing to do", flush=True)
        return 0

    device = resolve_device(args.device)
    print(f"nnU-Net device: {device}", flush=True)
    if args.dry_run:
        dry_work = args.output_dir / "_dry_run_work"
        run_nnunet(pending, args, device, dry_work)
        for case in pending:
            outputs = outputs_for_case(case, args.output_dir)
            print(f"Would export {case.case_id} -> {outputs.case_dir}")
        return 0

    cache_parent = Path(os.environ.get("SLURM_TMPDIR", args.output_dir / "_work"))
    cache_parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lesion-uncertainty-", dir=cache_parent) as temp:
        work_dir = Path(temp)
        prediction_dir = run_nnunet(pending, args, device, work_dir)
        for index, case in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] Exporting {case.case_id}", flush=True)
            outputs = outputs_for_case(case, args.output_dir)
            try:
                row = postprocess_case(case, prediction_dir, args)
                rows.append(row)
                print(
                    f"  expected={row['expected_lesion_volume_ml']:.2f} mL, "
                    f"hard={row['hard_lesion_volume_ml']:.2f} mL, "
                    f"entropy mass={row['entropy_mass_ml']:.2f} mL",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                write_case_error(case, outputs, error)
                print(
                    f"{case.case_id} FAILED: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    raise
            rebuild_manifests(args.output_dir)

    mark_completed_subjects(cases, args)
    print(f"Manifest: {args.output_dir / 'uncertainty_manifest.csv'}", flush=True)
    if failures:
        print(f"{failures} case(s) failed; see {args.output_dir / 'failures.csv'}", file=sys.stderr)
        return 1
    print(f"Completed {len(rows)} case(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

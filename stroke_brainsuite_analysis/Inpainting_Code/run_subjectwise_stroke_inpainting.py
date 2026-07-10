#!/usr/bin/env python3
"""Subjectwise stroke delineation followed by diffusion lesion inpainting.

For every discovered BIDS T1 image, this script creates one self-contained case
directory.  The following final images are all on the same 1 mm MNI grid:

* non-skull-stripped, N4-corrected T1;
* BrainSuite skull-stripped T1;
* BrainSuite skull-stripping (brain) mask;
* nnU-Net stroke mask;
* stroke mask dilated by 3 mm for inpainting;
* full-head inpainted T1;
* skull-stripped inpainted T1.

nnU-Net receives the non-skull-stripped MNI T1, as in the delineation pipeline.
The inpainting model receives the MNI-aligned BrainSuite brain and the notebook's
histogram-peak normalization, which avoids the blank-lesion failure caused by
normalizing a full-head image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import run_sample_arc_stroke_pipeline as delineation  # noqa: E402
import run_sample_arc_inpainting as inpainting  # noqa: E402


LESION_DILATION_MM = 3.0


@dataclass(frozen=True)
class Case:
    index: int
    case_id: str
    subject: str
    session: str
    source_path: Path


@dataclass(frozen=True)
class CaseFiles:
    case: Case
    case_dir: Path
    work_dir: Path
    std_t1: Path
    n4_t1: Path
    bse_brain_source: Path
    bse_mask_source: Path
    to_mni_mat: Path
    nnunet_input_dir: Path
    nnunet_input: Path
    nnunet_prediction_dir: Path
    nnunet_raw_prediction: Path
    model_input: Path
    model_mask: Path
    model_inpainted: Path
    full_t1_mni: Path
    brain_t1_mni: Path
    skullstrip_mask_mni: Path
    lesion_mask_mni: Path
    dilated_lesion_mask_mni: Path
    inpainted_t1_mni: Path
    brain_inpainted_t1_mni: Path
    metadata_json: Path
    error_json: Path


def parse_args() -> argparse.Namespace:
    default_input = Path("/deneb_disk/ARC")
    default_mni = delineation.first_existing(delineation.MNI_TEMPLATE_CANDIDATES)
    default_results = delineation.first_existing(delineation.NNUNET_RESULTS_CANDIDATES)
    default_fsl = delineation.first_existing(delineation.FSL_BIN_CANDIDATES)
    default_bse = delineation.executable_default("bse", delineation.BSE_CANDIDATES)
    default_predict = delineation.executable_default(
        "nnUNetv2_predict", delineation.NNUNET_PREDICT_CANDIDATES
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "With no input/output arguments, process every T1w scan under "
            "/deneb_disk/ARC and write to "
            "/deneb_disk/ARC/derivatives/stroke_inpainting."
        ),
    )
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <input-root>/derivatives/stroke_inpainting",
    )
    parser.add_argument("--modality", default="T1w")
    parser.add_argument(
        "--case-glob",
        default=None,
        help="Default: sub-*/ses-*/anat/*_<modality>.nii.gz",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Process only this case ID; may be specified multiple times",
    )

    parser.add_argument("--mni-template", type=Path, default=default_mni)
    parser.add_argument("--fsl-bin", type=Path, default=default_fsl)
    parser.add_argument("--bse-path", type=Path, default=default_bse)

    parser.add_argument("--nnunet-results", type=Path, default=default_results)
    parser.add_argument("--nnunet-predict", type=Path, default=default_predict)
    parser.add_argument("--dataset-id", default="Dataset001_Atlas2")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--folds", default="0")
    parser.add_argument("--nnunet-checkpoint", default="checkpoint_best.pth")
    parser.add_argument(
        "--nnunet-device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="auto uses CUDA only when at least 10 GiB is available",
    )
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument(
        "--enable-tta",
        action="store_true",
        help="Force TTA on CPU; by default it is disabled for practical runtime",
    )
    parser.add_argument("--disable-progress-bar", action="store_true")

    parser.add_argument(
        "--inpainting-checkpoint",
        type=Path,
        default=HERE / "Inpainting_model_inpaint_smart_ir_epoch800.pt",
    )
    parser.add_argument("--inpainting-device", default="cuda")
    parser.add_argument(
        "--inpainting-precision",
        choices=("auto", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--inpainting-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--intensity-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Robustly match generated tissue to a local perilesional T1 ring",
    )
    parser.add_argument(
        "--intensity-match-ring-mm",
        type=float,
        default=5.0,
        help="Width of the normal-tissue ring used for local intensity matching",
    )

    parser.add_argument(
        "--stop-after",
        choices=("preprocessing", "delineation", "inpainting"),
        default="inpainting",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed scan instead of recording it and continuing",
    )
    return parser.parse_args()


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def bids_entity(path: Path, prefix: str) -> str:
    for part in path.parts:
        if part.startswith(prefix):
            return part
    return "unknown"


def discover_cases(args: argparse.Namespace) -> list[Case]:
    pattern = args.case_glob or f"sub-*/ses-*/anat/*_{args.modality}.nii.gz"
    paths = sorted(args.input_root.glob(pattern))
    selected = set(args.case)
    cases: list[Case] = []
    for source_path in paths:
        case_id = nifti_stem(source_path)
        if selected and case_id not in selected:
            continue
        cases.append(
            Case(
                index=len(cases),
                case_id=case_id,
                subject=bids_entity(source_path, "sub-"),
                session=bids_entity(source_path, "ses-"),
                source_path=source_path.resolve(),
            )
        )
    if args.limit is not None:
        cases = cases[: args.limit]
    missing = selected - {case.case_id for case in cases}
    if missing:
        raise ValueError(f"Requested cases not found: {', '.join(sorted(missing))}")
    return cases


def files_for_case(case: Case, output_dir: Path) -> CaseFiles:
    # Keep all sessions underneath their BIDS subject, and retain a final scan
    # level because a small number of ARC sessions contain multiple T1w runs.
    case_dir = output_dir / case.subject / case.session / case.case_id
    work = case_dir / "work"
    nn_input_dir = work / "nnunet_input"
    nn_prediction_dir = work / "nnunet_prediction"
    model_dir = work / "inpainting_model_space"
    return CaseFiles(
        case=case,
        case_dir=case_dir,
        work_dir=work,
        std_t1=work / f"{case.case_id}_reoriented.nii.gz",
        n4_t1=work / f"{case.case_id}_n4.nii.gz",
        bse_brain_source=work / f"{case.case_id}_bse_brain_source.nii.gz",
        bse_mask_source=work / f"{case.case_id}_bse_mask_source.nii.gz",
        to_mni_mat=work / f"{case.case_id}_to_mni.mat",
        nnunet_input_dir=nn_input_dir,
        nnunet_input=nn_input_dir / f"{case.case_id}_0000.nii.gz",
        nnunet_prediction_dir=nn_prediction_dir,
        nnunet_raw_prediction=nn_prediction_dir / f"{case.case_id}.nii.gz",
        model_input=model_dir / f"{case.case_id}_model_input.nii.gz",
        model_mask=model_dir / f"{case.case_id}_stroke_model_mask.nii.gz",
        model_inpainted=model_dir / f"{case.case_id}_inpainted_model_space.nii.gz",
        full_t1_mni=case_dir / f"{case.case_id}_mni_1mm.nii.gz",
        brain_t1_mni=case_dir / f"{case.case_id}_brain_mni_1mm.nii.gz",
        skullstrip_mask_mni=case_dir / f"{case.case_id}_skullstrip_mask_mni_1mm.nii.gz",
        lesion_mask_mni=case_dir / f"{case.case_id}_stroke_mask_mni_1mm.nii.gz",
        dilated_lesion_mask_mni=(
            case_dir / f"{case.case_id}_stroke_mask_dilated_3mm_mni_1mm.nii.gz"
        ),
        inpainted_t1_mni=case_dir / f"{case.case_id}_inpainted_mni_1mm.nii.gz",
        brain_inpainted_t1_mni=case_dir / f"{case.case_id}_brain_inpainted_mni_1mm.nii.gz",
        metadata_json=case_dir / "processing_metadata.json",
        error_json=case_dir / "processing_error.json",
    )


def validate_args(args: argparse.Namespace) -> None:
    required = {
        "--input-root": args.input_root,
        "--mni-template": args.mni_template,
        "--fsl-bin": args.fsl_bin,
        "--bse-path": args.bse_path,
    }
    if args.stop_after != "preprocessing":
        required.update(
            {
                "--nnunet-results": args.nnunet_results,
                "--nnunet-predict": args.nnunet_predict,
            }
        )
    if args.stop_after == "inpainting":
        required["--inpainting-checkpoint"] = args.inpainting_checkpoint
    for label, path in required.items():
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for name in ("fslreorient2std", "flirt"):
        command = args.fsl_bin / name
        if not command.is_file():
            raise FileNotFoundError(command)
    if not 1 <= args.inpainting_steps <= 1000:
        raise ValueError("--inpainting-steps must be between 1 and 1000")
    if args.intensity_match_ring_mm <= 0:
        raise ValueError("--intensity-match-ring-mm must be positive")
    if args.disable_tta and args.enable_tta:
        raise ValueError("--disable-tta and --enable-tta are mutually exclusive")


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["FSLDIR"] = str(args.fsl_bin.parent)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    path_entries = [str(args.fsl_bin)]
    if args.nnunet_predict is not None:
        path_entries.append(str(args.nnunet_predict.parent))
    path_entries.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_entries)
    if args.nnunet_results is not None:
        env["nnUNet_results"] = str(args.nnunet_results)
    cache = args.output_dir / "_nnunet_cache"
    env.setdefault("nnUNet_raw", str(cache / "raw"))
    env.setdefault("nnUNet_preprocessed", str(cache / "preprocessed"))
    if not args.dry_run:
        Path(env["nnUNet_raw"]).mkdir(parents=True, exist_ok=True)
        Path(env["nnUNet_preprocessed"]).mkdir(parents=True, exist_ok=True)
    return env


def run_command(command: list[str | Path], env: dict[str, str], dry_run: bool) -> None:
    printable = shlex.join(str(part) for part in command)
    print(f"Running: {printable}", flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in command], check=True, env=env)


def copy_file(source: Path, destination: Path, overwrite: bool, dry_run: bool) -> None:
    if destination.is_file() and not overwrite:
        return
    print(f"Copying: {source} -> {destination}", flush=True)
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def canonicalize_brain_and_mask(files: CaseFiles) -> None:
    reference = nib.load(files.full_t1_mni)
    brain_image = nib.load(files.brain_t1_mni)
    mask_image = nib.load(files.skullstrip_mask_mni)
    if reference.shape != brain_image.shape or reference.shape != mask_image.shape:
        raise ValueError(f"{files.case.case_id}: brain extraction output shape mismatch")
    if not np.allclose(reference.affine, brain_image.affine, atol=1e-4, rtol=1e-5):
        raise ValueError(f"{files.case.case_id}: brain T1 affine mismatch")
    if not np.allclose(reference.affine, mask_image.affine, atol=1e-4, rtol=1e-5):
        raise ValueError(f"{files.case.case_id}: skull-strip mask affine mismatch")

    brain = np.asarray(brain_image.dataobj, dtype=np.float32).copy()
    mask = np.asarray(mask_image.dataobj) > 0.5
    brain[~mask] = 0
    inpainting.save_nifti(
        brain,
        reference.affine,
        files.brain_t1_mni,
        np.float32,
        header=reference.header,
    )
    inpainting.save_nifti(
        mask,
        reference.affine,
        files.skullstrip_mask_mni,
        np.uint8,
        header=reference.header,
    )


def preprocess_subject(
    files: CaseFiles, args: argparse.Namespace, env: dict[str, str]
) -> None:
    if not args.dry_run:
        files.case_dir.mkdir(parents=True, exist_ok=True)
        files.work_dir.mkdir(parents=True, exist_ok=True)
    reorient = args.fsl_bin / "fslreorient2std"
    flirt = args.fsl_bin / "flirt"

    if args.overwrite or not files.std_t1.is_file():
        run_command(
            [reorient, files.case.source_path, files.std_t1], env, args.dry_run
        )

    if args.overwrite or not files.n4_t1.is_file():
        print(f"Running N4 bias correction: {files.n4_t1}", flush=True)
        if not args.dry_run:
            delineation.n4_bias_correct(files.std_t1, files.n4_t1, overwrite=True)

    bse_complete = files.bse_brain_source.is_file() and files.bse_mask_source.is_file()
    if args.overwrite or not bse_complete:
        run_command(
            [
                args.bse_path,
                "-i",
                files.n4_t1,
                "-o",
                files.bse_brain_source,
                "--mask",
                files.bse_mask_source,
                "--auto",
                "-p",
            ],
            env,
            args.dry_run,
        )

    registration_complete = files.full_t1_mni.is_file() and files.to_mni_mat.is_file()
    if args.overwrite or not registration_complete:
        run_command(
            [
                flirt,
                "-in",
                files.n4_t1,
                "-ref",
                args.mni_template,
                "-out",
                files.full_t1_mni,
                "-omat",
                files.to_mni_mat,
                "-bins",
                "256",
                "-cost",
                "corratio",
                "-interp",
                "trilinear",
                "-dof",
                "12",
            ],
            env,
            args.dry_run,
        )

    if args.overwrite or not files.brain_t1_mni.is_file():
        run_command(
            [
                flirt,
                "-in",
                files.bse_brain_source,
                "-ref",
                files.full_t1_mni,
                "-applyxfm",
                "-init",
                files.to_mni_mat,
                "-interp",
                "trilinear",
                "-out",
                files.brain_t1_mni,
            ],
            env,
            args.dry_run,
        )

    if args.overwrite or not files.skullstrip_mask_mni.is_file():
        run_command(
            [
                flirt,
                "-in",
                files.bse_mask_source,
                "-ref",
                files.full_t1_mni,
                "-applyxfm",
                "-init",
                files.to_mni_mat,
                "-interp",
                "nearestneighbour",
                "-out",
                files.skullstrip_mask_mni,
            ],
            env,
            args.dry_run,
        )

    if not args.dry_run:
        canonicalize_brain_and_mask(files)


def canonicalize_lesion_mask(files: CaseFiles) -> None:
    reference = nib.load(files.full_t1_mni)
    prediction = nib.load(files.nnunet_raw_prediction)
    if reference.shape != prediction.shape:
        raise ValueError(f"{files.case.case_id}: nnU-Net output shape mismatch")
    if not np.allclose(reference.affine, prediction.affine, atol=1e-4, rtol=1e-5):
        raise ValueError(f"{files.case.case_id}: nnU-Net output affine mismatch")
    mask = np.asarray(prediction.dataobj) > 0
    inpainting.save_nifti(
        mask,
        reference.affine,
        files.lesion_mask_mni,
        np.uint8,
        header=reference.header,
    )


def dilate_mask_mm(
    mask: np.ndarray,
    voxel_sizes: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    """Dilate a binary mask by a Euclidean radius in physical millimetres."""
    mask = np.asarray(mask, dtype=bool)
    if radius_mm <= 0 or not mask.any():
        return mask.copy()
    distance = distance_transform_edt(~mask, sampling=voxel_sizes)
    return distance <= radius_mm


def create_dilated_lesion_mask(files: CaseFiles, overwrite: bool) -> None:
    if files.dilated_lesion_mask_mni.is_file() and not overwrite:
        return

    lesion_image = nib.load(files.lesion_mask_mni)
    brain_mask_image = nib.load(files.skullstrip_mask_mni)
    if lesion_image.shape != brain_mask_image.shape:
        raise ValueError(f"{files.case.case_id}: lesion/brain-mask shape mismatch")
    if not np.allclose(
        lesion_image.affine, brain_mask_image.affine, atol=1e-4, rtol=1e-5
    ):
        raise ValueError(f"{files.case.case_id}: lesion/brain-mask affine mismatch")

    lesion = np.asarray(lesion_image.dataobj) > 0
    brain_mask = np.asarray(brain_mask_image.dataobj) > 0
    voxel_sizes = tuple(
        float(value) for value in lesion_image.header.get_zooms()[:3]
    )
    dilated = dilate_mask_mm(lesion, voxel_sizes, LESION_DILATION_MM)
    # Do not replace extracranial full-head voxels. Preserve every original
    # lesion voxel even if a skull-stripping edge excludes it.
    dilated = lesion | (dilated & brain_mask)
    inpainting.save_nifti(
        dilated,
        lesion_image.affine,
        files.dilated_lesion_mask_mni,
        np.uint8,
        header=lesion_image.header,
    )


def delineate_subject(
    files: CaseFiles, args: argparse.Namespace, env: dict[str, str]
) -> None:
    if files.lesion_mask_mni.is_file() and not args.overwrite:
        return
    copy_file(
        files.full_t1_mni,
        files.nnunet_input,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    command: list[str | Path] = [
        args.nnunet_predict,
        "-i",
        files.nnunet_input_dir,
        "-o",
        files.nnunet_prediction_dir,
        "-d",
        args.dataset_id,
        "-c",
        args.configuration,
        "-tr",
        args.trainer,
        "-f",
        args.folds,
        "-chk",
        args.nnunet_checkpoint,
        "-device",
        args.resolved_nnunet_device,
    ]
    if args.save_probabilities:
        command.append("--save_probabilities")
    disable_tta = args.disable_tta or (
        args.resolved_nnunet_device == "cpu" and not args.enable_tta
    )
    if disable_tta:
        command.append("--disable_tta")
    if args.resolved_nnunet_device == "cpu":
        command.extend(["-npp", "1", "-nps", "1"])
    if args.disable_progress_bar:
        command.append("--disable_progress_bar")
    if not args.dry_run:
        files.nnunet_prediction_dir.mkdir(parents=True, exist_ok=True)
    run_command(command, env, args.dry_run)
    if not args.dry_run:
        if not files.nnunet_raw_prediction.is_file():
            raise FileNotFoundError(
                f"nnU-Net did not create {files.nnunet_raw_prediction}"
            )
        canonicalize_lesion_mask(files)


def prepare_inpainting_case(files: CaseFiles) -> inpainting.PreparedCase:
    brain_image = nib.load(files.brain_t1_mni)
    lesion_image = nib.load(files.dilated_lesion_mask_mni)
    if brain_image.shape != lesion_image.shape:
        raise ValueError(f"{files.case.case_id}: brain/lesion shape mismatch")
    if not np.allclose(brain_image.affine, lesion_image.affine, atol=1e-4, rtol=1e-5):
        raise ValueError(f"{files.case.case_id}: brain/lesion affine mismatch")

    brain = np.asarray(brain_image.dataobj, dtype=np.float32)
    lesion = np.asarray(lesion_image.dataobj) > 0
    raw_peak = inpainting.histogram_peak(brain.copy())
    peak = float(int(raw_peak))
    if peak <= 0:
        raise ValueError(f"{files.case.case_id}: invalid inpainting peak {peak}")

    cropped_brain, offset = inpainting.center_crop_or_pad(
        brain, inpainting.MODEL_CROP_SHAPE
    )
    cropped_lesion, lesion_offset = inpainting.center_crop_or_pad(
        lesion, inpainting.MODEL_CROP_SHAPE
    )
    if offset != lesion_offset:
        raise RuntimeError("Brain and lesion crop offsets differ")
    if int(cropped_lesion.sum()) != int(lesion.sum()):
        lost = int(lesion.sum()) - int(cropped_lesion.sum())
        raise ValueError(f"{files.case.case_id}: model crop removes {lost} lesion voxels")

    model_image = (
        inpainting.resample_model_grid(cropped_brain, is_label=False) / peak
    )
    model_mask = inpainting.resample_model_grid(
        cropped_lesion, is_label=True
    ) > 0.5
    normalized_mean = float(model_image.mean())
    if not 0.18 <= normalized_mean <= 0.45:
        raise ValueError(
            f"{files.case.case_id}: normalized input mean {normalized_mean:.3f} "
            "is outside 0.18-0.45; check brain extraction and normalization"
        )
    if lesion.any() and not model_mask.any():
        raise ValueError(f"{files.case.case_id}: lesion vanished at 2 mm")

    voxel_map = np.eye(4, dtype=np.float64)
    voxel_map[:3, :3] *= inpainting.MODEL_SPACING_MM
    voxel_map[:3, 3] = np.asarray(offset, dtype=np.float64) + 0.5
    model_affine = brain_image.affine @ voxel_map
    inpainting_paths = inpainting.CasePaths(
        files.case.index,
        files.case.case_id,
        files.full_t1_mni,
        files.bse_brain_source,
        files.to_mni_mat,
        files.dilated_lesion_mask_mni,
    )
    return inpainting.PreparedCase(
        paths=inpainting_paths,
        brain_mni_path=files.brain_t1_mni,
        source_image=brain_image,
        source_data=brain,
        source_mask=np.asarray(lesion, dtype=np.uint8),
        model_image=np.asarray(model_image, dtype=np.float32),
        model_mask=np.asarray(model_mask, dtype=np.uint8),
        model_affine=model_affine,
        peak=peak,
        crop_offset=offset,
    )


def harmonize_generated_intensity(
    generated: np.ndarray,
    target: np.ndarray,
    lesion: np.ndarray,
    brain_mask: np.ndarray,
    voxel_sizes: tuple[float, float, float],
    ring_width_mm: float,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Match generated intensity scale/offset to normal tissue around a lesion.

    An iteratively trimmed paired regression provides a robust affine mapping.
    The fit is estimated outside the lesion, so pathological lesion intensities
    never influence it. Only generated lesion voxels are later inserted.
    """
    outside_distance = distance_transform_edt(~lesion, sampling=voxel_sizes)
    ring = (
        (outside_distance > 0)
        & (outside_distance <= ring_width_mm)
        & brain_mask
        & np.isfinite(generated)
        & np.isfinite(target)
    )
    generated_ring = np.asarray(generated[ring], dtype=np.float64)
    target_ring = np.asarray(target[ring], dtype=np.float64)
    valid = (generated_ring > 0) & (target_ring > 0)
    generated_ring = generated_ring[valid]
    target_ring = target_ring[valid]

    scale = 1.0
    offset = 0.0
    matched = False
    if enabled and generated_ring.size >= 100:
        fit_generated = generated_ring
        fit_target = target_ring
        generated_bounds = np.quantile(fit_generated, (0.01, 0.99))
        target_bounds = np.quantile(fit_target, (0.01, 0.99))
        keep = (
            (fit_generated >= generated_bounds[0])
            & (fit_generated <= generated_bounds[1])
            & (fit_target >= target_bounds[0])
            & (fit_target <= target_bounds[1])
        )
        fit_generated = fit_generated[keep]
        fit_target = fit_target[keep]
        for _ in range(3):
            generated_centered = fit_generated - fit_generated.mean()
            variance = float(np.dot(generated_centered, generated_centered))
            if variance <= 1e-6:
                break
            target_centered = fit_target - fit_target.mean()
            scale = float(
                np.clip(
                    np.dot(generated_centered, target_centered) / variance,
                    0.5,
                    2.0,
                )
            )
            offset = float(np.median(fit_target - scale * fit_generated))
            residual = fit_target - (scale * fit_generated + offset)
            residual_center = float(np.median(residual))
            mad = float(np.median(np.abs(residual - residual_center)))
            if mad <= 1e-6:
                break
            inliers = np.abs(residual - residual_center) <= 3.0 * 1.4826 * mad
            if int(inliers.sum()) < 100 or bool(inliers.all()):
                break
            fit_generated = fit_generated[inliers]
            fit_target = fit_target[inliers]
        offset = float(np.median(target_ring - scale * generated_ring))
        candidate_parameters = [
            (1.0, 0.0),
            (1.0, float(np.median(target_ring - generated_ring))),
            (scale, offset),
        ]
        candidate_errors = [
            float(
                np.median(
                    np.abs(
                        candidate_scale * generated_ring
                        + candidate_offset
                        - target_ring
                    )
                )
            )
            for candidate_scale, candidate_offset in candidate_parameters
        ]
        best = int(np.argmin(candidate_errors))
        scale, offset = candidate_parameters[best]
        matched = True

    corrected = np.asarray(generated * scale + offset, dtype=np.float32)
    tissue = np.asarray(
        target[brain_mask & np.isfinite(target) & (target >= 0)], dtype=np.float64
    )
    if tissue.size:
        lower = min(0.0, float(np.quantile(tissue, 0.001)))
        upper = float(np.quantile(tissue, 0.999))
        if upper > lower:
            corrected = np.clip(corrected, lower, upper)
    else:
        lower = float("nan")
        upper = float("nan")

    corrected_ring = np.asarray(corrected[ring], dtype=np.float64)[valid]
    before_mae = (
        float(np.median(np.abs(generated_ring - target_ring)))
        if generated_ring.size
        else float("nan")
    )
    after_mae = (
        float(np.median(np.abs(corrected_ring - target_ring)))
        if corrected_ring.size
        else float("nan")
    )
    stats: dict[str, float | int | bool] = {
        "intensity_match_enabled": enabled,
        "intensity_match_applied": matched,
        "intensity_match_ring_voxels": int(generated_ring.size),
        "intensity_scale": scale,
        "intensity_offset": offset,
        "ring_median_abs_error_before": before_mae,
        "ring_median_abs_error_after": after_mae,
        "intensity_clip_lower": lower,
        "intensity_clip_upper": upper,
    }
    return corrected, stats


def save_subject_inpainting(
    files: CaseFiles,
    prepared: inpainting.PreparedCase,
    model_output: np.ndarray,
    intensity_match: bool,
    intensity_match_ring_mm: float,
) -> dict[str, float | int | bool]:
    inpainting.save_nifti(
        prepared.model_image,
        prepared.model_affine,
        files.model_input,
        np.float32,
    )
    inpainting.save_nifti(
        prepared.model_mask,
        prepared.model_affine,
        files.model_mask,
        np.uint8,
    )
    inpainting.save_nifti(
        model_output,
        prepared.model_affine,
        files.model_inpainted,
        np.float32,
    )

    generated_model = nib.Nifti1Image(
        np.asarray(model_output * prepared.peak, dtype=np.float32),
        prepared.model_affine,
    )
    reference = nib.load(files.full_t1_mni)
    generated_1mm = np.asarray(
        resample_from_to(
            generated_model,
            (reference.shape, reference.affine),
            order=3,
            mode="constant",
            cval=0.0,
        ).dataobj,
        dtype=np.float32,
    )
    source_lesion = np.asarray(nib.load(files.lesion_mask_mni).dataobj) > 0
    lesion = np.asarray(nib.load(files.dilated_lesion_mask_mni).dataobj) > 0
    full = np.asarray(reference.dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(files.brain_t1_mni).dataobj, dtype=np.float32)
    brain_mask = np.asarray(nib.load(files.skullstrip_mask_mni).dataobj) > 0
    voxel_sizes = tuple(float(value) for value in reference.header.get_zooms()[:3])
    full_generated, full_match_stats = harmonize_generated_intensity(
        generated_1mm,
        full,
        lesion,
        brain_mask,
        voxel_sizes,
        intensity_match_ring_mm,
        intensity_match,
    )
    brain_generated, brain_match_stats = harmonize_generated_intensity(
        generated_1mm,
        brain,
        lesion,
        brain_mask,
        voxel_sizes,
        intensity_match_ring_mm,
        intensity_match,
    )
    full_inpainted = full.copy()
    brain_inpainted = brain.copy()
    full_inpainted[lesion] = full_generated[lesion]
    brain_inpainted[lesion] = brain_generated[lesion]
    inpainting.save_nifti(
        full_inpainted,
        reference.affine,
        files.inpainted_t1_mni,
        np.float32,
        header=reference.header,
    )
    inpainting.save_nifti(
        brain_inpainted,
        reference.affine,
        files.brain_inpainted_t1_mni,
        np.float32,
        header=reference.header,
    )
    stats: dict[str, float | int | bool] = {
        "histogram_peak": prepared.peak,
        "normalized_input_mean": float(prepared.model_image.mean()),
        "source_lesion_voxels": int(source_lesion.sum()),
        "dilated_lesion_voxels": int(lesion.sum()),
        "model_lesion_voxels": int(prepared.model_mask.sum()),
        "model_lesion_mean": float(model_output[prepared.model_mask > 0].mean()),
        "full_outside_lesion_max_abs_change": float(
            np.max(np.abs(full_inpainted[~lesion] - full[~lesion]))
        ),
        "brain_outside_lesion_max_abs_change": float(
            np.max(np.abs(brain_inpainted[~lesion] - brain[~lesion]))
        ),
    }
    for key, value in full_match_stats.items():
        stats[f"full_{key}"] = value
    for key, value in brain_match_stats.items():
        stats[f"brain_{key}"] = value
    return stats


def copy_empty_lesion_outputs(files: CaseFiles) -> dict[str, float | int | bool]:
    shutil.copy2(files.full_t1_mni, files.inpainted_t1_mni)
    shutil.copy2(files.brain_t1_mni, files.brain_inpainted_t1_mni)
    return {
        "histogram_peak": 0.0,
        "normalized_input_mean": 0.0,
        "source_lesion_voxels": 0,
        "dilated_lesion_voxels": 0,
        "model_lesion_voxels": 0,
        "model_lesion_mean": 0.0,
        "full_outside_lesion_max_abs_change": 0.0,
        "brain_outside_lesion_max_abs_change": 0.0,
    }


def collect_inpainting_stats(files: CaseFiles) -> dict[str, float | int | bool]:
    source_lesion = np.asarray(nib.load(files.lesion_mask_mni).dataobj) > 0
    lesion = np.asarray(nib.load(files.dilated_lesion_mask_mni).dataobj) > 0
    full = np.asarray(nib.load(files.full_t1_mni).dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(files.brain_t1_mni).dataobj, dtype=np.float32)
    full_output = np.asarray(
        nib.load(files.inpainted_t1_mni).dataobj, dtype=np.float32
    )
    brain_output = np.asarray(
        nib.load(files.brain_inpainted_t1_mni).dataobj, dtype=np.float32
    )
    peak = float(int(inpainting.histogram_peak(brain.copy())))
    stats: dict[str, float | int | bool] = {
        "histogram_peak": peak,
        "source_lesion_voxels": int(source_lesion.sum()),
        "dilated_lesion_voxels": int(lesion.sum()),
        "full_outside_lesion_max_abs_change": float(
            np.max(np.abs(full_output[~lesion] - full[~lesion]))
        ),
        "brain_outside_lesion_max_abs_change": float(
            np.max(np.abs(brain_output[~lesion] - brain[~lesion]))
        ),
    }
    if files.model_input.is_file() and files.model_mask.is_file() and files.model_inpainted.is_file():
        model_input = np.asarray(nib.load(files.model_input).dataobj, dtype=np.float32)
        model_mask = np.asarray(nib.load(files.model_mask).dataobj) > 0
        model_output = np.asarray(
            nib.load(files.model_inpainted).dataobj, dtype=np.float32
        )
        stats.update(
            {
                "normalized_input_mean": float(model_input.mean()),
                "model_lesion_voxels": int(model_mask.sum()),
                "model_lesion_mean": float(model_output[model_mask].mean())
                if model_mask.any()
                else 0.0,
            }
        )
    else:
        stats.update(
            {
                "normalized_input_mean": 0.0,
                "model_lesion_voxels": 0,
                "model_lesion_mean": 0.0,
            }
        )
    return stats


def validate_case(files: CaseFiles, stop_after: str) -> dict[str, object]:
    outputs = [files.full_t1_mni, files.brain_t1_mni, files.skullstrip_mask_mni]
    if stop_after in ("delineation", "inpainting"):
        outputs.extend([files.lesion_mask_mni, files.dilated_lesion_mask_mni])
    if stop_after == "inpainting":
        outputs.extend([files.inpainted_t1_mni, files.brain_inpainted_t1_mni])
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final outputs: {missing}")

    reference = nib.load(files.full_t1_mni)
    geometry_match = True
    finite = True
    for path in outputs:
        image = nib.load(path)
        geometry_match &= image.shape == reference.shape
        geometry_match &= bool(
            np.allclose(image.affine, reference.affine, atol=1e-4, rtol=1e-5)
        )
        finite &= bool(np.isfinite(np.asarray(image.dataobj)).all())
    if not geometry_match:
        raise ValueError(f"{files.case.case_id}: final output geometry mismatch")
    if not finite:
        raise FloatingPointError(f"{files.case.case_id}: non-finite final image")
    if stop_after in ("delineation", "inpainting"):
        lesion = np.asarray(nib.load(files.lesion_mask_mni).dataobj) > 0
        dilated = np.asarray(nib.load(files.dilated_lesion_mask_mni).dataobj) > 0
        if np.any(lesion & ~dilated):
            raise ValueError(
                f"{files.case.case_id}: dilated mask does not contain lesion mask"
            )
    zooms = tuple(float(value) for value in reference.header.get_zooms()[:3])
    if not np.allclose(zooms, (1.0, 1.0, 1.0), atol=1e-3):
        raise ValueError(f"{files.case.case_id}: expected 1 mm MNI grid, got {zooms}")

    result: dict[str, object] = {
        "case_id": files.case.case_id,
        "subject": files.case.subject,
        "session": files.case.session,
        "source_path": str(files.case.source_path),
        "shape": "x".join(str(value) for value in reference.shape),
        "zooms": "x".join(f"{value:g}" for value in zooms),
        "orientation": "".join(nib.aff2axcodes(reference.affine)),
        "geometry_match": geometry_match,
        "finite": finite,
        "full_t1_mni": str(files.full_t1_mni),
        "brain_t1_mni": str(files.brain_t1_mni),
        "skullstrip_mask_mni": str(files.skullstrip_mask_mni),
        "lesion_mask_mni": str(files.lesion_mask_mni)
        if files.lesion_mask_mni.is_file()
        else "",
        "dilated_lesion_mask_mni": str(files.dilated_lesion_mask_mni)
        if files.dilated_lesion_mask_mni.is_file()
        else "",
        "inpainted_t1_mni": str(files.inpainted_t1_mni)
        if files.inpainted_t1_mni.is_file()
        else "",
        "brain_inpainted_t1_mni": str(files.brain_inpainted_t1_mni)
        if files.brain_inpainted_t1_mni.is_file()
        else "",
    }
    return result


def write_case_metadata(
    files: CaseFiles,
    args: argparse.Namespace,
    validation: dict[str, object],
    inpainting_stats: dict[str, float | int | bool] | None,
) -> dict[str, object]:
    metadata = {
        **validation,
        "mni_template": str(args.mni_template.resolve()),
        "nnunet_dataset": args.dataset_id,
        "nnunet_configuration": args.configuration,
        "nnunet_trainer": args.trainer,
        "nnunet_checkpoint": args.nnunet_checkpoint,
        "nnunet_device_requested": args.nnunet_device,
        "nnunet_device_used": args.resolved_nnunet_device,
        "nnunet_tta_enabled": not (
            args.disable_tta
            or (args.resolved_nnunet_device == "cpu" and not args.enable_tta)
        ),
        "inpainting_checkpoint": str(args.inpainting_checkpoint.resolve()),
        "inpainting_steps": args.inpainting_steps,
        "seed": args.seed + files.case.index,
        "intensity_match": args.intensity_match,
        "intensity_match_ring_mm": args.intensity_match_ring_mm,
        "lesion_dilation_mm": LESION_DILATION_MM,
        "dilation_growth_constrained_to_brain": True,
        "inpainting": inpainting_stats or {},
    }
    with files.metadata_json.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def write_global_manifest(rows: list[dict[str, object]], output_dir: Path) -> None:
    by_case: dict[str, dict[str, object]] = {}
    for metadata_path in sorted(output_dir.rglob("processing_metadata.json")):
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        case_id = metadata.get("case_id")
        if isinstance(case_id, str):
            by_case[case_id] = metadata
    for row in rows:
        case_id = row.get("case_id")
        if isinstance(case_id, str):
            by_case[case_id] = row
    if not by_case:
        return
    all_rows = [by_case[key] for key in sorted(by_case)]
    flattened: list[dict[str, object]] = []
    for row in all_rows:
        flat = {key: value for key, value in row.items() if key != "inpainting"}
        for key, value in dict(row.get("inpainting", {})).items():
            flat[f"inpainting_{key}"] = value
        flattened.append(flat)
    fieldnames: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def write_case_error(files: CaseFiles, error: Exception) -> None:
    files.case_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": files.case.case_id,
        "subject": files.case.subject,
        "session": files.case.session,
        "source_path": str(files.case.source_path),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    with files.error_json.open("w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def clear_case_error(files: CaseFiles) -> None:
    if files.error_json.is_file():
        files.error_json.unlink()


def write_failure_manifest(output_dir: Path) -> None:
    records: list[dict[str, object]] = []
    for error_path in sorted(output_dir.rglob("processing_error.json")):
        with error_path.open() as handle:
            record = json.load(handle)
        record.pop("traceback", None)
        records.append(record)

    manifest = output_dir / "failures.csv"
    if not records:
        if manifest.is_file():
            manifest.unlink()
        return

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def resolve_nnunet_device(args: argparse.Namespace) -> str:
    if args.nnunet_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nnU-Net requested but CUDA is unavailable")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gib < 10.0:
            raise RuntimeError(
                f"CUDA nnU-Net requested on a {total_gib:.1f} GiB GPU, but this "
                "model needs at least 10 GiB. Use --nnunet-device auto or cpu."
            )
        return "cuda"
    if args.nnunet_device != "auto":
        return args.nnunet_device
    if not torch.cuda.is_available():
        print("nnU-Net device: CPU (CUDA unavailable)", flush=True)
        return "cpu"
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total_gib < 10.0:
        print(
            f"nnU-Net device: CPU (GPU has {total_gib:.1f} GiB; "
            "the 128^3 model needs more workspace)",
            flush=True,
        )
        return "cpu"
    print(f"nnU-Net device: CUDA ({total_gib:.1f} GiB GPU)", flush=True)
    return "cuda"


def resolve_inpainting_dtype(args: argparse.Namespace) -> tuple[torch.device, torch.dtype]:
    device = torch.device(args.inpainting_device)
    precision = args.inpainting_precision
    if precision == "auto":
        precision = "float16" if device.type == "cuda" else "float32"
    dtype = torch.float16 if precision == "float16" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inpainting requested but CUDA is unavailable")
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inpainting is unsupported")
    return device, dtype


def inpainting_outputs_current(files: CaseFiles) -> bool:
    required = (
        files.dilated_lesion_mask_mni,
        files.inpainted_t1_mni,
        files.brain_inpainted_t1_mni,
        files.metadata_json,
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        with files.metadata_json.open() as handle:
            metadata = json.load(handle)
        return float(metadata.get("lesion_dilation_mm")) == LESION_DILATION_MM
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def process_case(
    files: CaseFiles,
    args: argparse.Namespace,
    env: dict[str, str],
    model: torch.nn.Module | None,
    device: torch.device | None,
    dtype: torch.dtype | None,
    number: int,
    total: int,
) -> tuple[dict[str, object] | None, torch.nn.Module | None]:
    case = files.case
    print(f"\n[{number}/{total}] {case.case_id}: preprocessing", flush=True)
    preprocess_subject(files, args, env)
    if args.stop_after == "preprocessing":
        if args.dry_run:
            return None, model
        validation = validate_case(files, args.stop_after)
        return write_case_metadata(files, args, validation, None), model

    print(f"[{number}/{total}] {case.case_id}: stroke delineation", flush=True)
    delineate_subject(files, args, env)
    if args.dry_run:
        print(
            f"Would dilate {files.lesion_mask_mni} by {LESION_DILATION_MM:g} mm "
            f"-> {files.dilated_lesion_mask_mni}",
            flush=True,
        )
    else:
        create_dilated_lesion_mask(files, args.overwrite)
    if args.stop_after == "delineation":
        if args.dry_run:
            return None, model
        validation = validate_case(files, args.stop_after)
        return write_case_metadata(files, args, validation, None), model

    print(f"[{number}/{total}] {case.case_id}: inpainting", flush=True)
    if args.dry_run:
        print(
            f"Would inpaint {files.dilated_lesion_mask_mni} "
            f"-> {files.inpainted_t1_mni}",
            flush=True,
        )
        return None, model

    inpainting_complete = inpainting_outputs_current(files)
    stats: dict[str, float | int | bool] | None = None
    if args.overwrite or not inpainting_complete:
        lesion = np.asarray(nib.load(files.lesion_mask_mni).dataobj) > 0
        if not lesion.any():
            stats = copy_empty_lesion_outputs(files)
            print(f"[{case.case_id}] empty lesion mask; copied input T1s", flush=True)
        else:
            prepared = prepare_inpainting_case(files)
            if model is None:
                assert device is not None and dtype is not None
                model = inpainting.load_model(
                    args.inpainting_checkpoint, device, dtype
                )
            assert device is not None and dtype is not None
            result = inpainting.inpaint_batch(
                [prepared],
                model,
                device,
                dtype,
                args.inpainting_steps,
                args.seed + case.index,
            )[0]
            stats = save_subject_inpainting(
                files,
                prepared,
                result,
                args.intensity_match,
                args.intensity_match_ring_mm,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        if files.metadata_json.is_file():
            with files.metadata_json.open() as handle:
                previous_metadata = json.load(handle)
            previous_stats = previous_metadata.get("inpainting")
            stats = previous_stats if isinstance(previous_stats, dict) else None
        if stats is None:
            stats = collect_inpainting_stats(files)

    validation = validate_case(files, args.stop_after)
    return write_case_metadata(files, args, validation, stats), model


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    if args.output_dir is None:
        args.output_dir = args.input_root / "derivatives" / "stroke_inpainting"
    args.output_dir = args.output_dir.resolve()
    args.mni_template = args.mni_template.resolve() if args.mni_template else None
    args.fsl_bin = args.fsl_bin.resolve() if args.fsl_bin else None
    args.bse_path = args.bse_path.resolve() if args.bse_path else None
    args.nnunet_results = (
        args.nnunet_results.resolve() if args.nnunet_results else None
    )
    args.nnunet_predict = (
        args.nnunet_predict.resolve() if args.nnunet_predict else None
    )
    args.inpainting_checkpoint = args.inpainting_checkpoint.resolve()
    validate_args(args)
    cases = discover_cases(args)
    if not cases:
        print(f"No {args.modality} images found under {args.input_root}", file=sys.stderr)
        return 1

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    args.resolved_nnunet_device = (
        "not_run"
        if args.stop_after == "preprocessing"
        else resolve_nnunet_device(args)
    )
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    model: torch.nn.Module | None = None
    if args.stop_after == "inpainting" and not args.dry_run:
        device, dtype = resolve_inpainting_dtype(args)

    subject_count = len({case.subject for case in cases})
    session_count = len({(case.subject, case.session) for case in cases})
    print(
        f"Found {len(cases)} T1w scan(s) across {subject_count} subject(s) "
        f"and {session_count} session(s)",
        flush=True,
    )
    print(f"Output root: {args.output_dir}", flush=True)
    rows: list[dict[str, object]] = []
    failures = 0
    for number, case in enumerate(cases, start=1):
        files = files_for_case(case, args.output_dir)
        try:
            row, model = process_case(
                files,
                args,
                env,
                model,
                device,
                dtype,
                number,
                len(cases),
            )
            if not args.dry_run:
                clear_case_error(files)
                if row is not None:
                    rows.append(row)
                    write_global_manifest(rows, args.output_dir)
                write_failure_manifest(args.output_dir)
        except Exception as error:
            failures += 1
            print(
                f"[{number}/{len(cases)}] {case.case_id} FAILED: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if not args.dry_run:
                write_case_error(files, error)
                write_failure_manifest(args.output_dir)
            if args.fail_fast:
                raise
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nSubjectwise processing complete", flush=True)
    if not args.dry_run:
        print(f"Manifest: {args.output_dir / 'manifest.csv'}", flush=True)
        if failures:
            print(f"Failures: {args.output_dir / 'failures.csv'}", flush=True)
    if failures:
        print(
            f"{failures} scan(s) failed; all remaining scans were attempted",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())

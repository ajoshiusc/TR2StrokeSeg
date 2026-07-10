#!/usr/bin/env python3
"""Run stroke delineation on sample ARC MRIs with the trained ATLAS2 nnU-Net."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent

DEFAULT_SAMPLE_ROOT = Path("/home/ajoshi/Desktop/sample_arc")
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "outputs" / "sample_arc"

NNUNET_RESULTS_CANDIDATES = [
    Path("/home/ajoshi/Projects/TR2preproc/supp_data/models/nnUNet_results"),
    Path("/home/ajoshi/project2_ajoshi_1183/data/TR2/nnUNet_results"),
]
MNI_TEMPLATE_CANDIDATES = [
    Path("/deneb_disk/TR2_data/ATLAS_Data/ATLAS_2/MNI152NLin2009aSym.nii.gz"),
    Path("/home/ajoshi/project2_ajoshi_1183/data/ATLAS_2/MNI152NLin2009aSym.nii.gz"),
    Path("/home/ajoshi/Software/fsl/data/standard/MNI152_T1_1mm.nii.gz"),
]
BSE_CANDIDATES = [
    Path("/home/ajoshi/Software/BrainSuite23a/bin/bse"),
    Path("/home/ajoshi/Software/BrainSuite21a/bin/bse"),
]
FSL_BIN_CANDIDATES = [Path("/home/ajoshi/Software/fsl/bin")]
NNUNET_PREDICT_CANDIDATES = [REPO_ROOT / ".venv" / "bin" / "nnUNetv2_predict"]


@dataclass(frozen=True)
class Case:
    case_id: str
    source_path: Path
    subject: str
    session: str


@dataclass(frozen=True)
class CasePaths:
    case: Case
    case_dir: Path
    std_t1: Path
    n4_t1: Path
    mni_t1: Path
    flirt_mat: Path
    inv_flirt_mat: Path
    bse_brain: Path
    bse_mask: Path
    nnunet_input: Path
    mni_prediction: Path
    source_prediction: Path


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def executable_default(name: str, candidates: list[Path]) -> Path | None:
    candidate = first_existing(candidates)
    if candidate is not None:
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def bids_entity(path: Path, prefix: str) -> str:
    for part in path.parts:
        if part.startswith(prefix):
            return part
    return "unknown"


def discover_cases(input_root: Path, modality: str, case_glob: str | None, limit: int | None) -> list[Case]:
    pattern = case_glob or f"sub-*/ses-*/anat/*_{modality}.nii.gz"
    files = sorted(input_root.glob(pattern))
    cases = [
        Case(
            case_id=nifti_stem(path),
            source_path=path.resolve(),
            subject=bids_entity(path, "sub-"),
            session=bids_entity(path, "ses-"),
        )
        for path in files
    ]
    if limit is not None:
        cases = cases[:limit]
    return cases


def copy_if_needed(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def run_command(cmd: list[str | Path], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"Running: {printable}")
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], check=True, env=env)


def fsl_command(name: str, fsl_bin: Path | None) -> Path:
    if fsl_bin is not None:
        candidate = fsl_bin / name
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"Could not find FSL command '{name}'. Pass --fsl-bin.")


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_results"] = str(args.nnunet_results)
    env.setdefault("nnUNet_raw", str(args.output_dir / "nnUNet_raw"))
    env.setdefault("nnUNet_preprocessed", str(args.output_dir / "nnUNet_preprocessed"))
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    if args.fsl_bin is not None:
        env.setdefault("FSLDIR", str(args.fsl_bin.parent))
        env["PATH"] = f"{args.fsl_bin}{os.pathsep}{env.get('PATH', '')}"
    if args.nnunet_predict is not None:
        env["PATH"] = f"{args.nnunet_predict.parent}{os.pathsep}{env.get('PATH', '')}"
    Path(env["nnUNet_raw"]).mkdir(parents=True, exist_ok=True)
    Path(env["nnUNet_preprocessed"]).mkdir(parents=True, exist_ok=True)
    return env


def n4_bias_correct(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.ReadImage(str(input_path))
    image = sitk.Cast(image, sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(image)
    finite = np.isfinite(data)

    positive = data[finite & (data > 0)]
    if positive.size:
        threshold = np.percentile(positive, 10)
        mask_data = (finite & (data > threshold)).astype(np.uint8)
    else:
        mask_data = finite.astype(np.uint8)
    if mask_data.sum() == 0:
        mask_data = np.ones_like(data, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_data)
    mask.CopyInformation(image)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(image, mask)
    sitk.WriteImage(corrected, str(output_path))


def image_summary(path: Path) -> tuple[str, str]:
    image = nib.load(str(path))
    shape = "x".join(str(v) for v in image.shape)
    zooms = "x".join(f"{float(v):.4g}" for v in image.header.get_zooms()[:3])
    return shape, zooms


def paths_for_case(case: Case, args: argparse.Namespace) -> CasePaths:
    case_dir = args.output_dir / "work" / case.case_id
    preproc_dir = case_dir / "preprocessed"
    brainsuite_dir = case_dir / "brainsuite"
    final_dir = args.output_dir / "stroke_masks_source_space"
    nnunet_input_dir = args.output_dir / "nnunet_input"
    nnunet_output_dir = args.output_dir / "nnunet_predictions_mni"

    return CasePaths(
        case=case,
        case_dir=case_dir,
        std_t1=preproc_dir / f"{case.case_id}_std.nii.gz",
        n4_t1=preproc_dir / f"{case.case_id}_n4.nii.gz",
        mni_t1=preproc_dir / f"{case.case_id}_mni_1mm.nii.gz",
        flirt_mat=preproc_dir / f"{case.case_id}_to_mni.mat",
        inv_flirt_mat=preproc_dir / f"{case.case_id}_mni_to_source.mat",
        bse_brain=brainsuite_dir / f"{case.case_id}_bse_brain.nii.gz",
        bse_mask=brainsuite_dir / f"{case.case_id}_bse_mask.nii.gz",
        nnunet_input=nnunet_input_dir / f"{case.case_id}_0000.nii.gz",
        mni_prediction=nnunet_output_dir / f"{case.case_id}.nii.gz",
        source_prediction=final_dir / f"{case.case_id}_stroke_source_space.nii.gz",
    )


def preprocess_case(case_paths: CasePaths, args: argparse.Namespace, env: dict[str, str]) -> None:
    reorient = fsl_command("fslreorient2std", args.fsl_bin)
    flirt = fsl_command("flirt", args.fsl_bin)

    case_paths.std_t1.parent.mkdir(parents=True, exist_ok=True)
    if not case_paths.std_t1.exists() or args.overwrite:
        run_command([reorient, case_paths.case.source_path, case_paths.std_t1], env=env, dry_run=args.dry_run)

    if not args.dry_run:
        n4_bias_correct(case_paths.std_t1, case_paths.n4_t1, args.overwrite)

    if args.run_bse and args.bse_path is not None:
        case_paths.bse_brain.parent.mkdir(parents=True, exist_ok=True)
        bse_done = case_paths.bse_brain.exists() and case_paths.bse_mask.exists()
        if not bse_done or args.overwrite:
            run_command(
                [
                    args.bse_path,
                    "-i",
                    case_paths.n4_t1,
                    "-o",
                    case_paths.bse_brain,
                    "--mask",
                    case_paths.bse_mask,
                    "--auto",
                    "-p",
                ],
                env=env,
                dry_run=args.dry_run,
            )

    flirt_done = case_paths.mni_t1.exists() and case_paths.flirt_mat.exists()
    if not flirt_done or args.overwrite:
        run_command(
            [
                flirt,
                "-in",
                case_paths.n4_t1,
                "-ref",
                args.mni_template,
                "-out",
                case_paths.mni_t1,
                "-omat",
                case_paths.flirt_mat,
                "-bins",
                "256",
                "-cost",
                "corratio",
                "-interp",
                "trilinear",
                "-dof",
                "12",
            ],
            env=env,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        copy_if_needed(case_paths.mni_t1, case_paths.nnunet_input, args.overwrite)


def run_nnunet(case_paths: list[CasePaths], args: argparse.Namespace, env: dict[str, str]) -> None:
    output_dir = args.output_dir / "nnunet_predictions_mni"
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = [paths.mni_prediction for paths in case_paths]
    if expected and all(path.exists() for path in expected) and not args.overwrite:
        print("Skipping nnU-Net: all expected MNI-space predictions already exist.")
        return

    cmd: list[str | Path] = [
        args.nnunet_predict,
        "-i",
        args.output_dir / "nnunet_input",
        "-o",
        output_dir,
        "-d",
        args.dataset_id,
        "-c",
        args.configuration,
        "-tr",
        args.trainer,
        "-f",
        args.folds,
        "-chk",
        args.checkpoint,
        "-device",
        args.device,
    ]
    if args.save_probabilities:
        cmd.append("--save_probabilities")
    if args.disable_tta:
        cmd.append("--disable_tta")
    if args.disable_progress_bar:
        cmd.append("--disable_progress_bar")

    run_command(cmd, env=env, dry_run=args.dry_run)


def inverse_transform_predictions(case_paths: list[CasePaths], args: argparse.Namespace, env: dict[str, str]) -> None:
    convert_xfm = fsl_command("convert_xfm", args.fsl_bin)
    flirt = fsl_command("flirt", args.fsl_bin)

    for paths in case_paths:
        if not paths.mni_prediction.exists():
            print(f"Warning: missing nnU-Net prediction for {paths.case.case_id}: {paths.mni_prediction}")
            continue

        paths.source_prediction.parent.mkdir(parents=True, exist_ok=True)
        if not paths.inv_flirt_mat.exists() or args.overwrite:
            run_command(
                [convert_xfm, "-inverse", "-omat", paths.inv_flirt_mat, paths.flirt_mat],
                env=env,
                dry_run=args.dry_run,
            )

        if not paths.source_prediction.exists() or args.overwrite:
            run_command(
                [
                    flirt,
                    "-in",
                    paths.mni_prediction,
                    "-ref",
                    paths.case.source_path,
                    "-applyxfm",
                    "-init",
                    paths.inv_flirt_mat,
                    "-interp",
                    "nearestneighbour",
                    "-out",
                    paths.source_prediction,
                ],
                env=env,
                dry_run=args.dry_run,
            )


def write_manifest(case_paths: list[CasePaths], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "subject",
        "session",
        "source_path",
        "source_shape",
        "source_zooms",
        "preprocessed_mni_t1",
        "flirt_mat",
        "bse_brain",
        "bse_mask",
        "nnunet_input",
        "mni_prediction",
        "source_space_prediction",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paths in case_paths:
            shape, zooms = image_summary(paths.case.source_path)
            writer.writerow(
                {
                    "case_id": paths.case.case_id,
                    "subject": paths.case.subject,
                    "session": paths.case.session,
                    "source_path": paths.case.source_path,
                    "source_shape": shape,
                    "source_zooms": zooms,
                    "preprocessed_mni_t1": paths.mni_t1,
                    "flirt_mat": paths.flirt_mat,
                    "bse_brain": paths.bse_brain if paths.bse_brain.exists() else "",
                    "bse_mask": paths.bse_mask if paths.bse_mask.exists() else "",
                    "nnunet_input": paths.nnunet_input,
                    "mni_prediction": paths.mni_prediction if paths.mni_prediction.exists() else "",
                    "source_space_prediction": paths.source_prediction if paths.source_prediction.exists() else "",
                }
            )


def validate_args(args: argparse.Namespace) -> None:
    required_paths = {
        "--input-root": args.input_root,
        "--mni-template": args.mni_template,
        "--nnunet-results": args.nnunet_results,
        "--nnunet-predict": args.nnunet_predict,
    }
    for label, path in required_paths.items():
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if args.run_bse and (args.bse_path is None or not args.bse_path.exists()):
        raise FileNotFoundError("--run-bse is enabled but --bse-path was not found.")

    if args.fsl_bin is not None and not args.fsl_bin.exists():
        raise FileNotFoundError(f"--fsl-bin does not exist: {args.fsl_bin}")


def parse_args() -> argparse.Namespace:
    default_nnunet_results = first_existing(NNUNET_RESULTS_CANDIDATES)
    default_mni_template = first_existing(MNI_TEMPLATE_CANDIDATES)
    default_bse = executable_default("bse", BSE_CANDIDATES)
    default_fsl_bin = first_existing(FSL_BIN_CANDIDATES)
    default_nnunet_predict = executable_default("nnUNetv2_predict", NNUNET_PREDICT_CANDIDATES)

    parser = argparse.ArgumentParser(
        description="Preprocess sample ARC T1 MRIs and delineate stroke with the trained ATLAS2 nnU-Net."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modality", default="T1w", help="BIDS suffix to discover under anat/. Default: T1w.")
    parser.add_argument("--case-glob", default=None, help="Override the default BIDS glob under --input-root.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N discovered scans.")
    parser.add_argument("--mni-template", type=Path, default=default_mni_template)
    parser.add_argument("--nnunet-results", type=Path, default=default_nnunet_results)
    parser.add_argument("--nnunet-predict", type=Path, default=default_nnunet_predict)
    parser.add_argument("--dataset-id", default="Dataset001_Atlas2")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--folds", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fsl-bin", type=Path, default=default_fsl_bin)
    parser.add_argument("--bse-path", type=Path, default=default_bse)
    parser.add_argument("--run-bse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess-only", action="store_true", help="Stop after creating nnU-Net inputs.")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    cases = discover_cases(args.input_root, args.modality, args.case_glob, args.limit)
    if not cases:
        print(f"No {args.modality} NIfTI files found under {args.input_root}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    case_paths = [paths_for_case(case, args) for case in cases]

    print(f"Found {len(case_paths)} {args.modality} scans.")
    print(f"Writing analysis outputs to: {args.output_dir}")
    print(f"Using nnU-Net results: {args.nnunet_results}")
    print(f"Using MNI template: {args.mni_template}")

    for index, paths in enumerate(case_paths, start=1):
        print(f"\n[{index}/{len(case_paths)}] Preprocessing {paths.case.case_id}")
        preprocess_case(paths, args, env)

    if not args.preprocess_only:
        print("\nRunning stroke nnU-Net inference")
        run_nnunet(case_paths, args, env)
        print("\nMapping stroke masks back to source image space")
        inverse_transform_predictions(case_paths, args, env)

    if not args.dry_run:
        manifest = args.output_dir / "manifest.csv"
        write_manifest(case_paths, manifest)
        print(f"\nManifest written to: {manifest}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

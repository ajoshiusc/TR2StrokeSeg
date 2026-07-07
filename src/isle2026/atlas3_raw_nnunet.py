"""
Minimal nnUNet v2 workflow for ATLAS3 raw subject-space stroke lesion data.

This module prepares ATLAS3 raw T1w/lesion-mask pairs in nnUNet format and wraps
the common plan, train, and predict commands. It deliberately avoids
registration. Optional resampling changes voxel spacing only and should be used
consistently for both training and inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOCAL_ATLAS3_ROOT = Path("/deneb_disk/ATLAS3")
CARC_ATLAS3_ROOT = Path("/project2/ajoshi_1183/data/ATLAS3")
ATLAS3_ROOT_CANDIDATES = (LOCAL_ATLAS3_ROOT, CARC_ATLAS3_ROOT)

LOCAL_WORK_DIR = LOCAL_ATLAS3_ROOT / "nnunet_isle2026"
CARC_WORK_DIR = Path("/project2/ajoshi_1183/data/ISLE2026_ATLAS3_nnunet")
WORK_DIR_CANDIDATES = (LOCAL_WORK_DIR, CARC_WORK_DIR)

DEFAULT_ATLAS3_ROOT = LOCAL_ATLAS3_ROOT
DEFAULT_WORK_DIR = LOCAL_WORK_DIR
DEFAULT_DATASET_ID = 326
DEFAULT_DATASET_NAME = "ATLAS3Raw"
DEFAULT_IMAGE_GLOB = "*space-orig_desc-brain_T1w.nii.gz"
DEFAULT_IMAGE_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
DEFAULT_LABEL_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"


@dataclass(frozen=True)
class Atlas3Case:
    case_id: str
    image: Path
    label: Path | None
    source_id: str
    site: str
    subject: str
    session: str


@dataclass(frozen=True)
class NnUNetDirs:
    raw: Path
    preprocessed: Path
    results: Path


def strip_nii_suffix(path: Path | str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path).stem


def sanitize_case_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return sanitized or "case"


def source_id_from_image(image: Path) -> str:
    stem = strip_nii_suffix(image)
    for marker in ("_space-", "_desc-", "_T1w"):
        index = stem.find(marker)
        if index > 0:
            return stem[:index]
    return stem


def parse_bids_parts(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    subject_index = next((i for i, part in enumerate(parts) if part.startswith("sub-")), None)
    subject = parts[subject_index] if subject_index is not None else ""
    site = parts[subject_index - 1] if subject_index and subject_index > 0 else ""
    session = next((part for part in parts if part.startswith("ses-")), "")
    return site, subject, session


def unique_case_id(base_id: str, used: set[str]) -> str:
    candidate = base_id
    suffix = 2
    while candidate in used:
        candidate = f"{base_id}_{suffix:02d}"
        suffix += 1
    used.add(candidate)
    return candidate


def format_path_candidates(paths: Sequence[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def resolve_atlas3_root(atlas3_root: Path | None) -> Path:
    if atlas3_root is not None:
        return atlas3_root.expanduser().resolve()

    env_root = os.environ.get("ATLAS3_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    for candidate in ATLAS3_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "ATLAS3 root not found. Set --atlas3-root or ATLAS3_ROOT. "
        f"Checked: {format_path_candidates(ATLAS3_ROOT_CANDIDATES)}"
    )


def resolve_atlas3_raw_root(atlas3_root: Path) -> Path:
    atlas3_root = atlas3_root.expanduser().resolve()
    raw_child = atlas3_root / "ATLAS3_Training_Raw"
    if raw_child.is_dir():
        return raw_child
    return atlas3_root


def default_work_dir_for_atlas3_root(atlas3_root: Path) -> Path:
    root_label = Path(os.path.normpath(str(atlas3_root.expanduser())))
    if root_label.name == "ATLAS3_Training_Raw":
        root_label = root_label.parent
    if root_label == LOCAL_ATLAS3_ROOT:
        return LOCAL_WORK_DIR
    if root_label == CARC_ATLAS3_ROOT:
        return CARC_WORK_DIR

    root = atlas3_root.expanduser().resolve()
    if root.name == "ATLAS3_Training_Raw":
        root = root.parent
    return root / "nnunet_isle2026"


def is_label_name(name: str) -> bool:
    lower = name.lower()
    return "label-lesion" in lower or "lesion" in lower or "mask" in lower or "seg" in lower


def discover_images(input_path: Path, image_glob: str = DEFAULT_IMAGE_GLOB) -> list[Path]:
    input_path = input_path.expanduser()
    if input_path.is_file():
        return [input_path.resolve()]

    files = sorted(input_path.rglob(image_glob))
    if files:
        return [p.resolve() for p in files]

    fallback = []
    for path in sorted(input_path.rglob("*T1w.nii*")):
        if path.name.endswith((".nii", ".nii.gz")) and not is_label_name(path.name):
            fallback.append(path.resolve())
    return fallback


def find_label_for_image(image: Path) -> Path | None:
    if image.name.endswith(DEFAULT_IMAGE_SUFFIX):
        candidate = image.with_name(image.name.replace(DEFAULT_IMAGE_SUFFIX, DEFAULT_LABEL_SUFFIX))
        if candidate.exists():
            return candidate.resolve()

    source_id = source_id_from_image(image)
    masks = []
    for candidate in image.parent.glob("*.nii*"):
        if candidate.name.endswith((".nii", ".nii.gz")) and is_label_name(candidate.name):
            if strip_nii_suffix(candidate).startswith(source_id):
                masks.append(candidate)
    if len(masks) == 1:
        return masks[0].resolve()
    return None


def discover_training_cases(
    atlas3_root: Path,
    image_glob: str = DEFAULT_IMAGE_GLOB,
    case_limit: int | None = None,
) -> list[Atlas3Case]:
    raw_root = resolve_atlas3_raw_root(atlas3_root)
    images = discover_images(raw_root, image_glob=image_glob)
    cases: list[Atlas3Case] = []
    used: set[str] = set()
    missing_labels = 0

    for image in images:
        label = find_label_for_image(image)
        if label is None:
            missing_labels += 1
            continue

        source_id = source_id_from_image(image)
        case_id = unique_case_id(sanitize_case_id(source_id), used)
        site, subject, session = parse_bids_parts(image)
        cases.append(
            Atlas3Case(
                case_id=case_id,
                image=image,
                label=label,
                source_id=source_id,
                site=site,
                subject=subject,
                session=session,
            )
        )
        if case_limit is not None and len(cases) >= case_limit:
            break

    if missing_labels:
        print(f"Skipped {missing_labels} image(s) without a matching lesion mask.")
    return cases


def discover_inference_cases(
    input_path: Path,
    image_glob: str = DEFAULT_IMAGE_GLOB,
    case_limit: int | None = None,
) -> list[Atlas3Case]:
    images = discover_images(input_path, image_glob=image_glob)
    cases: list[Atlas3Case] = []
    used: set[str] = set()

    for image in images:
        source_id = source_id_from_image(image)
        case_id = unique_case_id(sanitize_case_id(source_id), used)
        site, subject, session = parse_bids_parts(image)
        cases.append(
            Atlas3Case(
                case_id=case_id,
                image=image,
                label=None,
                source_id=source_id,
                site=site,
                subject=subject,
                session=session,
            )
        )
        if case_limit is not None and len(cases) >= case_limit:
            break
    return cases


def validate_case_geometry(cases: Iterable[Atlas3Case]) -> None:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("nibabel and numpy are required for geometry checks") from exc

    bad_cases: list[str] = []
    for case in cases:
        if case.label is None:
            continue
        image_nii = nib.load(str(case.image))
        label_nii = nib.load(str(case.label))
        same_shape = image_nii.shape == label_nii.shape
        same_affine = np.allclose(image_nii.affine, label_nii.affine, atol=1e-4)
        if not (same_shape and same_affine):
            bad_cases.append(
                f"{case.case_id}: image shape/affine {image_nii.shape} does not match label {label_nii.shape}"
            )

    if bad_cases:
        preview = "\n".join(bad_cases[:10])
        raise ValueError(
            "Image/label geometry mismatch. Use --target-spacing to rewrite pairs "
            "onto a shared grid, or inspect the source data.\n" + preview
        )


def write_dataset_json(dataset_folder: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "stroke_lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": DEFAULT_DATASET_NAME,
        "description": "ATLAS3 raw subject-space T1w stroke lesion segmentation",
        "reference": "ATLAS3",
        "licence": "See ATLAS3 data use terms",
        "release": "3",
    }
    with (dataset_folder / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)
        f.write("\n")


def write_mapping(mapping_path: Path, cases: Sequence[Atlas3Case]) -> None:
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "source_id",
                "site",
                "subject",
                "session",
                "image",
                "label",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "source_id": case.source_id,
                    "site": case.site,
                    "subject": case.subject,
                    "session": case.session,
                    "image": str(case.image),
                    "label": str(case.label) if case.label else "",
                }
            )


def install_file(src: Path, dst: Path, link_mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def make_resampled_reference(image, target_spacing: Sequence[float]):
    import SimpleITK as sitk

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    output_size = [
        max(1, int(round(size * spacing / target)))
        for size, spacing, target in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(v) for v in target_spacing))
    resampler.SetSize(output_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler


def resample_pair_to_spacing(
    image_path: Path,
    label_path: Path,
    output_image: Path,
    output_label: Path,
    target_spacing: Sequence[float],
) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))

    image_resampler = make_resampled_reference(image, target_spacing)
    image_resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = sitk.Cast(image_resampler.Execute(image), sitk.sitkFloat32)

    label_resampler = sitk.ResampleImageFilter()
    label_resampler.SetReferenceImage(resampled_image)
    label_resampler.SetTransform(sitk.Transform())
    label_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    label_resampler.SetDefaultPixelValue(0)
    resampled_label = sitk.Cast(label_resampler.Execute(label) > 0, sitk.sitkUInt8)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled_image, str(output_image), True)
    sitk.WriteImage(resampled_label, str(output_label), True)


def resample_image_to_spacing(image_path: Path, output_image: Path, target_spacing: Sequence[float]) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    image_resampler = make_resampled_reference(image, target_spacing)
    image_resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = sitk.Cast(image_resampler.Execute(image), sitk.sitkFloat32)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled_image, str(output_image), True)


def restore_prediction_to_source_grid(prediction: Path, source_image: Path, output_path: Path) -> None:
    import SimpleITK as sitk

    pred = sitk.ReadImage(str(prediction))
    reference = sitk.ReadImage(str(source_image))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    restored = sitk.Cast(resampler.Execute(pred) > 0, sitk.sitkUInt8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(restored, str(output_path), True)


def nnunet_dataset_folder(nnunet_raw: Path, dataset_id: int) -> Path:
    return nnunet_raw / f"Dataset{dataset_id:03d}_{DEFAULT_DATASET_NAME}"


def resolve_nnunet_dirs(
    work_dir: Path | None,
    nnunet_raw: Path | None,
    nnunet_preprocessed: Path | None,
    nnunet_results: Path | None,
    create: bool = True,
    atlas3_root: Path | None = None,
) -> NnUNetDirs:
    if work_dir is None:
        work_dir_env = os.environ.get("WORK_DIR")
        if work_dir_env:
            work_dir = Path(work_dir_env)
        elif nnunet_raw is None and nnunet_preprocessed is None and nnunet_results is None:
            env_raw = os.environ.get("nnUNet_raw")
            env_preprocessed = os.environ.get("nnUNet_preprocessed")
            env_results = os.environ.get("nnUNet_results")
            if env_raw is None and env_preprocessed is None and env_results is None:
                if atlas3_root is not None:
                    work_dir = default_work_dir_for_atlas3_root(atlas3_root)
                else:
                    for candidate in WORK_DIR_CANDIDATES:
                        if candidate.is_dir() or candidate.parent.is_dir():
                            work_dir = candidate
                            break
                    if work_dir is None:
                        raise ValueError(
                            "Could not choose a default work directory. Set --work-dir or WORK_DIR. "
                            f"Checked: {format_path_candidates(WORK_DIR_CANDIDATES)}"
                        )

    if work_dir is not None:
        work_dir = work_dir.expanduser().resolve()
        raw = nnunet_raw or work_dir / "nnUNet_raw"
        preprocessed = nnunet_preprocessed or work_dir / "nnUNet_preprocessed"
        results = nnunet_results or work_dir / "nnUNet_results"
    else:
        raw = nnunet_raw or os.environ.get("nnUNet_raw")
        preprocessed = nnunet_preprocessed or os.environ.get("nnUNet_preprocessed")
        results = nnunet_results or os.environ.get("nnUNet_results")

    if raw is None or preprocessed is None or results is None:
        raise ValueError(
            "Set --work-dir, or provide all three nnUNet dirs/env vars: "
            "nnUNet_raw, nnUNet_preprocessed, nnUNet_results."
        )

    dirs = NnUNetDirs(
        Path(raw).expanduser().resolve(),
        Path(preprocessed).expanduser().resolve(),
        Path(results).expanduser().resolve(),
    )
    if create:
        dirs.raw.mkdir(parents=True, exist_ok=True)
        dirs.preprocessed.mkdir(parents=True, exist_ok=True)
        dirs.results.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(dirs.raw)
    os.environ["nnUNet_preprocessed"] = str(dirs.preprocessed)
    os.environ["nnUNet_results"] = str(dirs.results)
    return dirs


def prepare_dataset(args: argparse.Namespace) -> None:
    atlas3_root = resolve_atlas3_root(args.atlas3_root)
    dirs = resolve_nnunet_dirs(
        args.work_dir,
        args.nnunet_raw,
        args.nnunet_preprocessed,
        args.nnunet_results,
        create=not args.dry_run,
        atlas3_root=atlas3_root,
    )
    raw_root = resolve_atlas3_raw_root(atlas3_root)
    cases = discover_training_cases(raw_root, args.image_glob, args.case_limit)

    print(f"ATLAS3 root: {atlas3_root}")
    print(f"ATLAS3 raw root: {raw_root}")
    print(f"Discovered {len(cases)} image/label pairs.")
    if not cases:
        raise SystemExit("No usable ATLAS3 raw training cases were found.")

    if not args.skip_geometry_check:
        validate_case_geometry(cases)
        print("Image/label geometry check passed.")

    dataset_folder = nnunet_dataset_folder(dirs.raw, args.dataset_id)
    print(f"nnUNet dataset: {dataset_folder}")

    if args.dry_run:
        for case in cases[:5]:
            print(f"  {case.case_id}: {case.image.name} / {case.label.name if case.label else ''}")
        if len(cases) > 5:
            print(f"  ... {len(cases) - 5} more")
        return

    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        assert case.label is not None
        output_image = images_tr / f"{case.case_id}_0000.nii.gz"
        output_label = labels_tr / f"{case.case_id}.nii.gz"
        if args.target_spacing:
            resample_pair_to_spacing(case.image, case.label, output_image, output_label, args.target_spacing)
        else:
            install_file(case.image, output_image, args.link_mode, args.overwrite)
            install_file(case.label, output_label, args.link_mode, args.overwrite)
        if index % 100 == 0 or index == len(cases):
            print(f"Prepared {index}/{len(cases)} cases")

    write_dataset_json(dataset_folder, len(cases))
    write_mapping(dataset_folder / "case_mapping.tsv", cases)

    print("Preparation complete.")
    print(f"Set nnUNet_raw={dirs.raw}")
    print(f"Run planning next: nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")


def run_command(cmd: Sequence[str]) -> None:
    print("$ " + shlex.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def add_plan_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    if args.configuration:
        cmd.extend(["-c", args.configuration])
    if args.num_processes is not None:
        cmd.extend(["-np", str(args.num_processes)])
    if args.verify_dataset_integrity:
        cmd.append("--verify_dataset_integrity")
    return cmd


def plan_dataset(args: argparse.Namespace) -> None:
    resolve_nnunet_dirs(args.work_dir, args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results)
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id)]
    run_command(add_plan_args(cmd, args))


def train_dataset(args: argparse.Namespace) -> None:
    resolve_nnunet_dirs(args.work_dir, args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results)
    cmd = [
        "nnUNetv2_train",
        str(args.dataset_id),
        args.configuration,
        str(args.fold),
        "-tr",
        args.trainer,
        "-p",
        args.plans,
        "-device",
        args.device,
    ]
    if args.num_gpus is not None:
        cmd.extend(["-num_gpus", str(args.num_gpus)])
    if args.continue_training:
        cmd.append("--c")
    if args.val:
        cmd.append("--val")
    if args.npz:
        cmd.append("--npz")
    run_command(cmd)


def parse_folds(folds: Sequence[str]) -> list[str]:
    parsed: list[str] = []
    for item in folds:
        parsed.extend(part for part in item.split(",") if part)
    return parsed or ["0"]


def prepare_prediction_input(args: argparse.Namespace) -> tuple[Path, Path, list[Atlas3Case]]:
    output_root = args.output.expanduser().resolve()
    input_dir = output_root / "nnunet_input"
    mapping_path = output_root / "case_mapping.tsv"
    cases = discover_inference_cases(args.input, args.image_glob, args.case_limit)

    if not cases:
        raise SystemExit(f"No T1w NIfTI files were found under {args.input}")

    input_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        output_image = input_dir / f"{case.case_id}_0000.nii.gz"
        if args.target_spacing:
            resample_image_to_spacing(case.image, output_image, args.target_spacing)
        else:
            install_file(case.image, output_image, args.link_mode, args.overwrite)

    write_mapping(mapping_path, cases)
    return input_dir, mapping_path, cases


def organized_prediction_name(case: Atlas3Case) -> str:
    return f"{sanitize_case_id(case.source_id)}_prediction.nii.gz"


def organize_predictions(
    prediction_dir: Path,
    final_dir: Path,
    cases: Sequence[Atlas3Case],
    restore_to_source_grid: bool,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        prediction = prediction_dir / f"{case.case_id}.nii.gz"
        if not prediction.exists():
            print(f"Warning: missing prediction for {case.case_id}: {prediction}")
            continue
        output_path = final_dir / organized_prediction_name(case)
        if restore_to_source_grid:
            restore_prediction_to_source_grid(prediction, case.image, output_path)
        else:
            shutil.copy2(prediction, output_path)


def predict_dataset(args: argparse.Namespace) -> None:
    resolve_nnunet_dirs(args.work_dir, args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    input_dir, _, cases = prepare_prediction_input(args)
    prediction_dir = args.output / "nnunet_predictions"
    final_dir = args.output / "final_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    folds = parse_folds(args.folds)
    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(prediction_dir),
        "-d",
        str(args.dataset_id),
        "-c",
        args.configuration,
        "-tr",
        args.trainer,
        "-p",
        args.plans,
        "-chk",
        args.checkpoint,
        "-device",
        args.device,
        "-f",
        *folds,
    ]
    if args.save_probabilities:
        cmd.append("--save_probabilities")
    run_command(cmd)
    organize_predictions(prediction_dir, final_dir, cases, args.restore_to_source_grid)
    print(f"Predictions saved to {final_dir}")


def add_common_nnunet_dir_args(parser: argparse.ArgumentParser) -> None:
    work_dir_help = (
        "Root for nnUNet_raw/preprocessed/results. Defaults to WORK_DIR, then first usable of: "
        f"{format_path_candidates(WORK_DIR_CANDIDATES)}"
    )
    parser.add_argument("--work-dir", type=Path, default=None, help=work_dir_help)
    parser.add_argument("--nnunet-raw", type=Path, default=None, help="Override nnUNet_raw")
    parser.add_argument("--nnunet-preprocessed", type=Path, default=None, help="Override nnUNet_preprocessed")
    parser.add_argument("--nnunet-results", type=Path, default=None, help="Override nnUNet_results")


def add_prepare_like_args(parser: argparse.ArgumentParser, include_labels: bool) -> None:
    parser.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB, help="Recursive glob for raw T1w images")
    parser.add_argument("--case-limit", type=int, default=None, help="Limit cases for smoke tests")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing formatted files")
    if include_labels:
        parser.add_argument("--skip-geometry-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run nnUNet v2 on ATLAS3 raw subject-space data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create Dataset###_ATLAS3Raw in nnUNet_raw")
    add_common_nnunet_dir_args(prepare)
    add_prepare_like_args(prepare, include_labels=True)
    atlas3_root_help = (
        "ATLAS3 root. Defaults to ATLAS3_ROOT, then first existing of: "
        f"{format_path_candidates(ATLAS3_ROOT_CANDIDATES)}"
    )
    prepare.add_argument("--atlas3-root", type=Path, default=None, help=atlas3_root_help)
    prepare.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=prepare_dataset)

    plan = subparsers.add_parser("plan", help="Run nnUNetv2_plan_and_preprocess")
    add_common_nnunet_dir_args(plan)
    plan.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    plan.add_argument("--configuration", default="3d_fullres")
    plan.add_argument("--num-processes", type=int, default=None)
    plan.add_argument("--verify-dataset-integrity", action="store_true", default=True)
    plan.set_defaults(func=plan_dataset)

    train = subparsers.add_parser("train", help="Run nnUNetv2_train")
    add_common_nnunet_dir_args(train)
    train.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    train.add_argument("--configuration", default="3d_fullres", choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"])
    train.add_argument("--fold", default="0")
    train.add_argument("--trainer", default="nnUNetTrainer")
    train.add_argument("--plans", default="nnUNetPlans")
    train.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    train.add_argument("--num-gpus", type=int, default=None)
    train.add_argument("--continue-training", action="store_true")
    train.add_argument("--val", action="store_true", help="Validate only")
    train.add_argument("--npz", action="store_true", help="Save softmax npz during validation")
    train.set_defaults(func=train_dataset)

    predict = subparsers.add_parser("predict", help="Format raw T1w images and run nnUNetv2_predict")
    add_common_nnunet_dir_args(predict)
    add_prepare_like_args(predict, include_labels=False)
    predict.add_argument("--input", type=Path, required=True, help="Raw T1w NIfTI or directory")
    predict.add_argument("--output", type=Path, required=True, help="Prediction working/output directory")
    predict.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    predict.add_argument("--configuration", default="3d_fullres", choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"])
    predict.add_argument("--folds", nargs="+", default=["0"], help="Fold list, comma list, or all")
    predict.add_argument("--trainer", default="nnUNetTrainer")
    predict.add_argument("--plans", default="nnUNetPlans")
    predict.add_argument("--checkpoint", default="checkpoint_final.pth")
    predict.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    predict.add_argument("--save-probabilities", action="store_true")
    predict.add_argument("--restore-to-source-grid", action="store_true")
    predict.set_defaults(func=predict_dataset)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()

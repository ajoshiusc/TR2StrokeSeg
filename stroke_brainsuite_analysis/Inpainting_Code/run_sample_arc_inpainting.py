#!/usr/bin/env python3
"""Run the bundled 3D diffusion inpainting model on sample_arc nnU-Net masks.

The preprocessing reproduces Inpainting_Script.ipynb without requiring TorchIO:

* transform the existing BrainSuite-extracted brain to the nnU-Net MNI grid;
* centrally crop/pad to 160 x 192 x 160;
* resample the T1 with cubic B-splines and the label with nearest-neighbor
  interpolation to 2 mm (80 x 96 x 80);
* divide T1 intensities by the lower edge of the dominant non-background
  histogram bin (20 bins), as in the notebook.

Unlike the notebook's save helper, this script records the crop translation and
half-voxel resampling offset in the output affine.  It also resamples generated
tissue to the original 1 mm MNI grid and replaces only lesion voxels, preserving
the original image exactly everywhere else.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to

warnings.filterwarnings("ignore", category=FutureWarning)

from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets.diffusion_model_unet import AttentionBlock
from generative.networks.schedulers import DDPMScheduler


MODEL_CROP_SHAPE = (160, 192, 160)
MODEL_SPACING_MM = 2.0
MODEL_SHAPE = (80, 96, 80)


@dataclass(frozen=True)
class CasePaths:
    index: int
    case_id: str
    full_head_t1_path: Path
    bse_brain_path: Path
    to_mni_mat_path: Path
    mask_path: Path


@dataclass
class PreparedCase:
    paths: CasePaths
    brain_mni_path: Path
    source_image: nib.Nifti1Image
    source_data: np.ndarray
    source_mask: np.ndarray
    model_image: np.ndarray
    model_mask: np.ndarray
    model_affine: np.ndarray
    peak: float
    crop_offset: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_sample = here.parent / "outputs" / "sample_arc"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=default_sample)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <sample-dir>/inpainted",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=here / "Inpainting_model_inpaint_smart_ir_epoch800.pt",
    )
    parser.add_argument(
        "--flirt",
        type=Path,
        default=Path("/home/ajoshi/Software/fsl/bin/flirt"),
        help="FSL flirt executable used to transform the BrainSuite brain to MNI",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="auto uses a float16 U-Net on CUDA with a float32 diffusion state",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Process only this case ID; may be supplied more than once",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Write model-space T1/masks and validate preprocessing without diffusion",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.sample_dir / "inpainted"
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def discover_cases(sample_dir: Path, selected: set[str]) -> list[CasePaths]:
    image_dir = sample_dir / "nnunet_input"
    mask_dir = sample_dir / "nnunet_predictions_mni"
    images = sorted(image_dir.glob("*_0000.nii.gz"))
    if not images:
        raise FileNotFoundError(f"No nnU-Net inputs found in {image_dir}")

    cases: list[CasePaths] = []
    for index, t1_path in enumerate(images):
        case_id = t1_path.name.removesuffix("_0000.nii.gz")
        if selected and case_id not in selected:
            continue
        mask_path = mask_dir / f"{case_id}.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {case_id}: {mask_path}")
        case_work = sample_dir / "work" / case_id
        bse_brain_path = case_work / "brainsuite" / f"{case_id}_bse_brain.nii.gz"
        to_mni_mat_path = case_work / "preprocessed" / f"{case_id}_to_mni.mat"
        if not bse_brain_path.is_file():
            raise FileNotFoundError(f"Missing BrainSuite brain for {case_id}: {bse_brain_path}")
        if not to_mni_mat_path.is_file():
            raise FileNotFoundError(f"Missing MNI transform for {case_id}: {to_mni_mat_path}")
        cases.append(
            CasePaths(
                index,
                case_id,
                t1_path,
                bse_brain_path,
                to_mni_mat_path,
                mask_path,
            )
        )

    found = {case.case_id for case in cases}
    missing = selected - found
    if missing:
        raise ValueError(f"Requested cases not found: {', '.join(sorted(missing))}")
    return cases


def histogram_peak(data: np.ndarray) -> float:
    # normalize_by_peak() in the notebook clips a temporary copy at zero and
    # deliberately ignores the background histogram bin.
    clipped = np.maximum(data, 0)
    hist, bins = np.histogram(clipped, bins=20)
    peak = float(bins[int(np.argmax(hist[1:])) + 1])
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError(f"Invalid histogram peak: {peak}")
    return peak


def center_crop_or_pad(
    data: np.ndarray, target: Sequence[int]
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Match TorchIO CropOrPad and return the target-to-input voxel offset."""
    slices: list[slice] = []
    pads: list[tuple[int, int]] = []
    offsets: list[int] = []
    for size, wanted in zip(data.shape, target, strict=True):
        delta = size - wanted
        if delta >= 0:
            # TorchIO assigns an odd extra crop voxel to the low-index side.
            low = (delta + 1) // 2
            high = delta - low
            slices.append(slice(low, size - high))
            pads.append((0, 0))
            offsets.append(low)
        else:
            total = -delta
            low = (total + 1) // 2
            high = total - low
            slices.append(slice(None))
            pads.append((low, high))
            offsets.append(-low)
    result = np.pad(data[tuple(slices)], pads, mode="constant")
    if result.shape != tuple(target):
        raise RuntimeError(f"Crop/pad produced {result.shape}, expected {tuple(target)}")
    return np.ascontiguousarray(result), tuple(offsets)


def resample_model_grid(data: np.ndarray, is_label: bool) -> np.ndarray:
    """Reproduce TorchIO Resample(2) for an identity-affine cropped array."""
    image = sitk.GetImageFromArray(np.asarray(data, dtype=np.float32).transpose(2, 1, 0))
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    output = sitk.Resample(
        image,
        MODEL_SHAPE,
        sitk.Transform(),
        interpolator,
        (0.5, 0.5, 0.5),
        (MODEL_SPACING_MM,) * 3,
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        0.0,
        sitk.sitkFloat32,
    )
    array = sitk.GetArrayFromImage(output).transpose(2, 1, 0)
    return np.ascontiguousarray(array)


def brain_mni_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "preprocessed_mni_1mm" / f"{case_id}_brain_mni_1mm.nii.gz"


def transform_brain_to_mni(
    paths: CasePaths,
    output_dir: Path,
    flirt: Path,
    overwrite: bool,
) -> Path:
    """Apply the already-estimated source-to-MNI transform to the BSE brain."""
    output_path = brain_mni_path(output_dir, paths.case_id)
    if output_path.is_file() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("FSLDIR", str(flirt.parent.parent))
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    command = [
        str(flirt),
        "-in",
        str(paths.bse_brain_path),
        "-ref",
        str(paths.full_head_t1_path),
        "-applyxfm",
        "-init",
        str(paths.to_mni_mat_path),
        "-interp",
        "trilinear",
        "-out",
        str(output_path),
    ]
    subprocess.run(command, check=True, env=env)
    return output_path


def prepare_case(
    paths: CasePaths,
    output_dir: Path,
    flirt: Path,
    overwrite_preprocessing: bool,
) -> PreparedCase:
    input_path = transform_brain_to_mni(
        paths, output_dir, flirt, overwrite_preprocessing
    )
    image = nib.load(input_path)
    mask_image = nib.load(paths.mask_path)
    if image.shape != mask_image.shape:
        raise ValueError(
            f"{paths.case_id}: T1 shape {image.shape} != mask shape {mask_image.shape}"
        )
    if not np.allclose(image.affine, mask_image.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{paths.case_id}: T1 and mask affines differ")

    data = np.asarray(image.dataobj, dtype=np.float32)
    raw_mask = np.asarray(mask_image.dataobj)
    unique = np.unique(raw_mask)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{paths.case_id}: expected a binary mask, found {unique.tolist()}")
    mask = np.asarray(raw_mask > 0, dtype=np.uint8)
    if not mask.any():
        raise ValueError(f"{paths.case_id}: lesion mask is empty")

    raw_peak = histogram_peak(data.copy())
    # The notebook casts the collated peak tensor to long before division.
    peak = float(int(raw_peak))
    if peak <= 0:
        raise ValueError(f"{paths.case_id}: integer histogram peak is not positive")
    cropped_data, offset = center_crop_or_pad(data, MODEL_CROP_SHAPE)
    cropped_mask, mask_offset = center_crop_or_pad(mask, MODEL_CROP_SHAPE)
    if offset != mask_offset:
        raise RuntimeError("Image and mask crop offsets unexpectedly differ")
    if int(cropped_mask.sum()) != int(mask.sum()):
        lost = int(mask.sum()) - int(cropped_mask.sum())
        raise ValueError(f"{paths.case_id}: central crop would remove {lost} lesion voxels")

    model_image = resample_model_grid(cropped_data, is_label=False) / peak
    model_mask = np.asarray(resample_model_grid(cropped_mask, is_label=True) > 0.5, dtype=np.uint8)
    if not model_mask.any():
        raise ValueError(f"{paths.case_id}: lesion vanished during 2 mm resampling")
    model_mean = float(model_image.mean())
    if not 0.18 <= model_mean <= 0.45:
        raise ValueError(
            f"{paths.case_id}: normalized model-input mean {model_mean:.3f} is outside "
            "the 0.18-0.45 range; check skull stripping and intensity normalization"
        )

    # A model-grid voxel i samples input voxel 2*i + offset + 0.5.
    voxel_map = np.eye(4, dtype=np.float64)
    voxel_map[:3, :3] *= MODEL_SPACING_MM
    voxel_map[:3, 3] = np.asarray(offset, dtype=np.float64) + 0.5
    model_affine = image.affine @ voxel_map

    return PreparedCase(
        paths=paths,
        brain_mni_path=input_path,
        source_image=image,
        source_data=data,
        source_mask=mask,
        model_image=np.asarray(model_image, dtype=np.float32),
        model_mask=model_mask,
        model_affine=model_affine,
        peak=peak,
        crop_offset=offset,
    )


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: Path,
    dtype: np.dtype,
    header: nib.Nifti1Header | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=dtype)
    output_header = None if header is None else header.copy()
    if output_header is not None:
        output_header.set_data_dtype(dtype)
    image = nib.Nifti1Image(array, affine, header=output_header)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)


def output_paths(output_dir: Path, case_id: str) -> dict[str, Path]:
    return {
        "model_input": output_dir / "model_space" / f"{case_id}_model_input.nii.gz",
        "model_mask": output_dir / "model_space" / f"{case_id}_stroke_model_mask.nii.gz",
        "model_inpainted": output_dir / "model_space" / f"{case_id}_inpainted_model_space.nii.gz",
        "mni_inpainted": output_dir / "mni_1mm" / f"{case_id}_inpainted_mni_1mm.nii.gz",
    }


def save_preprocessed(case: PreparedCase, output_dir: Path) -> None:
    paths = output_paths(output_dir, case.paths.case_id)
    save_nifti(case.model_image, case.model_affine, paths["model_input"], np.float32)
    save_nifti(case.model_mask, case.model_affine, paths["model_mask"], np.uint8)


def enable_memory_efficient_attention(dtype: torch.dtype) -> None:
    # AttentionBlock normally materializes an LxL matrix. Half precision can use
    # a fused CUDA kernel. Float32 is evaluated in query chunks using the exact
    # original baddbmm -> softmax -> bmm operations.
    def fused_attention(
        self: AttentionBlock,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1),
            dropout_p=0.0,
            scale=self.scale,
        ).squeeze(1)

    def chunked_attention(
        self: AttentionBlock,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty_like(query)
        key_transposed = key.transpose(-1, -2)
        chunk_size = 10000
        for start in range(0, query.shape[1], chunk_size):
            query_chunk = query[:, start : start + chunk_size]
            scores = torch.baddbmm(
                torch.empty(
                    query.shape[0],
                    query_chunk.shape[1],
                    key.shape[1],
                    dtype=query.dtype,
                    device=query.device,
                ),
                query_chunk,
                key_transposed,
                beta=0,
                alpha=self.scale,
            )
            result[:, start : start + chunk_size] = torch.bmm(
                scores.softmax(dim=-1), value
            )
        return result

    AttentionBlock._attention = fused_attention if dtype == torch.float16 else chunked_attention


def enable_cpu_offloaded_skips(model: DiffusionModelUNet, device: torch.device) -> None:
    """Keep float32 U-Net skip tensors on CPU until their matching up block."""
    if device.type != "cuda":
        return

    def offload_down_residuals(module, args, output):
        del module, args
        hidden, residuals = output
        return hidden, tuple(residual.cpu() for residual in residuals)

    def load_up_residuals(module, args, kwargs):
        del module
        residuals = kwargs["res_hidden_states_list"]
        kwargs["res_hidden_states_list"] = tuple(
            residual.to(device=device) for residual in residuals
        )
        return args, kwargs

    for block in model.down_blocks:
        block.register_forward_hook(offload_down_residuals)
    for block in model.up_blocks:
        block.register_forward_pre_hook(load_up_residuals, with_kwargs=True)


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> DiffusionModelUNet:
    enable_memory_efficient_attention(dtype)
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        num_channels=[32, 64, 64],
        attention_levels=[False, False, True],
        num_head_channels=[0, 0, 32],
        num_res_blocks=2,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    if dtype == torch.float32:
        enable_cpu_offloaded_skips(model, device)
    return model


def inpaint_batch(
    cases: list[PreparedCase],
    model: DiffusionModelUNet,
    device: torch.device,
    model_dtype: torch.dtype,
    steps: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    images = torch.from_numpy(np.stack([case.model_image for case in cases]))[:, None]
    lesions = torch.from_numpy(np.stack([case.model_mask for case in cases]))[:, None]
    # Keep the stochastic diffusion state in float32. A 1,000-step DDPM chain
    # accumulates unacceptable error in float16 even when every value remains
    # finite. Only the U-Net forward is cast to float16 for GPU memory savings.
    images = images.to(device=device, dtype=torch.float32)
    lesions = lesions.to(device=device, dtype=torch.float32)
    keep = 1.0 - lesions

    current = torch.randn(images.shape, device=device, dtype=torch.float32)
    made = images * keep + lesions * current

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0005,
        beta_end=0.0195,
    )
    scheduler.set_timesteps(num_inference_steps=steps)

    started = time.monotonic()
    with torch.inference_mode():
        for step_index, timestep in enumerate(scheduler.timesteps, start=1):
            combined = torch.cat((made, keep), dim=1).to(dtype=model_dtype)
            model_timesteps = torch.full(
                (len(cases),), int(timestep), device=device, dtype=torch.long
            )
            model_output = model(combined, timesteps=model_timesteps).float()
            current, _ = scheduler.step(model_output, timestep, made)
            made = images * keep + lesions * current
            if step_index == 1 or step_index % 100 == 0 or step_index == steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                rate = elapsed / step_index
                remaining = rate * (steps - step_index)
                names = ",".join(case.paths.case_id for case in cases)
                print(
                    f"[{names}] step {step_index:04d}/{steps} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                    flush=True,
                )

    if not bool(torch.isfinite(made).all()):
        raise FloatingPointError("Diffusion output contains non-finite values")
    return made[:, 0].float().cpu().numpy()


def save_inpainted(
    case: PreparedCase, model_output: np.ndarray, output_dir: Path
) -> dict[str, object]:
    paths = output_paths(output_dir, case.paths.case_id)
    save_nifti(model_output, case.model_affine, paths["model_inpainted"], np.float32)

    generated_scaled = np.asarray(model_output * case.peak, dtype=np.float32)
    generated_image = nib.Nifti1Image(generated_scaled, case.model_affine)
    generated_1mm = np.asarray(
        resample_from_to(
            generated_image,
            (case.source_image.shape, case.source_image.affine),
            order=3,
            mode="constant",
            cval=0.0,
        ).dataobj,
        dtype=np.float32,
    )
    composite = case.source_data.copy()
    lesion = case.source_mask.astype(bool)
    composite[lesion] = generated_1mm[lesion]
    save_nifti(
        composite,
        case.source_image.affine,
        paths["mni_inpainted"],
        np.float32,
        header=case.source_image.header,
    )

    outside_max_abs = float(np.max(np.abs(composite[~lesion] - case.source_data[~lesion])))
    inside = composite[lesion]
    return {
        "case_id": case.paths.case_id,
        "brain_mni_path": str(case.brain_mni_path),
        "full_head_t1_path": str(case.paths.full_head_t1_path),
        "bse_brain_path": str(case.paths.bse_brain_path),
        "mask_path": str(case.paths.mask_path),
        "model_input": str(paths["model_input"]),
        "model_mask": str(paths["model_mask"]),
        "model_inpainted": str(paths["model_inpainted"]),
        "mni_inpainted": str(paths["mni_inpainted"]),
        "source_shape": "x".join(str(value) for value in case.source_image.shape),
        "model_shape": "x".join(str(value) for value in MODEL_SHAPE),
        "crop_offset": "x".join(str(value) for value in case.crop_offset),
        "peak": case.peak,
        "source_lesion_voxels": int(case.source_mask.sum()),
        "model_lesion_voxels": int(case.model_mask.sum()),
        "inpainted_lesion_min": float(inside.min()),
        "inpainted_lesion_max": float(inside.max()),
        "inpainted_lesion_mean": float(inside.mean()),
        "outside_lesion_max_abs_change": outside_max_abs,
        "finite": bool(np.isfinite(composite).all()),
    }


def write_manifest(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    path = output_dir / "inpainting_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run_metadata(args: argparse.Namespace, cases: list[CasePaths], dtype: torch.dtype) -> None:
    metadata = {
        "sample_dir": str(args.sample_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "flirt": str(args.flirt.resolve()),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "cases": [case.case_id for case in cases],
        "num_cases": len(cases),
        "crop_shape": MODEL_CROP_SHAPE,
        "model_shape": MODEL_SHAPE,
        "model_spacing_mm": MODEL_SPACING_MM,
        "histogram_bins": 20,
        "histogram_peak_cast": "truncate_to_integer_like_notebook_long_tensor",
        "input_is_brainsuite_brain": True,
        "diffusion_steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(args.device),
        "dtype": str(dtype),
        "diffusion_state_dtype": str(torch.float32),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    args.sample_dir = args.sample_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.flirt = args.flirt.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.flirt.is_file():
        raise FileNotFoundError(args.flirt)

    cases = discover_cases(args.sample_dir, set(args.case))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available() and not args.preprocess_only:
        raise RuntimeError("CUDA was requested but is not available")
    precision = args.precision
    if precision == "auto":
        precision = "float16" if device.type == "cuda" else "float32"
    dtype = torch.float16 if precision == "float16" else torch.float32
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is not supported")

    print(f"Discovered {len(cases)} case(s) in {args.sample_dir}", flush=True)
    print(f"Writing results to {args.output_dir}", flush=True)
    write_run_metadata(args, cases, dtype)

    if args.preprocess_only:
        for case_paths in cases:
            case = prepare_case(
                case_paths,
                args.output_dir,
                args.flirt,
                args.overwrite,
            )
            save_preprocessed(case, args.output_dir)
            print(
                f"[{case_paths.case_id}] peak={case.peak:.6g} "
                f"mean={float(case.model_image.mean()):.3f} "
                f"lesion={int(case.source_mask.sum())} -> {int(case.model_mask.sum())} voxels",
                flush=True,
            )
        return 0

    model = load_model(args.checkpoint, device, dtype)
    rows: list[dict[str, object]] = []
    total_batches = math.ceil(len(cases) / args.batch_size)
    for batch_number, start in enumerate(range(0, len(cases), args.batch_size), start=1):
        batch_paths = cases[start : start + args.batch_size]
        completed = [
            output_paths(args.output_dir, case.case_id)["mni_inpainted"].is_file()
            and output_paths(args.output_dir, case.case_id)["model_inpainted"].is_file()
            for case in batch_paths
        ]
        if all(completed) and not args.overwrite:
            print(f"Skipping completed batch {batch_number}/{total_batches}", flush=True)
            continue

        print(f"Preparing batch {batch_number}/{total_batches}", flush=True)
        prepared = [
            prepare_case(case, args.output_dir, args.flirt, args.overwrite)
            for case in batch_paths
        ]
        for case in prepared:
            save_preprocessed(case, args.output_dir)
            print(
                f"[{case.paths.case_id}] peak={case.peak:.6g} "
                f"mean={float(case.model_image.mean()):.3f} "
                f"lesion={int(case.source_mask.sum())} -> {int(case.model_mask.sum())} voxels",
                flush=True,
            )

        outputs = inpaint_batch(
            prepared,
            model,
            device,
            dtype,
            args.steps,
            args.seed + batch_paths[0].index,
        )
        for case, output in zip(prepared, outputs, strict=True):
            row = save_inpainted(case, output, args.output_dir)
            rows.append(row)
            print(f"[{case.paths.case_id}] saved", flush=True)
        write_manifest(rows, args.output_dir)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Rebuild the manifest from current outputs when resuming is deliberately
    # left to the validation pass, which has access to every saved volume.
    print("Inpainting complete", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed batches remain on disk and can be resumed", file=sys.stderr)
        raise SystemExit(130)

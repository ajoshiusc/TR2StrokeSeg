#!/usr/bin/env python3
"""Create representative ARC MRI, label, and surface panels by lesion size.

Cases are selected objectively as the completed matched acquisitions closest to
the cohort 10th, 50th, and 90th percentiles of source-lesion volume. Direct
BrainSuite labels are carried from native space to the pre-inpainting MNI grid
with the affine used by the inpainting workflow. No generated-region quantity
is calculated: the figure is a qualitative illustration of pipeline outputs.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv

import analyze_arc_inpainting as arc


DEFAULT_ARC_ROOT = Path("/home/ajoshi/project2_ajoshi_1183/data/ARC")
DEFAULT_OUTPUT_DIR = Path("analysis/arc_inpainting/representative_outputs")
QUANTILES = (("Small", 0.10), ("Medium", 0.50), ("Large", 0.90))


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    direct_stats: Path
    inpainted_stats: Path
    direct_label: Path
    inpainted_label: Path
    direct_surface: Path
    inpainted_surface: Path
    original_mri: Path
    inpainted_mri: Path
    lesion_mask: Path
    native_to_mni: Path
    label_description: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-root", type=Path, default=DEFAULT_ARC_ROOT)
    parser.add_argument(
        "--case-metrics",
        type=Path,
        default=Path("analysis/arc_inpainting/case_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--flirt",
        type=Path,
        default=Path(
            shutil.which("flirt")
            or "/home/ajoshi/Software/fsl/share/fsl/bin/flirt"
        ),
        help="FSL FLIRT executable used to map direct labels to the MNI grid.",
    )
    return parser.parse_args()


def readdfs(path: Path) -> SimpleNamespace:
    """Read the geometry and vertex colors needed from a BrainSuite DFS file."""
    surface = SimpleNamespace()
    with path.open("rb") as stream:
        header = stream.read(12)
        if b"DFS" not in header:
            raise ValueError(f"Not a BrainSuite DFS file: {path}")
        header_size = struct.unpack("i", stream.read(4))[0]
        stream.read(8)  # metadata and patient-data offsets
        triangle_count = struct.unpack("i", stream.read(4))[0]
        vertex_count = struct.unpack("i", stream.read(4))[0]
        stream.read(8)  # strip count and strip size
        normals_offset = struct.unpack("i", stream.read(4))[0]
        uv_offset = struct.unpack("i", stream.read(4))[0]
        color_offset = struct.unpack("i", stream.read(4))[0]
        label_offset = struct.unpack("i", stream.read(4))[0]
        attribute_offset = struct.unpack("i", stream.read(4))[0]
        del normals_offset, uv_offset, label_offset, attribute_offset
        stream.seek(header_size)
        surface.faces = np.frombuffer(
            stream.read(triangle_count * 3 * 4), dtype=np.int32
        ).reshape(triangle_count, 3)
        surface.vertices = np.frombuffer(
            stream.read(vertex_count * 3 * 4), dtype=np.float32
        ).reshape(vertex_count, 3)
        if color_offset <= 0:
            raise ValueError(f"DFS file has no vertex colors: {path}")
        stream.seek(color_offset)
        surface.colors = np.frombuffer(
            stream.read(vertex_count * 3 * 4), dtype=np.float32
        ).reshape(vertex_count, 3)
    return surface


def paths_for_case(
    case_id: str,
    raw_files: dict[str, Path],
    inpainted_files: dict[str, Path],
    inpainting_root: Path,
) -> CasePaths:
    direct_stats = raw_files[case_id]
    inpainted_stats = inpainted_files[case_id]
    direct_prefix = direct_stats.name[: -len(arc.STATS_SUFFIX)]
    inpainted_prefix = inpainted_stats.name[: -len(arc.STATS_SUFFIX)]
    case_dir = arc.inpainting_case_dir(inpainting_root, case_id)
    return CasePaths(
        case_id=case_id,
        direct_stats=direct_stats,
        inpainted_stats=inpainted_stats,
        direct_label=direct_stats.with_name(f"{direct_prefix}.svreg.label.nii.gz"),
        inpainted_label=inpainted_stats.with_name(
            f"{inpainted_prefix}.svreg.label.nii.gz"
        ),
        direct_surface=direct_stats.with_name(
            f"{direct_prefix}.left.mid.cortex.svreg.dfs"
        ),
        inpainted_surface=inpainted_stats.with_name(
            f"{inpainted_prefix}.left.mid.cortex.svreg.dfs"
        ),
        original_mri=case_dir / f"{case_id}_brain_mni_1mm.nii.gz",
        inpainted_mri=case_dir / f"{case_id}_brain_inpainted_mni_1mm.nii.gz",
        lesion_mask=case_dir / f"{case_id}_stroke_mask_mni_1mm.nii.gz",
        native_to_mni=case_dir / "work" / f"{case_id}_to_mni.mat",
        label_description=direct_stats.with_name("brainsuite_labeldescription.xml"),
    )


def complete(paths: CasePaths) -> bool:
    return all(
        path.is_file()
        for path in (
            paths.direct_label,
            paths.inpainted_label,
            paths.direct_surface,
            paths.inpainted_surface,
            paths.original_mri,
            paths.inpainted_mri,
            paths.lesion_mask,
            paths.native_to_mni,
            paths.label_description,
        )
    )


def select_cases(
    metrics: pd.DataFrame,
    raw_files: dict[str, Path],
    inpainted_files: dict[str, Path],
    inpainting_root: Path,
) -> list[tuple[str, float, float, CasePaths]]:
    selected = []
    used: set[str] = set()
    for size, quantile in QUANTILES:
        target = float(metrics["lesion_volume_ml"].quantile(quantile))
        candidates = metrics.assign(
            distance=(metrics["lesion_volume_ml"] - target).abs()
        ).sort_values(["distance", "case_id"])
        for row in candidates.itertuples(index=False):
            if row.case_id in used:
                continue
            if row.case_id not in raw_files or row.case_id not in inpainted_files:
                continue
            paths = paths_for_case(
                row.case_id, raw_files, inpainted_files, inpainting_root
            )
            if complete(paths):
                selected.append(
                    (size, quantile, float(row.lesion_volume_ml), paths)
                )
                used.add(row.case_id)
                break
        else:
            raise RuntimeError(f"No complete case found near quantile {quantile}")
    return selected


def load_volume(path: Path, dtype: np.dtype | None = None) -> np.ndarray:
    data = np.asanyarray(nib.load(path).dataobj)
    return data.astype(dtype, copy=False) if dtype is not None else data


def map_direct_label(paths: CasePaths, flirt: Path, output_path: Path) -> np.ndarray:
    subprocess.run(
        [
            str(flirt),
            "-in",
            str(paths.direct_label),
            "-ref",
            str(paths.original_mri),
            "-applyxfm",
            "-init",
            str(paths.native_to_mni),
            "-interp",
            "nearestneighbour",
            "-out",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return load_volume(output_path, np.int32)


def label_colors(path: Path) -> dict[int, np.ndarray]:
    colors: dict[int, np.ndarray] = {}
    for element in ET.parse(path).getroot().findall("label"):
        label = int(element.attrib["id"])
        packed = int(element.attrib["color"], 16)
        colors[label] = np.array(
            [(packed >> 16) & 255, (packed >> 8) & 255, packed & 255],
            dtype=float,
        ) / 255.0
    return colors


def rgba_labels(labels: np.ndarray, colors: dict[int, np.ndarray]) -> np.ndarray:
    rgba = np.zeros((*labels.shape, 4), dtype=float)
    for label in np.unique(labels):
        if label == 0:
            continue
        color = colors.get(int(label))
        if color is None:
            canonical = int(label % 1000) if label >= 1000 else int(label)
            color = colors.get(canonical, np.array([0.95, 0.65, 0.10]))
        mask = labels == label
        rgba[mask, :3] = color
        rgba[mask, 3] = 0.52
    return rgba


def axial(array: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(array[:, :, index])


def display_crop(brain: np.ndarray) -> tuple[slice, slice]:
    coordinates = np.argwhere(brain > 0)
    if coordinates.size == 0:
        return slice(None), slice(None)
    low = np.maximum(coordinates.min(axis=0) - 4, 0)
    high = np.minimum(coordinates.max(axis=0) + 5, brain.shape)
    return slice(low[0], high[0]), slice(low[1], high[1])


def render_surface(surface_path: Path, screenshot_path: Path) -> np.ndarray:
    surface = readdfs(surface_path)
    faces = np.column_stack(
        [np.full(len(surface.faces), 3, dtype=np.int64), surface.faces]
    ).ravel()
    mesh = pv.PolyData(surface.vertices, faces)
    mesh.point_data["label_colors"] = (
        np.clip(surface.colors, 0, 1) * 255
    ).astype(np.uint8)
    plotter = pv.Plotter(off_screen=True, window_size=(720, 570))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="label_colors",
        rgb=True,
        smooth_shading=True,
        show_edges=False,
        ambient=0.30,
        diffuse=0.72,
        specular=0.04,
    )
    plotter.view_yz(negative=True)
    plotter.reset_camera()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.15)
    plotter.show(screenshot=str(screenshot_path), auto_close=True)
    image = plt.imread(screenshot_path)
    rgb = image[..., :3]
    foreground = np.any(rgb < 0.98, axis=2)
    coordinates = np.argwhere(foreground)
    if coordinates.size:
        low = np.maximum(coordinates.min(axis=0) - 8, 0)
        high = np.minimum(coordinates.max(axis=0) + 9, foreground.shape)
        image = image[low[0] : high[0], low[1] : high[1]]
    return image


def show_mri(
    axis: plt.Axes,
    image: np.ndarray,
    lesion: np.ndarray,
    crop: tuple[slice, slice],
    vmin: float,
    vmax: float,
) -> None:
    axis.imshow(image[crop], cmap="gray", vmin=vmin, vmax=vmax, interpolation="none")
    axis.contour(
        lesion[crop], levels=[0.5], colors=["#F04B3A"], linewidths=0.75
    )
    axis.text(0.02, 0.04, "L", transform=axis.transAxes, color="white", fontsize=6)
    axis.text(
        0.98,
        0.04,
        "R",
        transform=axis.transAxes,
        color="white",
        fontsize=6,
        ha="right",
    )
    axis.set_axis_off()


def show_labels(
    axis: plt.Axes,
    image: np.ndarray,
    labels: np.ndarray,
    lesion: np.ndarray,
    crop: tuple[slice, slice],
    vmin: float,
    vmax: float,
    colors: dict[int, np.ndarray],
) -> None:
    axis.imshow(image[crop], cmap="gray", vmin=vmin, vmax=vmax, interpolation="none")
    axis.imshow(rgba_labels(labels, colors)[crop], interpolation="none")
    axis.contour(lesion[crop], levels=[0.5], colors=["white"], linewidths=0.55)
    axis.set_axis_off()


def make_figure(
    selected: list[tuple[str, float, float, CasePaths]],
    flirt: Path,
    output_dir: Path,
) -> pd.DataFrame:
    pv.OFF_SCREEN = True
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    figure, axes = plt.subplots(6, 3, figsize=(7.2, 9.0))
    figure.subplots_adjust(
        left=0.145, right=0.995, bottom=0.018, top=0.925, wspace=0.025, hspace=0.035
    )
    titles = ("T1 MRI", "Volumetric atlas labels", "Midcortical surface labels")
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, fontsize=9, fontweight="bold", pad=5)

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="arc_representative_") as temporary:
        temporary_dir = Path(temporary)
        for group, (size, quantile, volume_ml, paths) in enumerate(selected):
            original = load_volume(paths.original_mri, np.float32)
            inpainted = load_volume(paths.inpainted_mri, np.float32)
            lesion = load_volume(paths.lesion_mask, np.uint8) > 0
            inpainted_label = load_volume(paths.inpainted_label, np.int32)
            direct_label = map_direct_label(
                paths, flirt, temporary_dir / f"{paths.case_id}_direct_label_mni.nii.gz"
            )
            if not (
                original.shape
                == inpainted.shape
                == lesion.shape
                == direct_label.shape
                == inpainted_label.shape
            ):
                raise ValueError(f"Mismatched MNI geometry for {paths.case_id}")
            slice_index = int(np.argmax(lesion.sum(axis=(0, 1))))
            original_slice = axial(original, slice_index)
            inpainted_slice = axial(inpainted, slice_index)
            lesion_slice = axial(lesion, slice_index)
            direct_label_slice = axial(direct_label, slice_index)
            inpainted_label_slice = axial(inpainted_label, slice_index)
            crop = display_crop((original_slice > 0) | (inpainted_slice > 0))
            intensity = np.concatenate(
                [original[original > 0].ravel(), inpainted[inpainted > 0].ravel()]
            )
            vmin, vmax = np.percentile(intensity, [1.0, 99.5])
            colors = label_colors(paths.label_description)
            direct_surface = render_surface(
                paths.direct_surface,
                temporary_dir / f"{paths.case_id}_direct_surface.png",
            )
            inpainted_surface = render_surface(
                paths.inpainted_surface,
                temporary_dir / f"{paths.case_id}_inpainted_surface.png",
            )

            original_row = group * 2
            inpainted_row = original_row + 1
            show_mri(
                axes[original_row, 0],
                original_slice,
                lesion_slice,
                crop,
                vmin,
                vmax,
            )
            show_labels(
                axes[original_row, 1],
                original_slice,
                direct_label_slice,
                lesion_slice,
                crop,
                vmin,
                vmax,
                colors,
            )
            axes[original_row, 2].imshow(direct_surface)
            axes[original_row, 2].set_axis_off()

            show_mri(
                axes[inpainted_row, 0],
                inpainted_slice,
                lesion_slice,
                crop,
                vmin,
                vmax,
            )
            show_labels(
                axes[inpainted_row, 1],
                inpainted_slice,
                inpainted_label_slice,
                lesion_slice,
                crop,
                vmin,
                vmax,
                colors,
            )
            axes[inpainted_row, 2].imshow(inpainted_surface)
            axes[inpainted_row, 2].set_axis_off()

            axes[original_row, 0].text(
                -0.11,
                0.50,
                f"{size}\n{volume_ml:.1f} mL\nOriginal",
                transform=axes[original_row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#333333",
            )
            axes[inpainted_row, 0].text(
                -0.11,
                0.50,
                "Inpainted",
                transform=axes[inpainted_row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#0B7285",
            )
            records.append(
                {
                    "size": size.lower(),
                    "target_quantile": quantile,
                    "case_id": paths.case_id,
                    "lesion_volume_ml": volume_ml,
                    "axial_slice_index": slice_index,
                }
            )

    for boundary_row in (1, 3):
        y = (axes[boundary_row, 0].get_position().y0 + axes[boundary_row + 1, 0].get_position().y1) / 2
        figure.add_artist(
            Line2D([0.04, 0.995], [y, y], transform=figure.transFigure, color="#C8C8C8", linewidth=0.65)
        )
    figure.text(
        0.55,
        0.992,
        "Representative ARC outputs across lesion sizes",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.995,
        0.004,
        "Red/white outline: original source-lesion mask; surface view: left lateral",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / "arc_representative_brainsuite_outputs.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output_dir / "arc_representative_brainsuite_outputs.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    if not args.flirt.is_file():
        raise FileNotFoundError(f"FSL FLIRT was not found: {args.flirt}")
    metrics = pd.read_csv(args.case_metrics)
    raw_root = args.arc_root / "derivatives/brainsuite_anatomical_raw"
    inpainted_root = args.arc_root / "derivatives/brainsuite_anatomical_bidsapp"
    inpainting_root = args.arc_root / "derivatives/stroke_inpainting"
    raw_files = arc.discover_stats(raw_root)
    inpainted_files = arc.discover_stats(inpainted_root)
    selected = select_cases(metrics, raw_files, inpainted_files, inpainting_root)
    records = make_figure(selected, args.flirt, args.output_dir)
    records.to_csv(args.output_dir / "representative_cases.csv", index=False)
    print(records.to_string(index=False))
    print(f"Wrote figure to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the manuscript figure for the completed ARC AQ comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULT_CANDIDATES = (
    Path("/project2/ajoshi_1183/data/ARC/derivatives/aq_mass_effect_comparison"),
    Path("/home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/aq_mass_effect_comparison"),
)

MODEL_ORDER = (
    "clinical_only",
    "lesion_standard",
    "lesion_plus_mass_effect",
    "lesion_plus_mass_effect_and_registration_qc",
    "lesion_plus_uncertainty",
    "lesion_uncertainty_plus_mass_effect",
)
MODEL_LABELS = {
    "clinical_only": "Clinical",
    "lesion_standard": "Lesion",
    "lesion_plus_mass_effect": "Lesion + deformation",
    "lesion_plus_mass_effect_and_registration_qc": "+ deformation + reg. QC",
    "lesion_plus_uncertainty": "Lesion + uncertainty",
    "lesion_uncertainty_plus_mass_effect": "+ uncertainty + deformation",
}


def first_results_dir() -> Path:
    for path in RESULT_CANDIDATES:
        if (path / "aq_mass_effect_metrics_by_repeat.csv").is_file():
            return path
    return RESULT_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=first_results_dir())
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE
        / "analysis"
        / "miccai_workshop"
        / "figures"
        / "aq_prediction_comparison.pdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = pd.read_csv(args.results_dir / "aq_mass_effect_metrics_by_repeat.csv")
    comparisons = pd.read_csv(args.results_dir / "aq_mass_effect_paired_comparisons.csv")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tr2strokeseg-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "clinical_only": "#999999",
        "lesion_standard": "#4477AA",
        "lesion_plus_mass_effect": "#228833",
        "lesion_plus_mass_effect_and_registration_qc": "#66AA55",
        "lesion_plus_uncertainty": "#CC6677",
        "lesion_uncertainty_plus_mass_effect": "#AA3377",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.25), constrained_layout=True)

    values = [
        metrics.loc[metrics["model"] == model, "mae"].to_numpy(float)
        for model in MODEL_ORDER
    ]
    positions = np.arange(len(MODEL_ORDER), 0, -1)
    boxes = axes[0].boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
    )
    for patch, model in zip(boxes["boxes"], MODEL_ORDER, strict=True):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.82)
    for median in boxes["medians"]:
        median.set_color("black")
        median.set_linewidth(1.3)
    axes[0].set_yticks(positions, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axes[0].set_xlabel("Repeated outer-CV MAE (AQ points)")
    axes[0].set_title("A  Held-out prediction error", loc="left", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.22)

    comparison_order = (
        "lesion_plus_mass_effect",
        "lesion_plus_mass_effect_and_registration_qc",
        "lesion_plus_uncertainty",
        "lesion_uncertainty_plus_mass_effect",
    )
    comparison_labels = {
        "lesion_plus_mass_effect": "+ deformation",
        "lesion_plus_mass_effect_and_registration_qc": "+ deformation + reg. QC",
        "lesion_plus_uncertainty": "+ uncertainty",
        "lesion_uncertainty_plus_mass_effect": "+ uncertainty + deformation",
    }
    selected = comparisons.set_index("comparison_model").loc[list(comparison_order)]
    estimates = selected["mean_mae_advantage_points"].to_numpy(float)
    low = selected["bootstrap_ci025"].to_numpy(float)
    high = selected["bootstrap_ci975"].to_numpy(float)
    ypos = np.arange(len(comparison_order), 0, -1)
    for index, (model, y, estimate, lower, upper) in enumerate(
        zip(comparison_order, ypos, estimates, low, high, strict=True)
    ):
        axes[1].errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - lower], [upper - estimate]]),
            fmt="o",
            color=colors[model],
            markersize=7,
            capsize=3,
            linewidth=1.8,
        )
        p_value = float(selected.iloc[index]["p_value_holm"])
        axes[1].text(
            upper + 0.05,
            y,
            f"$p_{{H}}$={p_value:.3f}",
            va="center",
            fontsize=8,
        )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(ypos, [comparison_labels[model] for model in comparison_order])
    axes[1].set_xlabel("Mean paired MAE advantage (AQ points)")
    axes[1].set_title("B  Incremental comparisons", loc="left", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].set_xlim(min(-0.75, float(low.min()) - 0.1), float(high.max()) + 0.55)

    figure.suptitle(
        "Repeated nested-CV aphasia prediction (n=210)",
        fontweight="bold",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, facecolor="white", bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

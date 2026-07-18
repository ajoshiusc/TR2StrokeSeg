#!/usr/bin/env python3
"""Paper-oriented comparison of WAB AQ models with lesion deformation.

The primary analysis compares the same ARC subjects under a conventional
lesion/clinical ridge model and that model augmented with a compact set of
contralateral-normalized deformation features. Ridge penalty selection,
imputation, and scaling are nested inside every outer fold. Repeated outer
cross-validation, paired subject-level error tests, multiplicity correction,
figures, machine-readable tables, and LaTeX result macros are written.

If a sufficiently complete lesion-uncertainty manifest is available, matched
soft-lesion/entropy models with and without deformation are included as a
secondary comparison. The deformation features are cross-sectional imaging
proxies and must not be described as physical ground-truth mass effect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DEFAULT_CLINICAL = HERE / "Inpainting_Code" / "analysis" / "arc_inpainting" / "case_metrics.csv"

CLINICAL_FEATURES = ["age_at_stroke", "log1p_wab_days"]
LESION_FEATURES = [
    "lesion_volume_ml",
    "left_language_lesion_ml",
    "right_language_lesion_ml",
    "lesion_laterality_index",
]
MASS_EFFECT_FEATURES = [
    "me_mass_effect_3_20mm_magnitude_mm_median",
    "me_mass_effect_3_20mm_magnitude_mm_p95",
    "me_mass_effect_3_20mm_radial_mm_median",
    "me_mass_effect_3_20mm_mean_absolute_radial_mm",
    "me_mass_effect_3_20mm_outward_integral_ml_mm",
    "me_mass_effect_3_20mm_inward_integral_ml_mm",
    "me_mass_effect_3_20mm_logjac_expansion_integral_ml",
    "me_mass_effect_3_20mm_logjac_compression_integral_ml",
]
REGISTRATION_QC_FEATURES = [
    "me_registration_sensitivity_3_20mm_mm_median",
    "me_contralateral_affine_fit_rmse_mm",
]
SOFT_LESION_FEATURES = [
    "unc_expected_lesion_volume_ml",
    "unc_expected_left_lesion_volume_ml",
    "unc_expected_right_lesion_volume_ml",
    "unc_maximum_lesion_probability",
    "unc_entropy_mass_ml",
    "unc_left_entropy_mass_ml",
    "unc_right_entropy_mass_ml",
    "unc_candidate_mean_entropy",
    "unc_boundary_mean_entropy",
    "unc_high_uncertainty_volume_ml",
    "unc_volume_p25_ml",
    "unc_volume_p75_ml",
]


def first_file(paths: list[Path | None]) -> Path | None:
    return next((path for path in paths if path is not None and path.is_file()), None)


def arc_root() -> Path:
    if os.environ.get("ARC_ROOT"):
        return Path(os.environ["ARC_ROOT"])
    return Path("/project2/ajoshi_1183/data/ARC")


def default_mass_manifest() -> Path:
    root = arc_root()
    return first_file(
        [
            Path(os.environ["MASS_EFFECT_OUTPUT_DIR"]) / "mass_effect_manifest.csv"
            if os.environ.get("MASS_EFFECT_OUTPUT_DIR")
            else None,
            root / "derivatives" / "lesion_mass_effect" / "mass_effect_manifest.csv",
            HERE / "outputs" / "mass_effect_local" / "mass_effect_manifest.csv",
        ]
    ) or root / "derivatives" / "lesion_mass_effect" / "mass_effect_manifest.csv"


def default_uncertainty_manifest() -> Path | None:
    root = arc_root()
    return first_file(
        [
            Path(os.environ["LESION_UNCERTAINTY_OUTPUT_DIR"]) / "uncertainty_manifest.csv"
            if os.environ.get("LESION_UNCERTAINTY_OUTPUT_DIR")
            else None,
            root / "derivatives" / "lesion_uncertainty" / "uncertainty_manifest.csv",
            HERE / "outputs" / "lesion_uncertainty_local" / "uncertainty_manifest.csv",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-effect-manifest", type=Path, default=default_mass_manifest())
    uncertainty_group = parser.add_mutually_exclusive_group()
    uncertainty_group.add_argument(
        "--uncertainty-manifest", type=Path, default=default_uncertainty_manifest()
    )
    uncertainty_group.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Run only the primary conventional-lesion/deformation comparison",
    )
    parser.add_argument("--clinical-table", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "outputs" / "aq_mass_effect_comparison"
    )
    parser.add_argument("--outcome", default="wab_aq")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=np.logspace(-4, 4, 17).tolist()
    )
    parser.add_argument(
        "--require-deformation-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use supported unilateral cases with valid, nonfolding proxy fields",
    )
    parser.add_argument("--maximum-folding-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-near-lesion-voxels", type=int, default=1000)
    parser.add_argument(
        "--minimum-uncertainty-coverage",
        type=float,
        default=0.90,
        help="Skip optional soft-lesion models if less of the mass-effect cohort is matched",
    )
    args = parser.parse_args()
    if args.no_uncertainty:
        args.uncertainty_manifest = None
    return args


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def check_unique(frame: pd.DataFrame, label: str) -> None:
    if "case_id" not in frame:
        raise ValueError(f"{label} has no case_id column")
    duplicate = frame["case_id"].duplicated(keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, "case_id"].astype(str).head(5).tolist()
        raise ValueError(f"{label} has duplicate case IDs, including {examples}")


def prefix_nonkeys(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = {"case_id", "subject", "session"}
    return frame.rename(columns={column: prefix + column for column in frame if column not in keys})


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def build_design(
    mass: pd.DataFrame,
    clinical: pd.DataFrame,
    uncertainty: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    if "participant_id" in clinical and "subject" not in clinical:
        clinical = clinical.rename(columns={"participant_id": "subject"})
    check_unique(mass, "mass-effect manifest")
    check_unique(clinical, "clinical table")
    if args.outcome not in clinical:
        raise ValueError(f"Clinical table has no outcome column {args.outcome!r}")

    original_mass_n = len(mass)
    mass_prefixed = prefix_nonkeys(mass, "me_")
    clinical_payload = clinical.drop(columns=["subject", "session"], errors="ignore")
    design = mass_prefixed.merge(clinical_payload, on="case_id", how="inner", validate="one_to_one")
    matched_n = len(design)

    required_mass_columns = [
        "me_laterality_supported",
        "me_normalized_field_folding_fraction",
        "me_mass_effect_3_20mm_magnitude_mm_n_voxels",
    ]
    missing = [column for column in required_mass_columns if column not in design]
    if missing:
        raise ValueError(
            "Mass-effect manifest predates required QC fields or is incomplete: " + ", ".join(missing)
        )
    if args.require_deformation_qc:
        supported = truthy(design["me_laterality_supported"])
        folding = pd.to_numeric(
            design["me_normalized_field_folding_fraction"], errors="coerce"
        )
        voxel_count = pd.to_numeric(
            design["me_mass_effect_3_20mm_magnitude_mm_n_voxels"], errors="coerce"
        )
        design = design.loc[
            supported
            & folding.le(args.maximum_folding_fraction)
            & voxel_count.ge(args.minimum_near_lesion_voxels)
        ].copy()
    qc_n = len(design)

    uncertainty_used = False
    uncertainty_coverage = 0.0
    if uncertainty is not None:
        check_unique(uncertainty, "uncertainty manifest")
        uncertainty_coverage = float(design["case_id"].isin(uncertainty["case_id"]).mean())
        if uncertainty_coverage >= args.minimum_uncertainty_coverage:
            uncertainty_payload = prefix_nonkeys(uncertainty, "unc_").drop(
                columns=["subject", "session"], errors="ignore"
            )
            design = design.merge(
                uncertainty_payload, on="case_id", how="inner", validate="one_to_one"
            )
            uncertainty_used = True

    design[args.outcome] = pd.to_numeric(design[args.outcome], errors="coerce")
    design["age_at_stroke"] = pd.to_numeric(design["age_at_stroke"], errors="coerce")
    design["log1p_wab_days"] = np.log1p(pd.to_numeric(design["wab_days"], errors="coerce"))
    numeric_columns = design.select_dtypes(include=[np.number]).columns
    design.loc[:, numeric_columns] = design.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    design = design.dropna(subset=[args.outcome])
    if design["subject"].duplicated().any():
        duplicates = design.loc[design["subject"].duplicated(False), "subject"].head().tolist()
        raise ValueError(f"More than one case remains for subjects including {duplicates}")
    design = design.sort_values(["subject", "case_id"]).reset_index(drop=True)
    audit: dict[str, int | float | bool] = {
        "mass_effect_rows": original_mass_n,
        "mass_effect_clinical_matches": matched_n,
        "rows_after_deformation_qc": qc_n,
        "uncertainty_coverage_before_matching": uncertainty_coverage,
        "uncertainty_models_used": uncertainty_used,
        "analysis_subjects": len(design),
    }
    return design, audit


def feature_sets(design: pd.DataFrame, uncertainty_used: bool) -> dict[str, list[str]]:
    standard = CLINICAL_FEATURES + LESION_FEATURES
    sets = {
        "clinical_only": CLINICAL_FEATURES,
        "lesion_standard": standard,
        "lesion_plus_mass_effect": standard + MASS_EFFECT_FEATURES,
    }
    if all(column in design and design[column].notna().any() for column in REGISTRATION_QC_FEATURES):
        sets["lesion_plus_mass_effect_and_registration_qc"] = (
            standard + MASS_EFFECT_FEATURES + REGISTRATION_QC_FEATURES
        )
    if uncertainty_used:
        # Keep the conventional lesion and language-network burden in both
        # models so this is an incremental uncertainty test, not a comparison
        # confounded by removing anatomical lesion-location information.
        uncertainty_augmented = standard + SOFT_LESION_FEATURES
        sets["lesion_plus_uncertainty"] = uncertainty_augmented
        sets["lesion_uncertainty_plus_mass_effect"] = (
            uncertainty_augmented + MASS_EFFECT_FEATURES
        )
    for model, features in sets.items():
        missing = [feature for feature in features if feature not in design]
        if missing:
            raise ValueError(f"Features missing for {model}: {missing}")
        empty = [feature for feature in features if not design[feature].notna().any()]
        if empty:
            raise ValueError(f"Features contain no finite values for {model}: {empty}")
    return sets


def pearson_r(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.isclose(np.std(observed), 0) or np.isclose(np.std(predicted), 0):
        return math.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def validate_cv(args: argparse.Namespace, n: int) -> None:
    if n < max(20, args.outer_folds):
        raise ValueError(f"Only {n} subjects remain; at least {max(20, args.outer_folds)} are required")
    if not 2 <= args.outer_folds <= n:
        raise ValueError("--outer-folds must be between 2 and the cohort size")
    if args.inner_folds < 2 or args.repeats < 1 or args.bootstrap_samples < 100:
        raise ValueError("Need inner-folds >=2, repeats >=1, and bootstrap-samples >=100")
    if not 0 <= args.maximum_folding_fraction <= 1:
        raise ValueError("--maximum-folding-fraction must be in [0,1]")
    if not 0 <= args.minimum_uncertainty_coverage <= 1:
        raise ValueError("--minimum-uncertainty-coverage must be in [0,1]")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in args.alpha_grid):
        raise ValueError("Every ridge alpha must be positive and finite")


def repeated_nested_cv(
    design: pd.DataFrame,
    sets: dict[str, list[str]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(design)
    validate_cv(args, n)
    y = design[args.outcome].to_numpy(float)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    identifiers = design[["case_id", "subject", "session"]].to_dict("records")

    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + 1009 * (repeat - 1)
        outer = list(
            KFold(args.outer_folds, shuffle=True, random_state=repeat_seed).split(np.arange(n))
        )
        for model, features in sets.items():
            x = design[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            predicted = np.full(n, np.nan)
            folds = np.full(n, -1, dtype=int)
            chosen_alpha = np.full(n, np.nan)
            for fold, (train, test) in enumerate(outer, start=1):
                inner_n = min(args.inner_folds, len(train))
                estimator = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge()),
                    ]
                )
                search = GridSearchCV(
                    estimator,
                    {"ridge__alpha": args.alpha_grid},
                    scoring="neg_mean_absolute_error",
                    cv=KFold(inner_n, shuffle=True, random_state=repeat_seed + fold),
                    n_jobs=args.n_jobs,
                    refit=True,
                )
                search.fit(x[train], y[train])
                predicted[test] = search.predict(x[test])
                alpha = float(search.best_params_["ridge__alpha"])
                chosen_alpha[test] = alpha
                folds[test] = fold
                coefficients = search.best_estimator_.named_steps["ridge"].coef_
                for feature, coefficient in zip(features, coefficients, strict=True):
                    coefficient_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": fold,
                            "model": model,
                            "feature": feature,
                            "standardized_coefficient": float(coefficient),
                            "alpha": alpha,
                        }
                    )
            for index in range(n):
                prediction_rows.append(
                    {
                        **identifiers[index],
                        "repeat": repeat,
                        "outer_fold": int(folds[index]),
                        "model": model,
                        "observed_aq": float(y[index]),
                        "predicted_aq": float(predicted[index]),
                        "absolute_error": float(abs(y[index] - predicted[index])),
                        "alpha": float(chosen_alpha[index]),
                    }
                )
            metric_rows.append(
                {
                    "repeat": repeat,
                    "model": model,
                    "n": n,
                    "mae": float(mean_absolute_error(y, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
                    "r2": float(r2_score(y, predicted)),
                    "pearson_r": pearson_r(y, predicted),
                }
            )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "n": int(group["n"].iloc[0])}
        for metric in ("mae", "rmse", "r2", "pearson_r"):
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd_across_repeats"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_p025_across_repeats"] = float(np.percentile(values, 2.5))
            row[f"{metric}_p975_across_repeats"] = float(np.percentile(values, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_comparisons(
    predictions: pd.DataFrame, reference: str, samples: int, seed: int
) -> pd.DataFrame:
    subject_errors = (
        predictions.groupby(["subject", "model"], as_index=False)["absolute_error"].mean()
        .pivot(index="subject", columns="model", values="absolute_error")
    )
    if reference not in subject_errors:
        raise ValueError(f"Reference model {reference!r} is unavailable")
    rng = np.random.default_rng(seed)
    rows = []
    for model in subject_errors.columns:
        if model == reference:
            continue
        pair = subject_errors[[reference, model]].dropna()
        advantage = pair[reference].to_numpy(float) - pair[model].to_numpy(float)
        indices = rng.integers(0, len(advantage), size=(samples, len(advantage)))
        bootstrap = advantage[indices].mean(axis=1)
        if np.allclose(advantage, 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(advantage, zero_method="pratt", alternative="two-sided")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "reference_model": reference,
                "comparison_model": model,
                "n_subjects": len(advantage),
                "mean_mae_advantage_points": float(np.mean(advantage)),
                "bootstrap_ci025": float(np.percentile(bootstrap, 2.5)),
                "bootstrap_ci975": float(np.percentile(bootstrap, 97.5)),
                "subjects_improved_fraction": float(np.mean(advantage > 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = holm_adjust(result["p_value"].to_numpy(float))
    return result


def display_name(value: str) -> str:
    return value.replace("_", " ").replace("mass effect", "deformation")


def make_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    output_stem: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tr2strokeseg-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = metrics["model"].drop_duplicates().tolist()
    colors = dict(zip(models, plt.cm.viridis(np.linspace(0.10, 0.90, len(models))), strict=True))
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    mae = [metrics.loc[metrics.model == model, "mae"].to_numpy(float) for model in models]
    box = axes[0, 0].boxplot(mae, tick_labels=[display_name(m) for m in models], patch_artist=True)
    for patch, model in zip(box["boxes"], models, strict=True):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.75)
    axes[0, 0].tick_params(axis="x", rotation=24)
    axes[0, 0].set_ylabel("Outer-CV MAE (AQ points)")
    axes[0, 0].set_title("A. Performance across repeated splits")
    axes[0, 0].grid(axis="y", alpha=0.2)

    averaged = predictions.groupby(["subject", "model"], as_index=False).agg(
        observed_aq=("observed_aq", "first"), predicted_aq=("predicted_aq", "mean")
    )
    shown = ["lesion_standard", "lesion_plus_mass_effect"]
    limits = [float(averaged.observed_aq.min()) - 3, float(averaged.observed_aq.max()) + 3]
    for model in shown:
        group = averaged[averaged.model == model]
        axes[0, 1].scatter(
            group.observed_aq,
            group.predicted_aq,
            s=20,
            alpha=0.55,
            label=display_name(model),
            color=colors[model],
        )
    axes[0, 1].plot(limits, limits, "k--", linewidth=1)
    axes[0, 1].set(xlim=limits, ylim=limits, xlabel="Observed WAB AQ", ylabel="Mean repeated-CV prediction")
    axes[0, 1].set_title("B. Out-of-sample predictions")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    subject_errors = predictions.groupby(["subject", "model"])["absolute_error"].mean().unstack()
    advantage = subject_errors["lesion_standard"] - subject_errors["lesion_plus_mass_effect"]
    axes[1, 0].hist(advantage, bins=22, color=colors["lesion_plus_mass_effect"], alpha=0.8)
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].axvline(advantage.mean(), color="#b2182b", linewidth=2, label=f"Mean={advantage.mean():.2f}")
    axes[1, 0].set_xlabel("Per-subject MAE advantage (standard − deformation), AQ points")
    axes[1, 0].set_ylabel("Subjects")
    axes[1, 0].set_title("C. Paired incremental benefit")
    axes[1, 0].legend(frameon=False)

    coefficient_model = coefficients[coefficients.model == "lesion_plus_mass_effect"]
    coefficient_model = coefficient_model[coefficient_model.feature.isin(MASS_EFFECT_FEATURES)]
    coefficient_summary = (
        coefficient_model.groupby("feature")["standardized_coefficient"]
        .agg(["mean", "std"])
        .sort_values("mean", key=lambda values: values.abs())
    )
    labels = [index.removeprefix("me_mass_effect_3_20mm_").replace("_", " ") for index in coefficient_summary.index]
    axes[1, 1].barh(
        labels,
        coefficient_summary["mean"],
        xerr=coefficient_summary["std"].fillna(0),
        color=colors["lesion_plus_mass_effect"],
        alpha=0.8,
    )
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set_xlabel("Standardized ridge coefficient (mean ± SD across folds)")
    axes[1, 1].set_title("D. Deformation feature stability")
    axes[1, 1].tick_params(axis="y", labelsize=8)
    axes[1, 1].grid(axis="x", alpha=0.2)

    figure.suptitle("Incremental value of lesion-associated deformation for ARC aphasia prediction", fontweight="bold")
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, facecolor="white", bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def latex_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_latex(summary: pd.DataFrame, comparisons: pd.DataFrame, path: Path) -> None:
    standard = summary.loc[summary.model == "lesion_standard"].iloc[0]
    augmented = summary.loc[summary.model == "lesion_plus_mass_effect"].iloc[0]
    paired = comparisons.loc[
        comparisons.comparison_model == "lesion_plus_mass_effect"
    ].iloc[0]
    lines = [
        "% Generated by compare_aq_mass_effect_models.py; do not edit by hand.",
        latex_command("AQAnalysisN", str(int(standard.n))),
        latex_command("AQStandardMAE", f"{standard.mae_mean:.2f}"),
        latex_command("AQDeformationMAE", f"{augmented.mae_mean:.2f}"),
        latex_command("AQDeformationAdvantage", f"{paired.mean_mae_advantage_points:.2f}"),
        latex_command(
            "AQDeformationAdvantageCI",
            f"{paired.bootstrap_ci025:.2f}--{paired.bootstrap_ci975:.2f}",
        ),
        latex_command("AQDeformationP", f"{paired.p_value_holm:.3g}"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.mass_effect_manifest.is_file():
        raise FileNotFoundError(f"--mass-effect-manifest not found: {args.mass_effect_manifest}")
    if not args.clinical_table.is_file():
        raise FileNotFoundError(f"--clinical-table not found: {args.clinical_table}")
    if args.uncertainty_manifest is not None and not args.uncertainty_manifest.is_file():
        raise FileNotFoundError(f"--uncertainty-manifest not found: {args.uncertainty_manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mass = read_table(args.mass_effect_manifest)
    clinical = read_table(args.clinical_table)
    uncertainty = read_table(args.uncertainty_manifest) if args.uncertainty_manifest else None
    design, audit = build_design(mass, clinical, uncertainty, args)
    sets = feature_sets(design, bool(audit["uncertainty_models_used"]))
    predictions, metrics, coefficients = repeated_nested_cv(design, sets, args)
    summary = summarize_metrics(metrics)
    comparisons = paired_comparisons(
        predictions, "lesion_standard", args.bootstrap_samples, args.seed + 99991
    )

    design.to_csv(args.output_dir / "aq_mass_effect_design.csv", index=False)
    predictions.to_csv(args.output_dir / "aq_mass_effect_predictions_long.csv", index=False)
    metrics.to_csv(args.output_dir / "aq_mass_effect_metrics_by_repeat.csv", index=False)
    summary.to_csv(args.output_dir / "aq_mass_effect_model_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "aq_mass_effect_paired_comparisons.csv", index=False)
    coefficients.to_csv(args.output_dir / "aq_mass_effect_coefficients.csv", index=False)
    make_figure(
        predictions,
        metrics,
        coefficients,
        args.output_dir / "aq_mass_effect_model_comparison",
    )
    write_latex(summary, comparisons, args.output_dir / "paper_results.tex")

    config = {
        **audit,
        "clinical_table": str(args.clinical_table.resolve()),
        "mass_effect_manifest": str(args.mass_effect_manifest.resolve()),
        "uncertainty_manifest": str(args.uncertainty_manifest.resolve())
        if args.uncertainty_manifest
        else None,
        "outcome": args.outcome,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "repeats": args.repeats,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "alpha_grid": args.alpha_grid,
        "feature_sets": sets,
        "primary_reference": "lesion_standard",
        "primary_incremental_model": "lesion_plus_mass_effect",
        "interpretation_warning": (
            "Cross-sectional contralateral-normalized lesion-associated deformation proxy; "
            "not physical ground-truth mass effect. Chronic remodeling/collapse may contribute."
        ),
        "inpainting_warning": (
            "The inpainted registration defines correspondences outside the synthetic target. "
            "Raw-vs-inpainted sensitivity is QC and is not treated as biological deformation."
        ),
        "uncertainty_warning": (
            "nnU-Net probabilities are predictive scores and are not externally calibrated p-values."
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Analyzed {len(design)} independent subjects")
    if uncertainty is not None and not audit["uncertainty_models_used"]:
        print(
            "Skipped optional uncertainty models: matched coverage "
            f"{audit['uncertainty_coverage_before_matching']:.1%} is below "
            f"{args.minimum_uncertainty_coverage:.1%}"
        )
    print(summary[["model", "mae_mean", "rmse_mean", "r2_mean", "pearson_r_mean"]].to_string(index=False))
    primary = comparisons.loc[
        comparisons.comparison_model == "lesion_plus_mass_effect"
    ].iloc[0]
    print(
        "Primary paired MAE advantage: "
        f"{primary.mean_mae_advantage_points:.2f} AQ points "
        f"(bootstrap 95% CI {primary.bootstrap_ci025:.2f} to {primary.bootstrap_ci975:.2f}; "
        f"Holm p={primary.p_value_holm:.3g})"
    )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

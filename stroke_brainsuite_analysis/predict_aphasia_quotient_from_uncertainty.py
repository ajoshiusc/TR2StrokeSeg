#!/usr/bin/env python3
"""Evaluate uncertainty-aware WAB Aphasia Quotient prediction on ARC.

This script joins ``uncertainty_manifest.csv`` from
``run_arc_lesion_uncertainty.py`` to the existing ARC clinical/case table and
compares nested cross-validated ridge-regression models using:

1. clinical timing/demographic covariates only;
2. a conventional hard lesion mask;
3. the soft lesion probability map summaries; and
4. probability plus predictive-entropy summaries.

All preprocessing and ridge-penalty selection occur inside each outer fold.
The output predictions are therefore out-of-sample estimates, not fitted
values from the complete cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DEFAULT_CASE_TABLE = (
    HERE / "Inpainting_Code" / "analysis" / "arc_inpainting" / "case_metrics.csv"
)
SESSION_RE = re.compile(r"^ses-([0-9]+(?:\.[0-9]+)?)$")


FEATURE_SETS: dict[str, list[str]] = {
    "clinical_only": ["age_at_stroke", "log1p_wab_days"],
    "hard_mask": [
        "age_at_stroke",
        "log1p_wab_days",
        "hard_lesion_volume_ml",
        "hard_left_lesion_volume_ml",
        "hard_right_lesion_volume_ml",
    ],
    "lesion_probability": [
        "age_at_stroke",
        "log1p_wab_days",
        "expected_lesion_volume_ml",
        "expected_left_lesion_volume_ml",
        "expected_right_lesion_volume_ml",
        "maximum_lesion_probability",
    ],
    "probability_plus_uncertainty": [
        "age_at_stroke",
        "log1p_wab_days",
        "expected_lesion_volume_ml",
        "expected_left_lesion_volume_ml",
        "expected_right_lesion_volume_ml",
        "maximum_lesion_probability",
        "entropy_mass_ml",
        "left_entropy_mass_ml",
        "right_entropy_mass_ml",
        "candidate_mean_entropy",
        "boundary_mean_entropy",
        "high_uncertainty_volume_ml",
        "volume_p25_ml",
        "volume_p75_ml",
    ],
}


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.is_file():
            return path
    return None


def default_uncertainty_manifest() -> Path | None:
    arc_root = Path(os.environ["ARC_ROOT"]) if os.environ.get("ARC_ROOT") else None
    candidates = [
        Path(os.environ["LESION_UNCERTAINTY_OUTPUT_DIR"]) / "uncertainty_manifest.csv"
        if os.environ.get("LESION_UNCERTAINTY_OUTPUT_DIR")
        else None,
        arc_root / "derivatives" / "lesion_uncertainty" / "uncertainty_manifest.csv"
        if arc_root
        else None,
        Path("/project2/ajoshi_1183/data/ARC/derivatives/lesion_uncertainty/uncertainty_manifest.csv"),
        HERE / "outputs" / "lesion_uncertainty_local" / "uncertainty_manifest.csv",
    ]
    return first_existing(candidates)


def default_clinical_table() -> Path:
    arc_root = Path(os.environ["ARC_ROOT"]) if os.environ.get("ARC_ROOT") else None
    return first_existing(
        [
            DEFAULT_CASE_TABLE,
            arc_root / "participants.tsv" if arc_root else None,
            Path("/project2/ajoshi_1183/data/ARC/participants.tsv"),
        ]
    ) or DEFAULT_CASE_TABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uncertainty-manifest", type=Path, default=default_uncertainty_manifest()
    )
    parser.add_argument(
        "--clinical-table",
        type=Path,
        default=default_clinical_table(),
        help="CSV case table (preferred) or BIDS participants.tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "outputs" / "aq_uncertainty_prediction"
    )
    parser.add_argument("--outcome", default="wab_aq")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=np.logspace(-4, 4, 17).tolist(),
        help="Positive ridge penalties searched independently inside each outer fold",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator)


def session_day(value: object) -> float:
    match = SESSION_RE.match(str(value))
    return float(match.group(1)) if match else math.nan


def one_scan_per_subject(frame: pd.DataFrame, wab_days_column: str = "wab_days") -> pd.DataFrame:
    """Choose the acquisition nearest WAB timing when no case ID is supplied."""
    if not frame["subject"].duplicated().any():
        return frame
    selected: list[pd.Series] = []
    for _, group in frame.groupby("subject", sort=True):
        group = group.copy()
        group["_session_day"] = group["session"].map(session_day)
        if wab_days_column in group and pd.to_numeric(group[wab_days_column], errors="coerce").notna().any():
            wab_days = pd.to_numeric(group[wab_days_column], errors="coerce")
            group["_distance"] = (group["_session_day"] - wab_days).abs()
            finite = group["_distance"].notna()
            chosen = group.loc[finite].sort_values(["_distance", "case_id"]).iloc[0] if finite.any() else group.sort_values("case_id").iloc[0]
        else:
            chosen = group.sort_values("case_id").iloc[0]
        selected.append(chosen)
    return pd.DataFrame(selected).drop(columns=["_session_day", "_distance"], errors="ignore")


def build_design(uncertainty: pd.DataFrame, clinical: pd.DataFrame, outcome: str) -> pd.DataFrame:
    uncertainty = uncertainty.copy()
    clinical = clinical.copy()
    if "participant_id" in clinical and "subject" not in clinical:
        clinical = clinical.rename(columns={"participant_id": "subject"})
    required_uncertainty = {"case_id", "subject", "session"}
    missing = required_uncertainty - set(uncertainty)
    if missing:
        raise ValueError(f"Uncertainty manifest is missing columns: {sorted(missing)}")
    if outcome not in clinical:
        raise ValueError(f"Clinical table has no outcome column '{outcome}'")

    if "case_id" in clinical:
        duplicate_cases = clinical["case_id"].duplicated(keep=False)
        if duplicate_cases.any():
            examples = clinical.loc[duplicate_cases, "case_id"].astype(str).head().tolist()
            raise ValueError(f"Clinical table has duplicate case IDs, including: {examples}")
        overlapping = [
            column
            for column in clinical.columns
            if column in uncertainty.columns and column not in {"case_id", "subject", "session"}
        ]
        clinical = clinical.drop(columns=overlapping)
        design = uncertainty.merge(
            clinical,
            on="case_id",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_clinical"),
        )
        for key in ("subject", "session"):
            clinical_key = f"{key}_clinical"
            if clinical_key in design:
                disagreement = design[key].astype(str) != design[clinical_key].astype(str)
                if disagreement.any():
                    raise ValueError(f"{key} disagrees between uncertainty and clinical tables")
                design = design.drop(columns=clinical_key)
    else:
        if "subject" not in clinical:
            raise ValueError("Clinical table needs either case_id or subject/participant_id")
        if clinical["subject"].duplicated().any():
            raise ValueError("Subject-level clinical table contains duplicate subjects")
        overlapping = [
            column for column in clinical.columns if column in uncertainty.columns and column != "subject"
        ]
        clinical = clinical.drop(columns=overlapping)
        design = uncertainty.merge(clinical, on="subject", how="inner", validate="many_to_one")
        design = one_scan_per_subject(design)

    if "wab_days" not in design or "age_at_stroke" not in design:
        raise ValueError("Clinical table must provide age_at_stroke and wab_days")
    design["log1p_wab_days"] = np.log1p(pd.to_numeric(design["wab_days"], errors="coerce"))
    design[outcome] = pd.to_numeric(design[outcome], errors="coerce")
    design = design.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome])
    design = one_scan_per_subject(design)
    if design["subject"].duplicated().any():
        raise RuntimeError("More than one observation per subject remains after case selection")
    return design.sort_values(["subject", "case_id"]).reset_index(drop=True)


def pearson_r(observed: np.ndarray, predicted: np.ndarray) -> float:
    if observed.size < 2 or np.isclose(np.std(observed), 0) or np.isclose(np.std(predicted), 0):
        return math.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def nested_cv_predictions(
    design: pd.DataFrame,
    outcome: str,
    feature_sets: dict[str, list[str]],
    outer_folds: int,
    inner_folds: int,
    alphas: list[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(design)
    if n < max(10, outer_folds):
        raise ValueError(
            f"Only {n} complete subjects are available; at least {max(10, outer_folds)} are required"
        )
    if not 2 <= outer_folds <= n:
        raise ValueError("--outer-folds must be between 2 and the number of subjects")
    if inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2")
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Every --alpha-grid value must be finite and positive")

    y = design[outcome].to_numpy(dtype=float)
    predictions = design[["case_id", "subject", "session", outcome]].copy()
    metrics: list[dict[str, object]] = []
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    for model_name, features in feature_sets.items():
        missing = [feature for feature in features if feature not in design]
        if missing:
            raise ValueError(f"Features missing for {model_name}: {missing}")
        x = design[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        predicted = np.full(n, np.nan, dtype=float)
        chosen_alpha = np.full(n, np.nan, dtype=float)
        fold_id = np.full(n, -1, dtype=int)
        for fold, (train, test) in enumerate(outer.split(x), start=1):
            n_inner = min(inner_folds, len(train))
            if n_inner < 2:
                raise ValueError("Not enough training subjects for inner cross-validation")
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge()),
                ]
            )
            search = GridSearchCV(
                pipeline,
                {"ridge__alpha": alphas},
                scoring="neg_mean_absolute_error",
                cv=KFold(n_splits=n_inner, shuffle=True, random_state=seed + fold),
                n_jobs=1,
                refit=True,
            )
            search.fit(x[train], y[train])
            predicted[test] = search.predict(x[test])
            chosen_alpha[test] = float(search.best_params_["ridge__alpha"])
            fold_id[test] = fold

        predictions[f"predicted_{model_name}"] = predicted
        predictions[f"outer_fold_{model_name}"] = fold_id
        predictions[f"alpha_{model_name}"] = chosen_alpha
        metrics.append(
            {
                "model": model_name,
                "n": n,
                "features": ";".join(features),
                "mae": float(mean_absolute_error(y, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
                "r2": float(r2_score(y, predicted)),
                "pearson_r": pearson_r(y, predicted),
                "outer_folds": outer_folds,
                "inner_folds": inner_folds,
            }
        )
    return predictions, pd.DataFrame(metrics)


def make_prediction_figure(
    predictions: pd.DataFrame, metrics: pd.DataFrame, outcome: str, output_path: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tr2strokeseg-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = metrics["model"].tolist()
    figure, axes = plt.subplots(1, len(model_names), figsize=(4.2 * len(model_names), 4), sharex=True, sharey=True, constrained_layout=True)
    if len(model_names) == 1:
        axes = [axes]
    observed = predictions[outcome].to_numpy(dtype=float)
    limits = [float(np.nanmin(observed)), float(np.nanmax(observed))]
    padding = max(2.0, 0.05 * (limits[1] - limits[0]))
    limits = [limits[0] - padding, limits[1] + padding]
    for axis, model_name in zip(axes, model_names, strict=True):
        predicted = predictions[f"predicted_{model_name}"].to_numpy(dtype=float)
        row = metrics.loc[metrics["model"] == model_name].iloc[0]
        axis.scatter(observed, predicted, s=22, alpha=0.7, edgecolor="none")
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(f"{model_name.replace('_', ' ')}\nMAE={row.mae:.2f}, R²={row.r2:.2f}")
        axis.set_xlabel("Observed WAB AQ")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Nested-CV predicted WAB AQ")
    figure.suptitle("ARC aphasia prediction with lesion uncertainty", fontweight="bold")
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.uncertainty_manifest is None or not args.uncertainty_manifest.is_file():
        raise FileNotFoundError(f"--uncertainty-manifest not found: {args.uncertainty_manifest}")
    if not args.clinical_table.is_file():
        raise FileNotFoundError(f"--clinical-table not found: {args.clinical_table}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    uncertainty = read_table(args.uncertainty_manifest)
    clinical = read_table(args.clinical_table)
    design = build_design(uncertainty, clinical, args.outcome)
    predictions, metrics = nested_cv_predictions(
        design,
        args.outcome,
        FEATURE_SETS,
        args.outer_folds,
        args.inner_folds,
        args.alpha_grid,
        args.seed,
    )
    design.to_csv(args.output_dir / "aq_uncertainty_design.csv", index=False)
    predictions.to_csv(args.output_dir / "aq_nested_cv_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "aq_nested_cv_metrics.csv", index=False)
    make_prediction_figure(
        predictions,
        metrics,
        args.outcome,
        args.output_dir / "aq_uncertainty_prediction.png",
    )
    audit = {
        "uncertainty_manifest": str(args.uncertainty_manifest.resolve()),
        "clinical_table": str(args.clinical_table.resolve()),
        "outcome": args.outcome,
        "n_subjects": len(design),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "alpha_grid": args.alpha_grid,
        "feature_sets": FEATURE_SETS,
        "probability_calibration_warning": (
            "nnU-Net softmax values are treated as predictive scores and were not "
            "externally calibrated on ARC."
        ),
    }
    with (args.output_dir / "aq_prediction_config.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")

    print(f"Analyzed {len(design)} independent subjects")
    print(metrics[["model", "mae", "rmse", "r2", "pearson_r"]].to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Stroke BrainSuite Analysis

Utilities for running the trained ATLAS2 stroke nnU-Net model on the sample ARC
MRI archive and keeping BrainSuite-side preprocessing outputs together.

Default input:

```bash
/home/ajoshi/Desktop/sample_arc
```

Default model:

```bash
/home/ajoshi/Projects/TR2preproc/supp_data/models/nnUNet_results
```

Run the full sample pipeline:

```bash
python stroke_brainsuite_analysis/run_sample_arc_stroke_pipeline.py
```

Useful options:

```bash
python stroke_brainsuite_analysis/run_sample_arc_stroke_pipeline.py \
  --limit 2 \
  --device cpu \
  --folds 0 \
  --checkpoint checkpoint_best.pth
```

The script writes preprocessed inputs, BrainSuite BSE outputs, nnU-Net MNI-space
predictions, inverse-transformed source-space stroke masks, and a CSV manifest
under `stroke_brainsuite_analysis/outputs/sample_arc` by default.

## Probabilistic lesion output and uncertainty

`run_arc_lesion_uncertainty.py` consumes the full-head 1 mm MNI T1 images made
by the ARC stroke-inpainting pipeline. For every scan it exports:

- uncalibrated nnU-Net lesion probability, `p_lesion`, as float32 NIfTI;
- normalized binary predictive entropy as float32 NIfTI;
- a `p_lesion >= 0.5` binary mask;
- scalar probability and uncertainty features in JSON/CSV; and
- a probability/entropy quality-control PNG.

Here `p_lesion` is a predictive probability, not the p-value from a statistical
hypothesis test. The entropy is zero for predictions near 0 or 1 and reaches
one at `p_lesion = 0.5`. The softmax probabilities have not been externally
calibrated on ARC and should be treated as predictive scores until calibration
is assessed on held-out manual delineations.

Run one local subject on CPU:

```bash
python stroke_brainsuite_analysis/run_arc_lesion_uncertainty.py \
  --mni-root stroke_brainsuite_analysis/outputs/subjectwise_stroke_inpainting.old \
  --output-dir stroke_brainsuite_analysis/outputs/lesion_uncertainty_local \
  --subject sub-M2300 \
  --device cpu \
  --disable-tta
```

On a CARC login node, preview and then submit one restartable GPU job per ARC
subject:

```bash
cd /project2/ajoshi_1183/Projects/TR2StrokeSeg

stroke_brainsuite_analysis/submit_arc_lesion_uncertainty_carc.sh \
  --dry-run -- --folds 0 --disable-tta
stroke_brainsuite_analysis/submit_arc_lesion_uncertainty_carc.sh \
  -- --folds 0 --disable-tta
```

The CARC defaults are:

```text
input:  /project2/ajoshi_1183/data/ARC/derivatives/stroke_inpainting
output: /project2/ajoshi_1183/data/ARC/derivatives/lesion_uncertainty
model:  /project2/ajoshi_1183/data/TR2/nnUNet_results
```

Monitor and audit the jobs with:

```bash
squeue -u "$USER" | grep pstroke
find /project2/ajoshi_1183/data/ARC/derivatives/lesion_uncertainty \
  -name processing_error.json -print
```

Completed subjects have a `lesion_uncertainty_complete` marker and are skipped
if the launcher is run again. Remove only a specific subject marker when that
subject intentionally needs to be resumed; pass `-- --overwrite` to recompute
all of that subject's outputs.

## Aphasia Quotient prediction

After all uncertainty jobs finish, compare clinical-only, hard-mask,
probability, and probability-plus-entropy feature sets with nested
cross-validation:

```bash
python stroke_brainsuite_analysis/predict_aphasia_quotient_from_uncertainty.py \
  --uncertainty-manifest \
    /project2/ajoshi_1183/data/ARC/derivatives/lesion_uncertainty/uncertainty_manifest.csv \
  --clinical-table \
    stroke_brainsuite_analysis/Inpainting_Code/analysis/arc_inpainting/case_metrics.csv \
  --output-dir \
    /project2/ajoshi_1183/data/ARC/derivatives/aq_uncertainty_prediction
```

The preferred clinical input is the case-level table above because it selects
the same one-scan-per-participant cohort used by the current WAB analysis. A
subject-level `participants.tsv` is also accepted; when multiple scans exist,
the script selects the scan closest to `wab_days`. Outputs include the joined
design table, out-of-fold predictions, model comparison metrics, a figure, and
an exact run-configuration JSON file.

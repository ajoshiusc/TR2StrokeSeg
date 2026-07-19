# MICCAI workshop manuscript assets

The manuscript is `../../LesionAwareStrokeMICCAIWorkshop2026.tex`.  From the
`Inpainting_Code` directory, build it with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  LesionAwareStrokeMICCAIWorkshop2026.tex
```

The completed-results PDF has ten manuscript pages; references start on page
11. This uses the user's allowance of up to two pages beyond the original
eight-page target.

## Representative-subject figure

From the repository root, regenerate the multipanel figure with:

```bash
.venv/bin/python stroke_brainsuite_analysis/make_representative_subject_figure.py \
  --case-id sub-M2278_ses-388_acq-tfl3p2_run-4_T1w
```

The script assembles the observed T1/lesion, lesion probability, entropy,
inpainted T1, atlas-mapped anatomy, deformation magnitude, radial displacement,
and log-Jacobian asymmetry for the same medium-volume subject. This is the
50th-percentile case in the existing small/medium/large inpainting figure; the
previous large-lesion case is not used because its inpainting failed visual QC.

## Cohort AQ-comparison results

The completed 210-participant nested-CV analysis is incorporated through:

```text
analysis/miccai_workshop/aq_results.tex
```

The source CARC results are under
`ARC/derivatives/aq_mass_effect_comparison`. The manuscript reports all six
models, bootstrap mean-advantage intervals, Holm-adjusted paired Wilcoxon tests,
and the direct uncertainty-versus-uncertainty-plus-deformation comparison.

Regenerate the paper-specific comparison figure from the saved CSV files with:

```bash
.venv/bin/python \
  stroke_brainsuite_analysis/Inpainting_Code/make_aq_paper_figure.py
```

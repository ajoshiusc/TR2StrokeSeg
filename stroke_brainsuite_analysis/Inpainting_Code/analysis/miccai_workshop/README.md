# MICCAI workshop manuscript assets

The manuscript is `../../LesionAwareStrokeMICCAIWorkshop2026.tex`.  From the
`Inpainting_Code` directory, build it with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  LesionAwareStrokeMICCAIWorkshop2026.tex
```

The current PDF has eight manuscript pages; references start on page 9.

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

## Inserting cohort AQ-comparison results

The paper intentionally does not invent results for CARC jobs that have not yet
finished.  When the nested-CV AQ comparison completes, copy its generated macro
file to:

```text
analysis/miccai_workshop/aq_results.tex
```

The manuscript detects that file automatically and replaces the prospective AQ
results paragraph with the cohort size, baseline and deformation-aware MAEs,
paired advantage and confidence interval, and permutation p-value. Rebuild the
PDF afterward with the command above.

# ARC inpainting validation

This directory contains the reproducible analysis added to
`LesionInpaintingPipelineBrainWorks2026.tex`.

Run from `Inpainting_Code` with an environment that provides NumPy, pandas,
SciPy, NiBabel, and Matplotlib:

```bash
MPLCONFIGDIR=/tmp/mplconfig python analyze_arc_inpainting.py
```

The default inputs are:

- `/home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/brainsuite_anatomical_raw`
- `/home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/brainsuite_anatomical_bidsapp`
- `/home/ajoshi/project2_ajoshi_1183/data/ARC/derivatives/stroke_inpainting`
- `/home/ajoshi/project2_ajoshi_1183/data/ARC/participants.tsv`

The script case-matches completed direct and inpainted BrainSuite results and
uses one observation per participant. It evaluates:

1. left/right asymmetry of BrainSuite tissue volume, mid-cortical area, and
   thickness in homologous ROI pairs entirely outside the dilated inpainting
   target (any target overlap excludes the pair);
2. partial Spearman associations between WAB Aphasia Quotient and the original
   binary lesion mask localized with post-inpainting BrainSuite labels.

Generated intensities and morphometry from the target are not treated as
biological measurements. Boundary continuity and exact preservation outside
the target are retained only as implementation quality controls, not study
endpoints.

`analysis_summary.json` records the exact paths, sample counts, thresholds,
random seed, software versions, estimates, confidence intervals, and tests.
`paper_results.tex` is generated from the same results and is imported by the
manuscript, avoiding manual transcription of numerical values.

Main outputs:

- `case_metrics.csv`: one row per matched participant;
- `roi_pair_metrics.csv`: all homologous ROI pairs and their target-overlap class;
- `spared_roi_metrics.csv`: zero-overlap pairs used in the morphometric analysis;
- `affected_roi_metrics.csv`: excluded target-overlapping/margin pairs retained
  as an audit table only;
- `summary_statistics.csv`: the three paired spared-anatomy endpoints;
- `clinical_associations.csv`: WAB construct-validity analyses;
- `arc_inpainting_validation.pdf`: manuscript figure;
- `paper_results.tex`: LaTeX result macros.

## Representative processing figure

Run the qualitative small/medium/large lesion figure with:

```bash
MPLCONFIGDIR=/tmp/mplconfig python make_representative_brainsuite_figure.py
```

Cases are chosen automatically as the completed matched acquisitions closest
to the lesion-volume 10th, 50th, and 90th percentiles. The direct labels are
mapped to the pre-inpainting MNI grid with FSL FLIRT so paired axial panels use
the same geometry. The output includes:

- `representative_outputs/arc_representative_brainsuite_outputs.pdf`;
- `representative_outputs/arc_representative_brainsuite_outputs.png`;
- `representative_outputs/representative_cases.csv`.

This figure is qualitative. Labels and surface geometry inside the outlined
generated region are not used as biological measurements.

No source ARC data are modified.

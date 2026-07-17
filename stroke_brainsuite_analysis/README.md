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

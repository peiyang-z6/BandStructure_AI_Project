# 2026-06-05 Probability Sharpening and Local Quadratic Fit

## Actions

- Implemented sharp learned temperature in `ExtremumExpectedGapHead` with `0.01 <= tau <= 0.5`.
- Added straight-through hard-extremum weights for the line-mode gap head, so forward gap predictions use the selected VBM/CBM k-point energies while gradients still flow through soft probabilities.
- Raised final entropy sharpening to `0.02` and extremum peak supervision to `1.0`.
- Fixed the topology rule to locate VBM/CBM zero-distance points after denormalizing k-distance channels.
- Replaced boundary-sensitive finite-difference curvature annotations with local quadratic fitting around extrema.
- Refreshed the full non-download pipeline with existing downloaded Materials Project data preserved.

## Final Command

```powershell
python scripts\run_full_pipeline.py --fresh
```

## Final OOD Metrics

- Line-mode Gap MAE: `0.171231 eV`
- Gap identity large residual rate: `0.067797`
- Direct Gap Recall: `1.000000`
- Type accuracy: `0.983051`
- Macro F1: `0.649749`
- Parity R2: `0.964435`
- Direct AUC: `0.982270`

The two locked north-star thresholds were met: gap identity residual rate is below `10%`, and Direct Gap Recall is above `0.90`.

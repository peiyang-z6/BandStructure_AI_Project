# Agent Log: Full SSL MBM Pretraining Run

- Date: 2026-06-04
- Scope: complete SSL MBM pretraining on current OOD tensors
- Data: `data_cache/ood_tensors/band_tensors_ood_split.npz`
- Report: `reports/ssl_mbm_pretraining_20260604.md`

## Run

- Epochs: 100
- Batch size: 16
- Mask ratio: 0.25
- Sign penalty weight: 2.0
- Consistency penalty weight: 0.2
- Model: `SSLEncoder(num_bands=4, seq_len=128, d_model=128, num_heads=4, num_layers=4, dff=256, projection_dim=64)`

## Results

- Best validation loss: `0.38232` at epoch 92.
- Final validation loss at epoch 100: `0.49985`.
- `models/ssl_mbm_pretrained.keras` now contains the best checkpoint weights.
- `models/ssl_mbm_final_epoch100.keras` preserves the final epoch-100 weights.
- `checkpoints/ssl_mbm/ckpt-best` and `checkpoints/ssl_mbm/ckpt-100` are retained.
- Previous test `ckpt-20` files were deleted.
- Old test TensorBoard event files were removed.

## Notes

- The mask replaces selected k-point feature vectors with zeros.
- Curvature channels are masked together with energy channels to avoid leakage, because curvature is derived from energy.
- Curvature sign loss remained zero during logged validation, indicating no detected VBM/CBM curvature sign violations at extrema.
- Training ran on CPU in native Windows because TensorFlow 2.21 has no native-Windows CUDA build.

# Agent Log: SSL MBM Pretraining

- Date: 2026-06-04
- Scope: SSL pretraining readiness, MBM trainer update, GPU diagnosis, first MBM run
- Files changed:
  - `scripts/train_ssl.py`
  - `src/models/band_structure_encoder.py`
- Outputs:
  - `models/ssl_mbm_pretrained.keras`
  - `models/ssl_mbm_norm_stats.json`
  - `checkpoints/ssl_mbm/ckpt-best`
  - `checkpoints/ssl_mbm/ckpt-10`
  - `checkpoints/ssl_mbm/ckpt-20`
  - `logs/ssl_mbm/`

## Readiness

- Data pipeline is ready:
  - `data_cache/ood_tensors/band_tensors_ood_split.npz`
  - Train tensor shape: `(141, 2, 128, 2)`
  - Test/OOD tensor shape: `(59, 2, 128, 2)`
  - Spacegroup overlap: empty

## Training Update

- `scripts/train_ssl.py` now defaults to `tensor_mbm` mode.
- Input is flattened from `(N, 2, 128, 2)` to `(N, 128, 4)`.
- Feature order: `[vbm_energy, vbm_curvature, cbm_energy, cbm_curvature]`.
- Masked Band Modeling uses a k-point mask ratio of 0.2.
- Loss includes:
  - masked reconstruction MSE for energy + curvature channels
  - VBM/CBM curvature sign penalty
  - curvature channel consistency penalty

## Run

Command:

```bash
conda run -n bandstructure_ai_project python scripts/train_ssl.py --mode tensor_mbm --epochs 20 --batch-size 16 --mask-ratio 0.2 --d-model 128 --num-heads 4 --num-layers 4 --dff 256 --projection-dim 64 --checkpoint-dir checkpoints/ssl_mbm --log-dir logs/ssl_mbm --model-dir models
```

Result:

- Best validation loss: `0.52001` at epoch 9.
- Final epoch 20 validation loss: `0.52196`.
- Model reload validation passed with `tf.keras.models.load_model(..., compile=False)`.

## GPU Diagnosis

- Native Windows conda env:
  - TensorFlow: `2.21.0`
  - `tf.test.is_built_with_cuda()`: `False`
  - `tf.config.list_physical_devices("GPU")`: `[]`
- `nvidia-smi` detects the RTX 4060 Laptop GPU on Windows.
- WSL2 Ubuntu also sees the NVIDIA GPU through `nvidia-smi`.
- Root cause: TensorFlow 2.11+ has no official native-Windows CUDA GPU support. Use WSL2/Linux for TensorFlow CUDA training.

# Artifact layout

The latest accepted formal experiment is:

`aflow_noleak_v5_30k_seed42`

Its byte-verified server return populates all four classified directories:

- `models/aflow_noleak_v5_30k_seed42/`: restored-best SSL encoder, SSL normalization statistics, and accepted supervised weights/config;
- `checkpoints/aflow_noleak_v5_30k_seed42/`: SSL best/last checkpoints and supervised best checkpoint;
- `reports/aflow_noleak_v5_30k_seed42/`: metrics, predictions, uncertainty/calibration figures, GPU/environment audits, manifests, import audit, result archive, and the dated formal report;
- `logs/aflow_noleak_v5_30k_seed42/`: full download, SSL, supervised, evaluation-resume, orchestration, TensorBoard and 15-second GPU-usage logs.

The previous accepted experiment `aflow_noleak_v4_seed42` remains immutable and reproducible in the same four classified directory types. It is a retained baseline, not the runtime latest model.

Legacy target-conditioned models, smoke checkpoints and superseded reports are not part of this runtime tree. Historical decisions remain in `PROJECT_BRAIN/agent_logs/`.

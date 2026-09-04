#!/usr/bin/env bash
# P0 3-seed training chain (server V100). Usage: ./p0_launch_3seeds.sh
# Each seed: SSL (60ep) + supervised train-only + evaluation-only +
#            seven-split frozen evaluation.
# Experiments: aflow_noleak_v7_60k_seed{42,2024,7} (seed 42 reuses its
# inherited v7 SSL encoder; seeds 2024/7 train SSL from scratch).
set -euo pipefail
PROJ=/home/zhao/BandStructure_AI_60k_20260903
cd "$PROJ"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bandstructure_gpu30k
TENSOR_NPZ=data/processed/aflow/ood_tensors_v7_60000_seed42/band_tensors_ood_split.npz
REPORT_DATE=20260904
PY=python
LOG=/home/zhao/p0_train_chain.log

for SEED in 42 2024 7; do
  EXPERIMENT="aflow_noleak_v7_60k_seed${SEED}"
  mkdir -p artifacts/checkpoints/$EXPERIMENT/ssl artifacts/checkpoints/$EXPERIMENT/supervised \
           artifacts/logs/$EXPERIMENT/ssl artifacts/models/$EXPERIMENT artifacts/reports/$EXPERIMENT
  echo "===== P0 seed ${SEED} =====" >> "$LOG"

  if [ "$SEED" = "42" ] && [ -f artifacts/models/aflow_noleak_v7_60k_seed42/ssl_mbm_pretrained.keras ]; then
    echo "seed 42: reusing accepted v7 SSL encoder (already trained)" >> "$LOG"
  else
    echo "----- SSL seed ${SEED} -----" >> "$LOG"
    $PY scripts/train_ssl.py --tensor-npz "$TENSOR_NPZ" --epochs 60 --batch-size 32 \
      --mask-ratio 0.25 --random-state "$SEED" --sign-weight 2.0 --consistency-weight 0.0 \
      --d-model 128 --num-heads 4 --num-layers 4 --dff 256 --projection-dim 64 \
      --strain-scale 0.01 --checkpoint-dir artifacts/checkpoints/$EXPERIMENT/ssl \
      --log-dir artifacts/logs/$EXPERIMENT/ssl --model-dir artifacts/models/$EXPERIMENT \
      --disable-strain-augmentation --require-gpu >> "$LOG" 2>&1 \
      || { echo "SSL seed $SEED FAILED" >> "$LOG"; exit 1; }
  fi

  if [ "$SEED" = "42" ]; then
    # P0 supersedes the v7 single-head supervised run under the same
    # experiment name (user-approved D2). Snapshot the accepted v7 artifacts
    # first so the v7 acceptance manifest hashes stay verifiable.
    SNAP="artifacts/models/aflow_noleak_v7_60k_seed42/_v7_accepted_snapshot"
    if [ ! -d "$SNAP" ]; then
      mkdir -p "$SNAP"
      cp -n artifacts/models/aflow_noleak_v7_60k_seed42/finetuned.weights.h5 "$SNAP/" 2>/dev/null || true
      cp -n artifacts/reports/aflow_noleak_v7_60k_seed42/metrics_summary.json "$SNAP/" 2>/dev/null || true
      cp -n artifacts/reports/aflow_noleak_v7_60k_seed42/inner_selection_manifest.json "$SNAP/" 2>/dev/null || true
      echo "seed 42: v7 accepted artifacts snapshotted to $SNAP" >> "$LOG"
    fi
  fi

  echo "----- supervised train-only seed ${SEED} -----" >> "$LOG"
  $PY scripts/finetune_supervised.py --tensor-npz "$TENSOR_NPZ" \
    --encoder artifacts/models/$EXPERIMENT/ssl_mbm_pretrained.keras \
    --norm artifacts/models/$EXPERIMENT/ssl_mbm_norm_stats.json \
    --epochs 60 --batch-size 32 --learning-rate 0.001 \
    --encoder-learning-rate 1e-05 --type-weight 2.0 --freeze-layers 2 \
    --topology-weight 0.3 --entropy-weight 0.02 --extremum-weight 1.0 \
    --random-state "$SEED" \
    --output-dir artifacts/reports/$EXPERIMENT \
    --checkpoint-dir artifacts/checkpoints/$EXPERIMENT/supervised \
    --model-path artifacts/models/$EXPERIMENT/finetuned.weights.h5 \
    --experiment-id "$EXPERIMENT" --source aflow \
    --report-date "$REPORT_DATE" --require-gpu --train-only >> "$LOG" 2>&1 \
    || { echo "supervised seed $SEED FAILED" >> "$LOG"; exit 1; }

  echo "----- evaluation-only seed ${SEED} -----" >> "$LOG"
  $PY scripts/finetune_supervised.py --tensor-npz "$TENSOR_NPZ" \
    --encoder artifacts/models/$EXPERIMENT/ssl_mbm_pretrained.keras \
    --norm artifacts/models/$EXPERIMENT/ssl_mbm_norm_stats.json \
    --batch-size 32 --random-state "$SEED" \
    --output-dir artifacts/reports/$EXPERIMENT \
    --checkpoint-dir artifacts/checkpoints/$EXPERIMENT/supervised \
    --model-path artifacts/models/$EXPERIMENT/finetuned.weights.h5 \
    --experiment-id "$EXPERIMENT" --source aflow \
    --report-date "$REPORT_DATE" --require-gpu --evaluation-only >> "$LOG" 2>&1 \
    || { echo "evaluation seed $SEED FAILED" >> "$LOG"; exit 1; }

  echo "----- seven-split frozen evaluation seed ${SEED} -----" >> "$LOG"
  $PY scripts/evaluate_seven_splits.py \
    --npz data/processed/aflow/ood_tensors_v7_60000_seed42/band_tensors_full.npz \
    --manifest data/processed/aflow/ood_tensors_v7_60000_seed42/ood_split_manifest.json \
    --splits data/processed/aflow/ood_tensors_v7_60000_seed42/seven_splits_test_ids.json \
    --labels data/processed/aflow/ood_tensors_v7_60000_seed42/three_task_labels.npz \
    --metadata data/raw/aflow/snapshots/aflow_60000_20260831/aflow_metadata.json \
    --model artifacts/models/$EXPERIMENT/finetuned.weights.h5 \
    --encoder artifacts/models/$EXPERIMENT/ssl_mbm_pretrained.keras \
    --norm artifacts/models/$EXPERIMENT/ssl_mbm_norm_stats.json \
    --output-dir artifacts/reports/$EXPERIMENT \
    --n-boot 1000 --random-state 42 --batch-size 256 >> "$LOG" 2>&1 \
    || { echo "seven-split seed $SEED FAILED" >> "$LOG"; exit 1; }
done
echo "P0 3-SEED CHAIN COMPLETE" >> "$LOG"
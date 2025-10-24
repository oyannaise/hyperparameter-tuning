#!/bin/bash

# Run training with best hyperparameters from Project 1
# These values are from models/lightning_logs/version_0/hparams.yaml

echo "Starting training with best hyperparameters..."
echo "=============================================="
echo "Learning Rate: 2.8e-5"
echo "Warmup Steps: 150"
echo "Scheduler: linear"
echo "Batch Size: 16"
echo "Weight Decay: 0.01"
echo "Epochs: 3"
echo "=============================================="

python main.py \
  --checkpoint_dir /app/checkpoints \
  --lr 2.8e-5 \
  --warmup_steps 150 \
  --scheduler linear \
  --batch_size 16 \
  --weight_decay 0.01 \
  --epochs 3 \
  --project_name mrpc-docker-training \
  --run_name docker-best-params

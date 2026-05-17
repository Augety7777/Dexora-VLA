#!/bin/bash

# Script to compute logpi values for the dataset
# This should be run before training the scoring model

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Starting logpi computation..."

python compute_logpi.py \
    --model_path "checkpoints/dexrdt-400m-v5" \
    --dataset_path "dataprocess/output/airbot_dexterous_bimanual_dexterous_manipulation" \
    --output_file "new_lerobot_logpi_values.json" \
    --batch_size 16 \
    --num_noise_steps 4 \
    --frame_stride 10
    # --verbose_timing \


echo "Logpi computation completed. Results saved to logpi_values.json"

#!/usr/bin/env bash
# Stage-3: data-quality-aware post-training (Dexora §III-D, Eq.(8)).
# Loads the pretrained policy (stage-1) and the frozen discriminator (stage-2),
# then fine-tunes the policy on the real dataset with DWBC-weighted MSE.

set -e

export NCCL_DEBUG=INFO
export NCCL_MIN_NCHANNELS=4
export NCCL_IB_DISABLE=1

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"

# --- Edit these to point at your trained stage-1 / stage-2 checkpoints ---
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/dexora-400m-pretrain/checkpoint-100000}"
SCORING_CKPT="${SCORING_CKPT:-checkpoints/scoring-model-v1/checkpoint-5000/scoring_model/pytorch_model.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/dexora-400m-posttrain}"

export WANDB_PROJECT="dexora-posttrain"
export WANDB_MODE=offline

mkdir -p "$OUTPUT_DIR"
echo "Stage-3 output -> $OUTPUT_DIR"
echo "  stage-1 ckpt: $STAGE1_CKPT"
echo "  scoring ckpt: $SCORING_CKPT"

accelerate launch main_posttrain.py \
    --config_path="./configs/base_400m.yaml" \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --stage1_ckpt="$STAGE1_CKPT" \
    --scoring_ckpt="$SCORING_CKPT" \
    --dwbc_eta=0.5 \
    --dwbc_w_min=0.0 \
    --dwbc_w_max=5.0 \
    --dwbc_warmup_steps=1000 \
    --train_batch_size=8 \
    --sample_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=50001 \
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=20 \
    --lr_scheduler="constant" \
    --learning_rate=5e-5 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from="lerobot" \
    --report_to=wandb

# export NCCL_SOCKET_IFNAME=eth1
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_NSOCKET_PERTHREAD=4
# export NCCL_MIN_NCHANNELS=4
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_TIMEOUT=14
export NCCL_DEBUG=INFO
export NCCL_MIN_NCHANNELS=4
export NCCL_IB_DISABLE=1

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/dexrdt-1115pap-v2"
# export OUTPUT_DIR="./checkpoints/debug"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="cutlass"

export WANDB_PROJECT="dex-rdt"
export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=7

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

# deepspeed --hostfile=hostfile.txt main.py \
    # --pretrained_model_name_or_path="checkpoints/dexrdt-400m-v5/checkpoint-95000" \
accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100001 \
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=30 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from="bson" \
    --report_to=wandb

    # --resume_from_checkpoint="checkpoint-45000" \
    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \

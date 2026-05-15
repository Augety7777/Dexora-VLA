#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Set paths (modify these according to your setup)
PRETRAINED_MODEL_PATH="checkpoints/dexrdt-400m-v1/checkpoint-95000"
LANG_EMBEDDINGS_PATH="outs/action4.pt"
DATA_DIR="data/ours"

# Create output directory
mkdir -p eval_results

# Run evaluation
python scripts/eval_action_curves.py \
    --pretrained-model-path $PRETRAINED_MODEL_PATH \
    --lang-embeddings-path $LANG_EMBEDDINGS_PATH \
    --data-dir $DATA_DIR \
    --episode-idx 150 \
    --inference-interval 32 \
    --output-dir eval_results

echo "Evaluation completed! Check eval_results/ for generated plots."

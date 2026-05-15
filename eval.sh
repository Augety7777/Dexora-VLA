python -m scripts.eval_action_curves \
    --pretrained-model-path checkpoints/dexrdt-400m-v5/checkpoint-95000 \
    --lang-embeddings-path outs/action15.pt \
    --data-dir "data/ours/true" \
    --episode-name "action15/episode_0" \
    --output-dir v5-95k-15
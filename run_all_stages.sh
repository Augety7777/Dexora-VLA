#!/usr/bin/env bash
# ============================================================================
# Dexora end-to-end three-stage training pipeline.
#
# This script chains:
#
#   1. Stage-1   pretrain on simulation data           -> train_ours.sh
#   2. Stage-2a  pre-screening (Aep, Jep) -> Spre      -> analyze_jerk.sh
#   3. Stage-2b  Spre -> Shigh post-validation         -> replay_validate.py
#   4. Stage-2c  log-pi proxy (Eq.(5))                 -> compute_logpi.py
#   5. Stage-2d  discriminator PU training (Eq.(7))    -> train_scoring.sh
#   6. Stage-3   quality-aware post-training (Eq.(8))  -> post_train.sh
#
# Override stages with the START_STAGE / END_STAGE env vars, e.g.
#
#   START_STAGE=4 END_STAGE=6 bash run_all_stages.sh
#
# All outputs land in $RUN_DIR (default: ./runs/dexora-$(date +%Y%m%d-%H%M%S)).
# ============================================================================

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-6}"
RUN_DIR="${RUN_DIR:-./runs/dexora-$(date +%Y%m%d-%H%M%S)}"

STAGE1_OUT="${STAGE1_OUT:-$RUN_DIR/stage1-pretrain}"
SPRE_OUT="${SPRE_OUT:-$RUN_DIR/spre.json}"
SHIGH_OUT="${SHIGH_OUT:-$RUN_DIR/shigh.json}"
LOGPI_OUT="${LOGPI_OUT:-$RUN_DIR/logpi.json}"
SCORING_OUT="${SCORING_OUT:-$RUN_DIR/stage2-scoring}"
STAGE3_OUT="${STAGE3_OUT:-$RUN_DIR/stage3-posttrain}"

REPLAY_VERIFIER="${REPLAY_VERIFIER:-trust_spre}"   # trust_spre / energy / mujoco
CONFIG_PATH="${CONFIG_PATH:-configs/base_400m.yaml}"

mkdir -p "$RUN_DIR"
echo "==> RUN_DIR=$RUN_DIR"
echo "==> Stages: $START_STAGE..$END_STAGE"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_stage() {
    local n="$1" name="$2"
    if (( n < START_STAGE || n > END_STAGE )); then
        echo "==> [skip ] stage $n: $name"
        return 0
    fi
    echo
    echo "================================================================"
    echo "==> [stage] $n: $name"
    echo "================================================================"
}

# ---------------------------------------------------------------------------
# 1. Stage-1: pretrain on simulation
# ---------------------------------------------------------------------------
run_stage 1 "Stage-1 pretrain on simulation (train_ours.sh)"
if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
    OUTPUT_DIR="$STAGE1_OUT" bash train_ours.sh
fi

# ---------------------------------------------------------------------------
# 2. Stage-2a: pre-screening (Aep, Jep) -> Spre
# ---------------------------------------------------------------------------
run_stage 2 "Stage-2a pre-screening (analyze_jerk.sh -> Spre)"
if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
    bash analyze_jerk.sh
    # analyze_jerk.sh writes to new_lerobot_jerk/complete_analysis_results.json
    cp -v new_lerobot_jerk/complete_analysis_results.json "$SPRE_OUT"
fi

# ---------------------------------------------------------------------------
# 3. Stage-2b: replay-based post-validation -> Shigh
# ---------------------------------------------------------------------------
run_stage 3 "Stage-2b replay-based post-validation -> Shigh"
if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
    python replay_validate.py \
        --pre_screening_file "$SPRE_OUT" \
        --output_file "$SHIGH_OUT" \
        --verifier "$REPLAY_VERIFIER"
fi

# ---------------------------------------------------------------------------
# 4. Stage-2c: log-pi proxy
# ---------------------------------------------------------------------------
run_stage 4 "Stage-2c log-pi proxy (compute_logpi.py)"
if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
    python compute_logpi.py \
        --config_path "$CONFIG_PATH" \
        --model_path "$STAGE1_OUT" \
        --output_file "$LOGPI_OUT" \
        --normalize_mode zscore
fi

# ---------------------------------------------------------------------------
# 5. Stage-2d: discriminator PU training
# ---------------------------------------------------------------------------
run_stage 5 "Stage-2d discriminator PU training (train_scoring.sh)"
if (( START_STAGE <= 5 && END_STAGE >= 5 )); then
    OUTPUT_DIR="$SCORING_OUT" \
    LOGPI_FILE="$LOGPI_OUT" \
    SHIGH_FILE="$SHIGH_OUT" \
        bash train_scoring.sh
fi

# ---------------------------------------------------------------------------
# 6. Stage-3: quality-aware post-training
# ---------------------------------------------------------------------------
run_stage 6 "Stage-3 quality-aware post-training (post_train.sh)"
if (( START_STAGE <= 6 && END_STAGE >= 6 )); then
    STAGE1_CKPT="$STAGE1_OUT" \
    SCORING_CKPT="$SCORING_OUT/final_model/pytorch_model.bin" \
    OUTPUT_DIR="$STAGE3_OUT" \
        bash post_train.sh
fi

echo
echo "==> All requested stages finished. Artifacts in $RUN_DIR"

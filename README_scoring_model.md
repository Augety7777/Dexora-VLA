# Scoring Model for Episode Quality Assessment

This scoring model implements a Positive-Unlabeled (PU) learning approach to assess episode quality in robotic demonstrations.

## Overview

The scoring model takes similar inputs to the main RDT model but outputs a 0-1 score indicating episode quality. It's designed for PU learning where:
- **Positive samples (De)**: Episodes from `episode_quality_analysis/complete_analysis_results.json` valid_episodes list
- **Unlabeled samples (Do)**: All other episodes (mixture of expert and non-expert)

## Architecture

- **Input**: State, action chunks, logpi (placeholder for future use)
- **Output**: Single score between 0-1 
- **Model**: Transformer-based architecture similar to RDT but smaller
- **Loss**: PU learning loss: `Ld = η*E[−log d] + E[−log(1−d)] − η*E[−log(1−d)]`

## Files Created

### Core Model
- `models/scoring_model.py` - ScoringModel and ScoringModelRunner classes
- `configs/scoring.yaml` - Configuration for scoring model

### Training
- `main_scoring.py` - Main training entry point
- `train/train_scoring.py` - Training loop with PU learning loss
- `train_scoring.sh` - Training shell script

### Evaluation
- `inference_scoring.py` - Inference script for model evaluation
- `test_scoring_model.py` - Unit tests for model functionality

## Usage

### Training
```bash
./train_scoring.sh
```

### Testing
```bash
python test_scoring_model.py
```

### Inference
```bash
python inference_scoring.py --model_path checkpoints/scoring-model-v1/final_model --num_samples 1000
```

## Key Features

1. **PU Learning Loss**: Implements the specified loss function with η parameter
2. **Logpi Interface**: Ready for future logpi inputs (currently uses zeros)
3. **Episode Quality Labels**: Automatically labels samples based on valid_episodes list
4. **Compatible Structure**: Similar to original RDT training pipeline
5. **Flexible Input**: Handles missing logpi gracefully

## Configuration

The scoring model uses smaller architecture than RDT:
- Hidden size: 512 (vs 1024 in RDT)
- Depth: 12 layers (vs 28 in RDT)
- Heads: 8 (vs 16 in RDT)

Adjust in `configs/scoring.yaml` under `model.scoring` section.

#!/usr/bin/env python
# coding=utf-8

import torch
import yaml
from models.scoring_model import ScoringModel, ScoringModelRunner


def test_scoring_model():
    """Test the scoring model with dummy data"""
    
    # Load config
    with open("configs/scoring.yaml", "r") as fp:
        config = yaml.safe_load(fp)
    
    # Create model
    runner = ScoringModelRunner(config)
    model = runner.model
    model.eval()
    
    # Create dummy inputs
    batch_size = 4
    state_dim = config['model']['state_token_dim']
    action_dim = config['model']['state_token_dim']
    action_chunk_size = config['common']['action_chunk_size']
    
    # Dummy data
    state = torch.randn(batch_size, state_dim)
    action_chunk = torch.randn(batch_size, action_chunk_size, action_dim)
    logpi_chunk = torch.randn(batch_size, 1)  # One logpi per data point
    
    print(f"Testing scoring model with:")
    print(f"  State shape: {state.shape}")
    print(f"  Action chunk shape: {action_chunk.shape}")
    print(f"  Logpi chunk shape: {logpi_chunk.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Test with logpi
        scores_with_logpi = model(state, action_chunk, logpi_chunk)
        print(f"  Scores with logpi: {scores_with_logpi.squeeze().tolist()}")
        
        # Test without logpi (should use zeros internally)
        scores_without_logpi = model(state, action_chunk)
        print(f"  Scores without logpi: {scores_without_logpi.squeeze().tolist()}")
    
    # Verify output shape and range
    assert scores_with_logpi.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {scores_with_logpi.shape}"
    assert torch.all(scores_with_logpi >= 0) and torch.all(scores_with_logpi <= 1), "Scores should be in [0, 1] range"
    
    print("✓ Scoring model test passed!")
    
    # Test model saving and loading
    save_path = "test_scoring_model_checkpoint"
    runner.save_pretrained(save_path)
    print(f"✓ Model saved to {save_path}")
    
    # Load and test
    new_runner = ScoringModelRunner(config)
    new_runner.load_pretrained(save_path)
    new_model = new_runner.model
    new_model.eval()
    
    with torch.no_grad():
        new_scores = new_model(state, action_chunk, logpi_chunk)
    
    # Verify loaded model produces same results
    assert torch.allclose(scores_with_logpi, new_scores, atol=1e-6), "Loaded model should produce same results"
    print("✓ Model save/load test passed!")
    
    # Clean up
    import shutil
    shutil.rmtree(save_path)
    
    return True


if __name__ == "__main__":
    test_scoring_model()

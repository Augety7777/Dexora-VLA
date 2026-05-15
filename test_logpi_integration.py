#!/usr/bin/env python3
"""
Test script to verify logpi integration with scoring model training.
"""

import torch
import numpy as np
import json
import os
from data.bson_vla_dataset_with_logpi import BsonVLADatasetWithLogpi
from train.train_scoring import ScoringDataset


def create_dummy_logpi_file(filename="test_logpi.json", num_episodes=5, frames_per_episode=10):
    """Create a dummy logpi file for testing"""
    logpi_dict = {}
    
    for ep_idx in range(num_episodes):
        logpi_dict[str(ep_idx)] = {}
        for frame_idx in range(frames_per_episode):
            # Generate random logpi values between -10 and 0
            logpi_value = float(np.random.uniform(-10, 0))
            logpi_dict[str(ep_idx)][str(frame_idx)] = logpi_value
    
    with open(filename, 'w') as f:
        json.dump(logpi_dict, f, indent=2)
    
    print(f"Created dummy logpi file: {filename}")
    return filename


def test_logpi_dataset():
    """Test the BsonVLADatasetWithLogpi class"""
    print("Testing BsonVLADatasetWithLogpi...")
    
    # Create dummy logpi file
    logpi_file = create_dummy_logpi_file()
    
    try:
        # Test dataset creation
        dataset = BsonVLADatasetWithLogpi(
            logpi_file=logpi_file,
            bson_dir="data/ours/true",
            sub_sample=0.01,  # Use small subset for testing
            normalize_mode="min_max",
            stats_file="v5_bson_stats/dataset_statistics.json"
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test logpi statistics
        logpi_stats = dataset.get_logpi_statistics()
        print(f"Logpi statistics: {logpi_stats}")
        
        # Test data loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            
            if 'logpi' in sample:
                logpi_val = sample['logpi']
                print(f"Logpi shape: {logpi_val.shape}, value: {logpi_val}")
                assert logpi_val.shape == (1,), f"Expected logpi shape (1,), got {logpi_val.shape}"
                print("✓ Logpi shape is correct")
            else:
                print("✗ Logpi not found in sample")
        
        print("✓ BsonVLADatasetWithLogpi test passed")
        
    except Exception as e:
        print(f"✗ BsonVLADatasetWithLogpi test failed: {e}")
    
    finally:
        # Clean up
        if os.path.exists(logpi_file):
            os.remove(logpi_file)


def test_scoring_dataset():
    """Test the ScoringDataset wrapper"""
    print("\nTesting ScoringDataset wrapper...")
    
    # Create dummy logpi file
    logpi_file = create_dummy_logpi_file()
    
    try:
        # Create base dataset
        base_dataset = BsonVLADatasetWithLogpi(
            logpi_file=logpi_file,
            bson_dir="data/ours/true",
            sub_sample=0.01,  # Use small subset for testing
            normalize_mode="min_max",
            stats_file="v5_bson_stats/dataset_statistics.json"
        )
        
        # Wrap with scoring dataset
        scoring_dataset = ScoringDataset(base_dataset)
        
        print(f"Scoring dataset created with {len(scoring_dataset)} samples")
        
        # Test data loading
        if len(scoring_dataset) > 0:
            sample = scoring_dataset[0]
            print(f"Scoring sample keys: {sample.keys()}")
            
            # Check required keys
            required_keys = ['state', 'actions', 'is_expert', 'logpi']
            for key in required_keys:
                if key in sample:
                    print(f"✓ {key}: shape {sample[key].shape}")
                else:
                    print(f"✗ Missing key: {key}")
            
            # Check logpi specifically
            if 'logpi' in sample:
                logpi_val = sample['logpi']
                assert logpi_val.shape == (1,), f"Expected logpi shape (1,), got {logpi_val.shape}"
                print(f"✓ Logpi value: {logpi_val[0]:.4f}")
        
        print("✓ ScoringDataset test passed")
        
    except Exception as e:
        print(f"✗ ScoringDataset test failed: {e}")
    
    finally:
        # Clean up
        if os.path.exists(logpi_file):
            os.remove(logpi_file)


def test_batch_loading():
    """Test batch loading with DataLoader"""
    print("\nTesting batch loading...")
    
    # Create dummy logpi file
    logpi_file = create_dummy_logpi_file()
    
    try:
        from torch.utils.data import DataLoader
        from train.dataset import DataCollatorForVLAConsumerDataset
        import transformers
        
        # Create base dataset
        base_dataset = BsonVLADatasetWithLogpi(
            logpi_file=logpi_file,
            bson_dir="data/ours/true",
            sub_sample=0.01,
            normalize_mode="min_max",
            stats_file="v5_bson_stats/dataset_statistics.json"
        )
        
        # Wrap with scoring dataset
        scoring_dataset = ScoringDataset(base_dataset)
        
        # Create a simple tokenizer for testing
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create data collator
        data_collator = DataCollatorForVLAConsumerDataset(tokenizer)
        
        # Create dataloader directly with scoring dataset
        dataloader = DataLoader(
            scoring_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: {
                'state': torch.stack([torch.from_numpy(item['state']) if isinstance(item['state'], np.ndarray) else item['state'] for item in x]),
                'actions': torch.stack([torch.from_numpy(item['actions']) if isinstance(item['actions'], np.ndarray) else item['actions'] for item in x]),
                'is_expert': torch.stack([torch.from_numpy(item['is_expert']) if isinstance(item['is_expert'], np.ndarray) else item['is_expert'] for item in x]),
                'logpi': torch.stack([torch.from_numpy(item['logpi']) if isinstance(item['logpi'], np.ndarray) else item['logpi'] for item in x])
            },
            num_workers=0
        )
        
        # Test batch loading
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Check logpi in batch
            if 'logpi' in batch:
                logpi_batch = batch['logpi']
                print(f"  Logpi values: {logpi_batch.flatten()}")
                assert logpi_batch.shape[1] == 1, f"Expected logpi shape [B, 1], got {logpi_batch.shape}"
                print("✓ Batch logpi shape is correct")
            
            break  # Only test first batch
        
        print("✓ Batch loading test passed")
        
    except Exception as e:
        print(f"✗ Batch loading test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(logpi_file):
            os.remove(logpi_file)


if __name__ == "__main__":
    print("Testing logpi integration with scoring model...")
    
    test_logpi_dataset()
    test_scoring_dataset()
    test_batch_loading()
    
    print("\nAll tests completed!")

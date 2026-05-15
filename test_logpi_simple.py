#!/usr/bin/env python
# coding=utf-8

import torch
import yaml
import numpy as np
from data.bson_vla_dataset import BsonVLADataset

def test_dataset_loading():
    """Test basic dataset loading functionality"""
    print("Testing dataset loading...")
    
    # Load dataset
    dataset = BsonVLADataset(
        bson_dir="data/ours/true",
        sub_sample=1.0,
        normalize_mode="min_max",
        stats_file="v5_bson_stats/dataset_statistics.json"
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Try to load a few samples
    for i in range(min(5, len(dataset))):
        try:
            print(f"\nTesting sample {i}...")
            result = dataset.get_item(i)
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                success, data = result
                if not success or data is None:
                    print(f"Sample {i}: Failed to load")
                    continue
            else:
                data = result
                if data is None:
                    print(f"Sample {i}: No data returned")
                    continue
            
            print(f"Sample {i} keys: {list(data.keys())}")
            
            # Check key data types and shapes
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value)}")
                    
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    print("\nDataset loading test completed.")

if __name__ == "__main__":
    test_dataset_loading()

async def main():
    await test_dataset_loading()
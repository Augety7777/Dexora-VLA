#!/usr/bin/env python3
"""
Debug script to identify and fix DataLoader segmentation fault issues
"""

import torch
import torch.multiprocessing as mp
import os
import sys
import traceback
import signal
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data.bson_vla_dataset import BsonVLADataset

class SimpleActionDataset(Dataset):
    """Simplified dataset for debugging"""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Pre-generate all data to avoid any I/O in workers
        print("Pre-generating dataset...")
        self.data = []
        for i in range(num_samples):
            # Generate dummy action chunks
            action_chunk = torch.randn(16, 36)  # [chunk_size, action_dim]
            self.data.append(action_chunk)
        print(f"Generated {len(self.data)} samples")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.data[idx % len(self.data)]

class BsonActionDataset(Dataset):
    """Dataset that uses BsonVLADataset"""
    def __init__(self, bson_dir, num_samples=100):
        print(f"Initializing BsonVLADataset from {bson_dir}...")
        try:
            self.bson_dataset = BsonVLADataset(bson_dir=bson_dir, sub_sample=0.1)
            self.num_samples = min(num_samples, len(self.bson_dataset))
            print(f"BsonVLADataset initialized with {len(self.bson_dataset)} episodes")
            
            # Test getting one sample
            test_sample = self.bson_dataset.get_item()
            self.action_shape = test_sample['actions'].shape
            print(f"Action shape: {self.action_shape}")
            
        except Exception as e:
            print(f"Error initializing BsonVLADataset: {e}")
            traceback.print_exc()
            raise
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        try:
            sample = self.bson_dataset.get_item()
            actions = torch.FloatTensor(sample['actions'])
            return actions
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            # Return dummy data
            return torch.zeros(self.action_shape)

def test_dataloader(dataset, num_workers=0, batch_size=4, test_name="Test"):
    """Test dataloader with given configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {test_name}")
    print(f"Workers: {num_workers}, Batch size: {batch_size}")
    print(f"{'='*60}")
    
    try:
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() and num_workers > 0,
            persistent_workers=num_workers > 0,
            timeout=30 if num_workers > 0 else 0
        )
        
        print(f"DataLoader created successfully")
        
        # Test iteration
        print("Testing iteration...")
        for i, batch in enumerate(loader):
            print(f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 2:  # Test only first few batches
                break
                
        print(f"✅ {test_name} PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {test_name} FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    print("DataLoader Debugging Script")
    print("="*60)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set")
    
    # Parse command line arguments
    bson_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ours"
    print(f"Using BSON directory: {bson_dir}")
    
    # Test 1: Simple dataset with no workers
    print("\n🔍 Test 1: Simple dataset, no workers")
    simple_dataset = SimpleActionDataset(100)
    test_dataloader(simple_dataset, num_workers=0, test_name="Simple dataset (0 workers)")
    
    # Test 2: Simple dataset with workers
    print("\n🔍 Test 2: Simple dataset, with workers")
    test_dataloader(simple_dataset, num_workers=2, test_name="Simple dataset (2 workers)")
    
    # Test 3: BSON dataset with no workers
    print("\n🔍 Test 3: BSON dataset, no workers")
    try:
        bson_dataset = BsonActionDataset(bson_dir, 50)
        test_dataloader(bson_dataset, num_workers=0, test_name="BSON dataset (0 workers)")
        
        # Test 4: BSON dataset with workers (if previous test passed)
        print("\n🔍 Test 4: BSON dataset, with workers")
        test_dataloader(bson_dataset, num_workers=1, test_name="BSON dataset (1 worker)")
        
    except Exception as e:
        print(f"❌ BSON dataset creation failed: {e}")
        traceback.print_exc()
    
    # Test 5: Environment and system info
    print("\n🔍 System Information")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Process ID: {os.getpid()}")
    
    # Test shared memory
    try:
        import psutil
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    except ImportError:
        print("psutil not available for memory info")
    
    print("\n" + "="*60)
    print("Debugging complete!")
    
    # Recommendations
    print("\n📋 Recommendations:")
    print("1. If only simple dataset works: Issue is with BSON file I/O in workers")
    print("2. If no multiprocessing works: Use num_workers=0")
    print("3. If BSON works with 0 workers: Pre-cache data or use persistent_workers=True")
    print("4. Check for memory leaks if crashes happen after some iterations")

if __name__ == "__main__":
    main()
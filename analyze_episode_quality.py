#!/usr/bin/env python3
"""
Script to analyze episode quality metrics from LeRobotDataset.
Calculates jerk and acceleration metrics for episodes and generates filtering threshold plots.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm

# Add lerobot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot', 'src'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from data.bson_vla_dataset import BsonVLADataset


class DataNormalizer:
    """
    Data normalizer following the approach from LeRobot VLA dataset.
    """
    
    def __init__(self, stats_file: Optional[str] = None, normalize_mode: str = "min_max"):
        self.normalize_mode = normalize_mode
        self.stats = None
        if stats_file:
            self._load_statistics(stats_file)
    
    def _load_statistics(self, stats_file: str):
        """Load statistics from JSON file for normalization."""
        try:
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded statistics from {stats_file}")
            print(f"State dim: {len(self.stats['state']['mean'])}, Action dim: {len(self.stats['action']['mean'])}")
        except Exception as e:
            print(f"Warning: Failed to load statistics from {stats_file}: {e}")
            self.stats = None
    
    def normalize_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Normalize data using loaded statistics."""
        if self.stats is None or self.normalize_mode is None:
            return data
            
        if data_type not in self.stats:
            print(f"Warning: No statistics found for {data_type}")
            return data
            
        stats_data = self.stats[data_type]
        
        if self.normalize_mode == 'mean_std':
            # Normalize using mean and standard deviation: (x - mean) / std
            mean = np.array(stats_data['mean'])
            std = np.array(stats_data['std'])
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            normalized = (data - mean) / std
            
        elif self.normalize_mode == 'min_max':
            # Normalize using percentiles as min/max: (x - min) / (max - min)
            min_val = np.array(stats_data['percentile_1'])
            max_val = np.array(stats_data['percentile_99'])
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val
            
        else:
            print(f"Warning: Unknown normalization mode {self.normalize_mode}")
            return data
            
        return normalized


def disable_video_loading(dataset):
    """Disable video loading to speed up data access."""
    object.__setattr__(dataset, 'video_keys', [])
    object.__setattr__(dataset, 'image_transforms', None)
    dataset.meta.info["features"]={}
    object.__setattr__(dataset, 'image_transforms', None)


def calculate_episode_metrics(states: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate jerk and acceleration metrics for an episode.
    
    Args:
        states: Episode states array of shape (T, state_dim)
        
    Returns:
        Dictionary containing jerk and acceleration RMS values per dimension
    """
    if len(states) < 3:
        # Need at least 3 timesteps to calculate jerk
        return {
            'jerk_rms': np.full(states.shape[1], np.inf),
            'acceleration_rms': np.full(states.shape[1], np.inf)
        }
    
    # Calculate velocity (first derivative)
    velocity = np.diff(states, axis=0)
    
    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    
    # Calculate jerk (third derivative)
    jerk = np.diff(acceleration, axis=0)
    
    # Calculate RMS for each dimension
    acceleration_rms = np.sqrt(np.mean(acceleration**2, axis=0))
    jerk_rms = np.sqrt(np.mean(jerk**2, axis=0))
    
    # Check for buggy episodes (all zeros)
    is_buggy = (np.allclose(jerk_rms, 0) and np.allclose(acceleration_rms, 0))
    
    return {
        'jerk_rms': jerk_rms,
        'acceleration_rms': acceleration_rms,
        'is_buggy': is_buggy
    }


def sample_episodes(dataset: LeRobotDataset, num_episodes: int = 100, normalizer: Optional[DataNormalizer] = None) -> List[Tuple[int, np.ndarray]]:
    """
    Sample random episodes from the dataset with optional normalization.
    
    Args:
        dataset: LeRobotDataset instance
        num_episodes: Number of episodes to sample
        normalizer: Optional data normalizer
        
    Returns:
        List of (episode_idx, states) tuples
    """
    if hasattr(dataset, "episode_data_index"):
        total_episodes = len(dataset.episode_data_index["from"])
    else:
        total_episodes = len(dataset)
    episode_indices = random.sample(range(total_episodes), min(num_episodes, total_episodes))
    
    sampled_episodes = []
    
    for ep_idx in tqdm(episode_indices, desc="Sampling episodes"):
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        
        # Extract states for this episode
        episode_states = []
        for i in tqdm(range(ep_start, ep_end), desc=f"Loading episode {ep_idx}", leave=False):
            try:
                sample = dataset[i]
                state = sample['states'].numpy()
                episode_states.append(state)
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue
        
        if len(episode_states) > 2:  # Need at least 3 timesteps
            states_array = np.array(episode_states)
            
            # Apply normalization if provided
            if normalizer is not None:
                states_array = normalizer.normalize_data(states_array, 'state')
            
            sampled_episodes.append((ep_idx, states_array))
    
    return sampled_episodes


def analyze_episodes(episodes: List[Tuple[int, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Analyze all episodes and collect metrics, detecting buggy episodes.
    
    Args:
        episodes: List of (episode_idx, states) tuples
        
    Returns:
        Dictionary containing all metrics and episode info
    """
    all_jerk_rms = []
    all_acceleration_rms = []
    episode_lengths = []
    episode_indices = []
    buggy_episodes = []
    valid_episodes = []
    
    for ep_idx, states in tqdm(episodes, desc="Analyzing episodes"):
        metrics = calculate_episode_metrics(states)
        
        if metrics['is_buggy']:
            buggy_episodes.append(ep_idx)
            print(f"Episode {ep_idx}: BUGGY (all zeros) - Length={len(states)}")
        else:
            all_jerk_rms.append(metrics['jerk_rms'])
            all_acceleration_rms.append(metrics['acceleration_rms'])
            episode_lengths.append(len(states))
            episode_indices.append(ep_idx)
            valid_episodes.append(ep_idx)
            
            print(f"Episode {ep_idx}: Length={len(states)}, "
                  f"Jerk RMS={np.mean(metrics['jerk_rms']):.4f}, "
                  f"Acc RMS={np.mean(metrics['acceleration_rms']):.4f}")
    
    print(f"\nFound {len(buggy_episodes)} buggy episodes: {buggy_episodes}")
    print(f"Valid episodes for analysis: {len(valid_episodes)}")
    
    return {
        'jerk_rms': np.array(all_jerk_rms),
        'acceleration_rms': np.array(all_acceleration_rms),
        'episode_lengths': np.array(episode_lengths),
        'episode_indices': np.array(episode_indices),
        'buggy_episodes': buggy_episodes,
        'valid_episodes': valid_episodes
    }


def calculate_filtering_thresholds(metrics: Dict[str, np.ndarray], target_ratio: float = 0.2) -> Dict[str, float]:
    """
    Calculate filtering thresholds to keep target_ratio of episodes.
    
    Args:
        metrics: Dictionary containing all metrics
        target_ratio: Target ratio of episodes to keep (default 0.2 for 20%)
        
    Returns:
        Dictionary containing jerk and acceleration thresholds
    """
    jerk_rms = metrics['jerk_rms'].squeeze(1)
    acceleration_rms = metrics['acceleration_rms'].squeeze(1)
    
    # For normalized data, we can calculate overall metrics across all dimensions
    # Take the mean across dimensions for each episode
    jerk_overall = np.mean(jerk_rms, axis=1)
    acc_overall = np.mean(acceleration_rms, axis=1)
    
    # Calculate thresholds for target ratio
    jerk_threshold = np.percentile(jerk_overall, target_ratio * 100)
    acc_threshold = np.percentile(acc_overall, target_ratio * 100)
    
    # Find episodes that pass both thresholds
    valid_mask = (jerk_overall <= jerk_threshold) & (acc_overall <= acc_threshold)
    valid_episode_indices = np.array(metrics['episode_indices'])[valid_mask]
    
    print(f"\n=== Filtering Results (Target: {target_ratio*100}%) ===")
    print(f"Jerk threshold: {jerk_threshold:.6f}")
    print(f"Acceleration threshold: {acc_threshold:.6f}")
    print(f"Episodes passing both thresholds: {len(valid_episode_indices)} ({len(valid_episode_indices)/len(metrics['episode_indices'])*100:.1f}%)")
    print(f"Valid episode indices: {valid_episode_indices.tolist()}")
    
    return {
        'jerk_threshold': jerk_threshold,
        'acceleration_threshold': acc_threshold,
        'valid_episodes': valid_episode_indices.tolist(),
        'num_valid': len(valid_episode_indices),
        'total_analyzed': len(metrics['episode_indices']),
        'actual_ratio': len(valid_episode_indices) / len(metrics['episode_indices'])
    }


def create_filtering_plots(metrics: Dict[str, np.ndarray], output_dir: str):
    """
    Create stacked plots showing filtering thresholds vs remaining episode ratios.
    
    Args:
        metrics: Dictionary containing all metrics
        output_dir: Output directory for plots
    """
    jerk_rms = metrics['jerk_rms']
    acceleration_rms = metrics['acceleration_rms']
    
    # Convert to proper 2D array format
    jerk_rms = np.array(jerk_rms)
    acceleration_rms = np.array(acceleration_rms)
    
    # Handle different array shapes
    if jerk_rms.ndim == 3:
        # Shape is (episodes, 1, dimensions) - squeeze out the middle dimension
        jerk_rms = jerk_rms.squeeze(axis=1)
        acceleration_rms = acceleration_rms.squeeze(axis=1)
    elif jerk_rms.ndim == 1:
        # Single dimension case - each episode has one value
        jerk_rms = jerk_rms.reshape(-1, 1)
        acceleration_rms = acceleration_rms.reshape(-1, 1)
    
    num_episodes, state_dim = jerk_rms.shape
    
    # For normalized data, also create overall metrics plots
    jerk_overall = np.mean(jerk_rms, axis=1)
    acc_overall = np.mean(acceleration_rms, axis=1)
    
    # Create threshold ranges
    jerk_thresholds = np.logspace(-3, 1, 50)  # From 0.001 to 10
    acc_thresholds = np.logspace(-3, 1, 50)   # From 0.001 to 10
    
    # Create normalized plots using overall metrics (mean across dimensions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall metrics plots (mean across dimensions) - these are the main plots for normalized data
    overall_jerk_thresholds = np.linspace(np.min(jerk_overall), np.max(jerk_overall), 50)
    overall_acc_thresholds = np.linspace(np.min(acc_overall), np.max(acc_overall), 50)
    
    overall_jerk_ratios = [np.sum(jerk_overall <= t) / len(jerk_overall) for t in overall_jerk_thresholds]
    overall_acc_ratios = [np.sum(acc_overall <= t) / len(acc_overall) for t in overall_acc_thresholds]
    
    # Main normalized plots
    ax1.fill_between(overall_jerk_thresholds, 0, overall_jerk_ratios, alpha=0.6, color='cyan')
    ax1.plot(overall_jerk_thresholds, overall_jerk_ratios, color='darkcyan', linewidth=2)
    ax1.set_xlabel('Overall Jerk RMS Threshold (Normalized)')
    ax1.set_ylabel('Remaining Episode Ratio')
    ax1.set_title('Episode Filtering by Overall Jerk RMS')
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(overall_acc_thresholds, 0, overall_acc_ratios, alpha=0.6, color='orange')
    ax2.plot(overall_acc_thresholds, overall_acc_ratios, color='darkorange', linewidth=2)
    ax2.set_xlabel('Overall Acceleration RMS Threshold (Normalized)')
    ax2.set_ylabel('Remaining Episode Ratio')
    ax2.set_title('Episode Filtering by Overall Acceleration RMS')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_quality_filtering_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save threshold data
    results = {
        'overall_jerk_thresholds': overall_jerk_thresholds.tolist(),
        'overall_acc_thresholds': overall_acc_thresholds.tolist(),
        'overall_jerk_ratios': overall_jerk_ratios,
        'overall_acc_ratios': overall_acc_ratios,
        'jerk_overall_values': jerk_overall.tolist(),
        'acc_overall_values': acc_overall.tolist(),
        'state_dimensions': state_dim,
        'total_episodes_analyzed': num_episodes
    }
    
    with open(os.path.join(output_dir, 'filtering_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Analyze episode quality metrics with normalization')
    parser.add_argument('num_episodes', type=int,
                        help='Number of episodes to sample')
    parser.add_argument('--output_dir', type=str, default='episode_quality_analysis',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for episode sampling')
    parser.add_argument('--stats_file', type=str, default='v4_lerobot_stats/dataset_statistics.json',
                        help='Path to dataset statistics file for normalization')
    parser.add_argument('--normalize_mode', type=str, choices=['mean_std', 'min_max'], default='min_max',
                        help='Normalization mode')
    parser.add_argument('--target_ratio', type=float, default=0.2,
                        help='Target ratio of episodes to keep (default: 0.2 for 20%)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize normalizer
    normalizer = DataNormalizer(stats_file=args.stats_file, normalize_mode=args.normalize_mode)
    
    delta_timestamps = {
        'states': [0],
    }
    # Load dataset
    # dataset = LeRobotDataset("","data/ours/true/output/airbot_dexterous_bimanual_dexterous_manipulation",delta_timestamps=delta_timestamps)
    dataset = LeRobotDataset("","dataprocess/output/airbot_dexterous_bimanual_dexterous_manipulation",delta_timestamps=delta_timestamps)
    disable_video_loading(dataset)
    # dataset = BsonVLADataset()
    print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    # print(f"Total episodes: {len(dataset.episode_data_index['from'])}")
    
    # Sample episodes with normalization
    print(f"Sampling {args.num_episodes} episodes with normalization...")
    episodes = sample_episodes(dataset, args.num_episodes, normalizer)
    print(f"Successfully sampled {len(episodes)} episodes")
    
    if len(episodes) == 0:
        print("No valid episodes found!")
        return
    
    # Analyze episodes
    print("Analyzing episode metrics...")
    metrics = analyze_episodes(episodes)
    
    if len(metrics['valid_episodes']) == 0:
        print("No valid episodes found for analysis!")
        return
    
    # Calculate filtering thresholds
    print("Calculating filtering thresholds...")
    filtering_results = calculate_filtering_thresholds(metrics, args.target_ratio)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total episodes sampled: {len(episodes)}")
    print(f"Valid episodes analyzed: {len(metrics['valid_episodes'])}")
    print(f"Buggy episodes found: {len(metrics['buggy_episodes'])}")
    print(f"State dimensions: {metrics['jerk_rms'].shape[1]}")
    print(f"Average episode length: {np.mean(metrics['episode_lengths']):.1f}")
    print(f"Jerk RMS - Mean: {np.mean(metrics['jerk_rms']):.6f}, Std: {np.std(metrics['jerk_rms']):.6f}")
    print(f"Acceleration RMS - Mean: {np.mean(metrics['acceleration_rms']):.6f}, Std: {np.std(metrics['acceleration_rms']):.6f}")
    
    # Create filtering plots
    print("Creating filtering threshold plots...")
    create_filtering_plots(metrics, args.output_dir)
    
    # Save complete results
    complete_results = {
        'filtering_thresholds': filtering_results,
        'buggy_episodes': metrics['buggy_episodes'],
        'all_valid_episodes': metrics['valid_episodes'],
        'summary_stats': {
            'total_sampled': len(episodes),
            'valid_analyzed': len(metrics['valid_episodes']),
            'buggy_found': len(metrics['buggy_episodes']),
            'state_dimensions': metrics['jerk_rms'].shape[1],
            'avg_episode_length': float(np.mean(metrics['episode_lengths'])),
            'jerk_rms_mean': float(np.mean(metrics['jerk_rms'])),
            'jerk_rms_std': float(np.std(metrics['jerk_rms'])),
            'acc_rms_mean': float(np.mean(metrics['acceleration_rms'])),
            'acc_rms_std': float(np.std(metrics['acceleration_rms']))
        },
        'normalization_settings': {
            'stats_file': args.stats_file,
            'normalize_mode': args.normalize_mode,
            'target_ratio': args.target_ratio
        }
    }
    
    with open(os.path.join(args.output_dir, 'complete_analysis_results.json'), 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Complete results saved to: {args.output_dir}/complete_analysis_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot state curves for buggy episodes identified by episode quality analysis.

This script loads the JSON results from analyze_episode_quality.py and plots
the state trajectories for episodes identified as buggy (all zeros).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add lerobot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot', 'src'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def disable_video_loading(dataset):
    """Disable video loading to speed up data access."""
    object.__setattr__(dataset, 'video_keys', [])
    object.__setattr__(dataset, 'image_transforms', None)


def load_analysis_results(json_file: str) -> Dict:
    """Load the analysis results from JSON file."""
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results


def load_episode_states(dataset: LeRobotDataset, episode_idx: int) -> np.ndarray:
    """Load states for a specific episode."""
    ep_start = dataset.episode_data_index["from"][episode_idx].item()
    ep_end = dataset.episode_data_index["to"][episode_idx].item()
    
    episode_states = []
    for i in tqdm(range(ep_start, ep_end), desc=f"Loading episode {episode_idx}", leave=False):
        try:
            sample = dataset[i]
            state = sample['states'].numpy()
            episode_states.append(state)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    return np.array(episode_states) if episode_states else np.array([])


def plot_individual_episodes(buggy_episodes: List[int], dataset: LeRobotDataset, 
                            output_dir: str, max_episodes: int = 10):
    """Create individual plots for each buggy episode showing all state dimensions."""
    print("Creating individual episode plots...")
    
    # Limit number of episodes to plot
    episodes_to_plot = buggy_episodes[:max_episodes]
    print(f"Plotting first {len(episodes_to_plot)} out of {len(buggy_episodes)} buggy episodes")
    
    for episode_idx in episodes_to_plot:
        states = load_episode_states(dataset, episode_idx)
        
        if len(states) > 0:
            states = np.array(states)
            print(f"Episode {episode_idx}: states shape: {states.shape}")
            
            if states.ndim == 1:
                # If 1D, reshape to (timesteps, 1)
                states = states.reshape(-1, 1)
            elif states.ndim > 2:
                # If more than 2D, flatten the extra dimensions
                states = states.reshape(states.shape[0], -1)
            
            num_timesteps, num_dims = states.shape
            print(f"Episode {episode_idx}: {num_timesteps} timesteps, {num_dims} dimensions")
            
            # Create subplot grid for all dimensions
            cols = 6  # 6 columns
            rows = (num_dims + cols - 1) // cols  # Ceiling division
            
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows))
            axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
            
            # Plot each dimension
            for dim in range(num_dims):
                ax = axes[dim]
                ax.plot(states[:, dim], 'b-', linewidth=1, alpha=0.8)
                ax.set_title(f'Dim {dim}', fontsize=10)
                ax.set_xlabel('Timestep', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                
                # Check if this dimension is all zeros
                if np.all(states[:, dim] == 0):
                    ax.set_facecolor('#ffeeee')  # Light red background
                    ax.text(0.5, 0.5, 'ALL ZEROS', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=8, color='red', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for dim in range(num_dims, len(axes)):
                axes[dim].set_visible(False)
            
            # Overall title
            is_all_zeros = np.all(states == 0)
            title = f'Episode {episode_idx} - All State Dimensions (Length: {num_timesteps})'
            if is_all_zeros:
                title += ' - ALL ZEROS DETECTED'
            fig.suptitle(title, fontsize=14, color='red' if is_all_zeros else 'black')
            
            plt.tight_layout()
            
            # Save plot
            filename = f'episode_{episode_idx}_all_states.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot: {filename}")
        else:
            print(f"Skipping episode {episode_idx} - no states loaded")


def main():
    parser = argparse.ArgumentParser(description='Plot state curves for buggy episodes')
    parser.add_argument('json_file', type=str, 
                        help='Path to JSON file with analysis results')
    parser.add_argument('--output_dir', type=str, default='buggy_episodes_plots',
                        help='Output directory for plots')
    parser.add_argument('--max_dims_per_plot', type=int, default=12,
                        help='Maximum dimensions to show per plot')
    parser.add_argument('--max_episodes', type=int, default=10,
                        help='Maximum number of episodes to plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load analysis results
    print(f"Loading analysis results from {args.json_file}...")
    try:
        results = load_analysis_results(args.json_file)
        buggy_episodes = results.get('buggy_episodes', [])
        
        if not buggy_episodes:
            print("No buggy episodes found in the analysis results!")
            return
            
        print(f"Found {len(buggy_episodes)} buggy episodes: {buggy_episodes}")
        
    except Exception as e:
        print(f"Error loading analysis results: {e}")
        return
    
    # Load dataset
    delta_timestamps = {
        'states': [0],
    }
    
    try:
        dataset = LeRobotDataset("", "data/ours/true/output/airbot_dexterous_bimanual_dexterous_manipulation", 
                               delta_timestamps=delta_timestamps)
        disable_video_loading(dataset)
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
        print(f"Total episodes: {len(dataset.episode_data_index['from'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    dataset.meta.info["features"] = {}
    object.__setattr__(dataset, 'image_transforms', None)
    
    # Plot individual episodes showing all state dimensions
    plot_individual_episodes(buggy_episodes, dataset, args.output_dir, args.max_episodes)
    
    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

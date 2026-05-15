import os
import fnmatch
import yaml
import numpy as np
import bson
import av
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import re
import pathlib
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random

from data.bson_vla_dataset import BsonVLADataset, EpisodeInfo


class BsonVLADatasetFrame(BsonVLADataset):
    """
    Extended BsonVLADataset that supports frame-specific access for logpi calculation.
    Allows getting specific frames from specific episodes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build episode-frame mapping
        self.episode_frame_mapping = self._build_episode_frame_mapping()
        
    def _build_episode_frame_mapping(self):
        """Build mapping from episode index to frame indices"""
        mapping = {}
        for ep_idx, episode_info in enumerate(self.episode_infos):
            try:
                episode_data = self._extract_data_from_episode(episode_info)
                if episode_data is None:
                    continue
                    
                qpos = episode_data["state"]
                num_steps = len(qpos)
                first_idx = self.IMG_HISTORY_SIZE
                
                # Valid frame indices for this episode
                valid_frames = list(range(first_idx - 1, num_steps))
                mapping[ep_idx] = {
                    'episode_info': episode_info,
                    'valid_frames': valid_frames,
                    'num_steps': num_steps
                }
            except Exception as e:
                print(f"Error processing episode {ep_idx}: {e}")
                continue
                
        return mapping
    
    def get_episode_frame(self, episode_idx: int, frame_idx: int):
        """
        Get specific frame from specific episode.
        
        Args:
            episode_idx: Episode index
            frame_idx: Frame index within the episode
            
        Returns:
            Dictionary with same format as __getitem__
        """
        if episode_idx not in self.episode_frame_mapping:
            raise ValueError(f"Episode {episode_idx} not found in mapping")
            
        ep_info = self.episode_frame_mapping[episode_idx]
        if frame_idx not in ep_info['valid_frames']:
            raise ValueError(f"Frame {frame_idx} not valid for episode {episode_idx}")
            
        episode_info = ep_info['episode_info']
        episode_data = self._extract_data_from_episode(episode_info)
        
        if episode_data is None:
            raise ValueError(f"Could not load data for episode {episode_idx}")
            
        return self._parse_episode_frame(episode_info, episode_data, frame_idx, episode_idx)
    
    def _parse_episode_frame(self, episode_info, episode_data, step_id, episode_idx):
        """Parse specific frame from episode data"""
        qpos = episode_data["state"]
        num_steps = len(qpos)
        
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "episode_id": episode_idx,
            "instruction": episode_info.action
        }
        
        actions_full = episode_data["action"]
        target_qpos = actions_full[step_id : step_id + self.CHUNK_SIZE]
        
        # Parse state and action
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = target_qpos

        if actions.shape[0] < self.CHUNK_SIZE:
            actions = np.pad(actions, ((0, self.CHUNK_SIZE - actions.shape[0]), (0, 0)), 'edge')

        state_dim = qpos.shape[1]
        state_indicator = np.ones(state_dim)

        # Parse images using the same logic as parent class
        def parse_img(key):
            img_info = episode_data["images_info"].get(key)
            
            if key == 'head_camera':
                if img_info is None:
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
                
                video_frames = self._get_decoded_video(episode_info, key, img_info)
                
                if video_frames.ndim != 4:
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
                
                start_idx = max(step_id - self.IMG_HISTORY_SIZE + 1, 0)
                imgs = video_frames[start_idx : step_id + 1]
            else:
                if img_info is None:
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
                
                imgs = []
                for i in range(max(step_id - self.IMG_HISTORY_SIZE + 1, 0), step_id + 1):
                    try:
                        img_path = os.path.join(episode_info.path, key, f"frame_{i:06d}.jpg")
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            img_array = np.array(img)
                            imgs.append(img_array)
                        else:
                            imgs.append(np.zeros((480, 640, 3)))
                    except Exception as e:
                        imgs.append(np.zeros((480, 640, 3)))
                
                imgs = np.array(imgs)
            
            # Pad if necessary
            if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                padding_shape = (self.IMG_HISTORY_SIZE - imgs.shape[0],) + imgs.shape[1:]
                padding = np.tile(imgs[0:1], (self.IMG_HISTORY_SIZE - imgs.shape[0], 1, 1, 1))
                imgs = np.concatenate([padding, imgs], axis=0)
            
            return imgs

        # Get camera configuration for this episode
        camera_config = self._validate_camera_config(episode_info.path)
        
        # Parse images based on camera configuration
        if camera_config == (0, 2, 6):  # 3-cam config 1
            cam_high = parse_img('head_camera')
            cam_left_wrist = parse_img('camera_0')
            cam_right_wrist = parse_img('camera_6') 
            cam_third_view = parse_img('camera_2')
        elif camera_config == (2, 4, 6):  # 3-cam config 2  
            cam_high = parse_img('head_camera')
            cam_left_wrist = parse_img('camera_2')
            cam_right_wrist = parse_img('camera_6')
            cam_third_view = parse_img('camera_4')
        elif camera_config == (0, 4, 6, 11):  # 4-cam config
            cam_high = parse_img('camera_0')  # head is external camera
            cam_left_wrist = parse_img('camera_11')
            cam_right_wrist = parse_img('camera_6') 
            cam_third_view = parse_img('camera_4')
        else:
            # Fallback to zeros
            cam_high = np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
            cam_left_wrist = np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
            cam_right_wrist = np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
            cam_third_view = np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))

        # Create masks
        cam_high_mask = np.ones((self.IMG_HISTORY_SIZE,))
        cam_left_wrist_mask = np.ones((self.IMG_HISTORY_SIZE,))
        cam_right_wrist_mask = np.ones((self.IMG_HISTORY_SIZE,))
        cam_third_view_mask = np.ones((self.IMG_HISTORY_SIZE,))

        return {
            'meta': meta,
            'state': state.flatten(),
            'state_std': state_std,
            'state_mean': state_mean,
            'state_norm': state_norm,
            'actions': actions,
            'state_indicator': state_indicator,
            'cam_high': cam_high,
            'cam_high_mask': cam_high_mask,
            'cam_left_wrist': cam_left_wrist,
            'cam_left_wrist_mask': cam_left_wrist_mask,
            'cam_right_wrist': cam_right_wrist,
            'cam_right_wrist_mask': cam_right_wrist_mask,
            'cam_third_view': cam_third_view,
            'cam_third_view_mask': cam_third_view_mask,
        }
    
    def get_all_episode_frame_pairs(self):
        """Get all valid (episode_idx, frame_idx) pairs"""
        pairs = []
        for ep_idx, ep_info in self.episode_frame_mapping.items():
            for frame_idx in ep_info['valid_frames']:
                pairs.append((ep_idx, frame_idx))
        return pairs

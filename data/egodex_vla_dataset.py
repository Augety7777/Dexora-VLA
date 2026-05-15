import os
import fnmatch
import yaml
import numpy as np
import h5py
from typing import List, Dict, Optional
import av

# Attempt to import torch and torchcodec, provide helpful error message if they fail.
try:
    import torch
    import torchcodec
except ImportError:
    print("="*50)
    print("ERROR: PyTorch and torchcodec are required for this dataset.")
    print("Please install them with:")
    print("pip install torch")
    print("pip install torchcodec")
    print("="*50)

class EgoDexVLADataset:
    """
    This class is used to sample episodes from the EgoDex dataset,
    which consists of paired HDF5 pose files and MP4 videos.
    The output format is compatible with HDF5VLADataset.
    It uses torchcodec for efficient video decoding.
    """
    def __init__(self, egodex_dir: str="data/egodex", sub_sample=1.0, cache_all_frames=True) -> None:
        """
        Initializes the EgoDexVLADataset.

        Args:
            egodex_dir (str): The path to the EgoDex dataset directory.
            sub_sample (float): The fraction of the dataset to use.
            cache_all_frames (bool): Whether to cache all frames in memory.
        """
        self.DATASET_NAME = "egodex"
        self.ext_image_names = ["cam_left_wrist", "cam_third_view", "cam_right_wrist"]
        self.L_FINGER_KEYS = ["leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip", "leftRingFingerTip", "leftLittleFingerTip"]
        self.R_FINGER_KEYS = ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip", "rightRingFingerTip", "rightLittleFingerTip"]
        self.POSE_KEYS = ["camera", "leftHand", "rightHand"] + self.L_FINGER_KEYS + self.R_FINGER_KEYS

        self._get_video = self._get_video_frames if cache_all_frames else self._get_video_decoder

        print("Finding EgoDex HDF5/MP4 pairs...")
        self.file_paths = [] # Stores paths to HDF5 files
        for root, _, files in os.walk(egodex_dir):
            for filename in fnmatch.filter(files, '*.hdf5'):
                hdf5_path = os.path.join(root, filename)
                mp4_path = hdf5_path.replace('.hdf5', '.mp4')
                if os.path.exists(mp4_path):
                    self.file_paths.append(hdf5_path)
        print(f"Found {len(self.file_paths)} paired HDF5/MP4 files.")

        # Sub-sampling logic
        indices = np.arange(len(self))
        np.random.seed(42)
        np.random.shuffle(indices)

        superior = sub_sample > 0.5
        if superior:
            sub_sample = 1 - sub_sample
        split = int(len(indices) * sub_sample)
        if superior:
            indices = indices[split:]
        else:
            indices = indices[:split]
        self.file_paths = [self.file_paths[i] for i in indices]
        print(f"Using {len(self.file_paths)} files after subsampling.")

        # Load config from YAML
        try:
            with open('configs/base.yaml', 'r') as file:
                config = yaml.safe_load(file)
            self.CHUNK_SIZE = config['common']['action_chunk_size']
            self.IMG_HISORY_SIZE = config['common']['img_history_size']
        except FileNotFoundError:
            print("Warning: `configs/base.yaml` not found. Using default values.")
            self.CHUNK_SIZE = 20
            self.IMG_HISORY_SIZE = 5

        self._video_cache = {}
        
        print("Pre-calculating episode lengths for sampling...")
        episode_lens = []
        valid_file_paths = []
        for hdf5_path in self.file_paths:
            valid, res = self.parse_egodex_file_state_only(hdf5_path)
            if valid:
                episode_lens.append(res['state'].shape[0])
                valid_file_paths.append(hdf5_path)
            else:
                print(f"Skipping invalid or too short episode: {hdf5_path}")
        
        self.file_paths = valid_file_paths
        if not episode_lens:
             raise ValueError("No valid episodes found in the provided directory.")
             
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
        print("Dataset initialized.")

    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """
        Get a training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]

            parser = self.parse_egodex_file if not state_only else self.parse_egodex_file_state_only
            valid, sample = parser(file_path)
            
            if valid:
                return sample
            else:
                if index is None:
                    print(f"Warning: Invalid sample from {file_path}, resampling...")
                    continue
                else:
                    raise RuntimeError(f"Episode at index {index} ({file_path}) is invalid.")

    def _mat_to_6d_rot(self, rot_mat: np.ndarray) -> np.ndarray:
        """Converts a batch of 3x3 rotation matrices to 6D format.
        ref: On the Continuity of Rotation Representations in Neural Networks"""
        return rot_mat[..., :2, :].reshape(*rot_mat.shape[:-2], 6)

    def _extract_data_from_hdf5(self, hdf5_path: str) -> Optional[Dict]:
        """
        Extracts and processes pose data from a single HDF5 file.
        Returns a dictionary with state/action trajectories and metadata.
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Load all transform data
                transforms = {key: f[f'transforms/{key}'][:] for key in self.POSE_KEYS}
                num_steps = transforms["camera"].shape[0]

                if num_steps == 0:
                    return None
                
                # --- Transform poses to camera frame ---
                T_world_cam = transforms["camera"]
                R_inv = T_world_cam[:, :3, :3].transpose(0, 2, 1) # inv(R) = R.T
                t_inv = -np.einsum('nij,nj->ni', R_inv, T_world_cam[:, :3, 3])
                
                T_cam_world = np.zeros_like(T_world_cam)
                T_cam_world[:, :3, :3] = R_inv
                T_cam_world[:, :3, 3] = t_inv
                T_cam_world[:, 3, 3] = 1

                # Apply transform: T_cam_obj = T_cam_world @ T_world_obj
                poses_in_cam = {}
                for key in self.POSE_KEYS:
                    if key == 'camera': continue
                    T_world_obj = transforms[key]
                    poses_in_cam[key] = np.einsum('nij,njk->nik', T_cam_world, T_world_obj)

                # --- Extract 48D state vector ---
                def extract_features(T_cam_obj):
                    pos = T_cam_obj[:, :3, 3]
                    rot_6d = self._mat_to_6d_rot(T_cam_obj[:, :3, :3])
                    return pos, rot_6d
                
                l_wrist_pos, l_wrist_rot = extract_features(poses_in_cam["leftHand"])
                r_wrist_pos, r_wrist_rot = extract_features(poses_in_cam["rightHand"])
                
                l_fingers_pos = np.stack([extract_features(poses_in_cam[k])[0] for k in self.L_FINGER_KEYS], axis=1)
                r_fingers_pos = np.stack([extract_features(poses_in_cam[k])[0] for k in self.R_FINGER_KEYS], axis=1)

                # Final state vector concatenation: [l_wrist(3+6), l_fingers(15), r_wrist(3+6), r_fingers(15)]
                state_vector = np.concatenate([
                    l_wrist_pos, l_wrist_rot, l_fingers_pos.reshape(num_steps, -1),
                    r_wrist_pos, r_wrist_rot, r_fingers_pos.reshape(num_steps, -1)
                ], axis=1, dtype=np.float32)

                # Get language instruction
                instruction = f.attrs.get('description', f.attrs['llm_description'])

            return {
                "state": state_vector,
                "action": np.copy(state_vector), # State and action are the same pose data
                "instruction": instruction,
                "episode_len": num_steps
            }
        except Exception as e:
            print(f"Error reading or processing HDF5 file {hdf5_path}: {e}")
            return None

    def _get_video_decoder(self, mp4_path: str):
        """Decodes an MP4 video file using torchcodec, with caching."""
        if mp4_path in self._video_cache:
            return self._video_cache[mp4_path]

        try:
            video_decoder = torchcodec.decoders.VideoDecoder(mp4_path)
        except Exception as e:
            print(f"Warning: Failed to decode video with torchcodec {mp4_path}. Error: {e}")
            video_decoder = None
        
        self._video_cache[mp4_path] = video_decoder
        return video_decoder
    
    def _get_video_frames(self, mp4_path):
        if mp4_path in self._video_cache:
            return self._video_cache[mp4_path]

        container = av.open(mp4_path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            img = np.transpose(img, (2, 0, 1))
            frames.append(img)
        container.close()

        video_array = np.stack(frames)
        self._video_cache[mp4_path] = video_array
        return video_array

    def parse_egodex_file(self, hdf5_path: str):
        """
        Parses an HDF5/MP4 pair to generate a training sample at a random timestep.
        """
        episode_data = self._extract_data_from_hdf5(hdf5_path)
        if not episode_data:
            return False, None

        qpos = episode_data["state"]
        num_steps = episode_data["episode_len"]

        if num_steps < 32: # Minimum episode length
            return False, None

        step_id = np.random.randint(0, num_steps)
        
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": episode_data["instruction"]
        }
        
        actions_full = episode_data["action"]
        
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = actions_full[step_id : step_id + self.CHUNK_SIZE]

        if actions.shape[0] < self.CHUNK_SIZE:
            actions = np.pad(actions, ((0, self.CHUNK_SIZE - actions.shape[0]), (0, 0)), 'edge')

        state_indicator = np.ones(qpos.shape[1], dtype=np.float32)

        # Parse images using torchcodec
        mp4_path = hdf5_path.replace('.hdf5', '.mp4')
        video = self._get_video(mp4_path)

        # Get image history
        start_idx = max(step_id - self.IMG_HISORY_SIZE + 1, 0)
        imgs = np.array(video[start_idx : step_id + 1])
        
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            pad_width = self.IMG_HISORY_SIZE - imgs.shape[0]
            imgs = np.pad(imgs, ((pad_width, 0), (0,0), (0,0), (0,0)), 'edge')
        
        imgs = np.transpose(imgs, (0, 2, 3, 1))

        # Create mask for valid historical images
        valid_len = min(step_id + 1, self.IMG_HISORY_SIZE)
        cam_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len, dtype=bool)
        
        # Return a sample compatible with VLA, using cam_high for egocentric view
        sample = {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": imgs,
            "cam_high_mask": cam_mask,
        }

        for cam_name in self.ext_image_names:
            sample[cam_name] = np.zeros_like(imgs)
            sample[cam_name + "_mask"] = np.zeros_like(cam_mask)

        return True, sample

    def parse_egodex_file_state_only(self, hdf5_path: str):
        """
        Parses an HDF5 file to generate full state and action trajectories.
        """
        episode_data = self._extract_data_from_hdf5(hdf5_path)
        if not episode_data or episode_data["episode_len"] < 32:
            return False, None
        
        return True, {
            "state": episode_data["state"],
            "action": episode_data["action"]
        }

if __name__ == "__main__":
    # --- Example Usage ---
    # Check for any HDF5 files to test with
    ds = EgoDexVLADataset()
    
    if len(ds) > 0:
        print(f"\n--- Testing get_item (state_only=False) ---")
        sample = ds.get_item()
        print("Sample keys:", sample.keys())
        print("Meta:", sample['meta'])
        print("State shape:", sample['state'].shape)
        print("Actions shape:", sample['actions'].shape)
        print("State Indicator shape:", sample['state_indicator'].shape)
        assert sample['state'].shape[1] == 48, f"State dimension is not 48! Got {sample['state'].shape[1]}"
        print("Ego-cam ('cam_high') shape:", sample['cam_high'].shape)
        print("Ego-cam mask:", sample['cam_high_mask'])

        print(f"\n--- Testing get_item (state_only=True) ---")
        state_sample = ds.get_item(state_only=True)
        print("State sample keys:", state_sample.keys())
        print("Full state trajectory shape:", state_sample['state'].shape)
        print("Full action trajectory shape:", state_sample['action'].shape)
        print("Test passed successfully!")
    else:
        print("\nDataset initialized but contains no valid episodes.")

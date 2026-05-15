#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RDT model deployment with ZMQ communication to MMK and XHand forwarders.
This script runs in the RDT environment and communicates with the robot forwarders via ZMQ.
"""

import argparse
import os
import sys
import time
import yaml
import base64
from collections import deque
import logging
import json

import numpy as np
import torch
from PIL import Image
import cv2
import zmq
import pyrealsense2 as rs

# Add project root to path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model creation utilities
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

# Global variables for observation buffering
observation_window = None
lang_embeddings = None
device = 'cuda'
dtype = torch.bfloat16

# Normalization globals
normalize_mode = 'min_max'
norm_stats = None
norm_tensors = {}


class ZMQRobotInterface:
    """Interface to communicate with MMK and XHand forwarders via ZMQ"""
    
    def __init__(self, config, mmk_host="localhost", mmk_port=5556, 
                 xhand_host="localhost", xhand_port=5557):
        # Save config
        self.config = config
        
        # ZMQ context
        self.context = zmq.Context()
        
        # MMK connection
        self.mmk_socket = self.context.socket(zmq.REQ)
        self.mmk_socket.connect(f"tcp://{mmk_host}:{mmk_port}")
        logging.info(f"Connected to MMK forwarder at {mmk_host}:{mmk_port}")
        
        # XHand connection
        self.xhand_socket = self.context.socket(zmq.REQ)
        self.xhand_socket.connect(f"tcp://{xhand_host}:{xhand_port}")
        logging.info(f"Connected to XHand forwarder at {xhand_host}:{xhand_port}")
        
        # Set socket timeouts
        self.mmk_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.xhand_socket.setsockopt(zmq.RCVTIMEO, 5000)
        
        # External camera configuration
        self.external_camera_names = ["cam_left_wrist", "cam_third_view", "cam_right_wrist"]
        self.external_camera_ids = config["ext_cam_ids"]
        self.external_cameras = {}
        
        # Initialize external cameras locally
        self._initialize_external_cameras()
    
    def _initialize_external_cameras(self):
        """Initialize external USB cameras locally in the main process"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        self.realsense_pipeline = pipeline
        logging.info("Initialized realsense camera")

        for i, cam_id in enumerate(self.external_camera_ids):
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {cam_id}")
            
            # Configure camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            self.external_cameras[self.external_camera_names[i]] = cap
            logging.info(f"Initialized external camera {self.external_camera_names[i]} with ID {cam_id}")
    
    def _capture_external_camera_images(self):
        """Capture images from external USB cameras locally"""
        images = {}
        
        for cam_name, cap in self.external_cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read from camera {cam_name}")
            images[cam_name] = frame
        
        # Add dummy images for any missing cameras
        for camera_name in self.external_camera_names:
            if camera_name not in images:
                images[camera_name] = np.zeros((480, 640, 3), dtype=np.uint8)
                logging.warning(f"Camera {camera_name} not found, using dummy image")
        
        return images
    
    def get_mmk_observations(self):
        """Get observations from MMK robot (qpos and head camera only)"""
        request = {'command': 'get_observations'}
        self.mmk_socket.send_json(request)
        
        try:
            response = self.mmk_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"MMK error: {response['error']}")
            
            # Decode head camera image from base64
            # head_camera_data = response['head_camera_image']
            # img_bytes = base64.b64decode(head_camera_data)
            # img_array = np.frombuffer(img_bytes, np.uint8)
            # head_camera_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            frames = self.realsense_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                head_camera_img = np.asanyarray(color_frame.get_data())
            else:
                head_camera_img = None
            
            # Get external camera images locally
            external_images = self._capture_external_camera_images()
            
            # Combine all images
            images = external_images
            images['head_camera'] = head_camera_img
            
            return {
                'qpos': np.array(response['qpos']),
                'images': images
            }
        except zmq.error.Again:
            raise RuntimeError("MMK forwarder timeout")
    
    def get_xhand_observations(self):
        """Get observations from XHand"""
        request = {'command': 'get_observations'}
        self.xhand_socket.send_json(request)
        
        try:
            response = self.xhand_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"XHand error: {response['error']}")
            
            return {
                'left_hand': response['left_hand'],
                'right_hand': response['right_hand']
            }
        except zmq.error.Again:
            raise RuntimeError("XHand forwarder timeout")
    
    def execute_mmk_action(self, action):
        """Execute action on MMK robot"""
        request = {
            'command': 'execute_action',
            'action': action.tolist() if isinstance(action, np.ndarray) else action
        }
        self.mmk_socket.send_json(request)
        
        try:
            response = self.mmk_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"MMK error: {response['error']}")
            return response
        except zmq.error.Again:
            raise RuntimeError("MMK forwarder timeout")
    
    def execute_xhand_action(self, left_hand_action, right_hand_action):
        """Execute action on XHand"""
        request = {
            'command': 'execute_action',
            'action_data': {
                'left_hand': left_hand_action.tolist() if isinstance(left_hand_action, np.ndarray) else left_hand_action,
                'right_hand': right_hand_action.tolist() if isinstance(right_hand_action, np.ndarray) else right_hand_action
            }
        }
        self.xhand_socket.send_json(request)
        
        try:
            response = self.xhand_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"XHand error: {response['error']}")
            return response
        except zmq.error.Again:
            raise RuntimeError("XHand forwarder timeout")
    
    def reset_mmk(self):
        """Reset MMK robot"""
        request = {'command': 'reset'}
        self.mmk_socket.send_json(request)
        
        try:
            response = self.mmk_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"MMK error: {response['error']}")
            return response
        except zmq.error.Again:
            raise RuntimeError("MMK forwarder timeout")
    
    def close(self):
        """Close ZMQ connections and release resources"""
        # Release external cameras
        for _, cap in self.external_cameras.items():
            if cap is not None:
                cap.release()
        
        # Close ZMQ sockets
        self.mmk_socket.close()
        self.xhand_socket.close()
        self.context.term()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_normalization(stats_file: str, mode: str = 'min_max'):
    """Load normalization statistics and prepare tensors on device.

    Args:
        stats_file: Path to dataset_statistics.json
        mode: 'none' | 'mean_std' | 'min_max'
    """
    global normalize_mode, norm_stats, norm_tensors, device, dtype
    normalize_mode = mode
    norm_tensors = {}

    if mode == 'none':
        logging.info("Normalization disabled (mode=none)")
        return

    try:
        with open(stats_file, 'r') as f:
            norm_stats = json.load(f)
        logging.info(f"Loaded normalization stats from {stats_file}")
    except Exception as e:
        logging.warning(f"Failed to load stats from {stats_file}: {e}. Falling back to no normalization.")
        normalize_mode = 'none'
        return

    # Prepare per-type tensors on device
    for key in ['state', 'action']:
        data = norm_stats.get(key, None)
        if data is None:
            continue
        if mode == 'mean_std':
            mean = torch.tensor(data['mean'], device=device, dtype=dtype)
            std = torch.tensor(data['std'], device=device, dtype=dtype)
            std = torch.where(std == 0, torch.ones_like(std), std)
            norm_tensors[key] = {'mean': mean, 'std': std}
        elif mode == 'min_max':
            min_val = torch.tensor(data['percentile_1'], device=device, dtype=dtype)
            max_val = torch.tensor(data['percentile_99'], device=device, dtype=dtype)
            range_val = max_val - min_val
            range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
            norm_tensors[key] = {'min': min_val, 'max': max_val, 'range': range_val}


def normalize_vector(x: torch.Tensor, key: str) -> torch.Tensor:
    """Normalize a vector/tensor along the last dimension using loaded stats for given key.

    x: torch tensor with last dim == dim size (e.g., [D], [B,D], [T,D], ...)
    key: 'state' or 'action'
    """
    if normalize_mode == 'none' or key not in norm_tensors:
        return x
    stats = norm_tensors[key]
    if normalize_mode == 'mean_std':
        return (x - stats['mean']) / stats['std']
    elif normalize_mode == 'min_max':
        return (x - stats['min']) / stats['range']
    
    return x


def denormalize_vector(x: torch.Tensor, key: str) -> torch.Tensor:
    """Inverse of normalize_vector for the given key."""
    if normalize_mode == 'none' or key not in norm_tensors:
        return x
    stats = norm_tensors[key]
    if normalize_mode == 'mean_std':
        return1 = x * stats['std'] + stats['mean']
        return torch.clip(return1, -2.0, 2.0)
    elif normalize_mode == 'min_max':
        return x * stats['range'] + stats['min']
    return x


def create_model(config, pretrained):
    """Initialize the RDT model directly from pretrained checkpoint"""
    global device, dtype
    logging.info("Creating RDT model...")
    
    # Create the vision encoder (Siglip)
    vision_encoder = SiglipVisionTower(vision_tower=config["vision_encoder_name"], args=None)
    image_processor = vision_encoder.image_processor
    
    # Directly load model from pretrained checkpoint
    logging.info(f"Loading pretrained model from {pretrained}")
    runner = RDTRunner.from_pretrained(pretrained)
    
    # Setup device
    runner = runner.to(device, dtype=dtype)
    vision_encoder = vision_encoder.to(device, dtype=dtype)
    
    # Set to evaluation mode
    runner.eval()
    vision_encoder.eval()
    
    return runner, vision_encoder, image_processor


def get_observations(robot_interface: ZMQRobotInterface):
    """Get observations from robot system via ZMQ"""
    # Get MMK observations
    mmk_obs = robot_interface.get_mmk_observations()
    arm_qpos = mmk_obs['qpos']
    images = mmk_obs['images']
    
    # Get XHand observations
    xhand_obs = robot_interface.get_xhand_observations()
    
    # Extract arm positions (first 6 dimensions of each arm)
    arm_left_pos = arm_qpos[:6]
    arm_right_pos = arm_qpos[6:12]
    
    return {
        'arm_left': arm_left_pos,
        'arm_right': arm_right_pos,
        'hand_left': xhand_obs['left_hand'],
        'hand_right': xhand_obs['right_hand'],
        'images': images
    }


def update_observation_window(config, robot_interface):
    """Update observation window with latest data"""
    global observation_window
    
    # Initialize observation window if not already created
    if observation_window is None:
        img_history_size = config.get("img_history_size", 2)
        observation_window = deque(maxlen=img_history_size)
        
        # Get first observation
        first_obs = get_observations(robot_interface)
        
        # Process images using JPEG transformation
        def jpeg_mapping(img):
            img = cv2.imencode('.jpg', img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img
        
        first_images = {}
        for camera_name in config["camera_names"]:
            img = first_obs['images'][camera_name]
            first_images[camera_name] = jpeg_mapping(img)
        
        # Combine arm and hand positions
        first_qpos = np.concatenate((
            np.array(first_obs['arm_left']),
            np.array(first_obs['hand_left']),
            np.array(first_obs['arm_right']),
            np.array(first_obs['hand_right'])
        ), axis=0)
        
        first_qpos = torch.from_numpy(first_qpos).float().cuda()
        
        # Convert images to tensors
        first_processed_images = {}
        for camera_name, img in first_images.items():
            first_processed_images[camera_name] = torch.from_numpy(img).float().cuda() / 255.0
        
        # Add the first observation twice to initialize
        first_observation = {
            'qpos': first_qpos,
            'images': first_processed_images
        }
        
        observation_window.append(first_observation)
        observation_window.append(first_observation.copy())
    
    # Get current observations
    obs = get_observations(robot_interface)
    
    # Process images
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    images = {}
    for camera_name in config["camera_names"]:
        img = obs['images'].get(camera_name, np.zeros((480, 640, 3), dtype=np.uint8))
        images[camera_name] = jpeg_mapping(img)
    
    # Combine positions
    qpos = np.concatenate((
        np.array(obs['arm_left']),
        np.array(obs['hand_left']),
        np.array(obs['arm_right']),
        np.array(obs['hand_right'])
    ), axis=0)
    
    qpos = torch.from_numpy(qpos).float().cuda()
    
    # Process images
    processed_images = {}
    for camera_name, img in images.items():
        processed_images[camera_name] = img
    
    observation_window.append({
        'qpos': qpos,
        'images': processed_images
    })


def inference_fn(config, policy, vision_encoder, image_processor):
    """Run model inference with current observations"""
    global observation_window, lang_embeddings, device, dtype
    
    # Fetch images from observation window
    image_arrs = []
    for camera_name in config["camera_names"]:
        for i in range(len(observation_window)):
            img = observation_window[i]['images'][camera_name]
            if img is None:
                logging.warning(f"Image {camera_name} is None, will use background image")
            image_arrs.append(img)
    
    # Background image for padding
    background_color = np.array([
        int(x*255) for x in image_processor.image_mean
    ], dtype=np.uint8).reshape(1, 1, 3)
    background_image = np.ones((
        image_processor.size["height"], 
        image_processor.size["width"], 3), dtype=np.uint8
    ) * background_color
    
    # Preprocess images
    image_tensor_list = []
    for image in image_arrs:
        if image is None:
            image = Image.fromarray(background_image)
        else:
            image = Image.fromarray(image)
        
        if config.get("image_aspect_ratio", "pad") == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_list.append(image)
    
    image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
    
    # Process with vision encoder
    image_embeds = vision_encoder(image_tensor).detach()
    image_embeds = image_embeds.reshape(-1, vision_encoder.hidden_size).unsqueeze(0)
    
    # Get proprioception (normalize to match training)
    proprio = observation_window[-1]['qpos'].to(device, dtype=dtype)
    proprio = normalize_vector(proprio, 'state')
    proprio = proprio.unsqueeze(0).unsqueeze(0)
    
    # Setup action mask
    action_mask = torch.ones(1, 1, config['state_dim'], device=device, dtype=dtype)
    
    # Setup control frequency
    ctrl_freq = torch.tensor([config['control_frequency']], device=device)
    
    # Run inference
    with torch.no_grad():
        actions = policy.predict_action(
            lang_tokens=lang_embeddings,
            lang_attn_mask=torch.ones(
                lang_embeddings.shape[:2], dtype=torch.bool,
                device=lang_embeddings.device),
            img_tokens=image_embeds,
            state_tokens=proprio,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freq
        )
        # Denormalize actions back to robot space
        actions = denormalize_vector(actions, 'action')
    
    return actions.squeeze(0).to(torch.float).cpu().numpy()


def interpolate_action(config, prev_action, cur_action):
    """Interpolate between previous and current action for smooth movement"""
    steps = np.concatenate((
        np.array(config['arm_steps_length']),
        np.array(config['hand_steps_length']),
        np.array(config['arm_steps_length']),
        np.array(config['hand_steps_length'])
    ), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    
    if step <= 1:
        return cur_action[np.newaxis, :]
    
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    logging.info(f"Interpolate action with {step} steps")
    return new_actions[1:]


def execute_actions(robot_interface, actions):
    """Execute actions on the robot via ZMQ"""
    # Split actions
    left_arm_action = actions[:6]
    left_hand_action = actions[6:18]
    right_arm_action = actions[18:24]
    right_hand_action = actions[24:36]
    
    # Execute hand actions
    logging.debug(f"Executing xHand action: L={left_hand_action}, R={right_hand_action}")
    robot_interface.execute_xhand_action(left_hand_action, right_hand_action)
    
    # Execute arm actions
    logging.debug(f"Executing MMK action: L={left_arm_action}, R={right_arm_action}")
    robot_interface.execute_mmk_action(np.concatenate([left_arm_action, right_arm_action]))


def model_inference_loop(config, robot_interface, policy, vision_encoder, image_processor):
    """Main inference loop for real-time control"""
    robot_interface.reset_mmk()
    
    # Initialize previous action
    update_observation_window(config, robot_interface)
    prev_action = observation_window[-1]['qpos'].cpu().numpy()
    # hand degree to rad
    prev_action[6:18] = np.deg2rad(prev_action[6:18])
    prev_action[24:36] = np.deg2rad(prev_action[24:36])
    
    # Inference loop
    t = 0
    chunk_size = config['chunk_size']
    max_steps = config['max_steps']
    
    action_buffer = np.zeros([chunk_size, config['state_dim']])
    
    logging.info("Starting inference loop...")
    while t < max_steps:
        loop_start_time = time.time()
        
        # Update observation window
        update_observation_window(config, robot_interface)
        
        # Run inference when needed
        if t % chunk_size == 0:
            for _ in range(5):
                update_observation_window(config, robot_interface)
            inference_start = time.time()
            action_buffer = inference_fn(
                config, policy, vision_encoder, image_processor
            ).copy()
            logging.info(f"Inference and get action {action_buffer.shape}")
            logging.debug(f"Inference time: {time.time() - inference_start:.4f}s")
        
        # Get current action
        raw_action = action_buffer[t % chunk_size]
        action = raw_action
        
        # Interpolate if needed
        if config['use_actions_interpolation']:
            interp_actions = interpolate_action(config, prev_action, action)
        else:
            interp_actions = action[np.newaxis, :]
        
        # Execute actions
        for act in interp_actions:
            execute_actions(robot_interface, act)
            time.sleep(1.0 / config['control_frequency'])
        
        # Update previous action
        prev_action = action.copy()
        
        # Increment time
        t += 1
        
        # Log status
        loop_time = time.time() - loop_start_time
        logging.debug(f"Step {t}/{max_steps}, loop time: {loop_time:.4f}s, FPS: {1/loop_time:.2f}")


def prepare_language_embeddings(args):
    """Prepare language embeddings for the model"""
    global lang_embeddings, device, dtype
    
    logging.info(f"Loading language embeddings from {args.lang_embeddings_path}")
    lang_dict = torch.load(args.lang_embeddings_path)
    logging.info(f"Using instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"].to(device, dtype=dtype)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RDT Model Deployment with ZMQ")
    
    # Model configuration
    parser.add_argument("--config-path", type=str, default="deploy/mmk_xhand_config.yaml", 
                        help="Path to model config YAML")
    parser.add_argument("--pretrained-model-path", type=str, required=True,
                        help="Path to pretrained model")
    
    # Language configuration
    parser.add_argument("--lang-embeddings-path", type=str, default=None,
                        help="Path to pre-computed language embeddings")
    
    # Normalization configuration
    parser.add_argument("--normalize-mode", type=str, default="min_max", choices=["none", "mean_std", "min_max"],
                        help="Normalization mode for qpos/action, should match training")
    parser.add_argument("--stats-file", type=str, default="bson_stats/dataset_statistics.json",
                        help="Path to dataset statistics JSON used for normalization")
    
    # ZMQ configuration
    parser.add_argument("--mmk-host", type=str, default="localhost",
                        help="MMK forwarder host")
    parser.add_argument("--mmk-port", type=int, default=5556,
                        help="MMK forwarder port")
    parser.add_argument("--xhand-host", type=str, default="localhost",
                        help="XHand forwarder host")
    parser.add_argument("--xhand-port", type=int, default=5557,
                        help="XHand forwarder port")
    
    args = parser.parse_args()
    return args


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    args = get_args()
    
    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config["seed"])
    
    # Load normalization (must match training dataset)
    load_normalization(args.stats_file, args.normalize_mode)
    
    # Create RDT model
    logging.info("Creating RDT model...")
    policy, vision_encoder, image_processor = create_model(
        config=config,
        pretrained=args.pretrained_model_path,
    )
    
    prepare_language_embeddings(args)
    
    # Initialize robot interface
    logging.info("Initializing ZMQ robot interface...")
    robot_interface = ZMQRobotInterface(
        config=config,
        mmk_host=args.mmk_host,
        mmk_port=args.mmk_port,
        xhand_host=args.xhand_host,
        xhand_port=args.xhand_port
    )
    
    # Test connections
    try:
        logging.info("Testing connections...")
        _ = robot_interface.get_mmk_observations()
        _ = robot_interface.get_xhand_observations()
        logging.info("All connections successful!")
    except Exception as e:
        logging.error(f"Connection test failed: {e}")
        robot_interface.close()
        return
    
    # Run inference
    logging.info("Starting RDT inference loop...")
    try:
        with torch.inference_mode():
            while input("Press enter to continue, input anything to exit") == "":
                model_inference_loop(
                    config, robot_interface,
                    policy, vision_encoder, image_processor
                )
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down...")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.info("Closing connections...")
        robot_interface.close()
        logging.info("Inference ended")


if __name__ == "__main__":
    main()
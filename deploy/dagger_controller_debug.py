#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RDT model deployment with Human-in-the-Loop intervention support (DAgger).
This script runs in the RDT environment and communicates with the robot forwarders via ZMQ.

Features:
- Real-time policy inference
- Human intervention mode with keyboard control
- Automatic data collection for DAgger training
- Smooth mode switching between policy and human control

Usage:
    python deploy/dagger_controller_debug.py \
        --pretrained-model-path v1-20k190/checkpoint-20000 \
        --lang-embeddings-path outs/action190.pt \
        --normalize-mode mean_std \
        --stats-file 20k190v1_bson_stats/1113action190/dataset_statistics.json

Keyboard Controls:
    Mode Control:
        - SPACE: Toggle between policy mode and human intervention mode
        - S: Save collected intervention data
        - Q: Exit program
    
    Teleoperation (in human mode):
        - 1/2/3: Select left/right/both arms
        - W/X/A/D: Move forward/back/left/right
        - R/F: Move up/down
        - I/K: Roll rotation （+/-）
        - J/L: Pitch rotation （+/-）
        - U/O: Yaw rotation （+/-）
        
    Hand Control (in human mode, v2 velocity control 2025-11-19):
        Finger Selection (multi-select, toggle on/off):
        - T: Toggle thumb 👍
        - Y: Toggle index finger ☝️
        - H: Toggle middle finger 🖕
        - N: Toggle ring finger 💍
        - M: Toggle pinky 🤙
        
        Control Selected Fingers (hold to move, release to stop):
        - Z: Hold to close selected fingers (~1.15°/cycle, smooth)
        - C: Hold to open selected fingers (~1.15°/cycle, smooth)
"""
import logging

# 立即配置logging（在任何logging调用之前）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Python 3.8+ 强制重新配置
)

import json
from pathlib import Path
import argparse
import os
import sys
import time
import yaml
import base64
from collections import deque
# import logging
import json
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image
import cv2
import zmq
import pyrealsense2 as rs
from pynput import keyboard

# Add project root to path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add mmk2_kdl_py to path
mmk2_kdl_path = os.path.join(os.path.dirname(__file__), 'mmk2_kdl_py-0.1.4', 'src')
sys.path.append(mmk2_kdl_path)

# Import model creation utilities
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

# Import MMK2 kinematics library
try:
    from mmk2_kdl_py import MMK2Kdl
    MMK2_KDL_AVAILABLE = True
    logging.info("MMK2 KDL library loaded successfully")
except ImportError as e:
    MMK2_KDL_AVAILABLE = False
    logging.warning(f"MMK2 KDL library not available: {e}. Using simplified IK/FK.")

# Global variables for observation buffering
observation_window = None
lang_embeddings = None
device = 'cuda'
dtype = torch.bfloat16

# Normalization globals
normalize_mode = 'min_max'
norm_stats = None
norm_tensors = {}

# Human-in-the-Loop control globals
control_mode = "policy"  # "policy" | "human"
mode_lock = Lock()
should_exit = False
intervention_data = []
current_episode_data = []

# Teleoperation control globals for wrist control
teleop_target_pose = {
    'left': {'pos': np.array([0.3, 0.3, 0.3]), 'rot': np.array([0., 0., 0.])},  # xyz, rpy
    'right': {'pos': np.array([0.3, -0.3, 0.3]), 'rot': np.array([0., 0., 0.])}
}
# 新增：速度控制变量（方案2）
teleop_velocity = {
    'left': {'pos': np.zeros(3), 'rot': np.zeros(3)},
    'right': {'pos': np.zeros(3), 'rot': np.zeros(3)}
}
active_arm = 'left'  # 'left' | 'right' | 'both'
teleop_step_size = 0.01  # 1cm per control cycle
teleop_rot_step = 0.05   # ~3 degrees per control cycle
teleop_hand_step = 5.0   # 5 degrees per keypress for hand joints

# 手部控制全局变量（新增 - 2025年实现，v2重新设计为手指选择模式）
teleop_hand_joints = {
    'left': np.zeros(12),   # 12个手部关节角度（弧度）
    'right': np.zeros(12)
}

# XHand 12关节到手指的映射（基于xhand_forwarder.py中的JOINT_LIMITS_RAD）
# joint 0: thumb_bend_joint [0, 105°]
# joint 1: thumb_rota_joint1 [-40°, 90°]
# joint 2: thumb_rota_joint2 [0, 90°]
# joint 3: index_bend_joint [-10°, 10°]
# joint 4-5: index_joint1-2 [0, 110°]
# joint 6-7: middle_joint1-2 [0, 110°]
# joint 8-9: ring_joint1-2 [0, 110°]
# joint 10-11: pinky_joint1-2 [0, 110°]
FINGER_JOINT_MAP = {
    'thumb': [0, 1, 2],      # 拇指: bend + 2个旋转关节
    'index': [3, 4, 5],      # 食指: 1个bend + 2个关节
    'middle': [6, 7],        # 中指: 2个关节
    'ring': [8, 9],          # 无名指: 2个关节
    'pinky': [10, 11]        # 小指: 2个关节
}

# 当前选中要控制的手指（可多选）
# 初始为空，让用户切换到人类模式后明确选择
selected_fingers = {
    'left': [],   # 空列表，用户手动按T/Y/H/N/M选择
    'right': []
}

teleop_hand_step = 2.0   # 每次按键2度（减小步长，更精细）

# 手指速度控制（方案B：速度模式）
teleop_hand_velocity = {
    'left': np.zeros(12),   # 12个关节的速度（弧度/周期）
    'right': np.zeros(12)
}
teleop_hand_speed = 0.02   # 约1.15度/周期（流畅的速度控制）

# MMK2 KDL instance (initialized in main)
mmk2_kdl = None


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


def setup_keyboard_listener():
    """设置键盘监听器用于模式切换和遥操作控制（速度控制 + 手指选择控制）"""
    global control_mode, should_exit, active_arm, teleop_velocity, teleop_hand_joints, selected_fingers, teleop_hand_velocity
    
    def on_press(key):
        global control_mode, should_exit, active_arm, teleop_velocity, teleop_hand_joints, selected_fingers, teleop_hand_velocity
        try:
            # ===== 模式控制 =====
            if key == keyboard.Key.space:
                # 切换模式
                with mode_lock:
                    if control_mode == "policy":
                        control_mode = "human"
                        arm_symbol = {"left": "👈", "right": "👉", "both": "👐"}[active_arm]
                        logging.info("=" * 70)
                        logging.info("🔴 切换到人类控制模式 - 键盘遥操作")
                        logging.info(f"  当前激活手臂: {arm_symbol} {active_arm.upper()}")
                        logging.info("  1=左臂 | 2=右臂 | 3=双臂")
                        logging.info("  按住WASDXRF=移动 | 按住IJKL/UO=旋转 | 松开=停止")
                        logging.info("  T/Y/H/N/M=选择手指 | Z/C=闭合/张开选中手指")
                        show_selected_fingers()
                        logging.info("=" * 70)
                    else:
                        control_mode = "policy"
                        # ✅ 清除手指选择状态，避免下次进入人类模式时保留历史
                        selected_fingers['left'] = []
                        selected_fingers['right'] = []
                        logging.info("🟢 切换回策略控制模式")
                        logging.info("  已清除手指选择状态")
                        
            elif hasattr(key, 'char') and key.char == 's':
                # 保存当前episode的干预数据
                save_intervention_data()
                logging.info("💾 干预数据已保存")
                
            elif hasattr(key, 'char') and key.char == 'q':
                # 退出程序
                logging.info("⛔ 用户请求退出程序（按Q键）")
                should_exit = True
            
            # ===== 手臂选择 (仅在人类模式下有效) =====
            elif control_mode == "human" and hasattr(key, 'char'):
                if key.char == '1':
                    active_arm = 'left'
                    logging.info(f"👈 已切换到左臂控制")
                    logging.info(f"   目标位置: [{teleop_target_pose['left']['pos'][0]:.3f}, {teleop_target_pose['left']['pos'][1]:.3f}, {teleop_target_pose['left']['pos'][2]:.3f}]")
                elif key.char == '2':
                    active_arm = 'right'
                    logging.info(f"👉 已切换到右臂控制")
                    logging.info(f"   目标位置: [{teleop_target_pose['right']['pos'][0]:.3f}, {teleop_target_pose['right']['pos'][1]:.3f}, {teleop_target_pose['right']['pos'][2]:.3f}]")
                elif key.char == '3':
                    active_arm = 'both'
                    logging.info(f"👐 已切换到双臂同步控制")
                
                # ===== 位置控制 (WASD + R/F) - 设置速度而非累积位置 =====
                elif key.char == 'w':  # +X (前)
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][0] = teleop_step_size
                    logging.info(f"➡️ 前进中... ({active_arm})")
                elif key.char == 'x':  # -X (后) 
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][0] = -teleop_step_size
                    logging.info(f"⬅️ 后退中... ({active_arm})")
                elif key.char == 'a':  # +Y (左)
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][1] = teleop_step_size
                    logging.info(f"⬅️ 左移中... ({active_arm})")
                elif key.char == 'd':  # -Y (右)
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][1] = -teleop_step_size
                    logging.info(f"➡️ 右移中... ({active_arm})")
                elif key.char == 'r':  # +Z (上)
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][2] = teleop_step_size
                    logging.info(f"⬆️ 上升中... ({active_arm})")
                elif key.char == 'f':  # -Z (下)
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][2] = -teleop_step_size
                    logging.info(f"⬇️ 下降中... ({active_arm})")
                
                # ===== 旋转控制 (IJKL + U/O) - 设置旋转速度 =====
                elif key.char == 'i':  # +Roll
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][0] = teleop_rot_step
                    logging.info(f"🔄 Roll+ 旋转中... ({active_arm})")
                elif key.char == 'k':  # -Roll
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][0] = -teleop_rot_step
                    logging.info(f"🔄 Roll- 旋转中... ({active_arm})")
                elif key.char == 'j':  # +Pitch
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][1] = teleop_rot_step
                    logging.info(f"🔄 Pitch+ 旋转中... ({active_arm})")
                elif key.char == 'l':  # -Pitch
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][1] = -teleop_rot_step
                    logging.info(f"🔄 Pitch- 旋转中... ({active_arm})")
                elif key.char == 'u':  # +Yaw
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][2] = teleop_rot_step
                    logging.info(f"🔄 Yaw+ 旋转中... ({active_arm})")
                elif key.char == 'o':  # -Yaw
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][2] = -teleop_rot_step
                    logging.info(f"🔄 Yaw- 旋转中... ({active_arm})")
                
                # ===== 手指选择控制 (T/Y/H/N/M) - v2设计 =====
                elif key.char == 't':  # Toggle 拇指
                    for arm in get_active_arms():
                        if 'thumb' in selected_fingers[arm]:
                            selected_fingers[arm].remove('thumb')
                            logging.info(f"❌ 取消拇指 ({arm})")
                        else:
                            selected_fingers[arm].append('thumb')
                            logging.info(f"✅ 选择拇指 ({arm})")
                    show_selected_fingers()
                
                elif key.char == 'y':  # Toggle 食指
                    for arm in get_active_arms():
                        if 'index' in selected_fingers[arm]:
                            selected_fingers[arm].remove('index')
                            logging.info(f"❌ 取消食指 ({arm})")
                        else:
                            selected_fingers[arm].append('index')
                            logging.info(f"✅ 选择食指 ({arm})")
                    show_selected_fingers()
                
                elif key.char == 'h':  # Toggle 中指
                    for arm in get_active_arms():
                        if 'middle' in selected_fingers[arm]:
                            selected_fingers[arm].remove('middle')
                            logging.info(f"❌ 取消中指 ({arm})")
                        else:
                            selected_fingers[arm].append('middle')
                            logging.info(f"✅ 选择中指 ({arm})")
                    show_selected_fingers()
                
                elif key.char == 'n':  # Toggle 无名指
                    for arm in get_active_arms():
                        if 'ring' in selected_fingers[arm]:
                            selected_fingers[arm].remove('ring')
                            logging.info(f"❌ 取消无名指 ({arm})")
                        else:
                            selected_fingers[arm].append('ring')
                            logging.info(f"✅ 选择无名指 ({arm})")
                    show_selected_fingers()
                
                elif key.char == 'm':  # Toggle 小指
                    for arm in get_active_arms():
                        if 'pinky' in selected_fingers[arm]:
                            selected_fingers[arm].remove('pinky')
                            logging.info(f"❌ 取消小指 ({arm})")
                        else:
                            selected_fingers[arm].append('pinky')
                            logging.info(f"✅ 选择小指 ({arm})")
                    show_selected_fingers()
                
                # ===== 控制选中的手指 (Z/C) - v2速度控制模式 =====
                elif key.char == 'z':  # 按住闭合选中的手指
                    for arm in get_active_arms():
                        if not selected_fingers[arm]:
                            logging.warning(f"⚠️ 没有选中任何手指 ({arm})，请先按T/Y/H/N/M选择")
                            continue
                        for finger in selected_fingers[arm]:
                            joint_indices = FINGER_JOINT_MAP[finger]
                            for idx in joint_indices:
                                teleop_hand_velocity[arm][idx] = teleop_hand_speed
                    fingers_str = ', '.join(selected_fingers[active_arm if active_arm != 'both' else 'left'])
                    logging.info(f"✊ 闭合中: {fingers_str} ({active_arm}) - 按住持续，松开停止")
                
                elif key.char == 'c':  # 按住张开选中的手指
                    for arm in get_active_arms():
                        if not selected_fingers[arm]:
                            logging.warning(f"⚠️ 没有选中任何手指 ({arm})，请先按T/Y/H/N/M选择")
                            continue
                        for finger in selected_fingers[arm]:
                            joint_indices = FINGER_JOINT_MAP[finger]
                            for idx in joint_indices:
                                teleop_hand_velocity[arm][idx] = -teleop_hand_speed
                    fingers_str = ', '.join(selected_fingers[active_arm if active_arm != 'both' else 'left'])
                    logging.info(f"🖐️  张开中: {fingers_str} ({active_arm}) - 按住持续，松开停止")
                
        except AttributeError:
            pass
    
    def on_release(key):
        """松开键时清零对应方向的速度"""
        global teleop_velocity, teleop_hand_velocity, active_arm
        try:
            if control_mode == "human" and hasattr(key, 'char'):
                # 位置控制键松开时清零对应轴的速度
                if key.char in ['w', 'x']:  # X轴
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][0] = 0
                    logging.info(f"⏸️ 停止 X轴移动")
                elif key.char in ['a', 'd']:  # Y轴
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][1] = 0
                    logging.info(f"⏸️ 停止 Y轴移动")
                elif key.char in ['r', 'f']:  # Z轴
                    for arm in get_active_arms():
                        teleop_velocity[arm]['pos'][2] = 0
                    logging.info(f"⏸️ 停止 Z轴移动")
                
                # 旋转控制键松开时清零对应轴的速度
                elif key.char in ['i', 'k']:  # Roll
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][0] = 0
                    logging.info(f"⏸️ 停止 Roll旋转")
                elif key.char in ['j', 'l']:  # Pitch
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][1] = 0
                    logging.info(f"⏸️ 停止 Pitch旋转")
                elif key.char in ['u', 'o']:  # Yaw
                    for arm in get_active_arms():
                        teleop_velocity[arm]['rot'][2] = 0
                    logging.info(f"⏸️ 停止 Yaw旋转")
                
                # ===== 手指控制键松开时清零速度（新增）=====
                elif key.char in ['z', 'c']:  # 手指闭合/张开
                    for arm in get_active_arms():
                        # 清零所有选中手指的速度
                        for finger in selected_fingers.get(arm, []):
                            joint_indices = FINGER_JOINT_MAP[finger]
                            for idx in joint_indices:
                                teleop_hand_velocity[arm][idx] = 0
                    logging.info(f"⏸️ 停止手指运动")
        except AttributeError:
            pass
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    logging.info("键盘监听器已启动（速度控制 + 手指速度控制）：")
    logging.info("  空格=切换模式 | S=保存 | Q=退出")
    logging.info("  1/2/3=选择手臂 | 按住WASD/RF=移动 | 按住IJKL/UO=旋转")
    logging.info("  T/Y/H/N/M=选择手指（可多选） | 按住Z/C=闭合/张开选中手指")
    logging.info("  💡 提示：先选手指(T/Y/H/N/M)，再按住Z/C流畅控制")
    return listener


def get_active_arms():
    """获取当前激活的手臂列表"""
    global active_arm
    if active_arm == 'both':
        return ['left', 'right']
    else:
        return [active_arm]


def show_selected_fingers():
    """显示当前选中的手指"""
    global selected_fingers, active_arm
    arm = active_arm if active_arm != 'both' else 'left'
    finger_names = {
        'thumb': '👍拇指', 'index': '☝️食指', 
        'middle': '🖕中指', 'ring': '💍无名指', 'pinky': '🤙小指'
    }
    selected = [finger_names[f] for f in selected_fingers[arm]]
    if selected:
        logging.info(f"🎯 当前选中手指 ({active_arm}): {' + '.join(selected)}")
    else:
        logging.info(f"🎯 当前选中手指 ({active_arm}): 无 (请按T/Y/H/N/M选择)")


def save_intervention_data():
    """保存干预数据用于后续DAgger训练"""
    global current_episode_data, intervention_data
    
    if not current_episode_data:
        logging.warning("没有干预数据可保存")
        return
    
    # 创建保存目录
    save_dir = Path("interventions")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"intervention_episode_{len(intervention_data)}_{timestamp}.npz"
    
    # 提取数据
    qpos_list = []
    policy_actions = []
    human_actions = []
    
    for data_point in current_episode_data:
        qpos_list.append(data_point['qpos'])
        policy_actions.append(data_point['policy_action'])
        human_actions.append(data_point['human_action'])
    
    # 保存为npz文件
    np.savez(
        save_path,
        qpos=np.array(qpos_list),
        policy_actions=np.array(policy_actions),
        human_actions=np.array(human_actions),
        metadata={
            'num_steps': len(current_episode_data),
            'timestamp': timestamp
        }
    )
    
    intervention_data.append(current_episode_data.copy())
    current_episode_data.clear()
    logging.info(f"✅ 干预数据已保存到 {save_path} (共 {len(qpos_list)} 步)")


def pose_to_transformation_matrix(pos, rpy):
    """
    将位置和RPY旋转转换为4x4变换矩阵
    
    Args:
        pos: 位置 [x, y, z]
        rpy: 旋转 [roll, pitch, yaw] (欧拉角，单位：弧度)
    
    Returns:
        T: 4x4 变换矩阵
    """
    roll, pitch, yaw = rpy
    
    # 旋转矩阵：R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # Roll (绕X轴)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (绕Y轴)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (绕Z轴)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵
    R = Rz @ Ry @ Rx
    
    # 构建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    
    return T


def transformation_matrix_to_pose(T):
    """
    将4x4变换矩阵转换为位置和RPY旋转
    
    Args:
        T: 4x4变换矩阵
    
    Returns:
        pos: 位置 [x, y, z]
        rpy: 旋转 [roll, pitch, yaw]
    """
    pos = T[:3, 3]
    R = T[:3, :3]
    
    # 从旋转矩阵提取欧拉角 (ZYX顺序)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    if np.abs(np.cos(pitch)) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock情况
        yaw = 0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    
    rpy = np.array([roll, pitch, yaw])
    
    return pos, rpy


def mmk2_ik_single_arm(target_pos, target_rot, current_joints, arm='left', spine_height=0.0):
    """
    使用MMK2 KDL求解单个手臂的IK
    
    Args:
        target_pos: 目标位置 [x, y, z]
        target_rot: 目标旋转 [roll, pitch, yaw]
        current_joints: 当前13维关节角度 [spine, left_arm(6), right_arm(6)]
        arm: 'left' or 'right'
        spine_height: 脊柱高度
    
    Returns:
        joint_angles: 6个手臂关节角度，失败返回current_joints对应手臂部分
    """
    global mmk2_kdl, MMK2_KDL_AVAILABLE
    
    if not MMK2_KDL_AVAILABLE or mmk2_kdl is None:
        raise RuntimeError("MMK2 KDL库未加载，无法进行IK求解！请安装casadi: pip install casadi")
    
    # 构建目标变换矩阵
    T_target = pose_to_transformation_matrix(target_pos, target_rot)
    
    # 准备参考位置（13维）
    ref_pos = current_joints.copy()
    ref_pos[0] = spine_height  # 设置脊柱高度
    
    try:
        # 调用MMK2 KDL求解IK
        if arm == 'left':
            ik_solutions = mmk2_kdl.inverse_kinematics(
                T_left=T_target,
                T_right=None,
                ref_pos=ref_pos,
                target_height=spine_height,
                force_calculate=True,
                use_clip=True
            )
        else:
            ik_solutions = mmk2_kdl.inverse_kinematics(
                T_left=None,
                T_right=T_target,
                ref_pos=ref_pos,
                target_height=spine_height,
                force_calculate=True,
                use_clip=True
            )
        
        if len(ik_solutions) > 0:
            # 返回第一个解的手臂部分
            solution = ik_solutions[0]
            # print("ik solutions", solution)
            # input()
            if arm == 'left':
                return solution[1:7]  # 左臂关节 1-6
            else:
                return solution[1:7]  
                # return solution[7:13]  # 右臂关节 7-12
        else:
            logging.warning(f"IK求解无解（{arm}臂），保持当前姿态")
            return current_joints[1:7] if arm == 'left' else current_joints[7:13]
    
    except Exception as e:
        logging.error(f"IK求解异常: {e}, 保持当前姿态")
        return current_joints[1:7] if arm == 'left' else current_joints[7:13]


def get_human_action_from_teleop(robot_interface,policy_action=None):
    """
    从键盘遥操作获取人类控制的动作（速度控制 + 手指速度控制）
    
    每个控制周期应用速度增量：
    - 手臂：更新目标位姿，通过IK求解关节角度
    - 手指：直接应用速度到关节角度（持续控制）
    """
    global teleop_target_pose, teleop_velocity, teleop_hand_joints, teleop_hand_velocity
    
    # ✅ 每个控制周期应用速度增量（关键：每次只应用一次）
    # 手臂速度
    for arm in ['left', 'right']:
        teleop_target_pose[arm]['pos'] += teleop_velocity[arm]['pos']
        teleop_target_pose[arm]['rot'] += teleop_velocity[arm]['rot']
    
    # ✅ 手指速度（新增：应用速度到关节角度）
    for arm in ['left', 'right']:
        teleop_hand_joints[arm] += teleop_hand_velocity[arm]
        # 限位到安全范围
        teleop_hand_joints[arm] = np.clip(teleop_hand_joints[arm], -np.pi, np.pi)
    
    # 获取当前机器人状态
    obs = get_observations(robot_interface)
    
    # 构建13维当前关节角度 [spine, left_arm(6), right_arm(6)]
    current_joints_13d = np.concatenate([
        [0.0],  # spine高度，暂时固定为0
        obs['arm_left'],
        obs['arm_right']
    ])
    
    # 左臂IK求解
    left_arm_joints = mmk2_ik_single_arm(
        teleop_target_pose['left']['pos'],
        teleop_target_pose['left']['rot'],
        current_joints_13d,
        arm='left',
        spine_height=0.0
    )
    left_arm_joints = np.array(left_arm_joints).flatten()
    
    # 右臂IK求解
    right_arm_joints = mmk2_ik_single_arm(
        teleop_target_pose['right']['pos'],
        teleop_target_pose['right']['rot'],
        current_joints_13d,
        arm='right',
        spine_height=0.0
    )
    right_arm_joints = np.array(right_arm_joints).flatten()
    
    # ✅ 手部姿态使用键盘控制的数据（新增功能）
    # 如果teleop_hand_joints全为0，说明用户未设置，则使用策略输出
    left_hand = teleop_hand_joints['left'].copy()
    right_hand = teleop_hand_joints['right'].copy()
    
    # 如果手部数据全为0，回退到策略输出（兼容性处理）
    if np.allclose(left_hand, 0) and policy_action is not None:
        left_hand = policy_action[6:18]
    if np.allclose(right_hand, 0) and policy_action is not None:
        right_hand = policy_action[24:36]
    
    # ✅ 维度检查
    assert left_arm_joints.shape[0] == 6, f"left_arm_joints维度错误: {left_arm_joints.shape[0]}, 期望6"
    assert right_arm_joints.shape[0] == 6, f"right_arm_joints维度错误: {right_arm_joints.shape[0]}, 期望6"
    assert left_hand.shape[0] == 12, f"left_hand维度错误: {left_hand.shape[0]}, 期望12"
    assert right_hand.shape[0] == 12, f"right_hand维度错误: {right_hand.shape[0]}, 期望12"
    
    # 组合成完整动作
    human_action = np.concatenate([
        left_arm_joints,   # 6
        left_hand,         # 12
        right_arm_joints,  # 6
        right_hand         # 12
    ])  # 总共36维
    
    return human_action


def initialize_teleop_from_current_state(robot_interface):
    """从当前机器人状态初始化遥操作目标位姿（使用MMK2 KDL FK）+ 手部状态"""
    global teleop_target_pose, teleop_hand_joints, mmk2_kdl, MMK2_KDL_AVAILABLE
    
    if not MMK2_KDL_AVAILABLE or mmk2_kdl is None:
        raise RuntimeError("MMK2 KDL库未加载，无法初始化遥操作！请安装casadi: pip install casadi")
    
    obs = get_observations(robot_interface)
    
    # 构建13维关节角度
    current_joints_13d = np.concatenate([
        [0.0],  # spine高度
        obs['arm_left'],
        obs['arm_right']
    ])
    
    # 使用MMK2 KDL计算正向运动学
    T_left, T_right = mmk2_kdl.forward_kinematics(current_joints_13d)
    
    # 从变换矩阵提取位置和旋转
    left_pos, left_rpy = transformation_matrix_to_pose(T_left)
    right_pos, right_rpy = transformation_matrix_to_pose(T_right)
    
    teleop_target_pose['left']['pos'] = left_pos
    teleop_target_pose['left']['rot'] = left_rpy
    teleop_target_pose['right']['pos'] = right_pos
    teleop_target_pose['right']['rot'] = right_rpy
    
    # ✅ 初始化手部姿态（从当前状态）- 新增功能
    # ⚠️ 关键：手部数据从机器人获取时是度数，需要转换为弧度！
    teleop_hand_joints['left'] = np.deg2rad(np.array(obs['hand_left']).flatten())
    teleop_hand_joints['right'] = np.deg2rad(np.array(obs['hand_right']).flatten())
    
    logging.info(f"✅ 初始化遥操作目标位姿 (MMK2 KDL FK):")
    logging.info(f"  左手腕: pos=[{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}], rot=[{left_rpy[0]:.3f}, {left_rpy[1]:.3f}, {left_rpy[2]:.3f}]")
    logging.info(f"  右手腕: pos=[{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}], rot=[{right_rpy[0]:.3f}, {right_rpy[1]:.3f}, {right_rpy[2]:.3f}]")
    logging.info(f"  左手部: avg_angle={np.rad2deg(np.mean(teleop_hand_joints['left'])):.1f}° (来自当前状态)")
    logging.info(f"  右手部: avg_angle={np.rad2deg(np.mean(teleop_hand_joints['right'])):.1f}° (来自当前状态)")


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
    # logging.debug(f"Executing xHand action: L={left_hand_action}, R={right_hand_action}")
    robot_interface.execute_xhand_action(left_hand_action, right_hand_action)
    
    # Execute arm actions
    # logging.debug(f"Executing MMK action: L={left_arm_action}, R={right_arm_action}")
    robot_interface.execute_mmk_action(np.concatenate([left_arm_action, right_arm_action]))


def model_inference_loop(config, robot_interface, policy, vision_encoder, image_processor):
    """Main inference loop for real-time control - 支持人类干预"""
    global control_mode, should_exit, current_episode_data
    
    robot_interface.reset_mmk()
    listener = setup_keyboard_listener()
    update_observation_window(config, robot_interface)
    initialize_teleop_from_current_state(robot_interface)
    prev_action = observation_window[-1]['qpos'].cpu().numpy()
    # hand degree to rad
    prev_action[6:18] = np.deg2rad(prev_action[6:18])
    prev_action[24:36] = np.deg2rad(prev_action[24:36])
    
    # Inference loop
    t = 0
    chunk_size = config['chunk_size']
    max_steps = config['max_steps']
    prev_mode = control_mode  # 记录上一次的模式，用于检测模式切换
    
    action_buffer = np.zeros([chunk_size, config['state_dim']])
    logging.info("=" * 70)
    logging.info("🚀 启动Human-in-the-Loop推理循环（手腕IK + 手指选择控制）")
    logging.info("=" * 70)
    logging.info("📋 模式控制：")
    logging.info("  空格键 - 切换 策略模式(AI) ↔ 人类控制模式(遥操作)")
    logging.info("  S键    - 保存干预数据")
    logging.info("  Q键    - 退出程序")
    logging.info("")
    logging.info("🎮 遥操作控制（人类模式下）：")
    logging.info("  手臂选择:")
    logging.info("    1 = 👈 左臂  |  2 = 👉 右臂  |  3 = 👐 双臂")
    logging.info("  位置控制 (按住移动，松开停止):")
    logging.info("    W/X = 前进/后退  |  A/D = 左移/右移  |  R/F = 上升/下降")
    logging.info("  旋转控制 (按住旋转，松开停止):")
    logging.info("    I/K = Roll(+/-)  |  J/L = Pitch(+/-)  |  U/O = Yaw(+/-)")
    logging.info("  手部控制 (手指选择模式 v2):")
    logging.info("    T/Y/H/N/M = 选择/取消 拇指/食指/中指/无名指/小指 (可多选)")
    logging.info("    Z = ✊ 闭合选中手指  |  C = 🖐️ 张开选中手指")
    logging.info("    💡 建议先按 T+Y 选择拇指和食指（捏取常用组合）")
    logging.info("")
    logging.info(f"📌 默认: 👈 LEFT 手臂 | 手指: 未选中（请按T/Y/H/N/M选择）")
    logging.info("=" * 70)
    
    try:
        while t < max_steps and not should_exit:
            loop_start_time = time.time()
            
            # Update observation window
            update_observation_window(config, robot_interface)
            
            # 获取策略动作（始终计算，用于对比）
            if t % chunk_size == 0:
                for _ in range(5):
                    update_observation_window(config, robot_interface)
                inference_start = time.time()
                action_buffer = inference_fn(
                    config, policy, vision_encoder, image_processor
                ).copy()
                logging.info(f"策略推理完成，动作形状: {action_buffer.shape}")
                logging.debug(f"推理时间: {time.time() - inference_start:.4f}s")
            
            policy_action = action_buffer[t % chunk_size]
            
            # 根据模式决定执行哪个动作
            with mode_lock:
                current_mode = control_mode
            
            # ✅ 检测模式切换：从policy → human时重新初始化位姿
            if prev_mode == "policy" and current_mode == "human":
                logging.info("🔄 检测到切换到人类模式，同步初始状态...")
                try:
                    # 1. 重新初始化目标位姿（基于当前状态）
                    initialize_teleop_from_current_state(robot_interface)
                    
                    # 2. ✅ 关键：立即获取对应的human_action并更新prev_action
                    # 避免第一次插值时有大的跳变
                    init_human_action = get_human_action_from_teleop(robot_interface,policy_action)
                    prev_action = init_human_action.copy()
                    
                    logging.info(f"✅ 无跳变切换完成，prev_action已同步")
                    logging.info("")
                    logging.info("🖐️ 手部控制提示：")
                    logging.info("  1️⃣ 先选择要控制的手指（按T/Y/H/N/M）")
                    logging.info("  2️⃣ 再用Z/C控制选中的手指")
                    logging.info("  💡 建议: T+Y（拇指+食指）适合捏取")
                    show_selected_fingers()
                except Exception as e:
                    logging.error(f"重新初始化失败: {e}")
            
            # ✅ 检测模式切换：从human → policy时强制重新推理
            elif prev_mode == "human" and current_mode == "policy":
                logging.info("🔄 检测到切换回策略模式，基于当前观察重新推理...")
                try:
                    # 强制重新获取观察并推理，确保策略基于当前状态
                    for _ in range(5):
                        update_observation_window(config, robot_interface)
                    
                    inference_start = time.time()
                    action_buffer = inference_fn(
                        config, policy, vision_encoder, image_processor
                    ).copy()
                    
                    # 重置chunk索引，从新推理的第一个动作开始
                    t = (t // chunk_size) * chunk_size
                    
                    # ✅ 关键：使用当前机器人实际qpos作为prev_action（而非策略输出）
                    # 这样插值会从当前位置平滑过渡到策略目标，避免跳变
                    current_qpos = observation_window[-1]['qpos'].cpu().numpy()
                    # 手部角度从度转弧度（与初始化时保持一致）
                    current_qpos[6:18] = np.deg2rad(current_qpos[6:18])
                    current_qpos[24:36] = np.deg2rad(current_qpos[24:36])
                    prev_action = current_qpos.copy()
                    
                    logging.info(f"✅ 策略推理完成，动作形状: {action_buffer.shape}")
                    logging.info(f"   推理时间: {time.time() - inference_start:.4f}s")
                    logging.info(f"   prev_action已设为当前qpos，确保平滑过渡")
                except Exception as e:
                    logging.error(f"重新推理失败: {e}")
            
            if current_mode == "policy":
                # 策略模式：执行策略动作
                action = policy_action
                
            else:  # human mode
                # 人类控制模式：获取人类动作
                human_action = get_human_action_from_teleop(robot_interface,policy_action)
                action = human_action
                
                # 记录干预数据（策略动作 vs 人类动作）
                current_episode_data.append({
                    'step': t,
                    'qpos': observation_window[-1]['qpos'].cpu().numpy(),
                    'policy_action': policy_action.copy(),
                    'human_action': human_action.copy(),
                    'timestamp': time.time()
                })
                
                # 只在有运动时打印日志
                is_moving = any(
                    np.any(teleop_velocity[arm]['pos'] != 0) or np.any(teleop_velocity[arm]['rot'] != 0)
                    for arm in ['left', 'right']
                )
                if is_moving:
                    logging.debug(f"人类干预中... 已记录 {len(current_episode_data)} 步")
            
            # Interpolate if needed (人类模式下禁用插值)
            if config['use_actions_interpolation'] and current_mode == "policy":
                # 只在策略模式下插值，人类模式直接执行以提高响应速度
                interp_actions = interpolate_action(config, prev_action, action)
            else:
                interp_actions = action[np.newaxis, :]
            
            # Execute actions
            for act in interp_actions:
                execute_actions(robot_interface, act)
                time.sleep(1.0 / config['control_frequency'])
            
            # Update previous action
            prev_action = action.copy()
            
            # Update previous mode (用于检测模式切换)
            prev_mode = current_mode
            
            # Increment time
            t += 1
            
            # 状态日志
            loop_time = time.time() - loop_start_time
            if t % 10 == 0:
                mode_emoji = "🟢" if current_mode == "policy" else "🔴"
                status_msg = (
                    f"{mode_emoji} 步数: {t}/{max_steps} | "
                    f"模式: {current_mode} | "
                    f"FPS: {1/loop_time:.2f} | "
                    f"干预数据: {len(current_episode_data)} 步"
                )
                
                # 在人类模式下显示详细控制信息
                if current_mode == "human":
                    arm_symbol = {"left": "👈", "right": "👉", "both": "👐"}[active_arm]
                    target = teleop_target_pose[active_arm if active_arm != 'both' else 'left']
                    vel = teleop_velocity[active_arm if active_arm != 'both' else 'left']
                    
                    # 检测是否有活动速度
                    pos_active = np.any(vel['pos'] != 0)
                    rot_active = np.any(vel['rot'] != 0)
                    
                    status_msg += (
                        f"\n  {arm_symbol} {active_arm.upper()} | "
                        f"位置: [{target['pos'][0]:.3f}, {target['pos'][1]:.3f}, {target['pos'][2]:.3f}]"
                    )
                    
                    if pos_active or rot_active:
                        status_msg += " | 🏃 运动中"
                        if pos_active:
                            status_msg += f" Vel: [{vel['pos'][0]:+.3f}, {vel['pos'][1]:+.3f}, {vel['pos'][2]:+.3f}]"
                        if rot_active:
                            status_msg += f" RotVel: [{vel['rot'][0]:+.2f}, {vel['rot'][1]:+.2f}, {vel['rot'][2]:+.2f}]"
                    else:
                        status_msg += " | ⏸️  静止"
                
                logging.info(status_msg)
    
    except KeyboardInterrupt:
        logging.info("⚠️ 检测到 Ctrl+C，正在退出...")
    
    except ValueError as e:
        logging.error("=" * 70)
        logging.error("❌ 值错误异常（通常是维度不匹配）")
        logging.error(f"错误信息: {e}")
        logging.error(f"发生在: 步数 {t}, 模式: {control_mode}")
        
        # 打印变量维度信息
        if 'prev_action' in locals():
            prev_shape = prev_action.shape if hasattr(prev_action, 'shape') else len(prev_action)
            logging.error(f"  prev_action: shape={prev_shape}")
        if 'action' in locals():
            act_shape = action.shape if hasattr(action, 'shape') else len(action)
            logging.error(f"  action: shape={act_shape}")
        if 'human_action' in locals():
            human_shape = human_action.shape if hasattr(human_action, 'shape') else len(human_action)
            logging.error(f"  human_action: shape={human_shape}")
        if 'policy_action' in locals():
            policy_shape = policy_action.shape if hasattr(policy_action, 'shape') else len(policy_action)
            logging.error(f"  policy_action: shape={policy_shape}")
        
        logging.error("=" * 70)
        
        import traceback
        logging.error("\n完整调用栈:")
        traceback.print_exc()
    
    except AssertionError as e:
        logging.error("=" * 70)
        logging.error("❌ 断言错误（维度检查失败）")
        logging.error(f"错误信息: {e}")
        logging.error(f"发生在: 步数 {t}, 模式: {control_mode}")
        logging.error("=" * 70)
        
        import traceback
        traceback.print_exc()
    
    except Exception as e:
        logging.error("=" * 70)
        logging.error(f"❌ 未知异常: {type(e).__name__}")
        logging.error(f"错误信息: {e}")
        logging.error(f"发生在: 步数 {t}, 模式: {control_mode}")
        logging.error("=" * 70)
        
        import traceback
        traceback.print_exc()
    
    finally:
        listener.stop()
        logging.info("✅ 键盘监听器已停止")
        
        # 如果有未保存的干预数据，提示保存
        if current_episode_data:
            logging.warning(f"⚠️ 有 {len(current_episode_data)} 步干预数据未保存！")
            try:
                save_prompt = input("是否保存？(y/n): ")
                if save_prompt.lower() == 'y':
                    save_intervention_data()
            except:
                logging.warning("无法获取用户输入，自动保存干预数据")
                save_intervention_data()


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
    
    # Initialize MMK2 KDL instance (必须成功)
    global mmk2_kdl, MMK2_KDL_AVAILABLE
    if not MMK2_KDL_AVAILABLE:
        logging.error("=" * 70)
        logging.error("❌ MMK2 KDL 库未加载！")
        logging.error("此程序需要MMK2 KDL库进行IK/FK求解")
        logging.error("请安装依赖: pip install casadi bidict xacrodoc lxml")
        logging.error("=" * 70)
        return
    
    try:
        mmk2_kdl = MMK2Kdl()
        logging.info("✅ MMK2 KDL 已成功初始化")
    except Exception as e:
        logging.error("=" * 70)
        logging.error(f"❌ MMK2 KDL 初始化失败: {e}")
        logging.error("请检查mmk2_kdl_py库是否正确安装")
        logging.error("=" * 70)
        import traceback
        traceback.print_exc()
        return
    
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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RDT模型部署，使用ZMQ与MMK和XHand转发器通信。
本脚本在RDT环境中运行，并通过ZMQ与机器人转发器通信。
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

# 添加项目根目录到路径，以支持相对导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型创建工具
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

# 全局变量，用于观测缓冲
observation_window = None
lang_embeddings = None
device = 'cuda'
dtype = torch.bfloat16


class ZMQRobotInterface:
    """通过ZMQ与MMK和XHand转发器通信的接口"""
    
    def __init__(self, config, mmk_host="localhost", mmk_port=5556, 
                 xhand_host="localhost", xhand_port=5557):
        # 保存配置
        self.config = config
        
        # ZMQ上下文
        self.context = zmq.Context()
        
        # MMK连接
        self.mmk_socket = self.context.socket(zmq.REQ)
        self.mmk_socket.connect(f"tcp://{mmk_host}:{mmk_port}")
        logging.info(f"已连接到MMK转发器 {mmk_host}:{mmk_port}")
        
        # XHand连接
        self.xhand_socket = self.context.socket(zmq.REQ)
        self.xhand_socket.connect(f"tcp://{xhand_host}:{xhand_port}")
        logging.info(f"已连接到XHand转发器 {xhand_host}:{xhand_port}")
        
        # 设置socket超时
        self.mmk_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        self.xhand_socket.setsockopt(zmq.RCVTIMEO, 5000)
        
        # 外部相机配置
        self.external_camera_names = ["cam_left_wrist", "cam_third_view", "cam_right_wrist"]
        self.external_camera_ids = config["ext_cam_ids"]
        self.external_cameras = {}
        
        # 在本地初始化外部相机
        self._initialize_external_cameras()
    
    def _initialize_external_cameras(self):
        """在主进程本地初始化外部USB相机"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        self.realsense_pipeline = pipeline
        logging.info("已初始化realsense相机")

        for i, cam_id in enumerate(self.external_camera_ids):
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开相机 {cam_id}")
            
            # 配置相机参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            self.external_cameras[self.external_camera_names[i]] = cap
            logging.info(f"已初始化外部相机 {self.external_camera_names[i]}，ID为 {cam_id}")
    
    def _capture_external_camera_images(self):
        """在本地采集外部USB相机的图像"""
        images = {}
        
        for cam_name, cap in self.external_cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法从相机 {cam_name} 读取图像")
            images[cam_name] = frame
        
        # 对于缺失的相机，添加占位图像
        for camera_name in self.external_camera_names:
            if camera_name not in images:
                images[camera_name] = np.zeros((480, 640, 3), dtype=np.uint8)
                logging.warning(f"未找到相机 {camera_name}，使用占位图像")
        
        return images
    
    def get_mmk_observations(self):
        """获取MMK机器人的观测（仅qpos和头部相机）"""
        request = {'command': 'get_observations'}
        self.mmk_socket.send_json(request)
        
        try:
            response = self.mmk_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"MMK错误: {response['error']}")
            
            # 解码头部相机图像（base64）
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
            
            # 本地获取外部相机图像
            external_images = self._capture_external_camera_images()
            
            # 合并所有图像
            images = external_images
            images['head_camera'] = head_camera_img
            
            return {
                'qpos': np.array(response['qpos']),
                'images': images
            }
        except zmq.error.Again:
            raise RuntimeError("MMK转发器超时")
    
    def get_xhand_observations(self):
        """获取XHand的观测"""
        request = {'command': 'get_observations'}
        self.xhand_socket.send_json(request)
        
        try:
            response = self.xhand_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"XHand错误: {response['error']}")
            
            return {
                'left_hand': response['left_hand'],
                'right_hand': response['right_hand']
            }
        except zmq.error.Again:
            raise RuntimeError("XHand转发器超时")
    
    def execute_mmk_action(self, action):
        """在MMK机器人上执行动作并保存到文件"""
        # 如果action是numpy数组则转为list
        action_list = action.tolist() if isinstance(action, np.ndarray) else action
        
        # 保存动作到文件
        if not hasattr(self, 'mmk_action_file'):
            # 首次执行时创建文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.mmk_action_file = f"joint_data_plt/mmk_actions_{timestamp}.jsonl"
        
        # 追加动作到文件
        with open(self.mmk_action_file, 'a') as f:
            json.dump({"timestamp": time.time(), "action": action_list}, f)
            f.write('\n')  # JSON Lines格式
        
        # 发送请求
        request = {
            'command': 'execute_action',
            'action': action_list
        }
        self.mmk_socket.send_json(request)
        
        try:
            response = self.mmk_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"MMK错误: {response['error']}")
            return response
        except zmq.error.Again:
            raise RuntimeError("MMK转发器超时")

    def execute_xhand_action(self, left_hand_action, right_hand_action):
        """在XHand上执行动作并保存到文件"""
        # 如果动作是numpy数组则转为list
        left_action_list = left_hand_action.tolist() if isinstance(left_hand_action, np.ndarray) else left_hand_action
        right_action_list = right_hand_action.tolist() if isinstance(right_hand_action, np.ndarray) else right_hand_action
        
        # 首次执行时创建文件
        if not hasattr(self, 'xhand_left_action_file'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.xhand_left_action_file = f"joint_data_plt/xhand_left_actions_{timestamp}.jsonl"
            self.xhand_right_action_file = f"joint_data_plt/xhand_right_actions_{timestamp}.jsonl"
        
        # 追加动作到文件
        with open(self.xhand_left_action_file, 'a') as f:
            json.dump({"timestamp": time.time(), "action": left_action_list}, f)
            f.write('\n')  # JSON Lines格式
        
        with open(self.xhand_right_action_file, 'a') as f:
            json.dump({"timestamp": time.time(), "action": right_action_list}, f)
            f.write('\n')  # JSON Lines格式
        
        # 发送请求
        request = {
            'command': 'execute_action',
            'action_data': {
                'left_hand': left_action_list,
                'right_hand': right_action_list
            }
        }
        self.xhand_socket.send_json(request)
        
        try:
            response = self.xhand_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"XHand错误: {response['error']}")
            return response
        except zmq.error.Again:
            raise RuntimeError("XHand转发器超时")
    
    def reset_mmk(self):
        """重置MMK机器人"""
        request = {'command': 'reset'}
        self.mmk_socket.send_json(request)
        
        try:
            response = self.mmk_socket.recv_json()
            if 'error' in response:
                raise RuntimeError(f"MMK错误: {response['error']}")
            return response
        except zmq.error.Again:
            raise RuntimeError("MMK转发器超时")
    
    def close(self):
        """关闭ZMQ连接并释放资源"""
        # 释放外部相机
        for _, cap in self.external_cameras.items():
            if cap is not None:
                cap.release()
        
        # 关闭ZMQ socket
        self.mmk_socket.close()
        self.xhand_socket.close()
        self.context.term()


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_model(config, pretrained):
    """直接从预训练权重初始化RDT模型"""
    global device, dtype
    logging.info("正在创建RDT模型...")
    
    # 创建视觉编码器（Siglip）
    vision_encoder = SiglipVisionTower(vision_tower=config["vision_encoder_name"], args=None)
    image_processor = vision_encoder.image_processor
    
    # 直接从预训练权重加载模型
    logging.info(f"从{pretrained}加载预训练模型")
    runner = RDTRunner.from_pretrained(pretrained)
    
    # 设置设备
    runner = runner.to(device, dtype=dtype)
    vision_encoder = vision_encoder.to(device, dtype=dtype)
    
    # 设置为评估模式
    runner.eval()
    vision_encoder.eval()
    
    return runner, vision_encoder, image_processor


def get_observations(robot_interface: ZMQRobotInterface):
    """通过ZMQ从机器人系统获取观测"""
    # 获取MMK观测
    mmk_obs = robot_interface.get_mmk_observations()
    arm_qpos = mmk_obs['qpos']
    images = mmk_obs['images']
    
    # 获取XHand观测
    xhand_obs = robot_interface.get_xhand_observations()
    
    # 提取机械臂位置（每个臂前6维）
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
    """用最新数据更新观测窗口"""
    global observation_window
    
    # 获取当前观测
    obs = get_observations(robot_interface)
    
    # 处理图像
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    images = {}
    for camera_name in config["camera_names"]:
        img = obs['images'].get(camera_name, np.zeros((480, 640, 3), dtype=np.uint8))
        images[camera_name] = jpeg_mapping(img)
    
    # 合并位置
    qpos = np.concatenate((
        np.array(obs['arm_left']),
        np.array(obs['hand_left']),
        np.array(obs['arm_right']),
        np.array(obs['hand_right'])
    ), axis=0)
    
    qpos = torch.from_numpy(qpos).float().cuda()
    
    # 处理图像
    processed_images = {}
    for camera_name, img in images.items():
        processed_images[camera_name] = img

    curr_obs = {
        'qpos': qpos,
        'images': processed_images
    }
    
    # 如果观测窗口未初始化则初始化
    if observation_window is None:
        img_history_size = config.get("img_history_size", 2)
        observation_window = deque(maxlen=img_history_size)
        
        observation_window.append(curr_obs)
        observation_window.append(curr_obs.copy())
    else:
        observation_window.append(curr_obs)


def inference_fn(config, policy, vision_encoder, image_processor):
    """用当前观测运行模型推理"""
    global observation_window, lang_embeddings, device, dtype
    
    # 从观测窗口获取图像
    image_arrs = []
    for camera_name in config["camera_names"]:
        for i in range(len(observation_window)):
            img = observation_window[i]['images'][camera_name]
            if img is None:
                logging.warning(f"图像 {camera_name} 为None，将使用背景图像")
            image_arrs.append(img)
    
    # 用于填充的背景图像
    background_color = np.array([
        int(x*255) for x in image_processor.image_mean
    ], dtype=np.uint8).reshape(1, 1, 3)
    background_image = np.ones((
        image_processor.size["height"], 
        image_processor.size["width"], 3), dtype=np.uint8
    ) * background_color
    
    # 预处理图像
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
    
    # 用视觉编码器处理
    image_embeds = vision_encoder(image_tensor).detach()
    image_embeds = image_embeds.reshape(-1, vision_encoder.hidden_size).unsqueeze(0)
    
    # 获取本体感知
    proprio = observation_window[-1]['qpos']
    proprio = proprio.unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    
    # 设置动作掩码
    action_mask = torch.ones(1, 1, config['state_dim'], device=device, dtype=dtype)
    
    # 设置控制频率
    ctrl_freq = torch.tensor([20], device=device)
    
    # 推理
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
    
    return actions.squeeze(0).to(torch.float).cpu().numpy()


def interpolate_action(config, prev_action, cur_action):
    """在前后动作间插值，实现平滑运动"""
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
    logging.info(f"插值动作，共{step}步")
    return new_actions[1:]


def execute_actions(robot_interface, actions):
    """通过ZMQ在机器人上执行动作"""
    # 拆分动作
    left_arm_action = actions[:6]
    left_hand_action = actions[6:18]
    right_arm_action = actions[18:24]
    right_hand_action = actions[24:36]
    
    # 执行手部动作
    logging.debug(f"执行xHand动作: L={left_hand_action}, R={right_hand_action}")
    robot_interface.execute_xhand_action(left_hand_action, right_hand_action)
    
    # 执行机械臂动作
    logging.debug(f"执行MMK动作: L={left_arm_action}, R={right_arm_action}")
    robot_interface.execute_mmk_action(np.concatenate([left_arm_action, right_arm_action]))


def model_inference_loop(config, robot_interface, policy, vision_encoder, image_processor):
    """实时控制的主推理循环"""
    robot_interface.reset_mmk()
    
    # 初始化前一动作
    update_observation_window(config, robot_interface)
    prev_action = observation_window[-1]['qpos'].cpu().numpy()
    
    # 推理循环
    t = 0
    chunk_size = config['chunk_size']
    max_steps = config['max_steps']
    
    action_buffer = np.zeros([chunk_size, config['state_dim']])
    
    logging.info("开始推理循环...")
    while t < max_steps:
        loop_start_time = time.time()
        
        # 更新观测窗口
        update_observation_window(config, robot_interface)
        
        # 需要时运行推理
        if t % chunk_size == 0:
            inference_start = time.time()
            action_buffer = inference_fn(
                config, policy, vision_encoder, image_processor
            ).copy()
            logging.info(f"推理并获得动作 {action_buffer.shape}")
            logging.debug(f"推理耗时: {time.time() - inference_start:.4f}s")
        
        # 获取当前动作
        raw_action = action_buffer[t % chunk_size]
        action = raw_action
        
        # 如需插值则插值
        if config['use_actions_interpolation']:
            interp_actions = interpolate_action(config, prev_action, action)
        else:
            interp_actions = action[np.newaxis, :]
        
        # 执行动作
        for act in interp_actions:
            execute_actions(robot_interface, act)
            time.sleep(1.0 / config['control_frequency'])
        
        # 更新前一动作
        prev_action = action.copy()
        
        # 时间步+1
        t += 1
        
        # 日志状态
        loop_time = time.time() - loop_start_time
        logging.debug(f"步骤 {t}/{max_steps}, 循环耗时: {loop_time:.4f}s, FPS: {1/loop_time:.2f}")


def prepare_language_embeddings(args):
    """为模型准备语言嵌入"""
    global lang_embeddings, device, dtype
    
    logging.info(f"从{args.lang_embeddings_path}加载语言嵌入")
    lang_dict = torch.load(args.lang_embeddings_path)
    logging.info(f"使用指令: \"{lang_dict['instruction']}\"，来自 \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"].to(device, dtype=dtype)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RDT模型部署（ZMQ）")
    
    # 模型配置
    parser.add_argument("--config-path", type=str, default="deploy/mmk_xhand_config.yaml", 
                        help="模型配置YAML路径")
    parser.add_argument("--pretrained-model-path", type=str, required=True,
                        help="预训练模型路径")
    
    # 语言配置
    parser.add_argument("--lang-embeddings-path", type=str, default=None,
                        help="预先计算的语言嵌入路径")
    
    # ZMQ配置
    parser.add_argument("--mmk-host", type=str, default="localhost",
                        help="MMK转发器主机")
    parser.add_argument("--mmk-port", type=int, default=5556,
                        help="MMK转发器端口")
    parser.add_argument("--xhand-host", type=str, default="localhost",
                        help="XHand转发器主机")
    parser.add_argument("--xhand-port", type=int, default=5557,
                        help="XHand转发器端口")
    
    args = parser.parse_args()
    return args


def main():
    """主入口"""
    logging.basicConfig(level=logging.INFO)
    
    # 解析参数
    args = get_args()
    
    # 加载配置
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config["seed"])
    
    # 创建RDT模型
    logging.info("正在创建RDT模型...")
    policy, vision_encoder, image_processor = create_model(
        config=config,
        pretrained=args.pretrained_model_path,
    )
    
    prepare_language_embeddings(args)
    
    # 初始化机器人接口
    logging.info("正在初始化ZMQ机器人接口...")
    robot_interface = ZMQRobotInterface(
        config=config,
        mmk_host=args.mmk_host,
        mmk_port=args.mmk_port,
        xhand_host=args.xhand_host,
        xhand_port=args.xhand_port
    )
    
    # 测试连接
    try:
        logging.info("正在测试连接...")
        _ = robot_interface.get_mmk_observations()
        _ = robot_interface.get_xhand_observations()
        logging.info("所有连接成功！")
    except Exception as e:
        logging.error(f"连接测试失败: {e}")
        robot_interface.close()
        return
    
    # 运行推理
    logging.info("开始RDT推理循环...")
    try:
        with torch.inference_mode():
            while input("按回车继续，输入任意内容退出") == "":
                model_inference_loop(
                    config, robot_interface,
                    policy, vision_encoder, image_processor
                )
    except KeyboardInterrupt:
        logging.info("用户中断，正在关闭...")
    except Exception as e:
        logging.error(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.info("正在关闭连接...")
        robot_interface.close()
        logging.info("推理结束")


if __name__ == "__main__":
    main()
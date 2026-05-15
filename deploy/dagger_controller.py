#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human-in-the-Loop Controller for DAgger/HG-DAgger
结合策略推理和人类干预的混合控制系统
"""

import argparse
import time
import logging
import numpy as np
import torch
from pathlib import Path
from collections import deque
from pynput import keyboard
from threading import Event, Lock

# 导入现有模块
from rdt_inference_zmq import (
    ZMQRobotInterface, create_model, 
    update_observation_window, inference_fn,
    execute_actions
)
from robots.common import make_robot_from_yaml

class HitLController:
    """Human-in-the-Loop控制器"""
    
    def __init__(self, config, robot_interface, policy, vision_encoder, image_processor, robot_teleop):
        self.config = config
        self.robot_interface = robot_interface
        self.policy = policy
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.robot_teleop = robot_teleop  # 遥操作Robot对象
        
        # 控制模式
        self.mode = "policy"  # "policy" | "human" | "blending"
        self.mode_lock = Lock()
        
        # 数据缓冲
        self.intervention_data = []
        self.current_episode = []
        
        # 干预检测
        self.intervention_event = Event()
        self.should_exit = False
         
        # 性能监控
        self.recent_errors = deque(maxlen=10)
        self.ood_threshold = 0.5  # 可调
        
    def setup_keyboard_listener(self):
        """设置键盘监听器"""
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    # 切换模式
                    with self.mode_lock:
                        if self.mode == "policy":
                            self.mode = "human"
                            self.robot_teleop.enter_passive_mode()
                            logging.info("🔴 切换到人类控制模式")
                        else:
                            self.mode = "policy"
                            logging.info("🟢 切换回策略控制模式")
                            
                elif key.char == 's':
                    # 保存当前episode的干预数据
                    self.save_intervention_data()
                    logging.info("💾 保存干预数据")
                    
                elif key == keyboard.Key.esc:
                    logging.info("⛔ 退出程序")
                    self.should_exit = True
                    
            except AttributeError:
                pass
                
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
    
    def detect_ood(self, observation, policy_action):
        """
        检测Out-of-Distribution情况
        可以基于多种指标：
        1. 模型置信度
        2. 观测与训练数据的距离
        3. 执行误差
        """
        # 示例：简单的基于误差的检测
        if len(self.recent_errors) >= 5:
            avg_error = np.mean(self.recent_errors)
            if avg_error > self.ood_threshold:
                return True
        return False
    
    def get_policy_action(self):
        """获取策略输出的动作"""
        # 更新观测
        update_observation_window(self.config, self.robot_interface)
        
        # 推理
        action_buffer = inference_fn(
            self.config, self.policy, 
            self.vision_encoder, self.image_processor
        )
        
        return action_buffer[0]  # 返回第一个动作
    
    def get_human_action(self):
        """获取人类遥操作的动作"""
        # 从遥操作Robot获取状态
        observation = self.robot_teleop.capture_observation()
        
        # 提取关节状态作为action
        left_arm = observation.get("/action/left_arm/joint_state", {}).get("data", [])
        right_arm = observation.get("/action/right_arm/joint_state", {}).get("data", [])
        
        # 从Vision Pro获取手部动作（需要集成receive_from_vision_pro.py）
        # 这里需要您的XHandTeleOps实例
        # left_hand = xhand_teleops.get_hand_action("hand_a")
        # right_hand = xhand_teleops.get_hand_action("hand_b")
        
        # 暂时返回占位符
        left_hand = np.zeros(12)
        right_hand = np.zeros(12)
        
        # 组合成完整action
        action = np.concatenate([left_arm, left_hand, right_arm, right_hand])
        return action
    
    def blend_actions(self, policy_action, human_action, alpha=0.5):
        """混合策略和人类动作（可选功能）"""
        return alpha * policy_action + (1 - alpha) * human_action
    
    def control_loop(self):
        """主控制循环"""
        logging.info("🚀 启动Human-in-the-Loop控制循环")
        logging.info("提示：按空格键切换策略/人类模式，按's'保存数据，按ESC退出")
        
        # 设置键盘监听
        listener = self.setup_keyboard_listener()
        
        # 初始化
        step = 0
        max_steps = self.config.get('max_steps', 10000)
        chunk_size = self.config['chunk_size']
        
        while step < max_steps and not self.should_exit:
            loop_start = time.time()
            
            # 获取当前观测
            obs = self.robot_interface.get_mmk_observations()
            
            # 获取策略动作
            policy_action = self.get_policy_action()
            
            # 决定使用哪个动作
            with self.mode_lock:
                current_mode = self.mode
            
            if current_mode == "policy":
                # 检测是否需要干预
                if self.detect_ood(obs, policy_action):
                    logging.warning("⚠️  检测到OOD，建议人类干预！")
                
                action = policy_action
                
            elif current_mode == "human":
                # 人类控制
                human_action = self.get_human_action()
                action = human_action
                
                # 记录干预数据
                self.current_episode.append({
                    'step': step,
                    'observation': obs,
                    'policy_action': policy_action,
                    'human_action': human_action,
                    'timestamp': time.time()
                })
                
            else:  # blending mode
                human_action = self.get_human_action()
                action = self.blend_actions(policy_action, human_action)
            
            # 执行动作
            execute_actions(self.robot_interface, action)
            
            # 控制频率
            dt = time.time() - loop_start
            sleep_time = max(0, 1.0 / self.config['control_frequency'] - dt)
            time.sleep(sleep_time)
            
            step += 1
            
            if step % 10 == 0:
                logging.info(f"步数: {step}/{max_steps}, 模式: {current_mode}, FPS: {1/(time.time()-loop_start):.2f}")
        
        listener.stop()
        logging.info("✅ 控制循环结束")
    
    def save_intervention_data(self):
        """保存干预数据用于DAgger训练"""
        if not self.current_episode:
            logging.warning("没有干预数据可保存")
            return
        
        # 保存为适合训练的格式
        save_path = Path(f"interventions/episode_{len(self.intervention_data)}.npz")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 转换为numpy数组
        observations = [d['observation'] for d in self.current_episode]
        policy_actions = np.array([d['policy_action'] for d in self.current_episode])
        human_actions = np.array([d['human_action'] for d in self.current_episode])
        
        np.savez(
            save_path,
            observations=observations,
            policy_actions=policy_actions,
            human_actions=human_actions,
            metadata={'num_steps': len(self.current_episode)}
        )
        
        self.intervention_data.append(self.current_episode)
        self.current_episode = []
        logging.info(f"✅ 干预数据已保存到 {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--lang-embeddings-path", type=str, required=True)
    parser.add_argument("--robot-config-path", type=str, required=True, 
                        help="遥操作Robot的配置文件")
    parser.add_argument("--mmk-host", type=str, default="localhost")
    parser.add_argument("--mmk-port", type=int, default=5556)
    parser.add_argument("--xhand-host", type=str, default="localhost")
    parser.add_argument("--xhand-port", type=int, default=5557)
    args = parser.parse_args()
    
    # 加载配置
    import yaml
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    # 初始化策略
    policy, vision_encoder, image_processor = create_model(config, args.pretrained_model_path)
    
    # 初始化机器人接口（用于策略执行）
    robot_interface = ZMQRobotInterface(
        config, args.mmk_host, args.mmk_port,
        args.xhand_host, args.xhand_port
    )
    
    # 初始化遥操作Robot（用于人类控制）
    robot_teleop = make_robot_from_yaml(args.robot_config_path)
    robot_teleop.connect()
    
    # 创建HitL控制器
    controller = HitLController(
        config, robot_interface, policy, 
        vision_encoder, image_processor, robot_teleop
    )
    
    try:
        # 运行控制循环
        controller.control_loop()
    finally:
        robot_teleop.exit()
        logging.info("程序结束")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
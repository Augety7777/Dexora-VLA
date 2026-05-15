#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器人数据重播脚本（整合机械臂运动+视频）

功能：
1. 从BSON文件加载录制的机械臂关节状态和视频数据
2. 同步重放机械臂运动和视频画面
3. 支持控制回放速度
4. 可视化重放过程
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import cv2
from tqdm import tqdm
from termcolor import colored
from airbot_data.io import load_bson

# 机械臂控制相关导入
from mmk2_types.types import MMK2Components
from mmk2_types.grpc_msgs import JointState, TrajectoryParams, GoalStatus
from airbot_py.airbot_mmk2 import AirbotMMK2

# 初始化日志
def init_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging

logging = init_logging()

class RobotReplayer:
    def __init__(self, robot_ip: str = "192.168.11.200", robot_port: int = 50055):
        """初始化机械臂控制接口"""
        try:
            self.mmk2 = AirbotMMK2(ip=robot_ip, port=robot_port)
            logging.info(f"成功连接到机械臂控制器 {robot_ip}:{robot_port}")
        except Exception as e:
            logging.error(f"连接机械臂失败: {str(e)}")
            raise

    def move_to_start_position(self):
        """移动到起始位置"""
        start_joint_action = {
            MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
            MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
            MMK2Components.SPINE: JointState(position=[0.1]),
        }
        
        if self.mmk2.set_goal(start_joint_action, TrajectoryParams()).value != GoalStatus.Status.SUCCESS:
            logging.error("移动到起始位置失败")
            return False
        time.sleep(2)  # 等待机械臂到位
        return True

    def replay_joint_states(self, joint_states: List[Dict], fps: int = 30):
        """
        重放关节状态序列
        :param joint_states: 关节状态列表
        :param fps: 目标帧率
        """
        if not joint_states:
            logging.warning("没有可重放的关节状态数据")
            return
        
        logging.info(f"开始重放机械臂运动，共 {len(joint_states)} 帧，目标FPS: {fps}")
        
        for i, state in enumerate(joint_states):
            start_time = time.perf_counter()
            
            # 转换为机械臂控制需要的格式
            action = {
                MMK2Components.LEFT_ARM: JointState(position=state['left_arm']),
                MMK2Components.RIGHT_ARM: JointState(position=state['right_arm']),
                # 可以添加其他关节...
            }
            
            # 发送控制指令
            result = self.mmk2.set_goal(action, TrajectoryParams())
            if result.value != GoalStatus.Status.SUCCESS:
                logging.warning(f"第 {i} 帧控制指令发送失败")
            
            # 控制播放速度
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, 1.0/fps - elapsed)
            time.sleep(sleep_time)

class DataReplayer:
    def __init__(self, root: str, repo_id: str):
        """初始化数据重播器"""
        self.local_dir = Path(root) / repo_id
        if not self.local_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.local_dir}")
        
        # 加载数据集信息
        self.rec_info_path = self.local_dir / "data_recording_info.json"
        self.rec_info = {}
        if self.rec_info_path.exists():
            with open(self.rec_info_path) as f:
                self.rec_info = json.load(f)
        
        # 获取所有episode文件
        self.episode_files = sorted(self.local_dir.glob("episode_*.bson"))
        if not self.episode_files:
            raise FileNotFoundError(f"未找到任何episode文件: {self.local_dir}")
        
        logging.info(f"找到 {len(self.episode_files)} 个episode文件")

    def load_episode(self, episode_index: int):
        """加载指定episode的数据"""
        if episode_index >= len(self.episode_files):
            raise IndexError(f"episode索引 {episode_index} 超出范围")
        
        file_path = self.episode_files[episode_index]
        logging.info(f"加载episode {episode_index}: {file_path}")
        
        try:
            data = load_bson(file_path)
            logging.info(f"成功加载episode {episode_index}")
            return data
        except Exception as e:
            logging.error(f"加载episode {episode_index} 失败: {str(e)}")
            raise

    def extract_joint_states(self, data: Dict) -> List[Dict]:
        """从数据中提取关节状态序列"""
        joint_states = []
        
        # 更安全的数据提取方式
        try:
            # 获取左右臂数据，如果不存在则返回空列表
            left_arm_states = data['data'].get('/observation/left_arm/joint_state', [])
            right_arm_states = data['data'].get('/observation/right_arm/joint_state', [])
            
            # 确保我们得到的是列表
            if not isinstance(left_arm_states, list) or not isinstance(right_arm_states, list):
                logging.error("关节状态数据格式不正确")
                return []
            
            # 确保数据长度一致
            min_length = min(len(left_arm_states), len(right_arm_states))
            for i in range(min_length):
                # 检查数据是否存在且格式正确
                if (isinstance(left_arm_states[i], dict) and 
                    isinstance(right_arm_states[i], dict) and
                    'data' in left_arm_states[i] and 
                    'data' in right_arm_states[i] and
                    'position' in left_arm_states[i]['data'] and
                    'position' in right_arm_states[i]['data']):
                    
                    joint_states.append({
                        'left_arm': left_arm_states[i]['data']['position'],
                        'right_arm': right_arm_states[i]['data']['position'],
                        'timestamp': left_arm_states[i].get('timestamp', 0)
                    })
                else:
                    logging.warning(f"第 {i} 帧关节状态数据格式不正确")
                    logging.debug(f"left_arm_states[{i}]: {left_arm_states[i]}")
                    logging.debug(f"right_arm_states[{i}]: {right_arm_states[i]}")
                    
            return joint_states
            
        except Exception as e:
            logging.error(f"提取关节状态时出错: {str(e)}")
            return []

    def replay_episode(
        self, 
        robot_replayer: RobotReplayer,
        episode_index: int, 
        fps: Optional[int] = None,
        num_rollouts: int = 1,
        show_video: bool = True
    ):
        
        """重放指定episode的数据（机械臂运动+视频）"""
        data = self.load_episode(episode_index)
        joint_states = self.extract_joint_states(data)
        if not joint_states:
            logging.error("没有可用的关节状态数据，无法重放")
            return
        frame_count = len(joint_states)
        
        # 获取视频数据
        image_keys = [k for k in data['data'] if 'image' in k]
        video_frames = {}
        for key in image_keys:
            video_frames[key] = [frame['data'] for frame in data['data'][key]]
        
        for rollout in range(num_rollouts):
            logging.info(f"开始重放episode {episode_index} (第 {rollout+1}/{num_rollouts} 次)")
            
            # 显示episode信息
            print(colored(f"\nEpisode {episode_index} 信息:", 'cyan'))
            print(f"总帧数: {frame_count}")
            print(f"录制时间: {data['timestamp']}")
            print(f"设备信息: {data['metadata']['station_id']}")
            
            # 创建进度条
            pbar = tqdm(total=frame_count, desc=f"重放episode {episode_index}")
            
            for frame_idx in range(frame_count):
                start_time = time.perf_counter()
                
                # 1. 控制机械臂运动
                robot_replayer.replay_joint_states([joint_states[frame_idx]], fps=fps)
                
                # 2. 显示视频帧
                if show_video and video_frames:
                    for cam_name, frames in video_frames.items():
                        if frame_idx < len(frames):
                            cv2.imshow(cam_name.split('/')[-1], frames[frame_idx])
                    
                    # 显示帧信息
                    info_img = np.zeros((200, 600, 3), dtype=np.uint8)
                    cv2.putText(info_img, f"Episode: {episode_index}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(info_img, f"Frame: {frame_idx}/{frame_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(info_img, f"FPS: {fps if fps else 'MAX'}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Replay Info", info_img)
                    
                    # 检查用户输入
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # 按q退出
                        logging.info("用户请求退出重放")
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('p'):  # 按p暂停
                        logging.info("暂停，按任意键继续...")
                        cv2.waitKey(0)
                
                # 更新进度条
                pbar.update(1)
                
                # 控制播放速度
                if fps is not None:
                    elapsed = time.perf_counter() - start_time
                    sleep_time = max(0, 1.0/fps - elapsed)
                    time.sleep(sleep_time)
            
            pbar.close()
            cv2.destroyAllWindows()
        
        logging.info(f"完成episode {episode_index}的重放")

def main():
    parser = argparse.ArgumentParser(description='机器人数据重播脚本（机械臂+视频）')
    parser.add_argument('--root', type=str, default='data',
                       help='数据根目录 (默认: data)')
    parser.add_argument('--repo-id', type=str, default='raw/example',
                       help='数据集ID (默认: raw/example)')
    parser.add_argument('--start-episode', type=int, default=0,
                       help='开始重放的episode索引 (默认: 0)')
    parser.add_argument('--num-episodes', type=int, default=1,
                       help='要重放的episode数量 (默认: 1)')
    parser.add_argument('--fps', type=int, default=20,
                       help='重放帧率 (默认: 20)')
    parser.add_argument('--num-rollouts', type=int, default=1,
                       help='每个episode的重放次数 (默认: 1)')
    parser.add_argument('--robot-ip', type=str, default="192.168.11.200",
                       help='机械臂控制器IP地址 (默认: 192.168.11.200)')
    parser.add_argument('--robot-port', type=int, default=50055,
                       help='机械臂控制器端口 (默认: 50055)')
    parser.add_argument('--no-video', type=bool, default=False,
                       help='不显示视频画面')
    # parser.add_argument('--no-video', action='store_true',
    #                    help='不显示视频画面')
    
    args = parser.parse_args()
    
    try:
        # 初始化机械臂控制
        robot_replayer = RobotReplayer(args.robot_ip, args.robot_port)
        if not robot_replayer.move_to_start_position():
            logging.error("无法移动到起始位置，退出")
            return
        
     
        # 初始化数据重播
        data_replayer = DataReplayer(args.root, args.repo_id)
        
        # 重放指定范围的episode
        for ep_idx in range(args.start_episode, 
                          args.start_episode + args.num_episodes):
            data_replayer.replay_episode(
                robot_replayer=robot_replayer,
                episode_index=ep_idx,
                fps=args.fps,
                num_rollouts=args.num_rollouts,
                show_video=not args.no_video
            )
            
    except Exception as e:
        logging.error(f"重放过程中发生错误: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
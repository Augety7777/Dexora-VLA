#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器人数据重播脚本

功能：
1. 从指定目录加载录制的BSON数据
2. 按帧回放机器人动作和传感器数据
3. 支持控制回放速度
4. 可选显示摄像头数据

使用方法：
python replay.py --root data --repo-id raw/example --start-episode 0 --num-episodes 1 --fps 20
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from airbot_data.io import load_bson
from termcolor import colored

# 初始化日志
def init_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging

logging = init_logging()

class DataReplayer:
    def __init__(self, root: str, repo_id: str):
        """
        初始化重播器
        :param root: 数据根目录
        :param repo_id: 数据集ID
        """
        self.local_dir = Path(root) / repo_id
        if not self.local_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.local_dir}")
        
        # 加载数据集信息
        self.rec_info_path = self.local_dir / "data_recording_info.json"
        if self.rec_info_path.exists():
            with open(self.rec_info_path) as f:
                self.rec_info = json.load(f)
            logging.info(f"找到录制信息，最后episode索引: {self.rec_info['last_episode_index']}")
        
        # 获取所有episode文件
        self.episode_files = sorted(self.local_dir.glob("episode_*.bson"))
        if not self.episode_files:
            raise FileNotFoundError(f"未找到任何episode文件: {self.local_dir}")
        
        logging.info(f"找到 {len(self.episode_files)} 个episode文件")

    def load_episode(self, episode_index: int):
        """
        加载指定episode的数据
        :param episode_index: episode索引
        :return: 加载的数据字典
        """
        if episode_index >= len(self.episode_files):
            raise IndexError(f"episode索引 {episode_index} 超出范围 (0-{len(self.episode_files)-1})")
        
        file_path = self.episode_files[episode_index]
        logging.info(f"加载episode {episode_index}: {file_path}")
        
        try:
            data = load_bson(file_path)
            logging.info(f"成功加载episode {episode_index}, 包含 {len(data['data'][next(iter(data['data']))])} 帧")
            return data
        except Exception as e:
            logging.error(f"加载episode {episode_index} 失败: {str(e)}")
            raise

    def replay_episode(
        self, 
        episode_index: int, 
        fps: Optional[int] = None,
        show_cameras: bool = True,
        num_rollouts: int = 1
    ):
        """
        重放指定episode的数据
        :param episode_index: episode索引
        :param fps: 回放帧率 (None表示尽可能快)
        :param show_cameras: 是否显示摄像头数据
        :param num_rollouts: 重复播放次数
        """
        data = self.load_episode(episode_index)
        frame_count = len(data['data'][next(iter(data['data']))])
        
        for rollout in range(num_rollouts):
            logging.info(f"开始重放episode {episode_index} (第 {rollout+1}/{num_rollouts} 次)")
            
            # 显示episode信息
            print(colored(f"\nEpisode {episode_index} 信息:", 'cyan'))
            print(f"总帧数: {frame_count}")
            print(f"录制时间: {data['timestamp']}")
            print(f"设备信息: {data['metadata']['station_id']}")
            print(f"版本: {data['metadata']['version']}")
            
            # 获取所有图像键
            image_keys = [k for k in data['data'] if 'image' in k]
            
            # 创建进度条
            pbar = tqdm(total=frame_count, desc=f"重放episode {episode_index}")
            
            for frame_idx in range(frame_count):
                start_time = time.perf_counter()
                
                # 显示图像数据
                if show_cameras:
                    for key in image_keys:
                        frame_data = data['data'][key][frame_idx]
                        if isinstance(frame_data['data'], np.ndarray):
                            img = frame_data['data']
                            cv2.imshow(key.split('/')[-1], img)
                    
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
            
            # 每次重放结束后清理窗口
            if show_cameras and rollout < num_rollouts - 1:
                cv2.destroyAllWindows()
        
        if show_cameras:
            cv2.destroyAllWindows()
        logging.info(f"完成episode {episode_index}的重放")

def main():
    parser = argparse.ArgumentParser(description='机器人数据重播脚本')
    parser.add_argument('--root', type=str, default='data',
                       help='数据根目录 (默认: data)')
    parser.add_argument('--repo-id', type=str, default='raw/example',
                       help='数据集ID (默认: raw/example)')
    parser.add_argument('--start-episode', type=int, default=0,
                       help='开始重放的episode索引 (默认: 0)')
    parser.add_argument('--num-episodes', type=int, default=1,
                       help='要重放的episode数量 (默认: 1)')
    parser.add_argument('--fps', type=int, default=None,
                       help='重放帧率 (None表示尽可能快，默认: None)')
    parser.add_argument('--num-rollouts', type=int, default=1,
                       help='每个episode的重放次数 (默认: 1)')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示摄像头画面')
    
    args = parser.parse_args()
    
    try:
        replayer = DataReplayer(args.root, args.repo_id)
        
        for ep_idx in range(args.start_episode, 
                          args.start_episode + args.num_episodes):
            replayer.replay_episode(
                episode_index=ep_idx,
                fps=args.fps,
                show_cameras=not args.no_display,
                num_rollouts=args.num_rollouts
            )
            
    except Exception as e:
        logging.error(f"重放过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
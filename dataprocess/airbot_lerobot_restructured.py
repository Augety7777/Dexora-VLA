#!/usr/bin/env python3
"""
Airbot数据转换器 - 重构版
支持4个独立LeRobot数据集的生成，包含neck/head数据和GPU编码

数据集架构:
- airbot_pick_and_place: 132 actions
- airbot_articulation: 19 actions  
- airbot_assemble: 17 actions
- airbot_dexterous: 18 actions

改进功能:
- 4个独立数据集的并行处理
- Neck/head关节数据提取
- GPU视频编码(h264_nvenc)
- 多进程并行处理
- 硬件资源监控
- 断点续传支持
"""
import json
import os
import sys
import numpy as np
import struct
import bson
import cv2
import tempfile
import shutil
import psutil
import threading
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
import datetime
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool, Manager, Lock, Process
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES

# 配置导入：
# 1. 优先使用同目录下的 airbot_config_restructured（相对导入，适配当前仓库）
# 2. 再尝试顶层模块
# 3. 再尝试 air_script 包
try:
    from .airbot_config_restructured import AirbotRestructuredConfig, AirbotDatasetConfig
except ImportError:
try:
    from airbot_config_restructured import AirbotRestructuredConfig, AirbotDatasetConfig
except ImportError:
        try:
            from air_script.airbot_config_restructured import (
                AirbotRestructuredConfig,
                AirbotDatasetConfig,
            )
        except ImportError:
            # 在我们使用自定义配置（AirbotFinal1129Config）时，
            # 不需要依赖默认的 AirbotRestructuredConfig，因此这里允许缺失。
            AirbotRestructuredConfig = None  # type: ignore
            AirbotDatasetConfig = None  # type: ignore


class HardwareResourceMonitor:
    """硬件资源监控器 - 实时监控CPU、内存、磁盘使用情况"""
    
    def __init__(self, config: AirbotRestructuredConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.monitoring = False
        self.monitor_thread = None
        self.warning_count = 0
        
    def check_resources(self) -> Tuple[bool, str]:
        """检查硬件资源是否满足要求"""
        issues = []
        
        # 检查内存
        memory = psutil.virtual_memory()
        free_memory_gb = memory.available / (1024**3)
        min_memory = 8.0  # 8GB最低要求
        if free_memory_gb < min_memory:
            issues.append(f"可用内存不足: {free_memory_gb:.2f}GB < {min_memory}GB")
        
        # 检查磁盘空间
        output_disk = psutil.disk_usage(str(self.config.output_root))
        free_disk_gb = output_disk.free / (1024**3)
        min_disk = 50.0  # 50GB最低要求
        if free_disk_gb < min_disk:
            issues.append(f"磁盘空间不足: {free_disk_gb:.2f}GB < {min_disk}GB")
        
        # 检查CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            issues.append(f"CPU使用率过高: {cpu_percent:.1f}% > 90%")
        
        if issues:
            return False, "; ".join(issues)
        return True, "资源充足"
    
    def get_recommended_workers(self) -> int:
        """根据当前硬件状态推荐worker数量"""
        cpu_count = psutil.cpu_count(logical=False) or 4
        memory = psutil.virtual_memory()
        free_memory_gb = memory.available / (1024**3)
        
        # 基于CPU核心数
        workers_by_cpu = max(1, cpu_count - 2)
        
        # 基于可用内存（假设每个worker需要2GB）
        workers_by_memory = max(1, int(free_memory_gb / 2))
        
        # 取较小值
        recommended = min(workers_by_cpu, workers_by_memory, 16)
        
        self.logger.info(f"硬件状态: CPU核心={cpu_count}, 可用内存={free_memory_gb:.1f}GB")
        self.logger.info(f"推荐worker数量: {recommended}")
        
        return recommended


class CategoryDatasetManager:
    """分类数据集管理器 - 管理单个category的LeRobot数据集"""
    
    def __init__(
        self,
        category_config: AirbotDatasetConfig,
        main_config: AirbotRestructuredConfig,
        logger: logging.Logger,
        action_instructions: Dict[str, List[str]]  # 新增：action到instructions的映射
    ):
        self.category_config = category_config
        self.main_config = main_config
        self.logger = logger
        self.action_instructions = action_instructions  # 保存instructions映射
        self.dataset = None
        
        # Episode到instruction的映射
        self.episode_instruction_mapping = []
        
        # 当前category的统计
        self.stats = {
            'total_episodes': 0,
            'processed_episodes': 0,
            'failed_episodes': 0
        }
        
    def setup_dataset(self) -> LeRobotDataset:
        """创建当前category的LeRobot数据集"""
        category = self.category_config.category
        repo_id = self.category_config.repo_id
        output_dir = self.category_config.output_dir
        
        self.logger.info(f"=== 创建 {category} 数据集 ===")
        self.logger.info(f"  Repo ID: {repo_id}")
        self.logger.info(f"  输出目录: {output_dir}")
        self.logger.info(f"  Actions数量: {len(self.category_config.actions)}")
        
        # 创建数据集（LeRobotDataset.create会自动创建metadata）
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_dir,
            fps=self.main_config.fps,
            robot_type=self.main_config.robot_type,
            features=self.main_config.features,
            use_videos=True,
            video_backend=self.main_config.video_backend
        )
        
        # 预先注册所有actions作为tasks (使用action名称而不是instruction)
        # 收集该category下所有actions的action名称
        self.logger.info("  预注册actions作为tasks...")
        registered_tasks = set()
        for action_id in self.category_config.actions:
            # 从action.jsonl获取action名称
            action_name = self.main_config.get_action_name(action_id)
            if action_name and action_name not in registered_tasks:
                self.dataset.meta.add_task(action_name)
                registered_tasks.add(action_name)
        
        self.logger.info(f"  已注册 {len(registered_tasks)} 个tasks (action names)")
        self.logger.info(f"{category} 数据集创建成功")
        return self.dataset
    
    def add_episode_mapping(self, episode_index: int, action_id: str, instruction: str, 
                           source_episode_path: str = "", output_episode_path: str = ""):
        """添加episode到instruction的映射（实时写入文件）"""
        action_name = self.main_config.get_action_name(action_id)
        
        mapping_entry = {
            "episode_index": episode_index,
            "action_id": action_id,
            "action_name": action_name,
            "instruction": instruction,
            "category": self.category_config.category,
            "source_episode_path": source_episode_path,
            "output_episode_path": output_episode_path
        }
        
        # 添加到内存列表（用于统计）
        self.episode_instruction_mapping.append(mapping_entry)
        
        # 实时追加写入文件
        output_dir = Path(self.category_config.output_dir)
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        mapping_file = meta_dir / "episode_instruction_mapping.jsonl"
        
        with open(mapping_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(mapping_entry, ensure_ascii=False) + '\n')
    
    def update_episode_instruction(self, episode_index: int, instruction: str):
        """更新episodes.jsonl中指定episode的tasks字段为instruction文本"""
        meta_dir = Path(self.category_config.output_dir) / "meta"
        episodes_file = meta_dir / "episodes.jsonl"
        
        if not episodes_file.exists():
            return
        
        # 读取所有episodes
        episodes = []
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        
        # 更新指定episode的tasks字段
        for episode in episodes:
            if episode['episode_index'] == episode_index:
                episode['tasks'] = [instruction]
                break
        
        # 写回文件
        with open(episodes_file, 'w', encoding='utf-8') as f:
            for episode in episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    def save_episode_mapping(self):
        """确认episode_instruction_mapping.jsonl已保存（实时写入模式下仅用于日志）"""
        output_dir = Path(self.category_config.output_dir)
        meta_dir = output_dir / "meta"
        mapping_file = meta_dir / "episode_instruction_mapping.jsonl"
        
        # 文件已经在每次add_episode_mapping时实时写入
        if mapping_file.exists():
            self.logger.info(f"Episode mapping文件: {mapping_file}")
            self.logger.info(f"  总计: {len(self.episode_instruction_mapping)} episodes")
        else:
            self.logger.warning(f"Episode mapping文件不存在: {mapping_file}")


class AirbotLeRobotRestructuredProcessor:
    """Airbot数据处理器 - 重构版，支持4个独立数据集"""
    
    def __init__(self, config: AirbotRestructuredConfig):
        self.config = config
        self.setup_logging()
        
        # 全局统计
        self.global_stats = {
            'total_episodes': 0,
            'processed_episodes': 0,
            'failed_episodes': 0,
            'skipped_episodes': 0
        }
        
        # 加载instruction文件（必须在创建manager之前）
        self._load_action_instructions()
        
        # 为每个category创建数据集管理器（传递action_instructions）
        self.category_managers = {}
        for dataset_config in self.config.dataset_configs:
            manager = CategoryDatasetManager(
                dataset_config, 
                self.config, 
                self.logger,
                self.action_instructions  # 传递instructions映射
            )
            self.category_managers[dataset_config.category] = manager
        
        # 硬件资源监控
        self.hardware_monitor = HardwareResourceMonitor(config, self.logger)
        
        # 动态调整worker数量
        self._adjust_worker_count()
    
    def setup_logging(self):
        """配置日志"""
        log_dir = self.config.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"restructured_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AirbotRestructured")
        self.logger.info("=== Airbot LeRobot Restructured Processor ===")
    
    def _load_action_instructions(self):
        """加载所有action的instruction文件（每个action有5个instruction）"""
        self.logger.info("加载instruction文件...")
        self.action_instructions = {}  # action_id -> list of 5 instructions
        
        for action_id in self.config.get_all_action_ids():
            action_dir = self.config.source_root / action_id
            
            if not action_dir.exists():
                self.logger.warning(f"Action目录不存在: {action_dir}")
                continue
            
            # 读取5个instruction文件
            instructions = []
            for i in range(1, 6):
                instruction_file = action_dir / f"instruction{i}.txt"
                
                if instruction_file.exists():
                    try:
                        with open(instruction_file, encoding='utf-8') as f:
                            instruction = f.read().strip()
                        instructions.append(instruction)
                    except Exception as e:
                        self.logger.warning(f"加载instruction失败 {instruction_file}: {e}")
                        instructions.append(f"Instruction {i} for {action_id}")
                else:
                    self.logger.warning(f"Instruction文件不存在: {instruction_file}")
                    instructions.append(f"Instruction {i} for {action_id}")
            
            if instructions:
                self.action_instructions[action_id] = instructions
        
        self.logger.info(f"加载了 {len(self.action_instructions)} 个action的instructions")
    
    def get_instruction_for_episode(self, action_id: str, episode_index_in_action: int, total_episodes_in_action: int) -> str:
        """根据episode在action中的索引获取对应的instruction
        
        每个action有5个instruction，episode会按一定规则分配到这5个instruction:
        - 前50个episode: 按比例均匀分配
        - 超过50个的: 随机分配
        
        Args:
            action_id: action ID
            episode_index_in_action: episode在该action中的索引（从0开始）
            total_episodes_in_action: 该action总共的episode数量
            
        Returns:
            instruction文本
        """
        instructions = self.action_instructions.get(action_id, [])
        if not instructions:
            return f"Default instruction for {action_id}"
        
        num_instructions = len(instructions)
        
        # 前50个episode按比例分配
        if episode_index_in_action < 50:
            # 计算每个instruction应该分配多少个episode
            base_count = 50 // num_instructions
            remainder = 50 % num_instructions
            
            # 构建每个instruction的分配区间
            current_pos = 0
            for inst_idx in range(num_instructions):
                # 当前instruction分配的数量
                count = base_count + (1 if inst_idx < remainder else 0)
                
                if current_pos <= episode_index_in_action < current_pos + count:
                    return instructions[inst_idx]
                
                current_pos += count
            
            # Fallback
            return instructions[0]
        else:
            # 超过50个的随机分配
            import random
            random.seed(episode_index_in_action)  # 使用episode索引作为seed，确保可复现
            return random.choice(instructions)
    
    def _adjust_worker_count(self):
        """根据硬件状况调整worker数量"""
        self.logger.info("=== 硬件配置检查 ===")
        
        ok, msg = self.hardware_monitor.check_resources()
        if not ok:
            self.logger.warning(f"硬件资源警告: {msg}")
        
        recommended = self.hardware_monitor.get_recommended_workers()
        
        if self.config.num_workers > recommended:
            self.logger.warning(
                f"配置的worker数量({self.config.num_workers})超过推荐值({recommended})，已调整"
            )
            self.config.num_workers = recommended
        
        self.logger.info(f"最终worker配置: {self.config.num_workers}")
    
    def setup_all_datasets(self):
        """创建所有category的数据集"""
        self.logger.info("=== 创建所有数据集 ===")
        
        for category, manager in self.category_managers.items():
            try:
                manager.setup_dataset()
            except Exception as e:
                self.logger.error(f"创建数据集失败 {category}: {e}", exc_info=True)
                raise
        
        self.logger.info("所有数据集创建完成")
    
    def extract_robot_joint_data(self, robot_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        从机器人BSON数据中提取关节数据 - 包含head/spine
        
        返回:
            {
                'left_arm_obs': (N, 6),
                'right_arm_obs': (N, 6),
                'left_arm_action': (N, 6),
                'right_arm_action': (N, 6),
                'head_obs': (N, 2),
                'head_action': (N, 2),
                'spine_obs': (N, 1),
                'spine_action': (N, 1)
            }
        """
        joint_data = {}
        
        try:
            data_section = robot_data.get('data', {})
            
            # 提取左臂关节数据
            left_arm_joint_obs = data_section.get('/observation/left_arm/joint_state', [])
            left_arm_joint_action = data_section.get('/action/left_arm/joint_state', [])
            
            # 提取右臂关节数据
            right_arm_joint_obs = data_section.get('/observation/right_arm/joint_state', [])
            right_arm_joint_action = data_section.get('/action/right_arm/joint_state', [])
            
            # 处理机械臂数据
            for key, data_list in [
                ('left_arm_obs', left_arm_joint_obs),
                ('left_arm_action', left_arm_joint_action),
                ('right_arm_obs', right_arm_joint_obs),
                ('right_arm_action', right_arm_joint_action)
            ]:
                if data_list:
                    positions = []
                    for frame_data in data_list:
                        if 'data' in frame_data and 'pos' in frame_data['data']:
                            pos_data = frame_data['data']['pos']
                            if len(pos_data) >= 6:
                                positions.append(pos_data[:6])
                            else:
                                padded_pos = pos_data + [0.0] * (6 - len(pos_data))
                                positions.append(padded_pos)
                    joint_data[key] = np.array(positions)
            
            # 提取head数据 (2维)
            head_joint_obs = data_section.get('/observation/head/joint_state', [])
            head_joint_action = data_section.get('/action/head/joint_state', [])
            
            # 处理head数据 - 保持2维 [value1, value2]
            for key, data_list in [
                ('head_obs', head_joint_obs),
                ('head_action', head_joint_action)
            ]:
                if data_list:
                    head_positions = []
                    
                    for frame_data in data_list:
                        if 'data' in frame_data and 'pos' in frame_data['data']:
                            pos_data = frame_data['data']['pos']
                            if len(pos_data) >= 2:
                                # head包含2个值
                                head_positions.append(pos_data[:2])
                            else:
                                # 使用默认值
                                head_positions.append([0.0, -1.0])
                    
                    joint_data[key] = np.array(head_positions)
                else:
                    # 如果没有head数据，使用默认值
                    num_frames = len(joint_data.get('left_arm_obs', []))
                    if num_frames > 0:
                        default_head = np.full((num_frames, 2), [0.0, -1.0])
                        joint_data[key] = default_head
            
            # 提取spine数据 (1维)
            spine_joint_obs = data_section.get('/observation/spine/joint_state', [])
            spine_joint_action = data_section.get('/action/spine/joint_state', [])
            
            # 处理spine数据 - 保持1维
            for key, data_list in [
                ('spine_obs', spine_joint_obs),
                ('spine_action', spine_joint_action)
            ]:
                if data_list:
                    spine_positions = []
                    
                    for frame_data in data_list:
                        if 'data' in frame_data and 'pos' in frame_data['data']:
                            pos_data = frame_data['data']['pos']
                            if len(pos_data) >= 1:
                                # spine只有1个值
                                spine_positions.append([pos_data[0]])
                            else:
                                # 使用默认值
                                spine_positions.append([0.15])
                    
                    joint_data[key] = np.array(spine_positions)
                else:
                    # 如果没有spine数据，使用默认值
                    num_frames = len(joint_data.get('left_arm_obs', []))
                    if num_frames > 0:
                        default_spine = np.full((num_frames, 1), 0.15)
                        joint_data[key] = default_spine
            
        except Exception as e:
            self.logger.error(f"提取机械臂关节数据失败: {e}", exc_info=True)
        
        return joint_data
    
    def extract_hand_data(self, hand_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """从灵巧手BSON数据中提取手部数据"""
        hand_joint_data = {}
        
        try:
            frames = hand_data.get('frames', [])
            
            left_hand_actions = []
            right_hand_actions = []
            left_hand_obs = []
            right_hand_obs = []
            
            for frame in frames:
                action_data = frame.get('action', {})
                left_action = action_data.get('left_hand', [])
                right_action = action_data.get('right_hand', [])
                
                left_action = self.pad_or_truncate(left_action, 12)
                right_action = self.pad_or_truncate(right_action, 12)
                
                left_hand_actions.append(left_action)
                right_hand_actions.append(right_action)
                
                obs_data = frame.get('observation', {})
                left_obs = obs_data.get('left_hand', [])
                right_obs = obs_data.get('right_hand', [])
                
                left_obs = self.pad_or_truncate(left_obs, 12)
                right_obs = self.pad_or_truncate(right_obs, 12)
                
                # 将手部observation从角度转换为弧度
                left_obs = [np.deg2rad(angle) for angle in left_obs]
                right_obs = [np.deg2rad(angle) for angle in right_obs]
                
                left_hand_obs.append(left_obs)
                right_hand_obs.append(right_obs)
            
            hand_joint_data['left_hand_action'] = np.array(left_hand_actions)
            hand_joint_data['right_hand_action'] = np.array(right_hand_actions)
            hand_joint_data['left_hand_obs'] = np.array(left_hand_obs)
            hand_joint_data['right_hand_obs'] = np.array(right_hand_obs)
            
        except Exception as e:
            self.logger.error(f"提取灵巧手数据失败: {e}", exc_info=True)
        
        return hand_joint_data
    
    def pad_or_truncate(self, data: List[float], target_length: int) -> List[float]:
        """填充或截断数据到目标长度"""
        if len(data) < target_length:
            return data + [0.0] * (target_length - len(data))
        return data[:target_length]
    
    def load_single_image(self, img_path: str) -> Optional[np.ndarray]:
        """加载单个图像（用于并行）"""
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # 转换BGR到RGB
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.warning(f"加载图像失败 {img_path}: {e}")
        return None
    
    def load_images_from_folders(self, episode_path: str, action_id: str) -> Dict[str, List[np.ndarray]]:
        """从episode的相机文件夹中加载图像（优化版：并行加载）"""
        images_by_camera = {}
        
        try:
            # 使用配置的并行参数
            use_parallel = self.config.enable_parallel_loading
            max_workers = self.config.image_load_workers if use_parallel else 1
            
            # 遍历配置的相机文件夹
            for folder_name, lerobot_name in self.config.camera_mapping.items():
                camera_dir = os.path.join(episode_path, folder_name)
                
                if not os.path.exists(camera_dir):
                    self.logger.warning(f"相机文件夹不存在: {camera_dir}")
                    continue
                
                # 获取所有图像文件路径（排序）
                image_files = sorted([f for f in os.listdir(camera_dir) 
                                     if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))])
                image_paths = [os.path.join(camera_dir, f) for f in image_files]
                
                if not image_paths:
                    continue
                
                # 并行加载图像
                camera_frames = []
                if use_parallel and len(image_paths) > 10:  # 少量图像不值得并行
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # 提交所有加载任务
                        future_to_idx = {executor.submit(self.load_single_image, path): idx 
                                        for idx, path in enumerate(image_paths)}
                        
                        # 按顺序收集结果
                        results = [None] * len(image_paths)
                        for future in as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            results[idx] = future.result()
                        
                        # 过滤掉None
                        camera_frames = [img for img in results if img is not None]
                else:
                    # 串行加载（少量图像或禁用并行）
                    for img_path in image_paths:
                        img = self.load_single_image(img_path)
                        if img is not None:
                            camera_frames.append(img)
                
                if camera_frames:
                    images_by_camera[lerobot_name] = camera_frames
                    self.logger.debug(f"  {lerobot_name}: {len(camera_frames)} frames")
        
        except Exception as e:
            self.logger.error(f"加载图像失败 {episode_path}: {e}", exc_info=True)
        
        return images_by_camera
    
    def process_single_episode(
        self,
        action_id: str,
        episode_name: str,
        category: str,
        episode_index_in_action: int,
        total_episodes_in_action: int
    ) -> bool:
        """处理单个episode
        
        Args:
            action_id: action ID (如 'action1')
            episode_name: episode文件夹名 (如 'episode_0')
            category: 所属category (如 'pick_and_place')
            episode_index_in_action: 该episode在action中的索引(从0开始)
            total_episodes_in_action: 该action总共的episode数量
        """
        try:
            episode_path = self.config.source_root / action_id / episode_name
            
            # 加载数据文件(使用实际的文件名)
            robot_bson = episode_path / self.config.robot_bson_name  # episode_0.bson
            hand_bson = episode_path / self.config.hand_bson_name  # xhand_control_data.bson
            
            if not all([robot_bson.exists(), hand_bson.exists()]):
                self.logger.warning(f"数据文件缺失: {episode_path}")
                return False
            
            # 提取关节数据
            with open(robot_bson, 'rb') as f:
                robot_data = bson.decode(f.read())
            joint_data = self.extract_robot_joint_data(robot_data)
            
            # 提取手部数据
            with open(hand_bson, 'rb') as f:
                hand_data = bson.decode(f.read())
            hand_joint_data = self.extract_hand_data(hand_data)
            
            # 加载图像（从文件夹）
            images_by_camera = self.load_images_from_folders(str(episode_path), action_id)
            
            # 获取帧数
            num_frames = len(joint_data.get('left_arm_obs', []))
            if num_frames == 0:
                self.logger.warning(f"Episode没有有效帧: {episode_path}")
                return False
            
            # 合并observation状态: 左臂(6) + 右臂(6) + 左手(12) + 右手(12) + head(2) + spine(1) = 39
            observation_states = []
            for i in range(num_frames):
                state = np.concatenate([
                    joint_data['left_arm_obs'][i],      # 6
                    joint_data['right_arm_obs'][i],     # 6
                    hand_joint_data['left_hand_obs'][i],  # 12
                    hand_joint_data['right_hand_obs'][i], # 12
                    joint_data['head_obs'][i],          # 2
                    joint_data['spine_obs'][i]          # 1
                ])
                observation_states.append(state)
            
            # 合并action: 同样的维度
            actions = []
            for i in range(num_frames):
                action = np.concatenate([
                    joint_data['left_arm_action'][i],
                    joint_data['right_arm_action'][i],
                    hand_joint_data['left_hand_action'][i],
                    hand_joint_data['right_hand_action'][i],
                    joint_data['head_action'][i],
                    joint_data['spine_action'][i]
                ])
                actions.append(action)
            
            # 获取对应的数据集管理器
            manager = self.category_managers[category]
            dataset = manager.dataset
            
            # 获取action名称（用于task_index）
            action_name = self.config.get_action_name(action_id)
            
            # 获取instruction（根据episode索引从5个instruction中选择）
            # instruction仅用于episodes.jsonl的tasks字段
            instruction = self.get_instruction_for_episode(
                action_id, 
                episode_index_in_action, 
                total_episodes_in_action
            )
            
            # 获取当前episode的索引（基于已处理的episode数量）
            episode_index = manager.stats['processed_episodes']
            
            for frame_idx in range(num_frames):
                frame_data = {
                    "observation.state": observation_states[frame_idx].astype(np.float32),
                    "action": actions[frame_idx].astype(np.float32)
                }
                
                # 添加图像 (camera_mapping的值已经是完整的键名，如 "observation.images.top")
                for lerobot_key, cam_frames in images_by_camera.items():
                    if frame_idx < len(cam_frames):
                        frame_data[lerobot_key] = cam_frames[frame_idx]
                
                # add_frame使用action_name作为task参数（映射到task_index）
                # timestamp由add_frame自动计算
                dataset.add_frame(frame_data, task=action_name, timestamp=frame_idx / self.config.fps)
            
            # 保存episode
            dataset.save_episode()
            
            # 更新episodes.jsonl中的tasks字段为instruction文本
            manager.update_episode_instruction(episode_index, instruction)
            
            # 记录episode mapping (添加source和output路径)
            source_path = str(episode_path)
            output_path = str(Path(self.category_managers[category].category_config.output_dir) / "data" / f"chunk-{episode_index:03d}")
            manager.add_episode_mapping(episode_index, action_id, instruction, source_path, output_path)
            
            # 更新统计
            manager.stats['processed_episodes'] += 1
            self.global_stats['processed_episodes'] += 1
            
            # 清理大对象内存
            del robot_data, joint_data, hand_data, hand_joint_data, images_by_camera
            del observation_states, actions, frame_data
            
            # 每10个episode触发一次垃圾回收
            if manager.stats['processed_episodes'] % 10 == 0:
                import gc
                gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理episode失败 {action_id}/{episode_name}: {e}", exc_info=True)
            return False
    
    def process_all_episodes(self):
        """处理所有episodes - 按category组织，考虑instruction分配"""
        self.logger.info("=== 开始处理所有episodes ===")
        
        # 首先扫描所有episodes，按category分组
        episodes_by_category = defaultdict(list)
        episodes_by_action = defaultdict(list)  # 用于计算每个action的episode总数
        
        for action_id in self.config.get_all_action_ids():
            action_path = self.config.source_root / action_id
            
            if not action_path.exists():
                self.logger.warning(f"Action目录不存在: {action_path}")
                continue
            
            # 获取category
            category = self.config.get_action_category(action_id)
            if not category:
                self.logger.warning(f"无法找到action的category: {action_id}")
                continue
            
            # 扫描所有episodes（排序以确保顺序一致）
            episodes = sorted([d for d in os.listdir(action_path) 
                              if os.path.isdir(os.path.join(action_path, d))])
            
            episodes_by_action[action_id] = episodes
            
            # 添加到category列表（包含episode索引信息）
            for idx, episode_name in enumerate(episodes):
                episodes_by_category[category].append((action_id, episode_name, idx, len(episodes)))
        
        # 打印统计信息
        self.logger.info("=== Episodes分布 ===")
        total_episodes = 0
        for category, episodes in episodes_by_category.items():
            count = len(episodes)
            total_episodes += count
            self.logger.info(f"  {category}: {count} episodes")
        self.logger.info(f"  总计: {total_episodes} episodes")
        
        self.global_stats['total_episodes'] = total_episodes
        
        # 处理每个category
        for category, episode_list in episodes_by_category.items():
            self.logger.info(f"\n=== 处理 {category} ({len(episode_list)} episodes) ===")
            
            # 使用tqdm显示进度
            for action_id, episode_name, episode_idx, total_eps in tqdm(episode_list, desc=category):
                success = self.process_single_episode(
                    action_id, 
                    episode_name, 
                    category,
                    episode_idx,  # episode在action中的索引
                    total_eps     # action总共的episode数
                )
                
                if not success:
                    self.global_stats['failed_episodes'] += 1
                    self.category_managers[category].stats['failed_episodes'] += 1
            
            # 保存当前category的数据集
            manager = self.category_managers[category]
            self.logger.info(f"保存 {category} 数据集...")
            manager.dataset.consolidate()
            manager.save_episode_mapping()
            
            self.logger.info(f"{category} 完成:")
            self.logger.info(f"  成功: {manager.stats['processed_episodes']}")
            self.logger.info(f"  失败: {manager.stats['failed_episodes']}")
    
    def process_single_category(self, category: str):
        """处理单个category的所有episodes（用于多进程）"""
        # 为这个进程创建独立的logger和processor
        import logging
        from pathlib import Path
        
        # 设置独立的logger
        log_file = Path(self.config.log_root) / f"airbot_restructured_{category}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(f"AirbotRestructured_{category}")
        logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        logger.handlers = []
        
        # 文件handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 控制台handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        logger.info(f"=== 开始处理 {category} 数据集 ===")
        
        try:
            # 获取该category的配置
            dataset_config = self.config.get_dataset_config(category)
            
            # 创建category manager（注意参数顺序：logger在action_instructions之前）
            manager = CategoryDatasetManager(dataset_config, self.config, logger, self.action_instructions)
            
            # 创建数据集
            manager.setup_dataset()
            
            # 扫描该category的所有episodes
            episode_list = []
            for action_id in dataset_config.actions:
                action_path = self.config.source_root / action_id
                
                if not action_path.exists():
                    logger.warning(f"Action目录不存在: {action_path}")
                    continue
                
                # 扫描所有episodes
                episodes = sorted([d for d in os.listdir(action_path) 
                                  if os.path.isdir(os.path.join(action_path, d))])
                
                # 添加到列表
                for idx, episode_name in enumerate(episodes):
                    episode_list.append((action_id, episode_name, idx, len(episodes)))
            
            # 如果设置了max_episodes_per_category，则限制处理数量
            if hasattr(self.config, 'max_episodes_per_category') and self.config.max_episodes_per_category:
                original_count = len(episode_list)
                episode_list = episode_list[:self.config.max_episodes_per_category]
                logger.info(f"{category} 限制处理: {len(episode_list)}/{original_count} episodes")
            else:
                logger.info(f"{category} 总episodes: {len(episode_list)}")
            
            # 处理所有episodes
            success_count = 0
            failed_count = 0
            
            for action_id, episode_name, episode_idx, total_eps in tqdm(episode_list, desc=category):
                success = self.process_single_episode_standalone(
                    action_id, 
                    episode_name, 
                    manager,
                    episode_idx,
                    total_eps,
                    logger
                )
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            
            # 保存数据集
            logger.info(f"保存 {category} 数据集...")
            # LeRobotDataset不需要consolidate，数据已经实时写入
            manager.save_episode_mapping()
            
            logger.info(f"{category} 完成:")
            logger.info(f"  成功: {success_count}")
            logger.info(f"  失败: {failed_count}")
            
            return True
        
        except KeyboardInterrupt:
            logger.warning(f"{category} 被用户中断，正在保存已处理的数据...")
            # 即使中断也要保存mapping（已实时写入，仅记录日志）
            try:
                manager.save_episode_mapping()
                logger.info(f"{category} 中断保存完成: {success_count} 成功, {failed_count} 失败")
            except Exception as save_error:
                logger.error(f"保存 {category} 数据时出错: {save_error}")
            raise  # 重新抛出以便主进程知道被中断
            
        except Exception as e:
            logger.error(f"处理 {category} 时出错: {e}", exc_info=True)
            # 发生异常时也尝试保存（已实时写入，仅记录日志）
            try:
                manager.save_episode_mapping()
                logger.info(f"{category} 异常保存完成: {success_count} 成功, {failed_count} 失败")
            except Exception as save_error:
                logger.error(f"保存 {category} 数据时出错: {save_error}")
            return False
    
    def process_single_episode_standalone(self, action_id: str, episode_name: str, 
                                          manager: 'CategoryDatasetManager',
                                          episode_index_in_action: int,
                                          total_episodes_in_action: int,
                                          logger: logging.Logger) -> bool:
        """独立处理单个episode（用于多进程，不依赖self的状态）"""
        try:
            # 这里复用process_single_episode的逻辑，但使用传入的manager和logger
            category = manager.category_config.category
            dataset = manager.dataset
            
            # 构建episode路径
            episode_path = self.config.source_root / action_id / episode_name
            
            # 读取BSON数据
            robot_bson = episode_path / self.config.robot_bson_name
            hand_bson = episode_path / self.config.hand_bson_name
            
            if not robot_bson.exists() or not hand_bson.exists():
                logger.warning(f"BSON文件不完整: {episode_path}")
                return False
            
            # 提取关节数据
            with open(robot_bson, 'rb') as f:
                robot_data = bson.decode(f.read())
            joint_data = self.extract_robot_joint_data(robot_data)
            
            with open(hand_bson, 'rb') as f:
                hand_data = bson.decode(f.read())
            hand_joint_data = self.extract_hand_data(hand_data)
            
            # 加载图像
            images_by_camera = self.load_images_from_folders(str(episode_path), action_id)
            
            # 获取帧数
            num_frames = len(joint_data.get('left_arm_obs', []))
            if num_frames == 0:
                logger.warning(f"Episode没有有效帧: {episode_path}")
                return False
            
            # 合并observation状态
            observation_states = []
            for i in range(num_frames):
                state = np.concatenate([
                    joint_data['left_arm_obs'][i],
                    joint_data['right_arm_obs'][i],
                    hand_joint_data['left_hand_obs'][i],
                    hand_joint_data['right_hand_obs'][i],
                    joint_data['head_obs'][i],
                    joint_data['spine_obs'][i]
                ])
                observation_states.append(state)
            
            # 合并action
            actions = []
            for i in range(num_frames):
                action = np.concatenate([
                    joint_data['left_arm_action'][i],
                    joint_data['right_arm_action'][i],
                    hand_joint_data['left_hand_action'][i],
                    hand_joint_data['right_hand_action'][i],
                    joint_data['head_action'][i],
                    joint_data['spine_action'][i]
                ])
                actions.append(action)
            
            # 获取action名称
            action_name = self.config.get_action_name(action_id)
            
            # 选择instruction
            instruction = self.get_instruction_for_episode(
                action_id, 
                episode_index_in_action, 
                total_episodes_in_action
            )
            
            # 获取当前episode索引
            episode_index = manager.stats['processed_episodes']
            
            # 添加frames
            for frame_idx in range(num_frames):
                frame_data = {
                    "observation.state": observation_states[frame_idx].astype(np.float32),
                    "action": actions[frame_idx].astype(np.float32)
                }
                
                # 添加图像
                for lerobot_key, cam_frames in images_by_camera.items():
                    if frame_idx < len(cam_frames):
                        frame_data[lerobot_key] = cam_frames[frame_idx]
                
                dataset.add_frame(frame_data, task=action_name, timestamp=frame_idx / self.config.fps)
            
            # 保存episode
            dataset.save_episode()
            
            # 更新episodes.jsonl中的tasks字段
            manager.update_episode_instruction(episode_index, instruction)
            
            # 记录mapping (添加source和output路径)
            source_path = str(episode_path)
            output_path = str(Path(manager.category_config.output_dir) / "data" / f"chunk-{episode_index:03d}")
            manager.add_episode_mapping(episode_index, action_id, instruction, source_path, output_path)
            
            # 更新统计
            manager.stats['processed_episodes'] += 1
            
            # 清理大对象内存
            del robot_data, joint_data, hand_data, hand_joint_data, images_by_camera
            del observation_states, actions, frame_data
            
            # 每10个episode触发一次垃圾回收
            if manager.stats['processed_episodes'] % 10 == 0:
                import gc
                gc.collect()
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("用户中断处理")
            raise
        except Exception as e:
            logger.error(f"处理episode失败 {action_id}/{episode_name}: {e}", exc_info=True)
            return False
    
    def print_summary(self):
        """打印处理摘要"""
        self.logger.info("\n" + "="*60)
        self.logger.info("处理完成摘要")
        self.logger.info("="*60)
        
        self.logger.info(f"总Episodes: {self.global_stats['total_episodes']}")
        self.logger.info(f"成功处理: {self.global_stats['processed_episodes']}")
        self.logger.info(f"失败: {self.global_stats['failed_episodes']}")
        
        self.logger.info("\n各数据集详情:")
        for category, manager in self.category_managers.items():
            self.logger.info(f"\n  {category}:")
            self.logger.info(f"    Actions: {len(manager.category_config.actions)}")
            self.logger.info(f"    Episodes: {manager.stats['processed_episodes']}")
            self.logger.info(f"    失败: {manager.stats['failed_episodes']}")
            self.logger.info(f"    输出: {manager.category_config.output_dir}")
        
        self.logger.info("\n" + "="*60)
    
    def run(self, parallel_datasets=True):
        """运行完整的转换流程
        
        Args:
            parallel_datasets: 是否并行处理多个数据集（默认True）
        """
        try:
            self.logger.info("=== Airbot LeRobot Restructured 转换开始 ===")
            
            if parallel_datasets:
                self.logger.info("=== 使用多进程并行处理4个数据集 ===")
                
                # 获取所有categories
                categories = list(self.config.datasets.keys())
                self.logger.info(f"将并行处理: {', '.join(categories)}")
                
                # 为每个category创建独立进程
                processes = []
                for category in categories:
                    p = Process(target=self.process_single_category, args=(category,))
                    p.start()
                    processes.append(p)
                    self.logger.info(f"启动进程处理 {category}")
                
                # 等待所有进程完成
                for p in processes:
                    p.join()
                
                self.logger.info("所有数据集处理完成")
                
            else:
                # 原有的串行处理方式
                self.logger.info("=== 使用单进程串行处理 ===")
                
                # 创建所有数据集
                self.setup_all_datasets()
                
                # 处理所有episodes
                self.process_all_episodes()
                
                # 打印摘要
                self.print_summary()
            
            self.logger.info("=== 转换完成 ===")
            
        except KeyboardInterrupt:
            self.logger.warning("用户中断处理")
        except Exception as e:
            self.logger.error(f"处理失败: {e}", exc_info=True)
            raise


def main():
    """主函数"""
    # 创建配置
    config = AirbotRestructuredConfig()
    
    # 打印配置摘要
    config.print_summary()
    
    # 创建处理器
    processor = AirbotLeRobotRestructuredProcessor(config)
    
    # 运行转换
    processor.run()


if __name__ == "__main__":
    main()

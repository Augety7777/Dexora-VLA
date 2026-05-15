"""
Airbot数据集重构配置 - 支持4个独立数据集（清理版）
"""
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json

@dataclass
class AirbotDatasetConfig:
    """单个数据集的配置"""
    category: str  # 'pick_and_place', 'articulation', 'assemble', 'dexterous'
    actions: List[str]  # 该category包含的action IDs
    output_dir: str  # 输出目录
    repo_id: str  # HuggingFace repo ID


class AirbotRestructuredConfig:
    """
    Airbot数据集重构配置
    
    支持4个独立数据集：
    - airbot_pick_and_place: 132 actions
    - airbot_articulation: 19 actions  
    - airbot_assemble: 17 actions
    - airbot_dexterous: 18 actions
    """
    
    def __init__(self):
        # ============ 基础路径配置 ============
        self.source_data_root = "/media/diy01/246ADCDD27AF9BC6"
        self.source_root = Path(self.source_data_root)
        self.output_data_root = "/media/diy01/246ADCDD27AF9BC6/lerobot_output_restructured"
        self.output_root = Path(self.output_data_root)
        self.log_root = "/media/diy01/246ADCDD27AF9BC6/logs"
        
        # ============ 元数据文件路径 ============
        self.task_jsonl_dir = "/media/diy01/246ADCDD27AF9BC6/task_jsonl"
        self.action_jsonl_path = "/media/diy01/246ADCDD27AF9BC6/action.jsonl"
        
        # ============ BSON文件名配置 ============
        self.robot_bson_name = "episode_0.bson"
        self.hand_bson_name = "xhand_control_data.bson"
        
        # ============ 相机配置 ============
        # 相机文件夹名称
        self.camera_folders = [
            'camera_third_view',
            'camera_left_wrist',
            'camera_right_wrist',
            'camera_head'
        ]
        
        # 相机名称映射：文件夹名 -> LeRobot相机名
        self.camera_mapping = {
            'camera_third_view': 'observation.images.top',
            'camera_left_wrist': 'observation.images.wrist_left',
            'camera_right_wrist': 'observation.images.wrist_right',
            'camera_head': 'observation.images.front'
        }
        
        # ============ 机器人配置 ============
        self.robot_type = "airbot_play"
        
        # 状态维度: 左臂6 + 右臂6 + 左手12 + 右手12 + neck1 + head1 = 38
        self.state_dim = 38
        
        # Neck/Head默认值（当数据缺失时）
        self.default_neck_head = [0.0, -1.0]
        
        # ============ 视频配置 ============
        self.fps = 20  # LeRobot会自动转换为Fraction
        self.video_backend = "pyav"
        self.video_codec = "h264_nvenc"  # GPU编码
        
        # ============ 处理配置 ============
        self.num_workers = 8
        self.queue_size = 10
        
        # ============ 性能优化配置 ============
        self.enable_parallel_loading = True  # 并行加载episode数据
        self.preload_batch_size = 4  # 预加载的episode数量
        self.image_load_workers = 4  # 图像加载的线程数
        
        # ============ 硬件监控 ============
        self.enable_hardware_monitoring = True
        self.min_free_memory_gb = 4.0
        self.min_free_disk_gb = 50.0
        
        # ============ 容错配置 ============
        self.skip_failed_episodes = True
        self.max_episode_retries = 2
        
        # ============ Checkpoint配置 ============
        self.enable_checkpoint = True
        self.checkpoint_interval = 50
        
        # ============ Features配置 ============
        self.features = {
            # 4个相机 (所有相机都是640x480)
            "observation.images.top": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.images.wrist_left": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.images.wrist_right": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            # 状态: 39维 (左臂6 + 右臂6 + 左手12 + 右手12 + head2 + spine1)
            "observation.state": {
                "dtype": "float32",
                "shape": (39,),
                "names": [
                    "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
                    "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
                    "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
                    "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
                    "left_hand_joint_1", "left_hand_joint_2", "left_hand_joint_3",
                    "left_hand_joint_4", "left_hand_joint_5", "left_hand_joint_6",
                    "left_hand_joint_7", "left_hand_joint_8", "left_hand_joint_9",
                    "left_hand_joint_10", "left_hand_joint_11", "left_hand_joint_12",
                    "right_hand_joint_1", "right_hand_joint_2", "right_hand_joint_3",
                    "right_hand_joint_4", "right_hand_joint_5", "right_hand_joint_6",
                    "right_hand_joint_7", "right_hand_joint_8", "right_hand_joint_9",
                    "right_hand_joint_10", "right_hand_joint_11", "right_hand_joint_12",
                    "head_joint_1", "head_joint_2", "spine_joint"
                ]
            },
            # 动作: 39维 (与状态相同)
            "action": {
                "dtype": "float32",
                "shape": (39,),
                "names": [
                    "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
                    "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
                    "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
                    "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
                    "left_hand_joint_1", "left_hand_joint_2", "left_hand_joint_3",
                    "left_hand_joint_4", "left_hand_joint_5", "left_hand_joint_6",
                    "left_hand_joint_7", "left_hand_joint_8", "left_hand_joint_9",
                    "left_hand_joint_10", "left_hand_joint_11", "left_hand_joint_12",
                    "right_hand_joint_1", "right_hand_joint_2", "right_hand_joint_3",
                    "right_hand_joint_4", "right_hand_joint_5", "right_hand_joint_6",
                    "right_hand_joint_7", "right_hand_joint_8", "right_hand_joint_9",
                    "right_hand_joint_10", "right_hand_joint_11", "right_hand_joint_12",
                    "head_joint_1", "head_joint_2", "spine_joint"
                ]
            }
        }
        
        # ============ 加载任务分类和action映射 ============
        self._load_task_categories()
        self._load_action_names()
        
        # ============ 创建4个数据集配置 ============
        self.datasets = self._create_dataset_configs()
        self.dataset_configs = list(self.datasets.values())
    
    def _load_task_categories(self):
        """加载任务分类（从task_jsonl目录）"""
        self.task_categories = {}
        
        category_files = {
            'pick_and_place': 'pick_and_place.json',
            'articulation': 'articulation.json',
            'assemble': 'assemble.json',
            'dexterous': 'dexterous.json'
        }
        
        for category, filename in category_files.items():
            filepath = Path(self.task_jsonl_dir) / filename
            if filepath.exists():
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)
                    # 文件格式: {"actions": ["action1", "action2", ...]}
                    if 'actions' in data:
                        self.task_categories[category] = data['actions']
                    else:
                        self.task_categories[category] = []
    
    def _load_action_names(self):
        """加载action ID到name的映射（从action.jsonl）"""
        self.action_names = {}
        
        with open(self.action_jsonl_path, encoding='utf-8') as f:
            # action.jsonl是一个完整的JSON文件，不是JSONL格式
            data = json.load(f)
            # data is a dict: {"action1": "name1", "action2": "name2", ...}
            self.action_names = data
    
    def _create_dataset_configs(self) -> Dict[str, AirbotDatasetConfig]:
        """创建4个数据集配置"""
        configs = {}
        
        for category, actions in self.task_categories.items():
            config = AirbotDatasetConfig(
                category=category,
                actions=actions,
                output_dir=str(self.output_root / f"airbot_{category}"),
                repo_id=f"RoboCoin-BAAI/airbot_{category}"
            )
            configs[category] = config
        
        return configs
    
    def get_action_category(self, action_id: str) -> str:
        """获取action所属的category"""
        for category, actions in self.task_categories.items():
            if action_id in actions:
                return category
        return None
    
    def get_action_name(self, action_id: str) -> str:
        """获取action的名称"""
        return self.action_names.get(action_id, action_id)
    
    def get_dataset_config(self, category: str) -> AirbotDatasetConfig:
        """获取指定category的数据集配置"""
        return self.datasets.get(category)
    
    def get_all_action_ids(self) -> List[str]:
        """获取所有action IDs"""
        all_actions = []
        for actions in self.task_categories.values():
            all_actions.extend(actions)
        return all_actions
    
    def print_summary(self):
        """打印配置摘要"""
        print("=" * 70)
        print("Airbot数据集重构配置")
        print("=" * 70)
        print(f"源数据路径: {self.source_data_root}")
        print(f"输出路径: {self.output_data_root}")
        print()
        print("数据集划分:")
        
        total = 0
        for category, actions in self.task_categories.items():
            count = len(actions)
            total += count
            print(f"  - {category:20s}: {count:3d} actions")
        
        print()
        print(f"总计: {total} actions")
        print()
        print(f"视频编码: GPU ({self.video_codec})")
        print(f"并行worker数: {self.num_workers}")
        print("=" * 70)
        
        # 打印几个示例映射
        print()
        print("示例action映射:")
        sample_actions = ['action1', 'action38', 'action87', 'action133']
        for action_id in sample_actions:
            category = self.get_action_category(action_id)
            action_name = self.get_action_name(action_id)
            if category:
                print(f"  {action_id} → {category:20s} → {action_name}")


if __name__ == "__main__":
    config = AirbotRestructuredConfig()
    config.print_summary()

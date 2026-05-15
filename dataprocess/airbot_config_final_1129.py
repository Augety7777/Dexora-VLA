"""
专门用于转换 `data/ours/final_11.29` 路径下数据的配置文件。

与 `airbot_config_restructured.AirbotRestructuredConfig` 保持接口兼容，
以便可以直接复用 `airbot_lerobot_restructured.AirbotLeRobotRestructuredProcessor`。
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json


@dataclass
class AirbotDatasetConfig:
    """单个数据集的配置（与原版保持一致接口）"""

    category: str  # 'pick_and_place', 'articulation', 'assemble', 'dexterous'
    actions: List[str]  # 该 category 包含的 action IDs
    output_dir: str  # 输出目录
    repo_id: str  # HuggingFace repo ID（这里主要占位，不一定实际上传）


class AirbotFinal1129Config:
    """
    面向 `data/ours/final_11.29` 的配置：

    - 源数据：repo_root / data / ours / final12.2
    - 类别划分：repo_root / data / ours / final/{pick_and_place,articulation,assemble,dexterous}.json
    - action → task 名：repo_root / dataprocess / tasks.json
    - 输出：repo_root / data / ours / lerobot_output_final_12_2/{airbot_pick_and_place,...}
    """

    def __init__(self):
        # 推断仓库根目录（当前文件在 Dex-RDT/dataprocess/ 下）
        self.repo_root = Path(__file__).resolve().parents[1]

        # ============ 基础路径配置 ============
        self.source_root = self.repo_root / "data" / "ours" / "final12.2"
        self.output_root = self.repo_root / "data" / "ours" / "lerobot_output_final_12_2"
        self.log_root = self.repo_root / "data" / "ours" / "lerobot_output_final_12_2" / "logs"

        # ============ 元数据 / 映射文件路径 ============
        # 类别划分 JSON（action 列表）
        self.task_jsonl_dir = self.repo_root / "data" / "ours" / "final"
        # action1-200 对应的 task 文本列表
        self.tasks_path = self.repo_root / "dataprocess" / "tasks.json"

        # ============ BSON 文件名配置 ============
        self.robot_bson_name = "episode_0.bson"
        self.hand_bson_name = "xhand_control_data.bson"

        # ============ 相机配置 ============
        self.camera_folders = [
            "camera_third_view",
            "camera_left_wrist",
            "camera_right_wrist",
            "camera_head",
        ]

        # 文件夹名 -> LeRobot 特征 key
        self.camera_mapping: Dict[str, str] = {
            "camera_third_view": "observation.images.top",
            "camera_left_wrist": "observation.images.wrist_left",
            "camera_right_wrist": "observation.images.wrist_right",
            "camera_head": "observation.images.front",
        }

        # ============ 机器人配置 ============
        self.robot_type = "airbot_play"

        # 关节状态维度：与 `airbot_lerobot_restructured.py` 中拼接逻辑保持一致
        # 左臂6 + 右臂6 + 左手12 + 右手12 + head2 + spine1 = 39
        self.state_dim = 39

        # head/spine 默认值（当数据缺失时在处理器中会用到）
        self.default_neck_head = [0.0, -1.0]

        # ============ 视频配置 ============
        self.fps = 20  # LeRobot 会自动转换为 Fraction
        self.video_backend = "pyav"
        # 注意：当前处理器用的是帧序列而非直接视频编码，codec 主要保留接口一致
        self.video_codec = "h264_nvenc"

        # ============ 处理 / 性能配置 ============
        self.num_workers = 8
        self.queue_size = 10

        # 并行加载图像
        self.enable_parallel_loading = True
        self.preload_batch_size = 4
        self.image_load_workers = 4

        # ============ 硬件监控 / 容错 / Checkpoint（接口占位，处理器里部分会使用） ============
        self.enable_hardware_monitoring = True
        self.min_free_memory_gb = 4.0
        self.min_free_disk_gb = 50.0

        self.skip_failed_episodes = True
        self.max_episode_retries = 2

        self.enable_checkpoint = True
        self.checkpoint_interval = 50

        # ============ Features 配置 ============
        # 与 `airbot_config_restructured.AirbotRestructuredConfig` 保持一致
        self.features = {
            # 4 路相机 (640x480)
            "observation.images.top": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_left": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_right": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            # 状态: 39 维 (左臂6 + 右臂6 + 左手12 + 右手12 + head2 + spine1)
            "observation.state": {
                "dtype": "float32",
                "shape": (39,),
                "names": [
                    "left_arm_joint_1",
                    "left_arm_joint_2",
                    "left_arm_joint_3",
                    "left_arm_joint_4",
                    "left_arm_joint_5",
                    "left_arm_joint_6",
                    "right_arm_joint_1",
                    "right_arm_joint_2",
                    "right_arm_joint_3",
                    "right_arm_joint_4",
                    "right_arm_joint_5",
                    "right_arm_joint_6",
                    "left_hand_joint_1",
                    "left_hand_joint_2",
                    "left_hand_joint_3",
                    "left_hand_joint_4",
                    "left_hand_joint_5",
                    "left_hand_joint_6",
                    "left_hand_joint_7",
                    "left_hand_joint_8",
                    "left_hand_joint_9",
                    "left_hand_joint_10",
                    "left_hand_joint_11",
                    "left_hand_joint_12",
                    "right_hand_joint_1",
                    "right_hand_joint_2",
                    "right_hand_joint_3",
                    "right_hand_joint_4",
                    "right_hand_joint_5",
                    "right_hand_joint_6",
                    "right_hand_joint_7",
                    "right_hand_joint_8",
                    "right_hand_joint_9",
                    "right_hand_joint_10",
                    "right_hand_joint_11",
                    "right_hand_joint_12",
                    "head_joint_1",
                    "head_joint_2",
                    "spine_joint",
                ],
            },
            # 动作: 39 维 (与状态相同)
            "action": {
                "dtype": "float32",
                "shape": (39,),
                "names": [
                    "left_arm_joint_1",
                    "left_arm_joint_2",
                    "left_arm_joint_3",
                    "left_arm_joint_4",
                    "left_arm_joint_5",
                    "left_arm_joint_6",
                    "right_arm_joint_1",
                    "right_arm_joint_2",
                    "right_arm_joint_3",
                    "right_arm_joint_4",
                    "right_arm_joint_5",
                    "right_arm_joint_6",
                    "left_hand_joint_1",
                    "left_hand_joint_2",
                    "left_hand_joint_3",
                    "left_hand_joint_4",
                    "left_hand_joint_5",
                    "left_hand_joint_6",
                    "left_hand_joint_7",
                    "left_hand_joint_8",
                    "left_hand_joint_9",
                    "left_hand_joint_10",
                    "left_hand_joint_11",
                    "left_hand_joint_12",
                    "right_hand_joint_1",
                    "right_hand_joint_2",
                    "right_hand_joint_3",
                    "right_hand_joint_4",
                    "right_hand_joint_5",
                    "right_hand_joint_6",
                    "right_hand_joint_7",
                    "right_hand_joint_8",
                    "right_hand_joint_9",
                    "right_hand_joint_10",
                    "right_hand_joint_11",
                    "right_hand_joint_12",
                    "head_joint_1",
                    "head_joint_2",
                    "spine_joint",
                ],
            },
        }

        # ============ 加载任务分类和 action 映射 ============
        self._load_task_categories()
        self._load_action_names_from_tasks()

        # ============ 创建 4 个数据集配置 ============
        self.datasets = self._create_dataset_configs()
        self.dataset_configs = list(self.datasets.values())

    # ------------------------------------------------------------------
    # 元数据加载
    # ------------------------------------------------------------------
    def _load_task_categories(self):
        """从 `data/ours/final/*.json` 读取每个 category 下的 action 列表。"""

        self.task_categories: Dict[str, List[str]] = {}

        category_files = {
            "pick_and_place": "pick_and_place.json",
            "articulation": "articulation.json",
            "assemble": "assemble.json",
            "dexterous": "dexterous.json",
        }

        for category, filename in category_files.items():
            filepath = self.task_jsonl_dir / filename
            if filepath.exists():
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    # 文件格式: {"actions": ["action1", "action2", ...]}
                    if "actions" in data:
                        self.task_categories[category] = data["actions"]
                    else:
                        self.task_categories[category] = []
            else:
                # 如果某个 json 不存在，则该 category 为空
                self.task_categories[category] = []

    def _load_action_names_from_tasks(self):
        """
        使用 dataprocess/tasks.json 中的 task 列表构造:
        action_id -> task_name 的映射。

        约定: tasks[0] 对应 action1, tasks[1] 对应 action2, ...
        """

        with open(self.tasks_path, encoding="utf-8") as f:
            data = json.load(f)
            tasks = data.get("tasks", [])

        self.action_names: Dict[str, str] = {}
        for idx, task_name in enumerate(tasks):
            action_id = f"action{idx + 1}"
            self.action_names[action_id] = task_name

    # ------------------------------------------------------------------
    # 接口: 被处理器调用
    # ------------------------------------------------------------------
    def _create_dataset_configs(self) -> Dict[str, AirbotDatasetConfig]:
        """根据类别映射创建 4 个数据集配置。"""

        configs: Dict[str, AirbotDatasetConfig] = {}

        for category, actions in self.task_categories.items():
            # 输出目录按 category 分开
            output_dir = self.output_root / f"airbot_{category}"
            repo_id = f"RoboCoin-BAAI/airbot_{category}_final_11_29"
            config = AirbotDatasetConfig(
                category=category,
                actions=actions,
                output_dir=str(output_dir),
                repo_id=repo_id,
            )
            configs[category] = config

        return configs

    def get_action_category(self, action_id: str) -> str:
        """获取某个 action 所属的 category。"""

        for category, actions in self.task_categories.items():
            if action_id in actions:
                return category
        return None

    def get_action_name(self, action_id: str) -> str:
        """获取 action 的可读名称（task 名）。"""

        return self.action_names.get(action_id, action_id)

    def get_dataset_config(self, category: str) -> AirbotDatasetConfig:
        """获取指定 category 的数据集配置。"""

        return self.datasets.get(category)

    def get_all_action_ids(self) -> List[str]:
        """获取所有 action IDs（来自各 category 汇总）。"""

        all_actions: List[str] = []
        for actions in self.task_categories.values():
            all_actions.extend(actions)
        return all_actions

    # ------------------------------------------------------------------
    # 调试用: 打印配置摘要
    # ------------------------------------------------------------------
    def print_summary(self):
        print("=" * 70)
        print("Airbot Final_11.29 数据集重构配置")
        print("=" * 70)
        print(f"源数据路径: {self.source_root}")
        print(f"输出路径: {self.output_root}")
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
        print(f"并行 worker 数: {self.num_workers}")
        print("=" * 70)


if __name__ == "__main__":
    config = AirbotFinal1129Config()
    config.print_summary()



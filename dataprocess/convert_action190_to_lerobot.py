#!/usr/bin/env python3
"""
通用 Action 数据转 LeRobot 数据集格式脚本

使用方式:
    python convert_action190_to_lerobot.py --source <输入目录> --output <输出目录>

示例:
    python convert_action190_to_lerobot.py \
        --source /path/to/action264 \
        --output /path/to/output/action264_lerobot


脚本会自动从输入目录名推断 action 名称，无需修改代码。

输入目录结构:
  - episode_xxx/
    - episode_0.bson
    - xhand_control_data.bson
    - camera_third_view/*.jpg|png
    - camera_left_wrist/*.jpg|png
    - camera_right_wrist/*.jpg|png
    - camera_head/*.jpg|png
  - instruction1.txt ... instruction5.txt

依赖:
- numpy, bson, opencv-python, lerobot
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import cv2
import bson

# LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class SingleActionConfig:
    """单动作转换配置 - 自动从输入路径推断 action 名称"""

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        fps: int = 20,
    ):
        self.source_root = Path(source_dir)
        self.output_root = Path(output_dir)
        self.fps = fps

        # 自动从输入目录名推断 action 名称
        self.action_name = self.source_root.name  # 例如 "action264"

        # 文件名与相机映射与现有重构版保持一致
        self.robot_bson_name = "episode_0.bson"
        self.hand_bson_name = "xhand_control_data.bson"

        # 相机映射: 源文件夹名 -> LeRobot键名
        self.camera_mapping: Dict[str, str] = {
            "camera_third_view": "observation.images.top",
            "camera_left_wrist": "observation.images.wrist_left",
            "camera_right_wrist": "observation.images.wrist_right",
            "camera_head": "observation.images.front",
        }

        # LeRobot features（与重构版一致）
        self.features: Dict[str, Dict[str, Any]] = {
            "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.images.wrist_left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.images.wrist_right": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            # 状态/动作: 左臂6 + 右臂6 + 左手12 + 右手12 + head2 + spine1 = 39
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


def setup_logger(output_root: Path, action_name: str) -> logging.Logger:
    """
    为转换过程创建日志记录器。

    注意：不要提前创建 output_root 本身，否则会与 LeRobotDataset.create
    内部对 root.mkdir(exist_ok=False) 的调用冲突，导致 FileExistsError。
    因此日志目录放在 output_root 同级的 logs 子目录中。
    """
    # 日志目录放在 output_root 的同级目录，避免抢先创建数据集根目录
    log_dir = output_root.parent / "logs" / output_root.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"convert_{action_name}.log"

    logger = logging.getLogger(f"Convert_{action_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def pad_or_truncate(data: List[float], target_length: int) -> List[float]:
    if len(data) < target_length:
        return data + [0.0] * (target_length - len(data))
    return data[:target_length]


def extract_robot_joint_data(robot_data: Dict[str, Any], default_head: Tuple[float, float] = (0.0, -1.0)) -> Dict[str, np.ndarray]:
    joint_data: Dict[str, np.ndarray] = {}
    data_section = robot_data.get("data", {})

    # Arms
    def collect_positions(seq: List[Dict[str, Any]], dims: int) -> np.ndarray:
        positions: List[List[float]] = []
        for frame_data in seq or []:
            pos = frame_data.get("data", {}).get("pos", [])
            if len(pos) >= dims:
                positions.append(pos[:dims])
            else:
                positions.append(pos + [0.0] * (dims - len(pos)))
        return np.array(positions) if positions else np.empty((0, dims), dtype=float)

    joint_data["left_arm_obs"] = collect_positions(data_section.get("/observation/left_arm/joint_state", []), 6)
    joint_data["left_arm_action"] = collect_positions(data_section.get("/action/left_arm/joint_state", []), 6)
    joint_data["right_arm_obs"] = collect_positions(data_section.get("/observation/right_arm/joint_state", []), 6)
    joint_data["right_arm_action"] = collect_positions(data_section.get("/action/right_arm/joint_state", []), 6)

    # Head (2)
    def collect_head(seq: List[Dict[str, Any]]) -> np.ndarray:
        positions: List[List[float]] = []
        for frame_data in seq or []:
            pos = frame_data.get("data", {}).get("pos", [])
            if len(pos) >= 2:
                positions.append(pos[:2])
            else:
                positions.append(list(default_head))
        return np.array(positions) if positions else np.empty((0, 2), dtype=float)

    head_obs = collect_head(data_section.get("/observation/head/joint_state", []))
    head_action = collect_head(data_section.get("/action/head/joint_state", []))
    joint_data["head_obs"] = head_obs
    joint_data["head_action"] = head_action

    # Spine (1)
    def collect_spine(seq: List[Dict[str, Any]]) -> np.ndarray:
        positions: List[List[float]] = []
        for frame_data in seq or []:
            pos = frame_data.get("data", {}).get("pos", [])
            if len(pos) >= 1:
                positions.append([pos[0]])
            else:
                positions.append([0.15])
        return np.array(positions) if positions else np.empty((0, 1), dtype=float)

    spine_obs = collect_spine(data_section.get("/observation/spine/joint_state", []))
    spine_action = collect_spine(data_section.get("/action/spine/joint_state", []))
    joint_data["spine_obs"] = spine_obs
    joint_data["spine_action"] = spine_action

    # If head/spine missing frames, align by left_arm frames if available
    num_frames = len(joint_data.get("left_arm_obs", []))
    if num_frames > 0:
        if len(joint_data["head_obs"]) == 0:
            joint_data["head_obs"] = np.full((num_frames, 2), default_head, dtype=float)
        if len(joint_data["head_action"]) == 0:
            joint_data["head_action"] = np.full((num_frames, 2), default_head, dtype=float)
        if len(joint_data["spine_obs"]) == 0:
            joint_data["spine_obs"] = np.full((num_frames, 1), 0.15, dtype=float)
        if len(joint_data["spine_action"]) == 0:
            joint_data["spine_action"] = np.full((num_frames, 1), 0.15, dtype=float)

    return joint_data


def extract_hand_data(hand_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    hand_joint_data: Dict[str, np.ndarray] = {}
    frames = hand_data.get("frames", [])

    left_hand_actions: List[List[float]] = []
    right_hand_actions: List[List[float]] = []
    left_hand_obs: List[List[float]] = []
    right_hand_obs: List[List[float]] = []

    for frame in frames:
        action_data = frame.get("action", {})
        left_action = pad_or_truncate(action_data.get("left_hand", []), 12)
        right_action = pad_or_truncate(action_data.get("right_hand", []), 12)
        left_hand_actions.append(left_action)
        right_hand_actions.append(right_action)

        obs_data = frame.get("observation", {})
        left_obs = pad_or_truncate(obs_data.get("left_hand", []), 12)
        right_obs = pad_or_truncate(obs_data.get("right_hand", []), 12)

        # 角度转弧度
        left_obs = [np.deg2rad(angle) for angle in left_obs]
        right_obs = [np.deg2rad(angle) for angle in right_obs]

        left_hand_obs.append(left_obs)
        right_hand_obs.append(right_obs)

    hand_joint_data["left_hand_action"] = np.array(left_hand_actions, dtype=float)
    hand_joint_data["right_hand_action"] = np.array(right_hand_actions, dtype=float)
    hand_joint_data["left_hand_obs"] = np.array(left_hand_obs, dtype=float)
    hand_joint_data["right_hand_obs"] = np.array(right_hand_obs, dtype=float)

    return hand_joint_data


def load_single_image(img_path: str, logger: logging.Logger) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(img_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            logger.error(f"图像加载返回 None: {img_path}")
    except Exception as e:
        logger.error(f"加载图像异常 {img_path}: {e}")
    return None


def load_images_from_folders(episode_path: Path, camera_mapping: Dict[str, str], logger: logging.Logger) -> tuple:
    """
    加载所有相机图像，并返回图像数据和帧数信息。
    
    返回: (images_by_camera, image_file_counts, load_success_counts)
        - images_by_camera: Dict[str, List[np.ndarray]] 成功加载的图像
        - image_file_counts: Dict[str, int] 每个相机的图像文件数量
        - load_success_counts: Dict[str, int] 每个相机成功加载的图像数量
    """
    images_by_camera: Dict[str, List[np.ndarray]] = {}
    image_file_counts: Dict[str, int] = {}
    load_success_counts: Dict[str, int] = {}
    
    for folder_name, lerobot_name in camera_mapping.items():
        camera_dir = episode_path / folder_name
        if not camera_dir.exists():
            logger.warning(f"相机文件夹不存在: {camera_dir}")
            image_file_counts[lerobot_name] = 0
            load_success_counts[lerobot_name] = 0
            continue
        
        image_files = sorted([f for f in os.listdir(camera_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        image_file_counts[lerobot_name] = len(image_files)
        
        if not image_files:
            logger.warning(f"相机文件夹无图像: {camera_dir}")
            load_success_counts[lerobot_name] = 0
            continue
        
        frames: List[np.ndarray] = []
        failed_images: List[str] = []
        for fname in image_files:
            arr = load_single_image(str(camera_dir / fname), logger)
            if arr is not None:
                frames.append(arr)
            else:
                failed_images.append(fname)
        
        load_success_counts[lerobot_name] = len(frames)
        
        if failed_images:
            logger.error(f"相机 {lerobot_name} 有 {len(failed_images)} 张图像加载失败: {failed_images[:5]}{'...' if len(failed_images) > 5 else ''}")
        
        if frames:
            images_by_camera[lerobot_name] = frames
    
    return images_by_camera, image_file_counts, load_success_counts


def read_instructions(action_dir: Path, action_name: str, logger: logging.Logger) -> List[str]:
    instructions: List[str] = []
    for i in range(1, 6):
        f = action_dir / f"instruction{i}.txt"
        if f.exists():
            try:
                text = f.read_text(encoding="utf-8").strip()
                instructions.append(text)
            except Exception as e:
                logger.warning(f"读取 {f} 失败: {e}")
                instructions.append(f"Instruction {i} for {action_name}")
        else:
            logger.warning(f"缺少 {f}, 将使用占位文本")
            instructions.append(f"Instruction {i} for {action_name}")
    return instructions


def choose_instruction(instructions: List[str], episode_index_in_action: int, total_episodes_in_action: int, action_name: str) -> str:
    if not instructions:
        return f"Default instruction for {action_name}"
    num_instructions = len(instructions)
    if episode_index_in_action < 50:
        base = 50 // num_instructions
        rem = 50 % num_instructions
        pos = 0
        for idx in range(num_instructions):
            count = base + (1 if idx < rem else 0)
            if pos <= episode_index_in_action < pos + count:
                return instructions[idx]
            pos += count
        return instructions[0]
    else:
        import random
        random.seed(episode_index_in_action)
        return random.choice(instructions)


def update_episode_instruction(meta_dir: Path, episode_index: int, instruction: str) -> None:
    episodes_file = meta_dir / "episodes.jsonl"
    if not episodes_file.exists():
        return
    episodes: List[Dict[str, Any]] = []
    with open(episodes_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    for ep in episodes:
        if ep.get("episode_index") == episode_index:
            ep["tasks"] = [instruction]
            break
    with open(episodes_file, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def add_episode_mapping(meta_dir: Path, episode_index: int, source_episode_path: str, output_episode_path: str, instruction: str, action_name: str) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = meta_dir / "episode_instruction_mapping.jsonl"
    entry = {
        "episode_index": episode_index,
        "action_id": action_name,
        "action_name": action_name,
        "instruction": instruction,
        "category": "single_action",
        "source_episode_path": source_episode_path,
        "output_episode_path": output_episode_path,
    }
    with open(mapping_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def convert_action(config: SingleActionConfig, logger: logging.Logger) -> None:
    """通用 action 转换函数，自动使用 config.action_name"""
    action_name = config.action_name
    logger.info(f"开始转换 {action_name}")
    logger.info(f"  源目录: {config.source_root}")
    logger.info(f"  输出目录: {config.output_root}")

    # 基础检查
    if not config.source_root.exists():
        raise FileNotFoundError(f"源目录不存在: {config.source_root}")

    # 读取指令
    instructions = read_instructions(config.source_root, action_name, logger)
    if any(text.startswith("Instruction") for text in instructions):
        logger.warning("instruction1-5.txt 不完整，已用占位文本代替缺失项。")

    # 创建 LeRobot 数据集（repo_id 自动使用 action_name）
    dataset = LeRobotDataset.create(
        repo_id=f"local/airbot_{action_name}",
        root=str(config.output_root),
        fps=config.fps,
        robot_type="airbot_play",
        features=config.features,
        use_videos=True,
        video_backend="pyav",
    )

    # 枚举 episodes
    episode_dirs = sorted([d for d in os.listdir(config.source_root) if d.startswith("episode_") and (config.source_root / d).is_dir()])
    logger.info(f"发现 {len(episode_dirs)} 个 episodes")

    processed = 0
    failed = 0
    for idx, ep_name in enumerate(episode_dirs):
        ep_path = config.source_root / ep_name
        robot_bson = ep_path / config.robot_bson_name
        hand_bson = ep_path / config.hand_bson_name

        # IMPORTANT:
        # If a previous episode failed during `dataset.save_episode()` after it popped keys like "size"/"task",
        # the internal `episode_buffer` can be left in a corrupted state and make subsequent episodes crash
        # immediately with `KeyError: 'size'` in `dataset.add_frame()`.
        # Reset/cleanup the buffer at the start of each episode to avoid cascading failures.
        try:
            if getattr(dataset, "episode_buffer", None) is None:
                dataset.episode_buffer = dataset.create_episode_buffer()
            else:
                dataset.clear_episode_buffer()
        except Exception:
            # Best-effort: don't block conversion due to buffer cleanup issues.
            dataset.episode_buffer = None

        if not robot_bson.exists() or not hand_bson.exists():
            logger.warning(f"缺少 BSON 文件: {ep_path} (需要: {config.robot_bson_name}, {config.hand_bson_name})")
            failed += 1
            continue

        try:
            # 读取 BSON
            with open(robot_bson, "rb") as f:
                robot_data = bson.decode(f.read())
            with open(hand_bson, "rb") as f:
                hand_data = bson.decode(f.read())

            joint_data = extract_robot_joint_data(robot_data)
            hand_joint = extract_hand_data(hand_data)

            # 图像
            images_by_camera, image_file_counts, load_success_counts = load_images_from_folders(ep_path, config.camera_mapping, logger)

            # 获取各数据源帧数
            robot_frames = len(joint_data.get("left_arm_obs", []))
            hand_frames = len(hand_joint.get("left_hand_obs", []))
            
            if robot_frames == 0:
                logger.error(f"无有效帧: {ep_path}")
                failed += 1
                continue

            # ========== 严格检查帧数对齐 ==========
            frame_counts = {
                "robot_bson": robot_frames,
                "hand_bson": hand_frames,
            }
            frame_counts.update({f"image_files_{k}": v for k, v in image_file_counts.items()})
            frame_counts.update({f"image_loaded_{k}": v for k, v in load_success_counts.items()})
            
            # 检查是否有图像加载失败
            has_load_failure = False
            for cam_key in image_file_counts:
                file_count = image_file_counts.get(cam_key, 0)
                loaded_count = load_success_counts.get(cam_key, 0)
                if file_count != loaded_count:
                    logger.error(f"图像加载失败: {cam_key} 文件数={file_count}, 成功加载={loaded_count}, 丢失={file_count - loaded_count}")
                    has_load_failure = True
            
            if has_load_failure:
                logger.error(f"跳过 episode (图像加载失败): {ep_path}")
                failed += 1
                continue
            
            # 检查所有数据源帧数是否一致
            unique_counts = set([robot_frames, hand_frames] + list(load_success_counts.values()))
            if len(unique_counts) > 1:
                logger.error(f"帧数不对齐: {frame_counts}")
                logger.error(f"跳过 episode (帧数不一致): {ep_path}")
                failed += 1
                continue
            
            num_frames = robot_frames
            logger.info(f"帧数验证通过: {num_frames} 帧")

            # 合并 observation.state 与 action
            observation_states: List[np.ndarray] = []
            actions: List[np.ndarray] = []
            for fi in range(num_frames):
                state_vec = np.concatenate(
                    [
                        joint_data["left_arm_obs"][fi],
                        joint_data["right_arm_obs"][fi],
                        hand_joint["left_hand_obs"][fi],
                        hand_joint["right_hand_obs"][fi],
                        joint_data["head_obs"][fi],
                        joint_data["spine_obs"][fi],
                    ]
                ).astype(np.float32)
                action_vec = np.concatenate(
                    [
                        joint_data["left_arm_action"][fi],
                        joint_data["right_arm_action"][fi],
                        hand_joint["left_hand_action"][fi],
                        hand_joint["right_hand_action"][fi],
                        joint_data["head_action"][fi],
                        joint_data["spine_action"][fi],
                    ]
                ).astype(np.float32)
                observation_states.append(state_vec)
                actions.append(action_vec)

            # 选择 instruction（作为本 episode 的任务描述）
            instruction_text = choose_instruction(instructions, idx, len(episode_dirs), action_name)

            # 写入帧，使用 instruction_text 作为 LeRobot 的 task 名称
            # 注意：此时已验证所有数据源帧数一致，无需条件检查
            for fi in range(num_frames):
                frame_data: Dict[str, Any] = {
                    "observation.state": observation_states[fi],
                    "action": actions[fi],
                }
                for lerobot_key, cam_frames in images_by_camera.items():
                    # 帧数已严格对齐，直接取对应帧
                    frame_data[lerobot_key] = cam_frames[fi]
                dataset.add_frame(frame_data, task=instruction_text, timestamp=fi / config.fps)

            # 保存 episode
            dataset.save_episode()

            # 更新 episodes.jsonl 的 tasks 字段，并记录 mapping
            meta_dir = Path(config.output_root) / "meta"
            episode_index = processed  # 当前数据集内 episode 顺序
            update_episode_instruction(meta_dir, episode_index, instruction_text)
            output_episode_path = str(Path(config.output_root) / "data" / f"chunk-{episode_index:03d}")
            add_episode_mapping(meta_dir, episode_index, str(ep_path), output_episode_path, instruction_text, action_name)

            processed += 1

            # 内存清理提示(依赖 Python GC 无需显式)
        except Exception as e:
            logger.error(f"处理失败 {ep_path}: {e}", exc_info=True)
            # Avoid poisoning the buffer and causing cascading KeyError on subsequent episodes.
            try:
                if getattr(dataset, "episode_buffer", None) is not None:
                    dataset.clear_episode_buffer()
                else:
                    dataset.episode_buffer = None
            except Exception:
                dataset.episode_buffer = None
            
            # 清理失败 episode 的残留数据
            try:
                # 当前 episode_index 是 processed（因为还没 +1）
                current_ep_idx = processed
                output_root = Path(config.output_root)
                
                # 清理残留的 parquet 文件
                parquet_path = output_root / f"data/chunk-000/episode_{current_ep_idx:06d}.parquet"
                if parquet_path.exists():
                    parquet_path.unlink()
                    logger.info(f"已清理残留 parquet: {parquet_path}")
                
                # 清理残留的视频文件
                for cam_key in config.camera_mapping.values():
                    video_path = output_root / f"videos/chunk-000/{cam_key}/episode_{current_ep_idx:06d}.mp4"
                    if video_path.exists():
                        video_path.unlink()
                        logger.info(f"已清理残留视频: {video_path}")
                
                # 清理残留的图像目录
                for cam_key in config.camera_mapping.values():
                    img_dir = output_root / f"images/{cam_key}/episode_{current_ep_idx:06d}"
                    if img_dir.exists():
                        import shutil
                        shutil.rmtree(img_dir, ignore_errors=True)
                        logger.info(f"已清理残留图像目录: {img_dir}")
            except Exception as cleanup_err:
                logger.warning(f"清理残留数据时出错: {cleanup_err}")
            
            failed += 1

    logger.info(f"完成: 成功 {processed}, 失败 {failed}, 输出目录: {config.output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="通用 Action 数据转 LeRobot 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_action190_to_lerobot.py --source /path/to/action264 --output /path/to/output
  
脚本会自动从 --source 目录名推断 action 名称（如 action264），
并用于 repo_id、日志文件名、mapping 等，无需手动修改代码。
""",
    )
    parser.add_argument("--source", type=str, required=True, help="源数据目录（如 /path/to/action1）")
    parser.add_argument("--output", type=str, required=True, help="输出数据集目录")
    parser.add_argument("--fps", type=int, default=20, help="视频帧率（默认 20）")
    args = parser.parse_args()

    config = SingleActionConfig(source_dir=args.source, output_dir=args.output, fps=args.fps)
    logger = setup_logger(config.output_root, config.action_name)

    logger.info(f"=== 转换配置 ===")
    logger.info(f"Action 名称: {config.action_name}")
    logger.info(f"源目录: {config.source_root}")
    logger.info(f"输出目录: {config.output_root}")
    logger.info(f"FPS: {config.fps}")

    # 必要文件快速检查
    missing: List[str] = []
    for name in ["instruction1.txt", "instruction2.txt", "instruction3.txt", "instruction4.txt", "instruction5.txt"]:
        if not (config.source_root / name).exists():
            missing.append(str(config.source_root / name))
    # 仅随机抽查一个 episode 的关键文件
    sample_ep = config.source_root / "episode_0"
    if not sample_ep.exists():
        logger.warning("未找到示例 episode_0 目录，仍将尝试遍历全部 episodes。")
    else:
        if not (sample_ep / config.robot_bson_name).exists():
            missing.append(str(sample_ep / config.robot_bson_name))
        if not (sample_ep / config.hand_bson_name).exists():
            missing.append(str(sample_ep / config.hand_bson_name))

    if missing:
        logger.warning("以下文件缺失，可能影响指令或数据读取：")
        for p in missing:
            logger.warning(f"  - {p}")

    convert_action(config, logger)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 MimicGen / robosuite 风格的 demo.hdf5
(demo_src_appletobowl_task_D0) 转换为 LeRobot v2.1 数据集。

目标是与 dexora 系列数据集的 schema 尽量一致：
- observation.state: 39 维
- action:           39 维
- 4 路 RGB 相机:  top / wrist_left / wrist_right / front
- 同时额外写入对应的 depth 通道（*_depth）

数据来源:
- HDF5: data/ours/hdf5_to_lerobot/demo_src_appletobowl_task_D0/demo.hdf5

输出:
- LeRobot 数据集根目录: data/ours/sim/appletobowl

使用方法 (在仓库根目录):

    cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT
    python -m dataprocess.convert_hdf5_appletobowl_to_lerobot
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import imageio.v2 as iio
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


@dataclass
class AppleBowlHDF5Config:
    """配置 demo_src_appletobowl_task_D0 的路径与映射信息。"""

    # 相对仓库根目录的路径
    hdf5_rel_path: str = "/baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/new_data/left/merge_action7/action7/action7.hdf5"
    output_root_rel: str = "data/ours/sim/left/action7"

    # LeRobot 参数
    fps: float = 20.0
    robot_type: str = "airbot_play"
    repo_id: str = "local/sim_appletobowl_left1_d1"

    # instruction / task
    instruction: str = "Using your left hand, grasp the purple gourd, then place it on the red-and-white plate "


def setup_logger(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = output_root / "convert_hdf5_appletobowl.log"

    logger = logging.getLogger("HDF5AppleBowlToLeRobot")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


def create_features() -> Dict[str, Dict]:
    """构造 LeRobot features，参考 dexora 的 39 维 state/action 定义，并加入 depth。"""

    # 与 dexora 中 39 维 state / action 的 names 对齐
    joint_names: List[str] = [
        # 左臂 6
        "left_arm_joint_1",
        "left_arm_joint_2",
        "left_arm_joint_3",
        "left_arm_joint_4",
        "left_arm_joint_5",
        "left_arm_joint_6",
        # 右臂 6
        "right_arm_joint_1",
        "right_arm_joint_2",
        "right_arm_joint_3",
        "right_arm_joint_4",
        "right_arm_joint_5",
        "right_arm_joint_6",
        # 左手 12
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
        # 右手 12
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
        # 头 / 脊柱 3
        "head_joint_1",
        "head_joint_2",
        "spine_joint",
    ]
    assert len(joint_names) == 39

    features: Dict[str, Dict] = {}

    # 4 路 RGB 相机: top, wrist_left, wrist_right, front
    img_keys = [
        "observation.images.top",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
        "observation.images.front",
    ]
    for key in img_keys:
        features[key] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    # 对应的 depth 通道（单通道）
    depth_img_keys = {
        "observation.images.top_depth",
        "observation.images.wrist_left_depth",
        "observation.images.wrist_right_depth",
        "observation.images.front_depth",
    }
    for key in depth_img_keys:
        features[key] = {
            "dtype": "video",
            # LeRobot 当前视频写入要求 3 通道，因此这里用 3 通道灰度 depth
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": True,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    # 39 维状态
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (39,),
        "names": joint_names,
    }

    # 39 维动作
    features["action"] = {
        "dtype": "float32",
        "shape": (39,),
        "names": joint_names,
    }

    # 补上默认的 meta 特征（timestamp / index / task_index 等）
    features.update(DEFAULT_FEATURES)

    return features


def extract_state_and_action(
    raw_state: np.ndarray,
    raw_action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据用户提供的索引映射，将原始 state(130) / action(38) 映射到 39 维."""

    # state: 索引映射（来自用户说明）
    # - state[12:18]  -> 左臂 6
    # - state[32:38]  -> 右臂 6
    # - state[20:32]  -> 左手 12
    # - state[40:52]  -> 右手 12
    left_arm_state = raw_state[12:18]
    right_arm_state = raw_state[32:38]
    left_hand_state = raw_state[20:32]
    right_hand_state = raw_state[40:52]

    # 固定值的三维
    head_spine = np.array([0.0, -1.0, 0.15], dtype=np.float64)

    state_39 = np.concatenate(
        [left_arm_state, right_arm_state, left_hand_state, right_hand_state, head_spine],
        axis=0,
    ).astype(np.float32)

    # action: 索引映射（来自用户说明，注意最终顺序与 state 对齐）
    # - [2:8]   -> 右臂 6
    # - [8:14]  -> 左臂 6
    # - [14:26] -> 右手 12
    # - [26:38] -> 左手 12
    right_arm_act = raw_action[2:8]
    left_arm_act = raw_action[8:14]
    right_hand_act = raw_action[14:26]
    left_hand_act = raw_action[26:38]

    action_39 = np.concatenate(
        [left_arm_act, right_arm_act, left_hand_act, right_hand_act, head_spine],
        axis=0,
    ).astype(np.float32)

    return state_39, action_39


def convert_hdf5_episode(
    ds: LeRobotDataset,
    cfg: AppleBowlHDF5Config,
    logger: logging.Logger,
    group: h5py.Group,
    video_dir: Path,
) -> None:
    """将单个 HDF5 group (data/demo_*) 转成一个 LeRobot episode。"""

    # 基本数组
    states_ds = group["states"]  # (T, 130)
    actions_ds = group["actions"]  # (T, 38)
    T = states_ds.shape[0]

    logger.info("Episode %s: T=%d", group.name, T)

    # 图像 / 深度：优先从 MP4 读取，若缺失再回退 HDF5
    obs = group["obs"]
    use_mp4 = True
    demo_name = group.name.split("/")[-1]  # e.g., demo_0
    # 数据集中 mp4 命名以 playback_demo_src_appletobowl_task_D0_demo_{k}_*.mp4 形式存在
    def vp(file_suffix: str) -> Path:
        return video_dir / f"{demo_name}_{file_suffix}.mp4"

    mp4_paths = {
        "top_rgb": vp("agentview1_rgb"),
        "front_rgb": vp("head_rgb"),
        "wrist_left_rgb": vp("left_wrist_rgb"),
        "wrist_right_rgb": vp("right_wrist_rgb"),
        "top_depth": vp("agentview1_depth"),
        "front_depth": vp("head_depth"),
        "wrist_left_depth": vp("left_wrist_depth"),
        "wrist_right_depth": vp("right_wrist_depth"),
    }
    # 若任何一个关键视频不存在，则尝试回退使用 HDF5
    for k, p in mp4_paths.items():
        if not p.is_file():
            use_mp4 = False
            break

    if not use_mp4:
        logger.info("未找到完整 MP4 视频集，回退使用 HDF5 中的图像/深度")
        agent_rgb = obs["agentview1_image"]  # (T, 480, 640, 4)
        # 某些数据集中 head/wrist 可能缺失，这里做存在性检查并用 top 代替以避免中断
        head_rgb = obs.get("robot0_head_cam_image", agent_rgb)
        lft_rgb = obs.get("robot0_lft_handeye_image", agent_rgb)
        rgt_rgb = obs.get("robot0_rgt_handeye_image", agent_rgb)

        agent_depth = obs.get("agentview1_depth")  # (T, 480, 640, 1)
        head_depth = obs.get("robot0_head_cam_depth", agent_depth)
        lft_depth = obs.get("robot0_lft_handeye_depth", agent_depth)
        rgt_depth = obs.get("robot0_rgt_handeye_depth", agent_depth)
    else:
        logger.info("使用 MP4 视频读取 %s", demo_name)
        # 打开 8 路视频迭代器
        video_readers = {k: iio.get_reader(str(p), format="ffmpeg") for k, p in mp4_paths.items()}
        video_iters = {k: iter(r) for k, r in video_readers.items()}
        last_frames: Dict[str, np.ndarray] = {}

    # 简单一致性检查
    assert actions_ds.shape[0] == T

    # 工具函数
    def rgb3(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        return arr

    def depth1(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)  # (H, W, [C])
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr2d = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            # 如果深度 mp4 是 3 通道（常见的可视化），先转灰度平均
            arr2d = arr.mean(axis=-1)
        else:
            arr2d = arr

        vmin = float(np.min(arr2d))
        vmax = float(np.max(arr2d))
        if vmax <= vmin:
            norm = np.zeros_like(arr2d, dtype=np.float32)
        else:
            norm = (arr2d - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0)

        u8 = (norm * 255.0).astype(np.uint8)  # (H, W)
        u8_3 = np.stack([u8, u8, u8], axis=-1)  # (H, W, 3)
        return u8_3

    for t in range(T):
        raw_state = states_ds[t]  # (130,)
        raw_action = actions_ds[t]  # (38,)
        state_39, action_39 = extract_state_and_action(raw_state, raw_action)

        frame: Dict[str, np.ndarray] = {}
        frame["observation.state"] = state_39
        frame["action"] = action_39

        if use_mp4:
            # 逐帧读取 mp4
            def next_frame(key: str) -> np.ndarray:
                try:
                    frm = next(video_iters[key])
                except StopIteration:
                    frm = last_frames.get(key, None)
                if frm is None:
                    # 用全 0 帧兜底
                    frm = np.zeros((480, 640, 3), dtype=np.uint8)
                last_frames[key] = frm
                return frm

            frame["observation.images.top"] = rgb3(next_frame("top_rgb"))
            frame["observation.images.front"] = rgb3(next_frame("front_rgb"))
            frame["observation.images.wrist_left"] = rgb3(next_frame("wrist_left_rgb"))
            frame["observation.images.wrist_right"] = rgb3(next_frame("wrist_right_rgb"))

            frame["observation.images.top_depth"] = depth1(next_frame("top_depth"))
            frame["observation.images.front_depth"] = depth1(next_frame("front_depth"))
            frame["observation.images.wrist_left_depth"] = depth1(next_frame("wrist_left_depth"))
            frame["observation.images.wrist_right_depth"] = depth1(next_frame("wrist_right_depth"))
        else:
            frame["observation.images.top"] = rgb3(agent_rgb[t])
            frame["observation.images.front"] = rgb3(head_rgb[t])
            frame["observation.images.wrist_left"] = rgb3(lft_rgb[t])
            frame["observation.images.wrist_right"] = rgb3(rgt_rgb[t])

            frame["observation.images.top_depth"] = depth1(agent_depth[t]) if agent_depth is not None else depth1(agent_rgb[t])
            frame["observation.images.front_depth"] = depth1(head_depth[t]) if head_depth is not None else depth1(head_rgb[t])
            frame["observation.images.wrist_left_depth"] = depth1(lft_depth[t]) if lft_depth is not None else depth1(lft_rgb[t])
            frame["observation.images.wrist_right_depth"] = depth1(rgt_depth[t]) if rgt_depth is not None else depth1(rgt_rgb[t])

        timestamp = t / cfg.fps
        ds.add_frame(frame, task=cfg.instruction, timestamp=timestamp)

    ds.save_episode()
    logger.info("Episode %s 转换完成，共 %d 帧", group.name, T)


def main() -> None:
    cfg = AppleBowlHDF5Config()

    # 仓库根目录: dataprocess 上一级
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]

    hdf5_path = repo_root / cfg.hdf5_rel_path
    output_root = repo_root / cfg.output_root_rel

    logger = setup_logger(output_root)
    logger.info("HDF5 路径: %s", hdf5_path)
    logger.info("输出根目录: %s", output_root)

    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 文件不存在: {hdf5_path}")

    # 创建 LeRobot 数据集
    features = create_features()
    if output_root.exists():
        # 简单起见，总是覆盖
        import shutil

        logger.info("输出目录已存在，删除重建: %s", output_root)
        shutil.rmtree(output_root)

    ds = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=int(cfg.fps),
        features=features,
        root=str(output_root),
        robot_type=cfg.robot_type,
        use_videos=True,
        video_backend="pyav",
    )

    # 遍历 HDF5 中的所有 demo_x group
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        demo_names = sorted(list(data_grp.keys()))
        logger.info("发现 episodes: %s", ", ".join(demo_names))

        for name in demo_names:
            grp = data_grp[name]
            convert_hdf5_episode(ds, cfg, logger, grp, hdf5_path.parent)

    logger.info(
        "全部完成: total_episodes=%d total_frames=%d",
        ds.num_episodes,
        len(ds),
    )


if __name__ == "__main__":
    main()



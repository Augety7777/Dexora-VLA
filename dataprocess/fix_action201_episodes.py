#!/usr/bin/env python3
"""
修复 action201 数据集中有问题的 8 个 episode

使用 LeRobot 原生的 encode_video_frames 函数确保编码格式一致

运行方式:
    conda activate rdt
    cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess
    python fix_action201_episodes.py
"""
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import cv2
import bson

# 使用 LeRobot 原生的视频编码函数
from lerobot.datasets.video_utils import encode_video_frames

DATASET_PATH = Path("/baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/data/action201")
FPS = 20

# 问题 episode 及其源数据路径
PROBLEM_EPISODES = {
    96: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_112",
    184: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_193",
    218: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_224",
    226: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_232",
    367: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_363",
    618: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_663",
    704: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_741",
    722: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201/episode_759",
}

CAMERA_MAPPING = {
    "camera_third_view": "observation.images.top",
    "camera_left_wrist": "observation.images.wrist_left",
    "camera_right_wrist": "observation.images.wrist_right",
    "camera_head": "observation.images.front",
}


def cleanup_episode(ep_idx: int):
    """删除 episode 的所有残留数据"""
    print(f"  清理 Episode {ep_idx} 残留数据...")
    
    # parquet
    parquet = DATASET_PATH / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    if parquet.exists():
        parquet.unlink()
        print(f"    ✅ 删除 parquet")
    
    # 视频
    for cam in ["front", "top", "wrist_left", "wrist_right"]:
        video = DATASET_PATH / f"videos/chunk-000/observation.images.{cam}/episode_{ep_idx:06d}.mp4"
        if video.exists():
            video.unlink()
            print(f"    ✅ 删除 {cam} 视频")
    
    # 残留图像目录
    for cam in ["front", "top", "wrist_left", "wrist_right"]:
        img_dir = DATASET_PATH / f"images/observation.images.{cam}/episode_{ep_idx:06d}"
        if img_dir.exists():
            shutil.rmtree(img_dir, ignore_errors=True)
            print(f"    ✅ 删除 {cam} 图像目录")


def get_video_frame_count(video_path: Path) -> Optional[int]:
    """获取视频帧数"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=120
        )
        return int(result.stdout.strip())
    except:
        return None


def get_min_frames(source_path: Path) -> int:
    """获取源数据的最小帧数（BSON 和图像的最小值）"""
    # BSON 帧数
    robot_bson = source_path / "episode_0.bson"
    with open(robot_bson, "rb") as f:
        robot_data = bson.decode(f.read())
    
    data_section = robot_data.get("data", {})
    bson_frames = len(data_section.get("/observation/left_arm/joint_state", []))
    
    # 图像帧数
    min_frames = bson_frames
    for cam_dir in CAMERA_MAPPING.keys():
        cam_path = source_path / cam_dir
        if cam_path.exists():
            images = [f for f in os.listdir(cam_path) if f.endswith(('.jpg', '.png'))]
            if len(images) < min_frames:
                min_frames = len(images)
    
    return min_frames


def regenerate_episode(ep_idx: int, source_path: str) -> bool:
    """重新生成单个 episode 的视频和 parquet"""
    print(f"\n重新生成 Episode {ep_idx} (源: {Path(source_path).name})...")
    
    source = Path(source_path)
    if not source.exists():
        print(f"  ❌ 源目录不存在: {source}")
        return False
    
    # 获取最小帧数
    num_frames = get_min_frames(source)
    print(f"  统一帧数: {num_frames}")
    
    # 1. 生成视频
    success = True
    for src_cam, dst_cam in CAMERA_MAPPING.items():
        src_dir = source / src_cam
        if not src_dir.exists():
            print(f"  ⚠️ 源相机目录不存在: {src_cam}")
            continue
        
        # 获取源图像列表
        images = sorted([f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))])[:num_frames]
        
        # 创建临时目录并复制/重命名图像为 LeRobot 格式 (frame_000000.png)
        tmp_dir = DATASET_PATH / f"tmp_encode_{ep_idx}_{dst_cam.replace('.', '_')}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        
        print(f"  正在处理 {dst_cam} ({len(images)} 帧)...", end=" ", flush=True)
        
        for i, img_name in enumerate(images):
            src_img = src_dir / img_name
            dst_img = tmp_dir / f"frame_{i:06d}.png"
            # 需要转换为 png
            img = cv2.imread(str(src_img))
            cv2.imwrite(str(dst_img), img)
        
        # 使用 LeRobot 原生编码函数生成视频
        video_path = DATASET_PATH / f"videos/chunk-000/{dst_cam}/episode_{ep_idx:06d}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            encode_video_frames(
                imgs_dir=tmp_dir,
                video_path=video_path,
                fps=FPS,
                vcodec="libsvtav1",
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                overwrite=True
            )
            
            # 验证帧数
            frame_count = get_video_frame_count(video_path)
            if frame_count == len(images):
                print(f"✅ {frame_count} 帧")
            else:
                print(f"⚠️ {frame_count}/{len(images)} 帧")
                success = False
        except Exception as e:
            print(f"❌ 编码失败: {e}")
            success = False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    if not success:
        return False
    
    # 2. 生成 parquet
    print(f"  重新生成 parquet...", end=" ", flush=True)
    
    try:
        robot_bson = source / "episode_0.bson"
        hand_bson = source / "xhand_control_data.bson"
        
        with open(robot_bson, "rb") as f:
            robot_data = bson.decode(f.read())
        with open(hand_bson, "rb") as f:
            hand_data = bson.decode(f.read())
        
        # 提取关节数据
        data_section = robot_data.get("data", {})
        
        def collect_positions(seq, dims):
            positions = []
            for frame_data in seq or []:
                pos = frame_data.get("data", {}).get("pos", [])
                if len(pos) >= dims:
                    positions.append(pos[:dims])
                else:
                    positions.append(pos + [0.0] * (dims - len(pos)))
            return np.array(positions) if positions else np.empty((0, dims), dtype=float)
        
        left_arm_obs = collect_positions(data_section.get("/observation/left_arm/joint_state", []), 6)
        right_arm_obs = collect_positions(data_section.get("/observation/right_arm/joint_state", []), 6)
        left_arm_action = collect_positions(data_section.get("/action/left_arm/joint_state", []), 6)
        right_arm_action = collect_positions(data_section.get("/action/right_arm/joint_state", []), 6)
        
        # Head
        def collect_head(seq):
            positions = []
            for frame_data in seq or []:
                pos = frame_data.get("data", {}).get("pos", [])
                positions.append(pos[:2] if len(pos) >= 2 else [0.0, -1.0])
            return np.array(positions) if positions else np.empty((0, 2), dtype=float)
        
        head_obs = collect_head(data_section.get("/observation/head/joint_state", []))
        head_action = collect_head(data_section.get("/action/head/joint_state", []))
        
        # Spine
        def collect_spine(seq):
            positions = []
            for frame_data in seq or []:
                pos = frame_data.get("data", {}).get("pos", [])
                positions.append([pos[0]] if len(pos) >= 1 else [0.15])
            return np.array(positions) if positions else np.empty((0, 1), dtype=float)
        
        spine_obs = collect_spine(data_section.get("/observation/spine/joint_state", []))
        spine_action = collect_spine(data_section.get("/action/spine/joint_state", []))
        
        # Hand data
        frames = hand_data.get("frames", [])
        left_hand_obs, right_hand_obs = [], []
        left_hand_action, right_hand_action = [], []
        
        for frame in frames:
            action_data = frame.get("action", {})
            obs_data = frame.get("observation", {})
            
            la = action_data.get("left_hand", [])[:12]
            ra = action_data.get("right_hand", [])[:12]
            lo = [np.deg2rad(a) for a in obs_data.get("left_hand", [])[:12]]
            ro = [np.deg2rad(a) for a in obs_data.get("right_hand", [])[:12]]
            
            la = la + [0.0] * (12 - len(la))
            ra = ra + [0.0] * (12 - len(ra))
            lo = lo + [0.0] * (12 - len(lo))
            ro = ro + [0.0] * (12 - len(ro))
            
            left_hand_action.append(la)
            right_hand_action.append(ra)
            left_hand_obs.append(lo)
            right_hand_obs.append(ro)
        
        left_hand_obs = np.array(left_hand_obs)
        right_hand_obs = np.array(right_hand_obs)
        left_hand_action = np.array(left_hand_action)
        right_hand_action = np.array(right_hand_action)
        
        # 如果数据不够，填充
        if len(head_obs) == 0:
            head_obs = np.full((num_frames, 2), [0.0, -1.0])
            head_action = np.full((num_frames, 2), [0.0, -1.0])
        if len(spine_obs) == 0:
            spine_obs = np.full((num_frames, 1), 0.15)
            spine_action = np.full((num_frames, 1), 0.15)
        
        # 构建数据
        states, actions = [], []
        for i in range(num_frames):
            state = np.concatenate([
                left_arm_obs[i], right_arm_obs[i],
                left_hand_obs[i], right_hand_obs[i],
                head_obs[i], spine_obs[i]
            ]).astype(np.float32)
            action = np.concatenate([
                left_arm_action[i], right_arm_action[i],
                left_hand_action[i], right_hand_action[i],
                head_action[i], spine_action[i]
            ]).astype(np.float32)
            states.append(state)
            actions.append(action)
        
        # 创建 DataFrame
        df = pd.DataFrame({
            "observation.state": states,
            "action": actions,
            "timestamp": [i / FPS for i in range(num_frames)],
            "frame_index": list(range(num_frames)),
            "episode_index": [ep_idx] * num_frames,
            "index": list(range(num_frames)),
            "task_index": [0] * num_frames,
        })
        
        # 保存 parquet
        parquet_path = DATASET_PATH / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        
        print(f"✅ {num_frames} 帧")
        return True
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_episode(ep_idx: int) -> bool:
    """验证 episode 数据一致性"""
    parquet = DATASET_PATH / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    video = DATASET_PATH / f"videos/chunk-000/observation.images.top/episode_{ep_idx:06d}.mp4"
    
    if not parquet.exists() or not video.exists():
        return False
    
    pq_frames = len(pd.read_parquet(parquet))
    vid_frames = get_video_frame_count(video)
    
    return pq_frames == vid_frames


def main():
    print("=" * 60)
    print("修复 action201 数据集中的问题 episode")
    print("使用 LeRobot 原生 encode_video_frames 函数")
    print("=" * 60)
    
    results = {}
    
    for ep_idx, source_path in PROBLEM_EPISODES.items():
        print(f"\n{'='*40}")
        print(f"处理 Episode {ep_idx}")
        print(f"{'='*40}")
        
        # 1. 清理残留
        cleanup_episode(ep_idx)
        
        # 2. 重新生成
        success = regenerate_episode(ep_idx, source_path)
        
        # 3. 验证
        if success and verify_episode(ep_idx):
            results[ep_idx] = "✅ 成功"
        elif success:
            results[ep_idx] = "⚠️ 生成成功但验证失败"
        else:
            results[ep_idx] = "❌ 失败"
    
    print("\n" + "=" * 60)
    print("修复结果汇总:")
    print("=" * 60)
    for ep_idx, status in results.items():
        print(f"  Episode {ep_idx}: {status}")
    
    success_count = sum(1 for s in results.values() if "成功" in s and "失败" not in s)
    print(f"\n成功: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n✅ 所有 episode 修复完成！")
    else:
        print("\n⚠️ 部分 episode 修复失败")


if __name__ == "__main__":
    main()


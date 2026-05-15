#!/usr/bin/env python3
"""
修复 3action190 数据集中有问题的 5 个 episode

使用 LeRobot 原生的 encode_video_frames 函数确保编码格式一致

运行方式:
    conda activate rdt
    cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess
    python fix_broken_episodes.py
"""
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# 使用 LeRobot 原生的视频编码函数
from lerobot.datasets.video_utils import encode_video_frames

DATASET_PATH = Path("/baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/data/3action190")
FPS = 20

# 问题 episode 及其源数据路径（从转换日志中提取）
PROBLEM_EPISODES = {
    284: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/new_action190/action264/episode_356",
    317: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/new_action190/action264/episode_387",
    325: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/new_action190/action264/episode_395",
    343: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/new_action190/action264/episode_411",
    406: "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/new_action190/action264/episode_47",
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
            capture_output=True, text=True, timeout=60
        )
        return int(result.stdout.strip())
    except:
        return None


def regenerate_episode(ep_idx: int, source_path: str) -> bool:
    """重新生成单个 episode 的视频（使用 LeRobot 原生编码）"""
    print(f"\n重新生成 Episode {ep_idx} (源: {Path(source_path).name})...")
    
    source = Path(source_path)
    if not source.exists():
        print(f"  ❌ 源目录不存在: {source}")
        return False
    
    success = True
    for src_cam, dst_cam in CAMERA_MAPPING.items():
        src_dir = source / src_cam
        if not src_dir.exists():
            print(f"  ⚠️ 源相机目录不存在: {src_cam}")
            continue
        
        # 获取源图像列表
        images = sorted([f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))])
        num_frames = len(images)
        
        # 创建临时目录并复制/重命名图像为 LeRobot 格式 (frame_000000.png)
        tmp_dir = DATASET_PATH / f"tmp_encode_{ep_idx}_{dst_cam.replace('.', '_')}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        
        print(f"  正在处理 {dst_cam} ({num_frames} 帧)...", end=" ", flush=True)
        
        for i, img_name in enumerate(images):
            src_img = src_dir / img_name
            # LeRobot 需要 .png 格式
            dst_img = tmp_dir / f"frame_{i:06d}.png"
            # 如果源是 jpg，需要转换为 png
            if img_name.endswith('.jpg'):
                import cv2
                img = cv2.imread(str(src_img))
                cv2.imwrite(str(dst_img), img)
            else:
            shutil.copy(src_img, dst_img)
        
        # 使用 LeRobot 原生编码函数生成视频
        video_path = DATASET_PATH / f"videos/chunk-000/{dst_cam}/episode_{ep_idx:06d}.mp4"
        
        try:
            encode_video_frames(
                imgs_dir=tmp_dir,
                video_path=video_path,
                fps=FPS,
                vcodec="libsvtav1",  # 与现有视频保持一致
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                overwrite=True
            )
            
            # 验证帧数
            frame_count = get_video_frame_count(video_path)
            if frame_count == num_frames:
                print(f"✅ {frame_count}/{num_frames} 帧")
            else:
                print(f"⚠️ {frame_count}/{num_frames} 帧 (不匹配!)")
                success = False
        except Exception as e:
            print(f"❌ 编码失败: {e}")
            success = False
        finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return success


def regenerate_parquet(ep_idx: int, source_path: str) -> bool:
    """重新生成 parquet 文件"""
    import bson
    import numpy as np
    
    source = Path(source_path)
    robot_bson = source / "episode_0.bson"
    hand_bson = source / "xhand_control_data.bson"
    
    if not robot_bson.exists() or not hand_bson.exists():
        print(f"  ❌ BSON 文件不存在")
        return False
    
    print(f"  重新生成 parquet...", end=" ", flush=True)
    
    try:
        # 读取 BSON 数据
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
        left_hand_obs = []
        right_hand_obs = []
        left_hand_action = []
        right_hand_action = []
        
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
        
        # 获取视频帧数来确定实际帧数
        video_path = DATASET_PATH / f"videos/chunk-000/observation.images.top/episode_{ep_idx:06d}.mp4"
        video_frames = get_video_frame_count(video_path)
        
        if video_frames is None:
            print(f"❌ 无法获取视频帧数")
            return False
        
        num_frames = min(len(left_arm_obs), video_frames)
        
        # 如果数据不够，填充
        if len(head_obs) == 0:
            head_obs = np.full((num_frames, 2), [0.0, -1.0])
            head_action = np.full((num_frames, 2), [0.0, -1.0])
        if len(spine_obs) == 0:
            spine_obs = np.full((num_frames, 1), 0.15)
            spine_action = np.full((num_frames, 1), 0.15)
        
        # 构建数据
        states = []
        actions = []
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
        
        # 读取现有 episodes.jsonl 获取 instruction
        episodes_file = DATASET_PATH / "meta" / "episodes.jsonl"
        instruction = f"Episode {ep_idx} task"
        with open(episodes_file) as f:
            for line in f:
                ep = json.loads(line)
                if ep.get("episode_index") == ep_idx:
                    instruction = ep.get("tasks", [instruction])[0]
                    break
        
        # 创建 DataFrame
        df_data = {
            "observation.state": states,
            "action": actions,
            "timestamp": [i / FPS for i in range(num_frames)],
            "frame_index": list(range(num_frames)),
            "episode_index": [ep_idx] * num_frames,
            "index": list(range(num_frames)),  # 会在后面更新
            "task_index": [0] * num_frames,
        }
        
        df = pd.DataFrame(df_data)
        
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


def main():
    print("=" * 60)
    print("修复 3action190 数据集中的问题 episode")
    print("使用 LeRobot 原生 encode_video_frames 函数")
    print("=" * 60)
    
    results = {}
    
    for ep_idx, source_path in PROBLEM_EPISODES.items():
        print(f"\n{'='*40}")
        print(f"处理 Episode {ep_idx}")
        print(f"{'='*40}")
        
        # 1. 清理残留
        cleanup_episode(ep_idx)
        
        # 2. 重新生成视频
        video_ok = regenerate_episode(ep_idx, source_path)
        
        # 3. 重新生成 parquet
        if video_ok:
            parquet_ok = regenerate_parquet(ep_idx, source_path)
            results[ep_idx] = "✅ 成功" if parquet_ok else "⚠️ 视频OK，parquet失败"
        else:
            results[ep_idx] = "❌ 视频生成失败"
    
    print("\n" + "=" * 60)
    print("修复结果汇总:")
    print("=" * 60)
    for ep_idx, status in results.items():
        print(f"  Episode {ep_idx}: {status}")
    
    success_count = sum(1 for s in results.values() if "成功" in s)
    print(f"\n成功: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n✅ 所有 episode 修复完成！可以重新运行训练。")
    else:
        print("\n⚠️ 部分 episode 修复失败，建议重新转换整个数据集。")


if __name__ == "__main__":
    main()

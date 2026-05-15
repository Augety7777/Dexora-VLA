#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a custom dataset into **LeRobot v2.0** standard format using the official API.

Key points:
- Forces the v2.0 metadata layout (meta/info.json with codebase_version=v2.0, meta/stats.json global).
- Writes one parquet per episode under data/chunk-XXX/episode_XXXXXX.parquet.
- Writes one mp4 per camera per episode under videos/chunk-XXX/observation.images.<camera_key>/episode_XXXXXX.mp4.
- Uses LeRobot v2.0 API (module path: lerobot.datasets.lerobot_dataset).
- Adds tasks to meta/tasks.jsonl; updates meta/episodes.jsonl and meta/stats.json via official methods.

Assumptions:
- You can adapt the `iterate_source_episodes(...)` generator to your original storage.
- Your original code already computes or can expose per-frame states/actions/flags/timestamps and RGB image frames per camera.
- FPS comes from your config (default 20.0).

Usage (example):
    python airbot_v2_0.py \
        --repo-id your_org/airbot_demo \
        --root /path/to/output_root \
        --source /path/to/source_root \
        --fps 20 \
        --robot-type {ROBOT_TYPE} \
        --cameras camera_high camera_left camera_right camera_front

If your environment only has v2.1 installed, you can install a commit/tag for v2.0,
or vendor the legacy module. This script tries the v2.0 import path first and aborts
if it is not available.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from airbot_config_v2_0 import FPS, ROBOT_TYPE, CAMERA_KEYS, CHUNKS_SIZE, REPO_ID
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Any

import numpy as np
import pandas as pd

# ---------- LeRobot v2.0 imports (required) ----------
try:
    # v2.0 module path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    _LEROBOT_VERSION_NAMESPACE = "v2.0"
except Exception as e:
    raise ImportError(
        "This script targets LeRobot v2.0 (module path: lerobot.datasets.lerobot_dataset). "
        "Please ensure you installed a v2.0-compatible release.\n"
        f"Original import error: {e}"
    )


# ---------- Config (you can adapt to your AirbotConfig) ----------
@dataclass
class AirbotConvertConfig:
    repo_id: str
    source_root: Path
    out_root: Path
    fps: float = float(FPS)
    robot_type: str = "{ROBOT_TYPE}"
    cameras: Tuple[str, ...] = tuple(CAMERA_KEYS)
    overwrite: bool = True
    chunks_size: int = int(CHUNKS_SIZE)  # same semantic as LeRobot default (episodes per chunk)

# ---------- Feature spec for LeRobot v2.0 ----------
def build_features_dict(cameras: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return the 'features' dict used by LeRobotDataset.create(...).

    v2.0 expects at least the following keys (you can add more if you have them):
      - 'states': float32[N] with optional 'names'
      - 'actions': float32[N] with optional 'names'
      - 'next.done': bool
      - 'timestamp': float32
      - for each camera: 'observation.images.<camera_key>' with dtype 'video'

    NOTE: Shapes and names should be set to your real dimensionalities.
    Here we assume 36-dim states/actions (6+12+6+12 like in your Airbot pipeline).
    """
    features: Dict[str, Dict[str, Any]] = {
        "states":   {"dtype": "float32", "shape": [36], "names": None},
        "actions":  {"dtype": "float32", "shape": [36], "names": None},
        "next.done": {"dtype": "bool",   "shape": [],   "names": None},
        "timestamp": {"dtype": "float32","shape": [],   "names": None},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [None, None, 3],  # H,W,3 (actual info will be inferred after first episode)
            "names": None,
        }
    return features

# ---------- Source iteration (ADAPT THIS TO YOUR DATA) ----------
def iterate_source_episodes(source_root: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield dictionaries with the following fields for each episode:
        - episode_index: int
        - states: List[np.ndarray (float32, shape [36])]
        - actions: List[np.ndarray (float32, shape [36])]
        - timestamps: List[float]
        - dones: List[bool]
        - images: Dict[camera_key -> List[np.ndarray (uint8 HxWx3, RGB)]]
        - tasks: List[str]  (natural language task(s) for this episode)
        - instruction: Optional[str] (if you have per-episode instruction)
    You should rewrite this function to reflect your original storage.
    """
    # Placeholder implementation: enumerate folders like "actionXX/episode_YYYYYY"
    # Replace this with your original directory walk and decoding (BSON -> arrays, jpg -> RGB images).
    for i, ep_dir in enumerate(sorted(source_root.rglob("episode_*"))):
        # Mock example, replace with real loading
        # Skip empty episodes
        # yield only structure here
        yield {
            "episode_index": i,
            "states": [],
            "actions": [],
            "timestamps": [],
            "dones": [],
            "images": { },  # {cam: [rgb_frames...]}
            "tasks": ["pick and place"],  # or from your action->task mapping
            "instruction": None,
            "original_path": str(ep_dir),
        }

# ---------- Stats computation (global & per-episode for v2.0) ----------
def compute_episode_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute stats per feature for one episode (v2.0 will aggregate into meta/stats.json).
    Only numerical tensor-like columns are summarized; booleans are skipped.
    """
    stats: Dict[str, Dict[str, float]] = {}
    # states/actions stored as sequences of arrays (object dtype). Stack along time.
    if "states" in df and len(df["states"]):
        S = np.stack(df["states"].to_list()).astype(np.float32)
        stats["states"] = {"mean": float(S.mean()), "std": float(S.std() + 1e-8),
                           "min": float(S.min()), "max": float(S.max())}
    if "actions" in df and len(df["actions"]):
        A = np.stack(df["actions"].to_list()).astype(np.float32)
        stats["actions"] = {"mean": float(A.mean()), "std": float(A.std() + 1e-8),
                            "min": float(A.min()), "max": float(A.max())}
    if "timestamp" in df and len(df["timestamp"]):
        T = df["timestamp"].astype(np.float32).to_numpy()
        stats["timestamp"] = {"mean": float(T.mean()), "std": float(T.std() + 1e-8),
                              "min": float(T.min()), "max": float(T.max())}
    return stats

# ---------- Video writing via LeRobot API (frames -> mp4) ----------
def encode_videos_via_api(ds_root: Path, episode_index: int, camera_keys: Iterable[str], fps: float) -> None:
    """
    Use the v2.0 dataset object's video encoder to convert temporary .png frames
    into .mp4 videos under videos/chunk-XXX/observation.images.<cam>/episode_XXXXXX.mp4
    """
    # In v2.0, videos are also handled via LeRobotDataset.encode_episode_videos
    dataset = LeRobotDataset(repo_id="LOCAL_ONLY", root=ds_root)  # local load
    dataset.encode_episode_videos(episode_index)  # will encode frames -> mp4

# ---------- Core conversion ----------
def convert(cfg: AirbotConvertConfig) -> None:
    ds_root = cfg.out_root / cfg.repo_id
    if cfg.overwrite and ds_root.exists():
        import shutil
        shutil.rmtree(ds_root)
    ds_root.mkdir(parents=True, exist_ok=True)

    # 1) Create metadata (forces v2.0 through the module you import)
    features = build_features_dict(cfg.cameras)
    meta = LeRobotDatasetMetadata.create(
        repo_id=cfg.repo_id,
        fps=int(cfg.fps),
        root=str(ds_root),
        robot_type=cfg.robot_type,
        features=features,
        use_videos=True,
    )
    # set chunk size (for path templating)
    meta.info["chunks_size"] = cfg.chunks_size
    # write back info.json
    from lerobot.datasets.lerobot_dataset import write_info  # v2.0 util
    write_info(meta.info, ds_root)

    # 2) Add tasks dictionary first (dedup by text)
    added_tasks: Dict[str, int] = {}
    def get_or_add_task(text: str) -> int:
        if text not in added_tasks:
            meta.add_task(text)  # writes to meta/tasks.jsonl and updates mapping
            added_tasks[text] = meta.get_task_index(text)
        return added_tasks[text]

    # 3) Iterate episodes in your source and write parquet/videos/meta
    for ep in iterate_source_episodes(cfg.source_root):
        ep_index = int(ep["episode_index"])
        states: List[np.ndarray] = ep["states"]
        actions: List[np.ndarray] = ep["actions"]
        timestamps: List[float] = ep["timestamps"]
        dones: List[bool] = ep["dones"]
        images: Dict[str, List[np.ndarray]] = ep["images"]
        tasks: List[str] = ep.get("tasks", [])
        original_path = ep.get("original_path")

        # a) build DataFrame (column order mirrors spec)
        df = pd.DataFrame({
            "states":    states,
            "actions":   actions,
            "next.done": dones,
            "timestamp": timestamps,
            "frame_index": list(range(len(states))),
            "episode_index": [ep_index] * len(states),
            "index": None,       # optional global index; you can fill with a running counter if needed
            "task_index": None,  # optional numeric ID; you can fill if you have a stable mapping
        })
        # ensure types
        if len(df):
            df["states"] = df["states"].astype(object)
            df["actions"] = df["actions"].astype(object)
            df["next.done"] = df["next.done"].astype(bool)
            df["timestamp"] = df["timestamp"].astype(np.float32)
            df["frame_index"] = df["frame_index"].astype(np.int64)
            df["episode_index"] = df["episode_index"].astype(np.int64)

        # b) write parquet to data/chunk-XXX/episode_XXXXXX.parquet
        # resolve chunk for this ep using meta (v2.0 relies on chunks_size)
        chunk_idx = meta.get_episode_chunk(ep_index)
        data_dir = ds_root / f"data/chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"episode_{ep_index:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        # c) write raw PNG frames temporarily under videos/... (LeRobot encoder expects PNGs)
        #    then encode to mp4 via LeRobot API
        for cam_key, frames in images.items():
            img_dir = ds_root / f"videos/chunk-{chunk_idx:03d}/observation.images.{cam_key}/episode_{ep_index:06d}"
            img_dir.mkdir(parents=True, exist_ok=True)
            # save frames as 000000.png, 000001.png, ...
            for i, rgb in enumerate(frames):
                from PIL import Image
                Image.fromarray(rgb).save(img_dir / f"{i:06d}.png")

        # d) encode to mp4
        encode_videos_via_api(ds_root, ep_index, cfg.cameras, cfg.fps)

        # e) meta: save episode (updates episodes.jsonl and stats.json)
        #    compute episode stats then save
        ep_stats = compute_episode_stats(df)
        # attach tasks
        for t in tasks:
            _ = get_or_add_task(t)
        meta.save_episode(
            episode_index=ep_index,
            episode_length=len(df),
            episode_tasks=tasks,
            episode_stats=ep_stats,
        )

        # optional: record the original path for traceability (v2.0 doesn't mandate it)
        # you can extend episodes.jsonl entry by re-writing after save if needed.

    print(f"[done] Converted to LeRobot v2.0 at: {ds_root}")

def parse_args() -> AirbotConvertConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, type=str)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--fps", default=20.0, type=float)
    p.add_argument("--robot-type", default="{ROBOT_TYPE}")
    p.add_argument("--cameras", nargs="*", default=["camera_high","camera_left","camera_right","camera_front"])
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--chunks-size", type=int, default=1000)
    args = p.parse_args()
    return AirbotConvertConfig(
        repo_id=args.repo_id,
        source_root=args.source,
        out_root=args.root,
        fps=args.fps,
        robot_type=args.robot_type,
        cameras=tuple(args.cameras),
        overwrite=args.overwrite,
        chunks_size=args.chunks_size,
    )

if __name__ == "__main__":
    cfg = parse_args()
    convert(cfg)

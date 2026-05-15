#!/usr/bin/env python3
"""
将多批次生成的 airbot_articulation lerobot 数据集合并为一个整体数据集。

核心目标：
- 保证 episode_index 在合并后连续（旧数据在前，新数据在后）
- 保证 episodes.jsonl / episodes_stats.jsonl / tasks.jsonl / stats.json / info.json 一致、对齐
- 保证 data/*.parquet 与 videos/*.mp4 的编号与新的 episode_index 对应
- 合并 episode_instruction_mapping.jsonl，episode_index 对应新的编号，并更新 output_episode_path

依赖：
- dataprocess/lerobot_split_merge_prcessor-main/lerobot_dataset_lib.py

用法示例（只合并 articulation，这里假设你当前在 Dex-RDT 根目录）：

python -m dataprocess.merge_airbot_articulation \
    --old /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/lerobot_output_restructured/airbot_assemble \
    --new /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/lerobot_output_final_11_29/airbot_assemble \
    --output /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/lerobot_output_restructured_merged/airbot_assemble 

确认合并结果正确后，可以用 mv 替换原来的 airbot_articulation 目录。
"""

import argparse
import os
from typing import Dict, List, Tuple

import sys

# 确保可以 import lerobot_dataset_lib
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_MERGE_DIR = os.path.join(CURRENT_DIR, "lerobot_split_merge_prcessor-main")
if SPLIT_MERGE_DIR not in sys.path:
    sys.path.append(SPLIT_MERGE_DIR)

from lerobot_dataset_lib import (  # type: ignore
    get_info,
    load_jsonl,
    save_jsonl,
    select_episodes,
    write_meta_and_copy,
)


def merge_episode_instruction_mappings(
    source_folders: List[str],
    episode_mapping: List[Tuple[str, int, int]],
    output_folder: str,
) -> None:
    """
    合并各源数据集的 episode_instruction_mapping.jsonl，并重写为新的 episode_index。

    - 使用 episode_mapping 中 (folder, old_idx, new_idx) 建立映射关系
    - 将每个源的 mapping 里的 episode_index 映射到 new_idx
    - 重写 output_episode_path 为合并后数据集中的 data/chunk-XXX
    """
    # 1. 构建从 (folder, old_ep_idx) -> new_ep_idx 的映射
    folder_ep_map: Dict[str, Dict[int, int]] = {}
    for folder, old_idx, new_idx in episode_mapping:
        folder_ep_map.setdefault(folder, {})[old_idx] = new_idx

    # 2. 从合并后的 info.json 读取 chunks_size
    info = get_info(output_folder)
    chunks_size = info.get("chunks_size", 1000)

    merged_records = []
    for folder in source_folders:
        meta_path = os.path.join(folder, "meta", "episode_instruction_mapping.jsonl")
        if not os.path.exists(meta_path):
            continue
        records = load_jsonl(meta_path)
        ep_map = folder_ep_map.get(folder, {})

        for rec in records:
            old_idx = rec.get("episode_index")
            if old_idx is None:
                continue
            if old_idx not in ep_map:
                # 这个 episode 没被选入合并（例如被 start_episodes/max_episodes 剪掉），跳过
                continue
            new_idx = ep_map[old_idx]
            new_rec = dict(rec)
            new_rec["episode_index"] = new_idx

            # 更新 output_episode_path：指向新数据集中的 data/chunk-XXX
            chunk_id = int(new_idx // chunks_size)
            new_rec["output_episode_path"] = os.path.join(
                output_folder, "data", f"chunk-{chunk_id:03d}"
            )
            merged_records.append(new_rec)

    # 写入合并后的 mapping
    out_meta_dir = os.path.join(output_folder, "meta")
    os.makedirs(out_meta_dir, exist_ok=True)
    out_path = os.path.join(out_meta_dir, "episode_instruction_mapping.jsonl")
    save_jsonl(merged_records, out_path)
    print(f"[merge_instruction] wrote {len(merged_records)} records to {out_path}")


def run_merge(old_folder: str, new_folder: str, output_folder: str) -> None:
    """
    将 old_folder 与 new_folder 这两个 lerobot 数据集按顺序合并到 output_folder。

    顺序：old 在前，new 接在后面。
    """
    source_folders = [old_folder, new_folder]

    # 1. 选取 / 重排 episodes & stats & tasks 等
    (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        all_tasks,
        all_stats_data,
        total_frames,
    ) = select_episodes(
        source_folders=source_folders,
        max_entries=None,
        max_episodes=None,
        start_entries=None,
        start_episodes=None,
    )

    # 2. 写 meta & 拷贝数据文件（parquet + videos），生成完整的合并数据集
    # 使用第一个源的数据集作为 base_info 模板
    fps = get_info(source_folders[0]).get("fps", 20)
    max_dim_cli = None  # 使用自动检测到的最大维度

    write_meta_and_copy(
        source_folders=source_folders,
        output_folder=output_folder,
        episode_mapping=episode_mapping,
        all_episodes=all_episodes,
        all_episodes_stats=all_episodes_stats,
        folder_dimensions=folder_dimensions,
        folder_task_mapping=folder_task_mapping,
        episode_to_frame_index=episode_to_frame_index,
        all_stats_data=all_stats_data,
        all_tasks=all_tasks,
        total_frames=total_frames,
        max_dim_cli=max_dim_cli,
        fps=fps,
    )

    # 3. 额外合并 episode_instruction_mapping.jsonl
    merge_episode_instruction_mappings(
        source_folders=source_folders,
        episode_mapping=episode_mapping,
        output_folder=output_folder,
    )

    print(
        f"[merge_done] merged from:\n  {old_folder}\n  {new_folder}\ninto:\n  {output_folder}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="合并 airbot_articulation lerobot 数据集（old + new -> output）"
    )
    parser.add_argument(
        "--old",
        required=True,
        help="旧的（已有）articulation 数据集路径，例如 lerobot_output_restructured/airbot_articulation",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="新增一批 articulation 数据集路径，例如 lerobot_output_final_12_2/airbot_articulation",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="合并后输出的数据集路径（建议先写到一个新目录，检查无误后再替换旧目录）",
    )

    args = parser.parse_args()
    run_merge(args.old, args.new, args.output)


if __name__ == "__main__":
    main()



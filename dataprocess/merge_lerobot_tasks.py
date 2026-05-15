#!/usr/bin/env python3
"""
通用的 LeRobot 数据集任务合并脚本（支持 articulation / assemble / dexterous / pick_and_place 等任意任务）。

设计目标：
- 给定「旧版」与「新增」两批 lerobot 输出根目录（old_root / new_root），
  把若干任务子目录（如 airbot_articulation 等）逐个合并到一个新的 output_root 下。
- 每个任务子目录的合并逻辑与 `merge_airbot_articulation.py` 相同：
  - episode_index 在旧数据后面继续编号；
  - episodes / episodes_stats / tasks / stats / info 等全部一致且对齐；
  - parquet / videos 与新的 episode_index 对应；
  - episode_instruction_mapping.jsonl 也会被重写为新的 episode_index，并更新 output_episode_path。

依赖：
- dataprocess/lerobot_split_merge_prcessor-main/lerobot_dataset_lib.py

使用示例（推荐）：

python -m dataprocess.merge_lerobot_tasks \\
  --old-root /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/lerobot_output_restructured \\
  --new-root /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/lerobot_output_final_12_2 \\
  --output-root /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/lerobot_output_restructured_merged \\
  --tasks airbot_articulation airbot_assemble airbot_dexterous airbot_pick_and_place

然后你会在 output-root 下得到对应的合并后任务目录。
确认无误后，可以用 mv 把单个任务目录替换掉旧的。
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

# 为了能够导入 lerobot_dataset_lib，加入其目录到 sys.path
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


def _merge_episode_instruction_mappings(
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
    folder_ep_map: Dict[str, Dict[int, int]] = {}
    for folder, old_idx, new_idx in episode_mapping:
        folder_ep_map.setdefault(folder, {})[old_idx] = new_idx

    info = get_info(output_folder)
    chunks_size = info.get("chunks_size", 1000)

    merged_records: List[dict] = []
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
                # 该 episode 未被选中（例如被 max_episodes/start_episodes 剪掉）
                continue
            new_idx = ep_map[old_idx]
            new_rec = dict(rec)
            new_rec["episode_index"] = new_idx

            chunk_id = int(new_idx // chunks_size)
            new_rec["output_episode_path"] = os.path.join(
                output_folder, "data", f"chunk-{chunk_id:03d}"
            )
            merged_records.append(new_rec)

    out_meta_dir = os.path.join(output_folder, "meta")
    os.makedirs(out_meta_dir, exist_ok=True)
    out_path = os.path.join(out_meta_dir, "episode_instruction_mapping.jsonl")
    save_jsonl(merged_records, out_path)
    print(f"[merge_instruction] wrote {len(merged_records)} records to {out_path}")


def _run_merge_single(old_folder: str, new_folder: str, output_folder: str) -> None:
    """
    将单个任务的 old_folder 与 new_folder 这两个 lerobot 数据集按顺序合并到 output_folder。

    顺序：old 在前，new 接在后面。
    """
    source_folders = [old_folder, new_folder]

    # 1. 选择 / 重排 episodes & stats & tasks 等
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

    # 2. 写 meta & 拷贝数据文件（parquet + videos）
    fps = get_info(source_folders[0]).get("fps", 20)
    max_dim_cli = None  # 自动检测维度

    os.makedirs(output_folder, exist_ok=True)
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
    _merge_episode_instruction_mappings(
        source_folders=source_folders,
        episode_mapping=episode_mapping,
        output_folder=output_folder,
    )

    print(
        f"[merge_done] merged from:\n  {old_folder}\n  {new_folder}\ninto:\n  {output_folder}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用的 lerobot 任务数据集合并脚本（old_root + new_root -> output_root）"
    )
    parser.add_argument(
        "--old-root",
        required=True,
        help="旧数据集根目录，例如 .../lerobot_output_restructured",
    )
    parser.add_argument(
        "--new-root",
        required=True,
        help="新增数据集根目录，例如 .../lerobot_output_final_12_2",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="合并后输出根目录，例如 .../lerobot_output_restructured_merged",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="要合并的任务子目录名称列表，例如 airbot_articulation airbot_assemble ...",
    )

    args = parser.parse_args()

    old_root = os.path.abspath(args.old_root)
    new_root = os.path.abspath(args.new_root)
    output_root = os.path.abspath(args.output_root)

    print(f"[config] old_root    = {old_root}")
    print(f"[config] new_root    = {new_root}")
    print(f"[config] output_root = {output_root}")
    print(f"[config] tasks       = {args.tasks}")

    for task in args.tasks:
        old_folder = os.path.join(old_root, task)
        new_folder = os.path.join(new_root, task)
        out_folder = os.path.join(output_root, task)

        if not os.path.isdir(old_folder):
            print(f"[skip] old task folder not found: {old_folder}")
            continue
        if not os.path.isdir(new_folder):
            print(f"[skip] new task folder not found: {new_folder}")
            continue

        print(f"\n====== Merging task: {task} ======")
        print(f"  old: {old_folder}")
        print(f"  new: {new_folder}")
        print(f"  out: {out_folder}")
        _run_merge_single(old_folder, new_folder, out_folder)


if __name__ == "__main__":
    main()





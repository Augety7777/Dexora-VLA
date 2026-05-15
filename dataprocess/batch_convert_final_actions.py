#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 data/ours/final 下的每个 actionX 目录单独转换为 LeRobot 数据集。

用法示例（在仓库根目录运行）:

    cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT
    python -m dataprocess.batch_convert_final_actions \
    --start_index 190 \
    --max_actions 1 \
    --max_workers 1 \
    --overwrite

行为约定:
- 源目录: data/ours/final/action1 ... action201
- 任务名: 从 dataprocess/tasks.json 的 \"tasks\" 字典中读取，
          key 为字符串形式的动作索引（如 \"1\"、\"2\" ...），
          value 为对应的任务名，第 i 个 action<i> 使用 key=str(i) 的任务名。
- 输出目录: dataprocess/output/<task_name>

注意:
- 本脚本默认串行处理 (--max_workers=1)，以避免显存/内存压力过大。
  如果你确认机器资源充足，可以把 --max_workers 调大 (例如 2~4)。
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from dataprocess.convert_action190_to_lerobot import SingleActionConfig, setup_logger, convert_action3


def load_tasks(tasks_json_path: Path) -> dict[int, str]:
    """读取 dataprocess/tasks.json 中的 tasks 字典，返回 {action_index: task_name}。"""
    with open(tasks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"{tasks_json_path} 中的 'tasks' 字段不是字典。")

    mapping: dict[int, str] = {}
    for k, v in tasks.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, str):
            continue
        mapping[idx] = v

    if not mapping:
        raise ValueError(f"{tasks_json_path} 中未解析到任何有效的任务映射。")
    return mapping


def convert_single_action(
    action_index: int,
    action_dir: Path,
    task_name: str,
    output_root: Path,
    fps: int,
    overwrite: bool,
) -> Tuple[str, bool, str]:
    """
    调用单动作转换逻辑，将一个 action 目录转换为 LeRobot 数据集。

    返回: (action_id, success, message)
    """
    action_id = f"action{action_index}"
    source_dir = action_dir
    output_dir = output_root / task_name

    # 1) 检查源目录是否存在
    if not source_dir.is_dir():
        return action_id, False, f"源目录不存在，跳过: {source_dir}"

    # 2) 处理输出目录
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            msg = f"输出目录已存在，已删除以便重建: {output_dir}"
        else:
            return action_id, False, f"输出目录已存在且未指定 --overwrite，跳过: {output_dir}"
    else:
        msg = f"输出目录: {output_dir}"

    # 3) 运行单动作转换
    config = SingleActionConfig(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        fps=fps,
    )
    logger = setup_logger(config.output_root)
    logger.info(f"=== 开始转换 {action_id} -> {task_name} ===")
    logger.info(msg)

    try:
        convert_action3(config, logger)
        logger.info(f"=== 完成 {action_id} -> {task_name} ===")
        return action_id, True, ""
    except Exception as e:  # noqa: BLE001
        logger.error(f"转换 {action_id} 失败: {e}", exc_info=True)
        return action_id, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "批量将 data/ours/final 下的 actionX 转为 LeRobot 数据集，并按 "
            "dataprocess/tasks.json 中的任务名重命名输出目录。"
        )
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=201,
        help="要处理的最大 action 数量（从 action1 开始按顺序）。默认: 201。",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="起始 action 序号（包含），即从 action<start_index> 开始。默认: 1。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="视频帧率，传递给单动作转换脚本。默认: 20。",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help=(
            "并行 worker 数量。默认 1（串行），避免显存/内存压力过大。"
            "如果机器资源富余，可以适当调大，例如 2~4。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果指定，则在输出目录已存在时先删除再重建；否则跳过已存在的输出。",
    )
    args = parser.parse_args()

    # 评估并发: 每个 action 转换会加载多路相机图像 + 关节数据，内存占用较高，
    # 201 个同时处理几乎肯定会 OOM 或严重影响 IO 性能，因此默认使用 max_workers=1 串行。
    if args.max_workers > 8:
        print(
            f"[警告] --max_workers={args.max_workers} 过大，"
            "建议不要超过 4，以免占用过多内存/显存。"
        )

    # 仓库根目录: dataprocess/ 上一级
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]

    final_root = repo_root / "data" / "ours" / "final"
    if not final_root.is_dir():
        raise FileNotFoundError(f"未找到源数据根目录: {final_root}")

    tasks_json_path = repo_root / "dataprocess" / "tasks.json"
    if not tasks_json_path.is_file():
        raise FileNotFoundError(f"未找到任务列表文件: {tasks_json_path}")

    tasks = load_tasks(tasks_json_path)  # {action_index: task_name}
    max_task_index = max(tasks.keys()) if tasks else 0
    if args.start_index > max_task_index:
        print(
            f"[警告] start_index={args.start_index} 大于 tasks.json 中的最大动作索引 {max_task_index}，"
            "不会有任何 action 被处理。"
        )

    # 输出根目录改为 data/ours/dexora
    output_root = repo_root / "data" / "ours" / "dexora"
    output_root.mkdir(parents=True, exist_ok=True)

    # 构造待处理的 (index, action_dir, task_name) 列表
    jobs: List[Tuple[int, Path, str]] = []
    for i in range(args.start_index, args.start_index + args.max_actions):
        if i not in tasks:
            # 超出 tasks.json 提供的范围时提前结束
            print(
                f"[提示] action{i} 在 tasks.json 中没有映射 (key='{i}')，"
                "后续 action 将不再处理。"
            )
            break
        action_id = f"action{i}"
        task_name = tasks[i]
        action_dir = final_root / action_id
        jobs.append((i, action_dir, task_name))

    print(f"计划处理 {len(jobs)} 个 action，从 action{args.start_index} 开始。")

    # 串行或有限并行执行
    results: List[Tuple[str, bool, str]] = []

    if args.max_workers <= 1:
        for i, action_dir, task_name in jobs:
            action_id, ok, msg = convert_single_action(
                action_index=i,
                action_dir=action_dir,
                task_name=task_name,
                output_root=output_root,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            status = "成功" if ok else f"失败: {msg}"
            print(f"[{action_id} -> {task_name}] {status}")
            results.append((action_id, ok, msg))
    else:
        # 使用线程池并行调度多个 action；实际计算仍在单进程中完成，
        # 但可以利用 IO 并发。注意不要把 max_workers 设得太大。
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            future_to_job = {
                ex.submit(
                    convert_single_action,
                    i,
                    action_dir,
                    task_name,
                    output_root,
                    args.fps,
                    args.overwrite,
                ): (i, action_dir, task_name)
                for i, action_dir, task_name in jobs
            }
            for fut in as_completed(future_to_job):
                i, _action_dir, task_name = future_to_job[fut]
                try:
                    action_id, ok, msg = fut.result()
                except Exception as e:  # noqa: BLE001
                    action_id, ok, msg = f"action{i}", False, str(e)
                status = "成功" if ok else f"失败: {msg}"
                print(f"[{action_id} -> {task_name}] {status}")
                results.append((action_id, ok, msg))

    # 汇总
    total = len(results)
    success = sum(1 for _aid, ok, _msg in results if ok)
    fail = total - success
    print(f"全部完成: 成功 {success} / {total}, 失败 {fail}")


if __name__ == "__main__":
    main()



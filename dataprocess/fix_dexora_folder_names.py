import json
import re
from pathlib import Path


DEXRDT_ROOT = Path(__file__).resolve().parents[1]
DEXORA_ROOT = DEXRDT_ROOT / "data" / "ours" / "dexora"
TASKS_JSON = DEXRDT_ROOT / "dataprocess" / "tasks.json"


def load_tasks() -> dict[int, str]:
    """Load action index -> folder name mapping from tasks.json."""
    with TASKS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data.get("tasks", {})
    mapping: dict[int, str] = {}
    for k, v in tasks.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        mapping[idx] = v
    return mapping


ACTION_RE = re.compile(r"action(\d+)")


def infer_action_index_from_mapping_file(mapping_file: Path) -> int | None:
    """Read first valid line from episode_instruction_mapping.jsonl and extract action index.

    Prefer parsing from 'source_episode_path' (e.g. .../action116/episode_1).
    Fall back to 'action_id': 'action116' if needed.
    """
    if not mapping_file.is_file():
        return None

    with mapping_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1) try source_episode_path
            src = obj.get("source_episode_path")
            if isinstance(src, str):
                m = ACTION_RE.search(src)
                if m:
                    return int(m.group(1))

            # 2) fallback: action_id like "action116"
            action_id = obj.get("action_id")
            if isinstance(action_id, str):
                m = ACTION_RE.search(action_id)
                if m:
                    return int(m.group(1))

            # if first valid JSON has no usable info, don't keep scanning too long
            break

    return None


def plan_renames(dry_run: bool = True) -> None:
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks from {TASKS_JSON}")

    if not DEXORA_ROOT.is_dir():
        raise SystemExit(f"dexora root not found: {DEXORA_ROOT}")

    planned: dict[Path, Path] = {}

    for sub in sorted(DEXORA_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name == "logs":
            continue

        mapping_file = sub / "meta" / "episode_instruction_mapping.jsonl"
        action_idx = infer_action_index_from_mapping_file(mapping_file)

        if action_idx is None:
            print(f"[WARN] Cannot infer action index for {sub.relative_to(DEXRDT_ROOT)}")
            continue

        if action_idx not in tasks:
            print(
                f"[WARN] Action index {action_idx} not found in tasks.json "
                f"for folder {sub.relative_to(DEXRDT_ROOT)}"
            )
            continue

        target_name = tasks[action_idx]
        target_path = sub.parent / target_name

        if target_path == sub:
            # already correct
            print(f"[OK] {sub.name} already matches action{action_idx} -> {target_name}")
        else:
            planned[sub] = target_path
            print(
                f"[PLAN] {sub.name}  ->  {target_name}  "
                f"(action{action_idx})"
            )

    if not planned:
        print("No renames planned.")
        return

    if dry_run:
        print("\nDry run only. No folders have been renamed.")
        return
    # ------------------------------------------------------------------
    # Two-phase renaming to safely resolve cyclic permutations:
    #   1) rename each original folder to a unique temporary name
    #   2) rename temps to their final targets
    # ------------------------------------------------------------------
    tmp_mapping: dict[Path, Path] = {}

    # Phase 1: to temporary names
    for src, dst in sorted(planned.items(), key=lambda kv: str(kv[0])):
        tmp = src.parent / f"__tmp_fix__{src.name}"
        if tmp.exists():
            raise RuntimeError(f"Temporary path already exists: {tmp}")
        print(f"[TMP] {src.name} -> {tmp.name}")
        src.rename(tmp)
        tmp_mapping[tmp] = dst

    # Phase 2: from temporary names to final destinations
    for tmp, dst in sorted(tmp_mapping.items(), key=lambda kv: str(kv[0])):
        if dst.exists():
            raise RuntimeError(
                f"Target path already exists when applying rename: {dst}. "
                "This indicates duplicate mappings; please inspect manually."
            )
        print(f"[RENAME] {tmp.name} -> {dst.name}")
        tmp.rename(dst)

    print("\nRenaming completed.")


if __name__ == "__main__":
    # default to dry-run; pass an argument "apply" to actually rename
    import sys

    do_apply = len(sys.argv) > 1 and sys.argv[1] == "apply"
    plan_renames(dry_run=not do_apply)



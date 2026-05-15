# -*- coding: utf-8 -*-
"""
Modified config for LeRobot v2.0 conversion using official API.
This file keeps your original knobs if present, and adds LeRobot-specific ones.
"""

# Try to import original config to reuse values
try:
    from airbot_config import *  # noqa
except Exception:
    pass

# Defaults (will be overridden by original if defined)
FPS = globals().get("FPS", 20.0)
ROBOT_TYPE = globals().get("ROBOT_TYPE", "airbot_dexterous")

# Cameras used in your dataset (update to match your real sources)
CAMERA_KEYS = globals().get("CAMERA_KEYS", ("camera_high","camera_left","camera_right","camera_front"))

# Output chunk size (episodes per chunk directory)
CHUNKS_SIZE = globals().get("CHUNKS_SIZE", 1000)

# Repository identifier (used by LeRobot metadata)
REPO_ID = globals().get("REPO_ID", "local/airbot_v2_0")

# Root dirs (can be overridden via CLI in airbot_v2_0.py)
SOURCE_ROOT = globals().get("SOURCE_ROOT", "./source")
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", "./output")

# Optional: mapping from action folder to task text (example)
TASK_TEXT_BY_ACTION = globals().get("TASK_TEXT_BY_ACTION", {
    # "action01": "pick and place",
})

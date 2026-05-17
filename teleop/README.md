# Dexora teleoperation & data-collection kit

This directory is the **on-robot** companion to the model code in the rest of
Dexora-VLA: it gathers the four-camera + 36-DoF demonstrations that the
training pipeline (`s1_pretrain.sh` → `s3_post_train.sh`) expects, plus the
playback / data-quality scripts referenced in the paper (§III-A / §III-B).

It bundles three originally-separate code bases —
[`mmk_dev/Imitate-All`](#)（robot + 4-camera recording framework）,
[`teleop_software_pkg`](#)（Vision Pro → XHand teleoperation）, and
[`mmk_xhand_record`](#)（top-level launchers + data triage scripts）— with
all hard-coded paths replaced by `PROJECT_ROOT`-anchored ones, so this
folder ports cleanly to a new machine.

To move it onto a new robot you only need to:
① edit `scripts/*.py` to point `ROBOT_PYTHON_PATH` / `HAND_PYTHON_PATH` at the
right conda envs;
② drop the XHand `auth_info.json` / `key.dat` into `teleop_pkg/` (see
`teleop_pkg/SECRETS.md` — never checked in);
③ install the two conda envs (see below).

The recorded data lands in `LeRobot v2.1` format — the same layout that
`Dexora/Dexora_Real-World_Dataset` ships on HuggingFace and that
`data/lerobot_vla_dataset.py` consumes — so a fresh dataset you collect here
can be fed straight into `s1_pretrain.sh`.

---

## 目录结构

```
mmk_teleop_record_kit/
├── README.md                       ← 你正在看的文件
├── requirements.txt                ← 顶层指引（两个 env 各自的 requirements 在子目录里）
├── .gitignore
│
├── scripts/                        ← 「外层启动器」(orchestrators)
│   ├── record_delete.py            ← 🟢 实际数采使用的主程序
│   ├── record.py                   ← 早期版本（复制不删源）
│   ├── record_intrpt.py            ← 带 ProcessManager 的版本
│   ├── replay.py                   ← 机器人 + 灵巧手 同步回放
│   ├── replay_lerobot.py           ← 支持 lerobot parquet 的回放
│   └── replay_only_robot.py        ← 仅机器人回放
│
├── imitate_all/                    ← 来自 mmk_dev/Imitate-All 的子集（实际录制 / 回放）
│   ├── record_4_rgb_cam.py         ← 四路 USB 相机 + MMK2 的录制脚本
│   ├── mmk_replay.py               ← BSON 轨迹回放
│   ├── mmk_replay_lerobot.py       ← Parquet 轨迹回放
│   ├── habitats/  robots/  data_process/  configurations/  envs/  utils/
│   ├── requirements/               ← imitall 环境的 pip 需求
│   ├── 99-camera-symlinks.rules    ← USB 相机 → /dev/camera_* 的 udev 规则
│   └── install_camera_symlinks.sh
│
├── teleop_pkg/                     ← 来自 teleop_software_pkg 的子集（灵巧手侧）
│   ├── receive_from_vision_pro.py  ← 从 Vision Pro 取手势 → 下发 XHand + 录制 bson
│   ├── control_from_bson.py        ← 灵巧手 BSON 回放
│   ├── config.yaml                 ← XHandTeleOps 入口配置
│   ├── env.yaml                    ← xhand_tele_env conda 环境快照
│   ├── xhand_tele_ops-*.whl        ← 厂家提供的 SDK wheel（仅 x86_64 / cp38）
│   ├── auth_info.example.json      ← 鉴权占位（真正的 auth_info.json 不入库）
│   └── SECRETS.md                  ← 鉴权 / 密钥文件如何配置
│
├── data_tools/                     ← 数据筛查 / BSON 处理
│   ├── validate_data_consistency.py    ← 校验 4 相机帧数 vs BSON 帧数
│   ├── swap_action_observation_bson.py ← BSON 内 action↔observation 对调（+ 角→弧度）
│   ├── bson_to_json_converter.py
│   └── sync_helper.py
│
├── video_tools/                    ← 视频/审查相关
│   ├── video.py                    ← 4 路图片 → 4 路 MP4
│   ├── video_rotate.py             ← 图片 → GIF（含旋转 / 抽帧）
│   ├── video_grid_merge.py         ← 4 路 MP4 → 2×2 网格 MP4
│   └── video_review_generator.py   ← 直接从图片一步出 2×2 审查视频
│
├── camera_tools/                   ← 相机调试
│   ├── usb_cameras.py              ← 同时打开多路 USB 相机预览
│   ├── camera_test.py              ← 多路同步采图 / 落盘吞吐测试
│   ├── tools_camera.py             ← 基于 AprilTag 的位姿对齐辅助（NCC / 偏差箭头）
│   └── v4.py                       ← AprilTag 调试 demo
│
├── samples/                        ← 调试 / 回放用的样例数据
│   ├── episode_0.bson              ← 机器人本体 1 个 episode
│   ├── episode_0.json              ← 上面 bson 的 json 镜像
│   ├── xhand_control_data.bson     ← 灵巧手 1 个 episode
│   └── xhand_control_data.json
│
└── docs/
    └── 验证脚本使用说明.md
```

---

## 完整数据流（一次数采的全过程）

```
┌────────────────────────────────────────────────────────────────────────┐
│   scripts/record_delete.py   ← 你在终端跑的入口                          │
│   (在 imitall env 里 fork 两个子进程)                                  │
└────────────────┬─────────────────────────────────────┬────────────────┘
                 │                                     │
       ROBOT_PYTHON_PATH                       HAND_PYTHON_PATH
                 │                                     │
     imitate_all/record_4_rgb_cam.py        teleop_pkg/receive_from_vision_pro.py
     ├─ 打开 4 路 USB 相机                  ├─ 连接 Vision Pro
     ├─ 通过 airbot_py 控本体              ├─ retarget 手势 → XHand 关节
     ├─ 落盘 imitate_all/data/raw/         ├─ 落盘 teleop_pkg/xhand_control_data.bson
     │     example/episode_0/{camera_*}    │     (含 action+observation+t)
     │     example/episode_0.bson          │
     └─ 录够帧或按 's' 退出                 └─ 按 's' 退出
                 │                                     │
                 └────────────── wait() ──────────────┘
                                  │
                          record_delete.py:copy()
                                  │
              cp 全部产物到 /media/slam/data/action6/episode_{N}/
              rm 源文件
```

回放方向：

```
scripts/replay.py
 ├─ imitate_all/mmk_replay.py        (读 episode_0.bson → 下发 MMK2 关节)
 └─ teleop_pkg/control_from_bson.py  (读 xhand_control_data.bson → 下发 XHand)
两路握手到 READY 后等用户回车，同步发 START。
```

数据筛查：

```
data_tools/validate_data_consistency.py   ← 跑完所有 action_* 下的 episode_*，
                                            检查每个 episode 的 camera_*/ 图片数
                                            与两个 BSON 的帧数是否一致，
                                            把异常项写到 logs/*.log
data_tools/swap_action_observation_bson.py ← 在某些标注阶段需要把 BSON 里的
                                              action / observation 对调（xhand 还会
                                              把角度制转弧度制）
```

---

## 环境安装

需要**两个独立的 conda 环境**，因为机器人侧和灵巧手侧的 Python 主版本不一样：

| 环境名 | Python | 用途 | 参考文件 |
| ------ | ------ | ---- | -------- |
| `imitall`         | 3.10 | 机器人本体 + 4 相机数采、本体回放 | `imitate_all/requirements/*.txt`（详细环境快照见原仓库 `environment.yml`） |
| `xhand_tele_env`  | 3.8  | XHand 遥操、Vision Pro 接入       | `teleop_pkg/requirements_x86_64.txt`、`teleop_pkg/env.yaml`、`teleop_pkg/xhand_tele_ops-*.whl` |

最小化安装步骤（机器是 x86_64 + CUDA 12.x）：

```bash
# === imitall ===
conda create -n imitall python=3.10 -y
conda activate imitall
pip install -r imitate_all/requirements/data_collection.txt
pip install -r imitate_all/requirements/realsense.txt
# airbot-data / airbot-py / mmk2-types 这几个私有包通常由 airbot 提供 wheel
# 详细完整的环境快照见原作者机器上的 ~/mmk_dev/environment.yml

# === xhand_tele_env ===
conda create -n xhand_tele_env python=3.8 -y
conda activate xhand_tele_env
pip install teleop_pkg/xhand_tele_ops-1.1.5-cp38-cp38-linux_x86_64.whl
pip install -r teleop_pkg/requirements_x86_64.txt
# 让 python 能监听原始网络包（pynput 处理 Vision Pro 推流必需）
sudo setcap cap_net_raw+ep "$(readlink -f "$(which python3)")"
```

然后把 `scripts/*.py` 文件顶部的下面两行改成实际路径：

```python
ROBOT_PYTHON_PATH = "/home/slam/miniconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH  = "/home/slam/miniconda3/envs/xhand_tele_env/bin/python"
```

---

## 系统级前置

### 1. USB 相机符号链接（4 路 USB 摄像头）

`imitate_all/record_4_rgb_cam.py` 走 `/dev/camera_{left,right,high,head}` 这种**固定符号链接**而不是 `/dev/video*`，这样插拔后编号不会乱：

```bash
cd imitate_all
sudo ./install_camera_symlinks.sh
# 然后插拔一下相机或 reboot，让规则生效
ls -la /dev/camera_*
```

> 注意：默认规则里 `head` 这个名字目前没有对应条目（只 left/right/high），需要按 `相机映射使用说明.md` 自行加上头相机的 devpath。

### 2. 机器人网络

MMK2 默认通过 gRPC 走 `192.168.11.200:50055`，确保主机和机器人本体能互通。

### 3. Vision Pro 连接

在 `teleop_pkg/config.yaml` 把 `avp_ip` 改成你的 Vision Pro 实际 IP，
确认 Vision Pro 端已经推流（参考 [avp-stream](https://pypi.org/project/avp-stream/)）。

### 4. 鉴权文件

参见 [`teleop_pkg/SECRETS.md`](teleop_pkg/SECRETS.md)。

---

## 使用方法

### 数采

```bash
conda activate imitall   # 仅用来给外层启动器一个解释器，子进程自己会切到对应 env
cd mmk_teleop_record_kit

# 录第 N 条 episode（--order 必须传，否则 copy 阶段会报错退出）
python scripts/record_delete.py --order 0
python scripts/record_delete.py --order 1
...
```

操作流程：

1. 程序拉起后，机器人侧会弹出按键提示，按空格开始录制本 episode。
2. 灵巧手侧另有自己的键盘监听：空格开始、`s` 保存退出、ESC 终止。
3. 两侧都按 `s` 保存后，外层会自动 `copy()` 归档到
   `ARCHIVE_ROOT/episode_{order}/`（默认 `/media/slam/data/action6/`，改 `record_delete.py` 顶部常量即可）。

### 回放

```bash
# 默认回放 samples/episode_0.bson + samples/xhand_control_data.bson
python scripts/replay.py
# 替换为自己的数据：改 scripts/replay.py 顶部的 DEFAULT_EPISODE_BSON
```

### 数据筛查

```bash
# 校验所有 action_* 文件夹
python data_tools/validate_data_consistency.py --path /media/slam/data
# 校验单个 action
python data_tools/validate_data_consistency.py --path /media/slam/data --action action6 -v
```

### 视频审查

```bash
# 直接从图片一步生成 2×2 网格的审查视频
python video_tools/video_review_generator.py \
    --input /media/slam/data --output review_videos --fps 20
```

---

## 建议改进（已知问题与可优化点）

**`scripts/record_delete.py` 的几个小坑（保留原行为未修，但值得修一下）：**

1. `--order` 是在 `copy()` 里临场 `argparse` 的，**数采全程跑完才会校验**。
   如果忘了传，本次采集会白干。建议挪到 `main()` 顶部统一解析。
2. `KeyboardInterrupt` 时只 `terminate()`，没有 `wait()` / `kill()` 兜底，
   也没有「一个子进程退了就把另一个一起带走」的逻辑。`record_intrpt.py` 有
   `ProcessManager` 的实现，可以借鉴。
3. `start-episode` 永远是 0，靠 `copy()` 改名累积 episode。
   建议直接根据 `--order` 动态生成 `--repo-id` 和保存路径，避免「永远写 episode_0
   再改名」的脆弱模式。

**同步性：**

4. 两个子进程之间没有时间起止握手或共享 trial_id。`sync_helper.py` 已经写了
   基于文件信号的同步原语，但没接进数采主流程。`replay.py` 已经接了 READY/START
   信号的同步，可以把同样的机制反向用到 record 流程里。

**数据落盘对齐：**

5. `xhand_control_data.bson` 是按帧顺序写的，没有显式 timestamp（只在
   `receive_from_vision_pro_timestemp.py` 里有）。如果想后期严格按时间对齐
   两路数据，建议默认切到 `_timestemp` 版本。

**可移植性：**

6. `scripts/*.py` 顶部的 `ROBOT_PYTHON_PATH` / `HAND_PYTHON_PATH` 仍是硬编码。
   可以改成读 `os.environ["ROBOT_PYTHON"]` / `HAND_PYTHON`，再在 README 里给出
   `.envrc` 模板，迁移到不同机器时只需要改环境变量。
7. `imitate_all/data/` 在录制阶段会被自动创建。把它加进 `.gitignore` 以避免
   误提交（已加）。
8. 工具脚本 `video_tools/video.py`、`video_rotate.py`、`camera_tools/v4.py` 里
   仍写死了 `/home/air/...` 风格的数据集路径，每次用前需要手动改。可以改成
   `argparse` 接收路径，更通用。

**鉴权 / 密钥：**

9. `teleop_pkg/auth_info.json`、`teleop_pkg/key.dat` 没有随项目分发（保密原因）。
   迁移到新机器一定要补上，见 `teleop_pkg/SECRETS.md`。

---

## 出处与归属

* `imitate_all/` 来自 [Airbot Imitate-All](https://github.com/airbots-org/Imitate-All) 的本地拷贝（LICENSE 见同目录）。
* `teleop_pkg/` 中的 `xhand_tele_ops` wheel 与 SDK 接口属第三方（RobotEra / XHand）。
* 其余外层启动器、数据筛查脚本、视频审查脚本由原作者
  （wzr / `mmk_xhand_record/`）编写，本仓库只做了路径整理。

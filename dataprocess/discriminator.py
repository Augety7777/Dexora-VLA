import bson
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# ===== 参数设置 =====
files = {
    "w/ discriminator": "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess/Discriminator/yuwxhand_control_data.bson",
    "w/o discriminator": "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess/Discriminator/yubuwenxhand_control_data.bson",
}
hand = "left_hand"   # "left_hand" 或 "right_hand"
joint_index = 4       # Right Hand Joint 9 → 索引 8
NORMALIZE = True      # 归一化：按“每个文件内所有关节”的全局 min/max 到 [0,1]

# ===== 读取函数（兼容 frames 为 dict 或 list），并统计文件内全关节的全局 min/max =====
def load_bson_data(filepath, hand, joint_index):
    with open(filepath, "rb") as f:
        data = bson.decode_all(f.read())

    times_sel, values_sel = [], []
    file_all_values = []

    for record in data:
        frames = record.get("frames")
        if frames is None:
            continue
        frames_iter = frames if isinstance(frames, list) else [frames]
        for fr in frames_iter:
            try:
                t = fr["t"]
                action = fr["action"]
            except (KeyError, TypeError):
                continue

            # 收集所选关节的曲线
            try:
                val_sel = action[hand][joint_index]
                times_sel.append(t)
                values_sel.append(val_sel)
            except Exception:
                pass

            # 累积该文件内“所有关节值”（左右手、所有关节、所有帧）
            if isinstance(action, dict):
                for h in ("left_hand", "right_hand"):
                    arr = action.get(h)
                    if isinstance(arr, (list, tuple)):
                        for v in arr:
                            if isinstance(v, (int, float)):
                                file_all_values.append(v)

    # 计算文件级全关节最小最大
    if file_all_values:
        fmin, fmax = min(file_all_values), max(file_all_values)
    else:
        fmin, fmax = 0.0, 0.0

    return times_sel, values_sel, fmin, fmax

# ===== 加载数据 =====
raw = {label: load_bson_data(path, hand, joint_index) for label, path in files.items()}

# ===== 归一化：每个文件用其“全关节全帧”的 min/max 做缩放 =====
if NORMALIZE:
    def normalize_by_file(values, fmin, fmax):
        rng = fmax - fmin
        if rng == 0:
            return [0.5] * len(values)
        return [(v - fmin) / rng for v in values]

    data_dict = {
        lbl: (times, normalize_by_file(values, fmin, fmax))
        for lbl, (times, values, fmin, fmax) in raw.items()
    }
else:
    data_dict = {lbl: (times, values) for lbl, (times, values, _, _) in raw.items()}

# ===== 动画绘制（横轴为 Time Steps，约20Hz 播放）=====
fig, ax = plt.subplots()
max_len = max((len(times) for times, _ in data_dict.values()), default=1)
ax.set_xlim(0, max_len)
if NORMALIZE:
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized Joint Value (per-file across all joints)")
else:
    all_vals_plot = [v for _, values in data_dict.values() for v in values]
    vmin = min(all_vals_plot) if all_vals_plot else -1.0
    vmax = max(all_vals_plot) if all_vals_plot else 1.0
    pad = 0.1 * (vmax - vmin + 1e-9)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_ylabel("Joint Value")
ax.set_xlabel("Time Steps")

colors = {"w/o discriminator": "orange", "w/ discriminator": "blue"}
lines = {label: ax.plot([], [], lw=2, color=colors.get(label, None), label=label)[0] for label in data_dict}
ax.legend()

def init():
    for line in lines.values():
        line.set_data([], [])
    return tuple(lines.values())

def update(i):
    for label, (times, values) in data_dict.items():
        n = min(i, len(values))
        x = list(range(n))     # Time Steps
        y = values[:n]
        lines[label].set_data(x, y)
    return tuple(lines.values())

ani = animation.FuncAnimation(
    fig, update,
    frames=max_len,
    init_func=init,
    blit=True,
    interval=50  # ~20Hz
)

# 保存为 20fps MP4
writer = FFMpegWriter(fps=20)
ani.save("cornleftHandJoint5.mp4", writer=writer)

plt.show()
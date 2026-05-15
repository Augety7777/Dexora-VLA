import subprocess
import os
import threading

# 替换下面这些路径为你的实际路径
ROBOT_PYTHON_PATH = "/home/air/anaconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH = "/home/air/anaconda3/envs/xhand_tele_env/bin/python"

# ROBOT_SCRIPT = os.path.abspath("/home/air/mmk_dev/Imitate-All/replay_wzr.py") # 如果例程不行，就用这个
ROBOT_SCRIPT = os.path.abspath("/home/air/mmk_dev/Imitate-All/.mmk_replaypy")
HAND_SCRIPT = os.path.abspath("/home/air/teleop_software_pkg/control_from_bson.py")

ROBOT_ARGS = [
    "data/raw/example/episode_0.bson",
    "--ip", "192.168.11.200",
    "--freq", "20",
]

def wait_for_ready(process, name, ready_event):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8").strip()
        print(f"[{name}] {decoded}")
        if decoded == "READY":
            ready_event.set()
            break
def main():
    from threading import Event
    
    print("🔧 启动机器人控制程序...")
    robot_proc = subprocess.Popen(
        [ROBOT_PYTHON_PATH, ROBOT_SCRIPT] + ROBOT_ARGS,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    print("🔧 启动灵巧手控制程序...")
    hand_proc = subprocess.Popen(
        [HAND_PYTHON_PATH, HAND_SCRIPT],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # 等待子程序输出 READY
    robot_ready = Event()
    hand_ready = Event()
    threading.Thread(target=wait_for_ready, args=(robot_proc, "Robot", robot_ready)).start()
    # threading.Thread(target=wait_for_ready, args=(hand_proc, "Hand", hand_ready)).start()
    threading.Thread(target=wait_for_ready, args=(hand_proc, "Hand", hand_ready)).start()

    robot_ready.wait()
    hand_ready.wait()

    # 用户输入验证
    print("✅ 两个子系统初始化完成。请输入启动命令。")
    print("🔔 请输入空行（直接按回车）以同步启动两个子系统。")
    while True:
        user_input = input("▶️ 等待回车启动 >> ").strip()
        if user_input == "":
            break
        else:
            print("❌ 非法输入。请输入空行（只按回车）以启动。")

    print("🚀 正在向两个子系统发送 START 指令...")
    robot_proc.stdin.write(b"START\n")
    robot_proc.stdin.flush()
    hand_proc.stdin.write(b"START\n")
    hand_proc.stdin.flush()

    # 等待两个进程完成
    try:
        robot_proc.wait()
        hand_proc.wait()
    except KeyboardInterrupt:
        print("🔴 检测到中断，终止两个进程...")
        robot_proc.terminate()
        hand_proc.terminate()

if __name__ == "__main__":
    main()

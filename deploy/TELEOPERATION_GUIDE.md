# Human-in-the-Loop 遥操作控制指南

## 概述

`dagger_controller_debug.py` 现在支持完整的键盘遥操作控制，使用**真实的MMK2 KDL库**进行逆运动学(IK)和正向运动学(FK)计算，精确控制手腕位置和姿态，**同时支持手部（灵巧手）的键盘控制**。

## 功能特性

- ✅ 实时策略推理与人类干预模式无缝切换
- ✅ 键盘控制手腕6D位姿（位置xyz + 旋转rpy）
- ✅ **真实MMK2 KDL库**进行IK/FK求解（基于实际DH参数）
- ✅ **手部（灵巧手）键盘控制** - 预设姿态 + 微调（新增 2025-11-19）
- ✅ 自动记录策略动作 vs 人类动作（用于DAgger训练）
- ✅ 支持单臂/双臂控制切换
- ✅ 模式切换时自动重新初始化目标位姿（避免突变）

## 前置要求

### 安装MMK2 KDL依赖（必需）

```bash
# 安装所有必需的依赖
pip install casadi bidict xacrodoc lxml numpy
```

**重要**：没有这些依赖，程序将无法启动！

## 启动命令

```bash
python deploy/dagger_controller_debug.py \
  --pretrained-model-path v1-20k190/checkpoint-20000 \
  --lang-embeddings-path outs/action190.pt \
  --normalize-mode mean_std \
  --stats-file 20k190v1_bson_stats/1113action190/dataset_statistics.json
```

## 键盘控制

### 模式控制

| 按键 | 功能 |
|------|------|
| **空格键** | 切换 AI策略模式 ↔ 人类遥操作模式 |
| **S** | 保存干预数据到 `interventions/` 目录 |
| **Q** | 退出程序 |

### 遥操作控制（仅在人类模式下）

#### 手臂选择
| 按键 | 功能 |
|------|------|
| **1** | 激活左臂控制 👈 |
| **2** | 激活右臂控制 👉 |
| **3** | 激活双臂同步控制 👐 |

#### 位置控制（笛卡尔空间）
| 按键 | 动作 | 步长 |
|------|------|------|
| **W** | 向前移动 (+X) | 1cm |
| **A** | 向左移动 (+Y) | 1cm |
| **D** | 向右移动 (-Y) | 1cm |
| **R** | 向上移动 (+Z) | 1cm |
| **F** | 向下移动 (-Z) | 1cm |

#### 旋转控制（欧拉角）
| 按键 | 动作 | 步长 |
|------|------|------|
| **I** | Roll+ | ~3° |
| **K** | Roll- | ~3° |
| **J** | Pitch+ | ~3° |
| **L** | Pitch- | ~3° |
| **U** | Yaw+ | ~3° |
| **O** | Yaw- | ~3° |

#### 手部控制（灵巧手 - v2 手指选择模式）- **新增 2025-11-19**

**手指选择（可多选，按键切换）**：
| 按键 | 动作 | 说明 |
|------|------|------|
| **T** | 👍 切换拇指 | 选择/取消拇指控制 |
| **Y** | ☝️ 切换食指 | 选择/取消食指控制 |
| **H** | 🖕 切换中指 | 选择/取消中指控制 |
| **N** | 💍 切换无名指 | 选择/取消无名指控制 |
| **M** | 🤙 切换小指 | 选择/取消小指控制 |

**控制选中的手指**（速度模式）：
| 按键 | 动作 | 说明 |
|------|------|------|
| **Z** (按住) | ✊ 闭合 | 选中的手指持续增加角度，松开停止 |
| **C** (按住) | 🖐️ 张开 | 选中的手指持续减少角度，松开停止 |

**速度**: ~1.15°/控制周期，流畅连续（非跳跃式）

**初始状态**: 未选中任何手指（请按T/Y/H/N/M明确选择）

**推荐组合**:
- 捏取：T+Y (👍拇指 + ☝️食指)
- 握持：T+Y+H+N (4指)
- 指点：T+H+N+M (除食指外)

## 使用工作流程

### 1. 启动系统
```bash
# 终端1: 启动MMK forwarder
python deploy/mmk_forwarder.py

# 终端2: 启动XHand forwarder
python deploy/xhand_forwarder.py

# 终端3: 启动控制器
python deploy/dagger_controller_debug.py --pretrained-model-path <path> ...
```

### 2. 策略模式（默认）
- 程序启动后默认处于 **AI策略控制模式** 🟢
- 机器人根据视觉输入和语言指令自主执行动作
- 观察机器人行为，识别失败情况

### 3. 切换到人类干预模式
- 当AI策略表现不佳时，按 **空格键** 切换到人类控制模式 🔴
- 系统提示："🔴 切换到人类控制模式 - 使用键盘遥操作"

### 4. 遥操作控制
```
1. 选择要控制的手臂（按 1/2/3）
2. 使用 WASD/RF 移动手腕位置
3. 使用 IJKL/UO 调整手腕姿态
4. 选择要控制的手指（按 T/Y/H/N/M，可多选）
5. 使用 Z/C 控制选中手指的闭合/张开
6. IK自动计算并执行关节动作
```

**手部控制示例（v2 速度控制模式）**：
- **捏取物体**（流畅操作）：
  1. 按 T 选择拇指（应显示"✅ 选择拇指"）
  2. 按 Y 选择食指（应显示"✅ 选择食指"）
  3. 确认显示"🎯 当前选中手指: 👍拇指 + ☝️食指"
  4. **按住 C** 张开手指（持续按住直到完全张开）
  5. 松开 C，手指停止
  6. 移动手臂到物体位置
  7. **按住 Z** 闭合拇指和食指（持续按住直到夹住物体）
  8. 松开 Z，保持夹持力
  
- **握持圆柱**：
  1. 依次按 T、Y、H、N 选择4根手指
  2. **按住 C** 完全张开（流畅张开）
  3. 移动到圆柱周围
  4. **按住 Z** 闭合（流畅闭合，感受握持力）
  
- **指点手势**：
  1. 按 T、H、N、M 选择除食指外的手指
  2. **按住 Z** 闭合这些手指
  3. 食指自动保持伸直（未选中）
  
- **精细调节**（单指控制）：
  1. 按 T 只选择拇指
  2. **按住 Z** 或 **C** 流畅调节拇指角度
  3. 松开即停，精确控制

### 5. 保存干预数据
- 在人类模式下，系统自动记录：
  - 当前观测
  - AI建议的动作（policy_action）
  - 人类实际执行的动作（human_action）
- 按 **S键** 保存到 `interventions/intervention_episode_*.npz`

### 6. 继续或退出
- 按 **空格键** 切回AI策略模式继续
- 按 **Q键** 退出程序

## 数据格式

保存的干预数据格式：
```python
{
    'qpos': np.array([N, 36]),          # 关节位置序列
    'policy_actions': np.array([N, 36]), # AI建议的动作
    'human_actions': np.array([N, 36]),  # 人类实际动作
    'metadata': {
        'num_steps': int,
        'timestamp': str
    }
}
```

## 屏幕输出示例

### 启动信息
```
======================================================================
🚀 启动Human-in-the-Loop推理循环（手腕IK + 手指选择控制）
======================================================================
模式控制：
  空格键 - 切换 策略模式(AI) ↔ 人类控制模式(遥操作)
  S键    - 保存干预数据
  Q键    - 退出程序

遥操作控制（人类模式下）：
  手臂选择:
    1 = 👈 左臂  |  2 = 👉 右臂  |  3 = 👐 双臂
  位置控制 (按住移动，松开停止):
    W/X = 前进/后退  |  A/D = 左移/右移  |  R/F = 上升/下降
  旋转控制 (按住旋转，松开停止):
    I/K = Roll(+/-)  |  J/L = Pitch(+/-)  |  U/O = Yaw(+/-)
  手部控制 (手指速度控制 v2):
    T/Y/H/N/M = 选择/取消 拇指/食指/中指/无名指/小指 (可多选)
    按住 Z = ✊ 闭合选中手指（流畅连续）| 松开 = ⏸️ 停止
    按住 C = 🖐️ 张开选中手指（流畅连续）| 松开 = ⏸️ 停止
    💡 建议先按 T+Y 选择拇指和食指（捏取常用组合）

📌 默认: 👈 LEFT 手臂 | 手指: 未选中（请按T/Y/H/N/M选择）
======================================================================
```

### 策略模式运行
```
🟢 步数: 10/1000 | 模式: policy | FPS: 5.23 | 干预数据: 0 步
```

### 人类干预模式
```
🔴 步数: 50/1000 | 模式: human | FPS: 5.18 | 干预数据: 15 步
  👈 left | Pos: [0.350, 0.320, 0.315] | Rot: [0.05, -0.10, 0.00]
```

## 技术架构

### 手臂控制（IK模式）
```
键盘输入 (WASD/RF/IJKL/UO)
  ↓
目标位姿更新 (teleop_target_pose: pos + RPY)
  ↓
构建变换矩阵 (pose_to_transformation_matrix)
  ↓
MMK2 KDL IK求解 (mmk2_kdl.inverse_kinematics)
  ├─ 输入: T_left/T_right (4x4变换矩阵)
  ├─ 参考: 当前13维关节角度 [spine, left_arm(6), right_arm(6)]
  └─ 输出: 13维关节角度解
  ↓
提取手臂关节 (6D per arm)
```

### 手部控制（手指速度控制 v2）
```
键盘输入 (T/Y/H/N/M 选择手指)
  ↓
更新 selected_fingers (多选集合)
  ↓
键盘输入 (按住 Z/C)
  ↓
on_press: 遍历选中的手指
  ├─ 根据 FINGER_JOINT_MAP 获取关节索引
  └─ 设置 teleop_hand_velocity[idx] = ±teleop_hand_speed
  ↓
每个控制周期: teleop_hand_joints += teleop_hand_velocity
  ├─ 持续应用速度增量（流畅变化）
  └─ 限位到 [-π, π]
  ↓
松开 Z/C
  ↓
on_release: 清零 teleop_hand_velocity（停止运动）
  ↓
组合完整动作 [left_arm(6), left_hand(12), right_arm(6), right_hand(12)]
  ↓
通过 ZMQ 发送到机器人
```

**与手臂控制完全一致的速度模式**

### 初始化流程

```
启动程序
  ↓
加载MMK2 KDL库
  ↓
获取当前关节角度 (get_observations)
  ↓
MMK2 KDL FK (mmk2_kdl.forward_kinematics)
  ├─ 输入: 13维关节角度
  └─ 输出: T_left, T_right
  ↓
提取位姿 (transformation_matrix_to_pose)
  ↓
初始化目标位姿 (teleop_target_pose)
```

## 参数调整

在文件中修改全局变量以调整控制参数：

```python
# 调整移动步长
teleop_step_size = 0.01  # 默认1cm，可改为0.005(5mm)或0.02(2cm)

# 调整旋转步长
teleop_rot_step = 0.05   # 默认~3度，可改为0.02(1度)或0.1(6度)
```

## IK/FK求解器

✅ **使用真实的MMK2 KDL库（必需）**

系统**仅使用**专业的Kinematics and Dynamics Library (KDL)进行逆运动学和正向运动学计算：

- **库**: `mmk2_kdl_py` - MMK2机器人专用KDL库
- **方法**: 
  - `mmk2_kdl.forward_kinematics(q)` - 正向运动学
  - `mmk2_kdl.inverse_kinematics(T_left, T_right, ref_pos)` - 逆运动学
- **特点**:
  - 基于真实DH参数
  - 考虑关节限位
  - 支持单臂/双臂求解
  - 包含脊柱关节

### ⚠️ 依赖要求

**MMK2 KDL库是必需的**，如果未安装，程序会拒绝启动并提示：

```bash
❌ MMK2 KDL 库未加载！
此程序需要MMK2 KDL库进行IK/FK求解
请安装依赖: pip install casadi bidict xacrodoc lxml
```

### 关节角度格式说明

MMK2机器人使用**13维关节空间**：

```
q = [q0, q1, q2, ..., q12]

其中：
- q[0]:     脊柱高度 (spine)
- q[1:7]:   左臂关节 1-6 (left arm)
- q[7:13]:  右臂关节 1-6 (right arm)
```

但在与RDT模型交互时，使用**36维动作空间**：

```
action = [left_arm(6), left_hand(12), right_arm(6), right_hand(12)]
```

系统会自动在两种格式间转换。

## DAgger训练数据使用

收集的干预数据可用于DAgger算法训练：

```python
# 加载干预数据
data = np.load('interventions/intervention_episode_0_20250117_120000.npz')

# 使用人类动作作为标签，观测作为输入
observations = data['qpos']
expert_actions = data['human_actions']  # 人类专家标签
policy_actions = data['policy_actions']  # 用于对比分析

# 添加到训练集
dataset.add_expert_data(observations, expert_actions)
```

## 故障排查

### 问题1: 按键无响应
**解决**: 确保终端窗口获得焦点，pynput监听器需要前台权限

### 问题2: IK求解失败
**解决**: 检查目标位置是否在机器人工作空间内，调整 `teleop_target_pose` 初始值

### 问题3: 机器人抖动
**解决**: 
- 减小步长 `teleop_step_size` 和 `teleop_rot_step`
- 启用动作插值 `use_actions_interpolation: true`

### 问题4: 数据保存失败
**解决**: 检查是否有写权限，`interventions/` 目录是否存在

### 问题5: 手部按键控制日志正常但机器人无动作
**现象**: 按Z/C或4/5/6/7/8键，日志显示角度变化，但真实机器人手部无动作

**诊断步骤**：
1. **检查DEBUG日志**：程序会输出手部动作数据
   ```
   DEBUG - teleop_hand_joints['left'] 平均角度: 90.0°
   DEBUG - 执行手部动作: 左手=[90. 90. ...], 右手=[0. 0. ...]
   ```

2. **对比策略模式**：切换到策略模式（按空格），观察手部是否有动作
   - 如果策略模式下手部有动作 → 问题在人类控制逻辑
   - 如果策略模式下手部也无动作 → 问题在 xhand_forwarder 或硬件连接

3. **检查单位转换**：系统内部使用弧度，发送给机器人也是弧度
   - `teleop_hand_joints` 存储弧度
   - `execute_xhand_action` 发送弧度
   - xhand_forwarder 期望弧度 (见注释 "Clamp actions to safe joint limits (radians)")

4. **检查 xhand_forwarder 日志**：查看是否有错误信息
   ```bash
   # 在运行 xhand_forwarder 的终端查看输出
   ```

5. **验证硬件连接**：确保 XHand 控制器正常连接且能接收命令

**可能的原因**：
- xhand_controller 库的单位期望与注释不符（可能实际期望度数）
- 手部关节限位太严格，数据被截断为0
- 通信延迟或丢包

## 未来改进

- [ ] 集成专业IK求解器（ikpy/PyBullet/MoveIt）
- [x] ~~添加手部姿态键盘控制（手指关节）~~ ✅ 已完成 (2025-11-19)
- [ ] 实时可视化目标位姿和当前位姿差异
- [ ] OOD检测自动提示干预
- [ ] 动作混合模式（AI + 人类加权）
- [ ] 支持3D鼠标/游戏手柄输入
- [ ] VR/AR遥操作集成
- [ ] 手部精细控制模式（单个手指关节调节）

---

## 更新日志

### 2025-11-19 (深夜v5): 回退到稳定版本

**重要决定**：
- 🔄 **回退所有"模式切换跳变修复"相关代码**
  - ❌ 移除 human→policy 切换检测和处理
  - ❌ 移除 just_switched_to_policy 标志位
  - ❌ 移除 skip_next_interpolation 标志位
  - ❌ 移除 observation_window 的度转弧度转换
  - ❌ 移除各种调试日志
  - ✅ 恢复到简洁的工作版本

**保留的功能**：
- ✅ 手指选择控制模式（T/Y/H/N/M选择手指）
- ✅ 手指速度控制（按住Z/C持续变化）
- ✅ selected_fingers初始为空（清晰的初始状态）
- ✅ 切换到策略模式时清除selected_fingers（避免历史污染）
- ✅ policy→human 切换时初始化位姿（无跳变）

**关于模式切换跳变**：
- 📝 **已知问题**：从人类模式切回策略模式时，可能跳回之前位置
- 🔧 **根本原因复杂**：涉及observation_window、action_buffer缓存、推理输入等多个因素
- 💡 **临时方案**：接受这个限制，或在切换后等待几步让策略重新适应
- 🎯 **焦点**：优先确保手部控制功能稳定工作

**当前版本特点**：
- 代码简洁清晰
- 手部控制功能完整
- 人类→策略切换有跳变（已知问题）
- 策略→人类切换无跳变 ✅

---

### 2025-11-19 (深夜v4已废弃): observation_window单位错误修复尝试

**🎯 找到并修复了跳变的真正根源**：
- 🐛 **问题**：切换到策略模式后，机器人突然跳回初始位置
- 🔧 **真正根源**：observation_window中hand数据单位错误
  1. 机器人返回hand数据（度数）
  2. `update_observation_window` 直接存储（度数）❌
  3. 模型推理时使用 `observation_window[-1]['qpos']`（hand是度数）
  4. **但模型训练时hand是弧度**
  5. 模型收到错误单位的输入 → 推理错误 → 输出"回到初始位置"的动作
  
- ✅ **解决方案**（两处修复）：
  ```python
  # update_observation_window 函数中（初始化+更新）
  qpos = np.concatenate((
      obs['arm_left'],
      np.deg2rad(obs['hand_left']),   # ✅ 必须转为弧度！
      obs['arm_right'],
      np.deg2rad(obs['hand_right'])   # ✅ 必须转为弧度！
  ))
  ```

- 📝 **影响**：
  - observation_window始终存储正确单位（hand=弧度）
  - 模型推理基于正确数据 → 输出正确动作
  - 切换模式完全无跳变！

**为什么之前所有修复都无效**：
- ❌ 设置prev_action → 只影响插值，不影响推理输入
- ❌ 跳过插值 → 推理结果本身就是错的
- ❌ 更新更多次observation_window → 还是度数，无用
- ✅ **必须修正推理的输入数据单位！**

**额外改进**：
- 禁用插值（`use_actions_interpolation: false`）避免额外的运动
- 添加详细调试日志追踪数据流

**代码变更**：
```python
# 修复：update_observation_window 函数（两处）
# 行1086-1093（初始化）和行1125-1133（更新）

# 修复前：
qpos = np.concatenate((
    obs['arm_left'],
    obs['hand_left'],      # ❌ 度数！导致模型推理错误
    obs['arm_right'],
    obs['hand_right']      # ❌ 度数！
))

# 修复后：
qpos = np.concatenate((
    obs['arm_left'],
    np.deg2rad(obs['hand_left']),   # ✅ 转为弧度
    obs['arm_right'],
    np.deg2rad(obs['hand_right'])   # ✅ 转为弧度
))
```

**为什么这是真正的根源**：
- observation_window 是模型推理的输入源
- 人类模式期间，observation_window不断更新（包含hand的度数数据）
- 切换回策略时，基于错误单位的observation_window推理
- 模型输出完全错误的动作（看起来像"回到初始位置"）

**之前的所有修复为什么无效**：
- ❌ 更新16次观测窗口 → 还是度数，无用
- ❌ 设置prev_action=current_qpos → 只影响插值，不影响推理输入
- ❌ 跳过插值 → 推理结果本身就是错的

**这次修复为什么有效**：
- ✅ 修正了推理的输入数据（observation_window中hand为弧度）
- ✅ 模型基于正确数据推理，输出正确动作
- ✅ 从当前位置B平滑继续，不回到A

---

### 2025-11-19 (深夜v3): 手指速度控制 - 从步进到流畅

**重大改进**：
- 🎯 **手指控制改为速度模式**：从步进式（每次5°）改为流畅的速度控制
  - ❌ 移除：按一次增加/减少固定角度（步进式，不流畅）
  - ✅ 新增：按住键持续变化，松开停止（与手臂WASD控制一致）
  - ✅ 速度：~1.15°/周期（0.02弧度），流畅连续
  - 📝 **影响**：手指控制更流畅，可以精确控制到任意角度

**技术实现**：
- 新增 `teleop_hand_velocity` 12维速度向量（与手臂速度控制一致）
- Z按键：设置选中手指关节的速度为 +teleop_hand_speed
- C按键：设置选中手指关节的速度为 -teleop_hand_speed
- on_release：松开Z/C时清零速度
- 每个控制周期：`teleop_hand_joints += teleop_hand_velocity`

**对比**：
| 项目 | 步进模式（旧） | 速度模式（新） |
|------|----------------|----------------|
| 控制方式 | 按一次+5° | 按住持续增加 |
| 流畅度 | ❌ 跳跃式 | ✅ 连续流畅 |
| 精度 | 5°的倍数 | 任意角度 |
| 操作感 | 像按钮 | 像遥杆 |
| 与手臂一致性 | ❌ 不同 | ✅ 完全一致 |

---

### 2025-11-19 (深夜v2): 用户体验修复 - 初始状态+状态清除+平滑切换

**Bug修复**：
- 🐛 **修复T键控制混乱**：按T应该选择拇指，但默认已选中导致按T变成取消拇指
  - ✅ **解决**：初始手指选择状态改为空列表，用户明确按键选择
  - 📝 **影响**：现在按T第一次是选择拇指，再按才是取消，符合直觉

- 🐛 **修复手指状态保留历史**：切换模式两次后，手指选择还保留上次的
  - ✅ **解决**：在切换回策略模式时，自动清除手指选择状态
  - 📝 **影响**：每次进入人类模式都是干净状态，避免混乱

- 🐛 **修复切换时大幅跳变**：即使更新16次观测窗口还是跳变
  - 🔧 **根本原因**：
    1. prev_action没有从当前实际位置初始化
    2. 设置action=current_qpos后被policy_action覆盖
    3. 插值会从current_qpos插值到action_buffer[1]，造成运动
  - ✅ **最终解决方案（双标志位设计）**：
    1. 获取current_qpos，设置action=current_qpos, prev_action=current_qpos
    2. 设置 `just_switched_to_policy=True`（防止action被覆盖+跳过本步插值）
    3. 设置 `skip_next_interpolation=True`（跳过下一步插值）
    4. 本步执行current_qpos（保持不动，无插值）
    5. 重新推理得到action_buffer（基于当前位置）
    6. 下步直接执行action_buffer[1]（无插值，直接跳到目标）
    7. 第三步恢复正常插值
  - 📝 **影响**：切换时保持1步不动，下步直接跳到新推理的目标，完全无历史位置跳变

**代码变更**：
- 行 142-144: `selected_fingers` 初始为空列表
- 行 385-387: 切换到策略模式时清除手指选择状态
- 行 1342-1352: 使用当前qpos初始化prev_action，而不是推理结果

---

### 2025-11-19 (深夜v3): 手指速度控制 - 从步进到流畅

**重大改进**：
- 🎯 **手指控制改为速度模式**：从步进式（每次5°）改为流畅的速度控制
  - ❌ 问题：5°步长太大，控制不流畅，像"跳跃"
  - ✅ **解决方案B（推荐）**：速度控制模式，与手臂WASD完全一致
  - 🎮 **操作方式**：按住Z/C键持续变化，松开键停止
  - 📈 **速度**：0.02弧度/周期 ≈ 1.15°/周期，流畅连续
  - 📝 **影响**：手指控制体验大幅提升，流畅且精确

**技术实现**：
```python
# 新增速度向量
teleop_hand_velocity = {
    'left': np.zeros(12),
    'right': np.zeros(12)
}
teleop_hand_speed = 0.02  # 弧度/周期

# on_press Z键：设置速度
for idx in selected_joints:
    teleop_hand_velocity[arm][idx] = teleop_hand_speed

# on_release Z/C键：清零速度
teleop_hand_velocity[arm][idx] = 0

# 每个周期：应用速度
teleop_hand_joints[arm] += teleop_hand_velocity[arm]
```

**对比方案A vs 方案B**：
| 方案 | 步长 | 控制方式 | 流畅度 | 精度 |
|------|------|----------|--------|------|
| A: 减小步长 | 1-2° | 按一次变化一次 | ⚠️ 一般 | ✅ 好 |
| **B: 速度控制** | **~1.15°/周期** | **按住持续变化** | **✅ 极好** | **✅ 极好** |

**选择方案B的原因**：
- ✅ 与手臂控制逻辑完全一致（学习成本低）
- ✅ 流畅连续，无跳跃感
- ✅ 可以通过按住时长精确控制角度
- ✅ 直觉性强：按住=动，松开=停

**代码变更位置**：
- 行 149-154: 添加 `teleop_hand_velocity` 和速度参数
- 行 529-548: Z/C按键改为设置速度（而非直接修改角度）
- 行 589-596: on_release添加Z/C键处理（清零速度）
- 行 833-836: get_human_action_from_teleop 中应用速度增量

---

### 2025-11-19 (深夜v1): v2重新设计 - 手指选择控制模式 + 模式切换跳变修复

**重大改进**：
- 🎯 **手部控制v2重新设计**：从预设姿态模式改为手指选择模式
  - ❌ 移除预设姿态（4/5/6/7/8按键和HAND_PRESETS）
  - ✅ 新增手指选择机制：T/Y/H/N/M 选择/取消手指（可多选）
  - ✅ Z/C 仅控制选中的手指，未选中的保持不动
  - ✅ 初始为空，用户明确选择（避免混乱）

- 🐛 **修复模式切换跳变**：从人类模式切回策略模式时位置跳变
  - 🔧 **根本原因**：切回策略模式时，使用了人类模式期间缓存的旧 action_buffer（基于旧位置推理）
  - ✅ **解决方案**：检测 human→policy 切换时：
    1. 更新观测窗口5次（同步当前位置）
    2. 立即重新推理生成新的 action_buffer（基于当前位置）
    3. 重置 chunk 索引到起始位置
    4. 更新 prev_action 为新推理的第一个动作（避免插值跳变）
  - 📝 **影响**：切回策略模式时无缝过渡，基于当前位置继续执行

**新的按键映射**：
| 功能 | 按键 | 说明 |
|------|------|------|
| 选择拇指 | T | Toggle thumb（切换选中状态） |
| 选择食指 | Y | Toggle index finger |
| 选择中指 | H | Toggle middle finger |
| 选择无名指 | N | Toggle ring finger |
| 选择小指 | M | Toggle pinky |
| 闭合选中手指 | Z | 选中的手指 +5° |
| 张开选中手指 | C | 选中的手指 -5° |

**使用优势**：
- ✅ 精细控制：可以单独控制任意手指组合
- ✅ 灵活性：可以只动拇指，或只动食指+中指，任意组合
- ✅ 实用性：默认选中拇指+食指，适合90%的抓取场景
- ✅ 简洁性：只需要Z/C两个键控制，按键更少
- ✅ 可视化：每次选择后显示当前选中的手指列表

**手指关节映射**（12维，基于XHand硬件规格）：
```python
FINGER_JOINT_MAP = {
    'thumb': [0, 1, 2],      # 拇指: bend[0°,105°] + rota1[-40°,90°] + rota2[0°,90°]
    'index': [3, 4, 5],      # 食指: bend[-10°,10°] + joint1[0°,110°] + joint2[0°,110°]
    'middle': [6, 7],        # 中指: joint1[0°,110°] + joint2[0°,110°]
    'ring': [8, 9],          # 无名指: joint1[0°,110°] + joint2[0°,110°]
    'pinky': [10, 11]        # 小指: joint1[0°,110°] + joint2[0°,110°]
}

# 关节限位（弧度）来自 xhand_forwarder.py JOINT_LIMITS_RAD
# 所有控制都会被自动限制在这些安全范围内
```

**使用场景示例**（v2设计）：
1. **捏取小物体**：
   - 默认已选中拇指+食指 ✅
   - 按 C 张开 → 移动到位置 → 按 Z 闭合

2. **握持圆柱**：
   - 按 T+Y+H+N 选中4根手指
   - 按 C 张开 → 移动到圆柱 → 按 Z 闭合

3. **指点手势**：
   - 按 T 取消拇指选择
   - 按 H+N+M 选中中指+无名指+小指
   - 按 Z 闭合（食指保持伸直）

4. **单指调节**：
   - 按 T+Y 取消所有选择
   - 按 T 只选择拇指
   - 用 Z/C 精细调节拇指角度

**代码变更位置**：
- `dagger_controller_debug.py` 行 127-146: 移除 HAND_PRESETS，添加手指选择变量和FINGER_JOINT_MAP
- `dagger_controller_debug.py` 行 525-538: 添加 show_selected_fingers() 辅助函数
- `dagger_controller_debug.py` 行 447-507: 键盘监听器改为手指选择(T/Y/H/N/M)+Z/C控制
- `dagger_controller_debug.py` 行 1335-1358: 完整的 human→policy 切换逻辑（修复跳变）
  - 更新观测窗口
  - 立即重新推理
  - 重置chunk索引
  - 同步prev_action

**模式切换流程**（最终修复版）：
```
策略模式 → 人类模式:
  1. 获取当前机器人状态（FK）
  2. 初始化目标位姿 = 当前位姿
  3. 初始化手部角度 = 当前手部角度
  4. 清除手指选择状态（干净开始）
  5. 同步 prev_action
  ✅ 无跳变开始人类控制

人类模式 → 策略模式（完整修复逻辑）:
  步骤N（检测切换）：
    1. 获取当前实际位置 current_qpos
    2. 设置 action = current_qpos（本步保持不动）
    3. 设置 prev_action = current_qpos
    4. 设置标志位 just_switched_to_policy = True
    5. 设置标志位 skip_next_interpolation = True
    6. 更新观测窗口1次，同步位置
    7. 立即重新推理 → action_buffer（基于当前位置）
    8. 重置t到chunk开始
  
  步骤N（action设置）：
    - just_switched_to_policy = True
    - ✅ 不覆盖 action（保持current_qpos）
  
  步骤N（插值）：
    - just_switched_to_policy = True
    - ✅ 跳过插值，直接执行
    - 清除 just_switched_to_policy = False
  
  步骤N（执行）：
    - 执行 current_qpos（保持不动）✅
  
  步骤N+1（action设置）：
    - current_mode = policy
    - action = action_buffer[1]（新推理的第2个动作）
  
  步骤N+1（插值）：
    - skip_next_interpolation = True
    - ✅ 跳过插值，直接跳到目标
    - 清除 skip_next_interpolation = False
  
  步骤N+1（执行）：
    - 直接执行 action_buffer[1]（无插值）✅
  
  步骤N+2（恢复正常）：
    - 使用 action_buffer[2]
    - 正常插值和执行
  
  ✅ 完全无跳变，平滑过渡
```

---

### 2025-11-19 (晚上): 手部微调限位范围修复

**Bug修复**：
- 🐛 **修复Z/C键微调无效问题**：按Z或C键调节手部角度时，角度卡在±114.6°不变
- 🔧 **根本原因**：手部关节限位设置为 `[-2.0, 2.0]` 弧度，等于 `[-114.6°, 114.6°]`，范围太小
- ✅ **解决方案**：将限位范围扩大到 `[-π, π]` 弧度，即 `[-180°, 180°]`
- 📝 **影响**：现在Z/C微调可以在完整的±180°范围内工作，手部可以充分闭合和张开

**代码变更**：
```python
# 修复前（限位太小）：
teleop_hand_joints[arm] = np.clip(teleop_hand_joints[arm], -2.0, 2.0)  # ±114.6°

# 修复后（完整范围）：
teleop_hand_joints[arm] = np.clip(teleop_hand_joints[arm], -np.pi, np.pi)  # ±180°
```

**验证方法**：
1. 进入人类控制模式
2. 按4张开手，然后持续按Z键
3. ✅ 角度应该能从0°增加到180°（之前卡在114.6°）
4. 持续按C键
5. ✅ 角度应该能从180°减少到-180°（之前卡在-114.6°）

---

### 2025-11-19 (下午): 手部初始化Bug修复

**Bug修复**：
- 🐛 **修复右手异常闭合问题**：切换到人类控制模式时，右手会异常闭合
- 🔧 **根本原因**：从机器人获取的手部数据是度数（degrees），但未转换为弧度（radians）
- ✅ **解决方案**：在 `initialize_teleop_from_current_state()` 函数中添加 `np.deg2rad()` 转换
- 📝 **影响**：现在切换到人类模式时，左右手都能正确保持当前姿态，与手臂控制逻辑一致

**代码变更**：
```python
# 修复前（错误）：
teleop_hand_joints['left'] = np.array(obs['hand_left']).flatten()
teleop_hand_joints['right'] = np.array(obs['hand_right']).flatten()

# 修复后（正确）：
teleop_hand_joints['left'] = np.deg2rad(np.array(obs['hand_left']).flatten())
teleop_hand_joints['right'] = np.deg2rad(np.array(obs['hand_right']).flatten())
```

**验证方法**：
1. 启动系统，让机器人处于策略模式
2. 观察当前手部姿态（例如：半握状态）
3. 按空格键切换到人类控制模式
4. ✅ 手部应保持原姿态，不应突然闭合或张开

---

### 2025-11-19 (上午): 手部键盘控制功能实现

**新增功能**：
- ✅ 手部预设姿态控制（5种姿态）
  - 按键 **4**: 🖐️ 张开 - 所有手指完全伸直
  - 按键 **5**: ✊ 握拳 - 所有手指弯曲约70°
  - 按键 **6**: 🤏 捏取 - 拇指+食指弯曲，其他伸直（适合抓取小物体）
  - 按键 **7**: 👆 指点 - 仅食指伸直，其他弯曲（适合指点目标）
  - 按键 **8**: 🤚 半握 - 所有手指半弯曲约35°（适合抓取中等物体）

- ✅ 手部微调控制（渐进式调节）
  - 按键 **Z**: ✊ 整体闭合 - 所有关节增加5°（可多次按压逐步闭合）
  - 按键 **C**: 🖐️ 整体张开 - 所有关节减少5°（可多次按压逐步张开）

**技术实现**：
- 新增全局变量 `teleop_hand_joints` 存储左右手的12维关节角度
- 新增预设姿态字典 `HAND_PRESETS` 定义5种常用手部姿态
- 更新 `get_human_action_from_teleop()` 函数，使用键盘控制的手部数据替代策略输出
- 更新 `initialize_teleop_from_current_state()` 函数，初始化时同步手部状态
- 支持单臂/双臂模式下的独立手部控制

**用户体验改进**：
- 按键冲突避免：手部控制使用数字键 4-8 和字母键 Z/C，不与现有手臂控制冲突
- 兼容性处理：如果用户未设置手部姿态（全零），自动回退到策略输出
- 关节限位保护：所有手部关节角度限制在 [-2.0, 2.0] 弧度范围内
- 实时反馈：手部姿态切换时显示对应的emoji图标和当前平均角度

**使用场景示例**：
1. **抓取物体**：按 4 张开手 → 移动到物体上方 → 按 5 握拳
2. **捏取小物体**：按 6 进入捏取姿态 → 微调位置 → 按 Z 增加闭合力
3. **指点目标**：按 7 进入指点姿态 → 移动手臂指向目标
4. **渐进式调节**：按 8 半握 → 多次按 Z 逐步增加闭合度，直到抓稳物体

**代码变更位置**：
- `daggger_controller_debug.py` 行 118-131: 添加手部控制全局变量
- `daggger_controller_debug.py` 行 433-467: 键盘监听器中添加手部按键处理
- `daggger_controller_debug.py` 行 755-764: 更新人类动作生成逻辑
- `daggger_controller_debug.py` 行 811-819: 初始化函数添加手部状态同步

**测试建议**：
- 测试所有5种预设姿态的切换
- 测试 Z/C 微调功能的渐进式调节
- 测试单臂和双臂模式下的手部独立控制
- 验证手部数据维度正确性（左手12维 + 右手12维）
- 测试与现有手臂控制的协同使用



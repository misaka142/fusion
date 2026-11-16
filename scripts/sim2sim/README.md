# SuperMimic × SUGAR 本地 MuJoCo Sim-to-Sim 部署指南

本目录提供 **零 Isaac Sim / 零 Isaac Lab 依赖** 的本地 Windows/Linux MuJoCo Sim-to-Sim 部署工具，原生支持 SUGAR 全部 **6 个人-物全身交互任务**：

| 任务名称 (`--task`) | 交互物体 | 物体质量 | 默认目标条件 (`use_target`) |
| :--- | :--- | :---: | :---: |
| `CarryBox` | `small_box` | `0.50 kg` | `True` |
| `PushBox` | `big_box` | `0.75 kg` | `True` |
| `KickBox` | `big_box` | `0.75 kg` | `False` |
| `PickBottle` | `bottle` (`angular_damping=0.2`) | `0.75 kg` | `False` |
| `StandBottle` | `bottle` (`angular_damping=0.2`) | `0.75 kg` | `False` |
| `SitChair` | `chair` (多 Box L 型凹面分解) | `3.00 kg` | `False` |

---

## 1. 本地 Windows 环境准备

你本地的 `D:\anaconda\envs\DRL_env` 已配置好所需轻量依赖；若在新环境部署，仅需执行：

```powershell
D:\anaconda\envs\DRL_env\python.exe -m pip install --upgrade-strategy only-if-needed mujoco trimesh diffusers hydra-core omegaconf dill imageio imageio-ffmpeg
# （可选）若希望直接从 descriptions/objects/*/obj_aligned.usd 提取原始物体网格为 STL：
D:\anaconda\envs\DRL_env\python.exe -m pip install usd-core
```

---

## 2. 放置资产与模型权重

将 Colab 训练导出的模型（或 SUGAR 官方 `demo_ckpts`）与静态资产按如下结构放在 `D:\robot\rework\fusion` 下（也可通过命令行参数任意指定路径）：

```text
D:\robot\rework\fusion\
├── descriptions\                  # 解压自 SUGAR descriptions.zip（可选，未放时自动使用内置 CAD 惯量 + 橡胶手碰撞体）
├── data\
│   └── CarryBox\data_000\...      # 模式一（replay）读取参考轨迹时使用
└── outputs\
    └── CarryBox_colab\ckpts\
        ├── tracker.pt             # 510 维底层 SuperMimic Tracker（同时兼容官方 demo_ckpts tracker.pt）
        └── generator.ckpt         # 10Hz 高层 Diffusion Generator（autonomous 模式使用）
```

---

## 3. 两种运行模式（Windows PowerShell）

首先进入项目根目录：

```powershell
cd D:\robot\rework\fusion
```

### 模式一：单 Tracker 参考轨迹回放（`--mode replay`）
不接入高层 `Generator`，直接从 `data/<TASK>` 读取 50Hz 参考动作指令喂给 `tracker.pt`，用于第一步独立验证物理步长、BFS↔SDK 关节双射映射、转子惯量 `armature` 与 500Hz 隐式 PD（对齐 IsaacLab `ImplicitActuatorCfg`）控制稳定性：

```powershell
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py `
  --task CarryBox `
  --mode replay `
  --tracker_ckpt outputs/CarryBox_colab/ckpts/tracker.pt `
  --motion_folder data/CarryBox `
  --motion_index 0
```
- `--task`：指定 6 个任务之一（`CarryBox` / `PushBox` / `KickBox` / `PickBottle` / `StandBottle` / `SitChair`）。
- `--motion_folder`：包含 `data_xxx` 子目录的数据集路径。
- `--motion_index`：选择回放第几条参考轨迹（默认 `0`）。

### 模式二：10Hz Generator + 50Hz Tracker 全自主闭环（`--mode autonomous`）
同时加载 `generator.ckpt` 与 `tracker.pt`，甚至**无需本地下载 `data/` 数据集**即可自定义初始物体位置与目标位置，完成端到端自主全身交互：

```powershell
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py `
  --task CarryBox `
  --mode autonomous `
  --tracker_ckpt outputs/CarryBox_colab/ckpts/tracker.pt `
  --generator_ckpt outputs/CarryBox_colab/ckpts/generator.ckpt `
  --obj_pos 0.45 0.0 0.15 `
  --target_pos 1.20 0.25 0.15
```
- 若同时传入 `--motion_folder data/CarryBox --motion_index 0`，则自动采用该轨迹第 0 帧状态作为起点、最后一帧物体位姿作为目标点。
- 若需后台无窗口评测并导出高清演示视频，追加 `--headless --steps 500 --record_mp4 outputs/carrybox_sim2sim.mp4`。

---

## 4. 3D 交互窗口 (`mujoco.viewer`) 操作技巧

1. **一键重置 (`R` 键)**：在 3D 窗口中按下键盘 `R`，机器人与物体立即回到初始状态重新执行。
2. **暂停 / 单步观察 (`Space` 键)**：按空格键随时暂停或恢复物理仿真。
3. **实时拖动目标点 (`target_vis`)**：在 `autonomous` 模式下，鼠标左键**双击绿色半透明目标虚影**，然后按住 **`Ctrl` + 鼠标右键拖拽**平移（或 **`Ctrl` + 鼠标左键拖拽**旋转），高层扩散规划器会在下一次重规划周期（每 `0.4s`）自动引导机器人把物体搬向新目标。
4. **外力抗干扰测试**：双击机器人躯干或手中的箱子，按住 **`Ctrl` + 鼠标右键拖拽**施加外力推搡，测试策略平衡恢复与重抓握能力。

---

## 5. 本地自动化验证（无需外部权重即可一键自检）

在拿到 Colab 训练权重之前，可随时运行内置的 5 项 Sim-to-Sim 端到端测试套件（涵盖全 6 任务场景编译、双格式权重加载、510 维观测 `1e-7` 级精度对齐、3.0s 闭环站立稳定与扩散规划器闭环 MP4 录制）：

```powershell
D:\anaconda\envs\DRL_env\python.exe D:\robot\rework\fusion\tests\test_mujoco_sim2sim.py
```

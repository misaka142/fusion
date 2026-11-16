# SuperMimic × SUGAR 本地 MuJoCo Sim-to-Sim 完整说明文档

本文档面向 **“本地代码上传 GitHub → Google Colab 训练 → 下载模型回本地 Windows 电脑运行 MuJoCo Sim-to-Sim”** 的工作流，详细说明如何在零 Isaac Sim / 零 Isaac Lab 依赖的本地 Windows 环境中完成模型部署、交互测试与视频录制。

---

## 1. 架构概览与支持的 6 大任务

### 1.1 多频闭环控制架构

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  高层规划器 (10Hz / 每 0.4s 重规划, 仅 autonomous 模式启用)             │
│  GeneratorWrapper (generator.ckpt)                                      │
│  输入: torso_link 局部系下当前物体位姿 + 目标物体位姿 + 上一帧 36 维指令 │
│  输出: 未来动作序列 36 维指令 [ref_q_bfs(29), v_b(3), w_b(3), contact(1)]│
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ 50Hz 逐步读取指令 (自动完成 BFS -> Policy 关节序对齐)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  底层全身跟踪器 (50Hz, control_dt = 0.02s)                              │
│  SuperMimicPolicy / 官方 RSL-RL Actor (tracker.pt)                      │
│  输入: 510 维观测 = 指令(36) + 5帧历史本体感知(465) + 局部系物体位姿(9)  │
│  输出: 29 维目标关节位置 q_target = q_default + action_scale * action    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ 每个控制步执行 decimation = 10 个物理子步
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  本地 MuJoCo 物理引擎 (500Hz, sim_dt = 0.002s, integrator=implicitfast) │
│  • 29 路隐式 PD 执行器 (<position kp kv forcerange>) + 电机转子惯量     │
│  • 橡胶手掌碰撞体 + 物体旋转角阻尼 + 实时回读鼠标拖拽的目标虚影位姿      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 全 6 个任务规格对照表

场景构建器 [`scripts/sim2sim/scene_builder.py`](../scripts/sim2sim/scene_builder.py) 已内置 SUGAR 全部 6 个任务的物理参数（与 `sugar_rl/assets/objects/objects.py` 及 `train.sh` 完全一致）：

| 任务名称 (`--task`) | 交互物体子目录 | 物体质量 | 旋转角阻尼 | 摩擦系数 `(滑动, 扭转, 滚动)` | 训练默认 `use_target` | 碰撞几何处理 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `CarryBox` | `small_box` | `0.50 kg` | `0.01` | `(0.75, 0.008, 0.0005)` | `True` | 原生 Box 碰撞体 |
| `PushBox` | `big_box` | `0.75 kg` | `0.01` | `(0.55, 0.005, 0.0005)` | `True` | 原生 Box 碰撞体 |
| `KickBox` | `big_box` | `0.75 kg` | `0.01` | `(0.55, 0.005, 0.0005)` | `False` | 原生 Box 碰撞体 |
| `PickBottle` | `bottle` | `0.75 kg` | `0.20` | `(0.85, 0.010, 0.0005)` | `False` | 瓶身 + 瓶颈双圆柱组合 |
| `StandBottle` | `bottle` | `0.75 kg` | `0.20` | `(0.85, 0.010, 0.0005)` | `False` | 瓶身 + 瓶颈双圆柱组合（初始横躺） |
| `SitChair` | `chair` | `3.00 kg` | `0.05` | `(0.85, 0.010, 0.0010)` | `False` | 座面 + 椅背 + 四腿 L 型多 Box 凹面分解 |

---

## 2. Sim-to-Sim 跨物理引擎关键对齐说明

从 IsaacLab (PhysX) 迁移到 MuJoCo 时，代码已在底层自动处理以下核心物理与工程差异：

1. **关节排列顺序双向置换（BFS ↔ SDK）**：
   - IsaacLab 解析 URDF 默认按广度优先（BFS）排列 29 个关节（`left_hip_pitch`, `right_hip_pitch`, `waist_yaw`, ...），SUGAR 数据集 `robot_50hz.npz` 与 `Generator` 输出的前 29 维也固定采用此顺序。
   - MuJoCo 运动学树按深度优先（SDK）排列关节（左腿 6 轴 → 右腿 6 轴 → 腰部 3 轴 → 左臂 7 轴 → 右臂 7 轴）。
   - [`policy_runner.py`](../scripts/sim2sim/policy_runner.py) 通过关节名称精确建立索引置换表，并支持通过 `default_joint_pos` 指纹自动识别未写入关节名元数据的旧版权重。
2. **510 维观测与局部坐标系严格同构（实测误差 `1.19e-07`）**：
   - 四元数统一采用 `wxyz` 顺序，物体 6D 朝向采用 SUGAR 行优先前两行展平定义，物体相对位姿以 `torso_link`（而非 `pelvis`）为锚点坐标系。
   - MuJoCo `<freejoint>` 的 `qvel[3:6]` 原生即为 `pelvis` 局部坐标系角速度，直接对齐 `base_ang_vel` 观测，避免二次逆旋转错误。
3. **隐式 PD 执行器与电机转子惯量对齐**：
   - 采用 MuJoCo `implicitfast` 积分器搭配 `<position kp="..." kv="..." forcelimited="true" forcerange="..."/>`，与 IsaacLab 的 `ImplicitActuatorCfg` 在刚度、隐式阻尼及力矩饱和限幅上完全等价（如膝关节限幅 `±139 N·m`、踝/腰关节 `±50 N·m`、手臂 `±25 N·m`、手腕 `±5 N·m`），并注入全部 29 个电机的转子等效惯量 `armature`。
4. **手部抓握碰撞与球铰簇自穿模排除**：
   - 显式开启 `left_rubber_hand` 与 `right_rubber_hand` 的高摩擦碰撞体（修复常见 mocap XML 关闭手部碰撞导致无法搬箱/抓瓶的问题），并自动排除复合 3 轴球铰关节簇（髋、腰、肩、腕、踝）内部的相邻凸包自穿模。

---

## 3. 本地 Windows 环境准备与自检验证

### 3.1 Python 环境依赖安装（Windows 本地 PowerShell）

你的本地环境 `D:\anaconda\envs\DRL_env` 已经安装配置完毕。若后续在其他电脑或新 Conda 环境复现，可在 **Windows PowerShell** 中执行：

```powershell
D:\anaconda\envs\DRL_env\python.exe -m pip install --upgrade-strategy only-if-needed `
  mujoco trimesh diffusers hydra-core omegaconf dill imageio imageio-ffmpeg
```
- **作用**：在不改动既有 PyTorch CUDA 版本的前提下，安装 MuJoCo 物理引擎、网格处理、扩散模型调度器与离屏 MP4 编码依赖。
- **参数含义**：`--upgrade-strategy only-if-needed` 确保已满足条件的包（如 `torch`、`numpy`）不被重新升级覆盖。
- **预期结果**：输出 `Successfully installed mujoco-...`，且 `torch.cuda.is_available()` 保持 `True`。
- **可选扩展**：若希望直接从 `descriptions/objects/*/obj_aligned.usd` 自动提取原始三维网格为 `.stl`，可额外执行 `D:\anaconda\envs\DRL_env\python.exe -m pip install usd-core`。

### 3.2 本地无权重一键自检测试（Windows 本地 PowerShell）

在尚未从 Colab 下载真实训练权重前，可随时运行端到端验证测试：

```powershell
D:\anaconda\envs\DRL_env\python.exe D:\robot\rework\fusion\tests\test_mujoco_sim2sim.py
```
- **作用**：一键验证全 6 任务场景编译、SuperMimic / 官方 RSL-RL 双格式权重加载、510 维观测对齐精度、500Hz 隐式 PD 站立稳定性以及 `autonomous` 闭环 MP4 录制。
- **参数含义**：直接运行脚本即可，内部使用临时目录自动构造测试检查点并在结束后自动清理。
- **预期结果**：依次通过 `[Test 1]` 至 `[Test 5]`，末尾打印 `ALL 5 MUJOCO SIM-TO-SIM END-TO-END TESTS PASSED SUCCESSFULLY!`。

---

## 4. 从 Colab 下载模型与本地目录结构

在 Colab 完成训练后，你只需要下载部署阶段所需的两个检查点（**注意：Sim-to-Sim 部署使用的是 510 维的 `tracker.pt`，不需要 Stage 1 的 890 维 `refiner.pt`**），推荐按如下结构摆放在 `D:\robot\rework\fusion` 中（也可放在任意位置并通过命令行参数指定路径）：

```text
D:\robot\rework\fusion\
├── sim2sim\
│   └── README.md                      # 本文档
├── scripts\sim2sim\
│   ├── run_sim2sim.py                 # Sim-to-Sim 统一启动入口脚本
│   ├── policy_runner.py               # 50Hz/10Hz/500Hz 闭环推理控制器
│   └── scene_builder.py               # 6 任务 MuJoCo XML 场景生成器
├── descriptions\                      # [可选] 解压 SUGAR descriptions.zip 后放入，自动启用精细 STL 外观网格
│   ├── robots\g1\meshes\*.STL
│   └── objects\small_box\...
├── data\                              # [replay 模式必需 / autonomous 模式可选] 参考动作数据集
│   └── CarryBox\data_000\...
└── outputs\
    └── CarryBox_colab\ckpts\
        ├── tracker.pt                 # [必需] Stage 2 训练导出的底层 SuperMimic Tracker (或官方 demo_ckpts tracker.pt)
        └── generator.ckpt             # [autonomous 模式必需] Stage 3 训练导出的高层扩散规划器
```

> **提示**：即使本地暂未放置 `descriptions\` 文件夹，程序也会自动使用内置的 G1 精确 CAD 质量惯量参数、脚掌接触垫与橡胶手掌碰撞体直接运行物理仿真。

---

## 5. 本地 Windows PowerShell 运行指南

每次运行前，先在 **Windows PowerShell** 中进入 `fusion` 项目根目录：

```powershell
cd D:\robot\rework\fusion
```

### 5.1 模式一：单 Tracker 参考轨迹回放（`--mode replay`）

建议拿到模型后**第一步先运行 `replay` 模式**：不接入高层 `Generator`，直接从 `data/<TASK>` 读取 50Hz 参考动作指令喂给 `tracker.pt`，独立检验底层控制策略在 MuJoCo 中的跟踪与平衡表现。

```powershell
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py `
  --task CarryBox `
  --mode replay `
  --tracker_ckpt outputs/CarryBox_colab/ckpts/tracker.pt `
  --motion_folder data/CarryBox `
  --motion_index 0
```
- **作用**：加载 `CarryBox` 第 `0` 条参考轨迹初始化机器人和箱子位姿，在 3D 交互窗口中以 50Hz 运行 `tracker.pt` 跟踪该参考动作（播完后自动循环重置）。
- **参数含义**：
  - `--task CarryBox`：选择任务场景与对应物体物理属性；
  - `--mode replay`：启用纯参考轨迹回放模式；
  - `--tracker_ckpt`：510 维底层跟踪策略 `.pt` 文件路径（同时兼容 SuperMimic 自训权重与官方 `demo_ckpts` 权重）；
  - `--motion_folder`：包含 `data_000` 等子目录的参考动作文件夹；
  - `--motion_index 0`：指定读取第几条轨迹（默认 `0`）。
- **预期结果**：终端打印 `[Sim2Sim] Task : CarryBox (mode='replay')` 与模型信息，弹出 `mujoco.viewer` 3D 窗口，机器人平稳站立并按参考轨迹完成全身动作。

---

### 5.2 模式二：10Hz Generator + 50Hz Tracker 全自主闭环（`--mode autonomous`）

同时加载高层扩散规划器 `generator.ckpt` 与底层跟踪器 `tracker.pt`。此模式**无需本地下载 `data/` 数据集**，可直接自定义初始物体坐标与目标点坐标：

```powershell
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py `
  --task CarryBox `
  --mode autonomous `
  --tracker_ckpt outputs/CarryBox_colab/ckpts/tracker.pt `
  --generator_ckpt outputs/CarryBox_colab/ckpts/generator.ckpt `
  --obj_pos 0.45 0.0 0.15 `
  --target_pos 1.20 0.25 0.15
```
- **作用**：在无参考轨迹输入的情况下，由高层扩散模型根据当前物体位姿和绿色目标虚影位姿每 `0.4s`（20 个控制步）在线规划未来全身关节与接触指令，驱动机器人在 MuJoCo 中自主搬箱到目标位置。
- **参数含义**：
  - `--mode autonomous`：启用高层扩散规划 + 底层跟踪全自主闭环模式；
  - `--generator_ckpt`：Stage 3 训练保存的 `.ckpt` 检查点路径；
  - `--obj_pos X Y Z`：自定义物体的初始世界坐标（单位：米）；
  - `--target_pos X Y Z`：自定义绿色目标虚影的世界坐标（单位：米）。（若同时传入 `--motion_folder data/CarryBox --motion_index 0`，则会默认取该轨迹首帧作为初始状态、末帧物体位姿作为目标点。）
- **预期结果**：弹出 `mujoco.viewer` 3D 窗口，场景中显示实体箱子与半透明绿色目标虚影，机器人自主走向箱子、弯腰抱起并将其搬运至目标虚影处，终端每 50 步打印实时高度与目标距离 `goal_dist`。

---

### 5.3 其余 5 个任务的自主闭环启动示例

只需替换 `--task` 与对应任务的权重路径即可切换全部 6 个任务（未指定 `--obj_pos` / `--target_pos` 时自动使用各任务合理的默认初始摆位）：

```powershell
# 1. PushBox（推大箱子至目标点，use_target=True）
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py --task PushBox --mode autonomous `
  --tracker_ckpt outputs/PushBox_colab/ckpts/tracker.pt --generator_ckpt outputs/PushBox_colab/ckpts/generator.ckpt

# 2. KickBox（踢大箱子，use_target=False）
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py --task KickBox --mode autonomous `
  --tracker_ckpt outputs/KickBox_colab/ckpts/tracker.pt --generator_ckpt outputs/KickBox_colab/ckpts/generator.ckpt

# 3. PickBottle（捡起地面水瓶，use_target=False）
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py --task PickBottle --mode autonomous `
  --tracker_ckpt outputs/PickBottle_colab/ckpts/tracker.pt --generator_ckpt outputs/PickBottle_colab/ckpts/generator.ckpt

# 4. StandBottle（扶正横躺在地面的水瓶，use_target=False）
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py --task StandBottle --mode autonomous `
  --tracker_ckpt outputs/StandBottle_colab/ckpts/tracker.pt --generator_ckpt outputs/StandBottle_colab/ckpts/generator.ckpt

# 5. SitChair（向后转身坐到身后的椅子上，use_target=False）
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py --task SitChair --mode autonomous `
  --tracker_ckpt outputs/SitChair_colab/ckpts/tracker.pt --generator_ckpt outputs/SitChair_colab/ckpts/generator.ckpt
```
- **作用**：分别加载其余 5 个全身交互任务的专属 MuJoCo 场景（自动切换物体质量、摩擦、初始朝向及椅子凹面碰撞分解）并启动自主闭环仿真。
- **参数含义**：`--task` 指定任务名，未传 `--obj_pos` 时自动采用 `SIM2SIM_TASK_SPECS` 内置默认坐标（如 `StandBottle` 水瓶默认横躺四元数 `(0.7071, 0, 0.7071, 0)`，`SitChair` 椅子默认位于机器人正后方 `(-0.48, 0, 0.41)`）。
- **预期结果**：对应任务的 3D 交互窗口正常启动，机器人与对应物体完成全身接触操作。

---

### 5.4 后台无窗口评测与高清 MP4 视频导出

若需要批量统计指标或直接保存汇报演示视频，可附加 `--headless` 与 `--record_mp4`：

```powershell
D:\anaconda\envs\DRL_env\python.exe scripts/sim2sim/run_sim2sim.py `
  --task CarryBox `
  --mode autonomous `
  --tracker_ckpt outputs/CarryBox_colab/ckpts/tracker.pt `
  --generator_ckpt outputs/CarryBox_colab/ckpts/generator.ckpt `
  --headless `
  --steps 500 `
  --record_mp4 outputs/videos/carrybox_sim2sim.mp4 `
  --video_width 1280 `
  --video_height 720
```
- **作用**：不弹出 GUI 窗口，在后台全速仿真 `500` 个控制步（即 `10.0` 秒仿真时长），使用自动跟随 `pelvis` 的离屏相机渲染并导出 50fps 高清 MP4 视频。
- **参数含义**：
  - `--headless`：关闭交互式 3D 窗口，全速后台运行；
  - `--steps 500`：运行 500 个 50Hz 控制步（共 5000 个 MuJoCo 物理子步）；
  - `--record_mp4`：输出 `.mp4` 文件路径；
  - `--video_width` / `--video_height`：导出视频分辨率（默认 `1280x720`）。
- **预期结果**：生成 `outputs/videos/carrybox_sim2sim.mp4` 视频文件，并在终端打印总步数、最低骨盆高度 `min_pelvis_z`、最终目标距离 `final_goal_dist` 与平均仿真帧率。

---

### 5.5 完整命令行参数速查表

| 参数 | 默认值 | 说明 |
| :--- | :---: | :--- |
| `--task` | 必填 | 任务名：`CarryBox` / `PushBox` / `KickBox` / `PickBottle` / `StandBottle` / `SitChair` |
| `--mode` | 自动推断 | `replay`（仅 Tracker）或 `autonomous`（传入 `--generator_ckpt` 时默认启用） |
| `--tracker_ckpt` | 必填 | 510 维 `tracker.pt` 路径（支持 SuperMimic 与官方 `demo_ckpts`） |
| `--generator_ckpt` | `None` | 高层扩散策略 `generator.ckpt` 路径（`autonomous` 模式必填） |
| `--descriptions_dir` | `None` | 可选的 `descriptions/` 资产根目录路径 |
| `--motion_folder` | `None` | `data/<TASK>` 参考轨迹目录（`replay` 模式必填，`autonomous` 可选） |
| `--motion_index` | `0` | 读取 `--motion_folder` 中的第几条 `data_xxx` 轨迹 |
| `--obj_pos` | 任务默认 | 覆盖初始物体位置 `X Y Z`（米） |
| `--obj_quat` | 任务默认 | 覆盖初始物体四元数 `W X Y Z` |
| `--target_pos` | 任务默认 | 覆盖目标位置 `X Y Z`（米） |
| `--target_quat` | 任务默认 | 覆盖目标四元数 `W X Y Z` |
| `--sim_dt` | `0.002` | MuJoCo 物理步长（`0.002s` = 500Hz） |
| `--control_dt` | `0.02` | 策略控制步长（`0.02s` = 50Hz） |
| `--generator_interval` | `20` | 高层扩散规划间隔步数（`20` 步 = `0.4s`） |
| `--device` | 自动检测 | 推理设备（优先 `cuda`，无 GPU 自动回退 `cpu`） |
| `--steps` | `None` | 最大运行控制步数（窗口模式默认不限步数，无头/录屏默认 `500` 步） |
| `--headless` | `False` | 无 3D 窗口后台运行 |
| `--no_realtime` | `False` | 在 3D 窗口中取消真实时间休眠锁帧，以最高算力速度运行 |
| `--record_mp4` | `None` | 录制 50fps `.mp4` 视频的保存路径 |

---

## 6. `mujoco.viewer` 3D 窗口交互技巧

在交互窗口运行期间，你可以直接使用键盘和鼠标进行在线交互实验：

1. **一键重置环境（`R` 键）**：按下键盘 `R`，机器人、交互物体与历史缓冲区立即回到初始状态重新开始。
2. **暂停 / 恢复仿真（`Space` 空格键）**：随时冻结画面检查接触点、关节力矩或相机视角，再按空格继续。
3. **实时拖动目标点在线变轨（仅 `autonomous` 且 `use_target=True` 任务）**：
   - 在窗口中用**鼠标左键双击**绿色的半透明目标虚影（`target_vis`）将其选中；
   - 按住 **`Ctrl` + 鼠标右键拖拽**可在水平/垂直方向平移目标点（按住 **`Ctrl` + 鼠标左键拖拽**可旋转目标朝向）；
   - `MuJoCoSim2SimRunner` 每步自动读取虚影的新坐标，高层 `Generator` 会在下一个重规划周期（`≤ 0.4s`）实时引导机器人把箱子搬往或推往新目标。
4. **外力推搡抗干扰测试**：鼠标左键双击机器人躯干（`torso_link`）或手中的物体，按住 **`Ctrl` + 鼠标右键拖拽**即可施加实时外力扰动，直观检验策略的平衡恢复与重抓握鲁棒性。

---

## 7. 常见问题排查（Troubleshooting）

- **报错提示传入了 890 维 `refiner.pt`**：
  Stage 1 的 `refiner.pt` 依赖仿真器未来多帧特权真值（890 维），仅用于生成精炼数据集和作为 Stage 2 的冻结教师网络；本地 Sim-to-Sim 部署必须加载 Stage 2 训练输出的 510 维 **`tracker.pt`**。
- **窗口中机器人显示为深灰/浅灰方块与胶囊骨架，而非精细银黑网格**：
  这是未放置 `descriptions/` 资产时的正常内置回退模式（CAD 质量惯量与手脚碰撞完全一致，不影响物理仿真）。若需精细外观，将 SUGAR 的 `descriptions.zip` 解压到 `D:\robot\rework\fusion\descriptions`（或传入 `--descriptions_dir`）即可自动切换为高精度 `.STL` 外观。
- **策略在 `autonomous` 下表现异常，不确定是 `Generator` 还是 `Tracker` 的问题**：
  先切回 `--mode replay` 喂真实数据集轨迹单独测试 `tracker.pt`。若 `replay` 稳定跟踪，说明底层物理与 `Tracker` 正常，需检查 `Generator` 训练轮数或目标坐标是否超出了训练分布范围（例如目标点距离初始点过远时可分步拖动绿色目标虚影引导）。

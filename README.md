# SuperMimic × SUGAR 融合工程 (`fusion`)

> **[重要源码来源与未开源缺失声明]**
> 1. **SuperMimic 官方源码缺失声明**：**SuperMimic 原论文官方代码尚未开源（官方源码缺失）**。本工程中的 `sugar_rl.supermimic` 包（基于用户本地参考目录 `allmimic-main` 重构移植）属于**依据论文算法原理自行编写的非官方融合实现，并非原论文作者发布的官方源码**。
> 2. **SUGAR 官方源码基础**：本工程的全身人-物交互仿真任务（`sugar_rl/tasks`）、URDF/MJCF 资产与高层扩散策略生成器（`sugar_il`）来自已核实的 **SUGAR 官方开源代码**。
> 3. **验证边界声明**：本融合工程已通过本地 PyTorch CUDA 9 项核心数学、运动学、观测切片对齐与 32 步人-物耦合 BPTT 回归测试；**尚未在远程服务器 Isaac Sim 5.1.0 环境上完成完整长周期训练验证**，请先按下方冒烟步骤在空闲 GPU 上验证后再启动完整训练。

本项目将 **SuperMimic**（基于可微学习世界模型的 32 步想象展开策略梯度 BPTT + 多头 Critic 引导）移植至 **SUGAR**（Synthesize Whole-Body Loco-Manipulation via Decoupled Tracking and Generation）全身人-物交互框架中，**同时替换原版 SUGAR 中的 Stage 1 `Refiner`（原 RSL-RL PPO）与 Stage 2 `Tracker`（原 RSL-RL BCPPO）**，并与原版 Rollout 数据转换管线及 Stage 3 扩散策略 `Generator`（`sugar_il`）保持 **100% 接口与数据格式兼容**。

---

## 1. 整体融合流水线架构

```text
原始人-物参考动作 (data/<TASK>)
   │
   ▼  [阶段 1] trainreg run -- python -u scripts/supermimic/train.py (--task *-Refiner)
Stage 1: SuperMimic Refiner
   • 890 维特权未来窗口观测（含启动时 18 项切片级对齐硬断言）+ 84 维人-物联合世界模型 JointWorldModel
   • H=32 步想象轨迹展开 BPTT（对齐 ref[t_cur] 奖励与软终止）+ Value-Clipped Ensemble Critic Bootstrap
   │
   ▼  [阶段 2] python -u scripts/supermimic/play.py (--task *-Refiner-Rollout) + process_refiner_rollout.py
物理精炼参考数据集 (rollout_datasets/refiner/rl_dataset)
   │
   ▼  [阶段 3] trainreg run -- python -u scripts/supermimic/train.py (--task *-Tracker --teacher_ckpt refiner.pt)
Stage 2: SuperMimic Tracker
   • 510 维单帧指令 + 5 帧滚动本体感知历史观测（可微滚动缓冲更新，精确补偿 per-env 关节零位偏置）
   • 同步利用 JointWorldModel 32 步 BPTT 梯度 + 冻结 Stage 1 SuperMimic Refiner 动态退火 BC 蒸馏
   │
   ▼  [阶段 4] python -u scripts/supermimic/play.py (--task *-Tracker-Rollout) + process_tracker_rollout.py
模仿学习 Zarr 数据集 (rollout_datasets/tracker/il_dataset)
   │
   ▼  [阶段 5] trainreg run -- python -u scripts/sugar_il/train.py
Stage 3: Diffusion Command Generator (sugar_il)
   • 根据当前机器人状态与场景物体位姿，扩散生成 36 维单帧全身关节+接触指令
   │
   ▼  [阶段 6] python -u scripts/supermimic/play.py (--task *-Inference --generator_checkpoint generator.ckpt)
50 Hz 全自主闭环推理与录屏评估 (Generator + SuperMimic Tracker)
```

---

## 2. 核心模块目录结构

```text
fusion/
├── SuperMimic_SUGAR_Repro.ipynb           # 交互式端到端复现 Notebook（支持 Colab 与远程 Ubuntu 服务器）
├── train_fusion.sh                        # 5 阶段全流程融合训练脚本（内置 trainreg 守护校验与退出码 139 容错）
├── inference_fusion.sh                    # 闭环推理与 MP4 视频录制一键脚本
├── tests/
│   └── test_supermimic_fusion.py          # 无需 Isaac Sim 的纯 PyTorch GPU/CPU 9 项核心与回归测试套件
├── scripts/
│   ├── supermimic/
│   │   ├── train.py                       # SuperMimic Refiner / Tracker 统一训练 CLI 入口（内置 trainreg 强校验）
│   │   └── play.py                        # SuperMimic Rollout 采集与闭环 Inference CLI 入口
│   ├── sugar_rl/                          # 原版数据处理脚本 (process_refiner_rollout.py / process_tracker_rollout.py)
│   └── sugar_il/                          # Stage 3 扩散生成器训练脚本（内置 trainreg 强校验）
└── source/
    ├── sugar_rl/
    │   └── sugar_rl/
    │       ├── supermimic/                # 【核心算法包（非官方融合实现）】
    │       │   ├── rotations.py           # 可微 wxyz 四元数、零角安全梯度、SUGAR 行优先 6D 旋转
    │       │   ├── urdf_kinematics.py     # 无 MuJoCo 依赖的 G1 29-DoF 可微正向运动学引擎（含刚体局部 COM 速度递推）
    │       │   ├── hoi_task.py            # 84 维人-物联合状态、18 项 890/510 维观测对齐、6 任务奖励（含 feet_air_time）
    │       │   ├── networks.py            # JointWorldModel、自包含 SuperMimicPolicy、Ensemble Critic（支持内部核编译）
    │       │   ├── replay.py              # GPU 驻留环形缓冲区（独立 valid_transitions 掩码与真实 ep_id 5 帧历史回溯）
    │       │   ├── env_bridge.py          # IsaacLab 桥接器（预缓存参考张量、on_contact 精准拦截、启动切片对齐断言）
    │       │   └── trainer.py             # SuperMimic WM + 32-step BPTT + Teacher BC 训练主循环
    │       └── tasks/                     # SUGAR 官方人-物交互任务环境定义
    └── sugar_il/                          # SUGAR 官方扩散策略生成器源码包
```

---

## 3. 操作命令详解（严格区分 Windows 本地 PowerShell 与 VS Code 远程 Bash）

> **环境与合规准则**：
> - **Windows 本地 PowerShell**：仅用于本地无仿真器 PyTorch CUDA 单元测试与代码静态检查。
> - **VS Code Remote-SSH 远程 Bash（`xiaoji@172.17.27.172`）**：默认工作目录为 `/home/xiaoji/YXM/gowork`（本项目位于 `/home/xiaoji/YXM/gowork/fusion`）。每次执行命令前先显式 `cd /home/xiaoji/YXM/gowork/fusion`；既有 Conda / Isaac Sim / Isaac Lab 环境只读使用、不得升级或覆盖；训练前先检查 GPU 空闲与磁盘余量，并**必须通过 `trainreg run --` 交互认证启动训练**（严禁手写 `dengji.txt` 或绕过登记）。

### 3.1 本地快速算法回归测试（Windows 本地 PowerShell，无需 Isaac Sim）

```powershell
& "D:\anaconda\envs\DRL_env\python.exe" "D:\robot\rework\fusion\tests\test_supermimic_fusion.py"
```
- **作用**：在不启动 Isaac Sim 的前提下，直接在本地 GPU 上验证旋转库零角梯度、URDF/MJCF 运动学与刚体局部 COM 线速度递推、per-env 关节零位随机偏置 `xi` 补偿、890 维特权观测 18 个切片对齐断言、6 大任务分阶段奖励（含 `PushBox` Refiner/Tracker 差异与 `feet_air_time`）、世界模型 SE(2) 不变性、环形缓冲区 `done/push` 历史回溯隔离、超时（Timeout）当前步价值 Bootstrap、断点续训 `it+1` 以及 Stage 1 → Stage 2 → Inference 完整闭环。
- **参数含义**：直接运行测试脚本，无需额外命令行参数（自动优先使用 `cuda`）。
- **预期结果**：依次打印 `[PASS] Test 1` 至 `[PASS] Test 9`，末尾输出 `ALL 9 SUPERMIMIC x SUGAR TESTS PASSED SUCCESSFULLY!`。

---

### 3.2 远程服务器训练前资源检查、包安装与实时额度查询（VS Code 远程 Bash）

#### （0）进入远程工作目录并检查 GPU / 磁盘余量与实时训练额度

```bash
cd /home/xiaoji/YXM/gowork/fusion
nvidia-smi
df -h /home/xiaoji/YXM/gowork
trainreg balance --name-id yexiaoming --json
```
- **作用**：显式进入远程默认工作目录下的 `fusion` 项目，检查当前空闲 GPU 编号与磁盘剩余空间（不占用他人繁忙 GPU），并实时查询 `yexiaoming` 账户的剩余训练点数。
- **参数含义**：
  - `df -h /home/xiaoji/YXM/gowork`：以人类可读单位查看工作区挂载盘可用容量；
  - `--name-id yexiaoming`：指定额度账户标识为 `yexiaoming`；
  - `--json`：以结构化 JSON 输出当前实时余额与活跃会话状态。
- **预期结果**：`nvidia-smi` 显示目标 GPU 显存空闲且无他人训练进程；`trainreg balance` 返回包含实时剩余点数的 JSON（严禁以历史余额代替本次实时查询结果）。

#### （1）在目标环境中注册本项目源码包（仅在新建专属环境或首次绑定时执行，不改动公共包依赖）

```bash
cd /home/xiaoji/YXM/gowork/fusion
python -m pip install --no-deps -e source/sugar_rl
python -m pip install --no-deps -e source/sugar_il
```
- **作用**：以可编辑且**不修改任何既有第三方包版本**（`--no-deps`）的安全方式，将本工作目录内的 `sugar_rl`（含 `sugar_rl.supermimic`）与 `sugar_il` 挂载到当前 Python 路径。
- **参数含义**：
  - `--no-deps`：禁止升级、卸载或覆盖既有 Conda 环境中的 PyTorch / IsaacLab 等公共依赖包；
  - `-e source/sugar_rl` / `-e source/sugar_il`：就地链接源码目录，修改代码实时生效。
- **预期结果**：输出 `Successfully installed sugar_rl-...` 与 `Successfully installed sugar_il-...`，既有环境包版本零变动。

---

### 3.3 分阶段训练与数据生成命令（VS Code 远程 Bash）

以下以代表性双手搬箱任务 `CarryBox` 为例（如需指定空闲卡，请在命令前加 `CUDA_VISIBLE_DEVICES=<空闲GPU号>`；在无 `trainreg` 的个人 Google Colab 环境中可省略前缀 `trainreg run --`）：

#### （1）Stage 1：SuperMimic Refiner 训练

```bash
cd /home/xiaoji/YXM/gowork/fusion
trainreg run -- python -u scripts/supermimic/train.py \
    --task Sugar-G129dof-CarryBox-Refiner \
    --num_envs 2048 \
    --log_dir outputs/CarryBox_sm/logs/refiner \
    --max_iterations 5001 \
    --motion_folder data/CarryBox \
    --headless
```
- **作用**：通过 `trainreg` 交互认证后，使用 SuperMimic（84 维人-物联合世界模型 + 32 步 BPTT）训练 890 维特权观测的动作精炼策略 `Refiner`。
- **参数含义**：
  - `trainreg run --`：远程服务器强制训练登记包装器（启动后在终端隐藏输入个人密钥并按 `y` 确认扣费登记）；
  - `--task`：Gymnasium 任务 ID，`*-Refiner` 自动启用 890 维特权未来窗口观测；
  - `--num_envs`：并行 IsaacLab 环境数量（冒烟测试可设 `64`，正式训练建议 `2048`）；
  - `--log_dir`：检查点 `model_*.pt` 与 TensorBoard 日志保存目录；
  - `--max_iterations`：训练总迭代轮数（正式训练 `5001`，冒烟验证可设 `20`）；
  - `--motion_folder`：原始人-物参考动作目录；
  - `--headless`：无头模式运行物理仿真。
- **预期结果**：终端先提示隐藏输入训练密钥并确认启动；随后 IsaacLab 初始化并打印 `[SuperMimic Bridge] Privileged obs alignment verified on ... clean envs: critic max_diff=... (tol=0.005)`；训练完成后生成 `outputs/CarryBox_sm/logs/refiner/model_5000.pt`（可复制为 `outputs/CarryBox_sm/ckpts/refiner.pt`）。

#### （2）Stage 2：SuperMimic Refiner Rollout 与物理过滤

```bash
cd /home/xiaoji/YXM/gowork/fusion
cp outputs/CarryBox_sm/logs/refiner/model_5000.pt outputs/CarryBox_sm/ckpts/refiner.pt

python -u scripts/supermimic/play.py \
    --task Sugar-G129dof-CarryBox-Refiner-Rollout \
    --num_envs 1000 \
    --checkpoint outputs/CarryBox_sm/ckpts/refiner.pt \
    --rollout_dir outputs/CarryBox_sm/rollout_datasets/refiner/raw_npz \
    --motion_folder data/CarryBox \
    --headless

python -u scripts/sugar_rl/process_refiner_rollout.py \
    --data_dir outputs/CarryBox_sm/rollout_datasets/refiner \
    --task_name CarryBox
```
- **作用**：用 Stage 1 训练出的 `refiner.pt` 在物理引擎中回放参考轨迹，筛选出物理可行的高质量人-物交互轨迹生成 `rl_dataset`。
- **参数含义**：
  - `--checkpoint`：Stage 1 导出的 `refiner.pt` 路径；
  - `--rollout_dir`：原始回放轨迹 `.npz` 输出目录；
  - `--data_dir` / `--task_name`：按任务成功判据过滤 `raw_npz` 并导出 `rl_dataset`。
- **预期结果**：生成 `outputs/CarryBox_sm/rollout_datasets/refiner/rl_dataset/` 目录及过滤后的 `.npz` 轨迹文件。

#### （3）Stage 3：SuperMimic Tracker 训练（32 步 WM-BPTT + 冻结 Teacher BC 蒸馏）

```bash
cd /home/xiaoji/YXM/gowork/fusion
trainreg run -- python -u scripts/supermimic/train.py \
    --task Sugar-G129dof-CarryBox-Tracker \
    --num_envs 2048 \
    --teacher_ckpt outputs/CarryBox_sm/ckpts/refiner.pt \
    --motion_folder outputs/CarryBox_sm/rollout_datasets/refiner/rl_dataset \
    --teacher_motion_folder data/CarryBox \
    --log_dir outputs/CarryBox_sm/logs/tracker \
    --max_iterations 5001 \
    --headless
```
- **作用**：训练仅依赖 510 维观测（36 维单帧指令 + 5 帧本体感知历史）的全身跟踪策略 `SuperMimic Tracker`，在 32 步想象展开中融合任务奖励 BPTT 梯度与冻结 `refiner.pt` 的动态退火 BC 蒸馏（默认按 `0.7 * max_iterations = 3500` 轮平滑退火至 `0.05`）。
- **参数含义**：
  - `--task`：`*-Tracker` 自动切换为 510 维可部署观测模式；
  - `--teacher_ckpt`：冻结的 Stage 1 SuperMimic Refiner 检查点；
  - `--motion_folder`：Stage 2 生成的物理精炼轨迹目录 `rl_dataset`；
  - `--teacher_motion_folder`：原始参考动作目录（仅预缓存 `pelvis` 锚点供 Teacher 观测对齐，显存占用降低 14 倍）。
- **预期结果**：启动时同时通过 `critic`、`teacher` 与无噪 `actions_history` 三重观测对齐断言；训练结束后生成 `outputs/CarryBox_sm/logs/tracker/model_5000.pt`（可复制为 `outputs/CarryBox_sm/ckpts/tracker.pt`）。

#### （4）Stage 4：SuperMimic Tracker Rollout 与模仿学习 Zarr 数据集生成

```bash
cd /home/xiaoji/YXM/gowork/fusion
cp outputs/CarryBox_sm/logs/tracker/model_5000.pt outputs/CarryBox_sm/ckpts/tracker.pt

python -u scripts/supermimic/play.py \
    --task Sugar-G129dof-CarryBox-Tracker-Rollout \
    --checkpoint outputs/CarryBox_sm/ckpts/tracker.pt \
    --num_envs 1000 \
    --rollout_dir outputs/CarryBox_sm/rollout_datasets/tracker/raw_npz \
    --motion_folder outputs/CarryBox_sm/rollout_datasets/refiner/rl_dataset \
    --teacher_motion_folder data/CarryBox \
    --headless

python -u scripts/sugar_rl/process_tracker_rollout.py \
    --data_dir outputs/CarryBox_sm/rollout_datasets/tracker
```
- **作用**：采集 SuperMimic Tracker 的闭环跟踪轨迹，并打包为扩散策略训练所需的 Zarr 数据集 `il_dataset`。
- **参数含义**：
  - `--checkpoint`：Stage 3 导出的 `tracker.pt`；
  - `--data_dir`：包含 `raw_npz` 的父目录，处理后在其下生成 `il_dataset`。
- **预期结果**：生成 `outputs/CarryBox_sm/rollout_datasets/tracker/il_dataset` Zarr 数据集目录。

#### （5）Stage 5：Diffusion Command Generator 训练（`sugar_il`）

```bash
cd /home/xiaoji/YXM/gowork/fusion
trainreg run -- python -u scripts/sugar_il/train.py \
    --config-name train_generator_workspace.yaml \
    task=CarryBox \
    use_target=True \
    num_epochs=1001 \
    log_path=outputs/CarryBox_sm/logs/generator \
    dataset_path=outputs/CarryBox_sm/rollout_datasets/tracker/il_dataset
```
- **作用**：在 SuperMimic Tracker 生成的 `il_dataset` 上训练高层扩散指令生成器 `Generator`。
- **参数含义**：
  - `task`：任务名称；
  - `use_target`：任务是否包含目标点位姿输入（`CarryBox`、`PickBox`、`PushBox` 为 `True`，其余为 `False`）；
  - `num_epochs`：扩散网络训练轮数（默认 `1001`）；
  - `dataset_path`：Stage 4 导出的 `il_dataset` 目录。
- **预期结果**：生成 `outputs/CarryBox_sm/logs/generator/epoch_checkpoints/epoch=1000.ckpt`（随后复制为 `outputs/CarryBox_sm/ckpts/generator.ckpt`）。

#### （6）Stage 6：闭环推理与 MP4 视频录制

```bash
cd /home/xiaoji/YXM/gowork/fusion
cp outputs/CarryBox_sm/logs/generator/epoch_checkpoints/epoch=1000.ckpt outputs/CarryBox_sm/ckpts/generator.ckpt

python -u scripts/supermimic/play.py \
    --task Sugar-G129dof-CarryBox-Inference \
    --checkpoint outputs/CarryBox_sm/ckpts/tracker.pt \
    --generator_checkpoint outputs/CarryBox_sm/ckpts/generator.ckpt \
    --motion_folder data/CarryBox \
    --num_envs 4 \
    --eval_random_motion \
    --headless \
    --video \
    --video_length 200
```
- **作用**：联合加载高层 `Generator` 与底层 `SuperMimic Tracker` 进行 50 Hz 全自主闭环交互并录制 `.mp4` 评估视频（或直接执行 `bash inference_fusion.sh CarryBox sm 4 200`）。
- **参数含义**：
  - `--checkpoint`：SuperMimic Tracker 权重路径；
  - `--generator_checkpoint`：扩散策略 Generator 权重路径；
  - `--eval_random_motion`：随机抽样测试动作初始场景；
  - `--video` / `--video_length 200`：开启离屏相机录制 200 步（4 秒，50 FPS）视频。
- **预期结果**：在 `outputs/CarryBox_sm/ckpts/videos/play/` 目录下生成 `.mp4` 闭环仿真交互视频。

---

## 4. 一键全流程脚本与交互式 Notebook

- **VS Code 远程 Bash 一键运行 5 阶段训练**：
  ```bash
  cd /home/xiaoji/YXM/gowork/fusion
  trainreg balance --name-id yexiaoming --json
  trainreg run -- bash train_fusion.sh CarryBox sm_run1 5001 2048
  ```
  - **作用**：在一次 `trainreg` 交互登记会话内自动串联执行 Stage 1 → Stage 5 全流程（含 Isaac Sim 退出码 `139` 产物校验容错）。
- **交互式复现手册 `SuperMimic_SUGAR_Repro.ipynb`**：
  - 在 **Google Colab GPU** 上可直接从 Step 0 至 Step 7 逐格运行；
  - 在安装了 `trainreg` 的**受控远程服务器**上，Notebook 会自动检测 `trainreg` 并打印带实时参数（`TASK_NAME` / `EXP_NAME` / `RL_ITERS`）的标准 `trainreg balance` 与 `trainreg run --` 交互终端命令，确保用户可在 VS Code 集成终端安全隐藏输入个人密钥，杜绝无 TTY 管道裸跑绕过。

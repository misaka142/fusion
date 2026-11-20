# SuperMimic × SUGAR 融合工程 (`fusion`)

> **[重要源码来源与未开源缺失声明]**
> 1. **SuperMimic 官方源码缺失声明**：**SuperMimic 原论文官方代码尚未开源（官方源码缺失）**。本工程中的 `sugar_rl.supermimic` 包（依据论文方法结合参考目录设计重构）属于**非官方融合实现，并非原论文作者发布的官方源码**。
> 2. **SUGAR 官方源码基础**：本工程的全身人-物交互仿真任务（`sugar_rl/tasks`）、URDF/MJCF 资产与高层扩散策略生成器（`sugar_il`）来自已核实的 **SUGAR** 官方开源代码。
> 3. **验证边界声明**：本融合工程已通过本地 PyTorch CUDA 15 项核心单元测试与 6 项 MuJoCo Sim-to-Sim 端到端测试；完整 Isaac Sim 5.1.0 长周期强化学习训练需按工作流在 **Google Colab GPU** 中完成验证，未在云端实际跑通收敛前不宣称已完成长周期任务收敛。

本项目将 **SuperMimic**（基于可微学习世界模型的 32 步想象展开策略梯度 BPTT + 双头悲观 Critic 引导）移植至 **SUGAR**（*Synthesize Whole-Body Loco-Manipulation via Decoupled Tracking and Generation*）全身人-物交互框架中，**同时替换原版 SUGAR 中的 Stage 1 `Refiner`（原 RSL-RL PPO）与 Stage 2 `Tracker`（原 RSL-RL BCPPO）**，并与原版 Rollout 数据转换管线及 Stage 3 扩散策略 `Generator`（`sugar_il`）保持 **100% 接口与数据格式兼容**。

---

## 1. 标准研发与训练工作流

本项目采用 **“本地 Windows 编写与调试 → 推送 GitHub → Google Colab 云端训练（Google Drive 持久化）→ 下载权重回本地 MuJoCo 验证”** 标准工作流：

```text
[本地 Windows PowerShell]
   编写/调试代码、生成平滑参考轨迹 data/OmniCarryBox、运行 15+6 项本地单元测试与 Sim2Sim 校验
   │
   ▼  git push origin v2-speed-opt
[GitHub 远程仓库]
   │
   ▼  在 Google Colab 打开 SuperMimic_SUGAR_Repro.ipynb
[Google Colab GPU (A100 / L4)]
   • Step 0~5：环境构建、数据加载/解压、GPU 单测、20 轮 Isaac Sim 实机冒烟校验
   • Stage 6.1：SuperMimic Refiner 训练 (890-D 特权观测 + 123-D SE(2) 世界模型 + 32 步 BPTT)
   • Stage 6.2：Refiner Rollout 与物理过滤 -> rl_dataset
   • Stage 6.3：SuperMimic Tracker 训练 (510-D 观测 + 32 步 BPTT + 冻结 Refiner 退火 BC 蒸馏)
   • Stage 6.4：Tracker Rollout 与 Zarr 打包 -> il_dataset
   • Stage 6.5：Diffusion Generator 训练 (sugar_il 10Hz 扩散规划器)
   • 权重、日志、视频自动同步至 Google Drive (`MyDrive/SuperMimic_SUGAR_Backup`)
   │
   ▼  下载 ckpts/*.pt / *.ckpt 回本地
[本地 Windows PowerShell]
   运行 scripts/sim2sim/run_sim2sim.py 在独立 MuJoCo 引擎上做 50Hz+500Hz 闭环验证
```

---

## 2. 核心算法增强与目录结构（v2）

### 2.1 v2 关键融合机制
- **123 维 `SE(2)` 防碰撞世界模型 + `delta_rms` 输出归一化**：`state_features` 在 108 维基础上拼入 `torso_link`、左右 `elbow_link` 与左右 `wrist_yaw_link` 在物体局部系下的 15 维坐标（共 123 维，保持 `SE(2)` 不变性）；`JointWorldModel` 默认 `(1024, 1024, 1024)` 隐层并内置 `delta_rms` 按维度反归一化、`vel_loss_scale=0.5` 及 MLP `bfloat16` 混合精度。
- **上下肢残差掩码解耦 + 确定性 BPTT 去抖**：`SuperMimicPolicy` 的 `residual_mask` 将双腿 12 关节设为 `0.0`（相对 `default_joint_pos` 自由动态平衡）、腰+臂 17 关节设为 `1.0`（保留 `q_ref` 前馈残差）；32 步 BPTT 内部采用确定性动作展开 + 纯网络残差平滑正则 + `clip_grad_value_(0.10)`，真实采样端使用上下肢分组退火探索噪声与余弦学习率衰减。
- **前沿边界采样与 DiPLO 起升平滑降速**：`HOIReplayBuffer` 支持 `frontier_frac=0.40` 前沿优先采样（突破抱箱前死亡进度壁垒）；`convert_diplo_to_sugar.py` 通过 `stretch_pickup_window` 对抱箱起升段做 `1.65x` `C1` 平滑时间拉伸并通过 MuJoCo 双足零滑移/防碰撞门禁，降低软 PD 刚度下的直立过冲惯性力。
- **物理随机化严格保持原版**：`BaseEventCfg` 与 `carry_box_*_env_cfg.py` 中的摩擦系数、物体质量与 `push_robot` / `push_object` 扰动严格沿用原版 SUGAR 配置。
- **整步零断点编译与矩阵化算子加速（`v2-speed-opt`）**：将 32 步想象推演单步 `_img_step_core` 与世界模型多步损失 `_wm_window_loss_core` 抽离为 0 Graph-Break 纯张量路径（支持 `--compile --compile_mode default` 整步 Triton 融合）；旋转与 `FastTreeKinematics` 改用广播乘加直接输出并消费旋转矩阵、预缓存 `MotionReferenceAdapter` 静态 `o2b` 与重投影重力 Z、启用 `AdamW(fused=True)` 并缓存 `HOIReplayBuffer` 窗口索引，**权重维度与 `v1-lafan-omni` 的 `sm_v2_diplo_lafan` 完全互通**。

### 2.2 核心模块目录结构

```text
fusion/
├── SuperMimic_SUGAR_Repro.ipynb           # Google Colab 端到端复现与断点续训 Notebook
├── COLAB_LOG_AND_VIDEO_GUIDE.md           # Colab TensorBoard 日志导出与 MP4 渲染手册
├── REFINER_DIAGNOSIS_AND_FIX_PLAN.md      # Refiner 训练诊断与 v2 根治方案说明
├── FUSION_LEARNING_GUIDE.md               # 零基础源码级精讲与面试指南
├── train_fusion.sh                        # Linux/Colab Bash 5 阶段串联训练脚本
├── inference_fusion.sh                    # Stage 6 闭环推理录屏脚本
├── tests/
│   ├── test_supermimic_fusion.py          # 15 项纯 PyTorch GPU/CPU 核心与回归测试套件
│   └── test_mujoco_sim2sim.py             # 6 项全任务 MuJoCo Sim-to-Sim 端到端测试套件
├── scripts/
│   ├── supermimic/
│   │   ├── train.py                       # SuperMimic Refiner / Tracker 统一训练 CLI
│   │   └── play.py                        # Rollout 采集与闭环 Inference CLI
│   ├── sugar_rl/                          # DiPLO+LAFAN1 清洗拼接、Rollout 过滤与打包脚本
│   ├── sugar_il/                          # Stage 5 扩散生成器训练脚本
│   └── sim2sim/                           # 零 IsaacLab 依赖的本地 MuJoCo Sim-to-Sim 部署器
└── source/
    ├── sugar_rl/sugar_rl/
    │   ├── supermimic/                    # 核心算法包（rotations / urdf_kinematics / hoi_task / networks / replay / env_bridge / trainer）
    │   └── tasks/                         # SUGAR 官方人-物交互任务环境定义
    └── sugar_il/                          # SUGAR 官方扩散策略生成器源码包
```

---

## 3. 操作命令详解（严格区分 Windows 本地 PowerShell 与 Google Colab）

> **环境保护准则**：使用本地既有的 `D:\anaconda\envs\DRL_env\python.exe` 运行测试与数据工具；未经确认不得在该环境中执行会覆盖核心包版本的 `pip install` 或降级操作。

### 3.1 本地单元测试与数据集准备（Windows 本地 PowerShell）

#### （1）运行 15 项 SuperMimic 核心算法回归测试（无需 Isaac Sim）

```powershell
& "D:\anaconda\envs\DRL_env\python.exe" "D:\robot\rework\fusion\tests\test_supermimic_fusion.py"
```
- **作用**：在本地 GPU/CPU 上验证四元数零角梯度、11 层深度向量化 URDF FK 与 `scale_grad` 等价性、890/510 维观测与奖励、123 维 `SE(2)` 不变性、`compute_state_delta`/`delta_rms` 往返一致、上下肢残差掩码、前沿边界采样、`OmniCarryBox` 凸包校验、起升平滑拉伸，以及矩阵化 FK/整步编译等价性与零 Dynamo Graph Break。
- **参数含义**：直接执行测试入口脚本（自动优先选择 `cuda`）。
- **预期结果**：依次打印 `[PASS] Test 1` 至 `[PASS] Test 15`，末尾输出 `ALL 15 SUPERMIMIC x SUGAR TESTS PASSED SUCCESSFULLY!`。

#### （2）运行 6 项 MuJoCo Sim-to-Sim 端到端测试

```powershell
& "D:\anaconda\envs\DRL_env\python.exe" "D:\robot\rework\fusion\tests\test_mujoco_sim2sim.py"
```
- **作用**：验证全 6 任务 MuJoCo 场景编译、双格式权重加载、510 维观测与 `SugarHOITask` 同构性、500Hz 隐式 PD 站立稳定、自主闭环录屏以及本地 `data/OmniCarryBox` 轨迹质心/防滑移/网格居中校验。
- **参数含义**：直接运行测试脚本，使用临时目录生成测试用场景与假权重。
- **预期结果**：末尾输出 `ALL 6 MUJOCO SIM-TO-SIM END-TO-END TESTS PASSED SUCCESSFULLY!`。

#### （3）生成平滑起升版 `data/OmniCarryBox`（仅重建数据集时执行）

```powershell
& "D:\anaconda\envs\DRL_env\python.exe" "D:\robot\rework\fusion\scripts\sugar_rl\convert_diplo_to_sugar.py"
```
- **作用**：从 `data_diplo_raw` 与 `data_lafan1_raw` 筛选纯 MuJoCo 模拟区间，拼接双足焊死弯腰与 `1.65x` 平滑起升拉伸、$D_4$ 对称扩增与 `LAFAN1` 停步接近轨迹，输出至 `data/OmniCarryBox`。
- **关键参数含义**：可通过 `--pickup_stretch 1.65` 调整抱箱起升拉伸倍数，`--blend_frames 22` 控制步态衔接窗长。
- **预期结果**：通过全部 MuJoCo 门禁并输出 `data/OmniCarryBox/data_000..`、`obj_aligned.usda`、`box_spec.json` 与 `manifest.json`（注意 `data/` 与 `outputs/` 已在 `.gitignore` 中排除，请通过压缩包同步至 Google Drive，不提交到 GitHub）。

---

### 3.2 Google Colab Notebook / 终端训练命令

在 Google Colab 中优先直接按顺序执行 `SuperMimic_SUGAR_Repro.ipynb`（Step 0 ~ Step 8）。如需在 Colab Notebook 单元格（首行加 `%%bash`）或左下角终端中手动分步执行核心训练：

#### （1）Stage 1：SuperMimic Refiner 训练（Colab `%%bash`）

```bash
%%bash
cd /content/fusion
/content/.venv_fusion/bin/python -u scripts/supermimic/train.py \
    --task Sugar-G129dof-CarryBox-Refiner \
    --num_envs 6144 \
    --wm_batch_size 3072 \
    --policy_batch_size 2048 \
    --log_dir outputs/CarryBox_sm_v2_diplo_lafan_full/logs/refiner \
    --max_iterations 5001 \
    --motion_folder data/OmniCarryBox \
    --compile \
    --compile_mode default \
    --auto_resume \
    --headless
```
- **作用**：使用 `123` 维 `SE(2)` 世界模型（`1024×3` + `delta_rms`）与确定性 32 步 BPTT 训练 890 维特权观测的 `Refiner` 策略。
- **关键参数含义**：
  - `--num_envs`：并行仿真环境数（A100-40GB 推荐 `6144`，A100-80GB 可设 `8192`）；
  - `--wm_batch_size` / `--policy_batch_size`：世界模型与 32 步想象展开的批大小（默认 `wm_updates=16, policy_updates=8`）；
  - `--motion_folder`：参考动作目录（`CarryBox` 默认使用 `data/OmniCarryBox`）；
  - `--auto_resume`：支持 Colab 断连后自动从 `model_latest.pt` 续训。
- **预期结果**：启动时通过 18 项特权观测对齐断言与 `obj_aligned.usda` 质量/传感器校验，训练结束后生成 `outputs/CarryBox_sm_v2_diplo_lafan_full/logs/refiner/model_5000.pt`。

#### （2）一键串联执行 Stage 1 ~ Stage 5 全流水线（Colab `%%bash`）

```bash
%%bash
cd /content/fusion
bash train_fusion.sh CarryBox sm_v2_diplo_lafan_full 5001 6144 2048 1001
```
- **作用**：按顺序串联执行 `Refiner` 训练 → `Refiner Rollout` 过滤 → `Tracker` 蒸馏训练 → `Tracker Rollout` 打包 → `Diffusion Generator` 训练（内置 Isaac Sim 退出码 `139` 容错校验）。
- **关键参数含义**：依次为 `TASK_NAME`、`EXP_NAME`、`RL_ITERS`、`NUM_ENVS`、`ROLLOUT_ENVS`、`IL_EPOCHS`。
- **预期结果**：在 `outputs/CarryBox_sm_v2_diplo_lafan_full/ckpts/` 下导出 `refiner.pt`、`tracker.pt` 与 `generator.ckpt`。

---

### 3.3 本地 MuJoCo Sim-to-Sim 闭环验证（Windows 本地 PowerShell）

将 Colab 训练好的 `tracker.pt` 与 `generator.ckpt` 从 Google Drive 下载到本地后，在 **Windows 本地 PowerShell** 运行独立 MuJoCo 验证：

```powershell
& "D:\anaconda\envs\DRL_env\python.exe" "D:\robot\rework\fusion\scripts\sim2sim\run_sim2sim.py" `
    --task CarryBox `
    --mode autonomous `
    --tracker_ckpt "D:\robot\rework\fusion\outputs\CarryBox_sm_v2_diplo_lafan_full\ckpts\tracker.pt" `
    --generator_ckpt "D:\robot\rework\fusion\outputs\CarryBox_sm_v2_diplo_lafan_full\ckpts\generator.ckpt" `
    --motion_folder "D:\robot\rework\fusion\data\OmniCarryBox"
```
- **作用**：脱离 Isaac Sim，在本地 500Hz MuJoCo 物理引擎上以 10Hz 扩散规划器 + 50Hz `Tracker` 进行全自主闭环交互验证（支持 `--headless --record_mp4 out.mp4` 导出视频）。
- **关键参数含义**：`--mode autonomous` 启用闭环扩散规划（若仅回放参考命令可改为 `--mode replay`）；`--motion_folder` 同步加载任务对应 `box_spec.json` 与居中网格。
- **预期结果**：打开实时 MuJoCo 3D 窗口展示机器人行走、弯腰抱起箱子并搬运至目标区域，无高频自激抖动或抱起仰翻摔倒。

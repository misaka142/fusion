# SuperMimic × SUGAR 第一阶段（Refiner）问题诊断与 v2 根治方案

> **状态**：**v2 代码与平滑起升数据集已在本地实现并完成本地单元测试（待推送 GitHub 并于 Google Colab A100 执行实机训练验证）**  
> **硬件与工作流**：本地 Windows 编写与单测 → 推送 GitHub → **Google Colab GPU（Google Drive 持久化）** 训练 → 下载权重回本地 MuJoCo Sim2Sim 验证

---

## 一、1400 轮实训日志与 `276 steps` 视频量化诊断结论

对 `CarryBox_sm_v1_lafan_omni_full`（日志 `tb_logs_CarryBox_sm_v1_lafan_omni_full_20261120_040633`，`8192` 环境、训练至 `1573` 轮）的 TensorBoard 全量曲线与 [CarryBox_refiner_model_1400_traj3_terminated_early_276steps.mp4](file:///E:/edge%20download/CarryBox_refiner_model_1400_traj3_terminated_early_276steps.mp4) 实测分析确认：

| 现象 | 核心根因（已定位到代码与轨迹物理参数） |
| :--- | :--- |
| **1. 能走路，但走路姿势高频抽风（零噪声推理仍抖）** | ① `JointWorldModel` 缺少按维度输出标准差归一化（`delta_rms`）且 `vel_loss_scale=0.1` 过小，成了对角速度跳变不敏感的低通滤波器，BPTT 趁机学出高频正负颤振刷分；<br>② 32 步想象展开每步注入 `deterministic=False` 白噪声并设 `entropy_coef=1e-3` 放大增益；<br>③ 双腿 12 关节强绑非动力学平衡的 `q_ref` 残差偏置，与手腕小 `action_scale (0.0745 rad)` 共同在 `action_rate_l2` 下产生对抗震荡；<br>④ `feet_air_time` 阈值（`0.5s`）高于正常步频摆动周期且无梯度截断，诱发低抬脚碎步抖动。 |
| **2. 能俯身抱箱子，但刚抱起（~255–276 步）就失重向后仰翻摔倒** | ① **DiPLO 起升动力学与 SUGAR 软 PD 刚度错配**：DiPLO 抓箱时躯干倾角中位数 `85.1°`，并在短短 `27 帧 (0.54s)` 内猛提至直立（箱子上升峰值速度 `1.15~1.66 m/s`，加速度 `14.7 m/s²`）；而在 SUGAR 超软 PD（腰/踝 `kp=28.5`, 肩/肘 `kp=14.25`）下，策略被迫输出巨大拉腰残差（`error_joint_pos` 在 1400 轮暴增至 `2.54 rad`），站直瞬间重力臂消失引发大力矩过冲，将重心甩出仅 `-5 cm` 的后脚跟支撑极限；<br>② **世界模型防碰撞特征盲区**：`state_features (108-D)` 漏掉了 `torso_link` 与前臂相对箱子的位置，BPTT 诱导把箱子往胸口拽，回真实环境撞胸后顶翻重心（`obj_ori` 终止占比升至 `34.5%`）；<br>③ **第 276 步经验池断墙**：全部环境都在抱起段终止，导致 Replay Buffer 与 `init_pool` 对持箱行走段采样几乎为 `0`。 |

---

## 二、已落地的 v2 根治改动清单（物理摩擦与外力推力严格沿用原版）

> **重要设计边界**：`BaseEventCfg` 与 `carry_box_*_env_cfg.py` 中的所有物理摩擦范围、物体质量随机化与 `push_robot` / `push_object` 均**保持 100% 原版 SUGAR 不动**。

1. **数据集起升段平滑时间拉伸（[convert_diplo_to_sugar.py](file:///D:/robot/rework/fusion/scripts/sugar_rl/convert_diplo_to_sugar.py)）**：
   - 新增 `stretch_pickup_window(..., stretch_factor=1.65)`，保持行走与持箱步态不变，仅对抓箱到直立站稳窗口施加 `C1` 平滑时间拉伸，将抬升耗时放缓至与原版 `CarryBox (0.94s)` 同量级、峰值惯性力下降约 `2.7~2.9x`，并重新通过全套 MuJoCo 双足零滑移与防碰撞门禁，生成 `diplo_sbto_lafan1_clean_v2` 数据集。
2. **防碰撞特征补齐与 `delta_rms` 速度还原（[hoi_task.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/hoi_task.py) & [networks.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/networks.py)）**：
   - `state_features` 从 `108` 维扩至 `123` 维（利用已算好的 `FKResult` 补入 `torso_link`、左右 `elbow_link`、左右 `wrist_yaw_link` 在箱子局部系的 15 维坐标，零额外 FK 开销且严格保持 `SE(2)` 不变性）；
   - 新增 `SugarHOITask.compute_state_delta`，配合 `JointWorldModel.delta_rms` 对 82 维增量做按维度标准差归一化，并将 `vel_loss_scale` 由 `0.1` 恢复到 `0.5`、`wm_contact_loss_weight` 升至 `0.5`，消除速度低通盲区；
   - 世界模型扩至 `(1024, 1024, 1024)` 并在 MLP 内部启用 CUDA `bfloat16` AMP。
3. **上下肢残差解耦 + 确定性 BPTT 去抖（[networks.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/networks.py) & [trainer.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/trainer.py)）**：
   - `SuperMimicPolicy` 引入持久缓冲区 `residual_mask`：下肢 12 关节设为 `0.0`（自由维持前后平衡），腰与双臂 17 关节设为 `1.0`（保留 `q_ref` 引导）；同时保持对无 `residual_mask` 旧检查点的 `strict=True` 加载兼容；
   - 32 步 BPTT 改为完全确定性展开（关闭想象白噪声与 `entropy_coef`），加入纯网络残差帧间平滑惩罚（`residual_rate_weight=0.02`）、`clip_grad_value_(0.10)` 与余弦学习率衰减（`lr_min_scale=0.15`）；真实环境采样端采用下肢 `0.12` / 上肢 `0.30` 分组线性退火噪声（终值 `0.25x`）。
4. **步态梯度解锁与想象引导（[hoi_task.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/hoi_task.py), [base_refiner_env_cfg.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/base_refiner_env_cfg.py), [base_tracker_env_cfg.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_tracker/base_tracker_env_cfg.py)）**：
   - `feet_air_time` 调为 `threshold=0.25, weight=1.5`，`feet_slide` 增强为 `-0.5`；`prev_feet_gnd` 执行 `.detach()` 切断贴地免罚梯度作弊；在想象 `update_policy` 阶段启用 `imagination_shaping=True` 强化脚踝抬高与双腕抓取局部位置引导。
5. **破除第 276 步断墙与提升吞吐（[replay.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/replay.py), [commands.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py), [train.py](file:///D:/robot/rework/fusion/scripts/supermimic/train.py)）**：
   - `HOIReplayBuffer.sample_imagination_starts` 新增 `frontier_frac=0.40` 前沿优先采样；`pool_warmup_steps` 缩至 `200*24`、`update_interval` 缩至 `24*20`；默认超参升级为 `wm_updates=16, wm_batch_size=2048, policy_updates=8, policy_batch_size=2048`。
6. **纯等价算子融合与整步零断点加速（`v2-speed-opt` 分支，[rotations.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/rotations.py), [urdf_kinematics.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/urdf_kinematics.py), [env_bridge.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/env_bridge.py), [trainer.py](file:///D:/robot/rework/fusion/source/sugar_rl/sugar_rl/supermimic/trainer.py)）**：
   - 保持所有模型尺寸与 `sm_v2_diplo_lafan` 权重格式 100% 不变：在 `hoi_task` 与 `FastTreeKinematics` 内部用 `mat_mul`/`mat_t_vec` 广播乘加与 `rotmat_error_magnitude` 直接基于 3×3 旋转矩阵计算观测与奖励（`with_quat=False` 跳过 13/14 刚体 `rotmat_to_quat`）、在 `MotionReferenceAdapter` 中预缓存 `o2b_pos_ref`/`o2b_rotmat_ref`/`anc_pg_z_ref` 并支持整窗一次性 `get_imagination_slice`、缓存 `HOIReplayBuffer` 窗口索引、`AdamW` 在 CUDA 上启用 `fused=True`；同时在 `SuperMimicTrainer` 抽离 0 Graph-Break 的 `_img_step_core` 与 `_wm_window_loss_core`，支持在 Colab Linux 下通过 `--compile --compile_mode default` 开启全步 Triton 算子融合。

---

## 三、Colab 重建与上线注意要点

1. **必须使用新数据集缓存与新实验名**：由于 `JointWorldModel` 输入特征已从 `108` 维扩至 `123` 维（且隐层扩至 `1024`、`SuperMimicPolicy` 新增 `residual_mask`），且 `data/OmniCarryBox` 已升级为起升平滑降速版 `diplo_sbto_lafan1_clean_v2`，请先将本地重新生成的 `outputs/OmniCarryBox_DiPLO_Clean_v2.zip` 上传至 Google Drive `MyDrive/SuperMimic_SUGAR_Backup/wheels_and_zips/`（或在 Colab 配置 `HF_TOKEN` 由 Step 3 自动重生成），并在 `SuperMimic_SUGAR_Repro.ipynb` 中使用全新隔离实验名 `sm_v2_diplo_lafan_full` 从第 0 轮起训。

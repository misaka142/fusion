# SuperMimic × SUGAR 第一阶段（Refiner）问题诊断与修改方案

> **状态**：待审核设计方案（尚未修改源码，待审核确认后执行代码修改、本地单元测试回归与 Colab 实机验证）  
> **目标硬件与工作流**：本地 Windows 编写与单测 → GitHub 同步 → **Google Colab A100 (`NUM_ENVS = 4096`)** 训练（Google Drive 实时持久化）

---

## 目录

1. [现象汇总与三大核心疑问定性结论](#1-现象汇总与三大核心疑问定性结论)
2. [第一性原理根因剖析](#2-第一性原理根因剖析)
   - [2.1 问题一：为什么走路姿势别扭、高频抖动、脚底板搓地？](#21-问题一为什么走路姿势别扭高频抖动脚底板搓地)
   - [2.2 问题二：为什么到 3500 轮仍抱不起箱子、箱子倒地、碰身体失败？](#22-问题二为什么到-3500-轮仍抱不起箱子箱子倒地碰身体失败)
   - [2.3 问题三：为什么 SuperMimic 替换 PPO 后没有加速反而变慢？](#23-问题三为什么-supermimic-替换-ppo-后没有加速反而变慢)
3. [分模块修改方案（遵循 KISS 原则）](#3-分模块修改方案遵循-kiss-原则)
   - [方案模块 M1：动作空间上下肢解耦 + 纯残差平滑约束（解决姿势别扭）](#方案模块-m1动作空间上下肢解耦--纯残差平滑约束解决姿势别扭)
   - [方案模块 M2：世界模型输出归一化 + 确定性 BPTT + 余弦退火（解决高频抖动）](#方案模块-m2世界模型输出归一化--确定性-bptt--余弦退火解决高频抖动)
   - [方案模块 M3：步态接触梯度解锁 + 脚踝摆动高度引导（解决脚底搓地）](#方案模块-m3步态接触梯度解锁--脚踝摆动高度引导解决脚底搓地)
   - [方案模块 M4：防碰撞几何感知 + 双掌夹紧引导 + 课程池加速（解决抱箱失败与碰身体）](#方案模块-m4防碰撞几何感知--双掌夹紧引导--课程池加速解决抱箱失败与碰身体)
   - [方案模块 M5：消除冗余 FK + `bf16` 混合精度 + 对齐 A100 高重用率 UTD（解决训练变慢）](#方案模块-m5消除冗余-fk--bf16-混合精度--对齐-a100-高重用率-utd解决训练变慢)
4. [涉及修改的文件清单与显存预算评估（Colab A100, 4096 Envs）](#4-涉及修改的文件清单与显存预算评估colab-a100-4096-envs)
5. [审核后验证与上线步骤](#5-审核后验证与上线步骤)

---

## 1. 现象汇总与三大核心疑问定性结论

在运行 `Sugar-G129dof-CarryBox-Refiner` 训练至约 `3500` 轮时，观察到两类核心异常：
- **异常 A（步态质量差）**：机器人走路姿态僵硬别扭、全身存在高频抖动、双脚贴着地面搓地滑行（不肯正常抬脚迈步）。
- **异常 B（交互操作失败）**：无法稳定抱起箱子，抬起瞬间箱子滑脱倒地，且试图抱箱子时频繁用躯干/前臂贴靠箱子（接触身体），导致奖励暴跌或箱子被身体顶翻触发终止失败。

针对三个核心疑问的直接核查结论：

| 你的疑问 | 核查定性结论 | 核心原因一句话总结 |
| :--- | :--- | :--- |
| **1. 姿势难看、抖动、搓地是否因为加速度？** | **仅占次要诱因** | 关节加速度权重（`-2.5e-7`）偏小且世界模型低估了速度跳变确实放纵了抖动；但**主因**是：**① 世界模型无输出归一化 + 32 步 BPTT 每步注入白噪声引发高频颤振**、**② 双腿强绑运动学参考残差与动态平衡对抗**、**③ `feet_air_time` 在一阶梯度下存在“不抬脚即零惩罚”的死锁陷阱**。 |
| **2. 3500 轮抱不起箱子、碰身体失败是否因为训练太少？** | **训练少占一半，模型梯度盲区占另一半** | 原版 SUGAR PPO 需 `4096` 环境训 `30,001` 轮，且 `init_pool` 前 1000 轮锁死、每 100 轮才慢速推进一次；**但更致命的是**：光滑箱子力闭合夹持存在接触梯度死区，且**世界模型特征完全漏掉了胸口躯干（`torso_link`）与前臂（`elbow_link`）相对箱子的坐标**，导致 BPTT 盲目把箱子往怀里搂，回真实物理即撞翻失败。 |
| **3. SuperMimic 没加速反而变慢，是错觉还是本该如此？** | **绝非错觉，当前代码确实变慢了** | 原版 SuperMimic 靠极高重用率（每 24 环境步消费 `125.8 万`想象步 + `2048×3` 大模型 + `bf16`/CUDA Graph）实现加速；当前 `fusion` 版将每轮想象步砍到了 `6.55 万`步（**缩水 19 倍，连当轮新采 9.8 万真实步的 5% 都没用上**），却在未编译 `fp32` 下每轮串行跑 **64 次** Python 循环版 34 连杆正向运动学，导致样本效率与单轮帧率双双下降。 |

---

## 2. 第一性原理根因剖析

### 2.1 问题一：为什么走路姿势别扭、高频抖动、脚底板搓地？

#### 根因 1：【高频抖动】世界模型缺失输出归一化 + 速度误差降权 + 32 步随机白噪声 BPTT（模型漏洞利用）
1. **世界模型变成了“低通滤波器”，吃掉了加速度惩罚**：
   - 在 `source/sugar_rl/sugar_rl/supermimic/networks.py`（第 108-160 行）中，`JointWorldModel` 直接线性输出 82 维跨数量级状态增量（单步位移 `dp_h` 约 `0.01 m`，而单步关节角速度跳变 `dqd` 达 `1.0 ~ 5.0 rad/s`，相差百倍），**缺失了原版 SuperMimic（见 `allmimic/networks.py` 第 121 行 `out_norm`）必备的按维度标准差归一化**。
   - 在 `source/sugar_rl/sugar_rl/supermimic/hoi_task.py`（第 447、468-482 行）中，`world_model_state_loss` 人为将所有速度误差乘以 `vel_loss_scale = 0.1` 后再送入 `beta = 0.05` 的 Huber Loss。
   - **后果**：世界模型对高频正负交替动作引起的剧烈角速度变化 $\Delta \dot{q}$ 严重欠拟合。当 BPTT 计算关节角加速度惩罚 `r_jacc = -2.5e-7 * (((qd_next - qd_prev) / 0.02)**2).sum()`（`hoi_task.py` 第 893-894 行）时，不仅 `-2.5e-7` 本身极弱，世界模型预测出的 `qd_next - qd_prev` 还远小于真实物理，一阶梯度趁机产生高频正负交替颤振（Chattering）去刷位置分。
2. **32 步想象展开内每步注入 50Hz 高斯白噪声**：
   - 原版 SuperMimic（`allmimic/train.py` 第 611-612 行）采用确定性策略网络（想象展开噪声为 `0`），配合余弦学习率衰减与按值梯度裁剪 `clip_grad_value_(0.1)`。
   - 当前 `trainer.py`（第 514、599-603 行）在 32 步想象展开的**每一步**都以 `deterministic=False` 注入 `std = 0.20` 的重参数化白噪声，并用 `entropy_coef = 1e-3` 阻止 `std` 下降、全程固定学习率 `3e-4` 不衰减，直接导致策略输出高频抖动。

#### 根因 2：【姿势别扭】双腿强绑运动学参考残差 `use_ref_residual=True` 与动态平衡及 `action_rate` 对抗
1. **双腿被非动力学平衡轨迹绑死**：
   - `networks.py`（第 228-232 行）对全部 29 个关节叠加了 `ref_bias = (q_ref - default_joint_pos) / action_scale`。
   - `data/CarryBox` 原始轨迹来自纯运动学重定向，并不满足动力学质心平衡（尤其负重前倾时）。双腿为了产生正确的地面反作用力维持不倒，必须调整落脚点，强加 `ref_bias` 会迫使网络时刻与参考轨迹拔河（对应 `allmimic/CHANGES.md` 第 `#20` 条实证结论：*"leg targets tied to the reference fight root balance"*）。
2. **`action_rate` 误伤小缩放关节**：
   - 手腕 pitch/yaw 的 `action_scale` 仅 `0.0745 rad`（`urdf_kinematics.py` 第 236-243 行）。`hoi_task.py`（第 887 行）对叠加 `ref_bias` 后的总输出 `raw_action` 计算 `-0.1 * ||raw_action - prev_raw_action||^2`，参考动作正常运动就会引发巨大的 `ref_bias` 变化率惩罚，迫使网络输出反向残差去抵消正常动作。

#### 根因 3：【脚底板搓地】`feet_air_time` 在一阶可微梯度下的“贴地免罚”死锁陷阱
查看 `hoi_task.py`（第 868-877 行）与 `rewards.py`（第 252-269 行）的腾空惩罚公式：
$$r_{\text{air}} = 5.0 \times \min(\text{air\_time} - 0.5, 0) \times (1 - p_{\text{prev\_gnd}}) \cdot p_{\text{curr\_gnd}}$$
- **数学死锁机制**：`r_air` 恒为**非正惩罚项**（权重高达 `5.0`）。在 `CarryBox` 正常步频下，单步摆动腾空时间通常仅 `0.25s ~ 0.40s`（小于阈值 `0.5s`）。这意味着**只要抬脚迈步，落地瞬间必定触发重罚**；而**如果双脚死死贴地滑行（`p_prev_gnd = 1`），则 `1 - p_prev_gnd = 0`，惩罚立刻清零**！
- 与之对比，惩罚滑行的 `feet_slide` 权重仅 `-0.1`（相差 50 倍）；而抬脚 `8 cm` 在 14 个跟踪刚体求平均的 `r_body_pos`（`hoi_task.py` 第 782-783 行）中仅贡献约 `0.0012` 的微弱分差。在一阶解析梯度 BPTT 下，优化器直接锁定了最划算的作弊解：**绝不抬脚，双脚贴地搓行**。

---

### 2.2 问题二：为什么到 3500 轮仍抱不起箱子、箱子倒地、碰身体失败？

#### 根因 1：总交互轮数差距 + `init_pool` 课程状态池推进过慢
1. 原版 SUGAR（`SUGAR-main/train.sh` 第 11-17 行）在 `4096` 环境下训练 **`30,001` 轮**。
2. 在 `base_refiner_env_cfg.py`（第 188-192 行）与 `commands.py`（第 1303-1370 行）中：
   - `init_with_ref = False`（注：这是为了防止原始未精炼 MoCap 在半空存在初始穿透或零夹紧力导致开局炸飞，设计初衷合理）；
   - 但 `pool_warmup_steps = 1000 * 24` 锁死了**前 1000 轮 100% 环境只能从 `t = 0`（远离箱子起点）出发**；
   - 1000 轮后，`update_interval = 24 * 100` 导致**每 100 轮才更新一次 `init_pool`**，且必须连续存活 `48` 步不终止才能解锁新帧。到 `3500` 轮总共只更新了 25 次——一旦在“弯腰抓箱子”那一刻卡住，后续几百帧“抱箱行走”的数据在 Replay Buffer 中占比为 **0%**，世界模型根本没见过持箱状态。

#### 根因 2：无把手光滑箱子力闭合夹持的“模型梯度死区”与上肢探索不足
- **物理机制**：G1 末端是无手指光滑橡胶掌（`left/right_rubber_hand`），箱子重 `0.5~1.0 kg`、摩擦系数最低随机至 `0.2`，且每 `1.5~3.0s` 遭受 `push_object`（`±0.5 m/s`、`±1.0 rad/s`）随机冲击（`carry_box_refiner_env_cfg.py` 第 66-82 行）。双手 PD 目标 `q_sp` 必须**穿透进箱子几何体内部持续向内挤压**，才能靠静摩擦力 $2\mu F_N > mg$ 托住箱子。
- **梯度断层**：运动学参考 `q_ref` 仅贴在箱子表面（法向力 $\approx 0$）。原版 PPO 靠 `init_noise_std = 1.0` 的大幅随机探索在真实 PhysX 中撞出“大力向内夹紧”样本并直接强化；而当前 `SuperMimic` 的 `init_noise_std` 仅 `0.20`（小 5 倍），Replay Buffer 中几乎全是未夹紧滑落样本，导致世界模型学不到“双手向内加力 $\Rightarrow$ 箱子随手上升”的接触相变，BPTT 导数 $\frac{\partial \text{obj\_pos}_{t+1}}{\partial a_t} \approx 0$，一抬手箱子便滑落倒地。

#### 根因 3：世界模型缺失躯干/前臂相对物体特征，诱导“怀搂碰身体”投机并引发碰撞翻箱
- G1 肩/肘扭矩上限仅 `25 N·m`、手腕仅 `5 N·m`，双掌悬空夹不住时，把箱子往胸口/肚子/前臂上搂（三点支撑）最省力。
- 在真实环境中，箱子一碰除双掌/双脚外的任何部位，不仅触发 `undesired_contacts`（阈值仅 `0.1 N`，权重 `-1.0`），还会产生刚体碰撞反弹把箱子顶歪，瞬间触发 `bad_obj_ori > 0.8` 或 `bad_obj_pos > 0.3` 终止失败！
- **代码特征盲区**：`hoi_task.py`（第 363-367 行）构建世界模型特征 `rel_c_obj` 时，仅传入了 `SUGAR_CONTACT_BODY_NAMES` 中的 7 个点（双掌、双脚、双髋、骨盆），**完全没有传入最常发生违规接触的 `torso_link`（胸口躯干）、`left/right_elbow_link`（前臂）与 `left/right_wrist_yaw_link`（手腕）**！加之 `wm_contact_loss_weight` 仅 `0.2`（分摊到第 5 维 `undesired` 仅 `0.04`），世界模型根本意识不到箱子贴近胸口/小臂会撞翻扣分，导致 BPTT 持续把策略往“贴身体抱箱”的死胡同里推。

---

### 2.3 问题三：为什么 SuperMimic 替换 PPO 后没有加速反而变慢？

在 `NUM_ENVS = 4096`（每轮采集 `4096 × 24 = 98,304` 步真实交互）下对比三者的真实计算与学习吞吐量：

| 对比项（折算至每 24 环境步 = 1 Iteration） | 原版 SUGAR PPO | 原版 SuperMimic (`allmimic`) | 当前 `fusion` 版 SuperMimic |
| :--- | :--- | :--- | :--- |
| **世界模型网络规模 & 输出归一化** | 无 | `(2048, 2048, 2048)` + `out_norm` | `(512, 512, 512)`（**小 16 倍**）+ **无 `out_norm`** |
| **每轮世界模型更新量** | 0 | ~96 次 × batch `2048` × `8` 步 | **仅 8 次** × batch `1024` × `8` 步 |
| **每轮策略更新样本吞吐量** | 20 次更新，刷 **`491,520`** 真实步 | ~9.6 次 × `4096 × 32` = **`1,258,291`** 想象步 | 4 次 × `512 × 32` = **仅 `65,536` 想象步** |
| **当轮新采真实状态抽样覆盖率** | **100%**（`98,304` 步重复刷 5 轮） | 高重用率（UTD 极高） | 仅抽 `4 × 512 = 2,048` 起点（**仅占新数据的 2.1%！**） |
| **单轮正向运动学与编译加速** | 无额外开销，极快 | `bf16` AMP + `torch.compile` CUDA Graph | `fp32` 未编译 + **每步重复算 2 次 FK（共 64 次 Python 串行循环）** |

**结论**：当前配置每轮采集了 `9.8 万`步真实物理数据，却只抽出 `2,048` 个起点做小批量想象（总想象步数比原版 PPO 刷过的真实步数还少 **7.5 倍**，比原版 SuperMimic 少 **19 倍**），同时在 `trainer.py`（第 500 与 535 行）的 32 步循环中**未复用上一步算好的 `fk_next`**，白白多算了一倍的 34 连杆 Python 循环正向运动学且未启用 `bf16` TensorCore，因此既慢又欠拟合。

---

## 3. 分模块修改方案（遵循 KISS 原则）

所有改动保持最小侵入性、向后兼容既有测试与检查点结构，不引入冗余抽象：

### 方案模块 M1：动作空间上下肢解耦 + 纯残差平滑约束（解决姿势别扭）
1. **上下肢混合动作模式（`SuperMimicPolicy`，位于 `networks.py`）**：
   - 根据 `joint_names` 构建 29 维掩码 `ref_residual_mask`（注册为持久 `buffer`）：
     - **双腿 12 关节（含 `hip`、`knee`、`ankle`）**：掩码设为 `0.0`，采用相对于 `default_joint_pos` 的绝对目标控制（彻底解放双腿动态平衡，不再与非物理 MoCap 参考打架）；
     - **腰部 + 双臂 17 关节**：掩码设为 `1.0`（可通过 CLI `--legs_residual` 切换回旧行为），保留上半身参考引导。
2. **想象奖励增加纯网络输出平滑项（`hoi_task.py`）**：
   - 在保留原有环境对齐奖励的基础上，支持对策略纯网络残差输出施加微小的幅值正则（`-0.02 * (residual**2).sum(-1)`），防止手腕小 `action_scale` 引起的跳变对抗。

### 方案模块 M2：世界模型输出归一化 + 确定性 BPTT + 余弦退火（解决高频抖动）
1. **为 `JointWorldModel` 增加 `delta_rms` 输出标准差归一化（`networks.py` & `hoi_task.py`）**：
   - 在 `SugarHOITask` 中新增解析函数 `compute_state_delta(s_prev, s_true) -> (B, 82)`（`apply_delta` 的精确逆运算）；
   - 在 `JointWorldModel` 中新增 `delta_rms = RunningMeanStd(DELTA_DIM, eps=1e-4)`：训练世界模型时用真实有效窗口增量更新 `delta_rms`，前向预测归一化增量后经 `delta = raw_delta * std + mean` 还原物理增量；
   - 在 `world_model_state_loss` 中去掉人为压低速度的 `vel_loss_scale = 0.1`，改为用 `delta_rms.std` 对位置误差与速度误差做均衡量纲归一化，消除模型对高频速度跳变的低通滤波盲区。
2. **确定性想象展开 + 上下肢差异化探索退火（`trainer.py`）**：
   - `update_policy` 32 步 BPTT 中统一使用 `deterministic=True`（不在想象轨迹内部注入每步白噪声，移除推高噪声的 `entropy_coef`）；
   - 真实环境 `collect_rollout` 采用按关节分组的退火探索噪声（随迭代进度从初始值平滑衰减至 `30%`）：**下肢 12 关节初始噪声 `0.12`（稳步态、消抖动），腰部与双臂 17 关节初始噪声 `0.35`（强力探索抓取夹紧）**。
3. **余弦学习率衰减 + 按值梯度裁剪（`trainer.py`）**：
   - 为 `wm_opt`、`policy_opt`、`critic_opt` 增加余弦学习率调度（衰减至初始学习率的 `0.1` 倍）；
   - 在 `policy_loss.backward()` 的范数裁剪基础上增加 `nn.utils.clip_grad_value_(params, 0.1)`，抑制 32 步链式导数尖峰。

### 方案模块 M3：步态接触梯度解锁 + 脚踝摆动高度引导（解决脚底搓地）
1. **调整步态奖励参数（同步更新 `base_refiner_env_cfg.py`、`base_tracker_env_cfg.py` 与 `hoi_task.py`）**：
   - 将 `feet_air_time` 的 `threshold` 从 `0.5s` 下调至 **`0.25s`**，权重从 `5.0` 调为 **`1.5`**（符合负重行走真实摆动周期，正常迈步不再被误罚）；
   - 将 `feet_slide` 权重从 `-0.1` 增强至 **`-0.5`**（搓地滑行惩罚放大 5 倍）。
2. **切断贴地免罚梯度死锁 + 精细抬脚引导（`hoi_task.py`）**：
   - 计算 `p_first_contact = (1.0 - prev_feet_gnd.detach()) * curr_feet_gnd` 时对 `prev_feet_gnd` 阻断反向作弊梯度；
   - 在想象 BPTT 奖励中，增加一项针对左右脚踝（`left/right_ankle_roll_link`）相对参考高度差 `z` 的局部跟踪引导项 `0.25 * exp(-err_feet_z / (0.06**2))`，弥补 14 刚体平均对 `8 cm` 抬脚动作激励的稀释。

### 方案模块 M4：防碰撞几何感知 + 双掌夹紧引导 + 课程池加速（解决抱箱失败与碰身体）
1. **补齐世界模型的躯干/前臂防碰撞相对特征（`hoi_task.py`）**：
   - 利用 `FKResult.tracked_pos_w` 中**已现成算出**的 5 个关键防碰刚体（`torso_link`、`left_elbow_link`、`right_elbow_link`、`left_wrist_yaw_link`、`right_wrist_yaw_link`），计算它们在箱子局部坐标系下的相对位置（`5 × 3 = 15` 维），拼入 `state_features`（`WM_STATE_FEAT_DIM: 108 -> 123`，零额外 FK 开销，依然严格满足 `SE(2)` 平移与偏航不变性！）；
   - 将 `wm_contact_loss_weight` 从 `0.2` 提升至 **`0.5`**，强化违规身体接触预测精度。
2. **想象 BPTT 物理交互整形引导（仅作用于 `update_policy` 想象梯度，`hoi_task.py`）**：
   - **双掌力闭合靠近与夹紧引导**：当 `hoi_mode == "hands"` 且 `contact_label == 1` 时，对左右 `rubber_hand` 在箱子局部坐标系下偏离参考抓取点的误差施加高灵敏度指数吸引项 `0.5 * exp(-err_hand_obj / (0.10**2))`，为未接触状态提供连续非零的靠近与向内挤压梯度；
   - **躯干与前臂防贴靠排斥项**：当箱子中心与 `torso_link` 距离小于安全净空（如 `0.20 m`）时施加平滑二次惩罚，从几何梯度上直接阻止“把箱子往胸口搂”。
3. **安全提速 `init_pool` 课程解锁 + 打散初始重置相位（`base_refiner_env_cfg.py`、`commands.py`、`env_bridge.py`）**：
   - 保持 `init_with_ref = False`（避免原始 MoCap 半空穿透），将 `base_refiner_env_cfg.py` 中 `pool_warmup_steps` 从 `1000 * 24` 缩短至 **`200 * 24`**；
   - 将 `commands.py`（第 1303 行）中 `init_pool` 的写入间隔 `update_interval` 从 `24 * 100`（每 100 轮）缩短至 **`24 * 25`（每 25 轮更新一次，解锁提速 4 倍）**；
   - 在 `SugarEnvBridge.__init__` 末尾随机初始化 `env.episode_length_buf`（对齐原版 RSL-RL `init_at_random_ep_len=True`），消除 4096 环境的同步重置波峰。

### 方案模块 M5：消除冗余 FK + `bf16` 混合精度 + 对齐 A100 高重用率 UTD（解决训练变慢）
1. **零损耗消除 32 步循环内的一半 FK 计算（`trainer.py` & `urdf_kinematics.py`）**：
   - 为 `FKResult` 增加轻量的直通梯度缩放方法 `scale_grad(alpha)`（与 `s_cur = s_next * alpha + s_next.detach() * (1 - alpha)` 数学严格等价）；
   - 在 `for tau in range(H):` 循环外仅计算 1 次初始 `fk_cur = self.task.run_fk(s_cur)`，循环内每步直接复用 `fk_cur = fk_next.scale_grad(alpha)`，**单次策略更新的 34 连杆 FK 调用次数直接从 64 次砍半至 33 次**！
2. **MLP 启用原生 `bfloat16` AMP（`networks.py`）**：
   - 参考 `allmimic/networks.py`，仅在 `JointWorldModel`、`SuperMimicPolicy`、`SuperMimicCritic` 的 MLP 矩阵乘内部开启 `torch.autocast("cuda", dtype=torch.bfloat16)`（输出立即转回 `float32`，FK、旋转四元数与 Loss 全程保持 `fp32`），在 Colab A100 上零精度风险吃满 TensorCore 吞吐并降低一半 MLP 激活显存。
3. **针对 Colab A100 (`NUM_ENVS = 4096`) 升级默认模型容量与重用率（`trainer.py`、`scripts/supermimic/train.py`、`SuperMimic_SUGAR_Repro.ipynb`）**：
   - `wm_hidden_dims`: `(512, 512, 512)` → **`(1024, 1024, 1024)`**
   - `wm_batch_size`: `1024` → **`2048`**，`wm_updates_per_iter`: `8` → **`16`**（世界模型每轮学习量提升 **4 倍**）
   - `policy_batch_size`: `512` → **`2048`**，`policy_updates_per_iter`: `4` → **`8`**（策略每轮消费 **`524,288`** 想象步，提升 **8 倍**，小环境冒烟测试时仍由 `train.py` 自动按比例向下缩放保护）
   - 更新 `SuperMimic_SUGAR_Repro.ipynb`：A100 显存探测下直接支持 `NUM_ENVS = 4096`，`full` 模式推荐 `RL_ITERS = 10001`（仅为原版 SUGAR PPO `30,001` 轮的 `1/3`）。

---

## 4. 涉及修改的文件清单与显存预算评估（Colab A100, 4096 Envs）

### 4.1 拟修改文件清单

| 序号 | 文件路径 | 核心修改内容 |
| :---: | :--- | :--- |
| 1 | `source/sugar_rl/sugar_rl/supermimic/urdf_kinematics.py` | `FKResult` 新增 `scale_grad(alpha)` 方法，支持 BPTT 跨步零误差复用 `fk_next`。 |
| 2 | `source/sugar_rl/sugar_rl/supermimic/hoi_task.py` | ① 新增 `compute_state_delta` 与按 `delta_rms` 归一化的 WM 损失；② `state_features` 补齐躯干/肘/腕 5 个刚体相对箱子坐标（`108 -> 123` 维，保持 `SE(2)` 不变性）；③ 更新 `feet_air_time`/`feet_slide` 参数并加入脚踝摆动高度、双掌夹紧与躯干防碰撞引导项。 |
| 3 | `source/sugar_rl/sugar_rl/supermimic/networks.py` | ① `JointWorldModel` 增加 `delta_rms` 输出反归一化并默认 `(1024,1024,1024)`；② `SuperMimicPolicy` 实现下肢绝对 + 上肢残差掩码；③ MLP 前向支持 CUDA `bfloat16` AMP。 |
| 4 | `source/sugar_rl/sugar_rl/supermimic/trainer.py` | ① 循环复用 `fk_next`；② 想象展开改为 `deterministic=True`；③ 真实采样采用上下肢分组退火探索噪声；④ 加入余弦学习率衰减与 `clip_grad_value_(0.1)`；⑤ 升级默认 UTD 与 batch size。 |
| 5 | `source/sugar_rl/sugar_rl/supermimic/env_bridge.py` | 初始化重置后随机打散 `episode_length_buf`，避免 4096 环境同步 timeout。 |
| 6 | `source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/base_refiner_env_cfg.py` | 同步 `feet_air_time`（`threshold=0.25, weight=1.5`）、`feet_slide`（`weight=-0.5`），并将 `pool_warmup_steps` 缩短为 `200 * 24`。 |
| 7 | `source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_tracker/base_tracker_env_cfg.py` | 同步 `feet_air_time` 与 `feet_slide` 步态奖励参数，保持 Stage 1 与 Stage 2 奖励一致。 |
| 8 | `source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py` | 将 `init_pool` 验证写入周期 `update_interval` 从 `24 * 100` 缩短至 `24 * 25`。 |
| 9 | `scripts/supermimic/train.py` & `SuperMimic_SUGAR_Repro.ipynb` | 同步 CLI 默认超参（`wm_updates=16, policy_updates=8, wm_batch_size=2048, policy_batch_size=2048`）及 Colab A100 `NUM_ENVS=4096`、`RL_ITERS=10001` 配置。 |
| 10 | `tests/test_supermimic_fusion.py` | 适配新特征维度并补充新增机制（`delta_rms` 往返一致性、上下肢残差掩码、`fk_next` 复用梯度等价性）的单元测试。 |

> **断点兼容性提示**：由于 `JointWorldModel` 输入特征从 `108` 维扩展至 `123` 维、隐层扩至 `1024` 且新增 `delta_rms`，升级代码后 Stage 1 Refiner 需**新建实验目录（或清理旧的 512 维 `logs/refiner/model_*.pt`）从第 0 轮重新起训**，不可直接加载旧结构检查点续训。

### 4.2 Colab A100 (`NUM_ENVS = 4096`) 显存安全预算

- **Isaac Sim 5.1 + PhysX 4096 环境刚体与接触传感器**：约 `18 ~ 21 GB`
- **`HOIReplayBuffer(capacity=256, num_envs=4096)`**：约 `1.0 GB`
- **模型权重 + AdamW 状态 + `policy_batch_size=2048` 32 步 BPTT 激活图（复用 `fk_next` + MLP `bf16` AMP）**：约 `3.0 ~ 4.5 GB`
- **峰值总显存预估**：**`22 ~ 26.5 GB`**，在 Colab 标准 **A100-40GB**（及 A100-80GB）上留有超过 `13 GB` 的安全余量，不会发生 OOM。

---

## 5. 审核后验证与上线步骤

待你审核确认本方案后，将严格按以下流程执行：

1. **本地代码实现与回归测试（Windows PowerShell）**：
   在本地完成上述 10 个文件的修改，并在本地运行完整单元测试套件（验证旋转梯度、URDF FK、890/510 维观测切片对齐、`SE(2)` 不变性、32 步 BPTT 梯度等价性与断点续训）：
   ```powershell
   cd D:\robot\rework\fusion
   python tests\test_supermimic_fusion.py
   ```
   - **作用**：在不启动重型物理引擎的前提下，秒级验证全部张量维度、前反向梯度有限性与检查点读写逻辑无误。
   - **预期结果**：全部测试用例输出 `[PASS]`。

2. **本地提交并推送至 GitHub（Windows PowerShell）**：
   ```powershell
   cd D:\robot\rework\fusion
   git status -s
   git add .
   git commit -m "perf(supermimic): fix gait jitter/shuffling, add torso collision features and scale UTD for A100"
   git push origin main
   ```
   - **作用**：确认无大体积数据混入后，将修复代码与更新后的 Notebook 推送至 GitHub `main` 分支。

3. **Colab A100 拉取更新与冒烟/正式训练（Google Colab Notebook Cell）**：
   在 Colab 中重新运行 `Step 1`（自动 `git pull` 同步最新提交）→ `Step 4`（GPU 单元测试）→ `Step 5`（20 轮 Isaac Sim 实机冒烟测试，核对启动时 18 项观测对齐断言与 FPS 提升）→ 将 `EXP_NAME` 设为新实验名（避免混用旧模型架构 checkpoint），以 `NUM_ENVS = 4096` 启动 `Stage 6.1` 训练并通过内嵌 TensorBoard 观察 `wm/obj_err_mm`、`collect/sim_step_reward` 与 `Step 7` 预览视频中的抬脚和抱箱表现。

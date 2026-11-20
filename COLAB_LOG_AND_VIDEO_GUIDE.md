# Google Colab 日志导出与模型 MP4 视频生成操作手册

本手册整理了在 **Google Colab 云端实例** 中随时打包下载 TensorBoard 训练日志、以及在终端使用任意检查点生成 `.mp4` 视频并保存至 Google Drive 的标准命令。所有终端命令均支持**在主 Notebook 后台训练不断开的情况下，打开 Colab 左下角「终端（Terminal）」并行运行**。

---

## 一、一键打包并下载 TensorBoard 日志（不含大体积权重）

### 1.1 在 Google Colab 终端（Bash）执行（推荐：不打断正在运行的训练）

打开 Colab 左下角的 **终端（Terminal）**（若在空白 Notebook Cell 中运行，请在首行加上 `%%bash`），复制粘贴以下命令：

```bash
cd /content/fusion

# ==================== 按需修改实验目录名 ====================
EXP_FOLDER="CarryBox_sm_v2_diplo_lafan_full"   # 对应 outputs/<EXP_FOLDER>；若留空或不存在则自动打包 outputs 下全部日志
# ==========================================================

STAMP=$(date +%Y%m%d_%H%M%S)
WORK_TMP="/tmp/tb_pack_${STAMP}"
ZIP_NAME="tb_logs_${EXP_FOLDER}_${STAMP}.zip"
LOCAL_ZIP="/content/${ZIP_NAME}"
DRIVE_EXPORT_DIR="/content/drive/MyDrive/SuperMimic_SUGAR_Backup/tb_exports"

mkdir -p "$WORK_TMP"
TARGET_ROOT="/content/fusion/outputs/${EXP_FOLDER}"
if [ ! -d "$TARGET_ROOT" ]; then
    TARGET_ROOT="/content/fusion/outputs"
fi

# 1. 仅提取 events.out.tfevents.* 与 *.log 文本（严格排除几百 MB 的 .pt / .ckpt / .npz 文件）
rsync -a --prune-empty-dirs \
    --include="*/" \
    --include="events.out.tfevents.*" \
    --include="*.log" \
    --exclude="*" \
    "$TARGET_ROOT/" "$WORK_TMP/raw_logs/"

# 2. 同时将全部 TensorBoard 标量曲线导出为 CSV 与 JSON 摘要，方便快速精确分析每一轮数值
/content/.venv_fusion/bin/python - "$WORK_TMP/raw_logs" "$WORK_TMP" << 'EOF'
import csv, json, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

raw_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
event_dirs = sorted({p.parent for p in raw_root.rglob("events.out.tfevents.*")})
summary = {}
csv_path = out_root / "tb_scalars_all.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["run", "tag", "step", "wall_time", "value"])
    for d in event_dirs:
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        ea.Reload()
        run_name = str(d.relative_to(raw_root))
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        summary[run_name] = {}
        for tag in sorted(tags):
            pts = ea.Scalars(tag)
            for p in pts:
                writer.writerow([run_name, tag, p.step, f"{p.wall_time:.3f}", f"{p.value:.6g}"])
            if pts:
                vals = [p.value for p in pts]
                summary[run_name][tag] = {
                    "count": len(pts),
                    "first_step": pts[0].step,
                    "first_val": round(pts[0].value, 6),
                    "last_step": pts[-1].step,
                    "last_val": round(pts[-1].value, 6),
                    "min_val": round(min(vals), 6),
                    "max_val": round(max(vals), 6),
                }

with open(out_root / "tb_scalars_summary.json", "w", encoding="utf-8") as f_js:
    json.dump(summary, f_js, indent=2, ensure_ascii=False)
print(f"[OK] Exported {len(summary)} TensorBoard run(s) -> CSV & JSON")
EOF

# 3. 打包成轻量 .zip 并同步至 /content/ 根目录与 Google Drive
python3 - "$WORK_TMP" "$LOCAL_ZIP" << 'EOF'
import sys, zipfile
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(src.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(src))
EOF

if [ -d "/content/drive/MyDrive" ]; then
    mkdir -p "$DRIVE_EXPORT_DIR"
    cp -f "$LOCAL_ZIP" "$DRIVE_EXPORT_DIR/"
    echo "[Google Drive] 已备份至: ${DRIVE_EXPORT_DIR}/${ZIP_NAME}"
fi

rm -rf "$WORK_TMP"
ls -lh "$LOCAL_ZIP"
```

- **命令作用**：只提取 `outputs/<EXP_FOLDER>` 下的 `events.out.tfevents.*` 与训练日志 `*.log`（跳过所有大体积模型权重），额外导出一份全量标量数据表 `tb_scalars_all.csv` 与统计摘要 `tb_scalars_summary.json`，打包成仅几百 KB ~ 几 MB 的 `.zip`，同时保存在 `/content/` 与 Google Drive。
- **关键参数含义**：
  - `EXP_FOLDER`：实验目录名（默认为 `"CarryBox_sm_v2_diplo_lafan_full"`；若改成其他实验名如 `"CarryBox_sm_v2_diplo_lafan_fast_verify"` 则只打包对应目录）。
  - `size_guidance={"scalars": 0}`：关闭 TensorBoard 默认抽样截断，100% 导出从第 0 轮到最新轮的每一个点。
- **预期结果与下载方式**：
  1. **方式 A（Colab 网页直接下载）**：点击 Colab 左侧边栏的文件夹图标（Files）并点刷新，在 `/content/` 根目录下找到 `tb_logs_CarryBox_...zip`，右键点击 **`Download`（下载）**；
  2. **方式 B（Google Drive 下载）**：进入 Google Drive 的 `SuperMimic_SUGAR_Backup/tb_exports/` 目录直接下载该 `.zip`；
  3. **发给我分析**：下载后将该 `.zip`（或解压后的文件夹）放入本地 `D:\robot\` 任意目录下（例如 `D:\robot\tb_logs\`），直接把本地路径发给我即可读取全量曲线分析。

### 1.2 （可选）训练暂停时在 Google Colab Notebook Python Cell 弹窗下载

如果主训练已经暂停或跑完，也可以在 Colab Notebook 新建一个普通 **Python 代码单元格**触发浏览器直接下载刚生成的最新日志包：

```python
from pathlib import Path
from google.colab import files

zips = sorted(Path("/content").glob("tb_logs_*.zip"), key=lambda p: p.stat().st_mtime)
assert zips, "请先在终端运行 1.1 命令生成日志压缩包！"
print(f"正在触发浏览器下载: {zips[-1]}")
files.download(str(zips[-1]))
```

### 1.3 （可选）在本地 Windows PowerShell 启动 TensorBoard 查看下载的日志

将 `.zip` 解压到本地（例如 `D:\robot\tb_logs\raw_logs`）后，可在 **Windows 本地 PowerShell** 运行：

```powershell
D:\anaconda\envs\DRL_env\Scripts\tensorboard.exe --logdir "D:\robot\tb_logs\raw_logs" --port 6006
```
- **作用与预期结果**：使用本地既有的 `DRL_env` 环境在 `http://localhost:6006` 打开离线 TensorBoard 面板（不修改任何本机包版本）。

---

## 二、在 Google Colab 终端生成 `.mp4` 视频并保存到 Google Drive

> **为什么不用 `--video`？** Google Colab 容器宿主机缺少 Vulkan 1.1 图形驱动，开启 Isaac Sim 原生相机渲染会报错退出。下方通用命令采用 **「Isaac Sim CUDA 真实物理执行（`num_envs=1`）+ MuJoCo `osmesa` 离屏渲染」**，物理 100% 运行在 Isaac Sim 上，视频生成后立即自动拷入你的 Google Drive（单环境仅占约 `4 GB` 显存，可在后台训练时于 Colab 终端并行执行）。

### 2.1 第一步：在 Colab 终端初始化通用渲染脚本（同一 Colab 会话只需运行一次）

在 **Google Colab 终端（Bash）** 中粘贴执行一次即可生成通用渲染器 `/tmp/render_sugar_mp4.py`：

```bash
cat << 'EOF' > /tmp/render_sugar_mp4.py
import os, shutil, sys
os.environ["MUJOCO_GL"] = "osmesa"
sys.path.insert(0, "/content/fusion")

import torch, gymnasium as gym, mujoco, imageio
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import sugar_rl.tasks  # noqa: F401
from sugar_rl.utils.parser_cfg import parse_env_cfg
from sugar_rl.supermimic.trainer import load_supermimic_policy
from sugar_rl.supermimic.env_bridge import SugarEnvBridge
from scripts.sim2sim.scene_builder import build_mujoco_scene_xml

(
    task_name, stage_mode, ckpt_path, gen_ckpt, motion_dir,
    local_out, drive_out, model_tag, n_trajs, max_inf_steps, stride
) = (
    sys.argv[1], sys.argv[2].lower(), sys.argv[3], sys.argv[4], sys.argv[5],
    sys.argv[6], sys.argv[7], sys.argv[8], int(sys.argv[9]), int(sys.argv[10]), int(sys.argv[11])
)

stage_suffix = {"refiner": "Refiner", "tracker": "Tracker", "inference": "Inference"}[stage_mode]
task_id = f"Sugar-G129dof-{task_name}-{stage_suffix}"
is_inf = (stage_mode == "inference")
entry_key = "play_env_cfg_entry_point" if is_inf else "env_cfg_entry_point"

env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=1, entry_point_key=entry_key)
env_cfg.episode_length_s = 120.0
if hasattr(env_cfg, "events"):
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "push_object"):
        env_cfg.events.push_object = None

if hasattr(env_cfg, "observations"):
    if hasattr(env_cfg.observations, "teacher"):
        env_cfg.observations.teacher = None
    if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False

m_cfg = env_cfg.commands.motion
m_cfg.motion_folder = motion_dir
from sugar_rl.assets.objects.objects import apply_dataset_box_spec
apply_dataset_box_spec(env_cfg, motion_dir)
if is_inf:
    assert gen_ckpt and os.path.isfile(gen_ckpt), f"inference 模式必须提供有效的 GEN_CKPT，当前为: '{gen_ckpt}'"
    m_cfg.generator_checkpoint_path = gen_ckpt
    if hasattr(m_cfg, "eval_random_motion"):
        m_cfg.eval_random_motion = True
else:
    m_cfg.start_init_env_ratio = 1.0
    m_cfg.pose_range = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    m_cfg.joint_position_range = (0.0, 0.0)

env = gym.make(task_id, cfg=env_cfg)
bridge = SugarEnvBridge(env=env, task_name_or_id=task_id, verify_obs_on_init=False)
policy = load_supermimic_policy(ckpt_path, device="cuda:0")
joint_names = list(bridge.fk.env_joint_names)

xml_path, _, _ = build_mujoco_scene_xml(task_name, motion_folder=motion_dir)
mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
mj_data = mujoco.MjData(mj_model)
p_adr = int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_freejoint")])
o_adr = int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")])
j_adrs = [int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in joint_names]
t_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_vis")
t_mocap_id = int(mj_model.body_mocapid[t_body_id]) if t_body_id >= 0 else -1

cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 2.8, -18.0, 135.0
renderer = mujoco.Renderer(mj_model, height=480, width=854)
fps_out = max(1, 50 // max(1, stride))
os.makedirs(local_out, exist_ok=True)
os.makedirs(drive_out, exist_ok=True)

for traj_idx in range(1, n_trajs + 1):
    obs_dict, _ = env.reset()
    bridge.current_obs_dict = obs_dict
    origin = bridge.env.scene.env_origins[0]
    goal_pos = (bridge.motion_cmd.obj_target_pos_w[0] - origin).detach().cpu().numpy()
    goal_quat = bridge.motion_cmd.obj_target_quat_w[0].detach().cpu().numpy()
    m_id = int(bridge.motion_cmd.motion_id[0].item())
    total_frames = max_inf_steps if is_inf else int(bridge.motion_cmd.motion.time_step_total_permotion[m_id].item())

    traj_s = [bridge.get_joint_state()[0].detach().cpu().numpy()]
    status = "max_steps"
    step_idx = 0
    try:
        while step_idx < (max_inf_steps if is_inf else (total_frames + 10)):
            with torch.no_grad():
                act = policy.act_inference(obs_dict["policy"])
            obs_dict, _, dones, timeouts, _, _ = bridge.step(act)
            if not is_inf:
                if step_idx <= 1 and bool(dones[0].item()):
                    origin = bridge.env.scene.env_origins[0]
                    goal_pos = (bridge.motion_cmd.obj_target_pos_w[0] - origin).detach().cpu().numpy()
                    goal_quat = bridge.motion_cmd.obj_target_quat_w[0].detach().cpu().numpy()
                    m_id = int(bridge.motion_cmd.motion_id[0].item())
                    total_frames = int(bridge.motion_cmd.motion.time_step_total_permotion[m_id].item())
                    traj_s = [bridge.get_joint_state()[0].detach().cpu().numpy()]
                    step_idx += 1
                    continue
                if bool(dones[0].item()):
                    status = "trajectory_complete" if bool(timeouts[0].item()) else "terminated_early"
                    break
            traj_s.append(bridge.get_joint_state()[0].detach().cpu().numpy())
            step_idx += 1
            if not is_inf and len(traj_s) >= total_frames:
                status = "trajectory_complete"
                break
    except SystemExit:
        status = "system_exit"

    sub_traj = traj_s[::max(1, stride)]
    frames = []
    for s in sub_traj:
        mj_data.qpos[p_adr:p_adr+3] = s[0:3]
        mj_data.qpos[p_adr+3:p_adr+7] = s[3:7]
        mj_data.qpos[j_adrs] = s[7:36]
        mj_data.qpos[o_adr:o_adr+3] = s[71:74]
        mj_data.qpos[o_adr+3:o_adr+7] = s[74:78]
        if t_mocap_id >= 0:
            mj_data.mocap_pos[t_mocap_id] = goal_pos
            mj_data.mocap_quat[t_mocap_id] = goal_quat
        mujoco.mj_forward(mj_model, mj_data)
        cam.lookat[:] = [float(s[0]), float(s[1]), 0.85]
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())
    if frames:
        frames.extend([frames[-1]] * (fps_out // 2))  # 末尾定格 0.5s 方便观察结束姿态

    fname = f"{task_name}_{stage_mode}_{model_tag}_traj{traj_idx}_{status}_{len(traj_s)}steps.mp4"
    local_mp4 = os.path.join(local_out, fname)
    drive_mp4 = os.path.join(drive_out, fname)
    if frames:
        imageio.mimsave(local_mp4, frames, fps=fps_out)
        shutil.copy2(local_mp4, drive_mp4)
        print(f"[SAVED] ({status}, {len(traj_s)}/{total_frames} steps) -> {drive_mp4}", flush=True)
    else:
        print(f"[WARN] 轨迹 {traj_idx} 未采集到有效帧，已跳过保存 ({status})", flush=True)

renderer.close()
env.close()
simulation_app.close()
EOF
echo "[OK] 通用渲染脚本已就绪: /tmp/render_sugar_mp4.py"
```

---

### 2.2 第二步：执行视频生成并保存到 Google Drive（每次录制运行这一段）

在 **Google Colab 终端（Bash）** 中运行（默认指向 `model_latest.pt`，换模型只需改顶部配置区；整段由 `( ... )` 子 Shell 包裹，失败时只会打印错误而**不会导致终端 `exit / disconnect`**）：

```bash
(
cd /content/fusion || exit 1

# ==================== 换模型生成视频时，修改以下配置区 ====================
TASK_NAME="CarryBox"          # 1. 任务名: CarryBox | PushBox | KickBox | PickBottle | StandBottle | SitChair
STAGE_MODE="refiner"          # 2. 阶段模式: refiner (Stage 1) | tracker (Stage 3) | inference (Stage 6 闭环)
CKPT_PATH="/content/fusion/outputs/CarryBox_sm_v2_diplo_lafan_full/logs/refiner/model_400.pt"  # 3. 策略 .pt 权重路径（也可指定 model_200.pt / model_400.pt / model_1400.pt）
GEN_CKPT=""                   # 4. 扩散生成器 .ckpt 路径（仅 STAGE_MODE="inference" 必填，其余保持 ""）
MOTION_DIR="data/OmniCarryBox" # 5. 参考轨迹目录（OmniRetarget 推荐用 data/OmniCarryBox；tracker 模式也可指向 .../rl_dataset）

NUM_TRAJS=3                   # 连续录制几条轨迹的 MP4
INF_STEPS=700                 # 仅 inference 闭环模式使用的最大步数（700 步 = 14 秒）
STRIDE=2                      # 抽帧步长：2 表示 25fps（设为 1 则输出 50fps）
# =========================================================================

test -f /tmp/render_sugar_mp4.py || { echo "[ERROR] 找不到 /tmp/render_sugar_mp4.py，请先运行上方 2.1 节命令生成它！"; exit 1; }
test -f "$CKPT_PATH" || { echo "[ERROR] 找不到权重文件: $CKPT_PATH"; exit 1; }
test -d "/content/drive/MyDrive" || { echo "[ERROR] Google Drive 未挂载 (/content/drive/MyDrive 不存在)！"; exit 1; }

# 自动提取权重文件名（如 model_latest / model_600 / refiner / tracker）建立独立目录，防止互相覆盖
MODEL_TAG=$(basename "$CKPT_PATH" .pt)
DRIVE_VIDEO_DIR="/content/drive/MyDrive/SuperMimic_SUGAR_Backup/videos/${TASK_NAME}_${STAGE_MODE}_${MODEL_TAG}"
LOCAL_VIDEO_DIR="/content/fusion/outputs/videos_preview/${TASK_NAME}_${STAGE_MODE}_${MODEL_TAG}"

OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
PYTHONPATH="/content/fusion:/content/fusion/source/sugar_rl:/content/fusion/source/sugar_il" \
/content/.venv_fusion/bin/python -u /tmp/render_sugar_mp4.py \
    "$TASK_NAME" "$STAGE_MODE" "$CKPT_PATH" "$GEN_CKPT" "$MOTION_DIR" \
    "$LOCAL_VIDEO_DIR" "$DRIVE_VIDEO_DIR" "$MODEL_TAG" "$NUM_TRAJS" "$INF_STEPS" "$STRIDE"
rc=$?

# 兼容 Isaac Sim 退出析构偶发返回码 139（前提是已实际生成目标 mp4），避免 `|| true` 静默吞掉真实错误
if [ "$rc" -ne 0 ] && ! { [ "$rc" -eq 139 ] && ls "$DRIVE_VIDEO_DIR"/*.mp4 >/dev/null 2>&1; }; then
    echo "[ERROR] 视频生成失败 (exit code=$rc)。若后台正在全量训练，请先检查 nvidia-smi 剩余显存（需约 6~8 GB 余量）。"
    exit "$rc"
fi

echo "=== 已保存到 Google Drive 的 MP4 文件列表 ==="
ls -lh "$DRIVE_VIDEO_DIR"/*.mp4
)
```

- **命令作用**：根据 `STAGE_MODE` 自动构建对应的无噪声 Isaac Sim 测试环境，加载 `CKPT_PATH` 连续回放 `NUM_TRAJS` 条轨迹，将包含机器人、物体与**半透明绿色目标位置框**的 `.mp4` 直接保存到 Google Drive 专属文件夹。
- **预期结果**：约 `1 ~ 2 分钟` 后，视频自动保存在你的 Google Drive：`MyDrive/SuperMimic_SUGAR_Backup/videos/<任务>_<阶段>_<模型名>/` 下，文件名自带状态（`trajectory_complete` 或 `terminated_early`）与实际存活步数。

---

## 三、换不同模型生成视频时，需要修改哪些地方？

每次换模型运行 **2.2 节命令** 时，**下方的执行代码与 `/tmp/render_sugar_mp4.py` 完全不用动**，只需要修改顶部 `# === 配置区 ===` 中的对应变量：

### 3.1 修改项速查对照表

| 你要切换的场景                                                                                                        | 需要修改的变量                                              | 具体怎么改                                                                                                                                                                                                                                          |
| :-------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **场景 A：同一阶段换不同轮数权重**<br>（如 `model_200.pt` → `model_400.pt` / `model_latest.pt` / `ckpts/refiner.pt`） | **仅改 `CKPT_PATH`（1 处）**                                | 把路径末尾文件名改为 `model_400.pt` 等即可。输出目录会自动按 `MODEL_TAG` 分开（如 `CarryBox_refiner_model_400`），不会覆盖旧视频。                                                                                                                  |
| **场景 B：换成 Stage 3 `Tracker` 模型**<br>（`510` 维观测的全身跟踪器，不含扩散生成器）                               | **改 `STAGE_MODE`、`CKPT_PATH`**<br>（可选改 `MOTION_DIR`） | 1. `STAGE_MODE="tracker"`（**必改**，否则报观测维度 890 vs 510 不匹配）<br>2. `CKPT_PATH` 指向 `logs/tracker/model_xxx.pt` 或 `ckpts/tracker.pt`<br>3. `MOTION_DIR` 可填精炼集 `outputs/.../rollout_datasets/refiner/rl_dataset` 或 `data/CarryBox` |
| **场景 C：换成 Stage 6 全自主闭环推理**<br>（`tracker.pt` + 扩散生成器 `generator.ckpt`）                             | **改 `STAGE_MODE`、`CKPT_PATH`、`GEN_CKPT`**                | 1. `STAGE_MODE="inference"`<br>2. `CKPT_PATH` 指向 `tracker.pt`（510 维）<br>3. `GEN_CKPT` 指向 `ckpts/generator.ckpt`（或官方 `demo_ckpts/CarryBox/generator.ckpt`）                                                                               |
| **场景 D：换成其他交互任务**<br>（如 `PushBox`、`PickBottle`、`SitChair` 等）                                         | **改 `TASK_NAME`、`MOTION_DIR`、`CKPT_PATH`**               | 例如推箱子：`TASK_NAME="PushBox"`，`MOTION_DIR="data/PushBox"`，并将 `CKPT_PATH` 指向对应任务的 `.pt` 权重。                                                                                                                                        |

---

### 3.2 四种常见切换的配置区复制模板

只需用下面任意一段替换 **2.2 节** 顶部的配置区即可：

#### 模板 1：查看 Stage 1 Refiner 其他轮数（例如 `model_400.pt` 或最新 `model_latest.pt`）
```bash
TASK_NAME="CarryBox"
STAGE_MODE="refiner"
CKPT_PATH="/content/fusion/outputs/CarryBox_sm_v2_diplo_lafan_full/logs/refiner/model_400.pt"
GEN_CKPT=""
MOTION_DIR="data/OmniCarryBox"
```

#### 模板 2：查看 Stage 3 Tracker 中途或最终权重（例如 `logs/tracker/model_1000.pt`）
```bash
TASK_NAME="CarryBox"
STAGE_MODE="tracker"
CKPT_PATH="/content/fusion/outputs/CarryBox_sm_v2_diplo_lafan_full/logs/tracker/model_1000.pt"
GEN_CKPT=""
MOTION_DIR="/content/fusion/outputs/CarryBox_sm_v2_diplo_lafan_full/rollout_datasets/refiner/rl_dataset"
```

#### 模板 3：查看 Stage 6 最终闭环自主推理（`tracker.pt` + `generator.ckpt`）
```bash
TASK_NAME="CarryBox"
STAGE_MODE="inference"
CKPT_PATH="/content/fusion/outputs/CarryBox_sm_v2_diplo_lafan_full/ckpts/tracker.pt"
GEN_CKPT="/content/fusion/outputs/CarryBox_sm_v2_diplo_lafan_full/ckpts/generator.ckpt"
MOTION_DIR="data/OmniCarryBox"
```

#### 模板 4：切换其他任务（例如 `PickBottle` 抓水瓶 Stage 1 Refiner，需提前准备好对应 `data/PickBottle`）
```bash
TASK_NAME="PickBottle"
STAGE_MODE="refiner"
CKPT_PATH="/content/fusion/outputs/PickBottle_sm_v2_diplo_lafan_full/logs/refiner/model_1000.pt"
GEN_CKPT=""
MOTION_DIR="data/PickBottle"
```

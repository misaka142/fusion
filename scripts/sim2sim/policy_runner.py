"""Zero-IsaacLab Policy & Physics Runner for Local MuJoCo Sim-to-Sim Deployment.

Supports:
1. Both checkpoint formats seamlessly:
   - SuperMimic fused checkpoints (`format="supermimic_sugar_v2"`, `policy_state_dict`)
   - Original SUGAR RSL-RL / BCPPO `demo_ckpts/<TASK>/tracker.pt` (`model_state_dict`, pure PyTorch)
2. Both Sim-to-Sim execution modes:
   - `replay`: Open-loop reference command playback from `data/<TASK>/data_xxx` (isolates low-level Tracker & PD physics)
   - `autonomous`: Closed-loop 10Hz `GeneratorWrapper` (`generator.ckpt`) + 50Hz `Tracker` (`tracker.pt`) + 500Hz MuJoCo
     (works with dataset trajectories OR completely standalone custom initial/target object poses + live mouse target dragging!)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_FUSION_ROOT = Path(__file__).resolve().parents[2]
_SUGAR_RL_SRC = _FUSION_ROOT / "source" / "sugar_rl"
_SUGAR_IL_SRC = _FUSION_ROOT / "source" / "sugar_il"
for _p in (_SUGAR_RL_SRC, _SUGAR_IL_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sugar_rl.pickle_compat import load_pickle_compat  # noqa: E402
from sugar_rl.supermimic.hoi_task import HISTORY_LEN, TRACKER_OBS_DIM, TrackerHistoryState  # noqa: E402
from sugar_rl.supermimic.rotations import (  # noqa: E402
    quat_apply_inverse,
    quat_normalize,
    rotmat_to_quat,
    subtract_frame_transforms,
    sugar_rot6d_from_quat,
)
from sugar_rl.supermimic.trainer import load_supermimic_policy  # noqa: E402
from sugar_rl.supermimic.urdf_kinematics import (  # noqa: E402
    G1_29DOF_SDK_JOINT_NAMES,
    build_g1_actuator_params,
)

from .scene_builder import (  # noqa: E402
    G1_ISAACLAB_BFS_JOINT_NAMES,
    Sim2SimTaskSpec,
    build_mujoco_scene_xml,
)


# -----------------------------------------------------------------------------
# Dual-Format Tracker Checkpoint Adapter (SuperMimic `.pt` + SUGAR `demo_ckpts`)
# -----------------------------------------------------------------------------

class _LegacyRslRlActorWrapper(nn.Module):
    """Pure-PyTorch inference wrapper for original SUGAR RSL-RL `demo_ckpts` (`model_state_dict`)."""

    def __init__(self, state_dict: dict[str, Tensor], obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        actor_w_keys = sorted(
            [k for k in state_dict.keys() if k.startswith("actor.") and k.endswith(".weight")],
            key=lambda x: int(x.split(".")[1]),
        )
        if not actor_w_keys:
            raise ValueError("Legacy RSL-RL checkpoint missing 'actor.*.weight' keys.")

        layers: list[nn.Module] = []
        for idx, w_key in enumerate(actor_w_keys):
            layer_idx = w_key.split(".")[1]
            b_key = f"actor.{layer_idx}.bias"
            w = state_dict[w_key]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(state_dict[b_key])
            layers.append(lin)
            if idx < len(actor_w_keys) - 1:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)

        # Optional empirical normalizer compatibility
        self.has_norm = "actor_obs_normalizer._mean" in state_dict and "actor_obs_normalizer._std" in state_dict
        if self.has_norm:
            self.register_buffer("obs_mean", state_dict["actor_obs_normalizer._mean"].view(-1).float())
            self.register_buffer("obs_std", state_dict["actor_obs_normalizer._std"].view(-1).float().clamp_min(1e-6))

    @torch.no_grad()
    def act_inference(self, obs: Tensor) -> Tensor:
        x = (obs - self.obs_mean) / self.obs_std if self.has_norm else obs
        return self.actor(x)


@dataclass
class LoadedTrackerBundle:
    """Unified Tracker policy + exact joint ordering & actuator parameters."""

    policy: nn.Module
    ckpt_format: str               # "supermimic" | "rsl_rl_legacy"
    joint_names: list[str]         # 29 joint names in policy observation/action order
    default_joint_pos: Tensor      # (29,) in `joint_names` order
    action_scale: Tensor           # (29,) in `joint_names` order
    stiffness: Tensor              # (29,) in `joint_names` order
    damping: Tensor                # (29,) in `joint_names` order
    effort_limit: Tensor           # (29,) in `joint_names` order


def _infer_joint_order_from_default_pos(default_q: Tensor) -> list[str]:
    """Distinguish IsaacLab BFS vs Unitree SDK DFS joint order via `default_joint_pos` signature."""
    q_cpu = default_q.detach().cpu().float().view(29)
    bfs_q = build_g1_actuator_params(G1_ISAACLAB_BFS_JOINT_NAMES, device="cpu").default_joint_pos
    sdk_q = build_g1_actuator_params(G1_29DOF_SDK_JOINT_NAMES, device="cpu").default_joint_pos
    err_bfs = float((q_cpu - bfs_q).abs().max().item())
    err_sdk = float((q_cpu - sdk_q).abs().max().item())
    return list(G1_ISAACLAB_BFS_JOINT_NAMES) if err_bfs <= err_sdk else list(G1_29DOF_SDK_JOINT_NAMES)


def load_tracker_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str = "cuda",
) -> LoadedTrackerBundle:
    """Load either a SuperMimic `tracker.pt` or an original SUGAR `demo_ckpts` `tracker.pt`."""
    dev = torch.device(device)
    payload = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint file (expected dict): {ckpt_path}")

    if "policy_state_dict" in payload:
        pol = load_supermimic_policy(ckpt_path, device=dev)
        if pol.obs_dim != TRACKER_OBS_DIM:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has obs_dim={pol.obs_dim} (likely a Stage 1 Refiner checkpoint of 890-D). "
                f"Sim-to-Sim requires a 510-D Stage 2 Tracker checkpoint (`tracker.pt`)."
            )
        j_names = getattr(pol, "joint_names", None)
        if not j_names or len(j_names) != 29:
            j_names = _infer_joint_order_from_default_pos(pol.default_joint_pos)
        act_params = build_g1_actuator_params(j_names, device=dev)
        return LoadedTrackerBundle(
            policy=pol,
            ckpt_format="supermimic",
            joint_names=list(j_names),
            default_joint_pos=pol.default_joint_pos.detach().clone().to(dev),
            action_scale=pol.action_scale.detach().clone().to(dev),
            stiffness=act_params.stiffness,
            damping=act_params.damping,
            effort_limit=act_params.effort_limit,
        )

    if "model_state_dict" in payload:
        sd = {k: v.to(dev) if isinstance(v, Tensor) else v for k, v in payload["model_state_dict"].items()}
        first_w = next(v for k, v in sorted(sd.items()) if k.startswith("actor.") and k.endswith(".weight"))
        obs_dim = int(first_w.shape[1])
        if obs_dim != TRACKER_OBS_DIM:
            raise ValueError(
                f"Legacy RSL-RL checkpoint '{ckpt_path}' has obs_dim={obs_dim}, expected {TRACKER_OBS_DIM} (`tracker.pt`)."
            )
        legacy_pol = _LegacyRslRlActorWrapper(sd, obs_dim=obs_dim, action_dim=29).to(dev)
        legacy_pol.eval()
        j_names = list(G1_ISAACLAB_BFS_JOINT_NAMES)
        act_params = build_g1_actuator_params(j_names, device=dev)
        return LoadedTrackerBundle(
            policy=legacy_pol,
            ckpt_format="rsl_rl_legacy",
            joint_names=j_names,
            default_joint_pos=act_params.default_joint_pos,
            action_scale=act_params.action_scale,
            stiffness=act_params.stiffness,
            damping=act_params.damping,
            effort_limit=act_params.effort_limit,
        )

    raise ValueError(
        f"Unrecognized Tracker checkpoint format in '{ckpt_path}' "
        f"(keys={list(payload.keys())[:8]})."
    )


# -----------------------------------------------------------------------------
# Reference Clip Loader (for `--mode replay` and optional `--motion_folder` init)
# -----------------------------------------------------------------------------

@dataclass
class MotionClipData:
    """Single 50Hz reference trajectory clip loaded from `data/<TASK>/data_xxx`."""

    clip_name: str
    length: int
    joint_pos_bfs: Tensor       # (T, 29) in IsaacLab BFS joint order
    joint_vel_bfs: Tensor       # (T, 29)
    root_pos_w: Tensor          # (T, 3) pelvis world pos
    root_quat_w: Tensor         # (T, 4) pelvis world quat (wxyz)
    root_lin_vel_w: Tensor      # (T, 3) pelvis world linear vel
    root_ang_vel_w: Tensor      # (T, 3) pelvis world angular vel
    obj_pos_w: Tensor           # (T, 3)
    obj_quat_w: Tensor          # (T, 4) wxyz
    obj_lin_vel_w: Tensor       # (T, 3)
    obj_ang_vel_w: Tensor       # (T, 3)
    contact_label: Tensor       # (T, 1) float in {0.0, 1.0}


def load_single_motion_clip(
    motion_folder: str | Path,
    motion_index: int = 0,
    device: torch.device | str = "cpu",
) -> MotionClipData:
    """Load one `data_xxx` clip from a SUGAR dataset folder (`data/<TASK>` or `rl_dataset`)."""
    folder = Path(motion_folder).resolve()
    if (folder / "robot_50hz.npz").exists():
        clip_dir = folder
    else:
        sub_dirs = sorted([d for d in folder.glob("data_*") if d.is_dir()])
        if not sub_dirs:
            raise FileNotFoundError(f"No 'data_*' trajectory folders found inside: {folder}")
        clip_dir = sub_dirs[int(motion_index) % len(sub_dirs)]

    with np.load(clip_dir / "robot_50hz.npz") as rob_npz:
        jpos = torch.as_tensor(rob_npz["joint_pos"], dtype=torch.float32, device=device)
        jvel = torch.as_tensor(rob_npz["joint_vel"], dtype=torch.float32, device=device)
        bpos = torch.as_tensor(rob_npz["body_pos_w"], dtype=torch.float32, device=device)
        bquat = torch.as_tensor(rob_npz["body_quat_w"], dtype=torch.float32, device=device)
        blin = torch.as_tensor(rob_npz["body_lin_vel_w"], dtype=torch.float32, device=device)
        bang = torch.as_tensor(rob_npz["body_ang_vel_w"], dtype=torch.float32, device=device)

    with open(clip_dir / "obj_motion_global_50hz.pkl", "rb") as f:
        obj_pkl = load_pickle_compat(f)

    contact_np = np.load(clip_dir / "contact_labels_50hz.npy")
    T = min(int(jpos.shape[0]), int(len(obj_pkl["obj_trans"])), int(len(contact_np)))

    obj_pos = torch.as_tensor(obj_pkl["obj_trans"][:T], dtype=torch.float32, device=device)
    obj_rot = torch.as_tensor(obj_pkl["obj_rot"][:T], dtype=torch.float32, device=device)
    obj_quat = quat_normalize(rotmat_to_quat(obj_rot))
    obj_lin = torch.as_tensor(obj_pkl["obj_lin_vel"][:T], dtype=torch.float32, device=device)
    obj_ang = torch.as_tensor(obj_pkl["obj_ang_vel"][:T], dtype=torch.float32, device=device)
    contact = torch.as_tensor(contact_np[:T], dtype=torch.float32, device=device).reshape(T, 1)

    # Index 0 of `body_*_w` is always `pelvis` in both 35-body raw and 14-body rollout datasets
    return MotionClipData(
        clip_name=clip_dir.name,
        length=T,
        joint_pos_bfs=jpos[:T],
        joint_vel_bfs=jvel[:T],
        root_pos_w=bpos[:T, 0],
        root_quat_w=quat_normalize(bquat[:T, 0]),
        root_lin_vel_w=blin[:T, 0],
        root_ang_vel_w=bang[:T, 0],
        obj_pos_w=obj_pos,
        obj_quat_w=obj_quat,
        obj_lin_vel_w=obj_lin,
        obj_ang_vel_w=obj_ang,
        contact_label=contact,
    )


# -----------------------------------------------------------------------------
# Main Closed-Loop MuJoCo Sim-to-Sim Runner
# -----------------------------------------------------------------------------

class MuJoCoSim2SimRunner:
    """50Hz Policy + 10Hz Diffusion Planner + 500Hz MuJoCo Physics Runner across all 6 SUGAR tasks."""

    def __init__(
        self,
        task_name: str,
        tracker_ckpt: str | Path,
        mode: str = "autonomous",
        generator_ckpt: str | Path | None = None,
        descriptions_dir: str | Path | None = None,
        motion_folder: str | Path | None = None,
        motion_index: int = 0,
        sim_dt: float = 0.002,
        control_dt: float = 0.02,
        generator_call_interval: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scene_xml_out: str | Path | None = None,
    ) -> None:
        if mode not in ("replay", "autonomous"):
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'replay' or 'autonomous'.")
        if mode == "autonomous" and generator_ckpt is None:
            raise ValueError("`--generator_ckpt` is required when running in `--mode autonomous`.")
        if mode == "replay" and motion_folder is None:
            raise ValueError("`--motion_folder` is required when running in `--mode replay`.")

        self.mode = mode
        self.device = torch.device(device)
        self.sim_dt = float(sim_dt)
        self.control_dt = float(control_dt)
        self.decimation = max(1, int(round(self.control_dt / self.sim_dt)))
        self.generator_call_interval = int(generator_call_interval)

        # 1. Build task-aligned MuJoCo scene XML & compile MjModel / MjData
        self.xml_path, self.spec, self.asset_report = build_mujoco_scene_xml(
            task_name=task_name,
            descriptions_dir=descriptions_dir,
            sim_dt=self.sim_dt,
            output_xml_path=scene_xml_out,
            motion_folder=motion_folder,
        )
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        # 2. Load Tracker policy & resolve exact 29-DoF joint bijection
        self.tracker = load_tracker_checkpoint(tracker_ckpt, device=self.device)
        self._setup_mujoco_indices()

        # 3. Optional dataset clip (used by `replay` and optional `autonomous` initial/target sampling)
        self.clip: MotionClipData | None = (
            load_single_motion_clip(motion_folder, motion_index=motion_index, device=self.device)
            if motion_folder is not None
            else None
        )

        # 4. Optional high-level Diffusion Generator (`sugar_il`)
        self.generator: Any = None
        self.downsample_rate: int = 5
        if self.mode == "autonomous":
            from sugar_il.wrapper.sugar_il_wrapper import DOWNSAMPLE_RATE, GeneratorObs, GeneratorWrapper

            self._GeneratorObs = GeneratorObs
            self.downsample_rate = int(DOWNSAMPLE_RATE)
            self.generator = GeneratorWrapper.load(str(generator_ckpt), device=str(self.device))
            n_obs = int(self.generator.n_obs_steps)
            n_act = int(self.generator.n_action_steps)
            obs_buf_len = (n_obs - 1) * self.downsample_rate + 1
            act_buf_len = (n_act - 1) * self.downsample_rate + 1
            if self.generator_call_interval > act_buf_len:
                self.generator_call_interval = act_buf_len

            self.gen_obs_buffer: dict[str, Tensor] = {
                "obj_pos_b": torch.zeros(1, obs_buf_len, 3, device=self.device),
                "obj_ori_b": torch.zeros(1, obs_buf_len, 4, device=self.device),
                "joint_pos": torch.zeros(1, obs_buf_len, 29, device=self.device),
                "project_gravity": torch.zeros(1, obs_buf_len, 3, device=self.device),
                "target_obj_pos_b": torch.zeros(1, obs_buf_len, 3, device=self.device),
                "target_obj_ori_b": torch.zeros(1, obs_buf_len, 4, device=self.device),
                "last_command": torch.zeros(1, obs_buf_len, 36, device=self.device),
            }
            self.gen_cmd_buffer = torch.zeros(1, act_buf_len, 36, device=self.device)
            self.gen_cmd_buffer_bfs = torch.zeros(1, act_buf_len, 36, device=self.device)

        # Runtime state buffers
        self.step_count: int = 0
        self.last_action = torch.zeros(1, 29, device=self.device)
        self.last_command = torch.zeros(1, 1, 36, device=self.device)
        self.current_cmd_36 = torch.zeros(1, 36, device=self.device)
        self.history: TrackerHistoryState | None = None
        self.target_pos_w = torch.tensor([self.spec.default_target_pos], dtype=torch.float32, device=self.device)
        self.target_quat_w = torch.tensor([self.spec.default_target_quat], dtype=torch.float32, device=self.device)

    def _setup_mujoco_indices(self) -> None:
        """Build exact index permutations between MuJoCo qpos/qvel/ctrl and Policy 29-DoF joint order."""
        self.pelvis_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.obj_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.target_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_vis")
        self.target_mocap_id = int(self.mj_model.body_mocapid[self.target_body_id])

        if min(self.pelvis_body_id, self.torso_body_id, self.obj_body_id) < 0:
            raise RuntimeError("MuJoCo model missing required bodies ('pelvis', 'torso_link', 'obj').")

        pelvis_jnt_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_freejoint")
        obj_jnt_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        self.pelvis_qpos_adr = int(self.mj_model.jnt_qposadr[pelvis_jnt_id])  # 0
        self.pelvis_dof_adr = int(self.mj_model.jnt_dofadr[pelvis_jnt_id])    # 0
        self.obj_qpos_adr = int(self.mj_model.jnt_qposadr[obj_jnt_id])        # 36
        self.obj_dof_adr = int(self.mj_model.jnt_dofadr[obj_jnt_id])          # 35

        # For each of the 29 policy joints, find its exact qpos, qvel, and actuator index in MuJoCo
        qpos_adrs, dof_adrs, ctrl_ids = [], [], []
        for j_name in self.tracker.joint_names:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, j_name)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"Joint or motor '{j_name}' not found in MuJoCo model!")
            qpos_adrs.append(int(self.mj_model.jnt_qposadr[jid]))
            dof_adrs.append(int(self.mj_model.jnt_dofadr[jid]))
            ctrl_ids.append(int(aid))

        self.pol_qpos_adrs = np.asarray(qpos_adrs, dtype=np.int64)
        self.pol_dof_adrs = np.asarray(dof_adrs, dtype=np.int64)
        self.pol_ctrl_ids = np.asarray(ctrl_ids, dtype=np.int64)

        # Permutation from IsaacLab BFS dataset joint order -> Policy joint order
        bfs_name_to_idx = {n: i for i, n in enumerate(G1_ISAACLAB_BFS_JOINT_NAMES)}
        self.bfs_to_pol_idx = torch.tensor(
            [bfs_name_to_idx[n] for n in self.tracker.joint_names],
            dtype=torch.long,
            device=self.device,
        )

        # Numpy actuator gains in Policy joint order & sync to MuJoCo `<position>` implicit PD actuators
        self.kp_np = self.tracker.stiffness.detach().cpu().numpy().astype(np.float64)
        self.kd_np = self.tracker.damping.detach().cpu().numpy().astype(np.float64)
        self.eff_np = self.tracker.effort_limit.detach().cpu().numpy().astype(np.float64)
        self.q_def_np = self.tracker.default_joint_pos.detach().cpu().numpy().astype(np.float64)
        self.scale_np = self.tracker.action_scale.detach().cpu().numpy().astype(np.float64)

        self._use_implicit_pos_actuators = bool(
            np.all(self.mj_model.actuator_biastype[self.pol_ctrl_ids] == int(mujoco.mjtBias.mjBIAS_AFFINE))
        )
        if self._use_implicit_pos_actuators:
            for idx_p, aid in enumerate(self.pol_ctrl_ids):
                kp, kd, eff = self.kp_np[idx_p], self.kd_np[idx_p], self.eff_np[idx_p]
                self.mj_model.actuator_gainprm[aid, 0] = kp
                self.mj_model.actuator_biasprm[aid, 0] = 0.0
                self.mj_model.actuator_biasprm[aid, 1] = -kp
                self.mj_model.actuator_biasprm[aid, 2] = -kd
                self.mj_model.actuator_forcerange[aid, 0] = -eff
                self.mj_model.actuator_forcerange[aid, 1] = eff

    # ------------------------------------------------------------------
    # State Reading & Coordinate Conversions
    # ------------------------------------------------------------------

    def sync_target_from_viewer_mocap(self) -> None:
        """Read back `target_vis` mocap pose so Ctrl+Mouse dragging in `mujoco.viewer` updates the goal live!"""
        if self.target_mocap_id >= 0:
            m_pos = self.mj_data.mocap_pos[self.target_mocap_id]
            m_quat = self.mj_data.mocap_quat[self.target_mocap_id]
            self.target_pos_w = torch.as_tensor(m_pos, dtype=torch.float32, device=self.device).view(1, 3)
            self.target_quat_w = quat_normalize(
                torch.as_tensor(m_quat, dtype=torch.float32, device=self.device).view(1, 4)
            )

    def _read_mujoco_tensors(self) -> dict[str, Tensor]:
        """Extract exact sensor tensors from `mj_data` (must be called after `mj_forward`)."""
        p_adr = self.pelvis_qpos_adr
        v_adr = self.pelvis_dof_adr
        o_adr = self.obj_qpos_adr

        root_pos_w = torch.as_tensor(self.mj_data.qpos[p_adr : p_adr + 3], dtype=torch.float32, device=self.device).view(1, 3)
        root_quat_w = quat_normalize(
            torch.as_tensor(self.mj_data.qpos[p_adr + 3 : p_adr + 7], dtype=torch.float32, device=self.device).view(1, 4)
        )
        # NOTE: In MuJoCo `<freejoint>`, `qvel[3:6]` is ALREADY in the local `pelvis` body frame!
        base_ang_vel_b = torch.as_tensor(
            self.mj_data.qvel[v_adr + 3 : v_adr + 6], dtype=torch.float32, device=self.device
        ).view(1, 3)

        q_pol = torch.as_tensor(self.mj_data.qpos[self.pol_qpos_adrs], dtype=torch.float32, device=self.device).view(1, 29)
        qd_pol = torch.as_tensor(self.mj_data.qvel[self.pol_dof_adrs], dtype=torch.float32, device=self.device).view(1, 29)

        anchor_pos_w = torch.as_tensor(self.mj_data.xpos[self.torso_body_id], dtype=torch.float32, device=self.device).view(1, 3)
        anchor_quat_w = quat_normalize(
            torch.as_tensor(self.mj_data.xquat[self.torso_body_id], dtype=torch.float32, device=self.device).view(1, 4)
        )

        obj_pos_w = torch.as_tensor(self.mj_data.qpos[o_adr : o_adr + 3], dtype=torch.float32, device=self.device).view(1, 3)
        obj_quat_w = quat_normalize(
            torch.as_tensor(self.mj_data.qpos[o_adr + 3 : o_adr + 7], dtype=torch.float32, device=self.device).view(1, 4)
        )

        grav_w = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)
        proj_grav = quat_apply_inverse(root_quat_w, grav_w)

        obj_pos_b, obj_quat_b = subtract_frame_transforms(anchor_pos_w, anchor_quat_w, obj_pos_w, obj_quat_w)
        obj_ori_6d = sugar_rot6d_from_quat(obj_quat_b)

        return {
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "base_ang_vel": base_ang_vel_b,
            "q": q_pol,
            "qd": qd_pol,
            "joint_pos_rel": q_pol - self.tracker.default_joint_pos.unsqueeze(0),
            "joint_vel_rel": qd_pol,
            "project_gravity": proj_grav,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "obj_pos_w": obj_pos_w,
            "obj_quat_w": obj_quat_w,
            "obj_pos_b": obj_pos_b,
            "obj_quat_b": obj_quat_b,
            "obj_ori_6d": obj_ori_6d,
        }

    # ------------------------------------------------------------------
    # Reset & Initialization
    # ------------------------------------------------------------------

    def reset(
        self,
        custom_obj_pos: tuple[float, float, float] | None = None,
        custom_obj_quat: tuple[float, float, float, float] | None = None,
        custom_target_pos: tuple[float, float, float] | None = None,
        custom_target_quat: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Tensor]:
        """Reset MuJoCo simulation, 5-step Tracker history, and (if autonomous) Generator buffers."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        p_adr = self.pelvis_qpos_adr
        v_adr = self.pelvis_dof_adr
        o_adr = self.obj_qpos_adr
        ov_adr = self.obj_dof_adr

        if self.clip is not None:
            # Initialize from frame 0 of the loaded SUGAR reference clip
            q0_pol = self.clip.joint_pos_bfs[0, self.bfs_to_pol_idx].detach().cpu().numpy()
            qd0_pol = self.clip.joint_vel_bfs[0, self.bfs_to_pol_idx].detach().cpu().numpy()
            r_pos0 = self.clip.root_pos_w[0].detach().cpu().numpy()
            r_quat0 = self.clip.root_quat_w[0].detach().cpu().numpy()
            r_lin0 = self.clip.root_lin_vel_w[0].detach().cpu().numpy()
            # Convert world angular velocity in clip -> local body frame for MuJoCo freejoint `qvel[3:6]`
            r_ang_b0 = quat_apply_inverse(
                self.clip.root_quat_w[0:1], self.clip.root_ang_vel_w[0:1]
            )[0].detach().cpu().numpy()

            self.mj_data.qpos[p_adr : p_adr + 3] = r_pos0
            self.mj_data.qpos[p_adr + 3 : p_adr + 7] = r_quat0
            self.mj_data.qpos[self.pol_qpos_adrs] = q0_pol
            self.mj_data.qvel[v_adr : v_adr + 3] = r_lin0
            self.mj_data.qvel[v_adr + 3 : v_adr + 6] = r_ang_b0
            self.mj_data.qvel[self.pol_dof_adrs] = qd0_pol

            o_pos0 = self.clip.obj_pos_w[0].detach().cpu().numpy()
            o_quat0 = self.clip.obj_quat_w[0].detach().cpu().numpy()
            o_lin0 = self.clip.obj_lin_vel_w[0].detach().cpu().numpy()
            o_ang_b0 = quat_apply_inverse(
                self.clip.obj_quat_w[0:1], self.clip.obj_ang_vel_w[0:1]
            )[0].detach().cpu().numpy()
            self.mj_data.qpos[o_adr : o_adr + 3] = o_pos0
            self.mj_data.qpos[o_adr + 3 : o_adr + 7] = o_quat0
            self.mj_data.qvel[ov_adr : ov_adr + 3] = o_lin0
            self.mj_data.qvel[ov_adr + 3 : ov_adr + 6] = o_ang_b0

            t_pos = self.clip.obj_pos_w[-1].detach().cpu().numpy()
            t_quat = self.clip.obj_quat_w[-1].detach().cpu().numpy()
        else:
            # Nominal standing init matching `sugar_rl/assets/robots/unitree.py` (`pos=(0, 0, 0.76)`)
            self.mj_data.qpos[p_adr : p_adr + 3] = np.array([0.0, 0.0, 0.76], dtype=np.float64)
            self.mj_data.qpos[p_adr + 3 : p_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.mj_data.qpos[self.pol_qpos_adrs] = self.q_def_np
            self.mj_data.qpos[o_adr : o_adr + 3] = np.asarray(self.spec.default_obj_pos, dtype=np.float64)
            self.mj_data.qpos[o_adr + 3 : o_adr + 7] = np.asarray(self.spec.default_obj_quat, dtype=np.float64)
            t_pos = np.asarray(self.spec.default_target_pos, dtype=np.float64)
            t_quat = np.asarray(self.spec.default_target_quat, dtype=np.float64)

        # Apply explicit user overrides if provided
        if custom_obj_pos is not None:
            self.mj_data.qpos[o_adr : o_adr + 3] = np.asarray(custom_obj_pos, dtype=np.float64)
            self.mj_data.qvel[ov_adr : ov_adr + 6] = 0.0
        if custom_obj_quat is not None:
            self.mj_data.qpos[o_adr + 3 : o_adr + 7] = np.asarray(custom_obj_quat, dtype=np.float64)
        if custom_target_pos is not None:
            t_pos = np.asarray(custom_target_pos, dtype=np.float64)
        if custom_target_quat is not None:
            t_quat = np.asarray(custom_target_quat, dtype=np.float64)

        self.target_pos_w = torch.as_tensor(t_pos, dtype=torch.float32, device=self.device).view(1, 3)
        self.target_quat_w = quat_normalize(
            torch.as_tensor(t_quat, dtype=torch.float32, device=self.device).view(1, 4)
        )
        if self.target_mocap_id >= 0:
            self.mj_data.mocap_pos[self.target_mocap_id] = self.target_pos_w[0].detach().cpu().numpy()
            self.mj_data.mocap_quat[self.target_mocap_id] = self.target_quat_w[0].detach().cpu().numpy()

        if self._use_implicit_pos_actuators:
            self.mj_data.ctrl[self.pol_ctrl_ids] = self.mj_data.qpos[self.pol_qpos_adrs]

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.step_count = 0
        self.last_action.zero_()
        self.last_command.zero_()

        st = self._read_mujoco_tensors()
        self.history = TrackerHistoryState(
            base_ang_vel=st["base_ang_vel"].unsqueeze(1).repeat(1, HISTORY_LEN, 1),
            joint_pos_rel=st["joint_pos_rel"].unsqueeze(1).repeat(1, HISTORY_LEN, 1),
            joint_vel_rel=st["joint_vel_rel"].unsqueeze(1).repeat(1, HISTORY_LEN, 1),
            actions=torch.zeros(1, HISTORY_LEN, 29, device=self.device),
            project_gravity=st["project_gravity"].unsqueeze(1).repeat(1, HISTORY_LEN, 1),
        )

        if self.mode == "autonomous":
            self._fill_generator_obs_buffer(st)
            self._run_generator_inference()
            self.current_cmd_36 = self.gen_cmd_buffer[:, 0, :].clone()
            self.last_command = self.gen_cmd_buffer_bfs[:, 0:1, :].clone()
        else:
            self.current_cmd_36 = self._get_replay_command(0)

        return st

    # ------------------------------------------------------------------
    # Command Generation (`replay` vs `autonomous` Diffusion Planner)
    # ------------------------------------------------------------------

    def _get_replay_command(self, t_idx: int) -> Tensor:
        """Construct 36-D `[ref_joint_pos(29), ref_root_lin_vel_b(3), ref_root_ang_vel_b(3), contact(1)]`."""
        assert self.clip is not None
        t = min(max(0, int(t_idx)), self.clip.length - 1)
        q_ref = self.clip.joint_pos_bfs[t : t + 1, self.bfs_to_pol_idx]
        rq = self.clip.root_quat_w[t : t + 1]
        v_b = quat_apply_inverse(rq, self.clip.root_lin_vel_w[t : t + 1])
        w_b = quat_apply_inverse(rq, self.clip.root_ang_vel_w[t : t + 1])
        c = self.clip.contact_label[t : t + 1]
        return torch.cat([q_ref, v_b, w_b, c], dim=-1)

    def _single_generator_obs_frame(self, st: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute current single-frame inputs matching `MotionCommand._get_generator_obs`."""
        t_pos_b, t_ori_b = subtract_frame_transforms(
            st["anchor_pos_w"], st["anchor_quat_w"], self.target_pos_w, self.target_quat_w
        )
        # Generator always expects 29-DoF joints in IsaacLab BFS order (`robot_joint_pos`)
        pol_to_bfs = torch.argsort(self.bfs_to_pol_idx)
        q_bfs = st["q"][:, pol_to_bfs]
        return {
            "obj_pos_b": st["obj_pos_b"],
            "obj_ori_b": st["obj_quat_b"],
            "joint_pos": q_bfs,
            "project_gravity": st["project_gravity"],
            "target_obj_pos_b": t_pos_b,
            "target_obj_ori_b": t_ori_b,
            "last_command": self.last_command.squeeze(1),
        }

    def _fill_generator_obs_buffer(self, st: dict[str, Tensor]) -> None:
        cur = self._single_generator_obs_frame(st)
        L = next(iter(self.gen_obs_buffer.values())).shape[1]
        for k, val in cur.items():
            self.gen_obs_buffer[k] = val.unsqueeze(1).repeat(1, L, 1).clone()

    def _update_generator_obs_buffer(self, st: dict[str, Tensor]) -> None:
        cur = self._single_generator_obs_frame(st)
        for k, val in cur.items():
            if self.gen_obs_buffer[k].shape[1] > 1:
                self.gen_obs_buffer[k][:, :-1] = self.gen_obs_buffer[k][:, 1:].clone()
            self.gen_obs_buffer[k][:, -1] = val

    def _run_generator_inference(self) -> None:
        gen_obs = self._GeneratorObs(
            obj_pos_b=self.gen_obs_buffer["obj_pos_b"],
            obj_ori_b=self.gen_obs_buffer["obj_ori_b"],
            joint_pos=self.gen_obs_buffer["joint_pos"],
            project_gravity=self.gen_obs_buffer["project_gravity"],
            target_obj_pos_b=self.gen_obs_buffer["target_obj_pos_b"],
            target_obj_ori_b=self.gen_obs_buffer["target_obj_ori_b"],
            last_command=self.gen_obs_buffer["last_command"],
        )
        pred_bfs = self.generator.predict(gen_obs)  # (1, L_act, 36) where [:29] is in IsaacLab BFS order
        self.gen_cmd_buffer_bfs = pred_bfs
        # Permute first 29 joint command dims into `self.tracker.joint_names` order if needed
        q_cmd_pol = pred_bfs[..., :29][:, :, self.bfs_to_pol_idx]
        self.gen_cmd_buffer = torch.cat([q_cmd_pol, pred_bfs[..., 29:]], dim=-1)

    # ------------------------------------------------------------------
    # 50Hz Control Step + 500Hz Inner MuJoCo Physics Substeps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self) -> dict[str, float]:
        """Execute one 50Hz policy control step (`decimation` MuJoCo physics substeps)."""
        assert self.history is not None
        self.sync_target_from_viewer_mocap()
        st = self._read_mujoco_tensors()

        # 1. Assemble 510-D `TrackerCfg` observation vector
        obs_510 = torch.cat(
            [
                self.current_cmd_36,
                self.history.base_ang_vel.reshape(1, -1),
                self.history.joint_pos_rel.reshape(1, -1),
                self.history.joint_vel_rel.reshape(1, -1),
                self.history.actions.reshape(1, -1),
                self.history.project_gravity.reshape(1, -1),
                st["obj_pos_b"],
                st["obj_ori_6d"],
            ],
            dim=-1,
        )

        # 2. Tracker forward pass -> raw action `a` (1, 29) -> PD target `q_target`
        raw_action = self.tracker.policy.act_inference(obs_510)
        raw_act_np = raw_action[0].detach().cpu().numpy().astype(np.float64)
        q_target_np = self.q_def_np + self.scale_np * raw_act_np

        # 3. Step MuJoCo physics `decimation` times with 500Hz implicit PD (`ImplicitActuatorCfg` isomorphic)
        ov_adr = self.obj_dof_adr
        ang_damp = float(self.spec.angular_damping)
        if self._use_implicit_pos_actuators:
            self.mj_data.ctrl[self.pol_ctrl_ids] = q_target_np
        for _ in range(self.decimation):
            if not self._use_implicit_pos_actuators:
                q_cur = self.mj_data.qpos[self.pol_qpos_adrs]
                qd_cur = self.mj_data.qvel[self.pol_dof_adrs]
                tau = np.clip(
                    self.kp_np * (q_target_np - q_cur) - self.kd_np * qd_cur,
                    -self.eff_np,
                    self.eff_np,
                )
                self.mj_data.ctrl[self.pol_ctrl_ids] = tau
            if ang_damp > 0.0:
                self.mj_data.qfrc_applied[ov_adr + 3 : ov_adr + 6] = (
                    -ang_damp * self.mj_data.qvel[ov_adr + 3 : ov_adr + 6]
                )
            mujoco.mj_step(self.mj_model, self.mj_data)

        # 4. Advance step counter, rolling 5-step proprioception history & next reference command
        self.step_count += 1
        self.last_action = raw_action.clone()
        st_next = self._read_mujoco_tensors()

        self.history = TrackerHistoryState(
            base_ang_vel=torch.cat([self.history.base_ang_vel[:, 1:], st_next["base_ang_vel"].unsqueeze(1)], dim=1),
            joint_pos_rel=torch.cat([self.history.joint_pos_rel[:, 1:], st_next["joint_pos_rel"].unsqueeze(1)], dim=1),
            joint_vel_rel=torch.cat([self.history.joint_vel_rel[:, 1:], st_next["joint_vel_rel"].unsqueeze(1)], dim=1),
            actions=torch.cat([self.history.actions[:, 1:], self.last_action.unsqueeze(1)], dim=1),
            project_gravity=torch.cat([self.history.project_gravity[:, 1:], st_next["project_gravity"].unsqueeze(1)], dim=1),
        )

        if self.mode == "autonomous":
            self._update_generator_obs_buffer(st_next)
            if self.step_count % self.generator_call_interval == 0:
                self._run_generator_inference()
            cmd_idx = self.step_count % self.generator_call_interval
            self.current_cmd_36 = self.gen_cmd_buffer[:, cmd_idx, :]
            self.last_command = self.gen_cmd_buffer_bfs[:, cmd_idx : cmd_idx + 1, :]
        else:
            self.current_cmd_36 = self._get_replay_command(self.step_count)

        goal_dist = float((st_next["obj_pos_w"] - self.target_pos_w).norm(dim=-1).item())
        return {
            "step": float(self.step_count),
            "sim_time_s": float(self.mj_data.time),
            "pelvis_z": float(st_next["root_pos_w"][0, 2].item()),
            "torso_z": float(st_next["anchor_pos_w"][0, 2].item()),
            "obj_z": float(st_next["obj_pos_w"][0, 2].item()),
            "goal_dist_m": goal_dist,
            "contact_cmd": float(self.current_cmd_36[0, 35].item()),
        }

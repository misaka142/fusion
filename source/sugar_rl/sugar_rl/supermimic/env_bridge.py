"""Bridge between IsaacLab `ManagerBasedRLEnv` (SUGAR tasks) and SuperMimic differentiable HOI engine.

Responsibilities:
1. Extract 84-D env-origin-normalized human-object state `(s_robot, s_obj)` from IsaacLab scene.
2. Extract 37-D physical domain randomization vector `xi` per environment at startup
   (foot collision-shape friction, object friction/mass scale, joint default offsets, zero-mean torso COM offset).
3. Extract 5-D ground-truth contact vector `[hoi_obj, feet_obj, lf_gnd, rf_gnd, undesired_body_fraction]`
   and 2-D `feet_air_time` from IsaacLab `ContactSensor` instances.
4. Intercept `interval` velocity-push perturbations (`push_robot`, `push_object`, `push_robot_sit`)
   with exact `contact_label` filtering for `on_contact` pushes.
5. Pre-cache `MotionLoader` tensors once on GPU to avoid repeated 35->14 body advanced-indexing copies,
   and run a strict per-slice startup alignment assertion on `PrivilegedCfg` (& `TeacherCfg`).
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import Tensor

from .hoi_task import (
    CONTACT_DIM,
    FUTURE_FRAMES,
    PRIVILEGED_OBS_DIM,
    PRIVILEGED_OBS_SLICES,
    RefFrame,
    RefFutureWindow,
    SugarHOITask,
    TRACKER_OBS_DIM,
    XI_DIM,
    resolve_task_profile,
)
from .urdf_kinematics import FastTreeKinematics


def assert_privileged_obs_aligned(
    obs_analytic: Tensor,
    obs_sim: Tensor,
    atol: float = 5e-3,
    label: str = "critic",
) -> tuple[float, str]:
    """Compare analytical vs simulator 890-D privileged observations across all 18 terms; raise on mismatch."""
    if obs_analytic.shape != obs_sim.shape:
        raise AssertionError(
            f"[{label}] Privileged obs shape mismatch: analytic={tuple(obs_analytic.shape)} vs sim={tuple(obs_sim.shape)}"
        )
    if not torch.isfinite(obs_analytic).all() or not torch.isfinite(obs_sim).all():
        raise AssertionError(f"[{label}] Non-finite values detected during startup observation alignment check!")

    diff = (obs_analytic - obs_sim).abs()
    worst_err = -1.0
    worst_name = "none"
    failing: list[str] = []

    for name, start, end in PRIVILEGED_OBS_SLICES:
        seg_err = float(diff[:, start:end].max().item())
        if seg_err > worst_err:
            worst_err = seg_err
            worst_name = f"{name}[{start}:{end}]"
        if seg_err > atol:
            failing.append(f"{name}[{start}:{end}]={seg_err:.6f}")

    if failing:
        raise AssertionError(
            f"[{label}] Privileged observation alignment failed (atol={atol:.1e}, worst={worst_name} @ {worst_err:.6f}). "
            f"Failing slices: {', '.join(failing)}"
        )
    return worst_err, worst_name


class MotionReferenceAdapter:
    """Fast batched slicer over SUGAR's `MotionLoader` tensors with one-time GPU tensor caching."""

    def __init__(
        self,
        motion_loader: Any,
        anchor_body_idx: int = 7,
        future_frames: int = FUTURE_FRAMES,
        anchor_only: bool = False,
    ) -> None:
        self.motion = motion_loader
        self.anchor_body_idx = anchor_body_idx
        self.future_frames = future_frames
        self.anchor_only = anchor_only

        self.lengths: Tensor = motion_loader.time_step_total_permotion.long()
        self.num_motions: int = int(motion_loader.num_motion)
        self.device = self.lengths.device

        # Cache direct tensor references once so `get_frame` / `get_future_window` never trigger
        # `MotionLoader.body_*_w` property copies (`_body_pos_w[:, :, _body_indexes]`) every step!
        self.joint_pos: Tensor = motion_loader.joint_pos
        self.joint_vel: Tensor = motion_loader.joint_vel
        self.obj_pos: Tensor = motion_loader.obj_pos
        self.obj_quat: Tensor = motion_loader.obj_quat
        self.obj_lin_vel: Tensor = motion_loader.obj_lin_vel
        self.obj_ang_vel: Tensor = motion_loader.obj_ang_vel
        self.contact_label: Tensor = motion_loader.contact_label

        if anchor_only and hasattr(motion_loader, "_body_pos_w") and hasattr(motion_loader, "_body_indexes"):
            raw_anc_idx = int(motion_loader._body_indexes[anchor_body_idx])
            self.body_pos_w: Tensor | None = None
            self.body_quat_w: Tensor | None = None
            self.body_lin_vel_w: Tensor | None = None
            self.body_ang_vel_w: Tensor | None = None
            self.anchor_pos_w: Tensor = motion_loader._body_pos_w[:, :, raw_anc_idx].contiguous()
            self.anchor_quat_w: Tensor = motion_loader._body_quat_w[:, :, raw_anc_idx].contiguous()
        else:
            self.body_pos_w = motion_loader.body_pos_w.contiguous()
            self.body_quat_w = motion_loader.body_quat_w.contiguous()
            self.body_lin_vel_w = motion_loader.body_lin_vel_w.contiguous()
            self.body_ang_vel_w = motion_loader.body_ang_vel_w.contiguous()
            self.anchor_pos_w = self.body_pos_w[:, :, anchor_body_idx]
            self.anchor_quat_w = self.body_quat_w[:, :, anchor_body_idx]

    def clamp_timesteps(self, motion_ids: Tensor, time_steps: Tensor) -> Tensor:
        max_t = (self.lengths[motion_ids] - 1).clamp_min(0)
        return time_steps.clamp(min=0).minimum(max_t)

    def is_done(self, motion_ids: Tensor, time_steps: Tensor) -> Tensor:
        return time_steps >= self.lengths[motion_ids]

    def get_frame(self, motion_ids: Tensor, time_steps: Tensor) -> RefFrame:
        if self.body_pos_w is None or self.body_quat_w is None or self.body_lin_vel_w is None or self.body_ang_vel_w is None:
            raise RuntimeError("`get_frame` called on an `anchor_only=True` MotionReferenceAdapter.")
        t = self.clamp_timesteps(motion_ids, time_steps)
        return RefFrame(
            joint_pos=self.joint_pos[motion_ids, t],
            joint_vel=self.joint_vel[motion_ids, t],
            body_pos_w=self.body_pos_w[motion_ids, t],
            body_quat_w=self.body_quat_w[motion_ids, t],
            body_lin_vel_w=self.body_lin_vel_w[motion_ids, t],
            body_ang_vel_w=self.body_ang_vel_w[motion_ids, t],
            obj_pos_w=self.obj_pos[motion_ids, t],
            obj_quat_w=self.obj_quat[motion_ids, t],
            obj_lin_vel_w=self.obj_lin_vel[motion_ids, t],
            obj_ang_vel_w=self.obj_ang_vel[motion_ids, t],
            contact_label=self.contact_label[motion_ids, t].float(),
        )

    def get_future_window(self, motion_ids: Tensor, time_steps: Tensor) -> RefFutureWindow:
        K = self.future_frames
        offsets = torch.arange(K, device=self.device, dtype=torch.long).unsqueeze(0)  # (1, K)
        max_t = (self.lengths[motion_ids] - 1).clamp_min(0).unsqueeze(1)              # (B, 1)
        fut_t = (time_steps.unsqueeze(1) + offsets).clamp(min=0).minimum(max_t)       # (B, K)
        m_ids = motion_ids.unsqueeze(1).expand(-1, K)                                 # (B, K)

        return RefFutureWindow(
            joint_pos=self.joint_pos[m_ids, fut_t],
            joint_vel=self.joint_vel[m_ids, fut_t],
            anchor_pos_w=self.anchor_pos_w[m_ids, fut_t],
            anchor_quat_w=self.anchor_quat_w[m_ids, fut_t],
            obj_pos_w=self.obj_pos[m_ids, fut_t],
            obj_quat_w=self.obj_quat[m_ids, fut_t],
            obj_lin_vel_w=self.obj_lin_vel[m_ids, fut_t],
            obj_ang_vel_w=self.obj_ang_vel[m_ids, fut_t],
        )


class SugarEnvBridge:
    """Adapter around IsaacLab `ManagerBasedRLEnv` for SuperMimic training & verification."""

    def __init__(
        self,
        env: Any,
        task_name_or_id: str,
        urdf_path: str | None = None,
        verify_obs_on_init: bool = True,
        obs_verify_atol: float = 5e-3,
    ) -> None:
        # Unwrap gym wrappers (e.g. RecordVideo) to reach ManagerBasedRLEnv
        self.wrapped_env = env
        self.env = env.unwrapped
        self.device: torch.device = torch.device(self.env.device)
        self.num_envs: int = int(self.env.num_envs)
        self.step_dt: float = float(self.env.step_dt)

        self.robot = self.env.scene["robot"]
        self.obj = self.env.scene["obj"]
        self.motion_cmd = self.env.command_manager.get_term("motion")

        # Build differentiable kinematics & sync nominal joint defaults, COM offsets, and PD gains
        self.fk = FastTreeKinematics(
            urdf_or_xml_path=urdf_path,
            env_joint_names=list(self.robot.joint_names),
            device=self.device,
        )
        live_scale = None
        try:
            jpos_term = self.env.action_manager.get_term("joint_pos")
            if isinstance(getattr(jpos_term, "_scale", None), Tensor):
                live_scale = jpos_term._scale[0] if jpos_term._scale.ndim == 2 else jpos_term._scale
        except Exception:
            live_scale = None
        self.fk.sync_from_isaaclab_robot(self.robot, action_scale=live_scale)
        self.task = SugarHOITask(self.fk, task_name_or_id=task_name_or_id, step_dt=self.step_dt)
        self.profile = self.task.profile

        # Reference motion adapters (Teacher adapter only needs anchor poses -> 14x less VRAM)
        self.motion_ref = MotionReferenceAdapter(
            self.motion_cmd.motion, anchor_body_idx=self.fk.anchor_tracked_idx, anchor_only=False
        )
        self.teacher_motion_ref = (
            MotionReferenceAdapter(
                self.motion_cmd.teacher_motion, anchor_body_idx=self.fk.anchor_tracked_idx, anchor_only=True
            )
            if getattr(self.motion_cmd, "teacher_motion", None) is not None
            else None
        )

        # Track interval velocity pushes per step
        self._pushed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._install_push_hooks()

        # Cache sensor body indices & sync live reward parameters (undesired body count / thresholds)
        self._setup_contact_indices()
        self._sync_live_reward_params()

        # Cache static startup domain randomization vector `xi`
        self.env_xi: Tensor = self.extract_xi()

        # Reset env once to initialize buffers cleanly
        obs_dict, _ = self.wrapped_env.reset()
        self.current_obs_dict: dict[str, Tensor] = obs_dict

        if verify_obs_on_init:
            self._verify_privileged_obs_alignment(atol=obs_verify_atol)

    def _install_push_hooks(self) -> None:
        """Wrap interval push event functions so `_pushed_mask` records exact pushed env_ids."""
        ev_mgr = getattr(self.env, "event_manager", None)
        if ev_mgr is None:
            return
        interval_cfgs = getattr(ev_mgr, "_mode_term_cfgs", {}).get("interval", [])
        interval_names = getattr(ev_mgr, "_mode_term_names", {}).get("interval", [])
        for name, term_cfg in zip(interval_names, interval_cfgs):
            if "push" in str(name).lower() and callable(getattr(term_cfg, "func", None)):
                orig_fn = term_cfg.func
                is_on_contact = "on_contact" in getattr(orig_fn, "__name__", "")

                @functools.wraps(orig_fn)
                def _hooked_fn(env_arg, env_ids, *args, _orig=orig_fn, _on_c=is_on_contact, **kwargs):
                    if env_ids is not None and len(env_ids) > 0:
                        if _on_c:
                            c_mask = self.motion_cmd.contact_label[env_ids].bool()
                            if c_mask.any():
                                self._pushed_mask[env_ids[c_mask]] = True
                        else:
                            self._pushed_mask[env_ids] = True
                    return _orig(env_arg, env_ids, *args, **kwargs)

                term_cfg.func = _hooked_fn

    def _setup_contact_indices(self) -> None:
        sensors = self.env.scene.sensors
        cf_sensor = sensors["contact_forces"]
        cf_names = list(cf_sensor.body_names)

        self._lf_idx = cf_names.index("left_ankle_roll_link")
        self._rf_idx = cf_names.index("right_ankle_roll_link")

        allowed = {"left_ankle_roll_link", "right_ankle_roll_link"}
        if self.profile.hoi_mode == "hands":
            allowed.update({"left_rubber_hand", "right_rubber_hand"})
            if self.profile.task_name == "StandBottle":
                allowed.update({"left_knee_link", "right_knee_link"})
        elif self.profile.hoi_mode == "sit":
            for n in cf_names:
                if "hip" in n or "pelvis" in n:
                    allowed.add(n)

        undesired_ids = [i for i, n in enumerate(cf_names) if n not in allowed]
        self._undesired_body_ids = torch.tensor(undesired_ids, dtype=torch.long, device=self.device)
        self._undesired_thresh = self.profile.undesired_threshold
        self.profile.num_undesired_bodies = max(1, len(undesired_ids))

    def _sync_live_reward_params(self) -> None:
        """Synchronize exact reward weights/thresholds from live `env.reward_manager` when present."""
        rew_mgr = getattr(self.env, "reward_manager", None)
        if rew_mgr is None:
            return
        term_names = getattr(rew_mgr, "_term_names", [])
        term_cfgs = getattr(rew_mgr, "_term_cfgs", [])
        cfg_map = dict(zip(term_names, term_cfgs))

        if "undesired_contacts" in cfg_map:
            uc = cfg_map["undesired_contacts"]
            self.profile.undesired_weight = float(uc.weight)
            if isinstance(getattr(uc, "params", None), dict) and "threshold" in uc.params:
                self._undesired_thresh = float(uc.params["threshold"])
                self.profile.undesired_threshold = self._undesired_thresh
        if "feet_slide" in cfg_map:
            self.profile.feet_slide_weight = float(cfg_map["feet_slide"].weight)
        elif term_names and "feet_slide" not in cfg_map:
            self.profile.feet_slide_weight = 0.0
        if "feet_air_time" in cfg_map:
            fa = cfg_map["feet_air_time"]
            self.profile.feet_air_time_weight = float(fa.weight)
            if isinstance(getattr(fa, "params", None), dict) and "threshold" in fa.params:
                self.profile.feet_air_time_threshold = float(fa.params["threshold"])
        elif term_names and "feet_air_time" not in cfg_map:
            self.profile.feet_air_time_weight = 0.0

    def _resolve_foot_shape_indices(self, total_shapes: int) -> list[int]:
        """Locate PhysX collision shape indices belonging to left/right `ankle_roll_link`."""
        shapes_per_body = getattr(self.robot, "_num_shapes_per_body", None)
        if shapes_per_body is not None and sum(shapes_per_body) == total_shapes:
            foot_indices: list[int] = []
            cursor = 0
            for b_name, count in zip(self.robot.body_names, shapes_per_body):
                if "ankle_roll" in b_name and count > 0:
                    foot_indices.extend(range(cursor, cursor + count))
                cursor += count
            if foot_indices:
                return foot_indices
        return list(range(total_shapes))

    @torch.no_grad()
    def extract_xi(self) -> Tensor:
        """Extract 37-D domain randomization parameter vector `xi` for all environments."""
        if hasattr(self, "env_xi"):
            return self.env_xi.clone()

        xi = torch.zeros((self.num_envs, XI_DIM), dtype=torch.float32, device=self.device)
        xi[:, 0] = 1.0  # default robot static friction
        xi[:, 1] = 1.0  # default robot dynamic friction
        xi[:, 2] = 0.5  # default obj static friction
        xi[:, 3] = 0.5  # default obj dynamic friction
        xi[:, 4] = 1.0  # default obj mass scale

        try:
            rob_mat = self.robot.root_physx_view.get_material_properties().to(self.device)
            foot_shapes = self._resolve_foot_shape_indices(int(rob_mat.shape[1]))
            xi[:, 0] = rob_mat[:, foot_shapes, 0].mean(dim=1)
            xi[:, 1] = rob_mat[:, foot_shapes, 1].mean(dim=1)
        except Exception:
            pass

        try:
            obj_mat = self.obj.root_physx_view.get_material_properties().to(self.device)
            xi[:, 2] = obj_mat[:, :, 0].mean(dim=1)
            xi[:, 3] = obj_mat[:, :, 1].mean(dim=1)
        except Exception:
            pass

        try:
            masses = self.obj.root_physx_view.get_masses().to(self.device).squeeze(-1)
            def_mass = self.obj.data.default_mass.to(self.device).squeeze(-1).clamp_min(1e-4)
            xi[:, 4] = masses / def_mass
        except Exception:
            pass

        nominal_jpos = getattr(self.robot.data, "default_joint_pos_nominal", None)
        if nominal_jpos is not None:
            nom = nominal_jpos.to(self.device).reshape(1, -1)
            xi[:, 5:34] = self.robot.data.default_joint_pos - nom

        try:
            torso_idx = self.robot.body_names.index("torso_link")
            coms = self.robot.root_physx_view.get_coms()[:, torso_idx, :3].to(self.device)
            if hasattr(self.robot.data, "default_com"):
                coms = coms - self.robot.data.default_com[:, torso_idx, :3].to(self.device)
            elif hasattr(self.robot.data, "default_com_pos"):
                coms = coms - self.robot.data.default_com_pos[:, torso_idx, :3].to(self.device)
            elif self.num_envs >= 2:
                # `randomize_rigid_body_com` samples zero-mean symmetric uniform offsets across envs
                coms = coms - coms.mean(dim=0, keepdim=True)
            else:
                coms = torch.zeros_like(coms)
            xi[:, 34:37] = coms
        except Exception:
            pass

        return xi

    @torch.no_grad()
    def get_joint_state(self) -> Tensor:
        """Read current 84-D env-origin-normalized human-object state."""
        origins = self.env.scene.env_origins
        r_data = self.robot.data
        o_data = self.obj.data
        return SugarHOITask.pack_state(
            root_pos=r_data.root_pos_w - origins,
            root_quat=r_data.root_quat_w,
            q=r_data.joint_pos,
            root_lin_vel=r_data.root_lin_vel_w,
            root_ang_vel=r_data.root_ang_vel_w,
            qd=r_data.joint_vel,
            obj_pos=o_data.root_pos_w - origins,
            obj_quat=o_data.root_quat_w,
            obj_lin_vel=o_data.root_lin_vel_w,
            obj_ang_vel=o_data.root_ang_vel_w,
        )

    @torch.no_grad()
    def get_motion_cursor(self) -> tuple[Tensor, Tensor]:
        return self.motion_cmd.motion_id.clone(), self.motion_cmd.time_steps.clone()

    @torch.no_grad()
    def get_last_action(self) -> Tensor:
        return self.env.action_manager.action.clone()

    @torch.no_grad()
    def get_feet_air_time(self) -> Tensor:
        """Read current air time `(num_envs, 2)` for `[left_ankle_roll_link, right_ankle_roll_link]`."""
        cf_data = self.env.scene.sensors["contact_forces"].data
        air_t = getattr(cf_data, "current_air_time", None)
        if air_t is None:
            return torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        return air_t[:, [self._lf_idx, self._rf_idx]].float()

    @torch.no_grad()
    def get_ground_truth_contacts(self) -> Tensor:
        """Read 5-D contact targets `[hoi_obj, feet_obj, lf_gnd, rf_gnd, undesired_fraction]` in [0, 1]."""
        sensors = self.env.scene.sensors
        out = torch.zeros((self.num_envs, CONTACT_DIM), dtype=torch.float32, device=self.device)

        def _obj_force_active(sensor_name: str, thresh: float = 0.1) -> Tensor:
            f_hist = sensors[sensor_name].data.force_matrix_w_history  # [N, H, 1, 1, 3]
            return f_hist[:, :, 0, 0, :].norm(dim=-1).max(dim=1)[0] > thresh

        lh = _obj_force_active("left_hand_forces")
        rh = _obj_force_active("right_hand_forces")
        lf_obj = _obj_force_active("left_foot_forces")
        rf_obj = _obj_force_active("right_foot_forces")

        if self.profile.hoi_mode == "hands":
            hoi_c = lh & rh
        elif self.profile.hoi_mode == "feet":
            hoi_c = lf_obj | rf_obj
        else:
            lhip = _obj_force_active("left_hip_forces")
            rhip = _obj_force_active("right_hip_forces")
            pelv = _obj_force_active("pelvis_forces")
            hoi_c = lhip | rhip | pelv

        cf_net = sensors["contact_forces"].data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]  # [N, B]
        lf_gnd = cf_net[:, self._lf_idx] > 1.0
        rf_gnd = cf_net[:, self._rf_idx] > 1.0
        if self._undesired_body_ids.numel() > 0:
            undes_count = (cf_net[:, self._undesired_body_ids] > self._undesired_thresh).float().sum(dim=-1)
            undes_frac = undes_count / float(self._undesired_body_ids.numel())
        else:
            undes_frac = torch.zeros_like(lf_gnd, dtype=torch.float32)

        out[:, 0] = hoi_c.float()
        out[:, 1] = (lf_obj | rf_obj).float()
        out[:, 2] = lf_gnd.float()
        out[:, 3] = rf_gnd.float()
        out[:, 4] = undes_frac.clamp(0.0, 1.0)
        return out

    @torch.no_grad()
    def _verify_privileged_obs_alignment(self, atol: float = 5e-3) -> None:
        """Step physics to flush derived link poses, assert per-slice alignment, then reset cleanly."""
        if "critic" not in self.current_obs_dict:
            return

        # Step unwrapped env twice with reference-tracking PD action so `body_pos_w` is freshly synced
        obs_dict = self.current_obs_dict
        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(2):
            q_ref_now = self.motion_cmd.joint_pos
            q_def_env = self.task.effective_default_joint_pos(self.env_xi)
            act_hold = (q_ref_now - q_def_env) / self.fk.action_scale.clamp_min(1e-6)
            obs_dict, _, terminated, truncated, _ = self.env.step(act_hold)
            valid = ~(terminated | truncated)

        if not valid.any():
            valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        s_now = self.get_joint_state()[valid]
        m_ids, t_steps = self.get_motion_cursor()
        m_ids_v = m_ids[valid]
        t_steps_v = t_steps[valid]
        last_act_v = self.get_last_action()[valid]
        xi_v = self.env_xi[valid]
        fk_v = self.task.run_fk(s_now)

        ref_fut = self.motion_ref.get_future_window(m_ids_v, t_steps_v)
        obs_analytic = self.task.build_privileged_obs(
            s_now, last_act_v, ref_fut, fk_res=fk_v, xi=xi_v
        )
        obs_sim = obs_dict["critic"][valid]
        max_diff, worst_term = assert_privileged_obs_aligned(
            obs_analytic, obs_sim, atol=atol, label="critic"
        )

        # In Stage 2 Tracker mode, also assert `TeacherCfg` (890-D) & noise-free action history slot!
        if "teacher" in obs_dict and self.teacher_motion_ref is not None:
            t_fut = self.teacher_motion_ref.get_future_window(m_ids_v, t_steps_v)
            obs_t_analytic = self.task.build_privileged_obs(
                s_now, last_act_v, t_fut, fk_res=fk_v, xi=xi_v
            )
            t_diff, t_worst = assert_privileged_obs_aligned(
                obs_t_analytic, obs_dict["teacher"][valid], atol=atol, label="teacher"
            )
            if t_diff > max_diff:
                max_diff, worst_term = t_diff, f"teacher:{t_worst}"

        if "policy" in obs_dict and int(obs_dict["policy"].shape[-1]) == TRACKER_OBS_DIM:
            # Slice 457:486 is the newest slot of `actions_history` (uncorrupted by sensor noise)
            act_hist_err = float((obs_dict["policy"][valid, 457:486] - last_act_v).abs().max().item())
            if act_hist_err > atol:
                raise AssertionError(
                    f"[tracker] Latest action history slice [457:486] mismatch: {act_hist_err:.6f} > {atol:.1e}"
                )

        print(
            f"[SuperMimic Bridge] Privileged obs alignment verified "
            f"(valid_envs={int(valid.sum().item())}/{self.num_envs}, max_diff={max_diff:.6f}, worst={worst_term})"
        )

        # Restore clean initial episode state before training begins
        self._pushed_mask.zero_()
        reset_obs, _ = self.wrapped_env.reset()
        self.current_obs_dict = reset_obs

    @torch.no_grad()
    def step(self, raw_action: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        """Step IsaacLab environment and return `(obs_dict, sim_reward, dones, timeouts, pushed, info)`."""
        self._pushed_mask.zero_()
        obs_dict, rew, terminated, truncated, extras = self.wrapped_env.step(raw_action)
        self.current_obs_dict = obs_dict
        dones = (terminated | truncated).bool()
        timeouts = extras.get("time_outs", truncated).bool()
        pushed = self._pushed_mask.clone()
        return obs_dict, rew.float(), dones, timeouts, pushed, extras

    def close(self) -> None:
        self.wrapped_env.close()

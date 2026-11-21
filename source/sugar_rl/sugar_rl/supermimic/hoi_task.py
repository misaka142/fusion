"""Differentiable Whole-Body Human-Object Interaction (HOI) Task Layer for SuperMimic x SUGAR.

Unifies:
1. 84-D joint robot-object state representation & SE(2)-invariant world-model features / deltas
2. Zero-mismatch differentiable observation builders for:
   - Stage 1 `PrivilegedCfg` / `TeacherCfg` (890-D)
   - Stage 2 `TrackerCfg` (510-D, including differentiable 5-step proprioception history shift)
3. Exact SUGAR multi-term analytical rewards + expected contact rewards across all 6 tasks
   (`CarryBox`, `KickBox`, `PickBottle`, `PushBox`, `SitChair`, `StandBottle`)
4. Smooth sigmoid soft terminations matching SUGAR's 5 termination criteria for 32-step BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from torch import Tensor

from .rotations import (
    axis_angle_to_quat,
    heading_inv_rotmat,
    heading_rotmat,
    mat_mul,
    mat_t_mul,
    mat_t_vec,
    mat_vec,
    quat_apply_inverse,
    quat_conj,
    quat_mul,
    quat_normalize,
    quat_to_axis_angle,
    quat_to_rotmat,
    rot6d_col,
    rotmat_error_magnitude,
    subtract_frame_transforms,
    sugar_anchor_vel_rotmat,
    sugar_rot6d_from_mat,
    sugar_rot6d_from_quat,
    yaw_quat,
)
from .urdf_kinematics import (
    FKResult,
    FastTreeKinematics,
    SUGAR_EE_TERMINATION_BODY_NAMES,
    SUGAR_TRACKED_BODY_NAMES,
)

# -----------------------------------------------------------------------------
# Dimensions & constants
# -----------------------------------------------------------------------------

ROBOT_SDIM = 71        # pos(3) + quat(4) + q(29) + lin_vel(3) + ang_vel(3) + qd(29)
OBJ_SDIM = 13          # obj_pos(3) + obj_quat(4) + obj_lin_vel(3) + obj_ang_vel(3)
SDIM = ROBOT_SDIM + OBJ_SDIM  # 84

ROBOT_DELTA_DIM = 70   # dp_h(3) + dtheta_local(3) + dq(29) + dv_h(3) + dw_h(3) + dqd(29)
OBJ_DELTA_DIM = 12     # dobj_p_h(3) + dobj_theta_local(3) + dobj_v_h(3) + dobj_w_h(3)
DELTA_DIM = ROBOT_DELTA_DIM + OBJ_DELTA_DIM  # 82

WM_STATE_FEAT_DIM = 123  # robot(71) + obj_rel(16) + contact_pts_in_obj(21) + anticol_pts_in_obj(15)
CONTACT_DIM = 5          # [hoi_obj, feet_obj, left_foot_gnd, right_foot_gnd, undesired]

# Indices inside `SUGAR_TRACKED_BODY_NAMES` for key anti-collision links:
# [7]=torso_link, [9]=left_elbow_link, [10]=left_wrist_yaw_link, [12]=right_elbow_link, [13]=right_wrist_yaw_link
_WM_ANTICOL_TRACKED_INDICES: tuple[int, ...] = (7, 9, 10, 12, 13)

PRIVILEGED_OBS_DIM = 890
TRACKER_OBS_DIM = 510
TRACKER_CMD_DIM = 36
HISTORY_LEN = 5
FUTURE_FRAMES = 8

# Exact 18-term layout of SUGAR's 890-D `PrivilegedCfg` / `TeacherCfg` for startup verification
PRIVILEGED_OBS_SLICES: tuple[tuple[str, int, int], ...] = (
    ("joint_pos_vel_future", 0, 464),
    ("motion_anchor_pos_b_future", 464, 488),
    ("motion_anchor_ori_b_future", 488, 536),
    ("ref_obj_pos_b_future", 536, 560),
    ("ref_obj_ori_b_future", 560, 608),
    ("ref_obj_lin_vel_b_future", 608, 632),
    ("ref_obj_ang_vel_b_future", 632, 656),
    ("body_pos", 656, 698),
    ("body_ori", 698, 782),
    ("base_lin_vel", 782, 785),
    ("base_ang_vel", 785, 788),
    ("joint_pos_rel", 788, 817),
    ("joint_vel_rel", 817, 846),
    ("last_action", 846, 875),
    ("obj_pos_b", 875, 878),
    ("obj_ori_b", 878, 884),
    ("obj_lin_vel_b", 884, 887),
    ("obj_ang_vel_b", 887, 890),
)

# Physical domain randomization context vector xi:
# [robot_static_fric(1), robot_dyn_fric(1), obj_static_fric(1), obj_dyn_fric(1),
#  obj_mass_scale(1), joint_default_offset(29), torso_com_offset(3)] = 37
XI_DIM = 37


@dataclass
class TaskRewardProfile:
    """Task- and stage-specific reward settings matching `sugar_rl/.../*_{refiner,tracker}_env_cfg.py`."""

    task_name: str
    hoi_mode: str                        # "hands" | "feet" | "sit"
    hoi_weight: float = 1.0
    undesired_weight: float = -0.1
    undesired_threshold: float = 1.0
    num_undesired_bodies: int = 26       # Overwritten by live ContactSensor body count in SugarEnvBridge
    feet_slide_weight: float = -0.5
    feet_air_time_weight: float = 1.5
    feet_air_time_threshold: float = 0.25
    avoid_feet_obj_weight: float = 0.0


TASK_PROFILES: dict[str, TaskRewardProfile] = {
    "CarryBox": TaskRewardProfile(
        "CarryBox",
        hoi_mode="hands",
        undesired_weight=-1.0,
        undesired_threshold=0.1,
        num_undesired_bodies=29,
        feet_slide_weight=-0.5,
    ),
    "PickBottle": TaskRewardProfile(
        "PickBottle", hoi_mode="hands", undesired_weight=-0.1, undesired_threshold=1.0, num_undesired_bodies=26
    ),
    "StandBottle": TaskRewardProfile(
        "StandBottle", hoi_mode="hands", undesired_weight=-0.1, undesired_threshold=1.0, num_undesired_bodies=24
    ),
    "PushBox": TaskRewardProfile(
        "PushBox",
        hoi_mode="hands",
        undesired_weight=-0.1,
        undesired_threshold=1.0,
        num_undesired_bodies=26,
        feet_slide_weight=-0.5,
        avoid_feet_obj_weight=1.0,
    ),
    "KickBox": TaskRewardProfile(
        "KickBox",
        hoi_mode="feet",
        undesired_weight=-0.1,
        undesired_threshold=1.0,
        num_undesired_bodies=28,
        feet_slide_weight=0.0,
    ),
    "SitChair": TaskRewardProfile(
        "SitChair", hoi_mode="sit", undesired_weight=-0.1, undesired_threshold=1.0, num_undesired_bodies=20
    ),
}


def resolve_task_profile(task_id_or_name: str, stage: str | None = None) -> TaskRewardProfile:
    """Extract independent task profile copy from Gym ID or name, accounting for Refiner vs Tracker diffs."""
    matched = TASK_PROFILES["CarryBox"]
    for key, prof in TASK_PROFILES.items():
        if key.lower() in task_id_or_name.lower():
            matched = prof
            break
    prof_copy = replace(matched)
    is_tracker = (stage == "tracker") or ("tracker" in task_id_or_name.lower())
    if prof_copy.task_name == "PushBox" and is_tracker:
        # `push_box_tracker_env_cfg.py` uses stronger undesired_contacts (-1.0, threshold=0.1) than Refiner
        prof_copy.undesired_weight = -1.0
        prof_copy.undesired_threshold = 0.1
    return prof_copy


# -----------------------------------------------------------------------------
# Reference trajectory data structures (env-origin normalized world frame)
# -----------------------------------------------------------------------------

@dataclass
class RefFrame:
    """Single-frame reference targets at control step `t` for batch `B`."""

    joint_pos: Tensor         # (B, 29)
    joint_vel: Tensor         # (B, 29)
    body_pos_w: Tensor        # (B, 14, 3)
    body_quat_w: Tensor       # (B, 14, 4)
    body_lin_vel_w: Tensor    # (B, 14, 3)
    body_ang_vel_w: Tensor    # (B, 14, 3)
    obj_pos_w: Tensor         # (B, 3)
    obj_quat_w: Tensor        # (B, 4)
    obj_lin_vel_w: Tensor     # (B, 3)
    obj_ang_vel_w: Tensor     # (B, 3)
    contact_label: Tensor     # (B,) or (B, 1) float in {0.0, 1.0}
    body_rotmat_w: Tensor | None = None    # (B, 14, 3, 3) optional precomputed rotation matrices
    o2b_pos_ref: Tensor | None = None      # (B, 14, 3) optional precomputed obj pos in each tracked body frame
    o2b_rotmat_ref: Tensor | None = None   # (B, 14, 3, 3) optional precomputed R_body^T @ R_obj
    anc_pg_z_ref: Tensor | None = None     # (B,) optional precomputed projected gravity Z in ref anchor frame

    def tracker_command_36(self) -> Tensor:
        """Build 36-D single-frame Tracker reference command `[q_ref(29), v_b(3), w_b(3), contact(1)]`."""
        root_q = self.body_quat_w[:, 0]
        v_b = quat_apply_inverse(root_q, self.body_lin_vel_w[:, 0])
        w_b = quat_apply_inverse(root_q, self.body_ang_vel_w[:, 0])
        c = self.contact_label.float().reshape(-1, 1)
        return torch.cat([self.joint_pos, v_b, w_b, c], dim=-1)


@dataclass
class RefFutureWindow:
    """8-frame lookahead window (`t, t+1, ..., t+7`) required by `PrivilegedCfg` / `TeacherCfg`."""

    joint_pos: Tensor         # (B, 8, 29)
    joint_vel: Tensor         # (B, 8, 29)
    anchor_pos_w: Tensor      # (B, 8, 3)
    anchor_quat_w: Tensor     # (B, 8, 4)
    obj_pos_w: Tensor         # (B, 8, 3)
    obj_quat_w: Tensor        # (B, 8, 4)
    obj_lin_vel_w: Tensor     # (B, 8, 3)
    obj_ang_vel_w: Tensor     # (B, 8, 3)


@dataclass
class TrackerHistoryState:
    """5-step rolling proprioceptive history (`t-4 .. t`) for `TrackerCfg`."""

    base_ang_vel: Tensor      # (B, 5, 3)
    joint_pos_rel: Tensor     # (B, 5, 29)
    joint_vel_rel: Tensor     # (B, 5, 29)
    actions: Tensor           # (B, 5, 29)
    project_gravity: Tensor   # (B, 5, 3)

    def detach(self) -> "TrackerHistoryState":
        return TrackerHistoryState(
            base_ang_vel=self.base_ang_vel.detach(),
            joint_pos_rel=self.joint_pos_rel.detach(),
            joint_vel_rel=self.joint_vel_rel.detach(),
            actions=self.actions.detach(),
            project_gravity=self.project_gravity.detach(),
        )


# -----------------------------------------------------------------------------
# Differentiable HOI Task Engine
# -----------------------------------------------------------------------------

class SugarHOITask:
    """Pure-PyTorch differentiable task layer bridging SuperMimic WM-BPTT and SUGAR."""

    def __init__(
        self,
        fk: FastTreeKinematics,
        task_name_or_id: str = "CarryBox",
        step_dt: float = 0.02,
        pos_term_temp: float = 0.03,
        ori_term_temp: float = 0.08,
        stage: str | None = None,
    ) -> None:
        self.fk = fk
        self.profile = resolve_task_profile(task_name_or_id, stage=stage)
        self.step_dt = step_dt
        self.pos_term_temp = pos_term_temp
        self.ori_term_temp = ori_term_temp
        self.anchor_idx = fk.anchor_tracked_idx  # 7 (`torso_link`)
        self.ee_indices = [SUGAR_TRACKED_BODY_NAMES.index(n) for n in SUGAR_EE_TERMINATION_BODY_NAMES]
        # Left / right inner rubber-palm contact centers relative to wrist_yaw_link (tracked indices [10, 13])
        self._palm_local_offset_from_wrist = torch.tensor(
            [[0.1065, -0.019, 0.0], [0.1065, 0.019, 0.0]],
            dtype=torch.float32,
            device=fk.default_joint_pos.device,
        )

    # ------------------------------------------------------------------
    # 1. State packing / unpacking & kinematics
    # ------------------------------------------------------------------

    @staticmethod
    def pack_state(
        root_pos: Tensor,
        root_quat: Tensor,
        q: Tensor,
        root_lin_vel: Tensor,
        root_ang_vel: Tensor,
        qd: Tensor,
        obj_pos: Tensor,
        obj_quat: Tensor,
        obj_lin_vel: Tensor,
        obj_ang_vel: Tensor,
    ) -> Tensor:
        return torch.cat(
            [
                root_pos,
                quat_normalize(root_quat),
                q,
                root_lin_vel,
                root_ang_vel,
                qd,
                obj_pos,
                quat_normalize(obj_quat),
                obj_lin_vel,
                obj_ang_vel,
            ],
            dim=-1,
        )

    @staticmethod
    def unpack_state(s_flat: Tensor) -> tuple[Tensor, ...]:
        root_pos = s_flat[..., 0:3]
        root_quat = quat_normalize(s_flat[..., 3:7])
        q = s_flat[..., 7:36]
        root_lin_vel = s_flat[..., 36:39]
        root_ang_vel = s_flat[..., 39:42]
        qd = s_flat[..., 42:71]
        obj_pos = s_flat[..., 71:74]
        obj_quat = quat_normalize(s_flat[..., 74:78])
        obj_lin_vel = s_flat[..., 78:81]
        obj_ang_vel = s_flat[..., 81:84]
        return (
            root_pos,
            root_quat,
            q,
            root_lin_vel,
            root_ang_vel,
            qd,
            obj_pos,
            obj_quat,
            obj_lin_vel,
            obj_ang_vel,
        )

    def run_fk(self, s_flat: Tensor, with_quat: bool = True) -> FKResult:
        root_pos, root_quat, q, root_v, root_w, qd, _, _, _, _ = self.unpack_state(s_flat)
        return self.fk(
            root_pos,
            root_quat,
            q,
            root_v,
            root_w,
            qd,
            ang_vel_in_world=True,
            with_quat=with_quat,
        )

    def effective_default_joint_pos(self, xi: Tensor | None = None) -> Tensor:
        """Return nominal default joint positions plus optional per-env calibration offset `xi[..., 5:34]`."""
        q_def = self.fk.default_joint_pos
        if xi is not None:
            return q_def + xi[..., 5:34]
        return q_def

    def raw_action_to_setpoint(self, raw_action: Tensor, xi: Tensor | None = None) -> Tensor:
        """Convert policy raw action `a` to joint PD target `q_default_nominal (+ offset) + scale * a`."""
        return self.effective_default_joint_pos(xi) + self.fk.action_scale * raw_action

    # ------------------------------------------------------------------
    # 2. SE(2)-invariant world model features, delta integration & loss
    # ------------------------------------------------------------------

    def state_features(self, s_flat: Tensor, fk_res: FKResult | None = None) -> Tensor:
        """Extract 123-D SE(2)-invariant robot-object-contact-anticollision feature vector for `JointWorldModel`."""
        if fk_res is None:
            fk_res = self.run_fk(s_flat, with_quat=False)
        (
            root_pos,
            root_quat,
            q,
            root_v,
            root_w,
            qd,
            obj_pos,
            obj_quat,
            obj_v,
            obj_w,
        ) = self.unpack_state(s_flat)

        R_h_inv = heading_inv_rotmat(root_quat)         # (B, 3, 3)
        R_root = fk_res.root_rotmat_w if fk_res.root_rotmat_w is not None else quat_to_rotmat(root_quat)
        R_obj = quat_to_rotmat(obj_quat)                # (B, 3, 3)

        # Robot features (71-D)
        z_root = root_pos[:, 2:3]
        rot6d_root = rot6d_col(mat_mul(R_h_inv, R_root))
        v_h = mat_vec(R_h_inv, root_v)
        w_h = mat_vec(R_h_inv, root_w)

        # Object features in robot heading frame + ground height (16-D)
        root_xy0 = torch.cat([root_pos[:, :2], torch.zeros_like(z_root)], dim=-1)
        obj_pos_h = mat_vec(R_h_inv, obj_pos - root_xy0)
        obj_rel_z = obj_pos[:, 2:3] - z_root
        rot6d_obj = rot6d_col(mat_mul(R_h_inv, R_obj))
        obj_v_h = mat_vec(R_h_inv, obj_v)
        obj_w_h = mat_vec(R_h_inv, obj_w)

        # 7 contact links + 5 torso/elbow/wrist anti-collision links expressed in object frame (21 + 15 = 36-D)
        anticol_pos_w = fk_res.tracked_pos_w[:, _WM_ANTICOL_TRACKED_INDICES]
        pts_w = torch.cat([fk_res.contact_pos_w, anticol_pos_w], dim=1)     # (B, 12, 3)
        rel_pts_w = pts_w - obj_pos.unsqueeze(1)
        rel_pts_obj = mat_t_vec(R_obj.unsqueeze(1), rel_pts_w).reshape(s_flat.shape[0], -1)

        return torch.cat(
            [
                z_root,
                rot6d_root,
                q,
                v_h,
                w_h,
                qd,
                obj_pos_h,
                obj_rel_z,
                rot6d_obj,
                obj_v_h,
                obj_w_h,
                rel_pts_obj,
            ],
            dim=-1,
        )

    @staticmethod
    def compute_state_delta(s_prev: Tensor, s_next: Tensor) -> Tensor:
        """Exact inverse of `apply_delta`: compute 82-D local-frame state delta from `(s_prev, s_next)`."""
        lead_shape = s_prev.shape[:-1]
        sp = s_prev.reshape(-1, SDIM)
        sn = s_next.reshape(-1, SDIM)
        (
            p0, q0, j0, v0, w0, jd0,
            op0, oq0, ov0, ow0,
        ) = SugarHOITask.unpack_state(sp)
        (
            p1, q1, j1, v1, w1, jd1,
            op1, oq1, ov1, ow1,
        ) = SugarHOITask.unpack_state(sn)

        R_h_inv = heading_inv_rotmat(q0)
        dp_h = mat_vec(R_h_inv, p1 - p0)
        dtheta = quat_to_axis_angle(quat_mul(quat_conj(q0), q1))
        dq = j1 - j0
        dv_h = mat_vec(R_h_inv, v1 - v0)
        dw_h = mat_vec(R_h_inv, w1 - w0)
        dqd = jd1 - jd0

        dobj_p_h = mat_vec(R_h_inv, op1 - op0)
        dobj_theta = quat_to_axis_angle(quat_mul(quat_conj(oq0), oq1))
        dobj_v_h = mat_vec(R_h_inv, ov1 - ov0)
        dobj_w_h = mat_vec(R_h_inv, ow1 - ow0)

        delta = torch.cat(
            [dp_h, dtheta, dq, dv_h, dw_h, dqd, dobj_p_h, dobj_theta, dobj_v_h, dobj_w_h],
            dim=-1,
        )
        return delta.reshape(*lead_shape, DELTA_DIM)

    @staticmethod
    def apply_delta(s_flat: Tensor, delta: Tensor) -> Tensor:
        """Integrate 82-D local state delta into next 84-D joint world state."""
        (
            root_pos,
            root_quat,
            q,
            root_v,
            root_w,
            qd,
            obj_pos,
            obj_quat,
            obj_v,
            obj_w,
        ) = SugarHOITask.unpack_state(s_flat)

        R_h = heading_rotmat(root_quat)

        dp_h = delta[..., 0:3]
        dtheta = delta[..., 3:6]
        dq = delta[..., 6:35]
        dv_h = delta[..., 35:38]
        dw_h = delta[..., 38:41]
        dqd = delta[..., 41:70]

        dobj_p_h = delta[..., 70:73]
        dobj_theta = delta[..., 73:76]
        dobj_v_h = delta[..., 76:79]
        dobj_w_h = delta[..., 79:82]

        next_root_pos = root_pos + mat_vec(R_h, dp_h)
        next_root_quat = quat_normalize(quat_mul(root_quat, axis_angle_to_quat(dtheta)))
        next_q = q + dq
        next_root_v = root_v + mat_vec(R_h, dv_h)
        next_root_w = root_w + mat_vec(R_h, dw_h)
        next_qd = qd + dqd

        next_obj_pos = obj_pos + mat_vec(R_h, dobj_p_h)
        next_obj_quat = quat_normalize(quat_mul(obj_quat, axis_angle_to_quat(dobj_theta)))
        next_obj_v = obj_v + mat_vec(R_h, dobj_v_h)
        next_obj_w = obj_w + mat_vec(R_h, dobj_w_h)

        return SugarHOITask.pack_state(
            next_root_pos,
            next_root_quat,
            next_q,
            next_root_v,
            next_root_w,
            next_qd,
            next_obj_pos,
            next_obj_quat,
            next_obj_v,
            next_obj_w,
        )

    @staticmethod
    def world_model_state_loss(
        s_pred: Tensor,
        s_true: Tensor,
        joint_weights: Tensor,
        vel_loss_scale: float = 0.5,
        obj_loss_weight: float = 1.5,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute local-frame Huber loss between predicted and ground-truth joint states."""
        (
            p_hat, q_hat_rot, j_hat, v_hat, w_hat, jd_hat,
            op_hat, oq_hat, ov_hat, ow_hat,
        ) = SugarHOITask.unpack_state(s_pred)
        (
            p_gt, q_gt_rot, j_gt, v_gt, w_gt, jd_gt,
            op_gt, oq_gt, ov_gt, ow_gt,
        ) = SugarHOITask.unpack_state(s_true)

        R_h_inv = heading_inv_rotmat(q_gt_rot)
        jw = joint_weights.view(1, 29)
        jw_sqrt = torch.sqrt(jw)

        # Robot position & geodesic axis-angle error
        err_p = mat_vec(R_h_inv, p_hat - p_gt)
        err_rot = quat_to_axis_angle(quat_mul(quat_conj(q_gt_rot), q_hat_rot))
        err_j = (j_hat - j_gt) * jw_sqrt
        err_v = mat_vec(R_h_inv, v_hat - v_gt) * vel_loss_scale
        err_w = mat_vec(R_h_inv, w_hat - w_gt) * vel_loss_scale
        err_jd = (jd_hat - jd_gt) * jw_sqrt * vel_loss_scale

        # Object position, rotation & velocity errors
        err_op = mat_vec(R_h_inv, op_hat - op_gt)
        err_orot = quat_to_axis_angle(quat_mul(quat_conj(oq_gt), oq_hat))
        err_ov = mat_vec(R_h_inv, ov_hat - ov_gt) * vel_loss_scale
        err_ow = mat_vec(R_h_inv, ow_hat - ow_gt) * vel_loss_scale

        rob_vec = torch.cat([err_p, err_rot, err_j, err_v, err_w, err_jd], dim=-1)
        obj_vec = torch.cat([err_op, err_orot, err_ov, err_ow], dim=-1)

        loss_rob = F.smooth_l1_loss(rob_vec, torch.zeros_like(rob_vec), beta=0.05)
        loss_obj = F.smooth_l1_loss(obj_vec, torch.zeros_like(obj_vec), beta=0.05)
        total = loss_rob + obj_loss_weight * loss_obj

        metrics = {
            "wm/loss_robot": loss_rob.detach(),
            "wm/loss_obj": loss_obj.detach(),
            "wm/pos_err_mm": (p_hat - p_gt).detach().norm(dim=-1).mean() * 1000.0,
            "wm/obj_err_mm": (op_hat - op_gt).detach().norm(dim=-1).mean() * 1000.0,
        }
        return total, metrics

    # ------------------------------------------------------------------
    # 3. Differentiable Observation Builders (100% aligned with SUGAR)
    # ------------------------------------------------------------------

    def build_privileged_obs(
        self,
        s_flat: Tensor,
        last_action: Tensor,
        ref_future: RefFutureWindow,
        fk_res: FKResult | None = None,
        xi: Tensor | None = None,
    ) -> Tensor:
        """Construct 890-D `PrivilegedCfg` / `TeacherCfg` observation vector.

        Matches every single term and reshape layout in `sugar_rl/tasks/locomanip/mdp/observations.py`.
        """
        if fk_res is None:
            fk_res = self.run_fk(s_flat, with_quat=False)
        (
            _,
            root_quat,
            q,
            root_v,
            root_w,
            qd,
            obj_pos_w,
            obj_quat_w,
            obj_v_w,
            obj_w_w,
        ) = self.unpack_state(s_flat)

        B = s_flat.shape[0]
        K = ref_future.joint_pos.shape[1]  # 8

        anchor_pos_w = fk_res.anchor_pos_w        # (B, 3)
        anchor_quat_w = fk_res.anchor_quat_w      # (B, 4)
        anchor_rotmat_w = fk_res.anchor_rotmat_w  # (B, 3, 3)

        # 1. joint_pos_vel_future: (B, 8, 58) -> (B, 464)
        joint_pos_vel_fut = torch.cat([ref_future.joint_pos, ref_future.joint_vel], dim=-1).reshape(B, -1)

        # 2 & 3. motion_anchor_pos_b_future (24) & motion_anchor_ori_b_future (48)
        anc_p_rep = anchor_pos_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3)
        anc_q_rep = anchor_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)
        fut_anc_p, fut_anc_q = subtract_frame_transforms(
            anc_p_rep,
            anc_q_rep,
            ref_future.anchor_pos_w.reshape(-1, 3),
            ref_future.anchor_quat_w.reshape(-1, 4),
        )
        motion_anchor_pos_b_fut = fut_anc_p.reshape(B, -1)
        motion_anchor_ori_b_fut = sugar_rot6d_from_quat(fut_anc_q).reshape(B, -1)

        # 4 & 5. ref_obj_pos_b_future (24) & ref_obj_ori_b_future (48) -- in CURRENT OBJECT frame!
        obj_p_rep = obj_pos_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3)
        obj_q_rep = obj_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)
        fut_obj_p, fut_obj_q = subtract_frame_transforms(
            obj_p_rep,
            obj_q_rep,
            ref_future.obj_pos_w.reshape(-1, 3),
            ref_future.obj_quat_w.reshape(-1, 4),
        )
        ref_obj_pos_fut = fut_obj_p.reshape(B, -1)
        ref_obj_ori_fut = sugar_rot6d_from_quat(fut_obj_q).reshape(B, -1)

        # 6 & 7. ref_obj_lin_vel_b_future (24) & ref_obj_ang_vel_b_future (24)
        R_anc_vel = sugar_anchor_vel_rotmat(anchor_quat_w)  # (B, 3, 3)
        ref_obj_v_fut = mat_vec(R_anc_vel.unsqueeze(1), ref_future.obj_lin_vel_w).reshape(B, -1)
        ref_obj_w_fut = mat_vec(R_anc_vel.unsqueeze(1), ref_future.obj_ang_vel_w).reshape(B, -1)

        # 8 & 9. robot_body_pos_b (42) & robot_body_ori_b (84) directly from FK rotation matrices
        body_pos_b = mat_t_vec(anchor_rotmat_w.unsqueeze(1), fk_res.tracked_pos_w - anchor_pos_w.unsqueeze(1))
        body_rot_b = mat_t_mul(anchor_rotmat_w.unsqueeze(1), fk_res.tracked_rotmat_w)
        obs_body_pos = body_pos_b.reshape(B, -1)
        obs_body_ori = sugar_rot6d_from_mat(body_rot_b).reshape(B, -1)

        # 10 & 11. base_lin_vel (3) & base_ang_vel (3) in pelvis frame
        R_root = fk_res.root_rotmat_w if fk_res.root_rotmat_w is not None else quat_to_rotmat(root_quat)
        base_lin_vel = mat_t_vec(R_root, root_v)
        base_ang_vel = mat_t_vec(R_root, root_w)

        # 12, 13, 14. joint_pos_rel (29), joint_vel_rel (29), last_action (29)
        joint_pos_rel = q - self.effective_default_joint_pos(xi)
        joint_vel_rel = qd

        # 15, 16, 17, 18. current object pose & velocity in anchor frame
        R_obj_w = quat_to_rotmat(obj_quat_w)
        cur_obj_pos_b = mat_t_vec(anchor_rotmat_w, obj_pos_w - anchor_pos_w)
        cur_obj_ori_b = sugar_rot6d_from_mat(mat_t_mul(anchor_rotmat_w, R_obj_w))
        cur_obj_v_b = mat_vec(R_anc_vel, obj_v_w)
        cur_obj_w_b = mat_vec(R_anc_vel, obj_w_w)

        return torch.cat(
            [
                joint_pos_vel_fut,
                motion_anchor_pos_b_fut,
                motion_anchor_ori_b_fut,
                ref_obj_pos_fut,
                ref_obj_ori_fut,
                ref_obj_v_fut,
                ref_obj_w_fut,
                obs_body_pos,
                obs_body_ori,
                base_lin_vel,
                base_ang_vel,
                joint_pos_rel,
                joint_vel_rel,
                last_action,
                cur_obj_pos_b,
                cur_obj_ori_b,
                cur_obj_v_b,
                cur_obj_w_b,
            ],
            dim=-1,
        )

    def single_step_proprio(
        self, s_flat: Tensor, last_action: Tensor, xi: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute current-step proprioceptive terms for `TrackerHistoryState`."""
        _, root_quat, q, _, root_w, qd, _, _, _, _ = self.unpack_state(s_flat)
        R_root = quat_to_rotmat(root_quat)
        base_ang_vel = mat_t_vec(R_root, root_w)
        joint_pos_rel = q - self.effective_default_joint_pos(xi)
        joint_vel_rel = qd
        proj_grav = -R_root[:, 2, :]
        return base_ang_vel, joint_pos_rel, joint_vel_rel, last_action, proj_grav

    def init_tracker_history_from_states(
        self, s_seq: Tensor, last_action_seq: Tensor, xi: Tensor | None = None
    ) -> TrackerHistoryState:
        """Build 5-frame `TrackerHistoryState` from state history `(B, 5, 84)` and actions `(B, 5, 29)`."""
        B, L, _ = s_seq.shape
        assert L == HISTORY_LEN
        xi_flat = (
            xi.unsqueeze(1).expand(-1, L, -1).reshape(B * L, XI_DIM)
            if xi is not None
            else None
        )
        ang_w, q_rel, qd_rel, act_h, p_grav = self.single_step_proprio(
            s_seq.reshape(B * L, SDIM), last_action_seq.reshape(B * L, 29), xi=xi_flat
        )
        return TrackerHistoryState(
            base_ang_vel=ang_w.reshape(B, L, 3),
            joint_pos_rel=q_rel.reshape(B, L, 29),
            joint_vel_rel=qd_rel.reshape(B, L, 29),
            actions=act_h.reshape(B, L, 29),
            project_gravity=p_grav.reshape(B, L, 3),
        )

    def advance_tracker_history(
        self, hist: TrackerHistoryState, s_next: Tensor, action_taken: Tensor, xi: Tensor | None = None
    ) -> TrackerHistoryState:
        """Differentiably shift 5-step history left and append new state `s_next` + `action_taken`."""
        ang_w, q_rel, qd_rel, act_t, p_grav = self.single_step_proprio(s_next, action_taken, xi=xi)
        return TrackerHistoryState(
            base_ang_vel=torch.cat([hist.base_ang_vel[:, 1:], ang_w.unsqueeze(1)], dim=1),
            joint_pos_rel=torch.cat([hist.joint_pos_rel[:, 1:], q_rel.unsqueeze(1)], dim=1),
            joint_vel_rel=torch.cat([hist.joint_vel_rel[:, 1:], qd_rel.unsqueeze(1)], dim=1),
            actions=torch.cat([hist.actions[:, 1:], act_t.unsqueeze(1)], dim=1),
            project_gravity=torch.cat([hist.project_gravity[:, 1:], p_grav.unsqueeze(1)], dim=1),
        )

    def build_tracker_obs(
        self,
        s_flat: Tensor,
        ref_cmd_36: Tensor,
        hist: TrackerHistoryState,
        fk_res: FKResult | None = None,
        add_noise: bool = False,
    ) -> Tensor:
        """Construct 510-D `TrackerCfg` observation vector matching SUGAR Tracker & Inference."""
        if fk_res is None:
            fk_res = self.run_fk(s_flat, with_quat=False)
        B = s_flat.shape[0]
        obj_pos_w = s_flat[..., 71:74]
        obj_quat_w = quat_normalize(s_flat[..., 74:78])

        cur_obj_pos_b = mat_t_vec(fk_res.anchor_rotmat_w, obj_pos_w - fk_res.anchor_pos_w)
        cur_obj_ori_b = sugar_rot6d_from_mat(mat_t_mul(fk_res.anchor_rotmat_w, quat_to_rotmat(obj_quat_w)))

        cmd = ref_cmd_36
        b_ang = hist.base_ang_vel.reshape(B, -1)
        j_pos = hist.joint_pos_rel.reshape(B, -1)
        j_vel = hist.joint_vel_rel.reshape(B, -1)
        acts = hist.actions.reshape(B, -1)
        p_grav = hist.project_gravity.reshape(B, -1)

        if add_noise:
            def _u(x: Tensor, lim: float) -> Tensor:
                return x + (torch.rand_like(x) * 2.0 - 1.0) * lim

            cmd = torch.cat(
                [
                    _u(cmd[:, 0:29], 0.01),
                    _u(cmd[:, 29:32], 0.05),
                    _u(cmd[:, 32:35], 0.10),
                    cmd[:, 35:36],
                ],
                dim=-1,
            )
            b_ang = _u(b_ang, 0.20)
            j_pos = _u(j_pos, 0.01)
            j_vel = _u(j_vel, 0.50)
            p_grav = _u(p_grav, 0.20)
            cur_obj_pos_b = _u(cur_obj_pos_b, 0.05)
            cur_obj_ori_b = _u(cur_obj_ori_b, 0.10)

        return torch.cat([cmd, b_ang, j_pos, j_vel, acts, p_grav, cur_obj_pos_b, cur_obj_ori_b], dim=-1)

    # ------------------------------------------------------------------
    # 4. Differentiable SUGAR Rewards & Soft Terminations
    # ------------------------------------------------------------------

    def _compute_relative_ref_bodies_mat(
        self, fk_res: FKResult, ref: RefFrame
    ) -> tuple[Tensor, Tensor]:
        """Fast rotation-matrix version of `_compute_relative_ref_bodies` (`body_pos_rel_w`, `body_rotmat_rel_w`)."""
        nb = len(SUGAR_TRACKED_BODY_NAMES)
        ref_anc_pos = ref.body_pos_w[:, self.anchor_idx]      # (B, 3)
        ref_anc_quat = ref.body_quat_w[:, self.anchor_idx]    # (B, 4)

        rob_anc_pos = fk_res.anchor_pos_w                     # (B, 3)
        rob_anc_quat = fk_res.anchor_quat_w                   # (B, 4)

        delta_pos_w = torch.cat([rob_anc_pos[:, :2], ref_anc_pos[:, 2:3]], dim=-1).unsqueeze(1).expand(-1, nb, -1)
        delta_ori_single = yaw_quat(quat_mul(rob_anc_quat, quat_conj(ref_anc_quat)))
        R_delta = quat_to_rotmat(delta_ori_single).unsqueeze(1)
        R_ref_b = ref.body_rotmat_w if ref.body_rotmat_w is not None else quat_to_rotmat(ref.body_quat_w)

        body_rotmat_rel_w = mat_mul(R_delta, R_ref_b)
        body_pos_rel_w = delta_pos_w + mat_vec(R_delta, ref.body_pos_w - ref_anc_pos.unsqueeze(1))
        return body_pos_rel_w, body_rotmat_rel_w

    def _compute_relative_ref_bodies(
        self, fk_res: FKResult, ref: RefFrame
    ) -> tuple[Tensor, Tensor]:
        """Reproduce `commands.py` lines 1381-1390 (`body_pos_relative_w`, `body_quat_relative_w`) out-of-place."""
        nb = len(SUGAR_TRACKED_BODY_NAMES)
        ref_anc_pos = ref.body_pos_w[:, self.anchor_idx]      # (B, 3)
        ref_anc_quat = ref.body_quat_w[:, self.anchor_idx]    # (B, 4)

        rob_anc_pos = fk_res.anchor_pos_w                     # (B, 3)
        rob_anc_quat = fk_res.anchor_quat_w                   # (B, 4)

        # delta_pos_w: robot (x, y) + reference anchor z
        delta_pos_w = torch.cat([rob_anc_pos[:, :2], ref_anc_pos[:, 2:3]], dim=-1).unsqueeze(1).expand(-1, nb, -1)
        delta_ori_single = yaw_quat(quat_mul(rob_anc_quat, quat_conj(ref_anc_quat)))
        R_delta = quat_to_rotmat(delta_ori_single).unsqueeze(1)
        delta_ori_w = delta_ori_single.unsqueeze(1).expand(-1, nb, -1)

        body_quat_rel_w = quat_mul(delta_ori_w, ref.body_quat_w)
        body_pos_rel_w = delta_pos_w + mat_vec(R_delta, ref.body_pos_w - ref_anc_pos.unsqueeze(1))
        return body_pos_rel_w, body_quat_rel_w

    def compute_reward_and_termination(
        self,
        s_prev: Tensor,
        s_next: Tensor,
        raw_action: Tensor,
        prev_raw_action: Tensor,
        ref_cur: RefFrame | None = None,
        contact_logits: Tensor | None = None,
        fk_next: FKResult | None = None,
        xi: Tensor | None = None,
        feet_air_time: Tensor | None = None,
        prev_feet_gnd: Tensor | None = None,
        time_steps: Tensor | None = None,
        *,
        ref_next: RefFrame | None = None,
        imagination_shaping: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Compute step reward (scaled by `step_dt=0.02`) and smooth termination probability in [0, 1].

        Note on SUGAR / IsaacLab step ordering:
        In `ManagerBasedRLEnv.step()`, `termination_manager.compute()` and `reward_manager.compute(dt)`
        execute AFTER physics advances to `s_next`, but BEFORE `command_manager.compute()` increments
        `MotionCommand.time_steps`. Therefore, `s_next` is evaluated against `ref_cur = ref[t_cur]`
        (the exact same reference index observed when choosing `raw_action`).
        """
        ref_target = ref_cur if ref_cur is not None else ref_next
        if ref_target is None:
            raise ValueError("`ref_cur` (current step reference frame) must be provided.")

        if fk_next is None:
            fk_next = self.run_fk(s_next, with_quat=False)

        _, _, q_prev, _, _, qd_prev, _, _, _, _ = self.unpack_state(s_prev)
        _, _, q_next, _, _, qd_next, obj_pos_w, obj_quat_w, obj_v_w, obj_w_w = self.unpack_state(s_next)

        ref_anc_pos = ref_target.body_pos_w[:, self.anchor_idx]
        ref_anc_quat = ref_target.body_quat_w[:, self.anchor_idx]
        ref_anc_rotmat = (
            ref_target.body_rotmat_w[:, self.anchor_idx]
            if ref_target.body_rotmat_w is not None
            else quat_to_rotmat(ref_anc_quat)
        )
        body_pos_rel_w, body_rotmat_rel_w = self._compute_relative_ref_bodies_mat(fk_next, ref_target)

        # ---- A. Tracking rewards (exact weights & stds from BaseRewardsCfg) ----
        err_jpos = ((ref_target.joint_pos - q_next) ** 2).sum(dim=-1)
        r_joint_pos = 0.20 * torch.exp(-err_jpos / (0.6**2))

        err_anc_p = ((ref_anc_pos - fk_next.anchor_pos_w) ** 2).sum(dim=-1)
        r_anc_pos = 0.45 * torch.exp(-err_anc_p / (0.3**2))

        err_anc_q = rotmat_error_magnitude(ref_anc_rotmat, fk_next.anchor_rotmat_w) ** 2
        r_anc_ori = 0.25 * torch.exp(-err_anc_q / (0.4**2))

        err_bpos = ((body_pos_rel_w - fk_next.tracked_pos_w) ** 2).sum(dim=-1).mean(dim=-1)
        r_body_pos = 0.45 * torch.exp(-err_bpos / (0.3**2))

        err_bori = (rotmat_error_magnitude(body_rotmat_rel_w, fk_next.tracked_rotmat_w) ** 2).mean(dim=-1)
        r_body_ori = 0.25 * torch.exp(-err_bori / (0.4**2))

        err_bv = ((ref_target.body_lin_vel_w - fk_next.tracked_lin_vel_w) ** 2).sum(dim=-1).mean(dim=-1)
        r_body_vel = 0.25 * torch.exp(-err_bv / (1.0**2))

        err_bw = ((ref_target.body_ang_vel_w - fk_next.tracked_ang_vel_w) ** 2).sum(dim=-1).mean(dim=-1)
        r_body_ang = 0.25 * torch.exp(-err_bw / (3.14**2))

        # ---- B. Object global tracking rewards ----
        R_obj_w = quat_to_rotmat(obj_quat_w)
        R_ref_obj_w = quat_to_rotmat(ref_target.obj_quat_w)
        err_op = ((obj_pos_w - ref_target.obj_pos_w) ** 2).sum(dim=-1)
        r_obj_pos = 0.5 * torch.exp(-err_op / (0.3**2))

        err_oq = rotmat_error_magnitude(R_obj_w, R_ref_obj_w) ** 2
        r_obj_ori = 0.5 * torch.exp(-err_oq / (0.4**2))

        err_ov = ((obj_v_w - ref_target.obj_lin_vel_w) ** 2).sum(dim=-1)
        r_obj_vel = 0.5 * torch.exp(-err_ov / (1.0**2))

        err_ow = ((obj_w_w - ref_target.obj_ang_vel_w) ** 2).sum(dim=-1)
        r_obj_ang = 0.5 * torch.exp(-err_ow / (3.14**2))

        # ---- C. Human-object relative interaction rewards (`obj2body_pos`, `obj2body_ori`) ----
        o2b_p = mat_t_vec(fk_next.tracked_rotmat_w, obj_pos_w.unsqueeze(1) - fk_next.tracked_pos_w)
        o2b_R = mat_t_mul(fk_next.tracked_rotmat_w, R_obj_w.unsqueeze(1))
        if ref_target.o2b_pos_ref is not None and ref_target.o2b_rotmat_ref is not None:
            ref_o2b_p = ref_target.o2b_pos_ref
            ref_o2b_R = ref_target.o2b_rotmat_ref
        else:
            R_ref_b = (
                ref_target.body_rotmat_w
                if ref_target.body_rotmat_w is not None
                else quat_to_rotmat(ref_target.body_quat_w)
            )
            ref_o2b_p = mat_t_vec(R_ref_b, ref_target.obj_pos_w.unsqueeze(1) - ref_target.body_pos_w)
            ref_o2b_R = mat_t_mul(R_ref_b, R_ref_obj_w.unsqueeze(1))
        err_o2b_p = ((o2b_p - ref_o2b_p) ** 2).sum(dim=-1).mean(dim=-1)
        r_o2b_pos = 0.25 * torch.exp(-err_o2b_p / (0.3**2))

        err_o2b_q = (rotmat_error_magnitude(o2b_R, ref_o2b_R) ** 2).mean(dim=-1)
        r_o2b_ori = 0.25 * torch.exp(-err_o2b_q / (0.4**2))

        # ---- D. Expected contact rewards (WM head + smooth kinematic proximity) ----
        c_label = ref_target.contact_label.float().reshape(-1)
        if contact_logits is not None:
            probs = torch.sigmoid(contact_logits)
            p_hoi = probs[:, 0]
            p_feet_obj = probs[:, 1]
            p_lf_gnd = probs[:, 2]
            p_rf_gnd = probs[:, 3]
            # Head 4 predicts fraction of undesired bodies in contact -> multiply by body count!
            expected_undesired_count = float(self.profile.num_undesired_bodies) * probs[:, 4]
        else:
            # Kinematic proximity fallback when evaluated without world model logits
            # SUGAR_CONTACT_BODY_NAMES: [lh(0), rh(1), lf(2), rf(3), lhip(4), rhip(5), pelv(6)]
            d_obj = (fk_next.contact_pos_w - obj_pos_w.unsqueeze(1)).norm(dim=-1)
            if self.profile.hoi_mode == "hands":
                p_hoi = torch.sigmoid((0.25 - d_obj[:, 0]) * 15.0) * torch.sigmoid((0.25 - d_obj[:, 1]) * 15.0)
            elif self.profile.hoi_mode == "feet":
                p_hoi = torch.max(
                    torch.sigmoid((0.30 - d_obj[:, 2]) * 15.0),
                    torch.sigmoid((0.30 - d_obj[:, 3]) * 15.0),
                )
            else:
                p_hoi = torch.sigmoid((0.35 - d_obj[:, 6]) * 15.0)
            p_feet_obj = torch.max(
                torch.sigmoid((0.25 - d_obj[:, 2]) * 15.0),
                torch.sigmoid((0.25 - d_obj[:, 3]) * 15.0),
            )
            p_lf_gnd = torch.sigmoid((0.06 - fk_next.contact_pos_w[:, 2, 2]) * 40.0)
            p_rf_gnd = torch.sigmoid((0.06 - fk_next.contact_pos_w[:, 3, 2]) * 40.0)
            expected_undesired_count = torch.zeros_like(c_label)

        r_hoi_contact = self.profile.hoi_weight * (c_label * p_hoi + (1.0 - c_label) * (1.0 - p_hoi))
        r_avoid_feet = self.profile.avoid_feet_obj_weight * (1.0 - p_feet_obj)
        r_undesired = self.profile.undesired_weight * expected_undesired_count

        # Feet horizontal slide penalty (`contact_pos_w[:, 2:4]` are left/right ankle_roll_link)
        v_lf_xy = torch.sqrt((fk_next.contact_lin_vel_w[:, 2, :2] ** 2).sum(dim=-1) + 1e-8)
        v_rf_xy = torch.sqrt((fk_next.contact_lin_vel_w[:, 3, :2] ** 2).sum(dim=-1) + 1e-8)
        r_feet_slide = self.profile.feet_slide_weight * (v_lf_xy * p_lf_gnd + v_rf_xy * p_rf_gnd)

        # Differentiable `feet_air_time_min_penalty` (gated by reference foot XY motion so stand/bend never taps feet)
        curr_feet_gnd = torch.stack([p_lf_gnd, p_rf_gnd], dim=-1)  # (B, 2)
        if feet_air_time is not None and self.profile.feet_air_time_weight != 0.0:
            if prev_feet_gnd is None:
                prev_feet_gnd = (feet_air_time <= 1e-4).float()
            # Detach `prev_feet_gnd` so 1st-order BPTT cannot exploit "stay permanently glued to floor"
            p_first_contact = (1.0 - prev_feet_gnd.detach()) * curr_feet_gnd
            air_before = feet_air_time + self.step_dt
            air_pen = (air_before - self.profile.feet_air_time_threshold).clamp_max(0.0) * p_first_contact
            ready_mask = (time_steps > 50).float() if time_steps is not None else 1.0
            ref_feet_vxy = torch.sqrt(
                (ref_target.body_lin_vel_w[:, [3, 6], :2] ** 2).sum(dim=-1) + 1e-8
            ).max(dim=-1)[0]
            walk_gate = ((ref_feet_vxy - 0.04) / 0.05).clamp(0.0, 1.0)
            r_feet_air_time = self.profile.feet_air_time_weight * air_pen.sum(dim=-1) * ready_mask * walk_gate
            next_feet_air_time = air_before * (1.0 - curr_feet_gnd)
        else:
            r_feet_air_time = torch.zeros_like(c_label)
            next_feet_air_time = (
                (feet_air_time + self.step_dt) * (1.0 - curr_feet_gnd)
                if feet_air_time is not None
                else (1.0 - curr_feet_gnd) * self.step_dt
            )

        # ---- E. Regularization rewards (action rate, joint limits, joint acc, PD torque) ----
        r_act_rate = -0.18 * ((raw_action - prev_raw_action) ** 2).sum(dim=-1)

        out_low = (self.fk.soft_joint_lower.unsqueeze(0) - q_next).clamp_min(0.0)
        out_high = (q_next - self.fk.soft_joint_upper.unsqueeze(0)).clamp_min(0.0)
        r_jlimit = -10.0 * (out_low + out_high).sum(dim=-1)

        qdd_approx = (qd_next - qd_prev) / self.step_dt
        r_jacc = -1.0e-6 * (qdd_approx**2).sum(dim=-1)

        q_sp = self.raw_action_to_setpoint(raw_action, xi=xi)
        tau_pd = (
            self.fk.stiffness.unsqueeze(0) * (q_sp - q_next)
            - self.fk.damping.unsqueeze(0) * qd_next
        ).clamp(-self.fk.effort_limit.unsqueeze(0), self.fk.effort_limit.unsqueeze(0))
        r_jtorque = -1.0e-5 * (tau_pd**2).sum(dim=-1)

        unscaled_reward = (
            r_joint_pos
            + r_anc_pos
            + r_anc_ori
            + r_body_pos
            + r_body_ori
            + r_body_vel
            + r_body_ang
            + r_obj_pos
            + r_obj_ori
            + r_obj_vel
            + r_obj_ang
            + r_o2b_pos
            + r_o2b_ori
            + r_hoi_contact
            + r_avoid_feet
            + r_undesired
            + r_feet_slide
            + r_feet_air_time
            + r_act_rate
            + r_jlimit
            + r_jacc
            + r_jtorque
        )
        if imagination_shaping:
            # Fine-grained ankle clearance Z tracking (indices [3, 6] = left/right ankle_roll_link)
            err_fz = ((body_pos_rel_w[:, [3, 6], 2] - fk_next.tracked_pos_w[:, [3, 6], 2]) ** 2).sum(dim=-1)
            unscaled_reward = unscaled_reward + 0.25 * torch.exp(-err_fz / (0.06**2))
            if self.profile.hoi_mode == "hands":
                # Focused inner rubber-palm position & orientation alignment in object frame
                R_ref_wrists = (
                    ref_target.body_rotmat_w[:, [10, 13]]
                    if ref_target.body_rotmat_w is not None
                    else quat_to_rotmat(ref_target.body_quat_w[:, [10, 13]])
                )
                palm_off = self._palm_local_offset_from_wrist.unsqueeze(0)
                palm_w = fk_next.tracked_pos_w[:, [10, 13]] + mat_vec(fk_next.tracked_rotmat_w[:, [10, 13]], palm_off)
                ref_palm_w = ref_target.body_pos_w[:, [10, 13]] + mat_vec(R_ref_wrists, palm_off)
                palm_in_obj = mat_t_vec(R_obj_w.unsqueeze(1), palm_w - obj_pos_w.unsqueeze(1))
                ref_palm_in_obj = mat_t_vec(R_ref_obj_w.unsqueeze(1), ref_palm_w - ref_target.obj_pos_w.unsqueeze(1))
                err_palm_p = ((palm_in_obj - ref_palm_in_obj) ** 2).sum(dim=-1).mean(dim=-1)
                err_palm_q = (rotmat_error_magnitude(o2b_R[:, [10, 13]], ref_o2b_R[:, [10, 13]]) ** 2).mean(dim=-1)
                unscaled_reward = unscaled_reward + c_label * (
                    0.55 * torch.exp(-err_palm_p / (0.07**2)) + 0.25 * torch.exp(-err_palm_q / (0.35**2))
                )
        step_reward = unscaled_reward * self.step_dt

        # ---- F. Smooth Sigmoid Soft Terminations (5 terms from BaseTerminationsCfg) ----
        # 1. bad_anchor_pos (threshold = 0.3m)
        d_anc_pos = torch.sqrt(err_anc_p + 1e-10)
        surv_anc_pos = torch.sigmoid((0.30 - d_anc_pos) / self.pos_term_temp)

        # 2. bad_anchor_ori (|proj_grav_z_ref - proj_grav_z_rob| > 0.8)
        if ref_target.anc_pg_z_ref is not None:
            pg_ref_z = ref_target.anc_pg_z_ref
        else:
            q_anc_n = quat_normalize(ref_anc_quat)
            pg_ref_z = 2.0 * (q_anc_n[:, 1] ** 2 + q_anc_n[:, 2] ** 2) - 1.0
        pg_rob_z = -fk_next.anchor_rotmat_w[:, 2, 2]
        d_anc_ori = torch.sqrt((pg_ref_z - pg_rob_z) ** 2 + 1e-10)
        surv_anc_ori = torch.sigmoid((0.80 - d_anc_ori) / self.ori_term_temp)

        # 3. ee_body_pos (any of 4 limbs relative error > 0.3m)
        ee_err = torch.sqrt(
            ((body_pos_rel_w[:, self.ee_indices] - fk_next.tracked_pos_w[:, self.ee_indices]) ** 2).sum(dim=-1)
            + 1e-10
        )  # (B, 4)
        surv_ee = torch.sigmoid((0.30 - ee_err) / self.pos_term_temp).prod(dim=-1)

        # 4. bad_obj_pos (threshold = 0.3m)
        d_obj_pos = torch.sqrt(err_op + 1e-10)
        surv_obj_pos = torch.sigmoid((0.30 - d_obj_pos) / self.pos_term_temp)

        # 5. bad_obj_ori (threshold = 0.8 rad)
        d_obj_ori = torch.sqrt(err_oq + 1e-10)
        surv_obj_ori = torch.sigmoid((0.80 - d_obj_ori) / self.ori_term_temp)

        p_survive = surv_anc_pos * surv_anc_ori * surv_ee * surv_obj_pos * surv_obj_ori
        p_term = (1.0 - p_survive).clamp(0.0, 1.0)

        info = {
            "rew/tracking_human": (r_joint_pos + r_anc_pos + r_anc_ori + r_body_pos + r_body_ori).detach(),
            "rew/tracking_obj": (r_obj_pos + r_obj_ori + r_obj_vel + r_obj_ang).detach(),
            "rew/interaction": (r_o2b_pos + r_o2b_ori + r_hoi_contact).detach(),
            "rew/feet_air_time": r_feet_air_time.detach(),
            "rew/reg": (r_act_rate + r_jlimit + r_jacc + r_jtorque + r_feet_slide + r_feet_air_time + r_undesired).detach(),
            "err/anchor_pos": d_anc_pos.detach(),
            "err/obj_pos": d_obj_pos.detach(),
            "term/p_term": p_term.detach(),
            "next_feet_air_time": next_feet_air_time,
            "curr_feet_gnd": curr_feet_gnd,
        }
        return step_reward, p_term, info

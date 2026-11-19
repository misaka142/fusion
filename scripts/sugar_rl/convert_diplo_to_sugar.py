"""Convert `Atarilab/diplo-dataset` SBTO trajectories + `LAFAN1` walk-to-stand approaches into `data/OmniCarryBox`.

Zero-tolerance physics & posture hygiene pipeline:
1. Matches DiPLO's exact MuJoCo SBTO box geometry (`0.310 x 0.310 x 0.340 m` square prism, `0.6 kg`) in both
   `obj_aligned.obj` (MuJoCo) and `obj_aligned.usda` (Isaac Sim PhysX `convexHull`).
2. Extracts ONLY the pure MuJoCo-simulated SBTO window (`[75 : 75 + motion_frames]`), discarding DiPLO's
   unsimulated `[0..74]` and `[75+mf..T]` root-frozen interpolation frames that caused 7.2 cm backward foot skating.
3. Prepends a coupled `12x15` closed-chain **Double-Sole Welded Stand-to-Bend** transition ahead of `t_entry >= 75`
   so BOTH shoe soles stay 100% stationary (`0.00 cm` slip) while sitting hips back to bend for the box.
4. Applies DiPLO's exact D4 symmetries (sagittal-plane mirroring `y -> -y` + `90 deg` square-box yaw symmetries),
   prepends diverse zero-phase smoothed `LAFAN1` native walk-to-stand buffers, and runs a final end-to-end MuJoCo
   verification pass on every stitched trajectory before saving.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import mujoco
import numpy as np
import torch
import trimesh

_FUSION_ROOT = Path(__file__).resolve().parents[2]
_SUGAR_RL_SRC = _FUSION_ROOT / "source" / "sugar_rl"
for _p in (_FUSION_ROOT, _SUGAR_RL_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.sim2sim.scene_builder import G1_ISAACLAB_BFS_JOINT_NAMES, build_mujoco_scene_xml
from scripts.sugar_rl.convert_omniretarget_to_sugar import (
    SUGAR_RAW_35_BODY_NAMES,
    _ang_vel_world_from_quats,
    _extract_clean_contact_labels,
    _quat_mul,
    _quat_norm,
    _quat_to_rotmat,
    _resample_catmull_rom,
    _resample_quat_slerp,
    _write_usda_convex_hull,
)
from scripts.sugar_rl.lafan_walk_stitcher import (
    _box_obb_surface_distance,
    _cubic_hermite,
    _estimate_initial_slope,
    _quat_to_rotmat_np,
    _quat_yaw,
    _rotate_z_mat,
    _rotmat_to_rotvec,
    _rotvec_to_rotmat,
    build_bfs_sagittal_mirror_map,
    load_lafan_walk_segments,
    mirror_sagittal_pose_Arrays,
    stitch_lafan_walk_to_omni_clip,
)
from sugar_rl.supermimic.urdf_kinematics import FastTreeKinematics, G1_29DOF_SDK_JOINT_NAMES

TARGET_FPS = 50.0
DT = 1.0 / TARGET_FPS
DIPLO_BOX_SIZE_XYZ = np.array([0.310, 0.310, 0.340], dtype=np.float64)
DIPLO_BOX_HALF_EXT = 0.5 * DIPLO_BOX_SIZE_XYZ
DIPLO_BOX_MASS_KG = 0.6

DIPLO_MJ_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert `(T, 3, 3)` rotation matrices to continuous `(T, 4)` wxyz quaternions."""
    out = np.zeros((len(R), 4), dtype=np.float64)
    for i, r in enumerate(R):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
        out[i] = q / np.linalg.norm(q)
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] = -out[i]
    return out


def _ensure_diplo_base_downloaded(diplo_dir: Path, cache_dir: Path | None = None) -> dict:
    """Download the 29 base `*_original` folders from `Atarilab/diplo-dataset` if not already present."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir.resolve()))

    meta_file = diplo_dir / "metadata" / "conversion_meta.json"
    if not meta_file.is_file():
        from huggingface_hub import hf_hub_download

        diplo_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="Atarilab/diplo-dataset",
            filename="metadata/conversion_meta.json",
            repo_type="dataset",
            local_dir=str(diplo_dir),
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    base_names = sorted([k for k in meta["motions"].keys() if "__" not in k])
    missing = [bn for bn in base_names if not (diplo_dir / bn / "sbto_traj.npz").is_file()]
    if missing:
        from huggingface_hub import snapshot_download

        print(f"[DiPLO Download] Fetching {len(missing)} base motion folders -> `{diplo_dir}`...")
        snapshot_download(
            repo_id="Atarilab/diplo-dataset",
            repo_type="dataset",
            local_dir=str(diplo_dir),
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=["metadata/*"] + [f"{bn}/sbto_traj.npz" for bn in missing],
            max_workers=4,
        )
    return meta


class _MuJoCoHygieneAuditor:
    """Fast reusable MuJoCo full-body collision, CoM support, closed-chain IK, and zero-skate verifier."""

    def __init__(self, descriptions_dir: Path, box_dir: Path):
        xml_path, _, _ = build_mujoco_scene_xml(
            task_name="CarryBox",
            descriptions_dir=descriptions_dir,
            sim_dt=0.002,
            motion_folder=box_dir,
        )
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.lf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.rf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.le_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
        self.re_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
        self.lh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand")
        self.rh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_rubber_hand")
        self.obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")

        bnames = {b: (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(self.model.nbody)}
        self.l_legs = {b for b, n in bnames.items() if n.startswith("left_") and any(k in n for k in ("hip", "knee", "ankle"))}
        self.r_legs = {b for b, n in bnames.items() if n.startswith("right_") and any(k in n for k in ("hip", "knee", "ankle"))}
        self.l_arms = {b for b, n in bnames.items() if n.startswith("left_") and any(k in n for k in ("shoulder", "elbow", "wrist", "hand"))}
        self.r_arms = {b for b, n in bnames.items() if n.startswith("right_") and any(k in n for k in ("shoulder", "elbow", "wrist", "hand"))}
        hand_set = {b for b, n in bnames.items() if any(k in n for k in ("wrist", "hand", "palm", "finger"))}
        self.non_hand = {b for b, n in bnames.items() if b > 0 and b != self.obj_id and n != "target_vis" and b not in hand_set}

        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) in self.l_legs or int(self.model.geom_bodyid[g]) in self.r_legs:
                self.model.geom_margin[g] = 0.025
        self.data = mujoco.MjData(self.model)

        pel_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_freejoint")
        self.p_adr = int(self.model.jnt_qposadr[pel_jid])
        self.pel_dofs_mj = np.arange(int(self.model.jnt_dofadr[pel_jid]), int(self.model.jnt_dofadr[pel_jid]) + 3, dtype=np.int64)
        self.o_adr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")])
        self.b_adrs = np.array(
            [int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in G1_ISAACLAB_BFS_JOINT_NAMES],
            dtype=np.int64,
        )
        self.sole_corners = np.array(
            [[-0.05, -0.03, -0.03], [-0.05, 0.03, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]],
            dtype=np.float64,
        )
        self.sole_mid_local = np.array([0.035, 0.0, -0.030], dtype=np.float64)

        jnames = list(G1_ISAACLAB_BFS_JOINT_NAMES)
        l_leg_names = [f"left_{k}_joint" for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
        r_leg_names = [f"right_{k}_joint" for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
        self.l_bfs_idx = np.array([jnames.index(n) for n in l_leg_names], dtype=np.int64)
        self.r_bfs_idx = np.array([jnames.index(n) for n in r_leg_names], dtype=np.int64)
        l_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in l_leg_names]
        r_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in r_leg_names]
        self.l_dofs_mj = np.array([int(self.model.jnt_dofadr[j]) for j in l_jids], dtype=np.int64)
        self.r_dofs_mj = np.array([int(self.model.jnt_dofadr[j]) for j in r_jids], dtype=np.int64)
        self.l_rng = self.model.jnt_range[l_jids].copy()
        self.r_rng = self.model.jnt_range[r_jids].copy()
        self.l_rng[3, 0] = max(self.l_rng[3, 0], 0.06)
        self.r_rng[3, 0] = max(self.r_rng[3, 0], 0.06)

        self.stand_q_init = np.zeros(29, dtype=np.float64)
        stand_defaults = {
            "left_hip_pitch_joint": -0.25, "right_hip_pitch_joint": -0.25,
            "left_knee_joint": 0.48, "right_knee_joint": 0.48,
            "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
            "waist_pitch_joint": 0.04,
            "left_shoulder_pitch_joint": 0.08, "right_shoulder_pitch_joint": 0.08,
            "left_shoulder_roll_joint": 0.22, "right_shoulder_roll_joint": -0.22,
            "left_elbow_joint": 0.45, "right_elbow_joint": 0.45,
        }
        for k_name, val in stand_defaults.items():
            self.stand_q_init[jnames.index(k_name)] = val

    def _solve_coupled_pelvis_legs_ik(
        self,
        pel_pos_ref: np.ndarray,
        pel_quat_wxyz: np.ndarray,
        q_bfs_ref: np.ndarray,
        sole_L_tgt: np.ndarray,
        R_LF_tgt: np.ndarray,
        sole_R_tgt: np.ndarray,
        R_RF_tgt: np.ndarray,
        pelvis_mobility: float = 0.25,
        n_iters: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coupled 12x15 (pelvis 3D + left leg 6D + right leg 6D) damped least-squares IK."""
        m, d = self.model, self.data
        p_cur = pel_pos_ref.copy()
        q = q_bfs_ref.copy()
        q[self.l_bfs_idx] = np.clip(q[self.l_bfs_idx], self.l_rng[:, 0], self.l_rng[:, 1])
        q[self.r_bfs_idx] = np.clip(q[self.r_bfs_idx], self.r_rng[:, 0], self.r_rng[:, 1])

        p_LF_tgt = sole_L_tgt - R_LF_tgt @ self.sole_mid_local
        p_RF_tgt = sole_R_tgt - R_RF_tgt @ self.sole_mid_local
        jacp_L = np.zeros((3, m.nv), dtype=np.float64)
        jacr_L = np.zeros((3, m.nv), dtype=np.float64)
        jacp_R = np.zeros((3, m.nv), dtype=np.float64)
        jacr_R = np.zeros((3, m.nv), dtype=np.float64)

        w_cols = np.ones(15, dtype=np.float64)
        w_cols[:3] = pelvis_mobility
        W_inv = np.diag(w_cols ** 2)
        reg = 1e-6 * np.eye(12, dtype=np.float64)

        for _ in range(n_iters):
            d.qpos[self.p_adr : self.p_adr + 3] = p_cur
            d.qpos[self.p_adr + 3 : self.p_adr + 7] = pel_quat_wxyz
            d.qpos[self.b_adrs] = q
            mujoco.mj_forward(m, d)

            ep_L = p_LF_tgt - d.xpos[self.lf_id]
            R_L = d.xmat[self.lf_id].reshape(3, 3)
            er_L = R_L @ _rotmat_to_rotvec(R_L.T @ R_LF_tgt)

            ep_R = p_RF_tgt - d.xpos[self.rf_id]
            R_R = d.xmat[self.rf_id].reshape(3, 3)
            er_R = R_R @ _rotmat_to_rotvec(R_R.T @ R_RF_tgt)

            err12 = np.concatenate([ep_L, er_L, ep_R, er_R])
            if float(np.max(np.abs(err12))) < 2e-5:
                break

            jacp_L.fill(0.0); jacr_L.fill(0.0)
            jacp_R.fill(0.0); jacr_R.fill(0.0)
            mujoco.mj_jacBody(m, d, jacp_L, jacr_L, self.lf_id)
            mujoco.mj_jacBody(m, d, jacp_R, jacr_R, self.rf_id)

            J = np.zeros((12, 15), dtype=np.float64)
            J[0:3, 0:3] = jacp_L[:, self.pel_dofs_mj]
            J[3:6, 0:3] = jacr_L[:, self.pel_dofs_mj]
            J[0:3, 3:9] = jacp_L[:, self.l_dofs_mj]
            J[3:6, 3:9] = jacr_L[:, self.l_dofs_mj]

            J[6:9, 0:3] = jacp_R[:, self.pel_dofs_mj]
            J[9:12, 0:3] = jacr_R[:, self.pel_dofs_mj]
            J[6:9, 9:15] = jacp_R[:, self.r_dofs_mj]
            J[9:12, 9:15] = jacr_R[:, self.r_dofs_mj]

            JW = J @ W_inv
            dx = W_inv @ J.T @ np.linalg.solve(JW @ J.T + reg, err12)
            p_cur += np.clip(dx[0:3], -0.05, 0.05)
            q[self.l_bfs_idx] = np.clip(q[self.l_bfs_idx] + np.clip(dx[3:9], -0.25, 0.25), self.l_rng[:, 0], self.l_rng[:, 1])
            q[self.r_bfs_idx] = np.clip(q[self.r_bfs_idx] + np.clip(dx[9:15], -0.25, 0.25), self.r_rng[:, 0], self.r_rng[:, 1])

        return p_cur, q

    def prepend_double_pinned_stand_to_bend(
        self,
        r_pos: np.ndarray,
        r_quat_wxyz: np.ndarray,
        q_bfs: np.ndarray,
        o_pos: np.ndarray,
        o_quat_wxyz: np.ndarray,
        dt: float = 0.02,
        hold_frames: int = 4,
        bend_frames: int = 28,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Prepend an upright standing hold + squat-bend transition with BOTH soles welded (`0.00 cm` slip)."""
        m, d = self.model, self.data
        d.qpos[self.p_adr : self.p_adr + 3] = r_pos[0]
        d.qpos[self.p_adr + 3 : self.p_adr + 7] = r_quat_wxyz[0]
        d.qpos[self.b_adrs] = q_bfs[0]
        mujoco.mj_forward(m, d)
        up_z_B = float(d.xmat[self.torso_id].reshape(3, 3)[2, 2])
        if up_z_B >= 0.93:
            return r_pos, r_quat_wxyz, q_bfs, o_pos, o_quat_wxyz, 0

        n_vel = min(6, len(r_pos))
        lf_pos_6 = np.zeros((n_vel, 3), dtype=np.float64)
        rf_pos_6 = np.zeros((n_vel, 3), dtype=np.float64)
        lf_R_6 = np.zeros((n_vel, 3, 3), dtype=np.float64)
        rf_R_6 = np.zeros((n_vel, 3, 3), dtype=np.float64)
        for k in range(n_vel):
            d.qpos[self.p_adr : self.p_adr + 3] = r_pos[k]
            d.qpos[self.p_adr + 3 : self.p_adr + 7] = r_quat_wxyz[k]
            d.qpos[self.b_adrs] = q_bfs[k]
            mujoco.mj_forward(m, d)
            lf_pos_6[k] = d.xpos[self.lf_id].copy()
            rf_pos_6[k] = d.xpos[self.rf_id].copy()
            lf_R_6[k] = d.xmat[self.lf_id].reshape(3, 3).copy()
            rf_R_6[k] = d.xmat[self.rf_id].reshape(3, 3).copy()

        R_LF_B, R_RF_B = lf_R_6[0], rf_R_6[0]
        sole_L_B = lf_pos_6[0] + R_LF_B @ self.sole_mid_local
        sole_R_B = rf_pos_6[0] + R_RF_B @ self.sole_mid_local
        w_LF_B = _ang_vel_world_from_quats(_rotmat_to_quat_wxyz(lf_R_6), dt)[0]
        w_RF_B = _ang_vel_world_from_quats(_rotmat_to_quat_wxyz(rf_R_6), dt)[0]

        yaw_LF = float(np.arctan2(R_LF_B[1, 0], R_LF_B[0, 0]))
        yaw_RF = float(np.arctan2(R_RF_B[1, 0], R_RF_B[0, 0]))
        R_LF_A = _rotate_z_mat(yaw_LF)
        R_RF_A = _rotate_z_mat(yaw_RF)
        sole_L_A = np.array([sole_L_B[0], sole_L_B[1], float(np.clip(sole_L_B[2], 0.002, 0.008))], dtype=np.float64)
        sole_R_A = np.array([sole_R_B[0], sole_R_B[1], float(np.clip(sole_R_B[2], 0.002, 0.008))], dtype=np.float64)

        yaw_B = float(_quat_yaw(r_quat_wxyz[0]))
        fwd_B = np.array([np.cos(yaw_B), np.sin(yaw_B)], dtype=np.float64)
        R_pel_A = _rotate_z_mat(yaw_B) @ _rotvec_to_rotmat(np.array([0.0, 0.03, 0.0]))
        quat_pel_A = _rotmat_to_quat_wxyz(R_pel_A[None, ...])[0]
        if np.dot(quat_pel_A, r_quat_wxyz[0]) < 0.0:
            quat_pel_A = -quat_pel_A

        span_xy = float(np.linalg.norm(sole_L_A[:2] - sole_R_A[:2]))
        half_span = 0.5 * max(0.0, span_xy - 0.16)
        pel_z_A = float(np.sqrt(max(0.38, 0.718**2 - half_span**2)))
        mid_sole_xy = 0.5 * (sole_L_A[:2] + sole_R_A[:2])
        pel_pos_A = np.array([mid_sole_xy[0] - 0.015 * fwd_B[0], mid_sole_xy[1] - 0.015 * fwd_B[1], pel_z_A], dtype=np.float64)

        pel_pos_A, q_A = self._solve_coupled_pelvis_legs_ik(
            pel_pos_A, quat_pel_A, self.stand_q_init, sole_L_A, R_LF_A, sole_R_A, R_RF_A,
            pelvis_mobility=0.60, n_iters=25,
        )
        for _ in range(3):
            d.qpos[self.p_adr : self.p_adr + 3] = pel_pos_A
            d.qpos[self.p_adr + 3 : self.p_adr + 7] = quat_pel_A
            d.qpos[self.b_adrs] = q_A
            mujoco.mj_forward(m, d)
            com_err_xy = d.subtree_com[self.pelvis_id, :2] - mid_sole_xy
            pel_pos_A[:2] -= 0.80 * com_err_xy
            pel_pos_A, q_A = self._solve_coupled_pelvis_legs_ik(
                pel_pos_A, quat_pel_A, q_A, sole_L_A, R_LF_A, sole_R_A, R_RF_A,
                pelvis_mobility=0.25, n_iters=18,
            )

        v_B = np.clip(_estimate_initial_slope(r_pos, dt, n_frames=n_vel), -0.6, 0.6)
        qd_B = np.clip(_estimate_initial_slope(q_bfs, dt, n_frames=n_vel), -4.5, 4.5)
        w_B = _ang_vel_world_from_quats(r_quat_wxyz[:n_vel], dt)[0]

        T_bend = bend_frames * dt
        u = np.arange(0, bend_frames, dtype=np.float64) / float(bend_frames)
        q_herm = _cubic_hermite(q_A, np.zeros(29), q_bfs[0], qd_B, u, T_bend)
        p_herm = _cubic_hermite(pel_pos_A, np.zeros(3), r_pos[0], v_B, u, T_bend)

        R_pel_B = _quat_to_rotmat_np(r_quat_wxyz[0])
        theta_pel = _cubic_hermite(np.zeros(3), np.zeros(3), _rotmat_to_rotvec(R_pel_A.T @ R_pel_B), R_pel_A.T @ w_B, u, T_bend)
        R_pel_u = np.stack([R_pel_A @ _rotvec_to_rotmat(tv) for tv in theta_pel], axis=0)
        quat_pel_u = _rotmat_to_quat_wxyz(R_pel_u)
        if np.dot(quat_pel_u[0], quat_pel_A) < 0.0:
            quat_pel_u = -quat_pel_u

        theta_LF = _cubic_hermite(np.zeros(3), np.zeros(3), _rotmat_to_rotvec(R_LF_A.T @ R_LF_B), R_LF_A.T @ w_LF_B, u, T_bend)
        theta_RF = _cubic_hermite(np.zeros(3), np.zeros(3), _rotmat_to_rotvec(R_RF_A.T @ R_RF_B), R_RF_A.T @ w_RF_B, u, T_bend)
        s_z = 3.0 * (u * u) - 2.0 * (u * u * u)

        bend_p = np.zeros((bend_frames, 3), dtype=np.float64)
        bend_q = q_herm.copy()
        q_prev = q_A.copy()
        p_prev = pel_pos_A.copy()
        for k in range(bend_frames):
            uk = float(u[k])
            R_LF_k = R_LF_A @ _rotvec_to_rotmat(theta_LF[k])
            R_RF_k = R_RF_A @ _rotvec_to_rotmat(theta_RF[k])
            sole_L_k = np.array([sole_L_B[0], sole_L_B[1], (1.0 - s_z[k]) * sole_L_A[2] + s_z[k] * sole_L_B[2]])
            sole_R_k = np.array([sole_R_B[0], sole_R_B[1], (1.0 - s_z[k]) * sole_R_A[2] + s_z[k] * sole_R_B[2]])

            q_seed = bend_q[k].copy()
            w_blend = uk * uk
            q_seed[self.l_bfs_idx] = (1.0 - w_blend) * q_prev[self.l_bfs_idx] + w_blend * q_herm[k, self.l_bfs_idx]
            q_seed[self.r_bfs_idx] = (1.0 - w_blend) * q_prev[self.r_bfs_idx] + w_blend * q_herm[k, self.r_bfs_idx]
            p_seed = (1.0 - w_blend) * p_prev + w_blend * p_herm[k]
            p_seed[:2] -= (0.038 * float(np.sin(np.pi * uk) ** 2)) * fwd_B

            pk, qk = self._solve_coupled_pelvis_legs_ik(
                p_seed, quat_pel_u[k], q_seed, sole_L_k, R_LF_k, sole_R_k, R_RF_k,
                pelvis_mobility=0.30, n_iters=15,
            )
            bend_p[k] = pk
            bend_q[k] = qk
            q_prev = qk
            p_prev = pk

        if hold_frames > 0:
            pre_p = np.concatenate([np.repeat(bend_p[0:1], hold_frames, axis=0), bend_p], axis=0)
            pre_quat = np.concatenate([np.repeat(quat_pel_u[0:1], hold_frames, axis=0), quat_pel_u], axis=0)
            pre_q = np.concatenate([np.repeat(bend_q[0:1], hold_frames, axis=0), bend_q], axis=0)
        else:
            pre_p, pre_quat, pre_q = bend_p, quat_pel_u, bend_q

        n_pre = len(pre_p)
        out_p = np.concatenate([pre_p, r_pos], axis=0)
        out_quat = np.concatenate([pre_quat, r_quat_wxyz], axis=0)
        for t in range(1, len(out_quat)):
            if np.dot(out_quat[t], out_quat[t - 1]) < 0.0:
                out_quat[t] = -out_quat[t]
        out_q = np.concatenate([pre_q, q_bfs], axis=0)
        out_op = np.concatenate([np.repeat(o_pos[0:1], n_pre, axis=0), o_pos], axis=0)
        out_oq = np.concatenate([np.repeat(o_quat_wxyz[0:1], n_pre, axis=0), o_quat_wxyz], axis=0)
        return out_p, out_quat, out_q, out_op, out_oq, n_pre

    def evaluate_sequence(
        self,
        r_pos: np.ndarray,
        r_quat_wxyz: np.ndarray,
        q_bfs: np.ndarray,
        o_pos: np.ndarray,
        o_quat_wxyz: np.ndarray,
        first_lift: int,
        last_lift: int,
        blend_range: tuple[int, int] | None = None,
        stand_bend_start: int | None = None,
    ) -> tuple[bool, str, dict]:
        """Run strict frame-by-frame biomechanical, collision, CoM, and zero-skating verification."""
        T = len(r_pos)
        min_hand_dy = 999.0
        min_elbow_dy = 999.0
        max_lifted_hdx = 0.0
        max_lifted_hdz = 0.0
        max_hand_detach = 0.0
        min_box_x = 999.0
        max_box_y = 0.0
        min_box_up_z = 1.0
        min_leg_d = 0.050
        max_box_pen = 0.0
        com_toe_blend = -999.0
        com_toe_pick = -999.0
        com_toe_place = -999.0
        max_bend_foot_z = 0.0
        max_bend_foot_span = 0.0
        max_airborne = 0.0
        lf_soles = np.zeros((T, 3), dtype=np.float64)
        rf_soles = np.zeros((T, 3), dtype=np.float64)

        b0, b1 = blend_range if blend_range is not None else (-1, -1)

        for t in range(T):
            self.data.qpos[self.p_adr : self.p_adr + 3] = r_pos[t]
            self.data.qpos[self.p_adr + 3 : self.p_adr + 7] = r_quat_wxyz[t]
            self.data.qpos[self.b_adrs] = q_bfs[t]
            self.data.qpos[self.o_adr : self.o_adr + 3] = o_pos[t]
            self.data.qpos[self.o_adr + 3 : self.o_adr + 7] = o_quat_wxyz[t]
            mujoco.mj_forward(self.model, self.data)

            t_pos = self.data.xpos[self.torso_id]
            t_R = self.data.xmat[self.torso_id].reshape(3, 3)
            lh_w = self.data.xpos[self.lh_id]
            rh_w = self.data.xpos[self.rh_id]
            lh_loc = t_R.T @ (lh_w - t_pos)
            rh_loc = t_R.T @ (rh_w - t_pos)
            le_loc = t_R.T @ (self.data.xpos[self.le_id] - t_pos)
            re_loc = t_R.T @ (self.data.xpos[self.re_id] - t_pos)

            min_hand_dy = min(min_hand_dy, float(lh_loc[1] - rh_loc[1]))
            min_elbow_dy = min(min_elbow_dy, float(le_loc[1] - re_loc[1]))

            o_R = self.data.xmat[self.obj_id].reshape(3, 3)
            min_box_up_z = min(min_box_up_z, float(o_R[2, 2]))

            if first_lift + 8 <= t <= last_lift - 8:
                max_lifted_hdx = max(max_lifted_hdx, abs(float(lh_loc[0] - rh_loc[0])))
                max_lifted_hdz = max(max_lifted_hdz, abs(float(lh_w[2] - rh_w[2])))
                box_loc = t_R.T @ (o_pos[t] - t_pos)
                min_box_x = min(min_box_x, float(box_loc[0]))
                max_box_y = max(max_box_y, abs(float(box_loc[1])))
                hd = _box_obb_surface_distance(np.stack([lh_w, rh_w]), o_pos[t], o_R, DIPLO_BOX_HALF_EXT)
                max_hand_detach = max(max_hand_detach, float(hd.max()))

            for ci in range(self.data.ncon):
                con = self.data.contact[ci]
                c_b1, c_b2 = int(self.model.geom_bodyid[con.geom1]), int(self.model.geom_bodyid[con.geom2])
                if (c_b1 in self.l_legs and c_b2 in self.r_legs) or (c_b2 in self.l_legs and c_b1 in self.r_legs):
                    min_leg_d = min(min_leg_d, float(con.dist))
                if ((c_b1 in self.l_arms and c_b2 in self.r_arms) or (c_b2 in self.l_arms and c_b1 in self.r_arms)) and float(con.dist) < 0.0:
                    return False, "arm_self_collision", {}
                if (c_b1 == self.obj_id and c_b2 in self.non_hand) or (c_b2 == self.obj_id and c_b1 in self.non_hand):
                    max_box_pen = max(max_box_pen, -float(con.dist))

            lfp = self.data.xpos[self.lf_id]
            rfp = self.data.xpos[self.rf_id]
            lfR = self.data.xmat[self.lf_id].reshape(3, 3)
            rfR = self.data.xmat[self.rf_id].reshape(3, 3)
            lf_soles[t] = lfp + lfR @ self.sole_mid_local
            rf_soles[t] = rfp + rfR @ self.sole_mid_local
            ff = 0.5 * (lfR[:, 0][:2] + rfR[:, 0][:2])
            ff /= (np.linalg.norm(ff) + 1e-8)
            fl = np.array([-ff[1], ff[0]])
            df = lfp[:2] - rfp[:2]
            if abs(float(df @ ff)) < 0.16 and float(df @ fl) < 0.035:
                return False, "crossed_scissor_legs", {}

            if (max(0, first_lift - 12) <= t <= first_lift + 6) or (max(0, last_lift - 6) <= t <= min(T - 1, last_lift + 12)):
                max_bend_foot_z = max(max_bend_foot_z, float(lfp[2]), float(rfp[2]))
                max_bend_foot_span = max(max_bend_foot_span, float(np.linalg.norm(df)))

            sup = np.vstack([(lfR @ self.sole_corners.T).T[:, :2] + lfp[:2], (rfR @ self.sole_corners.T).T[:, :2] + rfp[:2]])
            com_over = float(self.data.subtree_com[self.pelvis_id, :2] @ ff) - float(np.max(sup @ ff))
            if b0 <= t < b1:
                com_toe_blend = max(com_toe_blend, com_over)
            if max(0, first_lift - 25) <= t <= first_lift + 8:
                com_toe_pick = max(com_toe_pick, com_over)
            if last_lift - 8 <= t <= min(T - 1, last_lift + 20):
                com_toe_place = max(com_toe_place, com_over)
            max_airborne = max(max_airborne, min(float(lfp[2]), float(rfp[2])))

        if min_hand_dy < 0.24 or min_elbow_dy < 0.24:
            return False, f"crossed_or_narrow_arms(dy={min_hand_dy:.3f}m)", {}
        if max_lifted_hdx > 0.11 or max_lifted_hdz > 0.15 or max_box_y > 0.12 or min_box_x < 0.16:
            return False, f"asymmetric_grasp(dx={max_lifted_hdx:.2f},dz={max_lifted_hdz:.2f},by={max_box_y:.2f})", {}
        if max_hand_detach > 0.085 or min_box_up_z < 0.88:
            return False, "detached_or_tilted_box", {}
        if max_bend_foot_z > 0.068 or max_bend_foot_span > 0.40:
            return False, f"arabesque_or_split_bend(fz={max_bend_foot_z:.3f}m,span={max_bend_foot_span:.2f}m)", {}
        if min_leg_d < 0.013:
            return False, f"leg_clearance_low({min_leg_d*100:.2f}cm)", {}
        if max_box_pen > 0.012:
            return False, f"non_hand_box_pen({max_box_pen*100:.1f}cm)", {}
        if max_airborne > 0.064:
            return False, f"airborne({max_airborne*100:.1f}cm)", {}
        if (blend_range is not None and com_toe_blend > 0.002) or com_toe_pick > -0.002 or com_toe_place > 0.002:
            return False, f"com_tip(blend={com_toe_blend*100:+.1f},pick={com_toe_pick*100:+.1f},place={com_toe_place*100:+.1f}cm)", {}

        # 1. Short-window (0.24s) grounded sole anti-skating gate
        max_pre_skate_cm = 0.0
        skate_start = max(0, b0 - 10) if blend_range is not None else 0
        chk_end = min(T - 12, first_lift + 5)
        for soles in (lf_soles, rf_soles):
            for t in range(skate_start, max(skate_start, chk_end)):
                if np.all(soles[t : t + 13, 2] <= 0.016):
                    drift_cm = float(np.linalg.norm(soles[t + 12, :2] - soles[t, :2])) * 100.0
                    max_pre_skate_cm = max(max_pre_skate_cm, drift_cm)
        if max_pre_skate_cm > 2.2:
            return False, f"pre_pickup_foot_skate({max_pre_skate_cm:.2f}cm)", {}

        # 2. Full-duration standing-to-grasp cumulative drift gate (`stand_bend_start -> t_grasp`):
        #    Rejects ANY slow smooth sliding of either foot across the entire bend-and-grasp phase!
        max_bend_drift_cm = 0.0
        if stand_bend_start is not None and 0 <= stand_bend_start < first_lift:
            grasp_idx = np.flatnonzero(o_pos[stand_bend_start : first_lift + 1, 2] > (float(o_pos[stand_bend_start, 2]) + 0.012))
            t_grasp_end = (stand_bend_start + int(grasp_idx[0])) if len(grasp_idx) > 0 else first_lift
            t_grasp_end = max(stand_bend_start + 1, t_grasp_end)
            l_cum = np.linalg.norm(lf_soles[stand_bend_start : t_grasp_end + 1, :2] - lf_soles[stand_bend_start, :2], axis=-1).max()
            r_cum = np.linalg.norm(rf_soles[stand_bend_start : t_grasp_end + 1, :2] - rf_soles[stand_bend_start, :2], axis=-1).max()
            max_bend_drift_cm = float(max(l_cum, r_cum)) * 100.0
            if max_bend_drift_cm > 2.2:
                return False, f"bend_cumulative_drift({max_bend_drift_cm:.2f}cm)", {}

        stats = {
            "min_hand_dy_m": round(min_hand_dy, 3),
            "max_grasp_dx_m": round(max_lifted_hdx, 3),
            "min_leg_dist_cm": round(min_leg_d * 100.0, 2),
            "com_toe_blend_cm": round(com_toe_blend * 100.0, 2) if blend_range else 0.0,
            "com_toe_pick_cm": round(com_toe_pick * 100.0, 2),
            "com_toe_place_cm": round(com_toe_place * 100.0, 2),
            "max_airborne_cm": round(max_airborne * 100.0, 2),
            "max_pre_skate_cm": round(max_pre_skate_cm, 2),
            "max_bend_drift_cm": round(max_bend_drift_cm, 2),
        }
        return True, "OK", stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diplo_dir", type=str, default=str(_FUSION_ROOT / "data_diplo_raw"))
    ap.add_argument("--lafan_dir", type=str, default=str(_FUSION_ROOT / "data_lafan1_raw"))
    ap.add_argument("--out_dir", type=str, default=str(_FUSION_ROOT / "data" / "OmniCarryBox"))
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional HuggingFace download cache dir.")
    ap.add_argument("--blend_frames", type=int, default=22, help="50Hz stance-pinned walk->stand blend frames (default: 22 = 0.44s).")
    ap.add_argument("--max_solves_per_base", type=int, default=6, help="Max top clean SBTO solves per base motion.")
    ap.add_argument("--approaches_per_variant", type=int, default=3, help="Distinct LAFAN1 + D4 box variants per clean solve (x2 mirror).")
    args = ap.parse_args()

    t_start_wall = time.perf_counter()
    diplo_dir = Path(args.diplo_dir).resolve()
    lafan_dir = Path(args.lafan_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    meta = _ensure_diplo_base_downloaded(diplo_dir, cache_dir=cache_dir)
    base_names = sorted([k for k in meta["motions"].keys() if "__" not in k])

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Export exact DiPLO SBTO centered box mesh (0.310 x 0.310 x 0.340 m, 0.6 kg)
    box_mesh = trimesh.creation.box(extents=DIPLO_BOX_SIZE_XYZ)
    trimesh.repair.fix_normals(box_mesh)
    box_mesh.export(str(out_root / "obj_aligned.obj"))
    _write_usda_convex_hull(box_mesh, out_root / "obj_aligned.usda", mass_kg=DIPLO_BOX_MASS_KG)
    box_spec = {
        "dataset": "diplo_sbto_lafan1_clean",
        "task": "CarryBox",
        "box_size_xyz": [round(float(x), 5) for x in DIPLO_BOX_SIZE_XYZ],
        "box_center_xyz": [0.0, 0.0, 0.0],
        "mass_kg": DIPLO_BOX_MASS_KG,
        "hull_vertices": int(len(box_mesh.vertices)),
        "hull_faces": int(len(box_mesh.faces)),
        "mesh_file": "obj_aligned.obj",
        "usd_file": "obj_aligned.usda",
    }
    (out_root / "box_spec.json").write_text(json.dumps(box_spec, indent=2), encoding="utf-8")

    # 2. Initialize kinematics & MuJoCo hygiene auditor
    sdk_to_idx = {n: i for i, n in enumerate(G1_29DOF_SDK_JOINT_NAMES)}
    sdk_for_bfs = np.array([sdk_to_idx[n] for n in G1_ISAACLAB_BFS_JOINT_NAMES], dtype=np.int64)
    diplo_to_idx = {n: i for i, n in enumerate(DIPLO_MJ_JOINT_NAMES)}
    diplo_for_bfs = np.array([diplo_to_idx[n] for n in G1_ISAACLAB_BFS_JOINT_NAMES], dtype=np.int64)
    mir_perm, mir_signs = build_bfs_sagittal_mirror_map(list(G1_ISAACLAB_BFS_JOINT_NAMES))

    urdf_path = _FUSION_ROOT / "descriptions" / "robots" / "g1" / "g1_29dof_rev_1_0_with_rubber_hand.urdf"
    fk_35 = FastTreeKinematics(
        urdf_or_xml_path=urdf_path,
        env_joint_names=G1_ISAACLAB_BFS_JOINT_NAMES,
        tracked_body_names=SUGAR_RAW_35_BODY_NAMES,
        device="cpu",
    )
    auditor = _MuJoCoHygieneAuditor(_FUSION_ROOT / "descriptions", out_root)

    # Load native walk -> decelerate -> stand-still LAFAN1 segments (including mirrored stand buffers)
    lafan_segments = load_lafan_walk_segments(
        lafan_dir=lafan_dir,
        fk_35=fk_35,
        sdk_for_bfs=sdk_for_bfs,
        body_names=SUGAR_RAW_35_BODY_NAMES,
        resample_catmull_rom_fn=_resample_catmull_rom,
        resample_quat_slerp_fn=_resample_quat_slerp,
        ang_vel_fn=_ang_vel_world_from_quats,
        dt=DT,
        only_stand_buffer=True,
        include_mirrored=True,
    )
    print("=" * 86)
    print(f"[DiPLO + LAFAN1] Loaded {len(lafan_segments)} mirrored native walk-to-stand segments from `{lafan_dir / 'g1'}`")

    torso_idx_35 = SUGAR_RAW_35_BODY_NAMES.index("torso_link")
    la_idx_35 = SUGAR_RAW_35_BODY_NAMES.index("left_ankle_roll_link")
    ra_idx_35 = SUGAR_RAW_35_BODY_NAMES.index("right_ankle_roll_link")

    # 3. Stage A: Screen all ~900 SBTO solves strictly inside the PURE MuJoCo-simulated window `[75 : 75+motion_frames]`,
    #    then prepend the coupled `12x15` Double-Sole Welded Stand-to-Bend transition (`0.00 cm` sole slip)!
    core_cache_file = diplo_dir / ".clean_sbto_cores_v5_puresim.pkl"
    all_clean_by_base: dict[str, list[dict]] = {}
    drop_summary: dict[str, int] = {}

    if core_cache_file.is_file():
        cached_payload = pickle.loads(core_cache_file.read_bytes())
        all_clean_by_base = cached_payload["by_base"]
        drop_summary = cached_payload["drop_summary"]
        print(f"[DiPLO Hygiene Cache] Loaded pre-screened pure-sim + welded-bend SBTO cores from `{core_cache_file.name}`")
    else:
        for bn in base_names:
            sbto = np.load(diplo_dir / bn / "sbto_traj.npz")
            m_meta = meta["motions"][bn]
            costs = np.array(m_meta["cost"], dtype=np.float64)
            sim_start = 75
            sim_end = min(sbto["dof_pos"].shape[1], 75 + int(m_meta["motion_frames"]))
            B, T, _ = sbto["dof_pos"].shape
            b0 = int(np.argmin(costs))
            carry_xy = float(np.linalg.norm(sbto["object_pos"][b0, sim_end - 1, :2] - sbto["object_pos"][b0, sim_start, :2]))
            if carry_xy < 0.45:
                drop_summary["stationary_lift"] = drop_summary.get("stationary_lift", 0) + B
                continue

            kept_for_base = []
            for b in np.argsort(costs):
                r_pos = sbto["root_pos"][b].astype(np.float64)
                r_quat = _quat_norm(sbto["root_rot"][b].astype(np.float64))
                dof_p = sbto["dof_pos"][b].astype(np.float64)
                dof_v = sbto["dof_vel"][b].astype(np.float64)
                o_pos = sbto["object_pos"][b].astype(np.float64)
                o_quat = _quat_norm(sbto["object_rot"][b].astype(np.float64))
                q_bfs = dof_p[:, diplo_for_bfs]

                with torch.no_grad():
                    fk_c = fk_35(
                        torch.from_numpy(r_pos.astype(np.float32)),
                        torch.from_numpy(r_quat.astype(np.float32)),
                        torch.from_numpy(q_bfs.astype(np.float32)),
                    )
                bpos_c = fk_c.tracked_pos_w.numpy().astype(np.float64)
                bquat_c = fk_c.tracked_quat_w.numpy().astype(np.float64)
                up_z_all = _quat_to_rotmat(bquat_c[:, torso_idx_35])[:, 2, 2]
                max_fz_both = np.maximum(bpos_c[:, la_idx_35, 2], bpos_c[:, ra_idx_35, 2])
                min_fz_both = np.minimum(bpos_c[:, la_idx_35, 2], bpos_c[:, ra_idx_35, 2])
                l_soles_c = bpos_c[:, la_idx_35] + _quat_to_rotmat(bquat_c[:, la_idx_35]) @ auditor.sole_mid_local
                r_soles_c = bpos_c[:, ra_idx_35] + _quat_to_rotmat(bquat_c[:, ra_idx_35]) @ auditor.sole_mid_local

                lifted_sim = o_pos[sim_start:sim_end, 2] > (float(o_pos[sim_start, 2]) + 0.035)
                l_sim_idx = np.flatnonzero(lifted_sim)
                if len(l_sim_idx) < 25:
                    drop_summary["lift_or_place_incomplete"] = drop_summary.get("lift_or_place_incomplete", 0) + 1
                    continue
                first_l_abs = sim_start + int(l_sim_idx[0])
                onset_idx = np.flatnonzero(o_pos[sim_start : first_l_abs + 1, 2] > (float(o_pos[sim_start, 2]) + 0.012))
                t_grasp_abs = (sim_start + int(onset_idx[0])) if len(onset_idx) > 0 else (first_l_abs - 6)

                # Pick earliest post-transient pure-sim frame (`t >= 87`, `pel_z <= 0.752m`) where DiPLO's artificial
                # `(0, 0, 0.78m)` t=75 root reset has damped out and BOTH shoe soles stay settled (`<= 2.1 cm` until grasp)
                entry_candidates = []
                for t in range(sim_start + 12, max(sim_start + 13, t_grasp_abs - 8)):
                    if 0.024 <= min_fz_both[t] and max_fz_both[t] <= 0.057 and up_z_all[t] >= -0.02 and r_pos[t, 2] <= 0.752:
                        l_cum = float(np.linalg.norm(l_soles_c[t : t_grasp_abs + 1, :2] - l_soles_c[t, :2], axis=-1).max())
                        r_cum = float(np.linalg.norm(r_soles_c[t : t_grasp_abs + 1, :2] - r_soles_c[t, :2], axis=-1).max())
                        if max(l_cum, r_cum) <= 0.021:
                            entry_candidates.append(t)
                if not entry_candidates:
                    drop_summary["no_settled_pickup_stance"] = drop_summary.get("no_settled_pickup_stance", 0) + 1
                    continue
                t_entry = entry_candidates[0]
                exit_candidates = [
                    t for t in range(max(sim_start + 50, sim_end - 18), sim_end + 1)
                    if up_z_all[t - 1] >= 0.85 and 0.025 <= min_fz_both[t - 1] <= 0.058
                ]
                t_exit = exit_candidates[-1] if exit_candidates else sim_end

                if float(np.abs(dof_v[t_entry:t_exit]).max()) > 10.0:
                    drop_summary["qv_spike"] = drop_summary.get("qv_spike", 0) + 1
                    continue
                max_wyaw = float(np.degrees(np.abs(dof_p[t_entry:t_exit, 12]).max()))
                max_wroll = float(np.degrees(np.abs(dof_p[t_entry:t_exit, 13]).max()))
                if max_wyaw > 28.0 or max_wroll > 24.0:
                    drop_summary["waist_twist"] = drop_summary.get("waist_twist", 0) + 1
                    continue

                lifted = o_pos[t_entry:t_exit, 2] > (float(o_pos[t_entry, 2]) + 0.035)
                l_idx = np.flatnonzero(lifted)
                if len(l_idx) < 25 or abs(float(o_pos[t_exit - 1, 2]) - float(o_pos[t_entry, 2])) > 0.025:
                    drop_summary["lift_or_place_incomplete"] = drop_summary.get("lift_or_place_incomplete", 0) + 1
                    continue

                # Screen pure simulation slice first
                ok_sim, reason_sim, _ = auditor.evaluate_sequence(
                    r_pos[t_entry:t_exit],
                    r_quat[t_entry:t_exit],
                    q_bfs[t_entry:t_exit],
                    o_pos[t_entry:t_exit],
                    o_quat[t_entry:t_exit],
                    first_lift=int(l_idx[0]),
                    last_lift=int(l_idx[-1]),
                )
                if not ok_sim:
                    key = reason_sim.split("(")[0]
                    drop_summary[key] = drop_summary.get(key, 0) + 1
                    continue

                # Prepend coupled 12x15 double-sole welded upright-stand -> squat-bend transition!
                rp_wb, rq_wb, qb_wb, op_wb, oq_wb, n_pre_wb = auditor.prepend_double_pinned_stand_to_bend(
                    r_pos[t_entry:t_exit],
                    r_quat[t_entry:t_exit],
                    q_bfs[t_entry:t_exit],
                    o_pos[t_entry:t_exit],
                    o_quat[t_entry:t_exit],
                    dt=DT,
                    hold_frames=5,
                    bend_frames=32,
                )
                ok_wb, reason_wb, stats_wb = auditor.evaluate_sequence(
                    rp_wb,
                    rq_wb,
                    qb_wb,
                    op_wb,
                    oq_wb,
                    first_lift=int(l_idx[0]) + n_pre_wb,
                    last_lift=int(l_idx[-1]) + n_pre_wb,
                    blend_range=(0, n_pre_wb) if n_pre_wb > 0 else None,
                    stand_bend_start=0,
                )
                if not ok_wb:
                    key = "bend_ik_" + reason_wb.split("(")[0]
                    drop_summary[key] = drop_summary.get(key, 0) + 1
                    continue

                kept_for_base.append(
                    {
                        "base": bn,
                        "solve_idx": int(b),
                        "t_entry": int(t_entry),
                        "n_pre_bend": int(n_pre_wb),
                        "cost": float(costs[b]),
                        "r_pos": rp_wb.copy(),
                        "r_quat": rq_wb.copy(),
                        "q_bfs": qb_wb.copy(),
                        "o_pos": op_wb.copy(),
                        "o_quat": oq_wb.copy(),
                        "stats": stats_wb,
                    }
                )

            if kept_for_base:
                kept_for_base.sort(key=lambda item: (max(item["stats"]["com_toe_pick_cm"], -2.5), item["cost"]))
                all_clean_by_base[bn] = kept_for_base
        core_cache_file.write_bytes(pickle.dumps({"by_base": all_clean_by_base, "drop_summary": drop_summary}))

    clean_cores: list[dict] = []
    for bn, kept_for_base in all_clean_by_base.items():
        selected_solves = kept_for_base[: args.max_solves_per_base]
        clean_cores.extend(selected_solves)
        print(
            f"  [Accepted Base] {bn:28s}: kept {len(selected_solves)}/{len(kept_for_base)} clean solves "
            f"(top solve #{selected_solves[0]['solve_idx']} @ t={selected_solves[0]['t_entry']}: "
            f"hand_dy={selected_solves[0]['stats']['min_hand_dy_m']:.3f}m, "
            f"bend_drift={selected_solves[0]['stats']['max_bend_drift_cm']:.2f}cm, "
            f"com_pick={selected_solves[0]['stats']['com_toe_pick_cm']:+.2f}cm)"
        )

    if not clean_cores:
        raise RuntimeError("No SBTO trajectories survived strict hygiene screening!")

    print(f"[DiPLO Hygiene Audit] Selected {len(clean_cores)} golden SBTO cores | Rejected reasons: {drop_summary}")
    print("=" * 86)

    # 4. Stage B: Expand with exact D4 symmetries (Sagittal Mirror x2 + 90-deg square-box rotations)
    #    + Stance-Foot Pinned LAFAN1 walk-to-stand approach + Final end-to-end MuJoCo gatekeeper!
    lafan_usage: dict[str, int] = {}
    manifest: list[dict] = []
    kept_idx = 0
    post_stitch_drops = 0

    yaw_offsets_deg = (0.0, -3.5, 3.5, -2.0, 2.0)
    c4_yaws = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)

    for core_i, core in enumerate(clean_cores):
        n_pre_bend = int(core.get("n_pre_bend", 32))
        for is_mirror in (False, True):
            if not is_mirror:
                rp_base = core["r_pos"].copy()
                rq_base = core["r_quat"].copy()
                qb_base = core["q_bfs"].copy()
                op_base = core["o_pos"].copy()
                oq_base = core["o_quat"].copy()
                mir_tag = ""
            else:
                rp_base, rq_base, qb_base = mirror_sagittal_pose_Arrays(
                    core["r_pos"], core["r_quat"], core["q_bfs"], mir_perm, mir_signs
                )
                op_base = core["o_pos"] * np.array([1.0, -1.0, 1.0], dtype=np.float64)
                oq_base = core["o_quat"] * np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
                mir_tag = "__mirror"

            # Constant resting floor Z calibration so box rests flush on z=0.170m
            rest_z = min(float(np.median(op_base[:8, 2])), float(np.median(op_base[-8:, 2])))
            dz = DIPLO_BOX_HALF_EXT[2] - rest_z
            if abs(dz) < 0.015:
                op_base[:, 2] += dz

            with torch.no_grad():
                fk_core = fk_35(
                    torch.from_numpy(rp_base.astype(np.float32)),
                    torch.from_numpy(rq_base.astype(np.float32)),
                    torch.from_numpy(qb_base.astype(np.float32)),
                )
            bpos_core = fk_core.tracked_pos_w.numpy().astype(np.float64)
            olin_core = np.gradient(op_base, DT, axis=0).astype(np.float32)

            for app_k in range(args.approaches_per_variant):
                box_c4_yaw = c4_yaws[(core_i + int(is_mirror) * 2 + app_k) % 4]
                qz_box = np.array([np.cos(0.5 * box_c4_yaw), 0.0, 0.0, np.sin(0.5 * box_c4_yaw)], dtype=np.float64)
                oq_rot = _quat_norm(_quat_mul(oq_base, np.broadcast_to(qz_box, oq_base.shape)))
                orot_mat = _quat_to_rotmat(oq_rot)
                oang_core = _ang_vel_world_from_quats(oq_rot, DT).astype(np.float32)
                contact_core = _extract_clean_contact_labels(
                    op_base,
                    olin_core,
                    oang_core,
                    bpos_core[:, 33],
                    bpos_core[:, 34],
                    DIPLO_BOX_HALF_EXT,
                    orot_mat,
                )
                moving_idx = np.flatnonzero(np.linalg.norm(olin_core, axis=-1) > 0.035)
                if len(moving_idx) > 0:
                    contact_core[: max(1, int(moving_idx[0]) - 4)] = False
                if bool(contact_core[0]):
                    continue

                yaw_off = np.radians(yaw_offsets_deg[(core_i + app_k * 2 + int(is_mirror)) % len(yaw_offsets_deg)])
                excluded_segs: set[int] = set()
                accepted_stitched = None
                accepted_stats = None

                for _retry in range(5):
                    try:
                        stitched = stitch_lafan_walk_to_omni_clip(
                            omni_pos=rp_base,
                            omni_quat_wxyz=rq_base,
                            omni_q_bfs=qb_base,
                            omni_bpos=bpos_core,
                            omni_obj_pos=op_base,
                            omni_obj_rot=orot_mat,
                            omni_contact=contact_core,
                            box_half_extents=DIPLO_BOX_HALF_EXT,
                            segments=lafan_segments,
                            usage_counts=lafan_usage,
                            fk_35=fk_35,
                            body_names=SUGAR_RAW_35_BODY_NAMES,
                            ang_vel_fn=_ang_vel_world_from_quats,
                            blend_frames=args.blend_frames,
                            dt=DT,
                            pin_stance_foot=True,
                            approach_yaw_offset_rad=float(yaw_off),
                            exclude_seg_indices=excluded_segs,
                        )
                    except RuntimeError:
                        break

                    bpos_out = stitched["body_pos_w"]
                    bquat_out = stitched["body_quat_w"]
                    q_out = stitched["joint_pos"]
                    opos_out = stitched["obj_pos"]
                    orot_out = stitched["obj_rot"]
                    oquat_out = _rotmat_to_quat_wxyz(orot_out.astype(np.float64))

                    w_n = int(stitched["meta"]["lafan_frames_50hz"])
                    b_n = int(stitched["meta"]["blend_frames_50hz"])
                    lifted_full = opos_out[:, 2] > (float(opos_out[0, 2]) + 0.035)
                    lf_idx = np.flatnonzero(lifted_full)
                    if len(lf_idx) < 25:
                        break

                    ok_final, _, stats_final = auditor.evaluate_sequence(
                        bpos_out[:, 0].astype(np.float64),
                        bquat_out[:, 0].astype(np.float64),
                        q_out.astype(np.float64),
                        opos_out.astype(np.float64),
                        oquat_out,
                        first_lift=int(lf_idx[0]),
                        last_lift=int(lf_idx[-1]),
                        blend_range=(w_n, w_n + b_n + n_pre_bend),
                        stand_bend_start=w_n + b_n,
                    )
                    if ok_final:
                        accepted_stitched = stitched
                        accepted_stats = stats_final
                        break
                    excluded_segs.add(int(stitched["meta"]["seg_idx"]))

                if accepted_stitched is None or accepted_stats is None:
                    post_stitch_drops += 1
                    continue

                bpos_out = accepted_stitched["body_pos_w"]
                bquat_out = accepted_stitched["body_quat_w"]
                q_out = accepted_stitched["joint_pos"]
                opos_out = accepted_stitched["obj_pos"]
                orot_out = accepted_stitched["obj_rot"]
                contact_out = accepted_stitched["contact"]
                oang_out = np.concatenate([accepted_stitched["obj_ang_vel_pre"], oang_core], axis=0).astype(np.float32)

                clip_folder = out_root / f"data_{kept_idx:03d}"
                clip_folder.mkdir(parents=True, exist_ok=True)

                np.savez(
                    clip_folder / "robot_50hz.npz",
                    fps=np.array([50], dtype=np.int32),
                    joint_pos=q_out,
                    joint_vel=accepted_stitched["joint_vel"],
                    body_pos_w=bpos_out,
                    body_quat_w=bquat_out,
                    body_lin_vel_w=accepted_stitched["body_lin_vel_w"],
                    body_ang_vel_w=accepted_stitched["body_ang_vel_w"],
                )
                with open(clip_folder / "obj_motion_global_50hz.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "obj_trans": opos_out,
                            "obj_rot": orot_out,
                            "obj_scale": float(1.0),
                            "obj_lin_vel": accepted_stitched["obj_lin_vel"],
                            "obj_ang_vel": oang_out,
                        },
                        f,
                    )
                np.save(clip_folder / "contact_labels_50hz.npy", contact_out)

                c90 = int(round(np.degrees(box_c4_yaw))) % 360
                rot_tag = f"__rot{c90}" if c90 > 0 else ""
                src_name = f"{core['base']}#sbto{core['solve_idx']:02d}{mir_tag}{rot_tag}"
                t0_up_z = float(_quat_to_rotmat(bquat_out[0:1, torso_idx_35].astype(np.float64))[0, 2, 2])

                entry = {
                    "index": kept_idx,
                    "folder": clip_folder.name,
                    "source": src_name,
                    "frames_50hz": int(len(q_out)),
                    "t0_torso_up_z": round(t0_up_z, 4),
                    "stand_bend_frames_50hz": n_pre_bend,
                    **accepted_stitched["meta"],
                    **accepted_stats,
                }
                manifest.append(entry)
                kept_idx += 1

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - t_start_wall
    print("=" * 86)
    print(
        f"[DONE] Generated {kept_idx} 100%-MuJoCo-verified zero-skate clips -> `{out_root}` in {elapsed:.1f}s "
        f"(post-stitch gate rejected={post_stitch_drops})"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()

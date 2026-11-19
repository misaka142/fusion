"""Convert `Atarilab/diplo-dataset` SBTO trajectories + `LAFAN1` walk-to-stand approaches into `data/OmniCarryBox`.

Zero-tolerance physics & posture hygiene pipeline:
1. Matches DiPLO's exact MuJoCo SBTO box geometry (`0.310 x 0.310 x 0.340 m` square prism, `0.6 kg`) in both
   `obj_aligned.obj` (MuJoCo) and `obj_aligned.usda` (Isaac Sim PhysX `convexHull`).
2. Screens all ~900 SBTO trajectories across the 29 base folders in MuJoCo, strictly rejecting:
   - Stationary lift-in-place clips (`carry_disp < 0.45m`);
   - Crossed arms, narrow/one-sided hands, or twisted diagonal grasps (`y_L - y_R < 0.25m` or `|x_L - x_R| > 0.11m`);
   - Single-leg raised "arabesque" pickup/placement or wide split lunges (`max(lf_z, rf_z) > 0.068m` or foot span `> 0.40m` at bend);
   - Leg-vs-leg self-collision (`< 1.3 cm`), arm self-collision, or scissor cross-stepping;
   - Forward whole-body CoM tipping past grounded toes (`mj_data.subtree_com`) or airborne feet.
3. Applies DiPLO's exact D4 symmetries (sagittal-plane mirroring `y -> -y` + `90 deg` square-box yaw symmetries)
   and prepends diverse zero-phase smoothed `LAFAN1` native walk-to-stand buffers via **Stance-Foot Pinned Blending**
   (zero foot skating and zero stiff-knee forward lean across the seam).
4. Runs a final end-to-end MuJoCo verification pass on every stitched trajectory before saving.
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
    """Fast reusable MuJoCo full-body collision, CoM support-polygon, and symmetric-grasp verifier."""

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

        self.p_adr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_freejoint")])
        self.o_adr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")])
        self.b_adrs = np.array(
            [int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in G1_ISAACLAB_BFS_JOINT_NAMES],
            dtype=np.int64,
        )
        self.sole_corners = np.array(
            [[-0.05, -0.03, -0.03], [-0.05, 0.03, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]],
            dtype=np.float64,
        )

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
    ) -> tuple[bool, str, dict]:
        """Run strict frame-by-frame biomechanical & collision verification."""
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
            ff = 0.5 * (lfR[:, 0][:2] + rfR[:, 0][:2])
            ff /= (np.linalg.norm(ff) + 1e-8)
            fl = np.array([-ff[1], ff[0]])
            df = lfp[:2] - rfp[:2]
            if abs(float(df @ ff)) < 0.16 and float(df @ fl) < 0.035:
                return False, "crossed_scissor_legs", {}

            # Enforce grounded two-foot stance (no single-leg arabesque kick or deep split lunge!) at pickup & placement!
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
        if com_toe_blend > -0.010 or com_toe_pick > -0.002 or com_toe_place > 0.002:
            return False, f"com_tip(blend={com_toe_blend*100:+.1f},pick={com_toe_pick*100:+.1f},place={com_toe_place*100:+.1f}cm)", {}

        stats = {
            "min_hand_dy_m": round(min_hand_dy, 3),
            "max_grasp_dx_m": round(max_lifted_hdx, 3),
            "min_leg_dist_cm": round(min_leg_d * 100.0, 2),
            "com_toe_blend_cm": round(com_toe_blend * 100.0, 2) if blend_range else 0.0,
            "com_toe_pick_cm": round(com_toe_pick * 100.0, 2),
            "com_toe_place_cm": round(com_toe_place * 100.0, 2),
            "max_airborne_cm": round(max_airborne * 100.0, 2),
        }
        return True, "OK", stats


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diplo_dir", type=str, default=str(_FUSION_ROOT / "data_diplo_raw"))
    ap.add_argument("--lafan_dir", type=str, default=str(_FUSION_ROOT / "data_lafan1_raw"))
    ap.add_argument("--out_dir", type=str, default=str(_FUSION_ROOT / "data" / "OmniCarryBox"))
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional HuggingFace download cache dir.")
    ap.add_argument("--blend_frames", type=int, default=25, help="50Hz stance-pinned blend frames (default: 25 = 0.50s).")
    ap.add_argument("--max_solves_per_base", type=int, default=7, help="Max top clean SBTO solves per base motion.")
    ap.add_argument("--approaches_per_variant", type=int, default=2, help="Distinct LAFAN1 + D4 box variants per clean solve (x2 mirror).")
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

    # 3. Stage A: Screen all ~900 SBTO solves across the 29 base motions with zero tolerance
    core_cache_file = diplo_dir / ".clean_sbto_cores_v1.pkl"
    all_clean_by_base: dict[str, list[dict]] = {}
    drop_summary: dict[str, int] = {}

    if core_cache_file.is_file():
        cached_payload = pickle.loads(core_cache_file.read_bytes())
        all_clean_by_base = cached_payload["by_base"]
        drop_summary = cached_payload["drop_summary"]
        print(f"[DiPLO Hygiene Cache] Loaded pre-screened SBTO cores from `{core_cache_file.name}`")
    else:
        for bn in base_names:
            sbto = np.load(diplo_dir / bn / "sbto_traj.npz")
            costs = np.array(meta["motions"][bn]["cost"], dtype=np.float64)
            B, T, _ = sbto["dof_pos"].shape
            b0 = int(np.argmin(costs))
            carry_xy = float(np.linalg.norm(sbto["object_pos"][b0, T - 95, :2] - sbto["object_pos"][b0, 70, :2]))
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
                min_fz_all = np.minimum(bpos_c[:, la_idx_35, 2], bpos_c[:, ra_idx_35, 2])

                entry_candidates = [
                    t for t in range(52, 74)
                    if up_z_all[t] >= 0.86 and 0.026 <= min_fz_all[t] <= 0.056
                ]
                t_entry = entry_candidates[0] if entry_candidates else 56
                exit_candidates = [
                    t for t in range(T - 96, min(T - 70, T))
                    if up_z_all[t] >= 0.85 and 0.026 <= min_fz_all[t] <= 0.058
                ]
                t_exit = exit_candidates[0] if exit_candidates else (T - 90)

                if float(np.abs(dof_v[t_entry:t_exit]).max()) > 10.0:
                    drop_summary["qv_spike"] = drop_summary.get("qv_spike", 0) + 1
                    continue
                max_wyaw = float(np.degrees(np.abs(dof_p[t_entry:t_exit, 12]).max()))
                max_wroll = float(np.degrees(np.abs(dof_p[t_entry:t_exit, 13]).max()))
                if max_wyaw > 25.0 or max_wroll > 22.0:
                    drop_summary["waist_twist"] = drop_summary.get("waist_twist", 0) + 1
                    continue

                lifted = o_pos[t_entry:t_exit, 2] > (float(o_pos[t_entry, 2]) + 0.035)
                l_idx = np.flatnonzero(lifted)
                if len(l_idx) < 25 or abs(float(o_pos[t_exit - 1, 2]) - float(o_pos[t_entry, 2])) > 0.025:
                    drop_summary["lift_or_place_incomplete"] = drop_summary.get("lift_or_place_incomplete", 0) + 1
                    continue

                ok, reason, stats = auditor.evaluate_sequence(
                    r_pos[t_entry:t_exit],
                    r_quat[t_entry:t_exit],
                    q_bfs[t_entry:t_exit],
                    o_pos[t_entry:t_exit],
                    o_quat[t_entry:t_exit],
                    first_lift=int(l_idx[0]),
                    last_lift=int(l_idx[-1]),
                )
                if not ok:
                    key = reason.split("(")[0]
                    drop_summary[key] = drop_summary.get(key, 0) + 1
                    continue

                kept_for_base.append(
                    {
                        "base": bn,
                        "solve_idx": int(b),
                        "cost": float(costs[b]),
                        "r_pos": r_pos[t_entry:t_exit].copy(),
                        "r_quat": r_quat[t_entry:t_exit].copy(),
                        "q_bfs": q_bfs[t_entry:t_exit].copy(),
                        "o_pos": o_pos[t_entry:t_exit].copy(),
                        "o_quat": o_quat[t_entry:t_exit].copy(),
                        "stats": stats,
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
            f"(top solve #{selected_solves[0]['solve_idx']}: hand_dy={selected_solves[0]['stats']['min_hand_dy_m']:.3f}m, "
            f"leg_clr={selected_solves[0]['stats']['min_leg_dist_cm']:.2f}cm, com_pick={selected_solves[0]['stats']['com_toe_pick_cm']:+.2f}cm)"
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

    yaw_offsets_deg = (0.0, -10.0, 10.0, -6.0, 6.0)
    c4_yaws = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)

    for core_i, core in enumerate(clean_cores):
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
                # Ensure upright pre-bend buffer frames before box motion have contact=False
                moving_idx = np.flatnonzero(np.linalg.norm(olin_core, axis=-1) > 0.035)
                if len(moving_idx) > 0:
                    contact_core[: max(1, int(moving_idx[0]) - 4)] = False
                if bool(contact_core[0]):
                    continue

                yaw_off = np.radians(yaw_offsets_deg[(core_i + app_k * 2 + int(is_mirror)) % len(yaw_offsets_deg)])
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
                    )
                except RuntimeError:
                    post_stitch_drops += 1
                    continue

                # Final end-to-end MuJoCo full-trajectory gatekeeper (Walk + PinnedBlend + SBTO_Core)!
                bpos_out = stitched["body_pos_w"]
                bquat_out = stitched["body_quat_w"]
                q_out = stitched["joint_pos"]
                opos_out = stitched["obj_pos"]
                orot_out = stitched["obj_rot"]
                oquat_out = _rotmat_to_quat_wxyz(orot_out.astype(np.float64))
                contact_out = stitched["contact"]

                w_n = int(stitched["meta"]["lafan_frames_50hz"])
                b_n = int(stitched["meta"]["blend_frames_50hz"])
                lifted_full = opos_out[:, 2] > (float(opos_out[0, 2]) + 0.035)
                lf_idx = np.flatnonzero(lifted_full)
                if len(lf_idx) < 25:
                    post_stitch_drops += 1
                    continue

                ok_final, reason_final, stats_final = auditor.evaluate_sequence(
                    bpos_out[:, 0].astype(np.float64),
                    bquat_out[:, 0].astype(np.float64),
                    q_out.astype(np.float64),
                    opos_out.astype(np.float64),
                    oquat_out,
                    first_lift=int(lf_idx[0]),
                    last_lift=int(lf_idx[-1]),
                    blend_range=(w_n, w_n + b_n),
                )
                if not ok_final:
                    post_stitch_drops += 1
                    continue

                oang_out = np.concatenate([stitched["obj_ang_vel_pre"], oang_core], axis=0).astype(np.float32)
                clip_folder = out_root / f"data_{kept_idx:03d}"
                clip_folder.mkdir(parents=True, exist_ok=True)

                np.savez(
                    clip_folder / "robot_50hz.npz",
                    fps=np.array([50], dtype=np.int32),
                    joint_pos=q_out,
                    joint_vel=stitched["joint_vel"],
                    body_pos_w=bpos_out,
                    body_quat_w=bquat_out,
                    body_lin_vel_w=stitched["body_lin_vel_w"],
                    body_ang_vel_w=stitched["body_ang_vel_w"],
                )
                with open(clip_folder / "obj_motion_global_50hz.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "obj_trans": opos_out,
                            "obj_rot": orot_out,
                            "obj_scale": float(1.0),
                            "obj_lin_vel": stitched["obj_lin_vel"],
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
                    **stitched["meta"],
                    **stats_final,
                }
                manifest.append(entry)
                kept_idx += 1

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - t_start_wall
    print("=" * 86)
    print(
        f"[DONE] Generated {kept_idx} 100%-MuJoCo-verified clean clips -> `{out_root}` in {elapsed:.1f}s "
        f"(post-stitch gate rejected={post_stitch_drops})"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()

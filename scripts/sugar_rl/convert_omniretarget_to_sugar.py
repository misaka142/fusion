"""Convert `omniretarget/OmniRetarget_Dataset` (`robot-object/*.npz`) to SUGAR `CarryBox` 50Hz format.

Highlights:
1. Canonical OBB SE(3) re-alignment on `largebox.obj` 217-vertex convex hull (det(R)=+1, +Z up at rest)
   with machine-precision world-space equivalence verification, plus single-constant per-clip resting floor
   Z-offset calibration (no per-frame smoothing).
2. Exports both `obj_aligned.obj` (for MuJoCo) and standalone OpenUSD ASCII `obj_aligned.usda`
   (`PhysicsRigidBodyAPI` + `PhysicsMassAPI` + `PhysicsMeshCollisionAPI(convexHull)` with
   `physxConvexHullCollision:hullVertexLimit = 255`) plus `box_spec.json`.
3. Smooth C1 Cubic Catmull-Rom + Quaternion SLERP resampling from source FPS (30 Hz) -> 50 Hz.
4. Exact 29-DoF joint permutation from Unitree SDK order (`OmniRetarget`) -> IsaacLab PhysX BFS order (`SUGAR`).
5. Full 35-body forward kinematics (`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`) via
   `FastTreeKinematics`.
6. Clean `contact_labels_50hz.npy` extraction anchored on object motion + aligned OBB hand proximity.

Usage (Windows PowerShell):
    D:\\anaconda\\envs\\DRL_env\\python.exe scripts/sugar_rl/convert_omniretarget_to_sugar.py `
        --raw_dir data_omniretarget_raw `
        --out_dir data/OmniCarryBox `
        --subset pick_carry_place_original
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
import torch
import trimesh

_FUSION_ROOT = Path(__file__).resolve().parents[2]
_SUGAR_RL_SRC = _FUSION_ROOT / "source" / "sugar_rl"
for _p in (_FUSION_ROOT, _SUGAR_RL_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.sim2sim.scene_builder import G1_ISAACLAB_BFS_JOINT_NAMES
from scripts.sugar_rl.lafan_walk_stitcher import load_lafan_walk_segments, stitch_lafan_walk_to_omni_clip
from sugar_rl.supermimic.urdf_kinematics import FastTreeKinematics, G1_29DOF_SDK_JOINT_NAMES

TARGET_FPS = 50.0
DT = 1.0 / TARGET_FPS

SUGAR_RAW_35_BODY_NAMES: list[str] = [
    "pelvis", "left_hip_pitch_link", "pelvis_contour_link", "right_hip_pitch_link",
    "waist_yaw_link", "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "logo_link", "left_shoulder_pitch_link", "head_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link",
    "right_wrist_yaw_link", "left_rubber_hand", "right_rubber_hand",
]


# -----------------------------------------------------------------------------
# Smooth C1 Catmull-Rom Spline & Quaternion SLERP Resamplers (30Hz -> 50Hz)
# -----------------------------------------------------------------------------

def _quat_norm(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = _quat_norm(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def _resample_catmull_rom(y: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS) -> np.ndarray:
    """C1-continuous cubic Catmull-Rom interpolation from `src_fps` to `dst_fps`."""
    T_in = len(y)
    if abs(src_fps - dst_fps) < 1e-6 or T_in < 2:
        return y.copy()
    duration = (T_in - 1) / src_fps
    T_out = int(round(duration * dst_fps)) + 1
    t_out = np.linspace(0.0, T_in - 1.0, T_out, dtype=np.float64)

    i1 = np.clip(np.floor(t_out).astype(np.int64), 0, T_in - 2)
    u = (t_out - i1)[:, None]
    i0 = np.maximum(i1 - 1, 0)
    i2 = np.minimum(i1 + 1, T_in - 1)
    i3 = np.minimum(i1 + 2, T_in - 1)

    p0, p1, p2, p3 = y[i0], y[i1], y[i2], y[i3]
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _resample_quat_slerp(q_in: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS) -> np.ndarray:
    """Sign-consistent quaternion SLERP interpolation from `src_fps` to `dst_fps`."""
    q = _quat_norm(q_in.astype(np.float64))
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0.0:
            q[t] = -q[t]
    T_in = len(q)
    if abs(src_fps - dst_fps) < 1e-6 or T_in < 2:
        return q
    duration = (T_in - 1) / src_fps
    T_out = int(round(duration * dst_fps)) + 1
    t_out = np.linspace(0.0, T_in - 1.0, T_out, dtype=np.float64)

    i0 = np.clip(np.floor(t_out).astype(np.int64), 0, T_in - 2)
    u = t_out - i0
    qa = q[i0]
    qb = q[i0 + 1]

    dot = np.clip((qa * qb).sum(axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_th = np.sin(theta)
    small = sin_th < 1e-6

    w_a = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(small, 1.0, sin_th))
    w_b = np.where(small, u, np.sin(u * theta) / np.where(small, 1.0, sin_th))
    return _quat_norm(w_a[:, None] * qa + w_b[:, None] * qb)


def _ang_vel_world_from_quats(q_seq: np.ndarray, dt: float = DT) -> np.ndarray:
    q = _quat_norm(q_seq.copy())
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0.0:
            q[t] = -q[t]
    qd = np.gradient(q, dt, axis=0)
    prod = 2.0 * _quat_mul(qd, _quat_conj(q))
    return prod[:, 1:4]


# -----------------------------------------------------------------------------
# Canonical OBB Re-Alignment & OpenUSD ASCII (.usda) Convex Hull Writer
# -----------------------------------------------------------------------------

def _compute_canonical_obb_alignment(
    hull_raw: trimesh.Trimesh,
    ref_raw_rot0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh, np.ndarray]:
    """Compute right-handed +Z-up SE(3) alignment `(R_obb, p_obb)` mapping canonical frame -> raw frame.

    Invariant for every frame t:
        v_raw(t) = R_raw(t) @ v + t_raw(t)
                 == (R_raw(t) @ R_obb) @ v_aligned + (t_raw(t) + R_raw(t) @ p_obb)
    """
    obb = hull_raw.bounding_box_oriented
    T_obb = np.asarray(obb.primitive.transform, dtype=np.float64)
    R_cols = T_obb[:3, :3].copy()
    p_obb = T_obb[:3, 3].copy()

    # Gram-Schmidt orthonormalization for bit-tight SO(3)
    c0 = R_cols[:, 0] / np.linalg.norm(R_cols[:, 0])
    c1 = R_cols[:, 1] - np.dot(R_cols[:, 1], c0) * c0
    c1 /= np.linalg.norm(c1)
    c2 = np.cross(c0, c1)
    R_ortho = np.column_stack([c0, c1, c2])

    # Identify which OBB principal axis corresponds to world +Z when box is resting at ref_raw_rot0
    up_in_raw = ref_raw_rot0.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dots = up_in_raw @ R_ortho
    z_col_idx = int(np.argmax(np.abs(dots)))
    other_cols = [i for i in range(3) if i != z_col_idx]

    z_axis = R_ortho[:, z_col_idx].copy()
    if float(np.dot(up_in_raw, z_axis)) < 0.0:
        z_axis = -z_axis

    # Deterministic X/Y ordering among remaining two axes, flipping X if needed so det(R) == +1
    x_axis = R_ortho[:, other_cols[0]].copy()
    y_axis = R_ortho[:, other_cols[1]].copy()
    R_obb = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(R_obb)) < 0.0:
        R_obb[:, 0] = -R_obb[:, 0]

    assert np.allclose(R_obb.T @ R_obb, np.eye(3), atol=1e-6), "R_obb must be orthonormal"
    assert abs(float(np.linalg.det(R_obb)) - 1.0) < 1e-6, "det(R_obb) must be +1"
    assert float((ref_raw_rot0 @ R_obb)[2, 2]) > 0.95, "Canonical box +Z must align with world +Z at rest!"

    # Transform raw convex hull vertices into canonical box frame and absorb any sub-mm center residual
    v_al = (R_obb.T @ (np.asarray(hull_raw.vertices, dtype=np.float64) - p_obb).T).T
    c_res = 0.5 * (v_al.min(axis=0) + v_al.max(axis=0))
    p_obb = p_obb + R_obb @ c_res
    v_al = v_al - c_res

    hull_al = trimesh.Trimesh(vertices=v_al, faces=hull_raw.faces.copy(), process=False)
    trimesh.repair.fix_normals(hull_al)

    # Hard assertion: exact world-space equivalence against raw mesh at t=0
    pts_world_raw = (ref_raw_rot0 @ np.asarray(hull_raw.vertices, dtype=np.float64).T).T
    R_al0 = ref_raw_rot0 @ R_obb
    pts_world_al = (R_al0 @ np.asarray(hull_al.vertices, dtype=np.float64).T).T + (ref_raw_rot0 @ p_obb)
    equiv_err = float(np.max(np.abs(pts_world_raw - pts_world_al)))
    assert equiv_err < 1e-6, f"World-space OBB alignment residual too large: {equiv_err}"

    box_size_al = v_al.max(axis=0) - v_al.min(axis=0)
    return R_obb, p_obb, hull_al, box_size_al


def _write_usda_convex_hull(mesh: trimesh.Trimesh, out_path: Path, mass_kg: float = 0.5) -> None:
    """Write standalone OpenUSD ASCII asset matching IsaacLab `UsdFileCfg` rigid-body convexHull schema."""
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    pts_str = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in pts)
    counts_str = ", ".join(["3"] * len(faces))
    indices_str = ", ".join(str(int(i)) for i in faces.reshape(-1))

    usda = f"""#usda 1.0
(
    defaultPrim = "obj_aligned"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "obj_aligned" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = 1
    float physics:mass = {float(mass_kg):.6g}
    double3 xformOp:translate = (0, 0, 0)
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

    def Xform "geometry"
    {{
        def Mesh "mesh" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI"]
        )
        {{
            uniform token physics:approximation = "convexHull"
            bool physics:collisionEnabled = 1
            int physxConvexHullCollision:hullVertexLimit = 255
            color3f[] primvars:displayColor = [(0.80, 0.56, 0.32)]
            int[] faceVertexCounts = [{counts_str}]
            int[] faceVertexIndices = [{indices_str}]
            point3f[] points = [{pts_str}]
        }}
    }}
}}
"""
    out_path.write_text(usda, encoding="utf-8")


def _extract_clean_contact_labels(
    obj_pos: np.ndarray,
    obj_lin_vel: np.ndarray,
    obj_ang_vel: np.ndarray,
    lh_pos_w: np.ndarray,
    rh_pos_w: np.ndarray,
    box_half_extents: np.ndarray,
    obj_rot: np.ndarray,
) -> np.ndarray:
    """Infer binary `contact_labels_50hz` anchored on object motion + aligned OBB hand proximity."""
    T = len(obj_pos)
    v_norm = np.linalg.norm(obj_lin_vel, axis=-1)
    w_norm = np.linalg.norm(obj_ang_vel, axis=-1)
    z_floor_ref = min(float(np.percentile(obj_pos[:10, 2], 50)), float(np.percentile(obj_pos[-10:, 2], 50)))
    lifted = obj_pos[:, 2] > (z_floor_ref + 0.012)
    moving = (v_norm > 0.035) | (w_norm > 0.08) | lifted

    R_T = np.swapaxes(obj_rot, -1, -2)
    lh_local = np.abs(np.einsum("tij,tj->ti", R_T, lh_pos_w - obj_pos)) - box_half_extents[None, :]
    rh_local = np.abs(np.einsum("tij,tj->ti", R_T, rh_pos_w - obj_pos)) - box_half_extents[None, :]
    lh_surf_d = np.linalg.norm(np.maximum(lh_local, 0.0), axis=-1)
    rh_surf_d = np.linalg.norm(np.maximum(rh_local, 0.0), axis=-1)
    hands_close = (lh_surf_d < 0.085) & (rh_surf_d < 0.085)

    raw_c = moving | hands_close
    if moving.any():
        m_idx = np.flatnonzero(moving)
        t_first = max(0, int(m_idx[0]) - 5)
        t_last = min(T - 1, int(m_idx[-1]) + 3)
        contact = np.zeros(T, dtype=bool)
        contact[t_first : t_last + 1] = True
        while t_first < t_last and not hands_close[t_first] and not moving[t_first]:
            contact[t_first] = False
            t_first += 1
        while t_last > t_first and not hands_close[t_last] and not moving[t_last]:
            contact[t_last] = False
            t_last -= 1
        return contact
    return raw_c.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", type=str, default=str(_FUSION_ROOT / "data_omniretarget_raw"))
    ap.add_argument("--out_dir", type=str, default=str(_FUSION_ROOT / "data" / "OmniCarryBox"))
    ap.add_argument(
        "--subset",
        type=str,
        default="pick_carry_place_all",
        choices=["pick_carry_place_original", "all_original", "pick_carry_place_all", "all"],
        help=(
            "'pick_carry_place_all' (default, all valid floor-to-floor carry clips w/ spatial aug), "
            "'pick_carry_place_original' (original-only subset), "
            "'all_original' (312 clips), or 'all' (1826 clips)."
        ),
    )
    ap.add_argument("--max_clips", type=int, default=None, help="Optional cap on number of converted clips.")
    ap.add_argument(
        "--max_variants_per_base",
        type=int,
        default=3,
        help="Cap variants per base motion for balanced diversity (default: 3 -> ~159 clips; 0 = keep all smooth).",
    )
    ap.add_argument(
        "--max_joint_vel",
        type=float,
        default=10.0,
        help="Reject raw clips exceeding this joint speed (rad/s) when posture filtering is active (default: 10.0).",
    )
    ap.add_argument(
        "--max_joint_acc",
        type=float,
        default=260.0,
        help="Reject raw clips exceeding this joint accel (rad/s^2) when posture filtering is active (default: 260.0).",
    )
    ap.add_argument(
        "--no_floor_offset",
        action="store_true",
        default=False,
        help="Disable per-clip constant floor clearance offset on resting frames.",
    )
    ap.add_argument(
        "--no_posture_filter",
        action="store_true",
        default=False,
        help="Disable t=0 upright torso & non-contact filtering for pick_carry_place subsets.",
    )
    ap.add_argument(
        "--lafan_dir",
        type=str,
        default=str(_FUSION_ROOT / "data_lafan1_raw"),
        help="Path to Unitree G1 LAFAN1 walk CSV directory (`data_lafan1_raw`, expects `g1/walk*.csv`).",
    )
    ap.add_argument(
        "--no_lafan_walk",
        action="store_true",
        default=False,
        help="Disable prepending LAFAN1 straight-walk approach segments ahead of clean CarryBox clips.",
    )
    ap.add_argument(
        "--blend_frames",
        type=int,
        default=25,
        help="Number of 50Hz transition blend frames between LAFAN1 walk and OmniCarryBox (default: 25 = 0.50s).",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_dir).resolve()
    npz_dir = raw_root / "robot-object"
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"Missing `{npz_dir}`. Run `download_omniretarget.py` first.")

    largebox_obj = raw_root / "models" / "largebox" / "largebox.obj"
    if not largebox_obj.is_file():
        raise FileNotFoundError(f"Missing `{largebox_obj}`. Run `download_omniretarget.py` first.")

    # SDK -> IsaacLab BFS permutation
    sdk_to_idx = {n: i for i, n in enumerate(G1_29DOF_SDK_JOINT_NAMES)}
    sdk_for_bfs = np.array([sdk_to_idx[n] for n in G1_ISAACLAB_BFS_JOINT_NAMES], dtype=np.int64)

    urdf_path = _FUSION_ROOT / "descriptions" / "robots" / "g1" / "g1_29dof_rev_1_0_with_rubber_hand.urdf"
    fk_35 = FastTreeKinematics(
        urdf_or_xml_path=urdf_path if urdf_path.exists() else None,
        env_joint_names=G1_ISAACLAB_BFS_JOINT_NAMES,
        tracked_body_names=SUGAR_RAW_35_BODY_NAMES,
        device="cpu",
    )

    all_files = sorted(npz_dir.glob("*.npz"))
    candidate_paths: list[Path] = []
    for p in all_files:
        is_orig = p.stem.endswith("_original")
        if "original" in args.subset and not is_orig:
            continue
        if args.subset.startswith("pick_carry_place"):
            with np.load(p) as d:
                q = d["qpos"]
            obj_xyz = q[:, 40:43]
            z0, z1, zmax = float(obj_xyz[0, 2]), float(obj_xyz[-1, 2]), float(obj_xyz[:, 2].max())
            xy_disp = float(np.linalg.norm(obj_xyz[-1, :2] - obj_xyz[0, :2]))
            min_xy_disp = 0.30 if args.no_posture_filter else 0.50
            if not (z0 < 0.30 and z1 < 0.30 and (zmax - max(z0, z1)) > 0.12 and xy_disp >= min_xy_disp):
                continue
        candidate_paths.append(p)

    if not candidate_paths:
        raise RuntimeError(f"No clips matched subset='{args.subset}' under {npz_dir}")

    # Compute canonical +Z-up OBB alignment using first original (or first candidate) resting pose
    ref_clip = next((p for p in candidate_paths if p.stem.endswith("_original")), candidate_paths[0])
    with np.load(ref_clip) as d0:
        ref_raw_rot0 = _quat_to_rotmat(d0["qpos"][0:1, 36:40].astype(np.float64))[0]

    m_box = trimesh.load(str(largebox_obj), force="mesh", process=False)
    hull_raw = m_box.convex_hull
    R_obb, p_obb, hull_aligned, box_size_al = _compute_canonical_obb_alignment(hull_raw, ref_raw_rot0)
    half_ext_al = 0.5 * box_size_al

    # Quality & diversity pre-filtering before heavy resampling/stitching
    torso_raw_idx = SUGAR_RAW_35_BODY_NAMES.index("torso_link")
    la_raw_idx = SUGAR_RAW_35_BODY_NAMES.index("left_ankle_roll_link")
    ra_raw_idx = SUGAR_RAW_35_BODY_NAMES.index("right_ankle_roll_link")
    chk_names = ["pelvis", "torso_link", "head_link", "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"]
    chk_idxs = [SUGAR_RAW_35_BODY_NAMES.index(n) for n in chk_names]

    skipped_bent = 0
    skipped_floating = 0
    skipped_t0_contact = 0
    skipped_spike = 0
    skipped_pen = 0
    skipped_stitch = 0
    selected_scored: list[tuple[str, bool, float, Path]] = []

    if args.subset.startswith("pick_carry_place") and not args.no_posture_filter:
        for p in candidate_paths:
            with np.load(p) as d:
                q_raw = d["qpos"].astype(np.float64)
                fps_raw = float(d["fps"])
            q_bfs_raw = q_raw[:, 7:36][:, sdk_for_bfs]
            max_v = float(np.abs(np.diff(q_bfs_raw, axis=0) * fps_raw).max())
            max_a = float(np.abs(np.diff(q_bfs_raw, n=2, axis=0) * (fps_raw * fps_raw)).max())
            if max_v > args.max_joint_vel or max_a > args.max_joint_acc:
                skipped_spike += 1
                continue

            with torch.no_grad():
                fk_chk = fk_35(
                    torch.from_numpy(q_raw[:, 4:7].astype(np.float32)),
                    torch.from_numpy(q_raw[:, 0:4].astype(np.float32)),
                    torch.from_numpy(q_bfs_raw.astype(np.float32)),
                )
            bpos_raw = fk_chk.tracked_pos_w.numpy().astype(np.float64)
            bquat0 = fk_chk.tracked_quat_w[0].numpy().astype(np.float64)
            rot_torso0 = _quat_to_rotmat(bquat0[torso_raw_idx : torso_raw_idx + 1])[0]
            torso_up_z0 = float(rot_torso0[2, 2])
            if torso_up_z0 < 0.80:
                skipped_bent += 1
                continue
            min_foot_z0 = float(min(bpos_raw[0, la_raw_idx, 2], bpos_raw[0, ra_raw_idx, 2]))
            if not (0.026 <= min_foot_z0 <= 0.068):
                skipped_floating += 1
                continue

            oxyz = q_raw[:, 40:43]
            ov = np.linalg.norm(np.diff(oxyz[: min(5, len(oxyz))], axis=0) * fps_raw, axis=-1)
            if (len(ov) > 0 and ov[0] > 0.08) or (len(oxyz) > 1 and oxyz[1, 2] > float(oxyz[0, 2]) + 0.015):
                skipped_t0_contact += 1
                continue

            # Reject clips with deep non-hand penetration into largebox OBB
            orot_raw = _quat_to_rotmat(q_raw[:, 36:40])
            orot_al = np.einsum("tij,jk->tik", orot_raw, R_obb)
            opos_al = oxyz + np.einsum("tij,j->ti", orot_raw, p_obb)
            rel = bpos_raw[:, chk_idxs, :] - opos_al[:, None, :]
            loc = np.einsum("tij,tkj->tki", orot_al, rel)
            min_sd = float((np.abs(loc) - half_ext_al[None, None, :]).max(axis=-1).min())
            if min_sd < -0.02:
                skipped_pen += 1
                continue

            base_id = "_".join(p.stem.split("_")[:3])
            is_orig = p.stem.endswith("_original")
            smooth_cost = max_v + 0.02 * max_a
            selected_scored.append((base_id, is_orig, smooth_cost, p))

        if args.max_variants_per_base > 0:
            by_base: dict[str, list[tuple[bool, float, Path]]] = {}
            for base_id, is_orig, sc, p in selected_scored:
                by_base.setdefault(base_id, []).append((is_orig, sc, p))
            selected = []
            for base_id in sorted(by_base.keys()):
                grp = sorted(by_base[base_id], key=lambda item: (0 if item[0] else 1, item[1]))
                selected.extend(item[2] for item in grp[: args.max_variants_per_base])
        else:
            selected = [item[3] for item in selected_scored]
    else:
        selected = candidate_paths

    if args.max_clips is not None:
        selected = selected[: args.max_clips]
    if not selected:
        raise RuntimeError(f"All {len(candidate_paths)} candidates were filtered out under '{args.subset}'!")

    out_root = Path(args.out_dir).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Export canonical convex-hull meshes (`.obj` for MuJoCo, `.usda` for Isaac Sim)
    hull_aligned.export(str(out_root / "obj_aligned.obj"))
    _write_usda_convex_hull(hull_aligned, out_root / "obj_aligned.usda", mass_kg=0.5)

    box_spec = {
        "dataset": "omniretarget_omomo_largebox",
        "task": "CarryBox",
        "box_size_xyz": [round(float(x), 5) for x in box_size_al],
        "box_center_xyz": [0.0, 0.0, 0.0],
        "mass_kg": 0.5,
        "hull_vertices": int(len(hull_aligned.vertices)),
        "hull_faces": int(len(hull_aligned.faces)),
        "mesh_file": "obj_aligned.obj",
        "usd_file": "obj_aligned.usda",
    }
    (out_root / "box_spec.json").write_text(json.dumps(box_spec, indent=2), encoding="utf-8")

    print("=" * 82)
    print(f"[OmniRetarget -> SUGAR] Subset: {args.subset} | Selected: {len(selected)} / {len(candidate_paths)} candidates (total raw={len(all_files)})")
    print(
        f"[OmniRetarget -> SUGAR] Quality filters dropped: bent_t0={skipped_bent}, float_t0={skipped_floating}, "
        f"contact_t0={skipped_t0_contact}, spike={skipped_spike}, penetration={skipped_pen} | max_per_base={args.max_variants_per_base}"
    )
    print(
        f"[OmniRetarget -> SUGAR] Aligned OBB size: {box_spec['box_size_xyz']} m "
        f"| ConvexHull verts={len(hull_aligned.vertices)}, faces={len(hull_aligned.faces)}"
    )
    print(f"[OmniRetarget -> SUGAR] Floor calibration: {'OFF' if args.no_floor_offset else 'ON (per-clip const Z)'} -> {out_root}")
    use_lafan = args.subset.startswith("pick_carry_place") and not args.no_lafan_walk
    lafan_segments = []
    lafan_usage: dict[str, int] = {}
    if use_lafan:
        lafan_root = Path(args.lafan_dir).resolve()
        lafan_segments = load_lafan_walk_segments(
            lafan_dir=lafan_root,
            fk_35=fk_35,
            sdk_for_bfs=sdk_for_bfs,
            body_names=SUGAR_RAW_35_BODY_NAMES,
            resample_catmull_rom_fn=_resample_catmull_rom,
            resample_quat_slerp_fn=_resample_quat_slerp,
            ang_vel_fn=_ang_vel_world_from_quats,
            dt=DT,
        )
        print(f"[OmniRetarget + LAFAN1] Loaded {len(lafan_segments)} clean straight-walk windows from `{lafan_root / 'g1'}` (blend={args.blend_frames} frames)")
    print("=" * 82)

    t0 = time.perf_counter()
    verts_al = np.asarray(hull_aligned.vertices, dtype=np.float64)  # (V, 3)
    lk_bfs = G1_ISAACLAB_BFS_JOINT_NAMES.index("left_knee_joint")
    rk_bfs = G1_ISAACLAB_BFS_JOINT_NAMES.index("right_knee_joint")

    manifest = []
    floor_shifts_mm = []
    kept_idx = 0
    for idx, src_npz in enumerate(selected):
        with np.load(src_npz) as d:
            q_in = d["qpos"].astype(np.float64)
            src_fps = float(d["fps"])

        rob_quat_50 = _resample_quat_slerp(q_in[:, 0:4], src_fps, TARGET_FPS)
        rob_pos_50 = _resample_catmull_rom(q_in[:, 4:7], src_fps, TARGET_FPS)
        q_sdk_50 = _resample_catmull_rom(q_in[:, 7:36], src_fps, TARGET_FPS)
        obj_quat_raw_50 = _resample_quat_slerp(q_in[:, 36:40], src_fps, TARGET_FPS)
        obj_pos_raw_50 = _resample_catmull_rom(q_in[:, 40:43], src_fps, TARGET_FPS)

        q_bfs_50 = q_sdk_50[:, sdk_for_bfs]
        rob_lin_vel = np.gradient(rob_pos_50, DT, axis=0).astype(np.float32)
        rob_ang_vel = _ang_vel_world_from_quats(rob_quat_50, DT).astype(np.float32)
        q_vel_bfs = np.gradient(q_bfs_50, DT, axis=0).astype(np.float32)

        with torch.no_grad():
            fk_res = fk_35(
                torch.from_numpy(rob_pos_50.astype(np.float32)),
                torch.from_numpy(rob_quat_50.astype(np.float32)),
                torch.from_numpy(q_bfs_50.astype(np.float32)),
                torch.from_numpy(rob_lin_vel),
                torch.from_numpy(rob_ang_vel),
                torch.from_numpy(q_vel_bfs),
            )

        b_pos = fk_res.tracked_pos_w.numpy().astype(np.float32)
        b_quat = fk_res.tracked_quat_w.numpy().astype(np.float32)
        b_lin = fk_res.tracked_lin_vel_w.numpy().astype(np.float32)
        b_ang = fk_res.tracked_ang_vel_w.numpy().astype(np.float32)

        # Apply canonical OBB SE(3) re-alignment cleanly per frame
        obj_rot_raw = _quat_to_rotmat(obj_quat_raw_50)  # (T, 3, 3)
        obj_rot_al = np.einsum("tij,jk->tik", obj_rot_raw, R_obb)  # (T, 3, 3)
        obj_pos_al = obj_pos_raw_50 + np.einsum("tij,j->ti", obj_rot_raw, p_obb)  # (T, 3)

        # Single constant per-clip Z offset based on convex-hull bottom across resting start/end frames
        dz_const = 0.0
        if not args.no_floor_offset:
            rest_idx = np.concatenate([np.arange(min(10, len(obj_pos_al))), np.arange(max(0, len(obj_pos_al) - 10), len(obj_pos_al))])
            rest_z_min = []
            for ri in rest_idx:
                pts_w = (obj_rot_al[ri] @ verts_al.T).T + obj_pos_al[ri]
                rest_z_min.append(float(pts_w[:, 2].min()))
            z_rest_med = float(np.median(rest_z_min))
            if z_rest_med < 0.0:
                dz_const = -z_rest_med
                obj_pos_al[:, 2] += dz_const

        # Exact world angular velocity via constant-R identity (`qd` / `obj_rot_al`)
        obj_lin_vel = np.gradient(obj_pos_al, DT, axis=0).astype(np.float32)
        # Note: R_obj_al(t) = R_obj_raw(t) @ R_obb -> w_world(t) is identical to R_obj_raw(t)
        obj_ang_vel = _ang_vel_world_from_quats(obj_quat_raw_50, DT).astype(np.float32)

        contact_50 = _extract_clean_contact_labels(
            obj_pos_al,
            obj_lin_vel,
            obj_ang_vel,
            b_pos[:, 33].astype(np.float64),
            b_pos[:, 34].astype(np.float64),
            half_ext_al,
            obj_rot_al,
        )

        rot_torso0 = _quat_to_rotmat(b_quat[0:1, torso_raw_idx].astype(np.float64))[0]
        torso_up_z0 = float(rot_torso0[2, 2])
        if args.subset.startswith("pick_carry_place") and not args.no_posture_filter:
            if torso_up_z0 < 0.80:
                skipped_bent += 1
                continue
            if bool(contact_50[0]):
                skipped_t0_contact += 1
                continue

        stitch_meta: dict = {}
        if use_lafan:
            try:
                stitched = stitch_lafan_walk_to_omni_clip(
                    omni_pos=rob_pos_50,
                    omni_quat_wxyz=rob_quat_50,
                    omni_q_bfs=q_bfs_50,
                    omni_bpos=b_pos.astype(np.float64),
                    omni_obj_pos=obj_pos_al,
                    omni_obj_rot=obj_rot_al,
                    omni_contact=contact_50,
                    box_half_extents=half_ext_al,
                    segments=lafan_segments,
                    usage_counts=lafan_usage,
                    fk_35=fk_35,
                    body_names=SUGAR_RAW_35_BODY_NAMES,
                    ang_vel_fn=_ang_vel_world_from_quats,
                    blend_frames=args.blend_frames,
                    dt=DT,
                )
            except RuntimeError as exc:
                skipped_stitch += 1
                print(f"  [Skip stitch] {src_npz.name}: {exc}")
                continue
            q_out = stitched["joint_pos"]
            qd_out = stitched["joint_vel"]
            bpos_out = stitched["body_pos_w"]
            bquat_out = stitched["body_quat_w"]
            blin_out = stitched["body_lin_vel_w"]
            bang_out = stitched["body_ang_vel_w"]
            opos_out = stitched["obj_pos"]
            orot_out = stitched["obj_rot"]
            olin_out = stitched["obj_lin_vel"]
            oang_out = np.concatenate([stitched["obj_ang_vel_pre"], obj_ang_vel], axis=0).astype(np.float32)
            contact_out = stitched["contact"]
            stitch_meta = stitched["meta"]
        else:
            q_out = q_bfs_50.astype(np.float32)
            qd_out = q_vel_bfs
            bpos_out = b_pos
            bquat_out = b_quat
            blin_out = b_lin
            bang_out = b_ang
            opos_out = obj_pos_al.astype(np.float32)
            orot_out = obj_rot_al.astype(np.float32)
            olin_out = obj_lin_vel
            oang_out = obj_ang_vel
            contact_out = contact_50.astype(bool)

        floor_shifts_mm.append(dz_const * 1000.0)
        clip_folder = out_root / f"data_{kept_idx:03d}"
        clip_folder.mkdir(parents=True, exist_ok=True)

        np.savez(
            clip_folder / "robot_50hz.npz",
            fps=np.array([50], dtype=np.int32),
            joint_pos=q_out,
            joint_vel=qd_out,
            body_pos_w=bpos_out,
            body_quat_w=bquat_out,
            body_lin_vel_w=blin_out,
            body_ang_vel_w=bang_out,
        )
        with open(clip_folder / "obj_motion_global_50hz.pkl", "wb") as f:
            pickle.dump(
                {
                    "obj_trans": opos_out,
                    "obj_rot": orot_out,
                    "obj_scale": float(1.0),
                    "obj_lin_vel": olin_out,
                    "obj_ang_vel": oang_out,
                },
                f,
            )
        np.save(clip_folder / "contact_labels_50hz.npy", contact_out)

        knees_deg = np.degrees(q_out[:, [lk_bfs, rk_bfs]])
        entry = {
            "index": kept_idx,
            "folder": clip_folder.name,
            "source": src_npz.name,
            "frames_50hz": int(len(q_out)),
            "floor_z_shift_mm": round(dz_const * 1000.0, 3),
            "t0_torso_up_z": round(torso_up_z0, 4),
        }
        entry.update(stitch_meta)
        manifest.append(entry)
        kept_idx += 1

        if kept_idx % 20 == 0 or kept_idx == 1 or (idx + 1) == len(selected):
            walk_info = f" | walk=+{stitch_meta['pre_walk_path_m']:.2f}m ({stitch_meta['lafan_source']})" if stitch_meta else ""
            print(
                f"  [{kept_idx:3d} kept | scan {idx + 1:3d}/{len(selected)}] {clip_folder.name} <- {src_npz.name:34s} | "
                f"T={len(q_out):3d} | torso_up={torso_up_z0:.2f} | dz=+{dz_const*1000:4.2f}mm{walk_info}"
            )

    if not manifest:
        raise RuntimeError(f"All {len(selected)} candidate clips were filtered out! Check filter thresholds.")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("=" * 82)
    print(
        f"[DONE] Converted {len(manifest)} high-quality clips -> `{out_root}` in {time.perf_counter() - t0:.1f}s "
        f"(filtered: bent_t0={skipped_bent}, float_t0={skipped_floating}, contact_t0={skipped_t0_contact}, "
        f"spike={skipped_spike}, pen={skipped_pen}, stitch={skipped_stitch} | "
        f"floor dz median=+{np.median(floor_shifts_mm):.2f}mm, max=+{np.max(floor_shifts_mm):.2f}mm)"
    )
    print("=" * 82)


if __name__ == "__main__":
    main()

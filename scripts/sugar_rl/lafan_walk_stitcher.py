"""Phase-matched SE(2) smooth stitching of Unitree G1 LAFAN1 walking clips onto OmniCarryBox.

Prepends a clean straight-walking approach (total pre-contact distance ~1.2m - 1.8m)
from `lvhaidong/LAFAN1_Retargeting_Dataset` (`g1/walk*.csv`) before each clean
`OmniCarryBox` pickup clip using:
1. Heading-frame ankle-phase + forward-velocity + pelvis-height matching.
2. Analytic SE(2) root alignment so the walking heading enters `omni[0]` smoothly.
3. C1-continuous Cubic Hermite splines (joint positions, pelvis XYZ) + Lie-group
   cubic Hermite SO(3) orientation interpolation across a 0.50s (25-frame) blend window.
4. Full `FastTreeKinematics` forward kinematics recomputation + unified numerical gradients,
   followed by static-box OBB surface clearance & foot ground-clearance assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sugar_rl.supermimic.urdf_kinematics import FastTreeKinematics


@dataclass
class LafanWalkSegment:
    source_name: str
    start_frame_50hz: int
    end_frame_50hz: int
    pos_w: np.ndarray        # (T, 3) pelvis pos at 50Hz
    quat_wxyz: np.ndarray    # (T, 4) pelvis quat (w, x, y, z) at 50Hz
    q_bfs: np.ndarray        # (T, 29) BFS joint angles at 50Hz
    qd_bfs_end: np.ndarray   # (29,) ending joint velocity
    v_pelvis_end: np.ndarray # (3,) ending pelvis world velocity
    w_pelvis_end: np.ndarray # (3,) ending pelvis world angular velocity
    local_la_end: np.ndarray # (3,) ending left_ankle_roll_link in heading-aligned pelvis frame (z wrt ground)
    local_ra_end: np.ndarray # (3,) ending right_ankle_roll_link in heading-aligned pelvis frame (z wrt ground)
    v_fwd_end: float         # scalar forward speed along ending heading
    walk_disp_m: float       # net XY displacement of pure walking segment
    is_stand_buffer: bool = False  # True if clip naturally decelerates to double-support stand buffer


def com_to_ankle_margin(
    pelvis_xy: np.ndarray,
    torso_xy: np.ndarray,
    la_xy: np.ndarray,
    ra_xy: np.ndarray,
) -> np.ndarray:
    """Compute XY distance from upper-body COM proxy to the line segment connecting both ankles."""
    com_xy = 0.5 * (pelvis_xy + torso_xy)
    seg = ra_xy - la_xy
    seg2 = np.sum(seg * seg, axis=-1) + 1e-9
    t_proj = np.clip(np.sum((com_xy - la_xy) * seg, axis=-1) / seg2, 0.0, 1.0)
    closest = la_xy + t_proj[..., None] * seg
    return np.linalg.norm(com_xy - closest, axis=-1)


def build_bfs_sagittal_mirror_map(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build exact sagittal-plane (`y -> -y`) joint permutation and sign arrays for 29-DoF G1."""
    perm = []
    signs = []
    for jn in joint_names:
        if jn.startswith("left_"):
            partner = "right_" + jn[5:]
        elif jn.startswith("right_"):
            partner = "left_" + jn[6:]
        else:
            partner = jn
        perm.append(joint_names.index(partner))
        is_pitch_like = any(k in jn for k in ("pitch", "knee", "elbow"))
        signs.append(1.0 if is_pitch_like else -1.0)
    return np.array(perm, dtype=np.int64), np.array(signs, dtype=np.float64)


def mirror_sagittal_pose_Arrays(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    q_bfs: np.ndarray,
    perm: np.ndarray,
    signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflect root pose and 29-DoF joint angles across the XZ sagittal plane (`y -> -y`)."""
    pos_m = pos_w * np.array([1.0, -1.0, 1.0], dtype=pos_w.dtype)
    quat_m = quat_wxyz * np.array([1.0, -1.0, 1.0, -1.0], dtype=quat_wxyz.dtype)
    q_m = q_bfs[..., perm] * signs.astype(q_bfs.dtype)
    return pos_m, quat_m, q_m


def _quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
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


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    cos_a = float(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cos_a))
    if angle < 1e-6:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) / (2.0 * np.sin(angle))
    return angle * axis


def _rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(rv))
    if ang < 1e-7:
        wx, wy, wz = rv
        return np.eye(3, dtype=np.float64) + np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=np.float64)
    k = rv / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert `(..., 3, 3)` rotation matrices to sign-consistent `(..., 4)` wxyz quaternions."""
    batch = R.shape[:-2]
    Rm = R.reshape(-1, 3, 3)
    out = np.zeros((len(Rm), 4), dtype=np.float64)
    for i, r in enumerate(Rm):
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
    return out.reshape(*batch, 4)


def _quat_yaw(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z_mat(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _cubic_hermite(p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray, u: np.ndarray, T_dur: float) -> np.ndarray:
    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    return (
        h00[..., None] * p0
        + h10[..., None] * (T_dur * v0)
        + h01[..., None] * p1
        + h11[..., None] * (T_dur * v1)
    )


def _smooth_savgol_np(x: np.ndarray, window: int = 19, polyorder: int = 2) -> np.ndarray:
    """Zero-phase pure-NumPy Savitzky-Golay filter (no scipy/OpenMP dependency)."""
    if len(x) < window or window < 3:
        return x.copy()
    m = window // 2
    t = np.arange(-m, m + 1, dtype=np.float64)
    A = np.vander(t, polyorder + 1, increasing=True)
    kernel = (A @ np.linalg.pinv(A))[m]
    pad = np.pad(x, ((m, m), (0, 0)), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for col in range(x.shape[1]):
        out[:, col] = np.convolve(pad[:, col], kernel[::-1], mode="valid")
    return out


def _estimate_initial_slope(x: np.ndarray, dt: float, n_frames: int = 6) -> np.ndarray:
    """Estimate robust t=0 velocity via ordinary least-squares slope over the first `n_frames`."""
    k = min(n_frames, len(x))
    if k < 2:
        return np.zeros(x.shape[1:], dtype=np.float64)
    t = np.arange(k, dtype=np.float64) * dt
    t_c = t - t.mean()
    denom = float(np.dot(t_c, t_c))
    if denom < 1e-12:
        return np.zeros(x.shape[1:], dtype=np.float64)
    flat = x[:k].reshape(k, -1)
    slope = (t_c @ (flat - flat.mean(axis=0, keepdims=True))) / denom
    return slope.reshape(x.shape[1:])


def _local_ankle_features(pelvis_pos: np.ndarray, pelvis_yaw: float, l_ankle: np.ndarray, r_ankle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Rz_T = _rotate_z_mat(pelvis_yaw).T
    origin = np.array([pelvis_pos[0], pelvis_pos[1], 0.0], dtype=np.float64)
    return Rz_T @ (l_ankle - origin), Rz_T @ (r_ankle - origin)


def load_lafan_walk_segments(
    lafan_dir: Path,
    fk_35: FastTreeKinematics,
    sdk_for_bfs: np.ndarray,
    body_names: list[str],
    resample_catmull_rom_fn,
    resample_quat_slerp_fn,
    ang_vel_fn,
    dt: float = 0.02,
    arm_smooth_window: int = 19,
    max_arm_vel: float = 4.5,
    max_arm_acc: float = 40.0,
    only_stand_buffer: bool = False,
    include_mirrored: bool = True,
) -> list[LafanWalkSegment]:
    """Load LAFAN1 G1 `walk*.csv`, suppress upper-body IK jitter, and extract clean straight walks."""
    g1_dir = lafan_dir / "g1"
    csv_files = sorted(g1_dir.glob("walk*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No `walk*.csv` files found under `{g1_dir}`. Run `scripts/sugar_rl/download_lafan1_g1.py` first."
        )

    torso_idx = body_names.index("torso_link")
    la_idx = body_names.index("left_ankle_roll_link")
    ra_idx = body_names.index("right_ankle_roll_link")
    lh_idx = body_names.index("left_rubber_hand")
    rh_idx = body_names.index("right_rubber_hand")
    arm_idxs = [i for i, n in enumerate(fk_35.env_joint_names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    mir_perm, mir_signs = build_bfs_sagittal_mirror_map(list(fk_35.env_joint_names))

    segments: list[LafanWalkSegment] = []
    for csv_path in csv_files:
        arr = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 36 or len(arr) < 120:
            continue

        pos_50_base = resample_catmull_rom_fn(arr[:, 0:3], 30.0, 50.0)
        quat_wxyz_30 = arr[:, [6, 3, 4, 5]]  # QX, QY, QZ, QW -> W, X, Y, Z
        quat_50_base = resample_quat_slerp_fn(quat_wxyz_30, 30.0, 50.0)
        q_bfs_50_base = resample_catmull_rom_fn(arr[:, 7:36][:, sdk_for_bfs], 30.0, 50.0)

        # Suppress high-frequency mocap/IK wrist & arm chatter while keeping legs untouched
        if arm_idxs and arm_smooth_window >= 3:
            q_bfs_50_base[:, arm_idxs] = _smooth_savgol_np(q_bfs_50_base[:, arm_idxs], window=arm_smooth_window, polyorder=2)

        variants = [(csv_path.name, pos_50_base, quat_50_base, q_bfs_50_base)]
        if include_mirrored:
            p_m, q_m, j_m = mirror_sagittal_pose_Arrays(pos_50_base, quat_50_base, q_bfs_50_base, mir_perm, mir_signs)
            variants.append((f"{csv_path.stem}__mirror.csv", p_m, q_m, j_m))

        for src_label, pos_50, quat_50, q_bfs_50 in variants:
            lin_vel = np.gradient(pos_50, dt, axis=0)
            ang_vel = ang_vel_fn(quat_50, dt)
            qd_bfs = np.gradient(q_bfs_50, dt, axis=0)
            arm_vel = np.abs(np.diff(q_bfs_50[:, arm_idxs], axis=0) / dt).max(axis=1) if arm_idxs else np.zeros(len(pos_50) - 1)
            arm_acc = np.abs(np.diff(q_bfs_50[:, arm_idxs], n=2, axis=0) / (dt * dt)).max(axis=1) if arm_idxs else np.zeros(len(pos_50) - 2)

            with torch.no_grad():
                fk_res = fk_35(
                    torch.from_numpy(pos_50.astype(np.float32)),
                    torch.from_numpy(quat_50.astype(np.float32)),
                    torch.from_numpy(q_bfs_50.astype(np.float32)),
                )
            bpos = fk_res.tracked_pos_w.numpy().astype(np.float64)
            bquat = fk_res.tracked_quat_w.numpy().astype(np.float64)

            rot_torso = _quat_to_rotmat_np(bquat[:, torso_idx])
            torso_up_z = rot_torso[:, 2, 2]
            yaw = np.unwrap(_quat_yaw(quat_50))
            v_xy = np.linalg.norm(lin_vel[:, :2], axis=-1)
            min_foot_z = np.minimum(bpos[:, la_idx, 2], bpos[:, ra_idx, 2])

            # 1. Extract natural walk -> decelerate -> compact double-support stand-still buffers
            #    (strictly 4..18 slow tail frames = 0.08..0.36s; no ~2s lingering idle stand!)
            # 2. Optionally extract double-support smooth straight-walking endings (both ankles <= 0.060m)
            T50 = len(pos_50)
            stand_ends: list[tuple[int, bool]] = []
            last_stand = -100
            for t_end in range(130, T50 - 5, 3):
                if t_end - last_stand < 5:
                    continue
                if (
                    float(v_xy[t_end - 8 : t_end].max()) < 0.22
                    and float(bpos[t_end - 1, la_idx, 2]) <= 0.058
                    and float(bpos[t_end - 1, ra_idx, 2]) <= 0.058
                ):
                    slow_tail = int((v_xy[t_end - 35 : t_end] < 0.22).sum())
                    pre_mean_v = float(v_xy[t_end - 45 : t_end - 16].mean())
                    if 4 <= slow_tail <= 18 and pre_mean_v >= 0.24:
                        stand_ends.append((t_end, True))
                        last_stand = t_end

            ds_ends: list[tuple[int, bool]] = []
            if not only_stand_buffer:
                last_ds = -100
                for t_end in range(135, T50 - 5, 4):
                    if t_end - last_ds < 6:
                        continue
                    if float(bpos[t_end - 1, la_idx, 2]) <= 0.060 and float(bpos[t_end - 1, ra_idx, 2]) <= 0.060:
                        ds_ends.append((t_end, False))
                        last_ds = t_end

            for t_end, is_stand in stand_ends + ds_ends:
                win_choices = (105, 120, 140, 160) if is_stand else (120, 105, 135)
                disp_bounds = (0.78, 1.75) if is_stand else (1.02, 1.62)
                for win_len in win_choices:
                    t_start = t_end - win_len
                    if t_start < 5:
                        continue
                    if float(arm_vel[t_start : t_end - 1].max()) > max_arm_vel:
                        continue
                    if float(arm_acc[t_start : t_end - 2].max()) > max_arm_acc:
                        continue
                    dvec = pos_50[t_end - 1, :2] - pos_50[t_start, :2]
                    disp = float(np.linalg.norm(dvec))
                    if not (disp_bounds[0] <= disp <= disp_bounds[1]):
                        continue
                    u_move = dvec / disp
                    s_proj = (pos_50[t_start:t_end, :2] - pos_50[t_start, :2]) @ u_move
                    max_back = float((np.maximum.accumulate(s_proj) - s_proj).max())
                    if max_back > 0.008:
                        continue
                    v_win = v_xy[t_start:t_end]
                    if is_stand:
                        if float(v_win[: win_len // 2].mean()) < 0.32 or float(v_win.max()) > 1.05:
                            continue
                    else:
                        if v_win.min() < 0.25 or v_win.max() > 1.05 or not (0.42 <= float(v_win.mean()) <= 0.82):
                            continue
                    if float(torso_up_z[t_start:t_end].min()) < 0.92:
                        continue
                    if float(np.ptp(yaw[t_start:t_end])) > np.radians(24.0 if is_stand else 20.0):
                        continue
                    move_dir = float(np.arctan2(dvec[1], dvec[0]))
                    heading_err = np.abs(((yaw[t_start:t_end] - move_dir + np.pi) % (2.0 * np.pi)) - np.pi)
                    if float(heading_err.max()) > np.radians(26.0 if is_stand else 22.0):
                        continue
                    foot_slice = min_foot_z[t_start:t_end]
                    if float(foot_slice.min()) < 0.028 or float(foot_slice.max()) > 0.062:
                        continue

                    # Ensure uncrossed ankles & arms in final frame so arms stay tucked near sides
                    end_Rz_T = _rotate_z_mat(float(yaw[t_end - 1])).T
                    lh_rel = end_Rz_T @ (bpos[t_end - 1, lh_idx] - pos_50[t_end - 1])
                    rh_rel = end_Rz_T @ (bpos[t_end - 1, rh_idx] - pos_50[t_end - 1])
                    if max(float(lh_rel[0]), float(rh_rel[0])) > 0.34 or float(lh_rel[1] - rh_rel[1]) < 0.24:
                        continue

                    end_idx = t_end - 1
                    la_loc, ra_loc = _local_ankle_features(
                        pos_50[end_idx], float(yaw[end_idx]), bpos[end_idx, la_idx], bpos[end_idx, ra_idx]
                    )
                    if float(la_loc[1] - ra_loc[1]) < 0.08:
                        continue
                    fwd_vec = np.array([np.cos(yaw[end_idx]), np.sin(yaw[end_idx])], dtype=np.float64)
                    v_fwd = float(np.dot(lin_vel[end_idx, :2], fwd_vec))

                    segments.append(
                        LafanWalkSegment(
                            source_name=src_label,
                            start_frame_50hz=t_start,
                            end_frame_50hz=t_end,
                            pos_w=pos_50[t_start:t_end].copy(),
                            quat_wxyz=quat_50[t_start:t_end].copy(),
                            q_bfs=q_bfs_50[t_start:t_end].copy(),
                            qd_bfs_end=qd_bfs[end_idx].copy(),
                            v_pelvis_end=lin_vel[end_idx].copy(),
                            w_pelvis_end=ang_vel[end_idx].copy(),
                            local_la_end=la_loc,
                            local_ra_end=ra_loc,
                            v_fwd_end=v_fwd,
                            walk_disp_m=disp,
                            is_stand_buffer=is_stand,
                        )
                    )
                    break

    if not segments:
        raise RuntimeError(f"Failed to extract any clean straight walking windows from `{g1_dir}`.")
    return segments


def _box_obb_surface_distance(
    pts_w: np.ndarray,
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    box_half_extents: np.ndarray,
) -> np.ndarray:
    """Compute unsigned outside distance from points `(..., 3)` to aligned box OBB surface (`<=0` means inside)."""
    rel = pts_w - box_pos
    local = np.einsum("ij,...j->...i", box_rot.T, rel)
    d = np.abs(local) - box_half_extents
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


def stitch_lafan_walk_to_omni_clip(
    omni_pos: np.ndarray,
    omni_quat_wxyz: np.ndarray,
    omni_q_bfs: np.ndarray,
    omni_bpos: np.ndarray,
    omni_obj_pos: np.ndarray,
    omni_obj_rot: np.ndarray,
    omni_contact: np.ndarray,
    box_half_extents: np.ndarray,
    segments: list[LafanWalkSegment],
    usage_counts: dict[str, int],
    fk_35: FastTreeKinematics,
    body_names: list[str],
    ang_vel_fn,
    blend_frames: int = 25,
    dt: float = 0.02,
    pin_stance_foot: bool = True,
    approach_yaw_offset_rad: float = 0.0,
    exclude_seg_indices: set[int] | None = None,
) -> dict:
    """Select phase-matched LAFAN1 walk segment and smoothly stitch it ahead of `omni[0]`."""
    pelvis_idx = body_names.index("pelvis")
    la_idx = body_names.index("left_ankle_roll_link")
    ra_idx = body_names.index("right_ankle_roll_link")
    torso_idx = body_names.index("torso_link")
    lh_idx = body_names.index("left_rubber_hand")
    rh_idx = body_names.index("right_rubber_hand")
    check_body_names = [
        "pelvis",
        "torso_link",
        "head_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ]
    check_idxs = [body_names.index(n) for n in check_body_names]
    lower_mask = np.array(
        [n in ("pelvis", "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link") for n in check_body_names],
        dtype=bool,
    )

    yaw_omni0 = float(_quat_yaw(omni_quat_wxyz[0]))
    la_omni, ra_omni = _local_ankle_features(omni_pos[0], yaw_omni0, omni_bpos[0, la_idx], omni_bpos[0, ra_idx])
    fwd_omni = np.array([np.cos(yaw_omni0), np.sin(yaw_omni0)], dtype=np.float64)

    # Estimate omni initial velocities safely via OLS slope over the first few frames (avoids single-frame noise)
    v_omni_0 = _estimate_initial_slope(omni_pos, dt, n_frames=6)
    qd_omni_0 = _estimate_initial_slope(omni_q_bfs, dt, n_frames=6)
    w_omni_0 = ang_vel_fn(omni_quat_wxyz[: min(6, len(omni_quat_wxyz))], dt)[0]
    v_fwd_omni0 = float(np.dot(v_omni_0[:2], fwd_omni))

    # Clamp endpoint joint rates to physical motor range so Cubic Hermite never overshoots wildly
    qd_omni_0_c = np.clip(qd_omni_0, -4.5, 4.5)
    v_omni_0_c = v_omni_0.copy()
    v_omni_0_c[2] = float(np.clip(v_omni_0_c[2], -0.25, 0.25))

    # Baseline omni[0] grounding & COM stability quick check
    omni0_min_fz = float(min(omni_bpos[0, la_idx, 2], omni_bpos[0, ra_idx, 2]))
    if not (0.024 <= omni0_min_fz <= 0.058):
        raise RuntimeError(f"Omni[0] min foot Z={omni0_min_fz:.4f}m violates ground-contact limit [0.024, 0.058]m.")
    omni0_margin = float(
        com_to_ankle_margin(
            omni_pos[0, :2], omni_bpos[0, torso_idx, :2], omni_bpos[0, la_idx, :2], omni_bpos[0, ra_idx, :2]
        )
    )
    if omni0_margin > 0.16:
        raise RuntimeError(f"Omni[0] COM-to-ankle margin={omni0_margin:.4f}m exceeds 0.16m limit.")

    ranked: list[tuple[float, int]] = []
    la_gnd0 = float(omni_bpos[0, la_idx, 2]) <= 0.055
    ra_gnd0 = float(omni_bpos[0, ra_idx, 2]) <= 0.055
    if la_gnd0 and ra_gnd0:
        proj_la = float(np.dot(omni_bpos[0, la_idx, :2] - omni_pos[0, :2], fwd_omni))
        proj_ra = float(np.dot(omni_bpos[0, ra_idx, :2] - omni_pos[0, :2], fwd_omni))
        stance_idx = la_idx if proj_la >= proj_ra else ra_idx
    else:
        stance_idx = la_idx if float(omni_bpos[0, la_idx, 2]) <= float(omni_bpos[0, ra_idx, 2]) else ra_idx
    other_idx = ra_idx if stance_idx == la_idx else la_idx
    v_foot_B = np.clip(_estimate_initial_slope(omni_bpos[:, stance_idx], dt, n_frames=6), -0.15, 0.15)
    v_other_B = np.clip(_estimate_initial_slope(omni_bpos[:, other_idx], dt, n_frames=6), -0.15, 0.15)
    prefer_stand = float(np.linalg.norm(v_foot_B[:2])) <= 0.18
    omni_stance_span = la_omni - ra_omni
    for idx, seg in enumerate(segments):
        seg_span = seg.local_la_end - seg.local_ra_end
        ankle_err = float(np.linalg.norm(seg.local_la_end - la_omni) + np.linalg.norm(seg.local_ra_end - ra_omni))
        span_err = float(np.linalg.norm(seg_span[:2] - omni_stance_span[:2]))
        pz_err = abs(float(seg.pos_w[-1, 2]) - float(omni_pos[0, 2]))
        vel_err = abs(seg.v_fwd_end - v_fwd_omni0)
        div_pen = 0.045 * usage_counts.get(f"{seg.source_name}:{seg.end_frame_50hz}", 0) + 0.015 * usage_counts.get(seg.source_name, 0)
        stand_adj = -0.35 if (prefer_stand and seg.is_stand_buffer) else (0.15 if (not prefer_stand and seg.is_stand_buffer) else 0.0)
        score = 0.45 * ankle_err + 3.2 * span_err + 0.8 * pz_err + 0.35 * vel_err + div_pen + stand_adj
        ranked.append((score, idx))
    ranked.sort(key=lambda x: x[0])

    # Baseline omni[0] hand/body clearance so we know how close the clip starts at t=0
    omni0_dists = _box_obb_surface_distance(
        omni_bpos[0, check_idxs], omni_obj_pos[0], omni_obj_rot[0], box_half_extents
    )
    min_allowed_hand_clr = max(0.015, min(0.04, float(omni0_dists.min()) - 0.005))

    # Precompute constant box pose for pre-contact sequence and try top phase-matched candidates
    T_dur = (blend_frames + 1) * dt
    u = np.arange(1, blend_frames + 1, dtype=np.float64) / float(blend_frames + 1)

    for score, seg_idx in ranked[:180]:
        if exclude_seg_indices is not None and seg_idx in exclude_seg_indices:
            continue
        seg = segments[seg_idx]
        yaw_w_end = float(_quat_yaw(seg.quat_wxyz[-1]))
        target_walk_yaw = yaw_omni0 + float(approach_yaw_offset_rad)
        d_yaw = float((target_walk_yaw - yaw_w_end + np.pi) % (2.0 * np.pi) - np.pi)
        Rz = _rotate_z_mat(d_yaw)
        qz_delta = np.array([np.cos(0.5 * d_yaw), 0.0, 0.0, np.sin(0.5 * d_yaw)], dtype=np.float64)

        # Rotate walking segment in SE(2) around its final frame
        rel_w = seg.pos_w - seg.pos_w[-1]
        pos_w_rot = (Rz @ rel_w.T).T
        # Quaternion left multiplication qz_delta * quat_wxyz
        aw, ax, ay, az = qz_delta
        bw, bx, by, bz = seg.quat_wxyz[:, 0], seg.quat_wxyz[:, 1], seg.quat_wxyz[:, 2], seg.quat_wxyz[:, 3]
        quat_w_rot = np.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            axis=-1,
        )
        if np.dot(quat_w_rot[-1], omni_quat_wxyz[0]) < 0.0:
            quat_w_rot = -quat_w_rot

        v_A = Rz @ seg.v_pelvis_end
        v_A[2] = float(np.clip(v_A[2], -0.25, 0.25))
        w_A = Rz @ seg.w_pelvis_end
        qd_A = np.clip(seg.qd_bfs_end, -4.5, 4.5)

        # Cubic Hermite blend for 29 DoFs and Lie-group SO(3) orientation first
        blend_q_base = _cubic_hermite(seg.q_bfs[-1], qd_A, omni_q_bfs[0], qd_omni_0_c, u, T_dur)

        R_A = _quat_to_rotmat_np(quat_w_rot[-1])
        R_B = _quat_to_rotmat_np(omni_quat_wxyz[0])
        theta_rel = _rotmat_to_rotvec(R_A.T @ R_B)
        v_theta_0 = R_A.T @ w_A
        v_theta_1 = R_A.T @ w_omni_0
        theta_u = _cubic_hermite(np.zeros(3), v_theta_0, theta_rel, v_theta_1, u, T_dur)
        R_blend = np.stack([R_A @ _rotvec_to_rotmat(tv) for tv in theta_u], axis=0)
        quat_blend = _rotmat_to_quat_wxyz(R_blend)
        if np.dot(quat_blend[0], quat_w_rot[-1]) < 0.0:
            quat_blend = -quat_blend

        # Build candidate anchor foot pairings when stance-pinning a walk-to-stand clip
        use_pin = pin_stance_foot and seg.is_stand_buffer and float(np.linalg.norm(v_foot_B[:2])) <= 0.18
        if use_pin:
            anchor_pairs: list[tuple[int, int, np.ndarray] | None] = [(stance_idx, other_idx, v_foot_B)]
            if float(omni_bpos[0, other_idx, 2]) <= 0.055 and float(np.linalg.norm(v_other_B[:2])) <= 0.18:
                anchor_pairs.append((other_idx, stance_idx, v_other_B))
        else:
            anchor_pairs = [None]

        for anchor_opt in anchor_pairs:
            blend_q = blend_q_base.copy()
            if anchor_opt is not None:
                s_idx, o_idx, v_s_B = anchor_opt
                # Closed-chain stance-foot pinning: weld grounded stance foot on floor so knees/hips naturally sit back
                q_chain = np.concatenate([seg.q_bfs[-4:], blend_q, omni_q_bfs[0:1]], axis=0)
                quat_chain = np.concatenate([quat_w_rot[-4:], quat_blend, omni_quat_wxyz[0:1]], axis=0)
                for ti in range(1, len(quat_chain)):
                    if np.dot(quat_chain[ti], quat_chain[ti - 1]) < 0.0:
                        quat_chain[ti] = -quat_chain[ti]
                zero_p = np.zeros((len(q_chain), 3), dtype=np.float32)
                with torch.no_grad():
                    fk_rel = fk_35(
                        torch.from_numpy(zero_p),
                        torch.from_numpy(quat_chain.astype(np.float32)),
                        torch.from_numpy(q_chain.astype(np.float32)),
                    )
                r_stance_all = fk_rel.tracked_pos_w[:, s_idx, :].numpy().astype(np.float64)
                r_other_all = fk_rel.tracked_pos_w[:, o_idx, :].numpy().astype(np.float64)
                foot_tail_w = pos_w_rot[-4:] + r_stance_all[:4]
                v_foot_A = np.clip(_estimate_initial_slope(foot_tail_w[::-1], -dt, n_frames=4), -0.10, 0.10)
                r_A = r_stance_all[3]
                r_blend = r_stance_all[4:-1]
                r_B = r_stance_all[-1]
                foot_B = omni_pos[0] + r_B
                if float(np.linalg.norm(v_foot_A[:2])) <= 0.12 and float(np.linalg.norm(v_s_B[:2])) <= 0.12:
                    foot_A_z = float(seg.pos_w[-1, 2]) + float(r_A[2])
                    s_smooth = 3.0 * (u * u) - 2.0 * (u * u * u)
                    foot_blend_z = (1.0 - s_smooth) * foot_A_z + s_smooth * float(foot_B[2])
                    foot_blend = np.column_stack([np.full_like(u, foot_B[0]), np.full_like(u, foot_B[1]), foot_blend_z])
                    foot_A = np.array([foot_B[0], foot_B[1], foot_A_z], dtype=np.float64)
                else:
                    vf_A, vf_B = v_foot_A, v_s_B
                    foot_A_xy = foot_B[:2] - 0.5 * (vf_A[:2] + vf_B[:2]) * T_dur
                    foot_A = np.array([foot_A_xy[0], foot_A_xy[1], float(seg.pos_w[-1, 2]) + float(r_A[2])], dtype=np.float64)
                    foot_blend = _cubic_hermite(foot_A, vf_A, foot_B, vf_B, u, T_dur)
                pos_A = foot_A - r_A
                # Reject pin placement that overshoots past omni[0] along forward heading
                if float(np.dot(pos_A[:2] - omni_pos[0, :2], fwd_omni)) > 0.002:
                    continue
                pos_w_aligned = pos_w_rot + pos_A
                blend_pos = foot_blend - r_blend

                # If non-stance foot must relocate > 1.8cm to reach target stance, step cleanly via knee flexion (zero floor drag!)
                other_A_xy = (pos_A + r_other_all[3])[:2]
                other_B_xy = (omni_pos[0] + r_other_all[-1])[:2]
                other_shift = float(np.linalg.norm(other_B_xy - other_A_xy))
                if other_shift > 0.018:
                    side = "left" if o_idx == la_idx else "right"
                    jnames = list(fk_35.env_joint_names)
                    j_hp = jnames.index(f"{side}_hip_pitch_joint")
                    j_kn = jnames.index(f"{side}_knee_joint")
                    j_ap = jnames.index(f"{side}_ankle_pitch_joint")
                    amp = float(np.clip(3.2 * other_shift + 0.24, 0.30, 0.52))
                    bell = np.sin(np.pi * u) ** 2
                    blend_q[:, j_kn] += amp * bell
                    blend_q[:, j_hp] -= 0.35 * amp * bell
                    blend_q[:, j_ap] -= 0.45 * amp * bell
            else:
                pos_A_xy = omni_pos[0, :2] - 0.5 * (v_A[:2] + v_omni_0_c[:2]) * T_dur
                pos_A = np.array([pos_A_xy[0], pos_A_xy[1], float(seg.pos_w[-1, 2])], dtype=np.float64)
                pos_w_aligned = pos_w_rot + pos_A
                blend_pos = _cubic_hermite(pos_A, v_A, omni_pos[0], v_omni_0_c, u, T_dur)

            pre_pos = np.concatenate([pos_w_aligned, blend_pos], axis=0)
            pre_quat = np.concatenate([quat_w_rot, quat_blend], axis=0)
            pre_q = np.concatenate([seg.q_bfs, blend_q], axis=0)

            # Ensure sign-continuous quaternions across W -> blend -> omni
            full_quat = np.concatenate([pre_quat, omni_quat_wxyz], axis=0)
            for t in range(1, len(full_quat)):
                if np.dot(full_quat[t], full_quat[t - 1]) < 0.0:
                    full_quat[t] = -full_quat[t]
            full_pos = np.concatenate([pre_pos, omni_pos], axis=0)
            full_q = np.concatenate([pre_q, omni_q_bfs], axis=0)

            # Forward kinematics check on pre_pos + pre_q segment (with full velocity recomputation next)
            with torch.no_grad():
                fk_pre = fk_35(
                    torch.from_numpy(pre_pos.astype(np.float32)),
                    torch.from_numpy(full_quat[: len(pre_pos)].astype(np.float32)),
                    torch.from_numpy(pre_q.astype(np.float32)),
                )
            bpos_pre = fk_pre.tracked_pos_w.numpy().astype(np.float64)
            bquat_pre = fk_pre.tracked_quat_w.numpy().astype(np.float64)

            # Verify strict ground foot height across approach & blend (never airborne > 5.8cm or sunk < 2.2cm)
            min_fz = np.minimum(bpos_pre[:, la_idx, 2], bpos_pre[:, ra_idx, 2])
            if float(min_fz.min()) < 0.022 or float(min_fz.max()) > 0.058:
                continue

            # Verify uncrossed arms & uncrossed legs throughout approach + blend!
            R_torso_pre = _quat_to_rotmat_np(bquat_pre[:, torso_idx])
            lh_loc_y = np.einsum("ti,ti->t", R_torso_pre[:, :, 1], bpos_pre[:, lh_idx] - bpos_pre[:, torso_idx])
            rh_loc_y = np.einsum("ti,ti->t", R_torso_pre[:, :, 1], bpos_pre[:, rh_idx] - bpos_pre[:, torso_idx])
            if float((lh_loc_y - rh_loc_y).min()) < 0.22:
                continue
            ank_diff = bpos_pre[:, la_idx, :2] - bpos_pre[:, ra_idx, :2]
            yaw_pre = _quat_yaw(full_quat[: len(pre_pos)])
            fwd_pre = np.stack([np.cos(yaw_pre), np.sin(yaw_pre)], axis=-1)
            left_pre = np.stack([-np.sin(yaw_pre), np.cos(yaw_pre)], axis=-1)
            dx_ank = np.abs(np.sum(ank_diff * fwd_pre, axis=-1))
            dy_ank = np.sum(ank_diff * left_pre, axis=-1)
            if np.any((dx_ank < 0.16) & (dy_ank < 0.04)):
                continue

            # Reject if any foot slides along the floor (> 2.6cm) WITHOUT lifting (> 6.0cm) across the blend window!
            seam_s = max(0, len(seg.pos_w) - 1)
            if anchor_opt is not None:
                la_blend_drift = float(np.linalg.norm(omni_bpos[0, la_idx, :2] - bpos_pre[seam_s, la_idx, :2]))
                ra_blend_drift = float(np.linalg.norm(omni_bpos[0, ra_idx, :2] - bpos_pre[seam_s, ra_idx, :2]))
                la_peak_z = float(bpos_pre[seam_s:, la_idx, 2].max())
                ra_peak_z = float(bpos_pre[seam_s:, ra_idx, 2].max())
                if la_blend_drift > 0.026 and la_peak_z < 0.062:
                    continue
                if ra_blend_drift > 0.026 and ra_peak_z < 0.062:
                    continue

            # Verify upper-body COM projection stays safely within <= 0.14m of the ankle support segment across seam
            seam_margins = com_to_ankle_margin(
                bpos_pre[seam_s:, pelvis_idx, :2],
                bpos_pre[seam_s:, torso_idx, :2],
                bpos_pre[seam_s:, la_idx, :2],
                bpos_pre[seam_s:, ra_idx, :2],
            )
            max_seam_margin = float(max(float(seam_margins.max()), omni0_margin))
            if max_seam_margin > 0.140:
                continue

            # Verify monotonic forward approach to box (zero backward retreat > 1.0cm & compact stand settle <= 28 frames)
            d_start_box = float(np.linalg.norm(pre_pos[0, :2] - omni_obj_pos[0, :2]))
            d_omni0_box = float(np.linalg.norm(omni_pos[0, :2] - omni_obj_pos[0, :2]))
            if d_start_box < d_omni0_box + 0.68:
                continue

            pre_and_o0_xy = np.vstack([pre_pos[:, :2], omni_pos[0:1, :2]])
            d_box_seq = np.linalg.norm(pre_and_o0_xy - omni_obj_pos[0, :2], axis=-1)
            max_retreat_m = float((d_box_seq - np.minimum.accumulate(d_box_seq)).max())
            if max_retreat_m > 0.010:
                continue

            if seg.is_stand_buffer:
                v_pre_xy = np.linalg.norm(np.gradient(pre_pos[:, :2], dt, axis=0), axis=-1)
                slow_pre_frames = int((v_pre_xy[-min(45, len(v_pre_xy)) :] < 0.18).sum())
                if slow_pre_frames > 28:
                    continue

            # Verify static-box clearance across entire approach & blend sequence
            pts_chk = bpos_pre[:, check_idxs, :]  # (T_pre, K, 3)
            dists = _box_obb_surface_distance(pts_chk, omni_obj_pos[0], omni_obj_rot[0], box_half_extents)
            min_lower_clr = float(dists[:, lower_mask].min())
            min_any_clr = float(dists.min())
            if min_lower_clr < 0.045 or min_any_clr < min_allowed_hand_clr:
                continue

            # Recompute unified velocities & 35-body FK across the complete stitched trajectory
            full_lin_vel = np.gradient(full_pos, dt, axis=0).astype(np.float32)
            full_ang_vel = ang_vel_fn(full_quat, dt).astype(np.float32)
            full_q_vel = np.gradient(full_q, dt, axis=0).astype(np.float32)
            with torch.no_grad():
                fk_full = fk_35(
                    torch.from_numpy(full_pos.astype(np.float32)),
                    torch.from_numpy(full_quat.astype(np.float32)),
                    torch.from_numpy(full_q.astype(np.float32)),
                    torch.from_numpy(full_lin_vel),
                    torch.from_numpy(full_ang_vel),
                    torch.from_numpy(full_q_vel),
                )

            T_pre = len(pre_pos)
            full_obj_pos = np.concatenate([np.repeat(omni_obj_pos[0:1], T_pre, axis=0), omni_obj_pos], axis=0)
            full_obj_rot = np.concatenate([np.repeat(omni_obj_rot[0:1], T_pre, axis=0), omni_obj_rot], axis=0)
            full_obj_lin_vel = np.gradient(full_obj_pos, dt, axis=0).astype(np.float32)
            obj_ang_vel_pre = np.zeros((T_pre, 3), dtype=np.float32)
            full_contact = np.concatenate([np.zeros(T_pre, dtype=bool), omni_contact.astype(bool)], axis=0)

            pre_walk_path = float(np.linalg.norm(np.diff(pre_pos[:, :2], axis=0), axis=-1).sum())
            usage_counts[seg.source_name] = usage_counts.get(seg.source_name, 0) + 1
            win_key = f"{seg.source_name}:{seg.end_frame_50hz}"
            usage_counts[win_key] = usage_counts.get(win_key, 0) + 1

            return {
                "joint_pos": full_q.astype(np.float32),
                "joint_vel": full_q_vel,
                "body_pos_w": fk_full.tracked_pos_w.numpy().astype(np.float32),
                "body_quat_w": fk_full.tracked_quat_w.numpy().astype(np.float32),
                "body_lin_vel_w": fk_full.tracked_lin_vel_w.numpy().astype(np.float32),
                "body_ang_vel_w": fk_full.tracked_ang_vel_w.numpy().astype(np.float32),
                "obj_pos": full_obj_pos.astype(np.float32),
                "obj_rot": full_obj_rot.astype(np.float32),
                "obj_lin_vel": full_obj_lin_vel,
                "obj_ang_vel_pre": obj_ang_vel_pre,
                "contact": full_contact,
                "meta": {
                    "seg_idx": int(seg_idx),
                    "lafan_source": seg.source_name,
                    "is_stand_buffer": bool(seg.is_stand_buffer),
                    "lafan_frames_50hz": int(len(seg.pos_w)),
                    "blend_frames_50hz": int(blend_frames),
                    "pre_walk_path_m": round(pre_walk_path, 3),
                    "phase_score": round(float(score), 4),
                    "min_box_clearance_m": round(min_any_clr, 4),
                    "min_foot_z_m": round(float(min_fz.min()), 4),
                    "max_blend_min_foot_z_m": round(float(max(float(min_fz[seam_s:].max()), omni0_min_fz)), 4),
                    "max_blend_com_margin_m": round(max_seam_margin, 4),
                    "max_approach_retreat_cm": round(max_retreat_m * 100.0, 2),
                },
            }

    raise RuntimeError(
        f"No collision-free & ground-safe LAFAN1 walk match found among top-150 candidates (best score={ranked[0][0]:.3f})."
    )

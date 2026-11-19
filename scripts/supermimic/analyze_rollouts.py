"""Quantify gait quality and box placement error of SUGAR Refiner rollouts (`raw_npz/trajectory_complete/*.npz`).

Runs locally with numpy only (torch optional, used only to read joint names from a checkpoint).

Windows PowerShell example:
    python scripts/supermimic/analyze_rollouts.py `
        --rollout_dir D:/data/CarryBox_sm_full/rollout_datasets/refiner/raw_npz `
        --ref_dir D:/data/CarryBox `
        --ckpt D:/data/CarryBox_sm_full/ckpts/refiner.pt `
        --out_json rollout_metrics.json

Metrics (per trajectory, aggregated as median / mean / p90):
- Jitter: action second difference RMS, joint acceleration RMS and >5 Hz joint-velocity power ratio,
  each next to the same metric of the reference joint trajectory (`ref_joint_pos`).
- Feet (ankle_roll links): slip speed while on the ground, swing clearance, steps per second.
- Posture: joint position RMS error to the reference (per group when joint names are available).
- Placement: final object error to the reference target; release frame from the simulated hand contacts;
  with `--ref_dir`, the final error split into (1) robot root drift at release, (2) box-relative-to-root error
  at release, (3) box displacement after release beyond the reference, plus release timing vs the reference.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np

DT = 0.02
LEFT_ANKLE, RIGHT_ANKLE = 3, 6  # indices in MotionCommandCfg.body_names (rollout body order)
ANCHOR_IDX = 7                  # `torso_link` index in MotionCommandCfg.body_names (14-body rollout order)
REF_ANCHOR_IDX = 10             # `torso_link` index in SUGAR_RAW_35_BODY_NAMES (35-body reference `robot_50hz.npz`)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    c = q.copy()
    c[..., 1:] *= -1.0
    return c / np.clip(np.sum(q * q, axis=-1, keepdims=True), 1e-9, None)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[..., 0:1] * t + np.cross(xyz, t)


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _hf_ratio(vel: np.ndarray, cutoff_hz: float = 5.0) -> float:
    """Share of joint-velocity spectral power above `cutoff_hz` (vel: (T, J))."""
    if vel.shape[0] < 16:
        return float("nan")
    v = vel - vel.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freqs = np.fft.rfftfreq(v.shape[0], d=DT)
    total = power[1:].sum()
    return float(power[freqs > cutoff_hz].sum() / total) if total > 0 else float("nan")


def _last_release(label: np.ndarray) -> int | None:
    lab = label.reshape(-1).astype(bool)
    idx = np.nonzero(lab[:-1] & ~lab[1:])[0]
    return int(idx[-1] + 1) if idx.size else None


def _feet_metrics(body_pos: np.ndarray, body_vel: np.ndarray) -> dict[str, float]:
    slips, clears, n_steps = [], [], 0
    for i in (LEFT_ANKLE, RIGHT_ANKLE):
        z = body_pos[:, i, 2]
        base = np.percentile(z, 5)
        contact = z < base + 0.02
        v_xy = np.linalg.norm(body_vel[:, i, :2], axis=-1)
        if contact.any():
            slips.append(float(v_xy[contact].mean()))
        # swing segments = runs of non-contact frames (>= 5 frames)
        swing = ~contact
        edges = np.diff(np.concatenate([[0], swing.astype(int), [0]]))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= 5:
                clears.append(float((z[s:e] - base).max()))
                n_steps += 1
    duration = body_pos.shape[0] * DT
    return {
        "foot_slip_mps": float(np.mean(slips)) if slips else float("nan"),
        "swing_clearance_m": float(np.median(clears)) if clears else float("nan"),
        "steps_per_s": n_steps / duration if duration > 0 else float("nan"),
    }


def _joint_groups(joint_names: list[str] | None, n: int) -> dict[str, np.ndarray]:
    if not joint_names or len(joint_names) != n:
        return {"all": np.arange(n)}
    legs = [i for i, s in enumerate(joint_names) if any(k in s for k in ("hip", "knee", "ankle"))]
    waist = [i for i, s in enumerate(joint_names) if "waist" in s]
    arms = [i for i in range(n) if i not in legs and i not in waist]
    return {"all": np.arange(n), "legs": np.array(legs), "waist": np.array(waist), "arms": np.array(arms)}


class _RefData:
    """Reference clips in MotionLoader order (`sorted(glob(data_*))`): motion_id in rollout names is that index."""

    def __init__(self, ref_dir: str) -> None:
        self.folders = sorted(glob.glob(os.path.join(ref_dir, "data_*")))
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, motion_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if motion_id not in self._cache:
            folder = self.folders[motion_id]
            with np.load(os.path.join(folder, "robot_50hz.npz")) as f:
                bpos = f["body_pos_w"]
                bquat = f["body_quat_w"]
                root = bpos[:, 0, :]            # pelvis = articulation root
                anc_p = bpos[:, REF_ANCHOR_IDX, :]  # torso_link = anchor
                anc_q = bquat[:, REF_ANCHOR_IDX, :]
            with open(os.path.join(folder, "obj_motion_global_50hz.pkl"), "rb") as f:
                obj = np.asarray(pickle.load(f)["obj_trans"])
            t = min(len(root), len(obj))
            self._cache[motion_id] = (root[:t], anc_p[:t], anc_q[:t], obj[:t])
        return self._cache[motion_id]


def analyze_one(path: str, groups: dict[str, np.ndarray], ref: _RefData | None) -> dict[str, float]:
    d = np.load(path, allow_pickle=True)
    out: dict[str, float] = {"file": os.path.basename(path)}
    action = d["action"]
    q, qd, q_ref = d["joint_pos"], d["joint_vel"], d["ref_joint_pos"]

    # Jitter
    out["action_d2_rms"] = _rms(np.diff(action, n=2, axis=0))
    out["joint_acc_rms"] = _rms(np.diff(qd, axis=0) / DT)
    out["ref_joint_acc_rms"] = _rms(np.diff(q_ref, n=2, axis=0) / DT**2)
    out["hf_ratio"] = _hf_ratio(qd)
    out["ref_hf_ratio"] = _hf_ratio(np.diff(q_ref, axis=0) / DT)

    # Feet
    out.update(_feet_metrics(d["body_pos_w"], d["body_lin_vel_w"]))

    # Posture
    for name, idx in groups.items():
        if idx.size:
            out[f"joint_err_rms_{name}"] = _rms(q[:, idx] - q_ref[:, idx])

    # Placement
    obj_w, root_w = d["obj_pos_w"], d["root_pos_w"]
    out["final_obj_err_m"] = float(np.linalg.norm(d["obj_pos_b"][-1] - d["target_obj_pos_b"][-1]))
    r = _last_release(d["hands_contact_label"])
    out["release_frame"] = float(r) if r is not None else float("nan")
    if r is not None:
        out["box_drop_after_release_m"] = float(obj_w[r, 2] - obj_w[-1, 2])
        out["box_xy_move_after_release_m"] = float(np.linalg.norm(obj_w[-1, :2] - obj_w[r, :2]))
        out["seg3_post_release_disp_m"] = float(np.linalg.norm(obj_w[-1] - obj_w[r]))
    if ref is not None and r is not None:
        motion_id = int(d["motion_id"])
        t_start = int(d["rollout_start"]) if "rollout_start" in d.files else 0
        ref_root, ref_anc_p, ref_anc_q, ref_obj = ref.get(motion_id)
        last = len(ref_obj) - 1
        k = min(t_start + r + 1, last)  # rollout sample i is recorded after `time_steps += 1` -> ref frame t_start+i+1
        out["seg1_root_drift_at_release_m"] = float(np.linalg.norm(root_w[r, :2] - ref_root[k, :2]))
        if "anchor_pos_w" in d.files and "anchor_quat_w" in d.files:
            anc_w = d["anchor_pos_w"][r]
            anc_q = d["anchor_quat_w"][r]
            out["seg1_anchor_drift_at_release_m"] = float(np.linalg.norm(anc_w - ref_anc_p[k]))
            dyaw = (_yaw_from_quat(anc_q) - _yaw_from_quat(ref_anc_q[k]) + np.pi) % (2.0 * np.pi) - np.pi
            out["seg1_anchor_yaw_drift_deg"] = float(np.degrees(np.abs(dyaw)))
            ref_obj_in_anc = _quat_apply(_quat_inv(ref_anc_q[k]), ref_obj[k] - ref_anc_p[k])
            out["seg2_box_in_anchor_err_at_release_m"] = float(np.linalg.norm(d["obj_pos_b"][r] - ref_obj_in_anc))
        out["seg2_box_rel_root_err_at_release_m"] = float(
            np.linalg.norm((obj_w[r] - root_w[r]) - (ref_obj[k] - ref_root[k]))
        )
        out["seg3_post_release_excess_m"] = float(
            np.linalg.norm((obj_w[-1] - obj_w[r]) - (ref_obj[last] - ref_obj[k]))
        )
        out["box_height_above_ref_end_at_release_m"] = float(obj_w[r, 2] - ref_obj[last, 2])
        out["final_obj_err_world_m"] = float(np.linalg.norm(obj_w[-1] - ref_obj[last]))
        r_ref = _last_release(d["ref_contact_label"])
        if r_ref is not None:
            out["release_minus_ref_frames"] = float(r - r_ref)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout_dir", required=True, help="raw_npz dir (or its trajectory_complete subdir)")
    ap.add_argument("--ref_dir", default=None, help="Reference motion folder used for the rollout (data/<Task>)")
    ap.add_argument("--ckpt", default=None, help="SuperMimic checkpoint, only to read joint names (needs torch)")
    ap.add_argument("--out_json", default=None, help="Write aggregate + per-trajectory metrics here")
    args = ap.parse_args()

    root = args.rollout_dir
    sub = os.path.join(root, "trajectory_complete")
    files = sorted(glob.glob(os.path.join(sub if os.path.isdir(sub) else root, "*.npz")))
    if not files:
        # Exit 0 so the caller can report the zero-trajectory case with its own hint
        print(f"[analyze_rollouts] WARNING: no complete .npz rollouts under {root}, skip metrics.")
        return

    joint_names = None
    if args.ckpt:
        import torch

        joint_names = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("joint_names")
    n_joints = np.load(files[0], allow_pickle=True)["joint_pos"].shape[-1]
    groups = _joint_groups(joint_names, n_joints)
    ref = _RefData(args.ref_dir) if args.ref_dir else None

    rows = [analyze_one(p, groups, ref) for p in files]
    keys = [k for k in rows[0] if k != "file"]
    for row in rows[1:]:
        keys += [k for k in row if k != "file" and k not in keys]

    summary = {}
    print(f"{len(rows)} trajectories from {root}")
    print(f"{'metric':38s} {'median':>10s} {'mean':>10s} {'p90':>10s}")
    for k in keys:
        vals = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        summary[k] = {
            "median": float(np.median(vals)),
            "mean": float(vals.mean()),
            "p90": float(np.percentile(vals, 90)),
            "n": int(vals.size),
        }
        s = summary[k]
        print(f"{k:38s} {s['median']:10.4f} {s['mean']:10.4f} {s['p90']:10.4f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_trajectory": rows}, f, indent=2)
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()

"""Pure-Kinematic Local Viewer & MP4 Exporter for Raw SUGAR Reference Datasets (`data/<TASK>`).

Zero policy checkpoint (`tracker.pt`) and zero Isaac Sim dependencies!
Directly sets `mj_data.qpos` from `robot_50hz.npz` + `obj_motion_global_50hz.pkl` and runs `mj_forward`
so you can inspect raw retargeted trajectories, knee angles, foot-ground contact/penetration, and object poses.

Interactive `mujoco.viewer` shortcuts:
  - `Space` : Pause / Resume playback
  - `N`     : Next trajectory (`data_xxx`)
  - `P`     : Previous trajectory (`data_xxx`)
  - `R`     : Restart current trajectory from frame 0
  - `Right` / `Left` (when paused): Step +1 / -1 frame
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

_FUSION_ROOT = Path(__file__).resolve().parents[2]
if str(_FUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_FUSION_ROOT))

import mujoco
import numpy as np

from scripts.sim2sim.policy_runner import load_single_motion_clip
from scripts.sim2sim.scene_builder import (
    G1_ISAACLAB_BFS_JOINT_NAMES,
    SIM2SIM_TASK_SPECS,
    build_mujoco_scene_xml,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kinematic Replay of Raw SUGAR Datasets in Local MuJoCo.")
    p.add_argument("--task", type=str, default="CarryBox", help=f"Task ({', '.join(SIM2SIM_TASK_SPECS.keys())}).")
    p.add_argument("--motion_folder", type=str, default=None, help="Path to `data/<TASK>` (default: `data/<task>`).")
    p.add_argument("--descriptions_dir", type=str, default=None, help="Path to `descriptions/` (auto-detected if present).")
    p.add_argument("--motion_index", type=int, default=0, help="Starting trajectory index (default: 0).")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 0.5 for half-speed).")
    p.add_argument("--headless", action="store_true", help="Run offscreen without GUI window.")
    p.add_argument("--record_mp4", type=str, default=None, help="Optional path to export side+oblique dual-view MP4.")
    p.add_argument("--num_clips", type=int, default=1, help="Number of clips to record back-to-back in `--record_mp4`.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    motion_folder = Path(args.motion_folder) if args.motion_folder else (_FUSION_ROOT / "data" / args.task)
    if not motion_folder.exists():
        raise FileNotFoundError(f"Motion folder not found: {motion_folder}")

    desc_dir = Path(args.descriptions_dir) if args.descriptions_dir else (_FUSION_ROOT / "descriptions")
    if not desc_dir.exists():
        desc_dir = None

    sub_dirs = sorted([d for d in motion_folder.glob("data_*") if d.is_dir()])
    if not sub_dirs:
        if (motion_folder / "robot_50hz.npz").exists():
            sub_dirs = [motion_folder]
        else:
            raise FileNotFoundError(f"No `data_*` subfolders found in {motion_folder}")

    xml_path, spec, asset_report = build_mujoco_scene_xml(
        task_name=args.task,
        descriptions_dir=desc_dir,
        sim_dt=0.002,
        motion_folder=motion_folder,
    )
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)

    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    lf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    target_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_vis")
    target_mocap_id = int(mj_model.body_mocapid[target_id]) if target_id >= 0 else -1

    pelvis_qpos_adr = int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_freejoint")])
    obj_qpos_adr = int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")])

    bfs_qpos_adrs = np.array(
        [int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in G1_ISAACLAB_BFS_JOINT_NAMES],
        dtype=np.int64,
    )
    lknee_bfs = G1_ISAACLAB_BFS_JOINT_NAMES.index("left_knee_joint")
    rknee_bfs = G1_ISAACLAB_BFS_JOINT_NAMES.index("right_knee_joint")

    def set_kinematic_frame(clip, t: int) -> dict[str, float]:
        t = int(np.clip(t, 0, clip.length - 1))
        mj_data.qpos[pelvis_qpos_adr : pelvis_qpos_adr + 3] = clip.root_pos_w[t].numpy()
        mj_data.qpos[pelvis_qpos_adr + 3 : pelvis_qpos_adr + 7] = clip.root_quat_w[t].numpy()
        mj_data.qpos[bfs_qpos_adrs] = clip.joint_pos_bfs[t].numpy()
        mj_data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = clip.obj_pos_w[t].numpy()
        mj_data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = clip.obj_quat_w[t].numpy()
        if target_mocap_id >= 0:
            mj_data.mocap_pos[target_mocap_id] = clip.obj_pos_w[-1].numpy()
            mj_data.mocap_quat[target_mocap_id] = clip.obj_quat_w[-1].numpy()
        mujoco.mj_forward(mj_model, mj_data)
        q_bfs = clip.joint_pos_bfs[t].numpy()
        return {
            "lknee_deg": float(np.degrees(q_bfs[lknee_bfs])),
            "rknee_deg": float(np.degrees(q_bfs[rknee_bfs])),
            "pelvis_z": float(mj_data.xpos[pelvis_id, 2]),
            "lf_z": float(mj_data.xpos[lf_id, 2]),
            "rf_z": float(mj_data.xpos[rf_id, 2]),
            "contact": float(clip.contact_label[t, 0].item()),
        }

    print("=" * 80)
    print(f"[Raw Kinematic Replay] Task: {args.task} | Clips found: {len(sub_dirs)} in {motion_folder}")
    print(f"[Raw Kinematic Replay] Robot geometry : {asset_report['robot_geometry']}")
    print(f"[Raw Kinematic Replay] Object geometry: {asset_report['object_geometry']}")
    print("=" * 80)

    if args.record_mp4:
        import imageio

        mp4_path = Path(args.record_mp4).resolve()
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        # Dual-view side-by-side: Left = pure sagittal side view (best for seeing knee bend!), Right = oblique 3D view
        w_half, h_full = 640, 720
        renderer = mujoco.Renderer(mj_model, height=h_full, width=w_half)

        cam_side = mujoco.MjvCamera()
        cam_side.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam_side.trackbodyid = pelvis_id
        cam_side.distance = 2.3
        cam_side.elevation = -6.0
        cam_side.azimuth = 90.0  # Pure lateral profile to inspect knee flexion & foot ground clearance

        cam_oblique = mujoco.MjvCamera()
        cam_oblique.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam_oblique.trackbodyid = pelvis_id
        cam_oblique.distance = 2.6
        cam_oblique.elevation = -18.0
        cam_oblique.azimuth = 135.0

        fps = max(1, int(round(50.0 * args.speed)))
        writer = imageio.get_writer(str(mp4_path), fps=fps)
        start_idx = args.motion_index % len(sub_dirs)
        for c_off in range(args.num_clips):
            c_idx = (start_idx + c_off) % len(sub_dirs)
            clip = load_single_motion_clip(motion_folder, motion_index=c_idx, device="cpu")
            print(f"[Recording] Clip #{c_idx} ({clip.clip_name}, {clip.length} frames = {clip.length * 0.02:.2f}s)...")
            for t in range(clip.length):
                metrics = set_kinematic_frame(clip, t)
                renderer.update_scene(mj_data, camera=cam_side)
                img_side = renderer.render().copy()
                renderer.update_scene(mj_data, camera=cam_oblique)
                img_obl = renderer.render().copy()
                frame_img = np.concatenate([img_side, img_obl], axis=1)
                writer.append_data(frame_img)
                if t % 50 == 0 or t == clip.length - 1:
                    print(
                        f"  Frame {t:3d}/{clip.length} ({t*0.02:4.2f}s) | contact={int(metrics['contact'])} | "
                        f"L_knee={metrics['lknee_deg']:5.1f} deg, R_knee={metrics['rknee_deg']:5.1f} deg | "
                        f"pelvis_z={metrics['pelvis_z']:.3f}m | ankle_z=({metrics['lf_z']:.3f}, {metrics['rf_z']:.3f})m"
                    )
        writer.close()
        renderer.close()
        print(f"[Done] Saved dual-view raw dataset replay MP4 -> {mp4_path}")
        if args.headless:
            return

    if args.headless:
        clip = load_single_motion_clip(motion_folder, motion_index=args.motion_index, device="cpu")
        for t in range(0, clip.length, 25):
            m = set_kinematic_frame(clip, t)
            print(
                f"Frame {t:3d}/{clip.length} | contact={int(m['contact'])} | "
                f"L_knee={m['lknee_deg']:5.1f} deg, R_knee={m['rknee_deg']:5.1f} deg | pelvis_z={m['pelvis_z']:.3f}m"
            )
        return

    from mujoco import viewer as mj_viewer

    state = {
        "clip_idx": args.motion_index % len(sub_dirs),
        "frame": 0,
        "paused": False,
        "reload": True,
    }
    clip = load_single_motion_clip(motion_folder, motion_index=state["clip_idx"], device="cpu")

    def _on_key(keycode: int) -> None:
        if keycode == 32:  # Space
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):  # R/r
            state["frame"] = 0
        elif keycode in (78, 110):  # N/n
            state["clip_idx"] = (state["clip_idx"] + 1) % len(sub_dirs)
            state["reload"] = True
        elif keycode in (80, 112):  # P/p
            state["clip_idx"] = (state["clip_idx"] - 1) % len(sub_dirs)
            state["reload"] = True
        elif keycode == 262 and state["paused"]:  # Right arrow
            state["frame"] += 1
        elif keycode == 263 and state["paused"]:  # Left arrow
            state["frame"] = max(0, state["frame"] - 1)

    print("[Viewer] Controls: `Space`=Pause, `N`=Next clip, `P`=Prev clip, `R`=Restart, `Left/Right`=Step frame")
    dt_sleep = 0.02 / max(0.05, args.speed)
    with mj_viewer.launch_passive(mj_model, mj_data, key_callback=_on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 110.0

        while viewer.is_running():
            t0 = time.perf_counter()
            if state["reload"]:
                clip = load_single_motion_clip(motion_folder, motion_index=state["clip_idx"], device="cpu")
                state["frame"] = 0
                state["reload"] = False
                print(f"\n>>> Switched to Clip #{state['clip_idx']}: {clip.clip_name} ({clip.length} frames)")

            m = set_kinematic_frame(clip, state["frame"])
            viewer.sync()

            if not state["paused"]:
                if state["frame"] % 25 == 0:
                    print(
                        f"[{clip.clip_name} f={state['frame']:3d}/{clip.length}] "
                        f"contact={int(m['contact'])} | L_knee={m['lknee_deg']:5.1f} deg, R_knee={m['rknee_deg']:5.1f} deg | "
                        f"pelvis_z={m['pelvis_z']:.3f}m | ankle_z=({m['lf_z']:.3f}, {m['rf_z']:.3f})m"
                    )
                state["frame"] = (state["frame"] + 1) % clip.length

            rem = dt_sleep - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)


if __name__ == "__main__":
    main()

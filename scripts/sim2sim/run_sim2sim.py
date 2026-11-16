"""Unified CLI Entry Point for Local Windows/Linux MuJoCo Sim-to-Sim Deployment.

Supports all 6 SUGAR loco-manipulation tasks (`CarryBox`, `PushBox`, `KickBox`, `PickBottle`, `StandBottle`, `SitChair`)
and both Sim-to-Sim verification modes:
  1. `--mode replay`     : Single `tracker.pt` tracking a reference trajectory from `data/<TASK>`
  2. `--mode autonomous` : Closed-loop `generator.ckpt` (10Hz) + `tracker.pt` (50Hz) + MuJoCo (500Hz)

Interactive `mujoco.viewer` controls:
  - Press `R` (or Backspace) : Reset robot & object to initial pose
  - Press `Space`            : Pause / Resume physics
  - Double-click green ghost target + `Ctrl + Right-Drag` : Move target goal live during autonomous run!
  - Double-click robot/box + `Ctrl + Right-Drag`          : Apply external push perturbations!
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
import torch

from scripts.sim2sim.policy_runner import MuJoCoSim2SimRunner  # noqa: E402
from scripts.sim2sim.scene_builder import SIM2SIM_TASK_SPECS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local MuJoCo Sim-to-Sim Deployment for SuperMimic x SUGAR (All 6 Tasks)."
    )
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help=f"SUGAR task name ({', '.join(SIM2SIM_TASK_SPECS.keys())}).",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["replay", "autonomous"],
        default=None,
        help="Execution mode: 'replay' (Tracker only) or 'autonomous' (Generator + Tracker). Auto-inferred if omitted.",
    )
    p.add_argument("--tracker_ckpt", type=str, required=True, help="Path to trained 510-D `tracker.pt`.")
    p.add_argument("--generator_ckpt", type=str, default=None, help="Path to trained `generator.ckpt` (for autonomous mode).")
    p.add_argument("--descriptions_dir", type=str, default=None, help="Optional path to extracted `descriptions/` folder.")
    p.add_argument("--motion_folder", type=str, default=None, help="Path to `data/<TASK>` (required for replay; optional for autonomous).")
    p.add_argument("--motion_index", type=int, default=0, help="Trajectory index inside `--motion_folder` (default: 0).")

    p.add_argument("--obj_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Custom initial object position (m).")
    p.add_argument("--obj_quat", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"), help="Custom initial object quaternion (wxyz).")
    p.add_argument("--target_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Custom target object position (m).")
    p.add_argument("--target_quat", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"), help="Custom target object quaternion (wxyz).")

    p.add_argument("--sim_dt", type=float, default=0.002, help="MuJoCo physics timestep in seconds (default: 0.002 = 500Hz).")
    p.add_argument("--control_dt", type=float, default=0.02, help="Policy control timestep in seconds (default: 0.02 = 50Hz).")
    p.add_argument("--generator_interval", type=int, default=20, help="Generator replanning interval in 50Hz steps (default: 20 = 0.4s).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch inference device.")

    p.add_argument("--steps", type=int, default=None, help="Max 50Hz control steps to run (default: unlimited in viewer, 500 in headless/video).")
    p.add_argument("--headless", action="store_true", help="Run without opening interactive 3D GUI window.")
    p.add_argument("--no_realtime", action="store_true", help="Run as fast as possible in viewer without sleeping to match wall-clock.")
    p.add_argument("--record_mp4", type=str, default=None, help="Optional output path to record an offscreen 50fps `.mp4` video.")
    p.add_argument("--video_width", type=int, default=1280, help="Recorded video width (default: 1280).")
    p.add_argument("--video_height", type=int, default=720, help="Recorded video height (default: 720).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    mode = args.mode
    if mode is None:
        mode = "autonomous" if args.generator_ckpt is not None else "replay"

    runner = MuJoCoSim2SimRunner(
        task_name=args.task,
        tracker_ckpt=args.tracker_ckpt,
        mode=mode,
        generator_ckpt=args.generator_ckpt,
        descriptions_dir=args.descriptions_dir,
        motion_folder=args.motion_folder,
        motion_index=args.motion_index,
        sim_dt=args.sim_dt,
        control_dt=args.control_dt,
        generator_call_interval=args.generator_interval,
        device=args.device,
    )

    custom_obj_pos = tuple(args.obj_pos) if args.obj_pos is not None else None
    custom_obj_quat = tuple(args.obj_quat) if args.obj_quat is not None else None
    custom_target_pos = tuple(args.target_pos) if args.target_pos is not None else None
    custom_target_quat = tuple(args.target_quat) if args.target_quat is not None else None

    runner.reset(
        custom_obj_pos=custom_obj_pos,
        custom_obj_quat=custom_obj_quat,
        custom_target_pos=custom_target_pos,
        custom_target_quat=custom_target_quat,
    )

    print("=" * 78)
    print(f"[Sim2Sim] Task             : {runner.spec.task_name} (mode='{runner.mode}')")
    print(f"[Sim2Sim] Tracker format   : {runner.tracker.ckpt_format} | device={runner.device}")
    print(f"[Sim2Sim] Robot geometry   : {runner.asset_report['robot_geometry']}")
    print(f"[Sim2Sim] Object geometry  : {runner.asset_report['object_geometry']}")
    print(f"[Sim2Sim] Scene XML        : {runner.asset_report['scene_xml']}")
    print(f"[Sim2Sim] Timing           : policy={1.0 / runner.control_dt:.0f}Hz, physics={1.0 / runner.sim_dt:.0f}Hz (decimation={runner.decimation})")
    if runner.clip is not None:
        print(f"[Sim2Sim] Loaded clip      : {runner.clip.clip_name} ({runner.clip.length} frames @ 50Hz)")
    print("=" * 78)

    # Optional offscreen MP4 recorder
    mp4_writer = None
    offscreen_renderer = None
    offscreen_cam = None
    if args.record_mp4:
        import imageio

        mp4_path = Path(args.record_mp4).resolve()
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        offscreen_renderer = mujoco.Renderer(runner.mj_model, height=args.video_height, width=args.video_width)
        offscreen_cam = mujoco.MjvCamera()
        offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        offscreen_cam.trackbodyid = runner.pelvis_body_id
        offscreen_cam.distance = 2.6
        offscreen_cam.elevation = -18.0
        offscreen_cam.azimuth = 135.0
        mp4_writer = imageio.get_writer(str(mp4_path), fps=int(round(1.0 / runner.control_dt)))
        print(f"[Sim2Sim] Recording MP4 to : {mp4_path}")

    def _capture_frame() -> None:
        if offscreen_renderer is not None and mp4_writer is not None:
            offscreen_renderer.update_scene(runner.mj_data, camera=offscreen_cam)
            mp4_writer.append_data(offscreen_renderer.render())

    max_steps = args.steps
    if max_steps is None and (args.headless or args.record_mp4):
        max_steps = runner.clip.length if (runner.mode == "replay" and runner.clip is not None) else 500

    min_pelvis_z = 999.0
    t_wall_start = time.perf_counter()
    info: dict[str, float] = {}

    try:
        if args.headless:
            assert max_steps is not None
            _capture_frame()
            for s in range(max_steps):
                info = runner.step()
                min_pelvis_z = min(min_pelvis_z, info["pelvis_z"])
                _capture_frame()
                if (s + 1) % 50 == 0 or (s + 1) == max_steps:
                    elapsed = max(1e-6, time.perf_counter() - t_wall_start)
                    fps = (s + 1) / elapsed
                    print(
                        f"[Step {s + 1:4d}/{max_steps}] t={info['sim_time_s']:5.2f}s | "
                        f"pelvis_z={info['pelvis_z']:.3f}m | obj_z={info['obj_z']:.3f}m | "
                        f"goal_dist={info['goal_dist_m']:.3f}m | {fps:6.1f} Hz"
                    )
        else:
            from mujoco import viewer as mj_viewer

            state_flags = {"reset_requested": False, "paused": False}

            def _on_key(keycode: int) -> None:
                # ASCII 'R'/'r' (82/114) -> reset; Space (32) -> pause toggle
                if keycode in (82, 114):
                    state_flags["reset_requested"] = True
                elif keycode == 32:
                    state_flags["paused"] = not state_flags["paused"]

            print("[Sim2Sim] Launching interactive 3D viewer (`R`: Reset, `Space`: Pause, `Ctrl+Drag` green target to move goal)...")
            with mj_viewer.launch_passive(runner.mj_model, runner.mj_data, key_callback=_on_key) as viewer:
                viewer.cam.lookat[:] = np.array([0.4, 0.0, 0.75], dtype=np.float64)
                viewer.cam.distance = 2.8
                viewer.cam.elevation = -18.0
                viewer.cam.azimuth = 135.0

                _capture_frame()
                while viewer.is_running():
                    if max_steps is not None and runner.step_count >= max_steps:
                        break
                    step_t0 = time.perf_counter()

                    if state_flags["reset_requested"]:
                        state_flags["reset_requested"] = False
                        runner.reset(
                            custom_obj_pos=custom_obj_pos,
                            custom_obj_quat=custom_obj_quat,
                            custom_target_pos=custom_target_pos,
                            custom_target_quat=custom_target_quat,
                        )
                        viewer.sync()
                        print("[Sim2Sim] Environment reset.")
                        continue

                    if state_flags["paused"]:
                        viewer.sync()
                        time.sleep(0.02)
                        continue

                    info = runner.step()
                    min_pelvis_z = min(min_pelvis_z, info["pelvis_z"])
                    viewer.sync()
                    _capture_frame()

                    if runner.step_count % 50 == 0:
                        print(
                            f"[Step {runner.step_count:4d}] t={info['sim_time_s']:5.2f}s | "
                            f"pelvis_z={info['pelvis_z']:.3f}m | obj_z={info['obj_z']:.3f}m | "
                            f"goal_dist={info['goal_dist_m']:.3f}m | contact={info['contact_cmd']:.2f}"
                        )

                    # Auto-loop reference clip in replay mode if `--steps` was not capped
                    if (
                        max_steps is None
                        and runner.mode == "replay"
                        and runner.clip is not None
                        and runner.step_count >= runner.clip.length
                    ):
                        runner.reset(
                            custom_obj_pos=custom_obj_pos,
                            custom_obj_quat=custom_obj_quat,
                            custom_target_pos=custom_target_pos,
                            custom_target_quat=custom_target_quat,
                        )

                    if not args.no_realtime:
                        rem = runner.control_dt - (time.perf_counter() - step_t0)
                        if rem > 0.0:
                            time.sleep(rem)
    finally:
        if mp4_writer is not None:
            mp4_writer.close()
        if offscreen_renderer is not None:
            offscreen_renderer.close()

    total_elapsed = max(1e-6, time.perf_counter() - t_wall_start)
    if runner.step_count > 0:
        print("=" * 78)
        print(
            f"[Sim2Sim Summary] Completed {runner.step_count} steps ({runner.step_count * runner.control_dt:.2f}s sim) | "
            f"min_pelvis_z={min_pelvis_z:.3f}m | final_goal_dist={info.get('goal_dist_m', 0.0):.3f}m | "
            f"avg_rate={runner.step_count / total_elapsed:.1f} Hz"
        )
        print("=" * 78)


if __name__ == "__main__":
    main()

"""Rollout & Closed-Loop Inference player for SuperMimic policies in SUGAR environments.

Supports all three evaluation / data-collection modes identically to `scripts/sugar_rl/play.py`:
1. Stage 2 Refiner Rollout (`Sugar-G129dof-*-Refiner-Rollout`) -> generates `raw_npz` for `process_refiner_rollout.py`
2. Stage 4 Tracker Rollout (`Sugar-G129dof-*-Tracker-Rollout`) -> generates `raw_npz` for `process_tracker_rollout.py`
3. Stage 6 Closed-Loop Inference (`Sugar-G129dof-*-Inference` + `--generator_checkpoint`) -> Diffusion + SuperMimic Tracker!
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play / Rollout / Inference with a trained SuperMimic policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record MP4 video during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Isaac Sim Fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="SUGAR Gymnasium task ID.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained SuperMimic `.pt` checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match real-time dt.")
parser.add_argument("--replay", action="store_true", default=False, help="Force RobotPlayEnvCfg entry point.")

# SUGAR Rollout & Diffusion Generator options
parser.add_argument("--generator_checkpoint", type=str, default=None, help="Path to trained `sugar_il` generator `.ckpt`.")
parser.add_argument("--eval_random_motion", action="store_true", default=False, help="Randomly sample test motions.")
parser.add_argument("--eval_mode", action="store_true", default=False, help="Enable evaluation mode.")
parser.add_argument("--eval_max_time", type=int, default=None, help="Maximum evaluation time steps.")
parser.add_argument("--rollout_dir", type=str, default=None, help="Output directory for rollout `raw_npz` trajectories.")
parser.add_argument("--motion_folder", type=str, default=None, help="Path to motion dataset folder.")
parser.add_argument("--teacher_motion_folder", type=str, default=None, help="Path to teacher motion folder.")
parser.add_argument("--teacher_ckpt", type=str, default=None, help="Unused compatibility flag with original SUGAR CLI.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-Omniverse imports
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import sugar_rl.tasks  # noqa: F401
from sugar_rl.supermimic.trainer import load_supermimic_policy
from sugar_rl.utils.parser_cfg import parse_env_cfg


def main() -> None:
    if args_cli.replay:
        task_spec = gym.spec(args_cli.task)
        if task_spec and "play_env_cfg_entry_point" in task_spec.kwargs:
            module_path = task_spec.kwargs["play_env_cfg_entry_point"].rsplit(":", 1)[0]
            task_spec.kwargs["play_env_cfg_entry_point"] = f"{module_path}:RobotPlayEnvCfg"

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        m_cfg = env_cfg.commands.motion
        if args_cli.generator_checkpoint and hasattr(m_cfg, "generator_checkpoint_path"):
            m_cfg.generator_checkpoint_path = args_cli.generator_checkpoint
        if args_cli.eval_random_motion and hasattr(m_cfg, "eval_random_motion"):
            m_cfg.eval_random_motion = True
        if args_cli.eval_mode and hasattr(m_cfg, "eval_mode"):
            m_cfg.eval_mode = True
        if args_cli.eval_max_time is not None and hasattr(m_cfg, "eval_max_time"):
            m_cfg.eval_max_time = args_cli.eval_max_time
        if args_cli.rollout_dir is not None:
            m_cfg.rollout_dir = args_cli.rollout_dir
        if args_cli.motion_folder is not None:
            m_cfg.motion_folder = args_cli.motion_folder
        if args_cli.teacher_motion_folder is not None:
            m_cfg.teacher_motion_folder = args_cli.teacher_motion_folder

        from sugar_rl.assets.objects.objects import apply_dataset_box_spec

        apply_dataset_box_spec(env_cfg, m_cfg.motion_folder, m_cfg.teacher_motion_folder)

    ckpt_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.abspath(ckpt_path))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[SuperMimic Play] Recording video:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    device = env.unwrapped.device
    print(f"[SuperMimic Play] Loading policy from: {ckpt_path}")
    policy = load_supermimic_policy(ckpt_path, device=device)

    dt = env.unwrapped.step_dt
    obs_dict, _ = env.reset()
    obs_pol = obs_dict["policy"]

    timestep = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.no_grad():
                actions = policy.act_inference(obs_pol)
                obs_dict, _, _, _, _ = env.step(actions)
                obs_pol = obs_dict["policy"]

            if args_cli.video:
                timestep += 1
                if timestep >= args_cli.video_length:
                    print(f"[SuperMimic Play] Finished recording {timestep} video steps.")
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except SystemExit as exit_exc:
        # `MotionCommand` raises SystemExit once all rollout trajectories have been saved cleanly
        print(f"[SuperMimic Play] Rollout completed cleanly: {exit_exc}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

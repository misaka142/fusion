"""Train SUGAR Stage 1 (Refiner) or Stage 2 (Tracker) using SuperMimic differentiable World-Model BPTT.

Drop-in CLI replacement for `scripts/sugar_rl/train.py`:
- Stage 1 Refiner:
    python -u scripts/supermimic/train.py \
      --task Sugar-G129dof-CarryBox-Refiner \
      --num_envs 2048 \
      --log_dir outputs/CarryBox_sm/logs/refiner \
      --max_iterations 5001 \
      --motion_folder data/OmniCarryBox \
      --headless

- Stage 2 Tracker (with frozen SuperMimic Refiner Teacher):
    python -u scripts/supermimic/train.py \
      --task Sugar-G129dof-CarryBox-Tracker \
      --num_envs 2048 \
      --teacher_ckpt outputs/CarryBox_sm/ckpts/refiner.pt \
      --motion_folder outputs/CarryBox_sm/rollout_datasets/refiner/rl_dataset \
      --teacher_motion_folder data/OmniCarryBox \
      --log_dir outputs/CarryBox_sm/logs/tracker \
      --max_iterations 5001 \
      --headless
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
from datetime import datetime


def _has_trainreg_ancestor() -> bool:
    """Return True if any ancestor process in `/proc` is `trainreg`."""
    pid = os.getpid()
    visited: set[int] = set()
    while pid > 1 and pid not in visited:
        visited.add(pid)
        try:
            cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
            comm = pathlib.Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="ignore").strip()
            if "trainreg" in comm or "trainreg" in cmdline:
                return True
            stat_text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
            rparen = stat_text.rfind(")")
            ppid = int(stat_text[rparen + 2 :].split()[1])
            pid = ppid
        except Exception:
            break
    return False


def _enforce_trainreg_if_present() -> None:
    """Block unauthenticated training launches when `trainreg` is installed on the host."""
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        return
    if sys.platform.startswith("linux") and shutil.which("trainreg") is not None:
        if not _has_trainreg_ancestor():
            quoted_cmd = " ".join([shutil.which("python") or "python", *sys.argv])
            sys.stderr.write(
                "\n[trainreg Compliance Error] `trainreg` is installed on this server, "
                "so direct training execution without `trainreg run --` is forbidden.\n"
                "1) Check real-time quota first:\n"
                "     trainreg balance --name-id yexiaoming --json\n"
                "2) Launch training interactively via `trainreg run --`:\n"
                f"     trainreg run -- {quoted_cmd}\n\n"
            )
            sys.exit(2)


_enforce_trainreg_if_present()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

import gymnasium as gym
from isaaclab.app import AppLauncher

tasks = [
    spec.id
    for spec in gym.registry.values()
    if "Sugar" in spec.id and "Isaac" not in spec.id
]

parser = argparse.ArgumentParser(description="Train SUGAR Refiner / Tracker with SuperMimic WM-BPTT.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel IsaacLab environments.")
parser.add_argument("--task", type=str, required=True, choices=tasks, help="SUGAR Gymnasium task ID.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=5001, help="Number of SuperMimic training iterations.")
parser.add_argument("--log_dir", type=str, default=None, help="Directory to store checkpoints and TensorBoard logs.")
parser.add_argument("--resume_ckpt", type=str, default=None, help="Optional SuperMimic checkpoint to resume from.")
parser.add_argument(
    "--auto_resume",
    action="store_true",
    default=False,
    help="Automatically resume from model_latest.pt (or latest model_<iter>.pt) in --log_dir if present.",
)

# SUGAR motion & Teacher distillation arguments
parser.add_argument("--motion_folder", type=str, default=None, help="Path to reference motion dataset folder.")
parser.add_argument(
    "--teacher_motion_folder", type=str, default=None, help="Path to raw teacher motion folder (Stage 2 Tracker)."
)
parser.add_argument(
    "--teacher_ckpt", type=str, default=None, help="Path to Stage 1 SuperMimic Refiner `.pt` checkpoint."
)

# SuperMimic algorithm hyperparameters (v2 anti-chatter & high-UTD defaults)
parser.add_argument("--horizon", type=int, default=32, help="Differentiable imagination rollout horizon H.")
parser.add_argument("--wm_seq_len", type=int, default=8, help="Autoregressive world-model training sequence length k.")
parser.add_argument("--wm_updates", type=int, default=16, help="World model gradient updates per iteration.")
parser.add_argument("--policy_updates", type=int, default=8, help="Policy BPTT updates per iteration.")
parser.add_argument("--wm_batch_size", type=int, default=2048, help="Minibatch size for world-model updates.")
parser.add_argument("--policy_batch_size", type=int, default=2048, help="Batch size of 32-step imagination rollouts.")
parser.add_argument("--replay_capacity", type=int, default=256, help="Ring replay buffer steps per environment.")
parser.add_argument("--wm_hidden", type=str, default="1024,1024,1024", help="World model hidden widths.")
parser.add_argument("--policy_hidden", type=str, default="512,256,128", help="Policy hidden widths.")
parser.add_argument("--wm_lr", type=float, default=1e-3, help="World model learning rate.")
parser.add_argument("--policy_lr", type=float, default=3e-4, help="Policy BPTT learning rate.")
parser.add_argument("--critic_lr", type=float, default=5e-4, help="Critic learning rate.")
parser.add_argument("--bc_coef", type=float, default=1.0, help="Initial Teacher BC weight in Tracker mode.")
parser.add_argument(
    "--bc_anneal_iters",
    type=int,
    default=None,
    help="Iterations to anneal Teacher BC weight (defaults to 70%% of --max_iterations, e.g. 3500 for 5001).",
)
parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval (iterations).")
parser.add_argument("--no_ref_residual", action="store_true", default=False, help="Disable reference joint residual bias.")
parser.add_argument(
    "--legs_residual",
    action="store_true",
    default=False,
    help="Deprecated compatibility flag (leg reference residual is now enabled by default).",
)
parser.add_argument(
    "--no_legs_residual",
    action="store_true",
    default=False,
    help="Disable reference joint residual bias on leg joints (default: False).",
)
parser.add_argument("--no_verify_obs", action="store_true", default=False, help="Skip startup obs alignment assertion.")
parser.add_argument(
    "--obs_verify_atol",
    type=float,
    default=5e-3,
    help="Max allowed per-dimension absolute difference in startup 890-D privileged obs check.",
)
parser.add_argument("--compile", action="store_true", default=False, help="Enable whole-step torch.compile fusion.")
parser.add_argument(
    "--compile_mode",
    type=str,
    default="default",
    help="torch.compile inductor mode ('default' is safe with IsaacLab; 'reduce-overhead' enables CUDA Graphs).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
hydra_args.append("hydra.run.dir=.")
hydra_args.append("hydra.output_subdir=null")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-Omniverse imports
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

import sugar_rl.tasks  # noqa: F401
from sugar_rl.supermimic.env_bridge import SugarEnvBridge, scale_physx_cfg_for_num_envs
from sugar_rl.supermimic.trainer import SuperMimicTrainCfg, SuperMimicTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg) -> None:
    torch.manual_seed(args_cli.seed)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    scale_physx_cfg_for_num_envs(env_cfg)
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.motion_folder is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.motion_folder = args_cli.motion_folder

    if (
        args_cli.teacher_motion_folder is not None
        and hasattr(env_cfg, "commands")
        and hasattr(env_cfg.commands, "motion")
    ):
        env_cfg.commands.motion.teacher_motion_folder = args_cli.teacher_motion_folder

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        from sugar_rl.assets.objects.objects import apply_dataset_box_spec

        apply_dataset_box_spec(
            env_cfg,
            env_cfg.commands.motion.motion_folder,
            env_cfg.commands.motion.teacher_motion_folder,
        )

    if args_cli.log_dir is not None:
        log_dir = os.path.abspath(args_cli.log_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(os.path.join("logs", "supermimic", args_cli.task, stamp))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[SuperMimic] Logging to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[SuperMimic] Recording training videos:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Scale batch sizes gracefully when running small smoke tests (e.g. --num_envs 64 or 512)
    total_ring_transitions = env.unwrapped.num_envs * args_cli.replay_capacity
    wm_bs = min(args_cli.wm_batch_size, max(64, total_ring_transitions // 8))
    pol_bs = min(args_cli.policy_batch_size, max(32, env.unwrapped.num_envs))

    bc_anneal_iters = (
        args_cli.bc_anneal_iters
        if args_cli.bc_anneal_iters is not None
        else max(100, int(0.7 * args_cli.max_iterations))
    )

    def _dims(text: str) -> tuple[int, ...]:
        return tuple(int(x) for x in text.split(",") if x.strip())

    sm_cfg = SuperMimicTrainCfg(
        replay_capacity=args_cli.replay_capacity,
        wm_hidden_dims=_dims(args_cli.wm_hidden),
        wm_seq_len=args_cli.wm_seq_len,
        wm_updates_per_iter=args_cli.wm_updates,
        wm_batch_size=wm_bs,
        wm_lr=args_cli.wm_lr,
        policy_hidden_dims=_dims(args_cli.policy_hidden),
        imagination_horizon=args_cli.horizon,
        policy_updates_per_iter=args_cli.policy_updates,
        policy_batch_size=pol_bs,
        policy_lr=args_cli.policy_lr,
        critic_lr=args_cli.critic_lr,
        use_ref_residual=not args_cli.no_ref_residual,
        legs_use_residual=not args_cli.no_legs_residual,
        bc_init_coef=args_cli.bc_coef,
        bc_anneal_iters=bc_anneal_iters,
        save_interval=args_cli.save_interval,
        compile_models=args_cli.compile,
        compile_mode=args_cli.compile_mode,
    )

    bridge = SugarEnvBridge(
        env=env,
        task_name_or_id=args_cli.task,
        verify_obs_on_init=not args_cli.no_verify_obs,
        obs_verify_atol=args_cli.obs_verify_atol,
    )
    trainer = SuperMimicTrainer(
        bridge=bridge,
        log_dir=log_dir,
        cfg=sm_cfg,
        teacher_ckpt=args_cli.teacher_ckpt,
    )

    if args_cli.resume_ckpt is not None:
        resumed_it = trainer.load_checkpoint(args_cli.resume_ckpt)
        print(f"[SuperMimic] Resumed from {args_cli.resume_ckpt} at iteration {resumed_it}")
    elif args_cli.auto_resume:
        trainer.auto_resume()

    final_ckpt = trainer.learn(max_iterations=args_cli.max_iterations)
    print(f"[SuperMimic] Training complete! Final checkpoint saved at: {final_ckpt}")
    bridge.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

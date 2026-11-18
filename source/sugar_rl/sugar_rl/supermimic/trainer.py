"""Unified SuperMimic Trainer for SUGAR Stage 1 (Refiner) and Stage 2 (Tracker).

Replaces:
- Stage 1 RSL-RL `OnPolicyRunner` (PPO) -> `SuperMimicTrainer` (stage="refiner")
- Stage 2 SUGAR `BCPPO` -> `SuperMimicTrainer` (stage="tracker" + annealed Refiner-Teacher BC in imagination)

Algorithm per iteration:
1. Collect `steps_per_env` (default 24) transitions in IsaacLab. The first `collect_det_frac` envs act without
   noise (as deployed), the rest add fixed Gaussian action noise (allmimic #23). Transitions go into the GPU
   `HOIReplayBuffer`; GAE(gamma, lam) targets use exact current-step timeout value bootstrap.
2. Train `JointWorldModel` via `k=8`-step autoregressive rollout. The WM predicts the state delta normalized by
   the running std of the true one-step delta, and the loss is the squared state error in the same units.
3. Train `SuperMimicCritic` ensemble on real simulator TD(lambda) targets.
4. Train `SuperMimicPolicy` via a deterministic `H=32`-step differentiable rollout through the frozen WM
   + analytical SUGAR rewards + smooth soft terminations + value-clipped critic tail bootstrap
   (+ annealed Teacher BC in Tracker mode).
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hoi_task import (
    PRIVILEGED_OBS_DIM,
    SDIM,
    TRACKER_OBS_DIM,
    resolve_task_profile,
)
from .networks import JointWorldModel, SuperMimicCritic, SuperMimicPolicy
from .replay import HOIReplayBuffer

LEG_JOINT_KEYWORDS = ("hip", "knee", "ankle")


@dataclass
class SuperMimicTrainCfg:
    """Hyperparameters for SuperMimic x SUGAR training (aligned with HybridRobotics/supermimic & A100 4096 envs)."""

    stage: str = "auto"                              # "auto" | "refiner" | "tracker"
    steps_per_env: int = 24                          # Matches SUGAR num_steps_per_env=24 for 1:1 iteration parity
    replay_capacity: int = 512                       # Ring buffer length per env on GPU (~21 iters of coverage)
    collect_det_frac: float = 0.25                   # First 25% envs act without noise via EMA policy (protects init_pool)
    collect_action_noise: float = 0.15               # Fixed exploration noise std in raw action units (0.1-0.2)

    # World model settings (scaled for A100 + Bryson pose weighting from official HybridRobotics/supermimic)
    wm_hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    wm_seq_len: int = 8                              # Autoregressive unroll horizon k
    wm_updates_per_iter: int = 12
    wm_batch_size: int = 2048
    wm_lr: float = 1e-3
    wm_contact_loss_weight: float = 0.25
    wm_obj_loss_weight: float = 2.0
    wm_pose_loss_weight: float = 4.0                 # Boost integrated root/object pose channels (prevents XY drift)
    wm_warmup_iters: int = 8                         # Fit WM & Critic (and pure BC if Tracker) before enabling BPTT
    joint_weight_clip: float = 4.0                   # Clip WM joint error weights to [1/c, c] (allmimic #21)

    # Policy BPTT settings
    policy_hidden_dims: tuple[int, ...] = (512, 256, 128)
    use_ref_residual: bool = True
    legs_absolute: bool = True                       # Legs absolute for dynamic CoM balance, upper body residual (allmimic #20)
    init_noise_std: float = 0.20                     # Only initializes the (unused) `log_std` parameter
    imagination_horizon: int = 32                    # H=32 step differentiable imagination rollout
    policy_updates_per_iter: int = 6
    policy_batch_size: int = 2048
    policy_lr: float = 3e-4
    adam_beta2: float = 0.95                         # Faster Adam EMA recovery after contact gradient spikes (official)
    bptt_grad_decay: float = 1.0                     # Full unattenuated 32-step horizon gradients (official)
    imag_jerk_weight: float = 0.02                   # Second-order action smoothness `||a_t - 2a_{t-1} + a_{t-2}||^2`
    ema_decay: float = 0.995                         # Polyak EMA actor weights for smooth gait & checkpoint stability
    grad_value_clip: float | None = 0.1              # Elementwise gradient clip after norm clip (official)
    pessimistic_critic: bool = True                  # Min over critic heads in imagination bootstrap
    value_clip_range: tuple[float, float] | None = None  # None -> +-max_step_reward / (1 - gamma)

    # Critic TD(lambda) settings (jointly calibrated on real transitions + deterministic imagined rollouts)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_heads: int = 2
    critic_epochs: int = 4
    critic_minibatches: int = 4
    imag_critic_updates: int = 2                     # Extra GAE updates per policy step on 32-step imagined rollouts
    critic_lr: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95

    # Learning-rate schedule: cosine decay of all three optimizers to `lr_final_frac` x base over the run
    lr_final_frac: float = 0.03

    # Stage 2 Tracker Teacher-BC distillation settings (replaces BCPPO)
    bc_init_coef: float = 1.0
    bc_min_coef: float = 0.05
    bc_anneal_iters: int = 3500                      # Reaches bc_min_coef at 70% of standard 5001-iter run

    # Precision, checkpointing & compilation
    use_bf16: bool = True                            # CUDA bfloat16 autocast inside MLPs (FP32 state/FK/rotations)
    max_grad_norm: float = 1.0
    save_interval: int = 500
    enable_tensorboard: bool = True
    compile_models: bool = False                     # Optional torch.compile of the MLPs only
    compile_bptt: bool = False                       # Optional torch.compile of whole WM & imagination steps
    compile_backend: str = "inductor"                # Backend passed to `torch.compile`
    compile_mode: str = "default"                    # Mode for `compile_bptt`: "default" | "reduce-overhead" | "max-autotune"


def _unwrap_compiled(module: nn.Module) -> nn.Module:
    """Return underlying `nn.Module` if wrapped by `torch.compile` (`OptimizedModule`)."""
    return getattr(module, "_orig_mod", module)


def _strip_compiled_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip `_orig_mod.` prefix from state_dict keys so compiled & eager checkpoints interoperate."""
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def make_residual_mask(joint_names: list[str], legs_absolute: bool) -> Tensor:
    """(29,) mask: 1 = residual on the reference target, 0 = absolute action (leg joints when `legs_absolute`)."""
    mask = torch.ones(len(joint_names), dtype=torch.float32)
    if legs_absolute:
        for i, name in enumerate(joint_names):
            if any(k in name for k in LEG_JOINT_KEYWORDS):
                mask[i] = 0.0
    return mask


def _policy_state_dict_compat(state_dict: dict[str, Any], action_dim: int) -> dict[str, Any]:
    """Older checkpoints have no `residual_mask` buffer: they were trained fully residual (all ones)."""
    sd = _strip_compiled_prefix(state_dict)
    if "residual_mask" not in sd:
        sd["residual_mask"] = torch.ones(action_dim, dtype=torch.float32)
    return sd


def _zero_nonfinite_grads_(params: list[nn.Parameter], grad_norm: Tensor) -> None:
    """Zero all gradients when the total norm is non-finite, without a host sync."""
    finite = torch.isfinite(grad_norm).float()
    for p in params:
        if p.grad is not None:
            p.grad.nan_to_num_(0.0, 0.0, 0.0).mul_(finite)


def load_supermimic_policy(
    ckpt_path: str | Path, device: torch.device | str = "cuda"
) -> SuperMimicPolicy:
    """Load a self-contained `SuperMimicPolicy` from a saved `.pt` checkpoint for inference or Teacher BC."""
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "policy_state_dict" not in payload:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is not a SuperMimic checkpoint (missing 'policy_state_dict')."
        )
    action_dim = int(payload.get("action_dim", 29))
    policy = SuperMimicPolicy(
        obs_dim=int(payload["obs_dim"]),
        action_dim=action_dim,
        hidden_dims=tuple(payload.get("hidden_dims", (512, 256, 128))),
        use_ref_residual=bool(payload.get("use_ref_residual", True)),
    ).to(device)
    policy.load_state_dict(_policy_state_dict_compat(payload["policy_state_dict"], action_dim))
    if "joint_names" in payload and payload["joint_names"] is not None:
        policy.joint_names = list(payload["joint_names"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _freeze_params(module: nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(not freeze)


class SuperMimicTrainer:
    """End-to-end Model-Based BPTT trainer for SUGAR Refiner and Tracker stages."""

    def __init__(
        self,
        bridge: Any,
        log_dir: str | Path,
        cfg: SuperMimicTrainCfg | None = None,
        teacher_ckpt: str | Path | None = None,
    ) -> None:
        self.bridge = bridge
        self.task = bridge.task
        self.fk = bridge.fk
        self.device = bridge.device
        self.num_envs = bridge.num_envs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or SuperMimicTrainCfg()

        # Auto-detect Refiner vs Tracker strictly from policy observation dimension
        obs_policy_sample = bridge.current_obs_dict["policy"]
        policy_obs_dim = int(obs_policy_sample.shape[-1])
        if self.cfg.stage == "auto":
            if policy_obs_dim == TRACKER_OBS_DIM:
                self.stage = "tracker"
            elif policy_obs_dim == PRIVILEGED_OBS_DIM:
                self.stage = "refiner"
                if teacher_ckpt is not None:
                    print(
                        f"[SuperMimic Trainer] WARNING: Ignoring `--teacher_ckpt={teacher_ckpt}` "
                        f"because policy obs dim is {policy_obs_dim} (Stage 1 Refiner mode)."
                    )
            else:
                raise ValueError(
                    f"Unsupported policy observation dim {policy_obs_dim}; "
                    f"expected {PRIVILEGED_OBS_DIM} (Refiner) or {TRACKER_OBS_DIM} (Tracker)."
                )
        else:
            self.stage = self.cfg.stage

        # Ensure stage-dependent reward profile nuances (e.g. PushBox Tracker) match even on mock bridges
        if getattr(self.bridge, "env", None) is None or getattr(self.bridge.env, "reward_manager", None) is None:
            stage_prof = resolve_task_profile(self.task.profile.task_name, stage=self.stage)
            self.task.profile.undesired_weight = stage_prof.undesired_weight
            self.task.profile.undesired_threshold = stage_prof.undesired_threshold

        critic_obs_dim = int(bridge.current_obs_dict.get("critic", obs_policy_sample).shape[-1])
        assert (
            critic_obs_dim == PRIVILEGED_OBS_DIM
        ), f"Expected {PRIVILEGED_OBS_DIM}-D critic obs, got {critic_obs_dim}"

        # Joint error-propagation weights from G1 kinematic tree, clipped to [1/c, c] and mean 1 (allmimic #21)
        c = self.cfg.joint_weight_clip
        jw = self.fk.compute_joint_weights().to(self.device).clamp(1.0 / c, c)
        self.joint_weights = jw / jw.mean()

        # Value clip for every critic bootstrap: the reachable range of the discounted return (allmimic #29)
        if self.cfg.value_clip_range is not None:
            self.value_clip = tuple(self.cfg.value_clip_range)
        else:
            v_lim = self.task.max_step_reward() / (1.0 - self.cfg.gamma)
            self.value_clip = (-v_lim, v_lim)

        # Build World Model, Actor (+ Polyak EMA Actor), Critic
        eff_bf16 = bool(
            self.cfg.use_bf16
            and torch.device(self.device).type == "cuda"
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        self._raw_world_model = JointWorldModel(
            hidden_dims=self.cfg.wm_hidden_dims,
            use_bf16=eff_bf16,
        ).to(self.device)
        residual_mask = make_residual_mask(list(self.fk.env_joint_names), self.cfg.legs_absolute)
        self._raw_policy = SuperMimicPolicy(
            obs_dim=policy_obs_dim,
            action_dim=29,
            hidden_dims=self.cfg.policy_hidden_dims,
            default_joint_pos=self.fk.default_joint_pos,
            action_scale=self.fk.action_scale,
            use_ref_residual=self.cfg.use_ref_residual,
            init_std=self.cfg.init_noise_std,
            residual_mask=residual_mask,
            use_bf16=eff_bf16,
        ).to(self.device)
        self._raw_policy.log_std.requires_grad_(False)  # exploration noise is fixed (`collect_action_noise`)

        # Smooth Polyak EMA policy used for deterministic collection, evaluation & exported `policy_state_dict`
        self.ema_policy: SuperMimicPolicy = copy.deepcopy(self._raw_policy).to(self.device)
        self.ema_policy.use_bf16 = False  # exact FP32 inference parity with standalone `play.py` & MuJoCo Sim2Sim
        self.ema_policy.eval()
        _freeze_params(self.ema_policy, True)

        self._raw_critic = SuperMimicCritic(
            obs_dim=critic_obs_dim,
            hidden_dims=self.cfg.critic_hidden_dims,
            num_heads=self.cfg.critic_heads,
            use_bf16=eff_bf16,
        ).to(self.device)

        self.world_model: nn.Module = self._raw_world_model
        self.policy: nn.Module = self._raw_policy
        self.critic: nn.Module = self._raw_critic

        # Optional frozen Refiner Teacher for Stage 2 Tracker distillation
        self.teacher_policy: SuperMimicPolicy | None = None
        if self.stage == "tracker" and teacher_ckpt is not None:
            print(f"[SuperMimic Trainer] Loading frozen Refiner Teacher from: {teacher_ckpt}")
            self.teacher_policy = load_supermimic_policy(teacher_ckpt, device=self.device)

        if self.cfg.compile_models and not self.cfg.compile_bptt and hasattr(torch, "compile"):
            self._raw_world_model.enable_compile(backend=self.cfg.compile_backend)
            self._raw_policy.enable_compile(backend=self.cfg.compile_backend)
            self._raw_critic.enable_compile(backend=self.cfg.compile_backend)

        self._wm_step_fn = self._wm_step
        self._img_step_fn = self._imagine_step
        self._use_cudagraph_mark = False
        self._logged_wm_compile = False
        self._logged_img_compile = False
        if self.cfg.compile_bptt and hasattr(torch, "compile"):
            eff_mode = self.cfg.compile_mode
            if (
                self.cfg.compile_backend == "inductor"
                and eff_mode in ("reduce-overhead", "max-autotune")
                and (self.cfg.imagination_horizon > 4 or self.cfg.wm_seq_len > 4)
            ):
                fallback = "max-autotune-no-cudagraphs" if eff_mode == "max-autotune" else "default"
                print(
                    f"[SuperMimic Trainer] Note: switching compile_mode='{eff_mode}' -> '{fallback}' for "
                    f"multi-step BPTT (H={self.cfg.imagination_horizon}, k={self.cfg.wm_seq_len}) to avoid "
                    f"deep CUDAGraph tree capture stalls and VRAM pool bloat.",
                    flush=True,
                )
                eff_mode = fallback
            comp_kwargs: dict[str, Any] = {"backend": self.cfg.compile_backend}
            if self.cfg.compile_backend == "inductor" and eff_mode not in (None, "", "default"):
                comp_kwargs["mode"] = eff_mode
            self._use_cudagraph_mark = (self.cfg.compile_backend == "cudagraphs") or (
                self.cfg.compile_backend == "inductor"
                and eff_mode in ("reduce-overhead", "max-autotune")
            )
            self._wm_step_fn = torch.compile(self._wm_step, **comp_kwargs)
            self._img_step_fn = torch.compile(self._imagine_step, **comp_kwargs)

        use_fused_opt = torch.device(self.device).type == "cuda"
        betas = (0.9, float(self.cfg.adam_beta2))
        self.wm_opt = torch.optim.AdamW(
            self._raw_world_model.parameters(), lr=self.cfg.wm_lr, betas=betas, weight_decay=1e-5, fused=use_fused_opt
        )
        self.policy_opt = torch.optim.AdamW(
            [p for p in self._raw_policy.parameters() if p.requires_grad],
            lr=self.cfg.policy_lr,
            betas=betas,
            weight_decay=1e-5,
            fused=use_fused_opt,
        )
        self.critic_opt = torch.optim.AdamW(
            self._raw_critic.parameters(), lr=self.cfg.critic_lr, betas=betas, weight_decay=1e-5, fused=use_fused_opt
        )
        self._base_lrs = {
            "wm": self.cfg.wm_lr,
            "policy": self.cfg.policy_lr,
            "critic": self.cfg.critic_lr,
        }

        # GPU Ring Replay Buffer + Domain Randomization Context xi
        self.replay = HOIReplayBuffer(
            capacity=self.cfg.replay_capacity,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.replay.set_env_xi(self.bridge.extract_xi())

        # Monotonic GPU episode continuity counter per env (zero CPU-GPU syncs)
        self._ep_step_counter: int = 0
        self._env_index_vec = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.env_ep_ids = self._env_index_vec.clone()
        self.det_env_count = int(self.num_envs * self.cfg.collect_det_frac)

        # Deterministic envs that always restart at t=0 reproduce the Stage 6.2 rollout condition exactly:
        # their clip-completion rate is a live proxy of the rollout selection rate.
        start_count = getattr(getattr(self.bridge, "motion_cmd", None), "start_init_env_count", self.det_env_count)
        self.proxy_env_count = min(self.det_env_count, int(start_count))
        self._last_log_obj: Any = None

        # Optional TensorBoard writer
        self.tb_writer = None
        if self.cfg.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)
                print(f"[SuperMimic Trainer] TensorBoard logging enabled -> {self.log_dir}")
            except Exception as exc:
                print(f"[SuperMimic Trainer] WARNING: TensorBoard disabled ({exc})")
                self.tb_writer = None

        self.current_iter: int = 0

    def current_bc_coef(self, iteration: int) -> float:
        if self.teacher_policy is None:
            return 0.0
        progress = min(1.0, float(iteration) / max(1, self.cfg.bc_anneal_iters))
        coef = self.cfg.bc_init_coef * (1.0 - progress) + self.cfg.bc_min_coef * progress
        return float(max(self.cfg.bc_min_coef, coef))

    def _set_lr(self, iteration: int, max_iterations: int) -> float:
        """Cosine decay from 1 to `lr_final_frac` over the run; returns the current multiplier."""
        p = min(1.0, max(0.0, iteration / max(1, max_iterations - 1)))
        frac = self.cfg.lr_final_frac + (1.0 - self.cfg.lr_final_frac) * 0.5 * (1.0 + math.cos(math.pi * p))
        for opt, key in ((self.wm_opt, "wm"), (self.policy_opt, "policy"), (self.critic_opt, "critic")):
            for group in opt.param_groups:
                group["lr"] = self._base_lrs[key] * frac
        return frac

    @torch.no_grad()
    def _sync_ema_buffers(self) -> None:
        for b_ema, b_raw in zip(self.ema_policy.buffers(), self._raw_policy.buffers()):
            b_ema.copy_(b_raw)

    @torch.no_grad()
    def _update_ema_policy(self) -> None:
        """Zero-sync fused Polyak EMA update of actor weights + exact buffer mirroring."""
        decay = float(self.cfg.ema_decay)
        raw_params = list(self._raw_policy.parameters())
        ema_params = list(self.ema_policy.parameters())
        if decay <= 0.0:
            torch._foreach_copy_(ema_params, raw_params)
        else:
            torch._foreach_lerp_(ema_params, raw_params, 1.0 - decay)
        self._sync_ema_buffers()

    # ------------------------------------------------------------------
    # Phase 1: Real Environment Rollout Collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(self) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Collect `steps_per_env` transitions and compute GAE returns for Critic."""
        T = self.cfg.steps_per_env
        N = self.num_envs
        get_air_fn = getattr(self.bridge, "get_feet_air_time", None)
        lengths = self.bridge.motion_ref.lengths

        critic_obs_buf = torch.empty((T, N, PRIVILEGED_OBS_DIM), dtype=torch.float32, device=self.device)
        rewards_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)
        dones_buf = torch.empty((T, N), dtype=torch.bool, device=self.device)
        values_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)

        noise_scale = torch.full((N, 1), self.cfg.collect_action_noise, device=self.device)
        noise_scale[: self.det_env_count] = 0.0
        is_proxy = self._env_index_vec < self.proxy_env_count

        proxy_done = torch.zeros((), device=self.device)
        proxy_succ = torch.zeros((), device=self.device)
        proxy_prog = torch.zeros((), device=self.device)
        log_sums: dict[str, Tensor] = {}
        log_counts: dict[str, int] = {}

        for t in range(T):
            obs_dict = self.bridge.current_obs_dict
            obs_pol = obs_dict["policy"]
            obs_crit = obs_dict.get("critic", obs_pol)

            s_pre = self.bridge.get_joint_state()
            prev_act = self.bridge.get_last_action()
            m_id, t_step = self.bridge.get_motion_cursor()
            feet_air = get_air_fn() if callable(get_air_fn) else None

            # Update running normalizers with fresh on-policy simulator observations
            self._raw_policy.obs_rms.update(obs_pol)
            self._sync_ema_buffers()
            v_pred = self.critic(obs_crit, update_rms=True, pessimistic=False)

            # Deterministic first `det_env_count` envs use smooth EMA policy (as deployed); rest explore with noise
            det_n = self.det_env_count
            if det_n > 0 and self.cfg.ema_decay > 0.0:
                act_det = self.ema_policy.forward_mean(obs_pol[:det_n], update_rms=False)
                act_exp = self._raw_policy.forward_mean(obs_pol[det_n:], update_rms=False)
                act_mean = torch.cat([act_det, act_exp], dim=0)
            else:
                act_mean = self._raw_policy.forward_mean(obs_pol, update_rms=False)
            action = act_mean + noise_scale * torch.randn_like(act_mean)

            _, sim_rew, dones, timeouts, pushed, extras = self.bridge.step(action)
            s_post = self.bridge.get_joint_state()
            contacts = self.bridge.get_ground_truth_contacts()

            # Bootstrap timeout terminations using CURRENT step value `v_pred` ( matching RSL-RL;
            # `next_obs_dict` for reset envs already belongs to the newly reset episode!)
            sim_rew = sim_rew + self.cfg.gamma * v_pred * timeouts.float()

            # Preserve true episode ID of `s_pre` for 5-frame history, while marking broken transitions
            valid_trans = ~(dones | pushed)
            self.replay.add(
                state=s_pre,
                action=action,
                prev_action=prev_act,
                next_state=s_post,
                contact=contacts,
                motion_id=m_id,
                time_step=t_step,
                ep_id=self.env_ep_ids,
                valid_transition=valid_trans,
                feet_air_time=feet_air,
            )

            # Advance episode IDs ONLY for environments that actually reset (`dones`), with zero GPU-CPU sync
            self._ep_step_counter += N
            self.env_ep_ids = torch.where(dones, self._env_index_vec + self._ep_step_counter, self.env_ep_ids)

            # Rollout proxy on deterministic t=0 envs: success = clip finished (timeout) instead of terminated
            p_done = (dones & is_proxy).float()
            proxy_done = proxy_done + p_done.sum()
            proxy_succ = proxy_succ + (p_done * timeouts.float()).sum()
            prog = ((t_step + 1).float() / lengths[m_id].clamp_min(1).float()).clamp(max=1.0)
            proxy_prog = proxy_prog + (p_done * prog).sum()

            # IsaacLab writes a fresh `extras["log"]` dict only on steps with resets (identity check, no sync)
            log_obj = extras.get("log") if isinstance(extras, dict) else None
            if isinstance(log_obj, dict) and log_obj is not self._last_log_obj:
                self._last_log_obj = log_obj
                for key, val in log_obj.items():
                    if isinstance(val, (int, float, Tensor)):
                        v_t = torch.as_tensor(val, dtype=torch.float32, device=self.device).reshape(-1).mean()
                        log_sums[key] = log_sums.get(key, torch.zeros((), device=self.device)) + v_t
                        log_counts[key] = log_counts.get(key, 0) + 1

            critic_obs_buf[t].copy_(obs_crit)
            rewards_buf[t].copy_(sim_rew)
            dones_buf[t].copy_(dones)
            values_buf[t].copy_(v_pred)

        # Compute GAE(gamma, lambda) returns for Critic training
        last_obs_crit = self.bridge.current_obs_dict.get("critic", self.bridge.current_obs_dict["policy"])
        next_value = self.critic(last_obs_crit, update_rms=False, pessimistic=False)

        advantages = torch.zeros_like(rewards_buf)
        gae = torch.zeros(N, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            not_done = (~dones_buf[t]).float()
            v_next = next_value if t == T - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + self.cfg.gamma * v_next * not_done - values_buf[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * not_done * gae
            advantages[t] = gae

        returns_buf = advantages + values_buf
        stats = {
            "collect/sim_step_reward": float(rewards_buf.mean().item()),
            "collect/reset_rate": float(dones_buf.float().mean().item()),
        }
        n_done = float(proxy_done.item())
        if n_done > 0:
            stats["rollout_proxy/det_t0_success_rate"] = float(proxy_succ.item()) / n_done
            stats["rollout_proxy/det_t0_progress"] = float(proxy_prog.item()) / n_done
            stats["rollout_proxy/det_t0_episodes"] = n_done
        for key, val in log_sums.items():
            stats[f"env/{key}"] = float((val / log_counts[key]).item())
        return critic_obs_buf.reshape(T * N, -1), returns_buf.reshape(T * N), stats

    # ------------------------------------------------------------------
    # Phase 2: Autoregressive Joint World Model Update
    # ------------------------------------------------------------------

    def _mark_cudagraph_step(self) -> None:
        if self._use_cudagraph_mark and hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()

    def _wm_step(
        self,
        s_cur: Tensor,
        action_t: Tensor,
        next_state_t: Tensor,
        contact_t: Tensor,
        mask_t: Tensor,
        xi: Tensor,
        err_std: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """One autoregressive WM step + masked normalized state/contact loss (pure tensor ops for compilation)."""
        s_next, c_logits = self._raw_world_model.step(self.task, s_cur, action_t, xi, update_rms=False)
        l_s, metrics = self.task.world_model_state_loss(
            s_next,
            next_state_t,
            self.joint_weights,
            err_std,
            obj_loss_weight=self.cfg.wm_obj_loss_weight,
            pose_loss_weight=self.cfg.wm_pose_loss_weight,
            step_mask=mask_t,
        )
        bce_per = F.binary_cross_entropy_with_logits(c_logits, contact_t, reduction="none").mean(dim=-1)
        l_c = (bce_per * mask_t).sum() / mask_t.sum().clamp_min(1.0)
        s_out = torch.where(mask_t.unsqueeze(-1) > 0.5, s_next, s_cur.detach())
        return s_out, l_s, l_c, metrics

    def update_world_model(self) -> dict[str, float]:
        if self.replay.size < 2:
            return {}

        self.world_model.train()
        wm = self._raw_world_model
        wm_params = list(wm.parameters())
        agg_tensors: dict[str, Tensor] = {}
        k_req = self.cfg.wm_seq_len
        actual_updates = 0

        for u in range(self.cfg.wm_updates_per_iter):
            batch = self.replay.sample_wm_sequences(self.cfg.wm_batch_size, seq_len=k_req)
            if batch is None:
                continue
            eff_k = int(batch.actions.shape[1])

            # Update input & output normalizers once per iteration from valid ground-truth transitions
            if u == 0:
                with torch.no_grad():
                    m0 = batch.mask[:, 0]
                    wm.feat_rms.update(self.task.state_features(batch.s_init), mask=m0)
                    wm.xi_rms.update(batch.xi, mask=m0)
                    s_prev = batch.s_init
                    deltas = []
                    for t in range(eff_k):
                        deltas.append(self.task.compute_state_delta(s_prev, batch.next_states[:, t]))
                        s_prev = batch.next_states[:, t]
                    wm.update_delta_stats(torch.cat(deltas, dim=0), mask=batch.mask.transpose(0, 1).reshape(-1))
            err_std = wm.delta_std().detach()

            if self.cfg.compile_bptt and not self._logged_wm_compile:
                print("[SuperMimic Trainer] Compiling single-step world-model graph (one-time JIT)...", flush=True)
                self._logged_wm_compile = True

            # Match `s_cur.requires_grad` at t=0 with t>=1 so torch.compile traces `_wm_step` once, not twice
            s_cur = batch.s_init.detach().requires_grad_(True) if self.cfg.compile_bptt else batch.s_init
            loss_state_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            loss_contact_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            last_metrics: dict[str, Tensor] = {}

            # Mark step boundary ONCE per full rollout+backward (never inside the k-step loop,
            # or CUDAGraph Trees will overwrite earlier steps' outputs needed by autograd).
            self._mark_cudagraph_step()
            for t in range(eff_k):
                s_cur, l_s, l_c, last_metrics = self._wm_step_fn(
                    s_cur,
                    batch.actions[:, t],
                    batch.next_states[:, t],
                    batch.contacts[:, t],
                    batch.mask[:, t],
                    batch.xi,
                    err_std,
                )
                loss_state_sum = loss_state_sum + l_s
                loss_contact_sum = loss_contact_sum + l_c

            total_loss = (loss_state_sum + self.cfg.wm_contact_loss_weight * loss_contact_sum) / float(eff_k)

            self.wm_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(wm_params, self.cfg.max_grad_norm)
            _zero_nonfinite_grads_(wm_params, grad_norm)
            self.wm_opt.step()

            actual_updates += 1
            agg_tensors["wm/total_loss"] = agg_tensors.get(
                "wm/total_loss", torch.zeros_like(total_loss)
            ) + total_loss.detach()
            c_avg = (loss_contact_sum / float(eff_k)).detach()
            agg_tensors["wm/contact_bce"] = agg_tensors.get("wm/contact_bce", torch.zeros_like(c_avg)) + c_avg
            for key, val in last_metrics.items():
                agg_tensors[key] = agg_tensors.get(key, torch.zeros_like(val)) + val

        if actual_updates == 0:
            return {}
        denom = float(actual_updates)
        return {key: float((val / denom).item()) for key, val in agg_tensors.items()}

    # ------------------------------------------------------------------
    # Phase 3: Ensemble Critic Update
    # ------------------------------------------------------------------

    def update_critic(self, critic_obs_flat: Tensor, returns_flat: Tensor) -> dict[str, float]:
        self.critic.train()
        total_samples = critic_obs_flat.shape[0]
        mb_size = max(64, total_samples // self.cfg.critic_minibatches)
        loss_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        n_steps = 0
        critic_params = list(self._raw_critic.parameters())

        for _ in range(self.cfg.critic_epochs):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, mb_size):
                idx = perm[start : start + mb_size]
                obs_mb = critic_obs_flat[idx]
                ret_mb = returns_flat[idx]

                preds = self._raw_critic.forward_all(obs_mb, update_rms=False)  # (H_heads, B)
                target = ret_mb.unsqueeze(0).expand_as(preds)
                loss = F.smooth_l1_loss(preds, target, beta=1.0)

                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(critic_params, self.cfg.max_grad_norm)
                _zero_nonfinite_grads_(critic_params, grad_norm)
                self.critic_opt.step()

                loss_accum = loss_accum + loss.detach()
                n_steps += 1

        return {"critic/loss": float((loss_accum / float(max(1, n_steps))).item())}

    # ------------------------------------------------------------------
    # Phase 4: 32-Step Differentiable World-Model BPTT (+ Tracker BC)
    # ------------------------------------------------------------------

    def _policy_obs(self, s_cur, prev_a, m_ids, t_cur, xi, fk_cur, hist_cur) -> Tensor:
        if self.stage == "tracker":
            cmd_36 = self.bridge.motion_ref.get_frame(m_ids, t_cur).tracker_command_36()
            return self.task.build_tracker_obs(s_cur, cmd_36, hist_cur, fk_res=fk_cur, add_noise=True)
        ref_fut_cur = self.bridge.motion_ref.get_future_window(m_ids, t_cur)
        return self.task.build_privileged_obs(s_cur, prev_a, ref_fut_cur, fk_res=fk_cur, xi=xi)

    def _teacher_bc(self, s_cur, prev_a, m_ids, t_cur, xi, fk_cur, a_mean) -> Tensor:
        if self.teacher_policy is None:
            return torch.zeros(a_mean.shape[0], dtype=a_mean.dtype, device=a_mean.device)
        t_ref_adapter = self.bridge.teacher_motion_ref or self.bridge.motion_ref
        teacher_fut = t_ref_adapter.get_future_window(m_ids, t_cur)
        with torch.no_grad():
            obs_teacher = self.task.build_privileged_obs(s_cur, prev_a, teacher_fut, fk_res=fk_cur, xi=xi)
            a_teacher = self.teacher_policy.act_inference(obs_teacher)
        return ((a_mean - a_teacher) ** 2).mean(dim=-1)

    def _imagine_step(self, s_cur, fk_cur, prev_a, m_ids, t_cur, xi, hist_cur):
        """One deterministic imagined step (no host syncs, so it can be compiled as a single region)."""
        v_min, v_max = self.value_clip
        ref_cur = self.bridge.motion_ref.get_frame(m_ids, t_cur)
        obs_pol = self._policy_obs(s_cur, prev_a, m_ids, t_cur, xi, fk_cur, hist_cur)
        a_cur = self._raw_policy.forward_mean(obs_pol, update_rms=False)
        step_bc = self._teacher_bc(s_cur, prev_a, m_ids, t_cur, xi, fk_cur, a_cur)

        # Differentiable 1-step human-object rollout
        s_next, c_logits = self._raw_world_model.step(self.task, s_cur, a_cur, xi, fk_res=fk_cur, update_rms=False)
        fk_next = self.task.run_fk(s_next)

        # Evaluate `s_next` against `ref_cur = ref[t_cur]` (air-time penalty omitted in imagination per review)
        r_step, p_term, r_info = self.task.compute_reward_and_termination(
            s_prev=s_cur,
            s_next=s_next,
            raw_action=a_cur,
            prev_raw_action=prev_a,
            ref_cur=ref_cur,
            contact_logits=c_logits,
            fk_next=fk_next,
            xi=xi,
            time_steps=t_cur,
            include_air_time_penalty=False,
        )

        # Value bootstrap where the reference clip finishes naturally, masked instead of branched (no host sync)
        t_next = t_cur + 1
        traj_done = self.bridge.motion_ref.is_done(m_ids, t_next).float()
        ref_fut_done = self.bridge.motion_ref.get_future_window(m_ids, t_next)
        obs_crit_done = self.task.build_privileged_obs(s_next, a_cur, ref_fut_done, fk_res=fk_next, xi=xi)
        v_done = self.critic(obs_crit_done, update_rms=False, pessimistic=self.cfg.pessimistic_critic).clamp(
            v_min, v_max
        )

        hist_next = (
            self.task.advance_tracker_history(hist_cur, s_next, a_cur, xi=xi) if self.stage == "tracker" else None
        )
        return (
            s_next,
            fk_next,
            a_cur,
            r_step,
            p_term,
            traj_done,
            v_done,
            step_bc,
            r_info["rew/tracking_obj"].detach(),
            hist_next,
        )

    def _fit_imagined_critic(
        self,
        img_states: Tensor,
        img_prev_acts: Tensor,
        img_rewards: Tensor,
        img_not_done: Tensor,
        m_ids: Tensor,
        t_0: Tensor,
        xi: Tensor,
    ) -> Tensor:
        """Align Critic on deterministic H-step imagined GAE returns (official ActorCriticPolicyLearner._fit_critic)."""
        H, B = img_rewards.shape
        v_min, v_max = self.value_clip
        gamma = self.cfg.gamma
        lam = self.cfg.lam

        with torch.no_grad():
            offsets = torch.arange(H + 1, device=self.device, dtype=t_0.dtype).unsqueeze(1)  # (H+1, 1)
            t_all = (t_0.unsqueeze(0) + offsets).reshape(-1)
            m_all = m_ids.unsqueeze(0).expand(H + 1, B).reshape(-1)
            xi_all = xi.unsqueeze(0).expand(H + 1, B, -1).reshape((H + 1) * B, -1)
            s_all = img_states.reshape((H + 1) * B, SDIM)
            pa_all = img_prev_acts.reshape((H + 1) * B, -1)
            fut_all = self.bridge.motion_ref.get_future_window(m_all, t_all)
            obs_all = self.task.build_privileged_obs(s_all, pa_all, fut_all, xi=xi_all)
            v_all = self.critic(obs_all, update_rms=False, pessimistic=False).clamp(v_min, v_max).view(H + 1, B)

            adv = torch.zeros(B, dtype=torch.float32, device=self.device)
            imag_ret = torch.empty((H, B), dtype=torch.float32, device=self.device)
            for tau in reversed(range(H)):
                nd = img_not_done[tau]
                delta = img_rewards[tau] + gamma * v_all[tau + 1] * nd - v_all[tau]
                adv = delta + gamma * lam * nd * adv
                imag_ret[tau] = (adv + v_all[tau]).clamp(v_min, v_max)

            obs_steps = obs_all[: H * B]
            ret_steps = imag_ret.reshape(H * B)

        _freeze_params(self._raw_critic, False)
        self.critic.train()
        critic_params = list(self._raw_critic.parameters())
        total_n = H * B
        mb_n = min(4096, total_n)
        loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        for _ in range(self.cfg.imag_critic_updates):
            perm = torch.randperm(total_n, device=self.device)[:mb_n]
            preds = self._raw_critic.forward_all(obs_steps[perm], update_rms=False)
            target = ret_steps[perm].unsqueeze(0).expand_as(preds)
            loss_v = F.smooth_l1_loss(preds, target, beta=1.0)
            self.critic_opt.zero_grad(set_to_none=True)
            loss_v.backward()
            gn = nn.utils.clip_grad_norm_(critic_params, self.cfg.max_grad_norm)
            _zero_nonfinite_grads_(critic_params, gn)
            self.critic_opt.step()
            loss_acc = loss_acc + loss_v.detach()
        self.critic.eval()
        _freeze_params(self._raw_critic, True)
        return loss_acc / float(self.cfg.imag_critic_updates)

    def update_policy(self, iteration: int) -> dict[str, float]:
        if self.replay.size < 2:
            return {}

        warmup_only = iteration < self.cfg.wm_warmup_iters
        if warmup_only and self.teacher_policy is None:
            return {"policy/warmup_skip": 1.0}

        self.policy.train()
        self.world_model.eval()
        self.critic.eval()
        _freeze_params(self._raw_world_model, True)
        _freeze_params(self._raw_critic, True)

        is_tracker = self.stage == "tracker"
        bc_coef = self.current_bc_coef(iteration)
        H = self.cfg.imagination_horizon
        v_min, v_max = self.value_clip
        alpha = self.cfg.bptt_grad_decay
        gamma = self.cfg.gamma
        jerk_w = float(self.cfg.imag_jerk_weight)
        dt = float(self.task.step_dt)
        pol_params = [p for p in self._raw_policy.parameters() if p.requires_grad]

        ret_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        bc_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        surv_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        obj_rew_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        imag_v_accum = torch.zeros((), dtype=torch.float32, device=self.device)

        try:
            for _ in range(self.cfg.policy_updates_per_iter):
                starts = self.replay.sample_imagination_starts(
                    self.cfg.policy_batch_size, need_history=is_tracker
                )
                B = starts.states.shape[0]
                use_compiled_rollout = self.cfg.compile_bptt and not warmup_only
                if use_compiled_rollout and not self._logged_img_compile:
                    print(
                        "[SuperMimic Trainer] Compiling single-step imagination BPTT graph "
                        f"(first run at iter={iteration}, one-time JIT)...",
                        flush=True,
                    )
                    self._logged_img_compile = True

                # Match `requires_grad` at tau=0 with tau>=1 so torch.compile traces `_imagine_step` only once
                s_cur = starts.states.detach().requires_grad_(True) if use_compiled_rollout else starts.states
                prev_a = (
                    starts.prev_actions.detach().requires_grad_(True) if use_compiled_rollout else starts.prev_actions
                )
                prev2_a = prev_a
                m_ids = starts.motion_ids
                t_0 = starts.time_steps
                xi = starts.xi

                hist_cur = (
                    self.task.init_tracker_history_from_states(
                        starts.hist_states, starts.hist_last_actions, xi=xi
                    )
                    if is_tracker
                    else None
                )
                if hist_cur is not None and use_compiled_rollout:
                    hist_cur = hist_cur.detach().requires_grad_(True)
                fk_cur = self.task.run_fk(s_cur)

                imagined_return = torch.zeros(B, dtype=torch.float32, device=self.device)
                bc_loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
                discount_weight = torch.ones(B, dtype=torch.float32, device=self.device)
                step_obj_rew_sum = torch.zeros((), dtype=torch.float32, device=self.device)

                need_imag_critic = (not warmup_only) and (self.cfg.imag_critic_updates > 0)
                if need_imag_critic:
                    img_states = torch.empty((H + 1, B, SDIM), dtype=torch.float32, device=self.device)
                    img_prev_acts = torch.empty((H + 1, B, prev_a.shape[-1]), dtype=torch.float32, device=self.device)
                    img_rewards = torch.empty((H, B), dtype=torch.float32, device=self.device)
                    img_not_done = torch.empty((H, B), dtype=torch.float32, device=self.device)
                    img_states[0].copy_(starts.states.detach())
                    img_prev_acts[0].copy_(starts.prev_actions.detach())

                if warmup_only:
                    # Tracker pure-BC warmup: one step, no world model
                    obs_pol = self._policy_obs(s_cur, prev_a, m_ids, t_0, xi, fk_cur, hist_cur)
                    a_mean = self._raw_policy.forward_mean(obs_pol, update_rms=False)
                    bc_term = self._teacher_bc(s_cur, prev_a, m_ids, t_0, xi, fk_cur, a_mean).mean()
                    policy_loss = bc_term
                else:
                    # Mark step boundary ONCE per H-step rollout+backward so earlier steps' activations stay valid
                    self._mark_cudagraph_step()
                    for tau in range(H):
                        t_cur = t_0 + tau
                        (
                            s_next,
                            fk_next,
                            a_cur,
                            r_step,
                            p_term,
                            traj_done,
                            v_done,
                            step_bc,
                            obj_rew,
                            hist_next,
                        ) = self._img_step_fn(s_cur, fk_cur, prev_a, m_ids, t_cur, xi, hist_cur)
                        bc_loss_sum = bc_loss_sum + (discount_weight.detach() * step_bc).mean()
                        step_obj_rew_sum = step_obj_rew_sum + obj_rew.mean()

                        if jerk_w > 0.0 and tau >= 1:
                            r_actor = r_step - jerk_w * ((a_cur - 2.0 * prev_a + prev2_a) ** 2).sum(dim=-1) * dt
                        else:
                            r_actor = r_step

                        done_boot = gamma * (1.0 - p_term) * v_done * traj_done
                        not_done_step = (1.0 - p_term) * (1.0 - traj_done)

                        imagined_return = imagined_return + discount_weight * (r_actor + done_boot)
                        discount_weight = discount_weight * gamma * not_done_step

                        if need_imag_critic:
                            img_states[tau + 1].copy_(s_next.detach())
                            img_prev_acts[tau + 1].copy_(a_cur.detach())
                            img_rewards[tau].copy_((r_step + done_boot).detach())
                            img_not_done[tau].copy_(not_done_step.detach())

                        # Optional Jacobian damping; when alpha == 1.0, passes gradients through unattenuated
                        if alpha < 1.0:
                            s_cur = s_next * alpha + s_next.detach() * (1.0 - alpha)
                            fk_cur = fk_next.scale_grad(alpha, self.fk.anchor_tracked_idx)
                        else:
                            s_cur = s_next
                            fk_cur = fk_next
                        prev2_a = prev_a
                        prev_a = a_cur
                        hist_cur = hist_next

                    # Value-clipped tail bootstrap V(s_H) via frozen ensemble Critic
                    ref_fut_tail = self.bridge.motion_ref.get_future_window(m_ids, t_0 + H)
                    obs_tail = self.task.build_privileged_obs(s_cur, prev_a, ref_fut_tail, fk_res=fk_cur, xi=xi)
                    v_tail = self.critic(obs_tail, update_rms=False, pessimistic=self.cfg.pessimistic_critic).clamp(
                        v_min, v_max
                    )
                    imagined_return = imagined_return + discount_weight * v_tail

                    bc_term = bc_loss_sum / float(H)
                    policy_loss = -imagined_return.mean() + bc_coef * bc_term

                self.policy_opt.zero_grad(set_to_none=True)
                policy_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(pol_params, self.cfg.max_grad_norm)
                _zero_nonfinite_grads_(pol_params, grad_norm)
                if self.cfg.grad_value_clip is not None and self.cfg.grad_value_clip > 0.0:
                    nn.utils.clip_grad_value_(pol_params, self.cfg.grad_value_clip)
                self.policy_opt.step()
                self._update_ema_policy()

                if need_imag_critic:
                    imag_v_accum = imag_v_accum + self._fit_imagined_critic(
                        img_states, img_prev_acts, img_rewards, img_not_done, m_ids, t_0, xi
                    )

                ret_accum = ret_accum + imagined_return.mean().detach()
                bc_accum = bc_accum + bc_term.detach()
                surv_accum = surv_accum + discount_weight.mean().detach()
                obj_rew_accum = obj_rew_accum + (step_obj_rew_sum / float(H)).detach()
        finally:
            _freeze_params(self._raw_world_model, False)
            _freeze_params(self._raw_critic, False)

        denom = float(max(1, self.cfg.policy_updates_per_iter))
        metrics = {
            "policy/imagined_return": float((ret_accum / denom).item()),
            "policy/tail_survival_weight": float((surv_accum / denom).item()),
            "policy/imagined_obj_rew": float((obj_rew_accum / denom).item()),
            "policy/collect_noise": float(self.cfg.collect_action_noise),
        }
        if (not warmup_only) and self.cfg.imag_critic_updates > 0:
            metrics["critic/imag_loss"] = float((imag_v_accum / denom).item())
        if self.teacher_policy is not None:
            metrics["policy/bc_loss"] = float((bc_accum / denom).item())
            metrics["policy/bc_coef"] = bc_coef
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing & Main Training Loop
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, iteration: int) -> None:
        raw_pol = _unwrap_compiled(self._raw_policy)
        ema_pol = _unwrap_compiled(self.ema_policy)
        raw_crit = _unwrap_compiled(self._raw_critic)
        raw_wm = _unwrap_compiled(self._raw_world_model)

        # Sync latest nominal actuator & normalizer buffers before saving
        raw_pol.sync_actuator_buffers(self.fk.default_joint_pos, self.fk.action_scale)
        self._sync_ema_buffers()
        ema_pol.sync_actuator_buffers(self.fk.default_joint_pos, self.fk.action_scale)

        # Export smooth EMA policy as primary `policy_state_dict` so `play.py`, Teacher BC & MuJoCo Sim2Sim
        # automatically get jitter-free inference, while retaining `raw_policy_state_dict` for exact resume!
        export_pol = ema_pol if self.cfg.ema_decay > 0.0 else raw_pol
        payload = {
            "format": "supermimic_sugar_v4",
            "stage": self.stage,
            "task_name": self.task.profile.task_name,
            "joint_names": list(self.fk.env_joint_names),
            "iteration": iteration,
            "next_iteration": iteration + 1,
            "obs_dim": raw_pol.obs_dim,
            "action_dim": raw_pol.action_dim,
            "hidden_dims": raw_pol.hidden_dims,
            "use_ref_residual": raw_pol.use_ref_residual,
            "cfg": asdict(self.cfg),
            "policy_state_dict": _strip_compiled_prefix(export_pol.state_dict()),
            "raw_policy_state_dict": _strip_compiled_prefix(raw_pol.state_dict()),
            "critic_state_dict": _strip_compiled_prefix(raw_crit.state_dict()),
            "world_model_state_dict": _strip_compiled_prefix(raw_wm.state_dict()),
            "policy_opt_state_dict": self.policy_opt.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "wm_opt_state_dict": self.wm_opt.state_dict(),
        }
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, str(path))

    def load_checkpoint(self, path: str | Path) -> int:
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        raw_pol = _unwrap_compiled(self._raw_policy)
        ema_pol = _unwrap_compiled(self.ema_policy)

        ema_sd = _policy_state_dict_compat(payload["policy_state_dict"], raw_pol.action_dim)
        raw_sd = (
            _policy_state_dict_compat(payload["raw_policy_state_dict"], raw_pol.action_dim)
            if "raw_policy_state_dict" in payload
            else ema_sd
        )
        raw_pol.load_state_dict(raw_sd)
        ema_pol.load_state_dict(ema_sd)

        _unwrap_compiled(self._raw_critic).load_state_dict(_strip_compiled_prefix(payload["critic_state_dict"]))
        _unwrap_compiled(self._raw_world_model).load_state_dict(
            _strip_compiled_prefix(payload["world_model_state_dict"])
        )
        if "policy_opt_state_dict" in payload:
            self.policy_opt.load_state_dict(payload["policy_opt_state_dict"])
        if "critic_opt_state_dict" in payload:
            self.critic_opt.load_state_dict(payload["critic_opt_state_dict"])
        if "wm_opt_state_dict" in payload:
            self.wm_opt.load_state_dict(payload["wm_opt_state_dict"])

        if "next_iteration" in payload:
            self.current_iter = int(payload["next_iteration"])
        else:
            self.current_iter = int(payload.get("iteration", -1)) + 1
        return self.current_iter

    def auto_resume(self) -> Path | None:
        """Resume from `model_latest.pt` (or highest valid `model_<iter>.pt`) in `self.log_dir` if present."""
        candidates: list[Path] = []
        latest_p = self.log_dir / "model_latest.pt"
        if latest_p.is_file():
            candidates.append(latest_p)

        numbered: list[tuple[int, Path]] = []
        for p in self.log_dir.glob("model_*.pt"):
            stem_num = p.stem.removeprefix("model_")
            if stem_num.isdigit():
                numbered.append((int(stem_num), p))
        numbered.sort(key=lambda item: item[0], reverse=True)
        candidates.extend(p for _, p in numbered)

        for cand in candidates:
            try:
                next_it = self.load_checkpoint(cand)
                print(f"[SuperMimic] Auto-resumed from {cand} (next_iteration={next_it})", flush=True)
                return cand
            except Exception as exc:
                print(f"[SuperMimic] WARNING: Skipping unreadable checkpoint '{cand}' ({exc})", flush=True)
        return None

    def _sync(self) -> None:
        """Synchronize CUDA so the per-phase timers measure GPU work, not just kernel launches."""
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()

    def learn(self, max_iterations: int) -> Path:
        """Run full SuperMimic training loop for `max_iterations` and save SUGAR-compatible `model_*.pt`."""
        start_iter = self.current_iter
        steps_per_iter = self.cfg.steps_per_env * self.num_envs
        print(
            f"[SuperMimic Trainer] Starting stage='{self.stage}' | task='{self.task.profile.task_name}' | "
            f"envs={self.num_envs} | horizon={self.cfg.imagination_horizon} | "
            f"policy_batch={self.cfg.policy_batch_size}x{self.cfg.policy_updates_per_iter} | "
            f"wm={self.cfg.wm_hidden_dims} batch={self.cfg.wm_batch_size}x{self.cfg.wm_updates_per_iter} | "
            f"value_clip=({self.value_clip[0]:.2f}, {self.value_clip[1]:.2f}) | "
            f"iters={start_iter}..{max_iterations}"
        )

        last_ckpt_path = self.log_dir / f"model_{max(0, max_iterations - 1)}.pt"
        if start_iter >= max_iterations:
            if not last_ckpt_path.is_file():
                self.save_checkpoint(last_ckpt_path, iteration=max(0, max_iterations - 1))
                print(f"[SuperMimic] Saved checkpoint -> {last_ckpt_path} (iter={max(0, max_iterations - 1)})")
            if self.tb_writer is not None:
                self.tb_writer.flush()
                self.tb_writer.close()
            return last_ckpt_path

        min_warmup_steps = max(self.cfg.wm_seq_len, 5)
        if start_iter > 0 and self.replay.size < min_warmup_steps:
            while self.replay.size < min_warmup_steps:
                self.collect_rollout()

        for it in range(start_iter, max_iterations):
            lr_frac = self._set_lr(it, max_iterations)
            self._sync()
            t0 = time.perf_counter()
            critic_obs_flat, returns_flat, c_stats = self.collect_rollout()
            self._sync()
            t1 = time.perf_counter()
            wm_stats = self.update_world_model()
            self._sync()
            t2 = time.perf_counter()
            crit_stats = self.update_critic(critic_obs_flat, returns_flat)
            self._sync()
            t3 = time.perf_counter()
            pol_stats = self.update_policy(it)
            self._sync()
            t4 = time.perf_counter()
            dt = max(1e-6, t4 - t0)

            self.current_iter = it + 1
            total_timesteps = (it + 1) * steps_per_iter
            fps = int(steps_per_iter / dt)

            perf_stats = {
                "perf/fps": float(fps),
                "perf/iter_s": dt,
                "perf/collect_s": t1 - t0,
                "perf/wm_s": t2 - t1,
                "perf/critic_s": t3 - t2,
                "perf/policy_s": t4 - t3,
                "perf/lr_frac": lr_frac,
            }
            all_metrics = {**c_stats, **wm_stats, **crit_stats, **pol_stats, **perf_stats}
            should_log = (
                (it < max(10, self.cfg.wm_warmup_iters + 1))
                or (it % 10 == 0)
                or (it == max_iterations - 1)
            )
            if self.tb_writer is not None:
                for k, v in all_metrics.items():
                    self.tb_writer.add_scalar(k, v, it)
                if should_log:
                    self.tb_writer.flush()

            if should_log:
                bc_str = (
                    f" | bc={pol_stats.get('policy/bc_loss', 0.0):.4f} (w={pol_stats.get('policy/bc_coef', 0.0):.2f})"
                    if "policy/bc_loss" in pol_stats
                    else ""
                )
                proxy = c_stats.get("rollout_proxy/det_t0_success_rate")
                proxy_str = f" | t0_succ={proxy:.2f}" if proxy is not None else ""
                print(
                    f"Learning iteration {it}/{max_iterations} | Total timesteps: {total_timesteps} | "
                    f"FPS: {fps} | sim_rew={c_stats['collect/sim_step_reward']:.4f} | "
                    f"img_ret={pol_stats.get('policy/imagined_return', 0.0):.3f} | "
                    f"wm_pos={wm_stats.get('wm/pos_err_mm', 0.0):.1f}mm | "
                    f"wm_obj={wm_stats.get('wm/obj_err_mm', 0.0):.1f}mm | "
                    f"v_loss={crit_stats['critic/loss']:.4f}{proxy_str} | "
                    f"t(c/w/v/p)={t1 - t0:.1f}/{t2 - t1:.1f}/{t3 - t2:.1f}/{t4 - t3:.1f}s{bc_str}",
                    flush=True,
                )

            if (it > 0 and it % self.cfg.save_interval == 0) or (it == max_iterations - 1):
                ckpt_p = self.log_dir / f"model_{it}.pt"
                self.save_checkpoint(ckpt_p, iteration=it)
                self.save_checkpoint(self.log_dir / "model_latest.pt", iteration=it)
                last_ckpt_path = ckpt_p
                print(f"[SuperMimic] Saved checkpoint -> {ckpt_p} (iter={it})", flush=True)

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        return last_ckpt_path

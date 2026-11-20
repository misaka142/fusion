"""Unified SuperMimic Trainer for SUGAR Stage 1 (Refiner) and Stage 2 (Tracker).

Replaces:
- Stage 1 RSL-RL `OnPolicyRunner` (PPO) -> `SuperMimicTrainer` (stage="refiner")
- Stage 2 SUGAR `BCPPO` -> `SuperMimicTrainer` (stage="tracker" + annealed Refiner-Teacher BC in imagination)

Algorithm per iteration:
1. Collect `steps_per_env` (default 24) transitions in IsaacLab (mixed deterministic + exploratory envs),
   pushing transitions into GPU `HOIReplayBuffer` (preserving true `ep_id` while flagging reset/push breaks)
   and building GAE(gamma, lam) targets with exact current-step timeout value bootstrap.
2. Train `JointWorldModel` via `k=8`-step autoregressive rollout (local state Huber loss + contact BCE).
3. Train `SuperMimicCritic` ensemble on real simulator TD(lambda) targets (zero model value bias).
4. Train `SuperMimicPolicy` via `H=32`-step differentiable rollout through frozen `JointWorldModel`
   + analytical SUGAR human-object rewards (aligned with `ref[t_cur]`) + smooth soft terminations
   + value-clipped frozen `SuperMimicCritic` tail bootstrap (+ annealed Teacher BC in Tracker mode).
"""

from __future__ import annotations

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
    RefFrame,
    RefFutureWindow,
    TRACKER_OBS_DIM,
    TrackerHistoryState,
    resolve_task_profile,
)
from .networks import JointWorldModel, SuperMimicCritic, SuperMimicPolicy
from .replay import HOIReplayBuffer
from .urdf_kinematics import FKResult


@dataclass
class SuperMimicTrainCfg:
    """Hyperparameters for SuperMimic x SUGAR training (PyTorch 2.7 compatible)."""

    stage: str = "auto"                              # "auto" | "refiner" | "tracker"
    steps_per_env: int = 24                          # Matches SUGAR num_steps_per_env=24 for 1:1 iteration parity
    replay_capacity: int = 256                       # Ring buffer length per env on GPU (256 * 2048 ~= 524k steps)
    collect_det_frac: float = 0.25                   # First 25% envs run deterministic actions (protects init_pool)

    # World model settings
    wm_hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    wm_seq_len: int = 8                              # Autoregressive unroll horizon k for world model training
    wm_updates_per_iter: int = 16
    wm_batch_size: int = 2048
    wm_lr: float = 1e-3
    wm_contact_loss_weight: float = 0.5
    wm_obj_loss_weight: float = 1.5
    wm_warmup_iters: int = 8                         # Fit WM & Critic (and pure BC if Tracker) before enabling BPTT

    # Policy BPTT & decoupled exploration settings
    policy_hidden_dims: tuple[int, ...] = (512, 256, 128)
    use_ref_residual: bool = True
    legs_use_residual: bool = True                   # Apply smooth q_ref feed-forward baseline across all 29 joints
    init_noise_std: float = 0.20
    leg_explore_std_init: float = 0.12
    arm_explore_std_init: float = 0.30
    explore_std_final_scale: float = 0.25
    imagination_horizon: int = 32                    # H=32 step differentiable imagination rollout
    policy_updates_per_iter: int = 8
    policy_batch_size: int = 2048
    frontier_sample_frac: float = 0.40               # 40% imagination starts prioritize highest reached trajectory steps
    residual_rate_weight: float = 0.15               # Smoothness penalty on 1st-order residual rate (dampens oscillation)
    residual_jerk_weight: float = 0.10               # 2nd-order residual jerk penalty (suppresses 25Hz sawtooth chatter)
    policy_lr: float = 3e-4
    lr_min_scale: float = 0.15                       # Cosine LR decay floor multiplier
    bptt_grad_decay: float = 0.98                    # Mild state Jacobian damping across 32 steps for stability
    grad_value_clip: float = 0.10                    # Per-element BPTT gradient clamp before global norm clip
    entropy_coef: float = 0.0
    value_clip_range: tuple[float, float] = (-25.0, 15.0)  # Protects long BPTT runs from OOD critic spikes

    # Critic TD(lambda) settings
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_heads: int = 2
    critic_epochs: int = 4
    critic_minibatches: int = 4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95

    # Stage 2 Tracker Teacher-BC distillation settings (replaces BCPPO)
    bc_init_coef: float = 1.0
    bc_min_coef: float = 0.05
    bc_anneal_iters: int = 3500                      # Reaches bc_min_coef at 70% of standard 5001-iter run

    # Checkpointing & logging
    max_grad_norm: float = 1.0
    save_interval: int = 500
    enable_tensorboard: bool = True
    compile_models: bool = False                     # Optional torch.compile; False by default for IsaacSim safety
    compile_backend: str = "inductor"                # Backend passed to `torch.compile` when `compile_models=True`
    compile_mode: str = "default"                    # Inductor mode ('default' is safe with IsaacLab/PhysX)


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _build_adamw(
    params, lr: float, weight_decay: float, device: torch.device
) -> torch.optim.AdamW:
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _unwrap_compiled(module: nn.Module) -> nn.Module:
    """Return underlying `nn.Module` if wrapped by `torch.compile` (`OptimizedModule`)."""
    return getattr(module, "_orig_mod", module)


def _strip_compiled_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip `_orig_mod.` prefix from state_dict keys so compiled & eager checkpoints interoperate."""
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def _compat_policy_state_dict(
    state_dict: dict[str, Any], policy: SuperMimicPolicy | None = None
) -> dict[str, Any]:
    """Normalize policy checkpoint keys while preserving or backfilling `residual_mask`."""
    sd = _strip_compiled_prefix(state_dict)
    if "residual_mask" not in sd and policy is not None and hasattr(policy, "residual_mask"):
        sd["residual_mask"] = policy.residual_mask.clone()
    return sd


def load_supermimic_policy(
    ckpt_path: str | Path, device: torch.device | str = "cuda"
) -> SuperMimicPolicy:
    """Load a self-contained `SuperMimicPolicy` from a saved `.pt` checkpoint for inference or Teacher BC."""
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "policy_state_dict" not in payload:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is not a SuperMimic checkpoint (missing 'policy_state_dict')."
        )
    raw_sd = payload["policy_state_dict"]
    has_mask = any(k.endswith("residual_mask") for k in raw_sd)
    legs_res = bool(payload.get("legs_use_residual", not has_mask))
    policy = SuperMimicPolicy(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload.get("action_dim", 29)),
        hidden_dims=tuple(payload.get("hidden_dims", (512, 256, 128))),
        use_ref_residual=bool(payload.get("use_ref_residual", True)),
        joint_names=payload.get("joint_names"),
        legs_use_residual=legs_res,
    ).to(device)
    policy.load_state_dict(_compat_policy_state_dict(raw_sd, policy), strict=True)
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

        # Joint error-propagation weights from G1 kinematic tree
        self.joint_weights = self.fk.compute_joint_weights().to(self.device)

        # Build World Model, Actor, Critic (retaining uncompiled references for clean attribute/checkpoint access)
        self._raw_world_model = JointWorldModel(hidden_dims=self.cfg.wm_hidden_dims).to(self.device)
        self._raw_policy = SuperMimicPolicy(
            obs_dim=policy_obs_dim,
            action_dim=29,
            hidden_dims=self.cfg.policy_hidden_dims,
            default_joint_pos=self.fk.default_joint_pos,
            action_scale=self.fk.action_scale,
            use_ref_residual=self.cfg.use_ref_residual,
            joint_names=self.fk.env_joint_names,
            legs_use_residual=self.cfg.legs_use_residual,
            init_std=self.cfg.init_noise_std,
        ).to(self.device)
        self._raw_critic = SuperMimicCritic(
            obs_dim=critic_obs_dim,
            hidden_dims=self.cfg.critic_hidden_dims,
            num_heads=self.cfg.critic_heads,
        ).to(self.device)

        explore_vec = [
            self.cfg.leg_explore_std_init
            if any(tok in n for tok in ("hip_", "knee_", "ankle_"))
            else self.cfg.arm_explore_std_init
            for n in self.fk.env_joint_names
        ]
        self._explore_std_base = torch.tensor(explore_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.world_model: nn.Module = self._raw_world_model
        self.policy: nn.Module = self._raw_policy
        self.critic: nn.Module = self._raw_critic

        # Optional frozen Refiner Teacher for Stage 2 Tracker distillation
        self.teacher_policy: SuperMimicPolicy | None = None
        if self.stage == "tracker" and teacher_ckpt is not None:
            print(f"[SuperMimic Trainer] Loading frozen Refiner Teacher from: {teacher_ckpt}")
            self.teacher_policy = load_supermimic_policy(teacher_ckpt, device=self.device)

        self._wm_loss_fn = self._wm_window_loss_core
        self._img_step_fn = self._img_step_core
        self._zero_scalar = torch.zeros((), dtype=torch.float32, device=self.device)
        self._one_scalar = torch.ones((), dtype=torch.float32, device=self.device)

        if self.cfg.compile_models and hasattr(torch, "compile"):
            if self.cfg.compile_backend == "inductor" and (
                self.device.type != "cuda" or not _has_triton()
            ):
                print(
                    "[SuperMimic Trainer] WARNING: Triton/CUDA Inductor unavailable on this host; "
                    "running compiled step functions in eager mode."
                )
            else:
                compile_kwargs: dict[str, Any] = {
                    "dynamic": False,
                    "backend": self.cfg.compile_backend,
                }
                if self.cfg.compile_backend == "inductor" and self.cfg.compile_mode:
                    compile_kwargs["mode"] = self.cfg.compile_mode
                self._wm_loss_fn = torch.compile(self._wm_window_loss_core, **compile_kwargs)
                self._img_step_fn = torch.compile(self._img_step_core, **compile_kwargs)
                print(
                    f"[SuperMimic Trainer] Whole-step torch.compile enabled "
                    f"(backend={self.cfg.compile_backend}, mode={self.cfg.compile_mode})"
                )

        self.wm_opt = _build_adamw(
            self._raw_world_model.parameters(), lr=self.cfg.wm_lr, weight_decay=1e-5, device=self.device
        )
        self.policy_opt = _build_adamw(
            self._raw_policy.parameters(), lr=self.cfg.policy_lr, weight_decay=1e-5, device=self.device
        )
        self.critic_opt = _build_adamw(
            self._raw_critic.parameters(), lr=self.cfg.critic_lr, weight_decay=1e-5, device=self.device
        )

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

        # Read-only monitoring counters for deterministic t=0 rollout proxy and env logs
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

    def _iter_progress(self, iteration: int, max_iterations: int) -> float:
        return min(1.0, max(0.0, float(iteration) / float(max(1, max_iterations - 1))))

    def current_explore_std(self, iteration: int, max_iterations: int) -> Tensor:
        progress = self._iter_progress(iteration, max_iterations)
        scale = 1.0 - (1.0 - self.cfg.explore_std_final_scale) * progress
        return self._explore_std_base * scale

    def _apply_lr_schedule(self, iteration: int, max_iterations: int) -> float:
        progress = self._iter_progress(iteration, max_iterations)
        lr_min = self.cfg.lr_min_scale
        cos_scale = lr_min + 0.5 * (1.0 - lr_min) * (1.0 + math.cos(math.pi * progress))
        for opt, base_lr in (
            (self.wm_opt, self.cfg.wm_lr),
            (self.policy_opt, self.cfg.policy_lr),
            (self.critic_opt, self.cfg.critic_lr),
        ):
            for pg in opt.param_groups:
                pg["lr"] = base_lr * cos_scale
        return float(cos_scale)

    # ------------------------------------------------------------------
    # Phase 1: Real Environment Rollout Collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(
        self, iteration: int = 0, max_iterations: int = 1
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Collect `steps_per_env` transitions and compute GAE returns for Critic."""
        T = self.cfg.steps_per_env
        N = self.num_envs
        get_air_fn = getattr(self.bridge, "get_feet_air_time", None)
        lengths = self.bridge.motion_ref.lengths
        explore_std_cur = self.current_explore_std(iteration, max_iterations)

        critic_obs_buf = torch.empty((T, N, PRIVILEGED_OBS_DIM), dtype=torch.float32, device=self.device)
        rewards_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)
        dones_buf = torch.empty((T, N), dtype=torch.bool, device=self.device)
        values_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)

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
            v_pred = self.critic(obs_crit, update_rms=True, pessimistic=False)

            # Mixed deterministic (first 25% init_pool protectors) + annealed leg/arm exploration
            act_det = self._raw_policy.forward_mean(obs_pol, update_rms=False)
            if self.det_env_count < N:
                noise = torch.randn_like(act_det[self.det_env_count :]) * explore_std_cur
                act_stoch = act_det[self.det_env_count :] + noise
                action = (
                    torch.cat([act_det[: self.det_env_count], act_stoch], dim=0)
                    if self.det_env_count > 0
                    else act_stoch
                )
            else:
                action = act_det

            _, sim_rew, dones, timeouts, pushed, extras = self.bridge.step(action)
            s_post = self.bridge.get_joint_state()
            contacts = self.bridge.get_ground_truth_contacts()

            # Bootstrap timeout terminations using CURRENT step value `v_pred` (matching RSL-RL;
            # `next_obs_dict` for reset envs already belongs to the newly reset episode!)
            if timeouts.any():
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

            # Read-only rollout proxy on deterministic t=0 envs
            p_done = (dones & is_proxy).float()
            proxy_done = proxy_done + p_done.sum()
            proxy_succ = proxy_succ + (p_done * timeouts.float()).sum()
            prog = ((t_step + 1).float() / lengths[m_id].clamp_min(1).float()).clamp(max=1.0)
            proxy_prog = proxy_prog + (p_done * prog).sum()

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
            "collect/explore_std": float(explore_std_cur.mean().item()),
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

    def _wm_window_loss_core(
        self,
        s_init: Tensor,
        actions: Tensor,
        next_states: Tensor,
        contacts: Tensor,
        xi: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pure-tensor multi-step world-model autoregressive rollout loss (compilable)."""
        eff_k = int(actions.shape[1])
        s_cur = s_init
        loss_state_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        loss_contact_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        last_metrics: dict[str, Tensor] = {}

        for t in range(eff_k):
            s_cur, c_logits = self._raw_world_model.step(
                self.task,
                s_cur,
                actions[:, t],
                xi,
                update_rms=False,
            )
            l_s, last_metrics = self.task.world_model_state_loss(
                s_cur,
                next_states[:, t],
                self.joint_weights,
                obj_loss_weight=self.cfg.wm_obj_loss_weight,
            )
            l_c = F.binary_cross_entropy_with_logits(c_logits, contacts[:, t])
            loss_state_sum = loss_state_sum + l_s
            loss_contact_sum = loss_contact_sum + l_c

        denom_k = float(eff_k)
        total_loss = (loss_state_sum + self.cfg.wm_contact_loss_weight * loss_contact_sum) / denom_k
        c_avg = (loss_contact_sum / denom_k).detach()
        return (
            total_loss,
            c_avg,
            last_metrics["wm/loss_robot"],
            last_metrics["wm/loss_obj"],
            last_metrics["wm/pos_err_mm"],
            last_metrics["wm/obj_err_mm"],
        )

    def update_world_model(self) -> dict[str, float]:
        if self.replay.size < 2:
            return {}

        self.world_model.train()
        agg_tensors: dict[str, Tensor] = {}
        k_req = self.cfg.wm_seq_len
        actual_updates = 0

        for _ in range(self.cfg.wm_updates_per_iter):
            batch = self.replay.sample_wm_sequences(self.cfg.wm_batch_size, seq_len=k_req)
            if batch is None:
                continue
            eff_k = int(batch.actions.shape[1])
            with torch.no_grad():
                s_prev_seq = torch.cat([batch.s_init.unsqueeze(1), batch.next_states[:, :-1]], dim=1)
                true_deltas = self.task.compute_state_delta(s_prev_seq, batch.next_states)
                self._raw_world_model.delta_rms.update(true_deltas)
                fk_init = self.task.run_fk(batch.s_init, with_quat=False)
                sf_init = self.task.state_features(batch.s_init, fk_res=fk_init)
                self._raw_world_model.feat_rms.update(sf_init)
                self._raw_world_model.xi_rms.update(batch.xi)

            loss_fn = self._wm_loss_fn if eff_k == k_req else self._wm_window_loss_core
            total_loss, c_avg, l_rob, l_obj, pos_mm, obj_mm = loss_fn(
                batch.s_init, batch.actions, batch.next_states, batch.contacts, batch.xi
            )

            self.wm_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self._raw_world_model.parameters(), self.cfg.max_grad_norm)
            if torch.isfinite(grad_norm):
                self.wm_opt.step()

            actual_updates += 1
            for key, val in (
                ("wm/total_loss", total_loss.detach()),
                ("wm/contact_bce", c_avg),
                ("wm/loss_robot", l_rob),
                ("wm/loss_obj", l_obj),
                ("wm/pos_err_mm", pos_mm),
                ("wm/obj_err_mm", obj_mm),
            ):
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
                grad_norm = nn.utils.clip_grad_norm_(self._raw_critic.parameters(), self.cfg.max_grad_norm)
                if torch.isfinite(grad_norm):
                    self.critic_opt.step()

                loss_accum = loss_accum + loss.detach()
                n_steps += 1

        return {"critic/loss": float((loss_accum / float(max(1, n_steps))).item())}

    # ------------------------------------------------------------------
    # Phase 4: 32-Step Differentiable World-Model BPTT (+ Tracker BC)
    # ------------------------------------------------------------------

    def _img_step_core(
        self,
        s_cur: Tensor,
        prev_a: Tensor,
        fk_tpos: Tensor,
        fk_tmat: Tensor,
        fk_tvel: Tensor,
        fk_tang: Tensor,
        fk_cpos: Tensor,
        fk_cvel: Tensor,
        fk_aquat: Tensor,
        fk_rmat: Tensor,
        prev_res: Tensor,
        prev2_res: Tensor,
        step0_flag: Tensor,
        step_ge2_flag: Tensor,
        feet_air_cur: Tensor,
        prev_feet_gnd: Tensor,
        t_cur: Tensor,
        xi: Tensor,
        ref_jpos: Tensor,
        ref_jvel: Tensor,
        ref_bpos: Tensor,
        ref_bquat: Tensor,
        ref_brot: Tensor | None,
        ref_bv: Tensor,
        ref_bw: Tensor,
        ref_opos: Tensor,
        ref_oquat: Tensor,
        ref_ov: Tensor,
        ref_ow: Tensor,
        ref_contact: Tensor,
        ref_o2b_pos: Tensor | None,
        ref_o2b_rot: Tensor | None,
        ref_anc_pg_z: Tensor | None,
        fut_jpos: Tensor,
        fut_jvel: Tensor,
        fut_anc_pos: Tensor,
        fut_anc_quat: Tensor,
        fut_obj_pos: Tensor,
        fut_obj_quat: Tensor,
        fut_obj_v: Tensor,
        fut_obj_w: Tensor,
        hist_ang: Tensor | None = None,
        hist_q: Tensor | None = None,
        hist_qd: Tensor | None = None,
        hist_act: Tensor | None = None,
        hist_grav: Tensor | None = None,
        t_fut_jpos: Tensor | None = None,
        t_fut_jvel: Tensor | None = None,
        t_fut_anc_pos: Tensor | None = None,
        t_fut_anc_quat: Tensor | None = None,
        t_fut_obj_pos: Tensor | None = None,
        t_fut_obj_quat: Tensor | None = None,
        t_fut_obj_v: Tensor | None = None,
        t_fut_obj_w: Tensor | None = None,
    ) -> tuple[Any, ...]:
        """Single branchless 50Hz imagination step (policy + WM + FK + reward), ready for torch.compile."""
        anc_i = self.fk.anchor_tracked_idx
        fk_cur = FKResult(
            tracked_pos_w=fk_tpos,
            tracked_rotmat_w=fk_tmat,
            tracked_quat_w=None,
            tracked_lin_vel_w=fk_tvel,
            tracked_ang_vel_w=fk_tang,
            contact_pos_w=fk_cpos,
            contact_lin_vel_w=fk_cvel,
            anchor_pos_w=fk_tpos[:, anc_i],
            anchor_quat_w=fk_aquat,
            anchor_rotmat_w=fk_tmat[:, anc_i],
            anchor_lin_vel_w=fk_tvel[:, anc_i],
            anchor_ang_vel_w=fk_tang[:, anc_i],
            root_rotmat_w=fk_rmat,
        )
        ref_cur = RefFrame(
            joint_pos=ref_jpos,
            joint_vel=ref_jvel,
            body_pos_w=ref_bpos,
            body_quat_w=ref_bquat,
            body_lin_vel_w=ref_bv,
            body_ang_vel_w=ref_bw,
            obj_pos_w=ref_opos,
            obj_quat_w=ref_oquat,
            obj_lin_vel_w=ref_ov,
            obj_ang_vel_w=ref_ow,
            contact_label=ref_contact,
            body_rotmat_w=ref_brot,
            o2b_pos_ref=ref_o2b_pos,
            o2b_rotmat_ref=ref_o2b_rot,
            anc_pg_z_ref=ref_anc_pg_z,
        )
        ref_fut_cur = RefFutureWindow(
            joint_pos=fut_jpos,
            joint_vel=fut_jvel,
            anchor_pos_w=fut_anc_pos,
            anchor_quat_w=fut_anc_quat,
            obj_pos_w=fut_obj_pos,
            obj_quat_w=fut_obj_quat,
            obj_lin_vel_w=fut_obj_v,
            obj_ang_vel_w=fut_obj_w,
        )

        is_tracker = self.stage == "tracker"
        if is_tracker:
            hist_cur = TrackerHistoryState(
                base_ang_vel=hist_ang,
                joint_pos_rel=hist_q,
                joint_vel_rel=hist_qd,
                actions=hist_act,
                project_gravity=hist_grav,
            )
            cmd_36 = ref_cur.tracker_command_36()
            obs_pol = self.task.build_tracker_obs(
                s_cur, cmd_36, hist_cur, fk_res=fk_cur, add_noise=True
            )
        else:
            obs_pol = self.task.build_privileged_obs(
                s_cur, prev_a, ref_fut_cur, fk_res=fk_cur, xi=xi
            )

        a_mean, res_cur = self._raw_policy.forward_mean_and_residual(obs_pol, update_rms=False)
        a_cur = a_mean

        if self.teacher_policy is not None:
            t_fut_win = (
                RefFutureWindow(
                    joint_pos=t_fut_jpos,
                    joint_vel=t_fut_jvel,
                    anchor_pos_w=t_fut_anc_pos,
                    anchor_quat_w=t_fut_anc_quat,
                    obj_pos_w=t_fut_obj_pos,
                    obj_quat_w=t_fut_obj_quat,
                    obj_lin_vel_w=t_fut_obj_v,
                    obj_ang_vel_w=t_fut_obj_w,
                )
                if t_fut_jpos is not None
                else ref_fut_cur
            )
            with torch.no_grad():
                obs_teacher = self.task.build_privileged_obs(
                    s_cur, prev_a, t_fut_win, fk_res=fk_cur, xi=xi
                )
                a_teacher = self.teacher_policy.act_inference(obs_teacher)
            step_bc = ((a_mean - a_teacher) ** 2).mean(dim=-1)
        else:
            step_bc = self._zero_scalar

        step_res_rate = ((res_cur - prev_res) ** 2).sum(dim=-1) * (1.0 - step0_flag)
        step_res_jerk = ((res_cur - 2.0 * prev_res + prev2_res) ** 2).sum(dim=-1) * step_ge2_flag
        next_prev2_res = prev_res
        next_prev_res = res_cur * (1.0 - step0_flag) + res_cur.detach() * step0_flag

        s_next, c_logits = self._raw_world_model.step(
            self.task, s_cur, a_cur, xi, fk_res=fk_cur, update_rms=False
        )
        fk_next = self.task.run_fk(s_next, with_quat=False)

        r_step, p_term, r_info = self.task.compute_reward_and_termination(
            s_prev=s_cur,
            s_next=s_next,
            raw_action=a_cur,
            prev_raw_action=prev_a,
            ref_cur=ref_cur,
            contact_logits=c_logits,
            fk_next=fk_next,
            xi=xi,
            feet_air_time=feet_air_cur,
            prev_feet_gnd=prev_feet_gnd,
            time_steps=t_cur,
            imagination_shaping=True,
        )

        if is_tracker:
            hist_next = self.task.advance_tracker_history(hist_cur, s_next, a_cur, xi=xi)
            h_ang_n: Tensor | None = hist_next.base_ang_vel
            h_q_n: Tensor | None = hist_next.joint_pos_rel
            h_qd_n: Tensor | None = hist_next.joint_vel_rel
            h_act_n: Tensor | None = hist_next.actions
            h_grav_n: Tensor | None = hist_next.project_gravity
        else:
            h_ang_n = h_q_n = h_qd_n = h_act_n = h_grav_n = None

        alpha = self.cfg.bptt_grad_decay
        s_next_scaled = s_next * alpha + s_next.detach() * (1.0 - alpha)
        fk_next_scaled = fk_next.scale_grad(alpha, anc_i)

        return (
            s_next,
            s_next_scaled,
            fk_next_scaled.tracked_pos_w,
            fk_next_scaled.tracked_rotmat_w,
            fk_next_scaled.tracked_lin_vel_w,
            fk_next_scaled.tracked_ang_vel_w,
            fk_next_scaled.contact_pos_w,
            fk_next_scaled.contact_lin_vel_w,
            fk_next_scaled.anchor_quat_w,
            fk_next_scaled.root_rotmat_w,
            a_cur,
            next_prev_res,
            next_prev2_res,
            r_info["next_feet_air_time"],
            r_info["curr_feet_gnd"],
            r_step,
            p_term,
            step_bc,
            step_res_rate,
            step_res_jerk,
            r_info["rew/tracking_obj"].mean(),
            h_ang_n,
            h_q_n,
            h_qd_n,
            h_act_n,
            h_grav_n,
        )

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
        K = self.bridge.motion_ref.future_frames
        v_min, v_max = self.cfg.value_clip_range
        has_slice_api = hasattr(self.bridge.motion_ref, "get_imagination_slice")

        ret_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        bc_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        surv_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        obj_rew_accum = torch.zeros((), dtype=torch.float32, device=self.device)
        res_jerk_accum = torch.zeros((), dtype=torch.float32, device=self.device)

        try:
            for _ in range(self.cfg.policy_updates_per_iter):
                starts = self.replay.sample_imagination_starts(
                    self.cfg.policy_batch_size,
                    need_history=is_tracker,
                    frontier_frac=self.cfg.frontier_sample_frac,
                )
                B = starts.states.shape[0]
                s_cur = starts.states
                prev_a = starts.prev_actions
                m_ids = starts.motion_ids
                t_0 = starts.time_steps
                xi = starts.xi
                feet_air_cur = starts.feet_air_time
                prev_feet_gnd = (feet_air_cur <= 1e-4).float()

                hist_cur = (
                    self.task.init_tracker_history_from_states(
                        starts.hist_states, starts.hist_last_actions, xi=xi
                    )
                    if is_tracker
                    else None
                )
                fk_cur = self.task.run_fk(s_cur, with_quat=False)

                if warmup_only:
                    # Pure 1-step Teacher BC during early WM warmup iterations
                    ref0 = self.bridge.motion_ref.get_frame(m_ids, t_0)
                    obs_pol = self.task.build_tracker_obs(
                        s_cur, ref0.tracker_command_36(), hist_cur, fk_res=fk_cur, add_noise=True
                    )
                    a_mean, _ = self._raw_policy.forward_mean_and_residual(obs_pol, update_rms=False)
                    t_ref_adapter = self.bridge.teacher_motion_ref or self.bridge.motion_ref
                    t_fut = t_ref_adapter.get_future_window(m_ids, t_0)
                    with torch.no_grad():
                        obs_teacher = self.task.build_privileged_obs(
                            s_cur, prev_a, t_fut, fk_res=fk_cur, xi=xi
                        )
                        a_teacher = self.teacher_policy.act_inference(obs_teacher)
                    bc_term = ((a_mean - a_teacher) ** 2).mean()
                    policy_loss = bc_term
                    self.policy_opt.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    if self.cfg.grad_value_clip > 0.0:
                        nn.utils.clip_grad_value_(self._raw_policy.parameters(), self.cfg.grad_value_clip)
                    grad_norm = nn.utils.clip_grad_norm_(self._raw_policy.parameters(), self.cfg.max_grad_norm)
                    if torch.isfinite(grad_norm):
                        self.policy_opt.step()
                    bc_accum = bc_accum + bc_term.detach()
                    surv_accum = surv_accum + 1.0
                    continue

                if has_slice_api:
                    ref_slice = self.bridge.motion_ref.get_imagination_slice(
                        m_ids, t_0, H, need_ref_cur=True
                    )
                    done_taus = set(
                        torch.nonzero(ref_slice.traj_done.any(dim=0), as_tuple=False).flatten().tolist()
                    )
                    t_slice = None
                    if self.teacher_policy is not None:
                        t_ref_adapter = self.bridge.teacher_motion_ref or self.bridge.motion_ref
                        if t_ref_adapter is not self.bridge.motion_ref:
                            t_slice = t_ref_adapter.get_imagination_slice(
                                m_ids, t_0, H, need_ref_cur=False
                            )
                else:
                    ref_slice = None
                    t_slice = None
                    done_taus = set()

                imagined_return = torch.zeros(B, dtype=torch.float32, device=self.device)
                bc_loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
                res_smooth_sum = torch.zeros((), dtype=torch.float32, device=self.device)
                res_jerk_sum = torch.zeros((), dtype=torch.float32, device=self.device)
                discount_weight = torch.ones(B, dtype=torch.float32, device=self.device)
                step_obj_rew_sum = torch.zeros((), dtype=torch.float32, device=self.device)
                prev_res = torch.zeros_like(prev_a)
                prev2_res = torch.zeros_like(prev_a)

                fk_tpos = fk_cur.tracked_pos_w
                fk_tmat = fk_cur.tracked_rotmat_w
                fk_tvel = fk_cur.tracked_lin_vel_w
                fk_tang = fk_cur.tracked_ang_vel_w
                fk_cpos = fk_cur.contact_pos_w
                fk_cvel = fk_cur.contact_lin_vel_w
                fk_aquat = fk_cur.anchor_quat_w
                fk_rmat = fk_cur.root_rotmat_w

                h_ang = hist_cur.base_ang_vel if hist_cur is not None else None
                h_q = hist_cur.joint_pos_rel if hist_cur is not None else None
                h_qd = hist_cur.joint_vel_rel if hist_cur is not None else None
                h_act = hist_cur.actions if hist_cur is not None else None
                h_grav = hist_cur.project_gravity if hist_cur is not None else None

                for tau in range(H):
                    t_cur = t_0 + tau
                    if ref_slice is not None:
                        ref_jpos = ref_slice.fut_joint_pos[:, tau]
                        ref_jvel = ref_slice.fut_joint_vel[:, tau]
                        ref_bpos = ref_slice.body_pos_w[:, tau]
                        ref_bquat = ref_slice.body_quat_w[:, tau]
                        ref_brot = ref_slice.body_rotmat_w[:, tau] if ref_slice.body_rotmat_w is not None else None
                        ref_bv = ref_slice.body_lin_vel_w[:, tau]
                        ref_bw = ref_slice.body_ang_vel_w[:, tau]
                        ref_opos = ref_slice.fut_obj_pos_w[:, tau]
                        ref_oquat = ref_slice.fut_obj_quat_w[:, tau]
                        ref_ov = ref_slice.fut_obj_lin_vel_w[:, tau]
                        ref_ow = ref_slice.fut_obj_ang_vel_w[:, tau]
                        ref_contact = ref_slice.contact_label[:, tau]
                        ref_o2b_pos = ref_slice.o2b_pos_ref[:, tau] if ref_slice.o2b_pos_ref is not None else None
                        ref_o2b_rot = ref_slice.o2b_rotmat_ref[:, tau] if ref_slice.o2b_rotmat_ref is not None else None
                        ref_anc_pg_z = ref_slice.anc_pg_z_ref[:, tau] if ref_slice.anc_pg_z_ref is not None else None
                        fut_jpos = ref_slice.fut_joint_pos[:, tau : tau + K]
                        fut_jvel = ref_slice.fut_joint_vel[:, tau : tau + K]
                        fut_anc_pos = ref_slice.fut_anchor_pos_w[:, tau : tau + K]
                        fut_anc_quat = ref_slice.fut_anchor_quat_w[:, tau : tau + K]
                        fut_obj_pos = ref_slice.fut_obj_pos_w[:, tau : tau + K]
                        fut_obj_quat = ref_slice.fut_obj_quat_w[:, tau : tau + K]
                        fut_obj_v = ref_slice.fut_obj_lin_vel_w[:, tau : tau + K]
                        fut_obj_w = ref_slice.fut_obj_ang_vel_w[:, tau : tau + K]
                    else:
                        rc = self.bridge.motion_ref.get_frame(m_ids, t_cur)
                        rf = self.bridge.motion_ref.get_future_window(m_ids, t_cur)
                        ref_jpos, ref_jvel = rc.joint_pos, rc.joint_vel
                        ref_bpos, ref_bquat, ref_brot = rc.body_pos_w, rc.body_quat_w, rc.body_rotmat_w
                        ref_bv, ref_bw = rc.body_lin_vel_w, rc.body_ang_vel_w
                        ref_opos, ref_oquat, ref_ov, ref_ow = rc.obj_pos_w, rc.obj_quat_w, rc.obj_lin_vel_w, rc.obj_ang_vel_w
                        ref_contact = rc.contact_label
                        ref_o2b_pos, ref_o2b_rot, ref_anc_pg_z = rc.o2b_pos_ref, rc.o2b_rotmat_ref, rc.anc_pg_z_ref
                        fut_jpos, fut_jvel = rf.joint_pos, rf.joint_vel
                        fut_anc_pos, fut_anc_quat = rf.anchor_pos_w, rf.anchor_quat_w
                        fut_obj_pos, fut_obj_quat = rf.obj_pos_w, rf.obj_quat_w
                        fut_obj_v, fut_obj_w = rf.obj_lin_vel_w, rf.obj_ang_vel_w

                    if t_slice is not None:
                        t_fut_jpos = t_slice.fut_joint_pos[:, tau : tau + K]
                        t_fut_jvel = t_slice.fut_joint_vel[:, tau : tau + K]
                        t_fut_anc_pos = t_slice.fut_anchor_pos_w[:, tau : tau + K]
                        t_fut_anc_quat = t_slice.fut_anchor_quat_w[:, tau : tau + K]
                        t_fut_obj_pos = t_slice.fut_obj_pos_w[:, tau : tau + K]
                        t_fut_obj_quat = t_slice.fut_obj_quat_w[:, tau : tau + K]
                        t_fut_obj_v = t_slice.fut_obj_lin_vel_w[:, tau : tau + K]
                        t_fut_obj_w = t_slice.fut_obj_ang_vel_w[:, tau : tau + K]
                    else:
                        t_fut_jpos = t_fut_jvel = t_fut_anc_pos = t_fut_anc_quat = None
                        t_fut_obj_pos = t_fut_obj_quat = t_fut_obj_v = t_fut_obj_w = None

                    step0_flag = self._one_scalar if tau == 0 else self._zero_scalar
                    step_ge2_flag = self._one_scalar if tau >= 2 else self._zero_scalar
                    (
                        s_next_raw,
                        s_cur,
                        fk_tpos,
                        fk_tmat,
                        fk_tvel,
                        fk_tang,
                        fk_cpos,
                        fk_cvel,
                        fk_aquat,
                        fk_rmat,
                        prev_a,
                        prev_res,
                        prev2_res,
                        feet_air_cur,
                        prev_feet_gnd,
                        r_step,
                        p_term,
                        step_bc,
                        step_res_rate,
                        step_res_jerk,
                        step_obj_rew,
                        h_ang,
                        h_q,
                        h_qd,
                        h_act,
                        h_grav,
                    ) = self._img_step_fn(
                        s_cur,
                        prev_a,
                        fk_tpos,
                        fk_tmat,
                        fk_tvel,
                        fk_tang,
                        fk_cpos,
                        fk_cvel,
                        fk_aquat,
                        fk_rmat,
                        prev_res,
                        prev2_res,
                        step0_flag,
                        step_ge2_flag,
                        feet_air_cur,
                        prev_feet_gnd,
                        t_cur,
                        xi,
                        ref_jpos,
                        ref_jvel,
                        ref_bpos,
                        ref_bquat,
                        ref_brot,
                        ref_bv,
                        ref_bw,
                        ref_opos,
                        ref_oquat,
                        ref_ov,
                        ref_ow,
                        ref_contact,
                        ref_o2b_pos,
                        ref_o2b_rot,
                        ref_anc_pg_z,
                        fut_jpos,
                        fut_jvel,
                        fut_anc_pos,
                        fut_anc_quat,
                        fut_obj_pos,
                        fut_obj_quat,
                        fut_obj_v,
                        fut_obj_w,
                        h_ang,
                        h_q,
                        h_qd,
                        h_act,
                        h_grav,
                        t_fut_jpos,
                        t_fut_jvel,
                        t_fut_anc_pos,
                        t_fut_anc_quat,
                        t_fut_obj_pos,
                        t_fut_obj_quat,
                        t_fut_obj_v,
                        t_fut_obj_w,
                    )

                    if self.teacher_policy is not None:
                        bc_loss_sum = bc_loss_sum + (discount_weight.detach() * step_bc).mean()
                    res_smooth_sum = res_smooth_sum + (discount_weight.detach() * step_res_rate).mean()
                    res_jerk_sum = res_jerk_sum + (discount_weight.detach() * step_res_jerk).mean()
                    step_obj_rew_sum = step_obj_rew_sum + step_obj_rew
                    imagined_return = imagined_return + discount_weight * r_step

                    if ref_slice is not None:
                        if tau in done_taus:
                            traj_done = ref_slice.traj_done[:, tau]
                            ref_fut_done = RefFutureWindow(
                                joint_pos=ref_slice.fut_joint_pos[:, tau + 1 : tau + 1 + K],
                                joint_vel=ref_slice.fut_joint_vel[:, tau + 1 : tau + 1 + K],
                                anchor_pos_w=ref_slice.fut_anchor_pos_w[:, tau + 1 : tau + 1 + K],
                                anchor_quat_w=ref_slice.fut_anchor_quat_w[:, tau + 1 : tau + 1 + K],
                                obj_pos_w=ref_slice.fut_obj_pos_w[:, tau + 1 : tau + 1 + K],
                                obj_quat_w=ref_slice.fut_obj_quat_w[:, tau + 1 : tau + 1 + K],
                                obj_lin_vel_w=ref_slice.fut_obj_lin_vel_w[:, tau + 1 : tau + 1 + K],
                                obj_ang_vel_w=ref_slice.fut_obj_ang_vel_w[:, tau + 1 : tau + 1 + K],
                            )
                            fk_done = self.task.run_fk(s_next_raw, with_quat=False)
                            obs_crit_done = self.task.build_privileged_obs(
                                s_next_raw, prev_a, ref_fut_done, fk_res=fk_done, xi=xi
                            )
                            v_done = self.critic(obs_crit_done, update_rms=False, pessimistic=True).clamp(v_min, v_max)
                            imagined_return = (
                                imagined_return
                                + discount_weight * self.cfg.gamma * (1.0 - p_term) * v_done * traj_done.float()
                            )
                            discount_weight = (
                                discount_weight
                                * self.cfg.gamma
                                * (1.0 - p_term)
                                * (~traj_done).float()
                            )
                        else:
                            discount_weight = discount_weight * self.cfg.gamma * (1.0 - p_term)
                    else:
                        t_next = t_cur + 1
                        traj_done = self.bridge.motion_ref.is_done(m_ids, t_next)
                        if traj_done.any():
                            ref_fut_done = self.bridge.motion_ref.get_future_window(m_ids, t_next)
                            fk_done = self.task.run_fk(s_next_raw, with_quat=False)
                            obs_crit_done = self.task.build_privileged_obs(
                                s_next_raw, prev_a, ref_fut_done, fk_res=fk_done, xi=xi
                            )
                            v_done = self.critic(obs_crit_done, update_rms=False, pessimistic=True).clamp(v_min, v_max)
                            imagined_return = (
                                imagined_return
                                + discount_weight * self.cfg.gamma * (1.0 - p_term) * v_done * traj_done.float()
                            )
                        discount_weight = (
                            discount_weight
                            * self.cfg.gamma
                            * (1.0 - p_term)
                            * (~traj_done).float()
                        )

                # Value-clipped tail bootstrap V(s_H) via frozen ensemble Critic
                anc_i = self.fk.anchor_tracked_idx
                fk_tail = FKResult(
                    tracked_pos_w=fk_tpos,
                    tracked_rotmat_w=fk_tmat,
                    tracked_quat_w=None,
                    tracked_lin_vel_w=fk_tvel,
                    tracked_ang_vel_w=fk_tang,
                    contact_pos_w=fk_cpos,
                    contact_lin_vel_w=fk_cvel,
                    anchor_pos_w=fk_tpos[:, anc_i],
                    anchor_quat_w=fk_aquat,
                    anchor_rotmat_w=fk_tmat[:, anc_i],
                    anchor_lin_vel_w=fk_tvel[:, anc_i],
                    anchor_ang_vel_w=fk_tang[:, anc_i],
                    root_rotmat_w=fk_rmat,
                )
                if ref_slice is not None:
                    ref_fut_tail = RefFutureWindow(
                        joint_pos=ref_slice.fut_joint_pos[:, H : H + K],
                        joint_vel=ref_slice.fut_joint_vel[:, H : H + K],
                        anchor_pos_w=ref_slice.fut_anchor_pos_w[:, H : H + K],
                        anchor_quat_w=ref_slice.fut_anchor_quat_w[:, H : H + K],
                        obj_pos_w=ref_slice.fut_obj_pos_w[:, H : H + K],
                        obj_quat_w=ref_slice.fut_obj_quat_w[:, H : H + K],
                        obj_lin_vel_w=ref_slice.fut_obj_lin_vel_w[:, H : H + K],
                        obj_ang_vel_w=ref_slice.fut_obj_ang_vel_w[:, H : H + K],
                    )
                else:
                    ref_fut_tail = self.bridge.motion_ref.get_future_window(m_ids, t_0 + H)
                obs_tail = self.task.build_privileged_obs(s_cur, prev_a, ref_fut_tail, fk_res=fk_tail, xi=xi)
                v_tail = self.critic(obs_tail, update_rms=False, pessimistic=True).clamp(v_min, v_max)
                imagined_return = imagined_return + discount_weight * v_tail

                bc_term = bc_loss_sum / float(H)
                res_rate_loss = self.cfg.residual_rate_weight * self.task.step_dt * res_smooth_sum
                res_jerk_loss = self.cfg.residual_jerk_weight * self.task.step_dt * res_jerk_sum
                policy_loss = -imagined_return.mean() + bc_coef * bc_term + res_rate_loss + res_jerk_loss

                self.policy_opt.zero_grad(set_to_none=True)
                policy_loss.backward()
                if self.cfg.grad_value_clip > 0.0:
                    nn.utils.clip_grad_value_(self._raw_policy.parameters(), self.cfg.grad_value_clip)
                grad_norm = nn.utils.clip_grad_norm_(self._raw_policy.parameters(), self.cfg.max_grad_norm)
                if torch.isfinite(grad_norm):
                    self.policy_opt.step()

                ret_accum = ret_accum + imagined_return.mean().detach()
                bc_accum = bc_accum + bc_term.detach()
                surv_accum = surv_accum + discount_weight.mean().detach()
                obj_rew_accum = obj_rew_accum + (step_obj_rew_sum / float(max(1, H))).detach()
                res_jerk_accum = res_jerk_accum + (res_jerk_sum / float(max(1, H))).detach()
        finally:
            _freeze_params(self._raw_world_model, False)
            _freeze_params(self._raw_critic, False)

        denom = float(max(1, self.cfg.policy_updates_per_iter))
        metrics = {
            "policy/imagined_return": float((ret_accum / denom).item()),
            "policy/tail_survival_weight": float((surv_accum / denom).item()),
            "policy/imagined_obj_rew": float((obj_rew_accum / denom).item()),
            "policy/res_jerk": float((res_jerk_accum / denom).item()),
            "policy/action_std": float(self._raw_policy.std().mean().detach().item()),
        }
        if self.teacher_policy is not None:
            metrics["policy/bc_loss"] = float((bc_accum / denom).item())
            metrics["policy/bc_coef"] = bc_coef
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing & Main Training Loop
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, iteration: int) -> None:
        raw_pol = _unwrap_compiled(self._raw_policy)
        raw_crit = _unwrap_compiled(self._raw_critic)
        raw_wm = _unwrap_compiled(self._raw_world_model)

        # Sync latest nominal actuator buffers before saving
        raw_pol.sync_actuator_buffers(self.fk.default_joint_pos, self.fk.action_scale)
        payload = {
            "format": "supermimic_sugar_v2",
            "stage": self.stage,
            "task_name": self.task.profile.task_name,
            "joint_names": list(self.fk.env_joint_names),
            "iteration": iteration,
            "next_iteration": iteration + 1,
            "obs_dim": raw_pol.obs_dim,
            "action_dim": raw_pol.action_dim,
            "hidden_dims": raw_pol.hidden_dims,
            "use_ref_residual": raw_pol.use_ref_residual,
            "legs_use_residual": raw_pol.legs_use_residual,
            "cfg": asdict(self.cfg),
            "policy_state_dict": _strip_compiled_prefix(raw_pol.state_dict()),
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
        raw_pol.load_state_dict(
            _compat_policy_state_dict(payload["policy_state_dict"], raw_pol), strict=True
        )
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
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()

    def learn(self, max_iterations: int) -> Path:
        """Run full SuperMimic training loop for `max_iterations` and save SUGAR-compatible `model_*.pt`."""
        start_iter = self.current_iter
        steps_per_iter = self.cfg.steps_per_env * self.num_envs
        print(
            f"[SuperMimic Trainer] Starting stage='{self.stage}' | task='{self.task.profile.task_name}' | "
            f"envs={self.num_envs} | horizon={self.cfg.imagination_horizon} | "
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
            lr_scale = self._apply_lr_schedule(it, max_iterations)
            self._sync()
            t0 = time.perf_counter()
            critic_obs_flat, returns_flat, c_stats = self.collect_rollout(
                iteration=it, max_iterations=max_iterations
            )
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
                "policy/lr_scale": lr_scale,
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

"""Neural network modules for SuperMimic x SUGAR:
- `RunningMeanStd`: Differentiable observation/context normalizer
- `JointWorldModel`: Autoregressive human-object dynamics + auxiliary contact predictor
- `SuperMimicPolicy`: Unified Actor for Stage 1 Refiner (890-D) & Stage 2 Tracker (510-D)
- `SuperMimicCritic`: Multi-head ensemble value network for 32-step BPTT bootstrap
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
from torch import Tensor

from .hoi_task import (
    CONTACT_DIM,
    DELTA_DIM,
    PRIVILEGED_OBS_DIM,
    SugarHOITask,
    WM_STATE_FEAT_DIM,
    XI_DIM,
)
from .urdf_kinematics import FKResult


class RunningMeanStd(nn.Module):
    """Online running mean/variance normalizer (differentiable w.r.t. input `x`)."""

    def __init__(self, shape: int | Sequence[int], eps: float = 1e-5, clip: float = 10.0) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: Tensor, mask: Tensor | None = None) -> None:
        flat = x.detach().reshape(-1, self.mean.shape[-1]).float()
        if flat.shape[0] < 2:
            return
        if mask is None:
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0, unbiased=False)
            batch_count = torch.tensor(float(flat.shape[0]), dtype=torch.float64, device=flat.device)
            active = torch.ones((), dtype=torch.bool, device=flat.device)
        else:
            w = mask.detach().reshape(-1, 1).float().clamp(0.0, 1.0)
            w_sum = w.sum()
            active = w_sum >= 2.0
            denom = w_sum.clamp_min(1.0)
            batch_mean = (flat * w).sum(dim=0) / denom
            batch_var = (((flat - batch_mean) ** 2) * w).sum(dim=0) / denom
            batch_count = w_sum.to(torch.float64)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        safe_tot = tot_count.float().clamp_min(1e-6)
        bc_f = batch_count.float()

        new_mean = self.mean + delta * (bc_f / safe_tot)
        m_a = self.var * self.count.float()
        m_b = batch_var * bc_f
        m2 = m_a + m_b + (delta * delta) * (self.count.float() * bc_f / safe_tot)
        new_var = (m2 / safe_tot).clamp_min(1e-6)

        self.mean.copy_(torch.where(active, new_mean, self.mean))
        self.var.copy_(torch.where(active, new_var, self.var))
        self.count.copy_(torch.where(active, tot_count, self.count))

    def forward(self, x: Tensor) -> Tensor:
        std = torch.sqrt(self.var + self.eps)
        return ((x - self.mean) / std).clamp(-self.clip, self.clip)


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int | None = None,
    out_gain: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        lin = nn.Linear(prev, h)
        nn.init.orthogonal_(lin.weight, gain=1.414)
        nn.init.zeros_(lin.bias)
        layers.extend([lin, nn.LayerNorm(h), nn.SiLU()])
        prev = h
    if out_dim is not None:
        head = nn.Linear(prev, out_dim)
        nn.init.orthogonal_(head.weight, gain=out_gain)
        nn.init.zeros_(head.bias)
        layers.append(head)
    return nn.Sequential(*layers)


class JointWorldModel(nn.Module):
    """Differentiable joint robot-object world model with auxiliary contact head."""

    def __init__(
        self,
        state_feat_dim: int = WM_STATE_FEAT_DIM,
        action_dim: int = 29,
        xi_dim: int = XI_DIM,
        delta_dim: int = DELTA_DIM,
        contact_dim: int = CONTACT_DIM,
        hidden_dims: Sequence[int] = (512, 512, 512),
    ) -> None:
        super().__init__()
        self.in_dim = state_feat_dim + action_dim * 2 + xi_dim
        self.feat_rms = RunningMeanStd(state_feat_dim)
        self.xi_rms = RunningMeanStd(xi_dim)
        # Output normalizer: the head predicts the delta in units of the true one-step delta std (allmimic `out_norm`)
        self.delta_rms = RunningMeanStd(delta_dim)

        self.trunk = _build_mlp(self.in_dim, hidden_dims, out_dim=None)
        last_h = hidden_dims[-1]

        self.delta_head = nn.Linear(last_h, delta_dim)
        nn.init.orthogonal_(self.delta_head.weight, gain=0.01)
        nn.init.zeros_(self.delta_head.bias)

        self.contact_head = nn.Linear(last_h, contact_dim)
        nn.init.orthogonal_(self.contact_head.weight, gain=0.1)
        nn.init.zeros_(self.contact_head.bias)
        self._core_fn = self._core_forward

    def _core_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(x)
        return self.delta_head(h), self.contact_head(h)

    def enable_compile(self, **compile_kwargs) -> None:
        """Compile inner tensor MLP + dual heads without wrapping outer `nn.Module` state_dict keys."""
        if hasattr(torch, "compile"):
            self._core_fn = torch.compile(self._core_forward, **compile_kwargs)

    def delta_std(self, floor: float = 1e-4) -> Tensor:
        """Per-dim std of the true one-step delta (also the WM loss normalizer)."""
        return self.delta_rms.var.sqrt().clamp_min(floor)

    @torch.no_grad()
    def update_delta_stats(self, true_delta: Tensor, mask: Tensor | None = None) -> None:
        self.delta_rms.update(true_delta, mask=mask)

    def forward_delta(
        self,
        state_feat: Tensor,
        raw_action: Tensor,
        pd_error: Tensor,
        xi: Tensor,
        update_rms: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if update_rms:
            self.feat_rms.update(state_feat)
            self.xi_rms.update(xi)
        f_norm = self.feat_rms(state_feat)
        xi_norm = self.xi_rms(xi)
        x = torch.cat([f_norm, raw_action.clamp(-10.0, 10.0), pd_error.clamp(-5.0, 5.0), xi_norm], dim=-1)
        delta_n, contact_logits = self._core_fn(x)
        return delta_n * self.delta_std() + self.delta_rms.mean, contact_logits

    def step(
        self,
        task: SugarHOITask,
        s_flat: Tensor,
        raw_action: Tensor,
        xi: Tensor,
        fk_res: FKResult | None = None,
        update_rms: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Predict one 50Hz control step `(s_t, a_t, xi) -> (s_{t+1}, contact_logits)`."""
        state_feat = task.state_features(s_flat, fk_res=fk_res)
        q_cur = s_flat[..., 7:36]
        q_sp = task.raw_action_to_setpoint(raw_action, xi=xi)
        pd_err = q_sp - q_cur
        delta, contact_logits = self.forward_delta(
            state_feat, raw_action, pd_err, xi, update_rms=update_rms
        )
        s_next = task.apply_delta(s_flat, delta)
        return s_next, contact_logits


class SuperMimicPolicy(nn.Module):
    """Unified Actor network supporting both Stage 1 Refiner (890-D) and Stage 2 Tracker (510-D).

    Self-contained for standalone deployment in `play.py` / `Inference`:
    stores `obs_rms`, `default_joint_pos`, and `action_scale` inside its `state_dict`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        hidden_dims: Sequence[int] = (512, 256, 128),
        default_joint_pos: Tensor | None = None,
        action_scale: Tensor | None = None,
        use_ref_residual: bool = True,
        init_std: float = 0.20,
        min_std: float = 0.02,
        max_std: float = 1.0,
        action_limit: float = 8.0,
        residual_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)
        self.use_ref_residual = use_ref_residual
        self.min_std = min_std
        self.max_std = max_std
        self.action_limit = action_limit

        self.obs_rms = RunningMeanStd(obs_dim)
        self.backbone = _build_mlp(obs_dim, hidden_dims, out_dim=action_dim, out_gain=0.01)

        init_log_std = float(torch.tensor(init_std).log().item())
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std, dtype=torch.float32))

        if default_joint_pos is None:
            default_joint_pos = torch.zeros(action_dim, dtype=torch.float32)
        if action_scale is None:
            action_scale = torch.ones(action_dim, dtype=torch.float32)
        self.register_buffer("default_joint_pos", default_joint_pos.clone().float())
        self.register_buffer("action_scale", action_scale.clone().float())
        # 1 = residual on the reference joint target, 0 = absolute action around the default pose
        # (allmimic #20: legs absolute, upper body residual)
        if residual_mask is None:
            residual_mask = torch.ones(action_dim, dtype=torch.float32)
        self.register_buffer("residual_mask", residual_mask.clone().float())
        self._core_fn = self._core_forward

    def _core_forward(self, obs_n: Tensor) -> Tensor:
        residual = self.backbone(obs_n)
        return self.action_limit * torch.tanh(residual / self.action_limit)

    def enable_compile(self, **compile_kwargs) -> None:
        """Compile inner actor MLP without polluting outer `nn.Module` state_dict keys."""
        if hasattr(torch, "compile"):
            self._core_fn = torch.compile(self._core_forward, **compile_kwargs)

    def sync_actuator_buffers(self, default_joint_pos: Tensor, action_scale: Tensor) -> None:
        self.default_joint_pos.copy_(default_joint_pos.to(self.default_joint_pos.device))
        self.action_scale.copy_(action_scale.to(self.action_scale.device))

    def std(self) -> Tensor:
        return self.log_std.exp().clamp(self.min_std, self.max_std)

    def forward_mean(self, obs_raw: Tensor, update_rms: bool = False) -> Tensor:
        if update_rms:
            self.obs_rms.update(obs_raw)
        obs_n = self.obs_rms(obs_raw)
        # Smooth bound on residual to keep exploration and BPTT well-conditioned
        residual = self._core_fn(obs_n)
        if self.use_ref_residual:
            # In both PrivilegedCfg and TrackerCfg/Inference, obs_raw[:, 0:29] is current ref_joint_pos!
            q_ref = obs_raw[..., 0 : self.action_dim]
            ref_bias = (q_ref - self.default_joint_pos) / self.action_scale.clamp_min(1e-6)
            return ref_bias * self.residual_mask + residual
        return residual

    def act(
        self, obs_raw: Tensor, deterministic: bool = False, update_rms: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Return `(sampled_or_det_action, mean_action)` with reparameterized gradients."""
        mean = self.forward_mean(obs_raw, update_rms=update_rms)
        if deterministic:
            return mean, mean
        std = self.std()
        noise = torch.randn_like(mean)
        return mean + std * noise, mean

    @torch.no_grad()
    def act_inference(self, obs_raw: Tensor) -> Tensor:
        """Deterministic inference entry point compatible with RSL-RL `play.py` loop."""
        return self.forward_mean(obs_raw, update_rms=False)


class SuperMimicCritic(nn.Module):
    """Ensemble Privileged Value Network for tail bootstrap in 32-step WM-BPTT."""

    def __init__(
        self,
        obs_dim: int = PRIVILEGED_OBS_DIM,
        hidden_dims: Sequence[int] = (512, 256, 128),
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_heads = num_heads
        self.obs_rms = RunningMeanStd(obs_dim)
        self.heads = nn.ModuleList(
            [_build_mlp(obs_dim, hidden_dims, out_dim=1, out_gain=1.0) for _ in range(num_heads)]
        )
        self._core_fn = self._core_forward

    def _core_forward(self, obs_n: Tensor) -> Tensor:
        preds = [head(obs_n).squeeze(-1) for head in self.heads]
        return torch.stack(preds, dim=0)

    def enable_compile(self, **compile_kwargs) -> None:
        """Compile ensemble value heads without polluting outer `nn.Module` state_dict keys."""
        if hasattr(torch, "compile"):
            self._core_fn = torch.compile(self._core_forward, **compile_kwargs)

    def forward_all(self, obs_raw: Tensor, update_rms: bool = False) -> Tensor:
        """Return all head predictions stacked as `(num_heads, B)`."""
        if update_rms:
            self.obs_rms.update(obs_raw)
        obs_n = self.obs_rms(obs_raw)
        return self._core_fn(obs_n)

    def forward(self, obs_raw: Tensor, update_rms: bool = False, pessimistic: bool = False) -> Tensor:
        """Return scalar value `(B,)`: mean across heads (or min when `pessimistic=True`)."""
        all_v = self.forward_all(obs_raw, update_rms=update_rms)
        if pessimistic and self.num_heads > 1:
            return all_v.min(dim=0).values
        return all_v.mean(dim=0)

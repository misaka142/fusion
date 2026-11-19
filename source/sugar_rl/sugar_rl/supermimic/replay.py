"""GPU-resident ring replay buffer for SuperMimic x SUGAR human-object interaction.

Key capabilities:
1. Stores transition tuples `(s_t, a_t, prev_a_t, s_{t+1}, contacts_t, motion_id_t, time_step_t, ep_id_t, valid_t)`
   directly on GPU with zero CPU-GPU sync overhead.
2. Separates true episode identity (`ep_id >= 0`) from transition continuity (`valid_transitions = ~(done | push)`):
   - `sample_wm_sequences(k)` strictly samples unbroken physical windows within the same `ep_id`,
     gracefully shortening window length if needed and never crossing reset/push boundaries.
   - `sample_imagination_starts(need_history=True)` preserves true `ep_id` across pushes/terminations
     and pads episode-start history slots with the initial frame's state AND `prev_action`,
     matching IsaacLab's `ObservationManager` `CircularBuffer` reset behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .hoi_task import CONTACT_DIM, HISTORY_LEN, SDIM, XI_DIM


@dataclass
class WMSequenceBatch:
    """Contiguous unbroken transition windows of length `k <= seq_len` for `JointWorldModel` training."""

    s_init: Tensor              # (B, 84)
    actions: Tensor             # (B, k, 29)
    next_states: Tensor         # (B, k, 84)
    contacts: Tensor            # (B, k, 5) float in [0, 1]
    xi: Tensor                  # (B, 37)


@dataclass
class ImaginationStartBatch:
    """Sampled starting states (and optional 5-frame history) for 32-step BPTT."""

    states: Tensor              # (B, 84)
    prev_actions: Tensor        # (B, 29)
    motion_ids: Tensor          # (B,) long
    time_steps: Tensor          # (B,) long
    xi: Tensor                  # (B, 37)
    feet_air_time: Tensor       # (B, 2)
    hist_states: Tensor | None = None        # (B, 5, 84) when `need_history=True`
    hist_last_actions: Tensor | None = None  # (B, 5, 29) when `need_history=True`


class HOIReplayBuffer:
    """Ring buffer of shape `(capacity, num_envs, ...)` on GPU."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        action_dim: int = 29,
        state_dim: int = SDIM,
        contact_dim: int = CONTACT_DIM,
        xi_dim: int = XI_DIM,
        device: torch.device | str = "cuda",
    ) -> None:
        assert capacity >= 16, "Replay capacity must be at least 16 steps per env."
        self.capacity = capacity
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.contact_dim = contact_dim
        self.xi_dim = xi_dim
        self.device = torch.device(device)

        self.states = torch.zeros((capacity, num_envs, state_dim), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, num_envs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, num_envs, action_dim), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros((capacity, num_envs, action_dim), dtype=torch.float32, device=self.device)
        self.contacts = torch.zeros((capacity, num_envs, contact_dim), dtype=torch.float16, device=self.device)
        self.feet_air_times = torch.zeros((capacity, num_envs, 2), dtype=torch.float16, device=self.device)
        self.motion_ids = torch.zeros((capacity, num_envs), dtype=torch.int32, device=self.device)
        self.time_steps = torch.zeros((capacity, num_envs), dtype=torch.int32, device=self.device)
        self.ep_ids = torch.full((capacity, num_envs), -1, dtype=torch.int64, device=self.device)
        self.valid_transitions = torch.zeros((capacity, num_envs), dtype=torch.bool, device=self.device)

        self.env_xi = torch.zeros((num_envs, xi_dim), dtype=torch.float32, device=self.device)

        self.ptr: int = 0
        self.size: int = 0

    def set_env_xi(self, xi: Tensor) -> None:
        self.env_xi.copy_(xi.to(self.device))

    @torch.no_grad()
    def add(
        self,
        state: Tensor,
        action: Tensor,
        prev_action: Tensor,
        next_state: Tensor,
        contact: Tensor,
        motion_id: Tensor,
        time_step: Tensor,
        ep_id: Tensor,
        valid_transition: Tensor | None = None,
        feet_air_time: Tensor | None = None,
    ) -> None:
        idx = self.ptr
        ep_i64 = ep_id.to(torch.int64)
        if valid_transition is None:
            valid_t = ep_i64 >= 0
        else:
            valid_t = valid_transition.bool() & (ep_i64 >= 0)

        self.states[idx].copy_(state)
        self.actions[idx].copy_(action)
        self.prev_actions[idx].copy_(prev_action)
        self.next_states[idx].copy_(next_state)
        self.contacts[idx].copy_(contact.clamp(0.0, 1.0).to(torch.float16))
        if feet_air_time is not None:
            self.feet_air_times[idx].copy_(feet_air_time.clamp_min(0.0).to(torch.float16))
        else:
            self.feet_air_times[idx].zero_()
        self.motion_ids[idx].copy_(motion_id.to(torch.int32))
        self.time_steps[idx].copy_(time_step.to(torch.int32))
        self.ep_ids[idx].copy_(ep_i64)
        self.valid_transitions[idx].copy_(valid_t)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _chronological_indices(self) -> Tensor:
        """Return time indices ordered from oldest to newest of length `self.size`."""
        if self.size < self.capacity:
            return torch.arange(self.size, device=self.device, dtype=torch.long)
        return (torch.arange(self.capacity, device=self.device, dtype=torch.long) + self.ptr) % self.capacity

    @torch.no_grad()
    def sample_wm_sequences(self, batch_size: int, seq_len: int = 8) -> WMSequenceBatch | None:
        """Sample contiguous transitions without episode resets or push discontinuities.

        If no full `seq_len` unbroken window exists yet, safely shrinks window length `eff_k >= 1`
        rather than ever sampling across reset/push boundaries; returns `None` if zero valid steps exist.
        """
        if self.size < 1:
            return None

        chrono = self._chronological_indices()                  # (S,)
        ep_chrono = self.ep_ids[chrono]                         # (S, N)
        vt_chrono = self.valid_transitions[chrono]              # (S, N)

        max_k = min(seq_len, self.size)
        valid_pairs: Tensor | None = None
        eff_k = max_k

        for cand_k in range(max_k, 0, -1):
            num_starts = self.size - cand_k + 1
            ep_first = ep_chrono[:num_starts]
            valid = (ep_first >= 0) & vt_chrono[:num_starts]
            for offset in range(1, cand_k):
                valid = (
                    valid
                    & vt_chrono[offset : offset + num_starts]
                    & (ep_chrono[offset : offset + num_starts] == ep_first)
                )
            pairs = valid.nonzero(as_tuple=False)
            if pairs.shape[0] > 0:
                valid_pairs = pairs
                eff_k = cand_k
                break

        if valid_pairs is None or valid_pairs.shape[0] == 0:
            return None

        pick = torch.randint(0, valid_pairs.shape[0], (batch_size,), device=self.device)
        chosen = valid_pairs[pick]
        start_pos = chosen[:, 0]                                # (B,)
        env_ids = chosen[:, 1]                                  # (B,)

        offsets = torch.arange(eff_k, device=self.device, dtype=torch.long).unsqueeze(0)  # (1, eff_k)
        ring_idx = chrono[start_pos.unsqueeze(1) + offsets]     # (B, eff_k)
        env_exp = env_ids.unsqueeze(1).expand(-1, eff_k)        # (B, eff_k)

        s_init = self.states[ring_idx[:, 0], env_ids]
        actions = self.actions[ring_idx, env_exp]
        next_states = self.next_states[ring_idx, env_exp]
        contacts = self.contacts[ring_idx, env_exp].float()
        xi = self.env_xi[env_ids]

        return WMSequenceBatch(
            s_init=s_init,
            actions=actions,
            next_states=next_states,
            contacts=contacts,
            xi=xi,
        )

    @torch.no_grad()
    def sample_imagination_starts(
        self, batch_size: int, need_history: bool = False
    ) -> ImaginationStartBatch:
        """Sample valid starting states (plus 5-frame history for Tracker) for 32-step imagination."""
        if self.size < 1:
            raise RuntimeError("Replay buffer is empty.")

        chrono = self._chronological_indices()                  # (S,)
        S = chrono.shape[0]
        pos_idx = torch.randint(0, S, (batch_size,), device=self.device)
        env_ids = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

        ring_t = chrono[pos_idx]
        states = self.states[ring_t, env_ids]
        prev_actions = self.prev_actions[ring_t, env_ids]
        motion_ids = self.motion_ids[ring_t, env_ids].long()
        time_steps = self.time_steps[ring_t, env_ids].long()
        feet_air = self.feet_air_times[ring_t, env_ids].float()
        xi = self.env_xi[env_ids]

        if not need_history:
            return ImaginationStartBatch(
                states=states,
                prev_actions=prev_actions,
                motion_ids=motion_ids,
                time_steps=time_steps,
                xi=xi,
                feet_air_time=feet_air,
            )

        # Reconstruct 5-step history [t-4, t-3, t-2, t-1, t] within the same episode (`ep_id >= 0`),
        # latching BOTH earliest state and earliest prev_action to match IsaacLab's CircularBuffer reset fill.
        target_ep = self.ep_ids[ring_t, env_ids]                # (B,)
        hist_states = torch.empty((batch_size, HISTORY_LEN, self.state_dim), dtype=torch.float32, device=self.device)
        hist_actions = torch.empty((batch_size, HISTORY_LEN, self.action_dim), dtype=torch.float32, device=self.device)

        earliest_state = states.clone()
        earliest_action = prev_actions.clone()

        for step_back in range(HISTORY_LEN):
            slot = HISTORY_LEN - 1 - step_back
            if step_back == 0:
                hist_states[:, slot] = earliest_state
                hist_actions[:, slot] = earliest_action
                continue

            cand_pos = (pos_idx - step_back).clamp_min(0)
            cand_ring = chrono[cand_pos]
            same_ep = (
                (pos_idx >= step_back)
                & (target_ep >= 0)
                & (self.ep_ids[cand_ring, env_ids] == target_ep)
            )

            s_cand = self.states[cand_ring, env_ids]
            a_cand = self.prev_actions[cand_ring, env_ids]

            earliest_state = torch.where(same_ep.unsqueeze(-1), s_cand, earliest_state)
            earliest_action = torch.where(same_ep.unsqueeze(-1), a_cand, earliest_action)

            hist_states[:, slot] = earliest_state
            hist_actions[:, slot] = earliest_action

        return ImaginationStartBatch(
            states=states,
            prev_actions=prev_actions,
            motion_ids=motion_ids,
            time_steps=time_steps,
            xi=xi,
            feet_air_time=feet_air,
            hist_states=hist_states,
            hist_last_actions=hist_actions,
        )

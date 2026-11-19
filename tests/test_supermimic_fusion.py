"""Comprehensive GPU/CPU unit & integration test suite for SuperMimic x SUGAR (`fusion`).

Run standalone without Isaac Sim (requires only `torch` and `numpy`):
    python tests/test_supermimic_fusion.py
"""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile

import torch
import torch.nn as nn

# Add `source/sugar_rl` to sys.path so `sugar_rl.supermimic` imports cleanly without editable install
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source" / "sugar_rl"))
sys.path.insert(0, str(REPO_ROOT))

from sugar_rl.supermimic import (  # noqa: E402
    CONTACT_DIM,
    FUTURE_FRAMES,
    FastTreeKinematics,
    G1_29DOF_SDK_JOINT_NAMES,
    HOIReplayBuffer,
    JointWorldModel,
    PRIVILEGED_OBS_DIM,
    SDIM,
    SugarHOITask,
    SuperMimicPolicy,
    SuperMimicTrainCfg,
    SuperMimicTrainer,
    TASK_PROFILES,
    TRACKER_OBS_DIM,
    XI_DIM,
    assert_privileged_obs_aligned,
    load_supermimic_policy,
    resolve_task_profile,
)
from sugar_rl.supermimic.env_bridge import MotionReferenceAdapter  # noqa: E402
from sugar_rl.supermimic.rotations import (  # noqa: E402
    axis_angle_to_quat,
    quat_apply,
    quat_conj,
    quat_error_magnitude,
    quat_mul,
    quat_normalize,
    quat_to_axis_angle,
    quat_to_rotmat,
    rotmat_to_quat,
    subtract_frame_transforms,
    sugar_rot6d_from_mat,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_1_rotations_and_zero_angle_gradients() -> None:
    torch.manual_seed(0)
    q = quat_normalize(torch.randn(64, 4, device=DEVICE))
    R = quat_to_rotmat(q)
    q_rec = rotmat_to_quat(R)
    err = quat_error_magnitude(q, q_rec)
    assert err.max().item() < 1e-5, f"Quat <-> Rotmat roundtrip error too large: {err.max().item()}"

    # Axis-angle roundtrip including exact zero vector
    aa = torch.randn(32, 3, device=DEVICE) * 0.5
    aa[0].zero_()
    q_aa = axis_angle_to_quat(aa)
    aa_rec = quat_to_axis_angle(q_aa)
    assert (aa - aa_rec).abs().max().item() < 1e-5

    # Zero-error gradient finiteness check (critical for BPTT when tracking is accurate!)
    q_var = q.clone().detach().requires_grad_(True)
    loss_zero = (quat_error_magnitude(q_var, q.detach()) ** 2).sum()
    loss_zero.backward()
    assert q_var.grad is not None and torch.isfinite(q_var.grad).all(), "NaN gradient at zero rotation error!"

    # Verify SUGAR row-major 2-column 6D layout: [R00, R01, R10, R11, R20, R21]
    r6 = sugar_rot6d_from_mat(R[:1])
    expected = torch.tensor(
        [[R[0, 0, 0], R[0, 0, 1], R[0, 1, 0], R[0, 1, 1], R[0, 2, 0], R[0, 2, 1]]],
        device=DEVICE,
    )
    assert (r6 - expected).abs().max().item() < 1e-6

    # Verify subtract_frame_transforms
    t1 = torch.randn(16, 3, device=DEVICE)
    t2 = torch.randn(16, 3, device=DEVICE)
    q1 = quat_normalize(torch.randn(16, 4, device=DEVICE))
    q2 = quat_normalize(torch.randn(16, 4, device=DEVICE))
    p12, q12 = subtract_frame_transforms(t1, q1, t2, q2)
    t2_rec = t1 + quat_apply(q1, p12)
    q2_rec = quat_mul(q1, q12)
    assert (t2 - t2_rec).abs().max().item() < 1e-5
    assert quat_error_magnitude(q2, q2_rec).max().item() < 1e-5
    print("[PASS] Test 1: Differentiable rotations & zero-angle gradient safety")


def test_2_kinematics_urdf_mjcf_and_joint_permutation() -> None:
    torch.manual_seed(1)
    fk_sdk = FastTreeKinematics(env_joint_names=G1_29DOF_SDK_JOINT_NAMES, device=DEVICE)

    # Permute joint names arbitrarily (simulating PhysX BFS vs URDF DFS joint order)
    perm = torch.randperm(29)
    inv_perm = torch.argsort(perm)
    perm_names = [G1_29DOF_SDK_JOINT_NAMES[i] for i in perm.tolist()]
    fk_perm = FastTreeKinematics(env_joint_names=perm_names, device=DEVICE)

    B = 16
    root_pos = torch.randn(B, 3, device=DEVICE)
    root_quat = quat_normalize(torch.randn(B, 4, device=DEVICE))
    q_sdk = torch.randn(B, 29, device=DEVICE) * 0.3
    qd_sdk = torch.randn(B, 29, device=DEVICE) * 0.5
    root_v = torch.randn(B, 3, device=DEVICE)
    root_w = torch.randn(B, 3, device=DEVICE)

    q_perm = q_sdk[:, perm]
    qd_perm = qd_sdk[:, perm]

    res_a = fk_sdk(root_pos, root_quat, q_sdk, root_v, root_w, qd_sdk)
    res_b = fk_perm(root_pos, root_quat, q_perm, root_v, root_w, qd_perm)

    assert (res_a.tracked_pos_w - res_b.tracked_pos_w).abs().max().item() < 1e-6
    assert (res_a.tracked_lin_vel_w - res_b.tracked_lin_vel_w).abs().max().item() < 1e-6
    assert (res_a.tracked_ang_vel_w - res_b.tracked_ang_vel_w).abs().max().item() < 1e-6
    assert (res_a.contact_pos_w - res_b.contact_pos_w).abs().max().item() < 1e-6
    assert (fk_perm.action_scale[inv_perm] - fk_sdk.action_scale).abs().max().item() < 1e-6

    # Also test URDF XML parser via temporary standard URDF generated from G1 tree
    with tempfile.TemporaryDirectory() as tmpdir:
        urdf_file = pathlib.Path(tmpdir) / "g1_test.urdf"
        lines = ['<robot name="g1_29dof">']
        from sugar_rl.supermimic.urdf_kinematics import _BUILTIN_G1_29DOF_TREE

        for row in _BUILTIN_G1_29DOF_TREE:
            lines.append(f'  <link name="{row[1]}"/>')
        for row in _BUILTIN_G1_29DOF_TREE[1:]:
            parent_lk, child_lk, j_name, pos, quat, axis, lo, hi = row
            # Convert quat back to rpy for URDF <origin>
            R_m = quat_to_rotmat(torch.tensor(quat, dtype=torch.float64))
            pitch = math.asin(max(-1.0, min(1.0, -float(R_m[2, 0]))))
            roll = math.atan2(float(R_m[2, 1]), float(R_m[2, 2]))
            yaw = math.atan2(float(R_m[1, 0]), float(R_m[0, 0]))
            xyz_s = f"{pos[0]} {pos[1]} {pos[2]}"
            rpy_s = f"{roll} {pitch} {yaw}"
            if j_name is None:
                lines.append(f'  <joint name="fix_{child_lk}" type="fixed">')
                lines.append(f'    <parent link="{parent_lk}"/><child link="{child_lk}"/>')
                lines.append(f'    <origin xyz="{xyz_s}" rpy="{rpy_s}"/>')
                lines.append("  </joint>")
            else:
                ax_s = f"{axis[0]} {axis[1]} {axis[2]}"
                lines.append(f'  <joint name="{j_name}" type="revolute">')
                lines.append(f'    <parent link="{parent_lk}"/><child link="{child_lk}"/>')
                lines.append(f'    <origin xyz="{xyz_s}" rpy="{rpy_s}"/>')
                lines.append(f'    <axis xyz="{ax_s}"/>')
                lines.append(f'    <limit lower="{lo}" upper="{hi}" effort="88" velocity="32"/>')
                lines.append("  </joint>")
        lines.append("</robot>")
        urdf_file.write_text("\n".join(lines), encoding="utf-8")

        fk_urdf = FastTreeKinematics(urdf_or_xml_path=urdf_file, device=DEVICE)
        res_u = fk_urdf(root_pos, root_quat, q_sdk, root_v, root_w, qd_sdk)
        assert (res_a.tracked_pos_w - res_u.tracked_pos_w).abs().max().item() < 1e-4

    # Optional check against real MuJoCo MJCF file if present locally
    local_mjcf = REPO_ROOT.parent.parent / "papers" / "gmr" / "code-readonly" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    if local_mjcf.is_file():
        fk_mjcf = FastTreeKinematics(urdf_or_xml_path=local_mjcf, device=DEVICE)
        res_m = fk_mjcf(root_pos, root_quat, q_sdk, root_v, root_w, qd_sdk)
        assert (res_a.tracked_pos_w - res_m.tracked_pos_w).abs().max().item() < 1e-4

    # Check joint weights properties
    jw = fk_sdk.compute_joint_weights()
    assert abs(float(jw.mean().item()) - 1.0) < 1e-5
    hip_idx = G1_29DOF_SDK_JOINT_NAMES.index("left_hip_pitch_joint")
    wrist_idx = G1_29DOF_SDK_JOINT_NAMES.index("left_wrist_yaw_joint")
    assert jw[hip_idx] > jw[wrist_idx], "Hip joint weight must exceed distal wrist joint weight!"
    print("[PASS] Test 2: MuJoCo-free URDF FK, joint permutation & error-propagation weights")


class _MockMotionData:
    """Synthetic reference motion tensor container matching SUGAR's `MotionLoader` attributes."""

    def __init__(self, fk: FastTreeKinematics, num_motion: int = 4, T: int = 80, device: torch.device = DEVICE):
        self.num_motion = num_motion
        self.time_step_total_permotion = torch.full((num_motion,), T, dtype=torch.long, device=device)

        t_grid = torch.linspace(0.0, 1.5, T, device=device).view(1, T, 1)
        q_base = fk.default_joint_pos.view(1, 1, 29).expand(num_motion, T, 29)
        self.joint_pos = q_base + 0.1 * torch.sin(2.0 * math.pi * t_grid)
        self.joint_vel = 0.2 * math.pi * torch.cos(2.0 * math.pi * t_grid).expand(num_motion, T, 29)

        root_p = torch.zeros(num_motion, T, 3, device=device)
        root_p[..., 0] = 0.2 * t_grid.squeeze(-1)
        root_p[..., 2] = 0.76
        root_q = torch.zeros(num_motion, T, 4, device=device)
        root_q[..., 0] = 1.0
        root_v = torch.zeros(num_motion, T, 3, device=device)
        root_v[..., 0] = 0.13
        root_w = torch.zeros(num_motion, T, 3, device=device)

        fk_res = fk(
            root_p.reshape(-1, 3),
            root_q.reshape(-1, 4),
            self.joint_pos.reshape(-1, 29),
            root_v.reshape(-1, 3),
            root_w.reshape(-1, 3),
            self.joint_vel.reshape(-1, 29),
        )
        self.body_pos_w = fk_res.tracked_pos_w.reshape(num_motion, T, 14, 3)
        self.body_quat_w = fk_res.tracked_quat_w.reshape(num_motion, T, 14, 4)
        self.body_lin_vel_w = fk_res.tracked_lin_vel_w.reshape(num_motion, T, 14, 3)
        self.body_ang_vel_w = fk_res.tracked_ang_vel_w.reshape(num_motion, T, 14, 3)

        # Place box right between the robot's hands
        self.obj_pos = fk_res.anchor_pos_w.reshape(num_motion, T, 3).clone()
        self.obj_pos[..., 0] += 0.30
        self.obj_quat = root_q.clone()
        self.obj_lin_vel = root_v.clone()
        self.obj_ang_vel = root_w.clone()
        self.contact_label = torch.ones(num_motion, T, dtype=torch.bool, device=device)


def test_3_observations_and_all_6_task_rewards() -> None:
    torch.manual_seed(2)
    fk = FastTreeKinematics(device=DEVICE)
    mock_m = _MockMotionData(fk, device=DEVICE)
    ref_adapter = MotionReferenceAdapter(mock_m)

    B = 12
    m_ids = torch.arange(B, device=DEVICE) % mock_m.num_motion
    t_steps = torch.full((B,), 5, dtype=torch.long, device=DEVICE)
    ref_frame = ref_adapter.get_frame(m_ids, t_steps)
    ref_fut = ref_adapter.get_future_window(m_ids, t_steps)

    # Construct exact reference state s_perfect and perturbed state s_bad
    s_perfect = SugarHOITask.pack_state(
        root_pos=ref_frame.body_pos_w[:, 0],
        root_quat=ref_frame.body_quat_w[:, 0],
        q=ref_frame.joint_pos,
        root_lin_vel=ref_frame.body_lin_vel_w[:, 0],
        root_ang_vel=ref_frame.body_ang_vel_w[:, 0],
        qd=ref_frame.joint_vel,
        obj_pos=ref_frame.obj_pos_w,
        obj_quat=ref_frame.obj_quat_w,
        obj_lin_vel=ref_frame.obj_lin_vel_w,
        obj_ang_vel=ref_frame.obj_ang_vel_w,
    )
    s_bad = s_perfect.clone()
    s_bad[:, 0] += 1.0    # 1m robot anchor error -> triggers soft termination
    s_bad[:, 71] += 1.0   # 1m object pos error -> triggers soft termination

    last_act = torch.zeros(B, 29, device=DEVICE)

    for task_name in TASK_PROFILES:
        task = SugarHOITask(fk, task_name_or_id=task_name)
        obs_priv = task.build_privileged_obs(s_perfect, last_act, ref_fut)
        assert obs_priv.shape == (B, PRIVILEGED_OBS_DIM)
        assert torch.isfinite(obs_priv).all()

        hist = task.init_tracker_history_from_states(
            s_perfect.unsqueeze(1).expand(-1, 5, -1),
            last_act.unsqueeze(1).expand(-1, 5, -1),
        )
        hist = task.advance_tracker_history(hist, s_perfect, last_act)
        obs_trk = task.build_tracker_obs(s_perfect, ref_frame.tracker_command_36(), hist, add_noise=True)
        assert obs_trk.shape == (B, TRACKER_OBS_DIM)
        assert torch.isfinite(obs_trk).all()

        r_good, p_term_good, _ = task.compute_reward_and_termination(
            s_perfect, s_perfect, last_act, last_act, ref_frame
        )
        r_bad, p_term_bad, _ = task.compute_reward_and_termination(
            s_perfect, s_bad, last_act, last_act, ref_frame
        )
        assert (r_good > r_bad).all(), f"{task_name}: perfect tracking must yield higher reward than bad state!"
        assert p_term_good.mean().item() < 0.05
        assert p_term_bad.mean().item() > 0.99

    print("[PASS] Test 3: 890-D Refiner & 510-D Tracker observations + all 6 SUGAR task rewards/terminations")


def test_4_world_model_se2_invariance() -> None:
    torch.manual_seed(3)
    fk = FastTreeKinematics(device=DEVICE)
    task = SugarHOITask(fk, "CarryBox")

    B = 16
    s0 = SugarHOITask.pack_state(
        root_pos=torch.randn(B, 3, device=DEVICE),
        root_quat=quat_normalize(torch.randn(B, 4, device=DEVICE)),
        q=torch.randn(B, 29, device=DEVICE) * 0.2,
        root_lin_vel=torch.randn(B, 3, device=DEVICE),
        root_ang_vel=torch.randn(B, 3, device=DEVICE),
        qd=torch.randn(B, 29, device=DEVICE) * 0.2,
        obj_pos=torch.randn(B, 3, device=DEVICE),
        obj_quat=quat_normalize(torch.randn(B, 4, device=DEVICE)),
        obj_lin_vel=torch.randn(B, 3, device=DEVICE),
        obj_ang_vel=torch.randn(B, 3, device=DEVICE),
    )
    feat_0 = task.state_features(s0)

    # Apply random global horizontal translation (dx, dy, 0) and global yaw rotation d_psi
    psi = torch.randn(B, device=DEVICE) * math.pi
    q_yaw = torch.stack(
        [torch.cos(0.5 * psi), torch.zeros_like(psi), torch.zeros_like(psi), torch.sin(0.5 * psi)], dim=-1
    )
    shift_xy = torch.cat([torch.randn(B, 2, device=DEVICE) * 5.0, torch.zeros(B, 1, device=DEVICE)], dim=-1)

    rp, rq, q, rv, rw, qd, op, oq, ov, ow = SugarHOITask.unpack_state(s0)
    s_transformed = SugarHOITask.pack_state(
        root_pos=quat_apply(q_yaw, rp) + shift_xy,
        root_quat=quat_mul(q_yaw, rq),
        q=q,
        root_lin_vel=quat_apply(q_yaw, rv),
        root_ang_vel=quat_apply(q_yaw, rw),
        qd=qd,
        obj_pos=quat_apply(q_yaw, op) + shift_xy,
        obj_quat=quat_mul(q_yaw, oq),
        obj_lin_vel=quat_apply(q_yaw, ov),
        obj_ang_vel=quat_apply(q_yaw, ow),
    )
    feat_t = task.state_features(s_transformed)
    max_diff = (feat_0 - feat_t).abs().max().item()
    assert max_diff < 1e-4, f"SE(2) invariance violated: max_diff={max_diff}"
    print(f"[PASS] Test 4: JointWorldModel SE(2) translation/yaw invariance (max_diff={max_diff:.2e})")


def test_5_replay_buffer_push_isolation_and_history_padding() -> None:
    torch.manual_seed(4)
    buf = HOIReplayBuffer(capacity=32, num_envs=4, device=DEVICE)
    s_clean = torch.ones(4, SDIM, device=DEVICE)
    s_corrupt = torch.full((4, SDIM), 9999.0, device=DEVICE)
    act = torch.ones(4, 29, device=DEVICE)
    c = torch.zeros(4, CONTACT_DIM, device=DEVICE)
    m_id = torch.zeros(4, dtype=torch.long, device=DEVICE)

    ep = torch.arange(4, dtype=torch.int64, device=DEVICE)
    for t in range(20):
        if t == 9:
            # Simulate an external velocity push at t=9: keep true `ep` for 5-frame history, mark transition invalid
            buf.add(
                s_clean * float(t + 1),
                act,
                act * float(t + 1),
                s_corrupt,
                c,
                m_id,
                m_id + t,
                ep,
                valid_transition=torch.zeros(4, dtype=torch.bool, device=DEVICE),
            )
        else:
            buf.add(s_clean * float(t + 1), act, act * float(t + 1), s_clean * float(t + 2), c, m_id, m_id + t, ep)

    wm_batch = buf.sample_wm_sequences(batch_size=256, seq_len=8)
    assert wm_batch is not None
    assert (wm_batch.next_states < 1000.0).all(), "Broken/pushed step leaked into WM training sequence!"

    starts = buf.sample_imagination_starts(batch_size=128, need_history=True)
    assert starts.hist_states.shape == (128, 5, SDIM)
    assert starts.hist_last_actions.shape == (128, 5, 29)
    print("[PASS] Test 5: HOIReplayBuffer push/reset boundary isolation & 5-frame history padding")


def test_6_32_step_hoi_bptt_gradients() -> None:
    torch.manual_seed(5)
    fk = FastTreeKinematics(device=DEVICE)
    task = SugarHOITask(fk, "CarryBox")
    mock_m = _MockMotionData(fk, device=DEVICE)
    ref_adapter = MotionReferenceAdapter(mock_m)

    wm = JointWorldModel().to(DEVICE)
    # Give WM non-trivial coupling from actions to object motion (as learned after contact)
    with torch.no_grad():
        nn.init.normal_(wm.delta_head.weight, std=0.05)

    policy = SuperMimicPolicy(
        obs_dim=PRIVILEGED_OBS_DIM,
        default_joint_pos=fk.default_joint_pos,
        action_scale=fk.action_scale,
    ).to(DEVICE)

    B = 8
    m_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
    t_0 = torch.zeros(B, dtype=torch.long, device=DEVICE)
    ref0 = ref_adapter.get_frame(m_ids, t_0)
    s_cur = SugarHOITask.pack_state(
        ref0.body_pos_w[:, 0],
        ref0.body_quat_w[:, 0],
        ref0.joint_pos,
        ref0.body_lin_vel_w[:, 0],
        ref0.body_ang_vel_w[:, 0],
        ref0.joint_vel,
        ref0.obj_pos_w,
        ref0.obj_quat_w,
        ref0.obj_lin_vel_w,
        ref0.obj_ang_vel_w,
    )
    prev_a = torch.zeros(B, 29, device=DEVICE)
    xi = torch.zeros(B, XI_DIM, device=DEVICE)

    # Differentiate purely the OBJECT tracking reward over H=32 steps w.r.t. Policy weights!
    obj_return = torch.zeros(B, device=DEVICE)
    for tau in range(32):
        t_cur = t_0 + tau
        fk_cur = task.run_fk(s_cur)
        ref_cur = ref_adapter.get_frame(m_ids, t_cur)
        obs = task.build_privileged_obs(s_cur, prev_a, ref_adapter.get_future_window(m_ids, t_cur), fk_res=fk_cur)
        a_cur, _ = policy.act(obs, deterministic=False)
        s_next, c_logits = wm.step(task, s_cur, a_cur, xi, fk_res=fk_cur)
        obj_pos_err = ((s_next[:, 71:74] - ref_cur.obj_pos_w) ** 2).sum(dim=-1)
        r_obj_only = torch.exp(-obj_pos_err / 0.09) + 0.1 * torch.sigmoid(c_logits[:, 0])
        obj_return = obj_return + (0.99**tau) * r_obj_only
        s_cur = s_next
        prev_a = a_cur

    loss = -obj_return.mean()
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in policy.parameters() if p.grad is not None]
    total_gnorm = sum(grad_norms)
    assert total_gnorm > 0.0 and math.isfinite(total_gnorm), f"Invalid object-to-policy BPTT grad norm: {total_gnorm}"
    print(f"[PASS] Test 6: 32-step human-object BPTT gradients verified (||grad_theta J_obj|| = {total_gnorm:.4e})")


class _SyntheticSugarBridge:
    """Lightweight GPU physics bridge mimicking `SugarEnvBridge` for full trainer integration testing."""

    def __init__(self, stage: str, num_envs: int = 16, device: torch.device = DEVICE):
        self.device = device
        self.num_envs = num_envs
        self.fk = FastTreeKinematics(device=device)
        self.task = SugarHOITask(self.fk, "CarryBox")
        self.mock_m = _MockMotionData(self.fk, device=device)
        self.motion_ref = MotionReferenceAdapter(self.mock_m)
        self.teacher_motion_ref = self.motion_ref
        self.stage = stage

        self.m_ids = torch.arange(num_envs, device=device) % self.mock_m.num_motion
        self.t_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_act = torch.zeros(num_envs, 29, device=device)

        ref0 = self.motion_ref.get_frame(self.m_ids, self.t_steps)
        self.state = SugarHOITask.pack_state(
            ref0.body_pos_w[:, 0],
            ref0.body_quat_w[:, 0],
            ref0.joint_pos,
            ref0.body_lin_vel_w[:, 0],
            ref0.body_ang_vel_w[:, 0],
            ref0.joint_vel,
            ref0.obj_pos_w,
            ref0.obj_quat_w,
            ref0.obj_lin_vel_w,
            ref0.obj_ang_vel_w,
        )
        self.hist = self.task.init_tracker_history_from_states(
            self.state.unsqueeze(1).expand(-1, 5, -1),
            self.last_act.unsqueeze(1).expand(-1, 5, -1),
        )
        self.current_obs_dict = self._make_obs()

    def _make_obs(self) -> dict[str, torch.Tensor]:
        fut = self.motion_ref.get_future_window(self.m_ids, self.t_steps)
        priv = self.task.build_privileged_obs(self.state, self.last_act, fut)
        if self.stage == "refiner":
            return {"policy": priv, "critic": priv}
        cmd = self.motion_ref.get_frame(self.m_ids, self.t_steps).tracker_command_36()
        trk = self.task.build_tracker_obs(self.state, cmd, self.hist)
        return {"policy": trk, "critic": priv, "teacher": priv}

    def extract_xi(self) -> torch.Tensor:
        xi = torch.zeros(self.num_envs, XI_DIM, device=self.device)
        xi[:, :5] = 1.0
        return xi

    def get_joint_state(self) -> torch.Tensor:
        return self.state.clone()

    def get_motion_cursor(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.m_ids.clone(), self.t_steps.clone()

    def get_last_action(self) -> torch.Tensor:
        return self.last_act.clone()

    def get_ground_truth_contacts(self) -> torch.Tensor:
        c = torch.zeros(self.num_envs, CONTACT_DIM, device=self.device)
        c[:, 0] = 1.0
        c[:, 2:4] = 1.0
        return c

    def get_feet_air_time(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 2, device=self.device)

    def step(self, raw_action: torch.Tensor):
        # Simple damped spring towards PD target + coupled object motion
        q_sp = self.task.raw_action_to_setpoint(raw_action)
        rp, rq, q, rv, rw, qd, op, oq, ov, ow = SugarHOITask.unpack_state(self.state)
        qd_next = 0.8 * qd + 5.0 * (q_sp - q) * 0.02
        q_next = q + qd_next * 0.02
        rp_next = rp + rv * 0.02
        op_next = op + ov * 0.02 + 0.002 * raw_action[:, :3]

        s_next = SugarHOITask.pack_state(rp_next, rq, q_next, rv, rw, qd_next, op_next, oq, ov, ow)
        ref_cur = self.motion_ref.get_frame(self.m_ids, self.t_steps)
        rew, p_term, _ = self.task.compute_reward_and_termination(
            self.state, s_next, raw_action, self.last_act, ref_cur=ref_cur
        )

        self.t_steps = self.t_steps + 1
        timeouts = self.t_steps >= 60
        dones = timeouts | (p_term > 0.95)
        pushed = self.t_steps == 25

        if dones.any():
            self.t_steps[dones] = 0
            r_init = self.motion_ref.get_frame(self.m_ids[dones], self.t_steps[dones])
            s_next[dones] = SugarHOITask.pack_state(
                r_init.body_pos_w[:, 0],
                r_init.body_quat_w[:, 0],
                r_init.joint_pos,
                r_init.body_lin_vel_w[:, 0],
                r_init.body_ang_vel_w[:, 0],
                r_init.joint_vel,
                r_init.obj_pos_w,
                r_init.obj_quat_w,
                r_init.obj_lin_vel_w,
                r_init.obj_ang_vel_w,
            )

        self.state = s_next
        self.last_act = raw_action.clone()
        self.hist = self.task.advance_tracker_history(self.hist, self.state, self.last_act)
        self.current_obs_dict = self._make_obs()
        return self.current_obs_dict, rew, dones, timeouts, pushed, {"time_outs": timeouts}


def test_7_end_to_end_refiner_and_tracker_pipeline() -> None:
    torch.manual_seed(6)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)

        # 1. Stage 1: Train SuperMimic Refiner (890-D)
        refiner_bridge = _SyntheticSugarBridge(stage="refiner", num_envs=16, device=DEVICE)
        cfg = SuperMimicTrainCfg(
            steps_per_env=12,
            replay_capacity=48,
            wm_seq_len=6,
            wm_updates_per_iter=3,
            wm_batch_size=64,
            wm_warmup_iters=1,
            imagination_horizon=16,
            policy_updates_per_iter=2,
            policy_batch_size=16,
            save_interval=2,
            enable_tensorboard=False,
        )
        refiner_trainer = SuperMimicTrainer(refiner_bridge, log_dir=tmp / "refiner", cfg=cfg)
        assert refiner_trainer.stage == "refiner"
        refiner_ckpt = refiner_trainer.learn(max_iterations=3)
        assert refiner_ckpt.is_file()

        # 2. Stage 2: Train SuperMimic Tracker (510-D) with frozen Stage 1 Refiner Teacher!
        tracker_bridge = _SyntheticSugarBridge(stage="tracker", num_envs=16, device=DEVICE)
        tracker_trainer = SuperMimicTrainer(
            tracker_bridge,
            log_dir=tmp / "tracker",
            cfg=cfg,
            teacher_ckpt=refiner_ckpt,
        )
        assert tracker_trainer.stage == "tracker"
        assert tracker_trainer.teacher_policy is not None
        tracker_ckpt = tracker_trainer.learn(max_iterations=3)
        assert tracker_ckpt.is_file()

        # 3. Standalone Deployment Inference Load Check
        deployed_tracker = load_supermimic_policy(tracker_ckpt, device=DEVICE)
        obs_test = tracker_bridge.current_obs_dict["policy"]
        act_out = deployed_tracker.act_inference(obs_test)
        assert act_out.shape == (16, 29) and torch.isfinite(act_out).all()

    print("[PASS] Test 7: Full Stage 1 (SuperMimic Refiner) -> Stage 2 (SuperMimic Tracker + Teacher BC) -> Inference")


def test_8_review_regressions_obs_xi_com_and_rewards() -> None:
    torch.manual_seed(7)
    fk = FastTreeKinematics(device=DEVICE)
    mock_m = _MockMotionData(fk, device=DEVICE)

    # 1. H2 & L5: Strict per-slice alignment assertion + anchor_only MotionReferenceAdapter
    ref_full = MotionReferenceAdapter(mock_m, anchor_only=False)
    ref_anchor = MotionReferenceAdapter(mock_m, anchor_only=True)
    m_ids = torch.tensor([0, 1, 2, 3], device=DEVICE)
    t_s = torch.tensor([3, 7, 11, 15], device=DEVICE)
    w_full = ref_full.get_future_window(m_ids, t_s)
    w_anch = ref_anchor.get_future_window(m_ids, t_s)
    assert (w_full.anchor_pos_w - w_anch.anchor_pos_w).abs().max().item() < 1e-6
    assert (w_full.joint_pos - w_anch.joint_pos).abs().max().item() < 1e-6

    obs_base = torch.randn(4, PRIVILEGED_OBS_DIM, device=DEVICE)
    diff_ok, _ = assert_privileged_obs_aligned(obs_base, obs_base.clone(), atol=5e-3)
    assert diff_ok == 0.0
    obs_bad = obs_base.clone()
    obs_bad[2, 876] += 0.02  # Inject 2cm mismatch into `obj_pos_b` (875:878)
    raised = False
    try:
        assert_privileged_obs_aligned(obs_base, obs_bad, atol=5e-3, label="critic")
    except AssertionError as exc:
        raised = True
        assert "obj_pos_b" in str(exc), f"Expected offending slice name 'obj_pos_b' in error: {exc}"
    assert raised, "assert_privileged_obs_aligned failed to raise on 0.02 slice mismatch!"

    # 2. H4 & L1: `default_joint_pos_nominal` vs randomized env0 + body COM linear velocity lever-arm
    nominal = torch.linspace(-0.2, 0.2, 29, device=DEVICE)
    delta_per_env = torch.randn(4, 29, device=DEVICE) * 0.05
    delta_per_env[0] = 0.05  # Deliberate non-zero env0 bias!
    com_offsets = torch.zeros(4, fk.num_links, 3, device=DEVICE)
    com_offsets[:, :, 2] = 0.08  # 8cm local +Z COM offset on every link

    class _MockRobotData:
        default_joint_pos_nominal = nominal
        default_joint_pos = nominal.unsqueeze(0) + delta_per_env
        body_com_pos_b = com_offsets

    class _MockRobot:
        joint_names = list(G1_29DOF_SDK_JOINT_NAMES)
        body_names = list(fk.link_names)
        data = _MockRobotData()

    fk_com = FastTreeKinematics(device=DEVICE)
    fk_com.sync_from_isaaclab_robot(_MockRobot(), action_scale=fk.action_scale)
    assert (fk_com.default_joint_pos - nominal).abs().max().item() < 1e-6, "env0 random bias leaked into nominal!"

    task_cb = SugarHOITask(fk_com, "CarryBox", stage="refiner")
    xi = torch.zeros(4, XI_DIM, device=DEVICE)
    xi[:, 5:34] = delta_per_env
    act = torch.randn(4, 29, device=DEVICE)
    q_sp = task_cb.raw_action_to_setpoint(act, xi=xi)
    expected_sp = nominal.unsqueeze(0) + delta_per_env + fk_com.action_scale.unsqueeze(0) * act
    assert (q_sp - expected_sp).abs().max().item() < 1e-6

    # Verify rigid-body COM velocity lever-arm: pure angular spin around Y at q=0, qd=0
    rp = torch.zeros(4, 3, device=DEVICE)
    rq = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE).expand(4, 4)
    q_zero = torch.zeros(4, 29, device=DEVICE)
    qd_zero = torch.zeros(4, 29, device=DEVICE)
    w_spin = torch.tensor([[0.0, 2.0, 0.0]], device=DEVICE).expand(4, 3)
    # Root origin at (0,0,0) motionless -> root COM at (0,0,0.08) has v_root_com = w x c_root = (0.16, 0, 0)
    v_root_com = torch.cross(w_spin, com_offsets[:, 0], dim=-1)
    fk_spin = fk_com(rp, rq, q_zero, v_root_com, w_spin, qd_zero)
    for k, lk_idx in enumerate(fk_com.tracked_link_indices.tolist()):
        com_w = torch.einsum("bij,bj->bi", fk_spin.tracked_rotmat_w[:, k], com_offsets[:, lk_idx])
        r_com_w = fk_spin.tracked_pos_w[:, k] + com_w
        v_exact = torch.cross(w_spin, r_com_w, dim=-1)
        assert (fk_spin.tracked_lin_vel_w[:, k] - v_exact).abs().max().item() < 1e-5

    # Verify `joint_pos_rel` in both 890-D PrivilegedCfg [788:817] and Tracker proprio subtracts `nominal + delta_env`
    s_test = SugarHOITask.pack_state(
        rp, rq, nominal.unsqueeze(0) + delta_per_env + 0.123, v_root_com, w_spin, qd_zero, rp, rq, rp, rp
    )
    obs_priv = task_cb.build_privileged_obs(s_test, act, w_full, xi=xi)
    assert (obs_priv[:, 788:817] - 0.123).abs().max().item() < 1e-6
    _, q_rel_prop, _, _, _ = task_cb.single_step_proprio(s_test, act, xi=xi)
    assert (q_rel_prop - 0.123).abs().max().item() < 1e-6

    # 3. M1 & M2: Unbroken same-episode history across push + earliest-action padding on new episode start
    rb_ep10 = HOIReplayBuffer(capacity=16, num_envs=1, device=DEVICE)
    s1 = torch.ones(1, SDIM, device=DEVICE)
    c1 = torch.zeros(1, CONTACT_DIM, device=DEVICE)
    z_id = torch.zeros(1, dtype=torch.long, device=DEVICE)
    ep0 = torch.tensor([10], dtype=torch.int64, device=DEVICE)
    for step_i in range(5):
        valid_tr = torch.tensor([step_i != 1], dtype=torch.bool, device=DEVICE)  # push at step 1!
        a_i = torch.full((1, 29), float(step_i + 1), device=DEVICE)
        rb_ep10.add(s1 * float(step_i + 1), a_i, a_i, s1 * float(step_i + 2), c1, z_id, z_id + step_i, ep0, valid_tr)
    # Manually check when ring index 4 (5th step) is sampled:
    torch.manual_seed(0)
    for _ in range(32):
        st_ep10 = rb_ep10.sample_imagination_starts(64, need_history=True)
        mask_idx4 = st_ep10.time_steps == 4
        if mask_idx4.any():
            # All 5 history states [1, 2, 3, 4, 5] must be preserved across the push at step 1!
            expected_5 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)
            assert (st_ep10.hist_states[mask_idx4][0, :, 0] - expected_5).abs().max().item() < 1e-5
            break

    # Force check on single-frame new episode padding:
    rb_new = HOIReplayBuffer(capacity=16, num_envs=1, device=DEVICE)
    rb_new.add(
        s1 * 9.0,
        torch.ones(1, 29, device=DEVICE),
        torch.full((1, 29), 0.42, device=DEVICE),
        s1,
        c1,
        z_id,
        z_id,
        torch.tensor([77], dtype=torch.int64, device=DEVICE),
    )
    st_new = rb_new.sample_imagination_starts(4, need_history=True)
    assert (st_new.hist_last_actions - 0.42).abs().max().item() < 1e-3, "First-step action padding reverted to zero!"

    # 4. M3: Stage-aware PushBox profile, undesired contact body count scaling, and `feet_air_time` penalty
    p_ref = resolve_task_profile("PushBox", stage="refiner")
    p_trk = resolve_task_profile("PushBox", stage="tracker")
    assert p_ref.undesired_weight == -0.1 and p_ref.undesired_threshold == 1.0
    assert p_trk.undesired_weight == -1.0 and p_trk.undesired_threshold == 0.1

    task_push_trk = SugarHOITask(fk, "PushBox", stage="tracker")
    f0 = ref_full.get_frame(m_ids, t_s)
    s_perf = SugarHOITask.pack_state(
        f0.body_pos_w[:, 0], f0.body_quat_w[:, 0], f0.joint_pos,
        f0.body_lin_vel_w[:, 0], f0.body_ang_vel_w[:, 0], f0.joint_vel,
        f0.obj_pos_w, f0.obj_quat_w, f0.obj_lin_vel_w, f0.obj_ang_vel_w,
    )
    logits_no_undes = torch.tensor([[10.0, -10.0, -10.0, -10.0, -20.0]], device=DEVICE).expand(4, 5)
    logits_half_undes = torch.tensor([[10.0, -10.0, -10.0, -10.0, 0.0]], device=DEVICE).expand(4, 5)  # sigmoid(0)=0.5 -> 6 bodies
    r_clean, _, _ = task_push_trk.compute_reward_and_termination(s_perf, s_perf, act, act, ref_cur=f0, contact_logits=logits_no_undes)
    r_undes, _, _ = task_push_trk.compute_reward_and_termination(s_perf, s_perf, act, act, ref_cur=f0, contact_logits=logits_half_undes)
    expected_diff = 0.5 * float(task_push_trk.profile.num_undesired_bodies) * 1.0 * 0.02  # 6 * 1.0 * 0.02 = 0.12
    assert abs(float((r_clean[0] - r_undes[0]).item()) - expected_diff) < 1e-4

    # Feet air time: short swing (0.18s < 0.5s) landing at t=60 (>50) incurs negative penalty; long swing (0.6s) incurs 0
    logits_land = torch.tensor([[10.0, -10.0, 20.0, -20.0, -20.0]], device=DEVICE).expand(4, 5)
    prev_swing = torch.zeros(4, 2, device=DEVICE)
    air_short = torch.tensor([[0.18, 0.0]], device=DEVICE).expand(4, 2)
    air_long = torch.tensor([[0.60, 0.0]], device=DEVICE).expand(4, 2)
    t_late = torch.full((4,), 60, dtype=torch.long, device=DEVICE)
    r_short, _, info_s = task_push_trk.compute_reward_and_termination(
        s_perf, s_perf, act, act, ref_cur=f0, contact_logits=logits_land,
        feet_air_time=air_short, prev_feet_gnd=prev_swing, time_steps=t_late,
    )
    r_long, _, info_l = task_push_trk.compute_reward_and_termination(
        s_perf, s_perf, act, act, ref_cur=f0, contact_logits=logits_land,
        feet_air_time=air_long, prev_feet_gnd=prev_swing, time_steps=t_late,
    )
    assert float(info_l["rew/feet_air_time"].abs().max().item()) < 1e-5
    assert float((r_long[0] - r_short[0]).item()) > 0.02
    print("[PASS] Test 8: Per-slice obs assertion, xi/COM kinematics, history padding & PushBox/feet_air_time rewards")


def test_9_timeout_bootstrap_resume_and_compile() -> None:
    torch.manual_seed(8)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)

        # 1. H1: Exact current-step timeout bootstrap immune to post-reset `next_obs` contamination
        class _TimeoutContaminatedBridge(_SyntheticSugarBridge):
            def step(self, raw_action: torch.Tensor):
                N = self.num_envs
                rew = torch.full((N,), 2.0, device=self.device)
                dones = torch.ones(N, dtype=torch.bool, device=self.device)
                timeouts = torch.ones(N, dtype=torch.bool, device=self.device)
                pushed = torch.zeros(N, dtype=torch.bool, device=self.device)
                # Contaminate post-reset `current_obs_dict` with extreme values!
                bad_obs = torch.full((N, PRIVILEGED_OBS_DIM), 1e6, device=self.device)
                self.current_obs_dict = {"policy": bad_obs, "critic": bad_obs}
                return self.current_obs_dict, rew, dones, timeouts, pushed, {"time_outs": timeouts}

        t_bridge = _TimeoutContaminatedBridge(stage="refiner", num_envs=8, device=DEVICE)
        cfg_1step = SuperMimicTrainCfg(steps_per_env=1, replay_capacity=16, gamma=0.99, enable_tensorboard=False)
        t_trainer = SuperMimicTrainer(t_bridge, log_dir=tmp / "timeout_test", cfg=cfg_1step)
        # Set critic to return a distinct linear function of input so bad_obs (1e6) would blow up if used!
        with torch.no_grad():
            for head in t_trainer._raw_critic.heads:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.constant_(m.weight, 0.05)
                        nn.init.constant_(m.bias, 1.25)
            obs0 = t_bridge.current_obs_dict["critic"].clone()
            t_trainer._raw_critic.obs_rms.update(obs0)
            v0_expected = t_trainer.critic(obs0, update_rms=False, pessimistic=False)

        # Freeze obs_rms count huge so the single update inside collect_rollout doesn't shift v0_expected
        t_trainer._raw_critic.obs_rms.count.fill_(1e12)
        _, returns_flat, _ = t_trainer.collect_rollout()
        expected_ret = 2.0 + 0.99 * v0_expected
        assert (returns_flat - expected_ret).abs().max().item() < 1e-4, (
            f"Timeout bootstrap polluted by post-reset obs! got {returns_flat}, expected {expected_ret}"
        )

        # 2. M8 (`stage="auto"` ignores accidental `teacher_ckpt` on 890-D Refiner), M6 (value clip), M7 (resume & compile)
        ref_bridge = _SyntheticSugarBridge(stage="refiner", num_envs=8, device=DEVICE)
        cfg_comp = SuperMimicTrainCfg(
            steps_per_env=6,
            replay_capacity=24,
            wm_seq_len=4,
            wm_updates_per_iter=1,
            wm_batch_size=16,
            wm_warmup_iters=0,
            imagination_horizon=4,
            policy_updates_per_iter=1,
            policy_batch_size=8,
            value_clip_range=(-25.0, 15.0),
            enable_tensorboard=False,
            compile_models=True,
            compile_backend="aot_eager",
        )
        dummy_ckpt = tmp / "dummy_refiner.pt"
        t_trainer.save_checkpoint(dummy_ckpt, iteration=7)

        # Passing `teacher_ckpt` to an 890-D Refiner bridge must stay in "refiner" stage
        trainer_comp = SuperMimicTrainer(
            ref_bridge, log_dir=tmp / "comp_refiner", cfg=cfg_comp, teacher_ckpt=dummy_ckpt
        )
        assert trainer_comp.stage == "refiner" and trainer_comp.teacher_policy is None

        # Verify resume advances to `iteration + 1 = 8`
        next_it = trainer_comp.load_checkpoint(dummy_ckpt)
        assert next_it == 8 and trainer_comp.current_iter == 8

        # Spike Critic head biases to +5000 to verify `value_clip_range=(-25, 15)` bounds imagined_return
        with torch.no_grad():
            for head in trainer_comp._raw_critic.heads:
                head[-1].bias.fill_(5000.0)
        trainer_comp.collect_rollout()
        pol_metrics = trainer_comp.update_policy(iteration=8)
        assert abs(pol_metrics["policy/imagined_return"]) < 100.0, (
            f"Critic spike leaked past value_clip_range: {pol_metrics['policy/imagined_return']}"
        )

        # Run compiled `learn` from iteration 8..10 and test `_orig_mod.` prefix stripping on load
        out_ckpt = trainer_comp.learn(max_iterations=10)
        raw_payload = torch.load(out_ckpt, map_location=DEVICE, weights_only=False)
        assert all(not k.startswith("_orig_mod.") for k in raw_payload["policy_state_dict"])
        prefixed_payload = dict(raw_payload)
        prefixed_payload["policy_state_dict"] = {
            f"_orig_mod.{k}": v for k, v in raw_payload["policy_state_dict"].items()
        }
        prefixed_ckpt = tmp / "prefixed_model.pt"
        torch.save(prefixed_payload, prefixed_ckpt)
        loaded_pol = load_supermimic_policy(prefixed_ckpt, device=DEVICE)
        act_check = loaded_pol.act_inference(ref_bridge.current_obs_dict["policy"])
        assert act_check.shape == (8, 29) and torch.isfinite(act_check).all()

    print("[PASS] Test 9: Timeout bootstrap safety, value clipping, it+1 resume & torch.compile interop")


class _SyntheticGeneratorDataset(torch.utils.data.Dataset):
    """In-memory low-dim dataset matching `GeneratorDataset` schema without requiring `zarr`."""

    def __init__(self, length: int = 12, horizon: int = 8, n_obs_steps: int = 1, use_target: bool = True) -> None:
        super().__init__()
        self.length = length
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.use_target = use_target
        rng = torch.Generator().manual_seed(123)
        self._raw = {
            "action": torch.randn(length * horizon, 36, generator=rng).numpy(),
            "last_action": torch.randn(length * horizon, 36, generator=rng).numpy(),
            "obj_pos_b": torch.randn(length * horizon, 3, generator=rng).numpy(),
            "obj_ori_b": torch.randn(length * horizon, 6, generator=rng).numpy(),
            "joint_pos": torch.randn(length * horizon, 29, generator=rng).numpy(),
            "project_gravity": torch.randn(length * horizon, 3, generator=rng).numpy(),
            "target_obj_pos_b": torch.randn(length * horizon, 3, generator=rng).numpy(),
            "target_obj_ori_b": torch.randn(length * horizon, 6, generator=rng).numpy(),
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        s = idx * self.horizon
        e = s + self.horizon
        t_obs = s + self.n_obs_steps
        obs = {
            "obj_pos_b": torch.from_numpy(self._raw["obj_pos_b"][s:t_obs]),
            "obj_ori_b": torch.from_numpy(self._raw["obj_ori_b"][s:t_obs]),
            "last_action": torch.from_numpy(self._raw["last_action"][s:t_obs]),
        }
        if self.use_target:
            obs["target_obj_pos_b"] = torch.from_numpy(self._raw["target_obj_pos_b"][s:t_obs])
            obs["target_obj_ori_b"] = torch.from_numpy(self._raw["target_obj_ori_b"][s:t_obs])
        return {"obs": obs, "action": torch.from_numpy(self._raw["action"][s:e])}

    def get_validation_dataset(self) -> "_SyntheticGeneratorDataset":
        return _SyntheticGeneratorDataset(length=4, horizon=self.horizon, n_obs_steps=self.n_obs_steps, use_target=self.use_target)

    def get_normalizer(self, mode: str = "limits", **kwargs):
        from sugar_il.model.common.normalizer import LinearNormalizer

        norm = LinearNormalizer()
        norm.fit(data=self._raw, last_n_dims=1, mode=mode, **kwargs)
        return norm


def test_10_colab_interrupt_auto_resume_and_generator_ckpt() -> None:
    import types
    from omegaconf import OmegaConf

    torch.manual_seed(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)

        # --- Part A: SuperMimic auto_resume, replay warmup, corrupted-latest fallback & idempotent completion ---
        rl_dir = tmp / "refiner_interrupt_test"
        cfg_rl = SuperMimicTrainCfg(
            steps_per_env=3,           # Deliberately < wm_seq_len=6 to exercise auto-warmup on resume!
            replay_capacity=24,
            wm_seq_len=6,
            wm_updates_per_iter=1,
            wm_batch_size=16,
            wm_warmup_iters=0,
            imagination_horizon=4,
            policy_updates_per_iter=1,
            policy_batch_size=8,
            save_interval=2,
            enable_tensorboard=False,
        )
        b1 = _SyntheticSugarBridge(stage="refiner", num_envs=8, device=DEVICE)
        t1 = SuperMimicTrainer(b1, log_dir=rl_dir, cfg=cfg_rl)
        assert t1.auto_resume() is None  # Clean start returns None
        t1.learn(max_iterations=3)       # Runs it=0,1,2 -> saves model_2.pt & model_latest.pt (next_iteration=3)
        assert (rl_dir / "model_latest.pt").is_file() and (rl_dir / "model_2.pt").is_file()
        assert not list(rl_dir.glob("*.tmp")), "Atomic save left stray .tmp files!"

        # Simulate Colab runtime reset & resume from model_latest.pt up to max_iterations=5
        b2 = _SyntheticSugarBridge(stage="refiner", num_envs=8, device=DEVICE)
        t2 = SuperMimicTrainer(b2, log_dir=rl_dir, cfg=cfg_rl)
        resumed_p = t2.auto_resume()
        assert resumed_p == (rl_dir / "model_latest.pt") and t2.current_iter == 3
        final_p = t2.learn(max_iterations=5)  # Warms up empty replay to >=6 steps, then runs it=3,4
        assert t2.replay.size >= 6 and final_p == (rl_dir / "model_4.pt") and final_p.is_file()

        # Simulate corrupted model_latest.pt -> must gracefully fall back to highest valid model_4.pt!
        (rl_dir / "model_latest.pt").write_bytes(b"truncated_corrupt_bytes")
        b3 = _SyntheticSugarBridge(stage="refiner", num_envs=8, device=DEVICE)
        t3 = SuperMimicTrainer(b3, log_dir=rl_dir, cfg=cfg_rl)
        fallback_p = t3.auto_resume()
        assert fallback_p == (rl_dir / "model_4.pt") and t3.current_iter == 5
        # Calling learn(max_iterations=5) when already at iter 5 must return immediately & backfill missing target if needed
        (rl_dir / "model_4.pt").unlink()
        re_saved = t3.learn(max_iterations=5)
        assert re_saved == (rl_dir / "model_4.pt") and re_saved.is_file()

        # --- Part B: Diffusion Generator (`sugar_il`) mid-run resume & `GeneratorWrapper.load` interop ---
        sugar_il_src = str(REPO_ROOT / "source" / "sugar_il")
        if sugar_il_src not in sys.path:
            sys.path.insert(0, sugar_il_src)

        # Provide minimal lightweight `accelerate` shim only if `accelerate` is not installed in local env
        injected_acc = False
        try:
            import accelerate  # noqa: F401
        except ImportError:
            import diffusers  # noqa: F401  (import before shimming sys.modules['accelerate'])
            import importlib.machinery

            injected_acc = True
            acc_mod = types.ModuleType("accelerate")
            acc_mod.__spec__ = importlib.machinery.ModuleSpec("accelerate", loader=None)
            acc_utils = types.ModuleType("accelerate.utils")
            acc_utils.__spec__ = importlib.machinery.ModuleSpec("accelerate.utils", loader=None)

            class _ProjectConfiguration:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class _MiniAccelerator:
                def __init__(self, **kwargs):
                    self.device = DEVICE
                    self.is_main_process = True
                    self.mixed_precision = "no"

                def init_trackers(self, *args, **kwargs):
                    pass

                def wait_for_everyone(self):
                    pass

                def prepare(self, *objs):
                    return tuple(o.to(self.device) if isinstance(o, nn.Module) else o for o in objs)

                def backward(self, loss):
                    loss.backward()

                def unwrap_model(self, m):
                    return m

                def gather(self, x):
                    return x

                def log(self, *args, **kwargs):
                    pass

                def end_training(self):
                    pass

            acc_mod.Accelerator = _MiniAccelerator
            acc_utils.ProjectConfiguration = _ProjectConfiguration
            acc_mod.utils = acc_utils
            sys.modules["accelerate"] = acc_mod
            sys.modules["accelerate.utils"] = acc_utils

        try:
            from sugar_il.workspace.train_generator_workspace import TrainGeneratorWorkspace
            from sugar_il.wrapper.sugar_il_wrapper import GeneratorObs, GeneratorWrapper

            gen_dir = tmp / "generator_interrupt_test"
            gen_dir.mkdir(parents=True, exist_ok=True)

            def _make_gen_cfg(num_epochs: int) -> OmegaConf:
                return OmegaConf.create(
                    {
                        "name": "train_generator",
                        "task_name": "CarryBox",
                        "n_action_steps": 8,
                        "n_obs_steps": 1,
                        "use_last_action": True,
                        "use_target": True,
                        "num_epochs": num_epochs,
                        "shape_meta": {"obs": {"n_obs_steps": 1}, "action": {"shape": [36], "horizon": 8}},
                        "policy": {
                            "_target_": "sugar_il.policy.generator.Generator",
                            "shape_meta": {"obs": {"n_obs_steps": 1}, "action": {"shape": [36], "horizon": 8}},
                            "noise_scheduler": {
                                "_target_": "diffusers.DDIMScheduler",
                                "num_train_timesteps": 10,
                                "beta_start": 0.0001,
                                "beta_end": 0.02,
                                "beta_schedule": "squaredcos_cap_v2",
                                "clip_sample": True,
                                "set_alpha_to_one": True,
                                "steps_offset": 0,
                                "prediction_type": "epsilon",
                            },
                            "obs_encoder": {
                                "_target_": "sugar_il.model.encoder.generator_state_encoder.GeneratorStateObsEncoder",
                                "shape_meta": {"obs": {"n_obs_steps": 1}, "action": {"shape": [36], "horizon": 8}},
                                "feature_dim": 64,
                                "use_last_action": True,
                                "use_target": True,
                            },
                            "num_inference_steps": 4,
                            "n_layer": 2,
                            "n_head": 4,
                            "p_drop_attn": 0.0,
                            "use_attn_mask": True,
                            "start_ckpt_path": None,
                        },
                        "task": {
                            "name": "CarryBox",
                            "dataset": {
                                "_target_": f"{__name__}._SyntheticGeneratorDataset",
                                "length": 8,
                                "horizon": 8,
                                "n_obs_steps": 1,
                                "use_target": True,
                            },
                            "env_runner": {"_target_": "sugar_il.env_runner.generator_runner.GeneratorRunner"},
                        },
                        "dataloader": {"batch_size": 4, "num_workers": 0, "shuffle": False},
                        "val_dataloader": {"batch_size": 4, "num_workers": 0, "shuffle": False},
                        "optimizer": {"lr": 5e-4, "weight_decay": 1e-4, "betas": [0.95, 0.999]},
                        "training": {
                            "device": str(DEVICE),
                            "seed": 42,
                            "debug": False,
                            "resume": True,
                            "lr_scheduler": "cosine",
                            "lr_warmup_steps": 2,
                            "num_epochs": num_epochs,
                            "gradient_accumulate_every": 1,
                            "use_ema": False,
                            "rollout_every": 10,
                            "checkpoint_every": 2,
                            "val_every": 100,
                            "sample_every": 100,
                            "gen_attn_map": False,
                            "max_train_steps": None,
                            "max_val_steps": None,
                            "tqdm_interval_sec": 10.0,
                        },
                        "logging": {"project": "train_generator", "name": "unit_test"},
                        "checkpoint": {
                            "topk": {
                                "monitor_key": "train_loss",
                                "mode": "min",
                                "k": 2,
                                "format_str": "epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt",
                            },
                            "save_last_ckpt": True,
                            "save_last_snapshot": False,
                        },
                    }
                )

            # Run epochs 0, 1, 2 (num_epochs=3, 2 batches/epoch -> global_step=6, epoch=3 in latest.ckpt)
            ws1 = TrainGeneratorWorkspace(_make_gen_cfg(num_epochs=3))
            ws1._output_dir = str(gen_dir)
            ws1.run()
            assert ws1.epoch == 3 and ws1.global_step == 6
            assert (gen_dir / "checkpoints" / "latest.ckpt").is_file()
            assert (gen_dir / "epoch_checkpoints" / "epoch=2.ckpt").is_file()
            assert not list(gen_dir.rglob("*.tmp")), "Generator atomic save left stray .tmp files!"

            # Simulate Colab disconnect & resume to num_epochs=5 -> must start at epoch=3, global_step=6!
            ws2 = TrainGeneratorWorkspace(_make_gen_cfg(num_epochs=5))
            ws2._output_dir = str(gen_dir)
            ws2.run()
            assert ws2.epoch == 5 and ws2.global_step == 10
            final_gen_ckpt = gen_dir / "epoch_checkpoints" / "epoch=4.ckpt"
            assert final_gen_ckpt.is_file()

            # Verify closed-loop `GeneratorWrapper.load()` works cleanly with resumed checkpoint containing optimizer state
            wrapper = GeneratorWrapper.load(str(final_gen_ckpt), device=str(DEVICE))
            obs_eval = GeneratorObs(
                obj_pos_b=torch.zeros(2, 5, 3, device=DEVICE),
                obj_ori_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE).view(1, 1, 4).expand(2, 5, 4),
                joint_pos=torch.zeros(2, 5, 29, device=DEVICE),
                project_gravity=torch.tensor([0.0, 0.0, -1.0], device=DEVICE).view(1, 1, 3).expand(2, 5, 3),
                target_obj_pos_b=torch.zeros(2, 5, 3, device=DEVICE),
                target_obj_ori_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE).view(1, 1, 4).expand(2, 5, 4),
                last_command=torch.zeros(2, 5, 36, device=DEVICE),
            )
            pred_interp = wrapper.predict(obs_eval)
            assert pred_interp.shape == (2, 36, 36) and torch.isfinite(pred_interp).all()
        finally:
            if injected_acc:
                sys.modules.pop("accelerate", None)
                sys.modules.pop("accelerate.utils", None)

    print("[PASS] Test 10: Colab interrupt atomic save, RL auto-resume/fallback & Diffusion Generator resume")


def test_11_fast_fk_depth_vectorization_and_scale_grad() -> None:
    """Verify 11-layer depth-batched FastTreeKinematics and `FKResult.scale_grad(alpha)` exact 1st-order gradient equivalence."""
    torch.manual_seed(11)
    fk = FastTreeKinematics(device=DEVICE)
    assert fk.max_depth == 11, f"Expected G1 kinematic tree max_depth=11, got {fk.max_depth}"
    task = SugarHOITask(fk, "CarryBox")

    B = 8
    s_base = SugarHOITask.pack_state(
        root_pos=torch.randn(B, 3, device=DEVICE),
        root_quat=quat_normalize(torch.randn(B, 4, device=DEVICE)),
        q=torch.randn(B, 29, device=DEVICE) * 0.25,
        root_lin_vel=torch.randn(B, 3, device=DEVICE) * 0.3,
        root_ang_vel=torch.randn(B, 3, device=DEVICE) * 0.3,
        qd=torch.randn(B, 29, device=DEVICE) * 0.5,
        obj_pos=torch.randn(B, 3, device=DEVICE),
        obj_quat=quat_normalize(torch.randn(B, 4, device=DEVICE)),
        obj_lin_vel=torch.randn(B, 3, device=DEVICE) * 0.2,
        obj_ang_vel=torch.randn(B, 3, device=DEVICE) * 0.2,
    )
    alpha = 0.98

    # Method A (reference): decay state tensor directly, then call run_fk
    s_ref = s_base.clone().detach().requires_grad_(True)
    s_decayed = s_ref * alpha + s_ref.detach() * (1.0 - alpha)
    fk_ref = task.run_fk(s_decayed)
    loss_ref = (
        fk_ref.tracked_pos_w.pow(2).sum()
        + fk_ref.tracked_quat_w.pow(2).sum()
        + fk_ref.tracked_lin_vel_w.pow(2).sum()
        + fk_ref.contact_pos_w.pow(2).sum()
    )
    g_ref = torch.autograd.grad(loss_ref, s_ref)[0]

    # Method B (reused FKResult): compute FK once, then scale_grad(alpha)
    s_reuse = s_base.clone().detach().requires_grad_(True)
    fk_raw = task.run_fk(s_reuse)
    fk_scaled = fk_raw.scale_grad(alpha, fk.anchor_tracked_idx)
    loss_reuse = (
        fk_scaled.tracked_pos_w.pow(2).sum()
        + fk_scaled.tracked_quat_w.pow(2).sum()
        + fk_scaled.tracked_lin_vel_w.pow(2).sum()
        + fk_scaled.contact_pos_w.pow(2).sum()
    )
    g_reuse = torch.autograd.grad(loss_reuse, s_reuse)[0]

    grad_diff = (g_ref - g_reuse).abs().max().item()
    assert grad_diff < 1e-5, f"FKResult.scale_grad gradient mismatch: {grad_diff}"
    print(f"[PASS] Test 11: Depth-batched FK (depth={fk.max_depth}) & FKResult.scale_grad equivalence (diff={grad_diff:.2e})")


def test_12_omniretarget_clean_usda_and_bridge_sync() -> None:
    """Verify v1 CarryBox strict undesired contacts (20 bodies), hands-only contact sensor,
    fail-fast USD enforcement, rollout propagation, and clean OmniCarryBox dataset invariants.
    """
    import ast
    import json
    import subprocess
    from sugar_rl.supermimic.env_bridge import _resolve_undesired_body_ids

    cf_35 = [
        "pelvis", "left_hip_pitch_link", "pelvis_contour_link", "right_hip_pitch_link",
        "waist_yaw_link", "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
        "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
        "right_knee_link", "logo_link", "left_shoulder_pitch_link", "head_link",
        "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
        "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
        "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
        "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link",
        "right_wrist_yaw_link", "left_rubber_hand", "right_rubber_hand",
    ]

    class _SCfg:
        def __init__(self, body_names, body_ids=slice(None)):
            self.body_names = body_names
            self.body_ids = body_ids

    class _Term:
        def __init__(self, sensor_cfg):
            self.params = {"sensor_cfg": sensor_cfg}

    class _RewMgr:
        def __init__(self, names, cfgs):
            self._term_names = names
            self._term_cfgs = cfgs

    # 1. Verify CarryBox undesired_contacts regex keeps head/torso/elbows/pelvis/legs (25 bodies),
    #    matching both RewardManager regex and SugarEnvBridge fallback!
    cb_pats = [
        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_rubber_hand$)(?!right_rubber_hand$)(?!left_wrist_.*_link$)(?!right_wrist_.*_link$).+$"
    ]
    rm_cb = _RewMgr(["undesired_contacts"], [_Term(_SCfg(cb_pats))])
    cb_ids = _resolve_undesired_body_ids(cf_35, "CarryBox", "hands", rew_mgr=rm_cb)
    assert cb_ids == _resolve_undesired_body_ids(cf_35, "CarryBox", "hands", rew_mgr=None)
    assert len(cb_ids) == 25, f"Expected 25 penalized bodies (35 - 10 exempt), got {len(cb_ids)}"
    cb_names = {cf_35[i] for i in cb_ids}
    # Verify head, torso, elbows, pelvis, knees ARE penalized, while palms, wrists, and ankle_roll feet are exempt
    for must_penalize in ("head_link", "torso_link", "waist_yaw_link", "left_elbow_link", "right_elbow_link", "pelvis", "left_knee_link"):
        assert must_penalize in cb_names, f"{must_penalize} missing from CarryBox undesired_contacts!"
    for must_exempt in ("left_ankle_roll_link", "right_ankle_roll_link", "left_rubber_hand", "right_rubber_hand", "left_wrist_yaw_link", "right_wrist_roll_link"):
        assert must_exempt not in cb_names, f"{must_exempt} should be exempt from CarryBox undesired_contacts!"

    p_cb = resolve_task_profile("Sugar-G129dof-CarryBox-Refiner")
    assert p_cb.undesired_weight == -1.0 and p_cb.undesired_threshold == 0.1 and p_cb.num_undesired_bodies == 25

    # Verify carry_box_*_env_cfg no longer uses 5-link forearm/wrist contact sensor override
    for rel_cfg in (
        "source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/carry_box_refiner_env_cfg.py",
        "source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_tracker/carry_box_tracker_env_cfg.py",
        "source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/inference/carry_box_inference_env_cfg.py",
    ):
        cfg_txt = (REPO_ROOT / rel_cfg).read_text(encoding="utf-8")
        assert "rubber_hand|wrist_.*_link|elbow_link" not in cfg_txt, f"Option 2 sensor override still in {rel_cfg}"

    # 2. Verify `apply_dataset_box_spec` fail-fast guard (no CuboidCfg fallback)
    obj_src = (REPO_ROOT / "source" / "sugar_rl" / "sugar_rl" / "assets" / "objects" / "objects.py").read_text(encoding="utf-8")
    assert "sim_utils.CuboidCfg" not in obj_src and "FileNotFoundError" in obj_src
    mod_ast = ast.parse(obj_src)
    fn_node = next(n for n in mod_ast.body if isinstance(n, ast.FunctionDef) and n.name == "apply_dataset_box_spec")
    fn_src = ast.get_source_segment(obj_src, fn_node)
    assert fn_src is not None

    class _CfgNode:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    ns = {
        "RigidObjectCfg": type("RigidObjectCfg", (_CfgNode,), {"InitialStateCfg": _CfgNode}),
        "UsdFileCfg": _CfgNode,
        "sim_utils": _CfgNode(RigidBodyPropertiesCfg=_CfgNode, MassPropertiesCfg=_CfgNode),
    }
    exec(fn_src, ns)
    fn_apply = ns["apply_dataset_box_spec"]

    with tempfile.TemporaryDirectory() as box_tmp:
        b_dir = pathlib.Path(box_tmp) / "missing_usd_ds"
        b_dir.mkdir(parents=True)
        (b_dir / "box_spec.json").write_text(json.dumps({"usd_file": "obj_aligned.usda", "mass_kg": 0.5}), encoding="utf-8")
        stub_env_cfg = _CfgNode(scene=_CfgNode(obj=_CfgNode(prim_path="{ENV_REGEX_NS}/Obj")))
        caught_fnf = False
        try:
            fn_apply(stub_env_cfg, str(b_dir))
        except FileNotFoundError:
            caught_fnf = True
        assert caught_fnf, "Expected FileNotFoundError when box_spec.json exists without convex-hull USD!"

        (b_dir / "obj_aligned.usda").write_text("#usda 1.0\n", encoding="utf-8")
        assert fn_apply(stub_env_cfg, str(b_dir)) is True
        assert abs(stub_env_cfg.scene.obj.spawn.rigid_props.max_depenetration_velocity - 1.0) < 1e-6

    # 3. Verify `process_refiner_rollout.py --ref_dir` propagates convex-hull assets
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        ref_dir = tmp / "ref_omni"
        ref_dir.mkdir(parents=True)
        (ref_dir / "box_spec.json").write_text(json.dumps({"dataset": "test", "usd_file": "obj_aligned.usda", "mass_kg": 0.5}))
        (ref_dir / "obj_aligned.obj").write_text("v 0 0 0\n")
        (ref_dir / "obj_aligned.usda").write_text("#usda 1.0\n")

        data_root = tmp / "refiner_rollout"
        tc_dir = data_root / "raw_npz" / "trajectory_complete"
        tc_dir.mkdir(parents=True)
        import numpy as np
        T = 20
        np.savez(
            tc_dir / "motion_000_env_000_t0-19_idx_0.npz",
            joint_pos=np.zeros((T, 29), dtype=np.float32),
            joint_vel=np.zeros((T, 29), dtype=np.float32),
            body_pos_w=np.zeros((T, 14, 3), dtype=np.float32),
            body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T, 14, 1)),
            body_lin_vel_w=np.zeros((T, 14, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((T, 14, 3), dtype=np.float32),
            obj_pos_w=np.zeros((T, 3), dtype=np.float32),
            obj_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T, 1)),
            obj_lin_vel_w=np.zeros((T, 3), dtype=np.float32),
            obj_ang_vel_w=np.zeros((T, 3), dtype=np.float32),
            hands_contact_label=np.ones((T,), dtype=bool),
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "sugar_rl" / "process_refiner_rollout.py"),
                "--data_dir", str(data_root),
                "--task_name", "CarryBox",
                "--ref_dir", str(ref_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for fname in ("box_spec.json", "obj_aligned.obj", "obj_aligned.usda"):
            assert (data_root / "rl_dataset" / fname).is_file(), f"Missing propagated {fname} in rl_dataset!"

    # 4. Verify `scripts/supermimic/train.py` CLI motion/teacher_motion_folder override
    train_src = (REPO_ROOT / "scripts" / "supermimic" / "train.py").read_text(encoding="utf-8")
    train_mod = ast.parse(train_src)
    main_fn = next(n for n in train_mod.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    override_ifs = [
        ast.get_source_segment(train_src, st)
        for st in main_fn.body
        if isinstance(st, ast.If)
        and (
            "motion_folder" in (ast.get_source_segment(train_src, st) or "")
            or "teacher_motion_folder" in (ast.get_source_segment(train_src, st) or "")
        )
        and "apply_dataset_box_spec" not in (ast.get_source_segment(train_src, st) or "")
    ]
    assert len(override_ifs) == 2
    stub_motion = _CfgNode(motion_folder=None, teacher_motion_folder=None)
    stub_cfg = _CfgNode(commands=_CfgNode(motion=stub_motion))
    stub_args = _CfgNode(motion_folder="rl_ds_dir", teacher_motion_folder="data/OmniCarryBox")
    exec("\n".join(override_ifs), {"args_cli": stub_args, "env_cfg": stub_cfg})
    assert stub_motion.motion_folder == "rl_ds_dir"
    assert stub_motion.teacher_motion_folder == "data/OmniCarryBox"

    # 5. Verify `scene_builder.py` fails fast when box_spec.json exists without obj_aligned.obj
    with tempfile.TemporaryDirectory() as sb_tmp:
        from scripts.sim2sim.scene_builder import build_mujoco_scene_xml

        bad_ds = pathlib.Path(sb_tmp) / "broken_ds"
        bad_ds.mkdir(parents=True)
        (bad_ds / "box_spec.json").write_text(json.dumps({"mesh_file": "missing.obj", "box_size_xyz": [0.3, 0.3, 0.3]}))
        caught_sb = False
        try:
            build_mujoco_scene_xml("CarryBox", output_xml_path=pathlib.Path(sb_tmp) / "s.xml", motion_folder=bad_ds)
        except FileNotFoundError:
            caught_sb = True
        assert caught_sb, "scene_builder silently fell back to cuboid when mesh_file was missing!"

    print("[PASS] Test 12: CarryBox v1 strict anti-collision, hands-only sensor, fail-fast USDA & train.py override")


def test_13_lafan_walk_stitcher_continuity_and_clearance() -> None:
    """Verify SE(2) phase-matched stitching produces C1-smooth positions, consistent gradients, and safe box clearance."""
    import numpy as np
    from scripts.sugar_rl.lafan_walk_stitcher import (
        LafanWalkSegment,
        _box_obb_surface_distance,
        _smooth_savgol_np,
        stitch_lafan_walk_to_omni_clip,
    )
    from scripts.sugar_rl.convert_omniretarget_to_sugar import SUGAR_RAW_35_BODY_NAMES, _ang_vel_world_from_quats

    # 0. Verify Savitzky-Golay upper-body smoothing suppresses high-frequency chatter while keeping DC/low-freq swing
    t_test = np.arange(150, dtype=np.float64) * 0.02
    clean_sig = 0.25 * np.sin(2.0 * np.pi * 0.8 * t_test)[:, None]
    noisy_sig = clean_sig + 0.08 * ((-1.0) ** np.arange(150))[:, None]
    smoothed_sig = _smooth_savgol_np(noisy_sig, window=19, polyorder=2)
    raw_acc_rms = float(np.sqrt(np.mean((np.diff(noisy_sig[10:-10], n=2, axis=0) / 0.0004) ** 2)))
    sm_acc_rms = float(np.sqrt(np.mean((np.diff(smoothed_sig[10:-10], n=2, axis=0) / 0.0004) ** 2)))
    recon_rmse = float(np.sqrt(np.mean((smoothed_sig[10:-10] - clean_sig[10:-10]) ** 2)))
    assert sm_acc_rms < 0.12 * raw_acc_rms and recon_rmse < 0.015, (
        f"SavGol failed to attenuate jitter cleanly: sm_acc={sm_acc_rms:.1f}, raw={raw_acc_rms:.1f}, recon={recon_rmse:.4f}"
    )

    fk = FastTreeKinematics(device="cpu")
    dt = 0.02
    T_w, T_b, T_o = 120, 25, 40

    # Synthetic LAFAN walk along +X at 0.65 m/s in standing posture
    t_w = np.arange(T_w, dtype=np.float64) * dt
    pos_w = np.zeros((T_w, 3), dtype=np.float64)
    pos_w[:, 0] = 0.65 * t_w
    pos_w[:, 2] = 0.79
    quat_w = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T_w, 1))
    q_w = np.zeros((T_w, 29), dtype=np.float64)
    q_w[:, 0] = 0.15 * np.sin(2.0 * np.pi * t_w / 0.8)  # subtle walking swing
    with torch.no_grad():
        fk_w = fk(
            torch.from_numpy(pos_w.astype(np.float32)),
            torch.from_numpy(quat_w.astype(np.float32)),
            torch.from_numpy(q_w.astype(np.float32)),
        )
    bpos_w = fk_w.tracked_pos_w.numpy().astype(np.float64)
    fk_35 = FastTreeKinematics(
        urdf_or_xml_path=(REPO_ROOT / "descriptions" / "robots" / "g1" / "g1_29dof_rev_1_0_with_rubber_hand.urdf")
        if (REPO_ROOT / "descriptions" / "robots" / "g1" / "g1_29dof_rev_1_0_with_rubber_hand.urdf").is_file()
        else None,
        tracked_body_names=[n for n in SUGAR_RAW_35_BODY_NAMES if n in FastTreeKinematics(device="cpu").link_to_idx],
        device="cpu",
    )
    body_names_35 = list(fk_35.tracked_body_names)
    la_i = body_names_35.index("left_ankle_roll_link")
    ra_i = body_names_35.index("right_ankle_roll_link")
    with torch.no_grad():
        fk_w35 = fk_35(
            torch.from_numpy(pos_w.astype(np.float32)),
            torch.from_numpy(quat_w.astype(np.float32)),
            torch.from_numpy(q_w.astype(np.float32)),
        )
    bpos_w35 = fk_w35.tracked_pos_w.numpy().astype(np.float64)
    vel_w = np.gradient(pos_w, dt, axis=0)
    qd_w = np.gradient(q_w, dt, axis=0)
    ang_w = _ang_vel_world_from_quats(quat_w, dt)

    seg = LafanWalkSegment(
        source_name="walk_synth.csv",
        start_frame_50hz=0,
        end_frame_50hz=T_w,
        pos_w=pos_w,
        quat_wxyz=quat_w,
        q_bfs=q_w,
        qd_bfs_end=qd_w[-1],
        v_pelvis_end=vel_w[-1],
        w_pelvis_end=ang_w[-1],
        local_la_end=bpos_w35[-1, la_i] - np.array([pos_w[-1, 0], pos_w[-1, 1], 0.0]),
        local_ra_end=bpos_w35[-1, ra_i] - np.array([pos_w[-1, 0], pos_w[-1, 1], 0.0]),
        v_fwd_end=0.65,
        walk_disp_m=float(pos_w[-1, 0] - pos_w[0, 0]),
    )

    # Synthetic rotated OmniCarryBox clip starting upright 0.48m behind box along yaw=pi/4
    yaw_o = np.pi / 4.0
    t_o = np.arange(T_o, dtype=np.float64) * dt
    pos_o = np.zeros((T_o, 3), dtype=np.float64)
    pos_o[:, 0] = 0.25 * t_o * np.cos(yaw_o)
    pos_o[:, 1] = 0.25 * t_o * np.sin(yaw_o)
    pos_o[:, 2] = 0.785
    quat_o = np.tile(np.array([np.cos(0.5 * yaw_o), 0.0, 0.0, np.sin(0.5 * yaw_o)]), (T_o, 1))
    q_o = np.zeros((T_o, 29), dtype=np.float64)
    with torch.no_grad():
        fk_o35 = fk_35(
            torch.from_numpy(pos_o.astype(np.float32)),
            torch.from_numpy(quat_o.astype(np.float32)),
            torch.from_numpy(q_o.astype(np.float32)),
        )
    bpos_o35 = fk_o35.tracked_pos_w.numpy().astype(np.float64)
    obj_pos = np.repeat(np.array([[0.48 * np.cos(yaw_o), 0.48 * np.sin(yaw_o), 0.18]]), T_o, axis=0)
    obj_rot = np.repeat(np.eye(3)[None, ...], T_o, axis=0)
    contact_o = np.zeros(T_o, dtype=bool)
    contact_o[15:] = True

    out = stitch_lafan_walk_to_omni_clip(
        omni_pos=pos_o,
        omni_quat_wxyz=quat_o,
        omni_q_bfs=q_o,
        omni_bpos=bpos_o35,
        omni_obj_pos=obj_pos,
        omni_obj_rot=obj_rot,
        omni_contact=contact_o,
        box_half_extents=np.array([0.17, 0.17, 0.18]),
        segments=[seg],
        usage_counts={},
        fk_35=fk_35,
        body_names=body_names_35,
        ang_vel_fn=_ang_vel_world_from_quats,
        blend_frames=T_b,
        dt=dt,
    )
    assert out["joint_pos"].shape == (T_w + T_b + T_o, 29)
    assert out["meta"]["pre_walk_path_m"] >= 1.20
    assert not out["contact"][: T_w + T_b].any() and out["contact"][-1]
    # Check smooth velocity across join (no sudden step discontinuity)
    q_step_diff = np.abs(np.diff(out["joint_pos"], axis=0) / dt).max()
    assert q_step_diff < 5.0, f"Excessive joint speed spike at blend seam: {q_step_diff:.2f} rad/s"
    pelvis_v_diff = np.abs(np.diff(out["body_lin_vel_w"][:, 0, :], axis=0)).max()
    assert pelvis_v_diff < 0.08, f"Excessive pelvis accel jump across blend: {pelvis_v_diff:.4f} m/s/step"
    clr = _box_obb_surface_distance(
        out["body_pos_w"][: T_w + T_b, [la_i, ra_i], :], obj_pos[0], obj_rot[0], np.array([0.17, 0.17, 0.18])
    ).min()
    assert clr >= 0.035, f"Approach feet violated box clearance: {clr:.4f}m"
    assert out["meta"]["max_blend_min_foot_z_m"] <= 0.058, f"Blend feet airborne: {out['meta']}"
    assert out["meta"]["max_blend_com_margin_m"] <= 0.14, f"Blend COM margin exceeded: {out['meta']}"
    print(f"[PASS] Test 13: LAFAN1 walk stitcher SE(2) phase blend smooth (pre_walk={out['meta']['pre_walk_path_m']:.2f}m, clr={clr:.3f}m, com={out['meta']['max_blend_com_margin_m']:.3f}m)")


if __name__ == "__main__":
    print(f"Running SuperMimic x SUGAR test suite on device: {DEVICE} (PyTorch {torch.__version__})")
    test_1_rotations_and_zero_angle_gradients()
    test_2_kinematics_urdf_mjcf_and_joint_permutation()
    test_3_observations_and_all_6_task_rewards()
    test_4_world_model_se2_invariance()
    test_5_replay_buffer_push_isolation_and_history_padding()
    test_6_32_step_hoi_bptt_gradients()
    test_7_end_to_end_refiner_and_tracker_pipeline()
    test_8_review_regressions_obs_xi_com_and_rewards()
    test_9_timeout_bootstrap_resume_and_compile()
    test_10_colab_interrupt_auto_resume_and_generator_ckpt()
    test_11_fast_fk_depth_vectorization_and_scale_grad()
    test_12_omniretarget_clean_usda_and_bridge_sync()
    test_13_lafan_walk_stitcher_continuity_and_clearance()
    print("\nALL 13 SUPERMIMIC x SUGAR TESTS PASSED SUCCESSFULLY!")

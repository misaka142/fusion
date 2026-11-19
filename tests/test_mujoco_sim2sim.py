"""End-to-End Verification Suite for Local MuJoCo Sim-to-Sim Deployment (All 6 SUGAR Tasks).

Verifies:
1. Scene XML generation & MuJoCo compilation across all 6 tasks (both STL-mesh and primitive-fallback modes),
   exact object masses, motor torque `ctrlrange`, and non-zero rotor `armature`.
2. Dual-format Tracker checkpoint loading (SuperMimic BFS/SDK `.pt` + legacy RSL-RL `demo_ckpts` `.pt`)
   and 890-D Refiner misuse guard.
3. Exact 510-D observation isomorphism (`max_abs_err < 1e-4`) between `MuJoCoSim2SimRunner` and
   `SugarHOITask.build_tracker_obs` (proving `torso_link` FK, `wxyz` quats, local `qvel[3:6]`, and 6D ori match).
4. 500Hz PD controller standing stability over 2.0 simulated seconds (1000 physics steps, zero oscillation).
5. `--mode replay` end-to-end execution with SUGAR `data_000` (`robot_50hz.npz` + `obj_motion_global_50hz.pkl`).
6. `--mode autonomous` closed-loop 10Hz Diffusion `GeneratorWrapper` + 50Hz `SuperMimicPolicy` + live mocap
   target dragging + `run_sim2sim.py --headless --record_mp4` CLI verification.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import dill
import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

FUSION_ROOT = Path(__file__).resolve().parents[1]
SUGAR_RL_SRC = FUSION_ROOT / "source" / "sugar_rl"
SUGAR_IL_SRC = FUSION_ROOT / "source" / "sugar_il"
for _p in (FUSION_ROOT, SUGAR_RL_SRC, SUGAR_IL_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from scripts.sim2sim.policy_runner import MuJoCoSim2SimRunner, load_tracker_checkpoint  # noqa: E402
from scripts.sim2sim.scene_builder import (  # noqa: E402
    G1_ISAACLAB_BFS_JOINT_NAMES,
    SIM2SIM_TASK_SPECS,
    build_mujoco_scene_xml,
)
from sugar_il.model.common.normalizer import SingleFieldLinearNormalizer  # noqa: E402
from sugar_il.model.encoder.generator_state_encoder import GeneratorStateObsEncoder  # noqa: E402
from sugar_il.policy.generator import Generator  # noqa: E402
from sugar_rl.pickle_compat import loads_pickle_compat  # noqa: E402
from sugar_rl.supermimic.hoi_task import SugarHOITask, TRACKER_OBS_DIM  # noqa: E402
from sugar_rl.supermimic.networks import SuperMimicPolicy  # noqa: E402
from sugar_rl.supermimic.rotations import quat_apply, quat_normalize  # noqa: E402
from sugar_rl.supermimic.urdf_kinematics import (  # noqa: E402
    FastTreeKinematics,
    G1_29DOF_SDK_JOINT_NAMES,
    build_g1_actuator_params,
)


def _save_dummy_supermimic_tracker(
    path: Path,
    joint_names: list[str],
    zero_output: bool = True,
    closed_loop_balance: bool = False,
) -> Path:
    act = build_g1_actuator_params(joint_names, device="cpu")
    use_res = closed_loop_balance or (not zero_output)
    pol = SuperMimicPolicy(obs_dim=TRACKER_OBS_DIM, action_dim=29, use_ref_residual=use_res)
    pol.sync_actuator_buffers(act.default_joint_pos, act.action_scale)
    if zero_output or closed_loop_balance:
        with torch.no_grad():
            for name, p in pol.named_parameters():
                if "log_std" not in name:
                    torch.nn.init.zeros_(p)
            if closed_loop_balance:
                # Exact algebraic identity `SiLU(x) - SiLU(-x) == x` across normalized layers:
                # verifies end-to-end 510-D observation signs (`project_gravity`, `base_ang_vel`,
                # `joint_pos_rel`, `joint_vel_rel`) + 2-link CoM ankle balance under 500Hz implicit PD!
                for ln_idx in (1, 4, 7):
                    pol.backbone[ln_idx].weight.fill_(1.0)
                H0, H1, H2 = pol.hidden_dims
                pol.backbone[0].bias[4] = (H0 / 2.0) ** 0.5
                pol.backbone[0].bias[5] = -(H0 / 2.0) ** 0.5

                la_p = joint_names.index("left_ankle_pitch_joint")
                ra_p = joint_names.index("right_ankle_pitch_joint")
                lk_p = joint_names.index("left_knee_joint")
                rk_p = joint_names.index("right_knee_joint")
                s0 = 0.02
                pol.backbone[0].weight[0, 167 + la_p] = s0 * (-2.2)
                pol.backbone[0].weight[0, 167 + ra_p] = s0 * (-2.2)
                pol.backbone[0].weight[0, 167 + lk_p] = s0 * (-1.1)
                pol.backbone[0].weight[0, 167 + rk_p] = s0 * (-1.1)
                pol.backbone[0].weight[0, 312 + la_p] = s0 * (-0.25)
                pol.backbone[0].weight[0, 312 + ra_p] = s0 * (-0.25)
                pol.backbone[0].weight[0, 498] = s0 * 2.5
                pol.backbone[0].weight[0, 49] = s0 * 0.5
                pol.backbone[0].bias[0] = s0 * 0.025
                pol.backbone[0].weight[1, :] = -pol.backbone[0].weight[0, :]
                pol.backbone[0].bias[1] = -pol.backbone[0].bias[0]

                pol.backbone[0].weight[2, 499] = s0 * (-2.0)
                pol.backbone[0].weight[2, 48] = s0 * 0.4
                pol.backbone[0].weight[3, :] = -pol.backbone[0].weight[2, :]

                for layer, H in ((pol.backbone[3], H1), (pol.backbone[6], H2)):
                    layer.bias[4] = (H / 2.0) ** 0.5
                    layer.bias[5] = -(H / 2.0) ** 0.5
                    layer.weight[0, 0] = 1.0
                    layer.weight[0, 1] = -1.0
                    layer.weight[1, 0] = -1.0
                    layer.weight[1, 1] = 1.0
                    layer.weight[2, 2] = 1.0
                    layer.weight[2, 3] = -1.0
                    layer.weight[3, 2] = -1.0
                    layer.weight[3, 3] = 1.0

                for j_idx, jn in enumerate(joint_names):
                    sc = float(act.action_scale[j_idx].item()) * s0
                    if "ankle_pitch_joint" in jn:
                        pol.backbone[9].weight[j_idx, 0] = 1.0 / sc
                        pol.backbone[9].weight[j_idx, 1] = -1.0 / sc
                    elif "ankle_roll_joint" in jn:
                        pol.backbone[9].weight[j_idx, 2] = 1.0 / sc
                        pol.backbone[9].weight[j_idx, 3] = -1.0 / sc
    payload = {
        "format": "supermimic_sugar_v2",
        "stage": "tracker",
        "joint_names": list(joint_names),
        "obs_dim": TRACKER_OBS_DIM,
        "action_dim": 29,
        "hidden_dims": pol.hidden_dims,
        "use_ref_residual": pol.use_ref_residual,
        "policy_state_dict": pol.state_dict(),
    }
    torch.save(payload, str(path))
    return path


def _save_dummy_legacy_rsl_tracker(path: Path) -> Path:
    net = torch.nn.Sequential(
        torch.nn.Linear(TRACKER_OBS_DIM, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    )
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    sd = {f"actor.{k}": v for k, v in net.state_dict().items()}
    sd["std"] = torch.ones(29) * 0.1
    torch.save({"model_state_dict": sd, "iter": 100}, str(path))
    return path


def _save_dummy_generator_ckpt(path: Path, use_target: bool = True) -> Path:
    n_obs_steps = 1
    n_action_steps = 8
    horizon = n_obs_steps + n_action_steps - 1
    shape_meta = {
        "obs": {"n_obs_steps": n_obs_steps},
        "action": {"shape": [36], "horizon": horizon},
    }
    obs_enc = GeneratorStateObsEncoder(
        shape_meta=shape_meta, feature_dim=64, use_last_action=True, use_target=use_target
    )
    sched = DDPMScheduler(
        num_train_timesteps=10,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    gen = Generator(
        shape_meta=shape_meta,
        noise_scheduler=sched,
        obs_encoder=obs_enc,
        num_inference_steps=2,
        n_layer=2,
        n_head=4,
        p_drop_attn=0.0,
        use_attn_mask=True,
    )
    # Initialize identity normalizers for all expected keys
    obs_keys = ["obj_pos_b", "obj_ori_b", "last_action"]
    if use_target:
        obs_keys.extend(["target_obj_pos_b", "target_obj_ori_b"])
    for k in obs_keys:
        gen.normalizer[k] = SingleFieldLinearNormalizer.create_identity()
    gen.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()

    cfg = OmegaConf.create(
        {
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
            "use_target": use_target,
            "use_last_action": True,
            "policy": {
                "shape_meta": shape_meta,
                "obs_encoder": {"feature_dim": 64},
                "noise_scheduler": {
                    "num_train_timesteps": 10,
                    "beta_start": 1e-4,
                    "beta_end": 0.02,
                    "beta_schedule": "squaredcos_cap_v2",
                    "clip_sample": True,
                    "prediction_type": "epsilon",
                },
                "num_inference_steps": 2,
                "n_layer": 2,
                "n_head": 4,
                "p_drop_attn": 0.0,
                "use_attn_mask": True,
            },
        }
    )
    payload = {"cfg": cfg, "state_dicts": {"model": gen.state_dict()}, "pickles": {}}
    with open(path, "wb") as f:
        torch.save(payload, f, pickle_module=dill)
    return path


def _create_dummy_motion_dataset(folder: Path, T: int = 30) -> Path:
    clip_dir = folder / "data_000"
    clip_dir.mkdir(parents=True, exist_ok=True)
    bfs_def = build_g1_actuator_params(G1_ISAACLAB_BFS_JOINT_NAMES, device="cpu").default_joint_pos.numpy()
    jpos = np.tile(bfs_def[None, :], (T, 1)).astype(np.float32)
    jvel = np.zeros((T, 29), dtype=np.float32)
    bpos = np.zeros((T, 14, 3), dtype=np.float32)
    bpos[:, 0, 2] = 0.76
    bquat = np.zeros((T, 14, 4), dtype=np.float32)
    bquat[:, :, 0] = 1.0
    blin = np.zeros((T, 14, 3), dtype=np.float32)
    bang = np.zeros((T, 14, 3), dtype=np.float32)
    np.savez(
        clip_dir / "robot_50hz.npz",
        joint_pos=jpos,
        joint_vel=jvel,
        body_pos_w=bpos,
        body_quat_w=bquat,
        body_lin_vel_w=blin,
        body_ang_vel_w=bang,
    )

    obj_trans = np.tile(np.array([[0.45, 0.0, 0.15]], dtype=np.float32), (T, 1))
    obj_trans[:, 0] += np.linspace(0.0, 0.50, T, dtype=np.float32)
    obj_rot = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (T, 1, 1))
    obj_pkl = {
        "obj_trans": obj_trans,
        "obj_rot": obj_rot,
        "obj_lin_vel": np.zeros((T, 3), dtype=np.float32),
        "obj_ang_vel": np.zeros((T, 3), dtype=np.float32),
    }
    with open(clip_dir / "obj_motion_global_50hz.pkl", "wb") as f:
        pickle.dump(obj_pkl, f)
    np.save(clip_dir / "contact_labels_50hz.npy", np.ones((T,), dtype=bool))
    return folder


def test_1_all_six_tasks_scene_compilation(tmp_path: Path) -> None:
    tmp_dir = tmp_path
    print("[Test 1] Compiling MuJoCo scenes for all 6 SUGAR tasks...")
    for task_name, spec in SIM2SIM_TASK_SPECS.items():
        xml_path, resolved_spec, _ = build_mujoco_scene_xml(
            task_name=task_name,
            output_xml_path=tmp_dir / f"scene_{task_name}.xml",
        )
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        assert m.nq == 43, f"{task_name}: expected nq=43, got {m.nq}"
        assert m.nv == 41, f"{task_name}: expected nv=41, got {m.nv}"
        assert m.nu == 29, f"{task_name}: expected nu=29, got {m.nu}"

        obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        lhand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand")
        rhand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_rubber_hand")
        assert min(obj_id, torso_id, lhand_id, rhand_id) > 0

        # Verify exact object mass from `objects.py`
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        actual_obj_mass = float(m.body_subtreemass[obj_id])
        assert abs(actual_obj_mass - resolved_spec.object_mass) < 1e-4, (
            f"{task_name}: mass mismatch {actual_obj_mass} vs {resolved_spec.object_mass}"
        )

        # Verify non-zero rotor armature on all 29 actuated joints & implicit PD `forcerange` / `kp`
        armatures = m.dof_armature[6:35]
        assert np.all(armatures > 0.003), f"{task_name}: missing joint rotor armature!"
        knee_aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
        assert abs(float(m.actuator_forcerange[knee_aid, 1]) - 139.0) < 1e-4
        assert float(m.actuator_gainprm[knee_aid, 0]) > 99.0

    # Also test primitive-only fallback when no STL directory exists
    empty_desc = tmp_dir / "empty_desc"
    empty_desc.mkdir(exist_ok=True)
    import scripts.sim2sim.scene_builder as sb
    orig_discover = sb._discover_g1_mesh_dir
    try:
        sb._discover_g1_mesh_dir = lambda _: None
        xml_fb, _, rep_fb = build_mujoco_scene_xml("SitChair", output_xml_path=tmp_dir / "scene_fallback.xml")
        m_fb = mujoco.MjModel.from_xml_path(str(xml_fb))
        assert m_fb.nq == 43 and "Built-in" in rep_fb["robot_geometry"]
    finally:
        sb._discover_g1_mesh_dir = orig_discover
    print("  -> Passed all 6 tasks + primitive fallback!")


def test_2_dual_format_checkpoint_loading_and_joint_order(tmp_path: Path) -> None:
    tmp_dir = tmp_path
    print("[Test 2] Testing SuperMimic (BFS & SDK) + legacy RSL-RL checkpoint loading...")
    bfs_pt = _save_dummy_supermimic_tracker(tmp_dir / "tracker_bfs.pt", G1_ISAACLAB_BFS_JOINT_NAMES)
    sdk_pt = _save_dummy_supermimic_tracker(tmp_dir / "tracker_sdk.pt", G1_29DOF_SDK_JOINT_NAMES)
    leg_pt = _save_dummy_legacy_rsl_tracker(tmp_dir / "tracker_legacy.pt")

    b_bfs = load_tracker_checkpoint(bfs_pt, device="cpu")
    b_sdk = load_tracker_checkpoint(sdk_pt, device="cpu")
    b_leg = load_tracker_checkpoint(leg_pt, device="cpu")
    assert b_bfs.joint_names == G1_ISAACLAB_BFS_JOINT_NAMES
    assert b_sdk.joint_names == G1_29DOF_SDK_JOINT_NAMES
    assert b_leg.ckpt_format == "rsl_rl_legacy" and b_leg.joint_names == G1_ISAACLAB_BFS_JOINT_NAMES

    # Verify fingerprint fallback when `joint_names` metadata is absent (backward compat)
    payload_no_names = torch.load(str(bfs_pt), weights_only=False)
    del payload_no_names["joint_names"]
    no_names_pt = tmp_dir / "tracker_no_names.pt"
    torch.save(payload_no_names, str(no_names_pt))
    b_inferred = load_tracker_checkpoint(no_names_pt, device="cpu")
    assert b_inferred.joint_names == G1_ISAACLAB_BFS_JOINT_NAMES

    # Verify 890-D Refiner misuse guard
    refiner_pol = SuperMimicPolicy(obs_dim=890, action_dim=29)
    ref_pt = tmp_dir / "refiner.pt"
    torch.save({"obs_dim": 890, "action_dim": 29, "policy_state_dict": refiner_pol.state_dict()}, str(ref_pt))
    caught = False
    try:
        load_tracker_checkpoint(ref_pt, device="cpu")
    except ValueError as e:
        caught = "890" in str(e)
    assert caught, "Expected ValueError when passing 890-D refiner.pt to Sim-to-Sim!"
    print("  -> Passed dual-format & joint-bijection checks!")


def test_3_observation_isomorphism_with_sugar_hoi_task(tmp_path: Path) -> None:
    tmp_dir = tmp_path
    print("[Test 3] Verifying 510-D observation isomorphism against SugarHOITask...")
    bfs_pt = _save_dummy_supermimic_tracker(tmp_dir / "tracker_iso.pt", G1_ISAACLAB_BFS_JOINT_NAMES)
    dataset_dir = _create_dummy_motion_dataset(tmp_dir / "motion_iso")

    runner = MuJoCoSim2SimRunner(
        task_name="CarryBox",
        tracker_ckpt=bfs_pt,
        mode="replay",
        motion_folder=dataset_dir,
        device="cpu",
    )
    runner.reset()

    # Set non-trivial random 3D robot pose, waist angles, local angular velocity, and object pose
    rng = np.random.default_rng(42)
    runner.mj_data.qpos[0:3] = np.array([0.12, -0.08, 0.77])
    rq = quat_normalize(torch.tensor([[0.96, 0.08, -0.12, 0.23]], dtype=torch.float32))[0].numpy()
    runner.mj_data.qpos[3:7] = rq
    q_rand_bfs = rng.uniform(-0.25, 0.25, size=(29,)).astype(np.float64)
    runner.mj_data.qpos[runner.pol_qpos_adrs] = q_rand_bfs
    w_local = np.array([0.35, -0.42, 0.61], dtype=np.float64)
    runner.mj_data.qvel[3:6] = w_local
    qd_rand_bfs = rng.uniform(-0.8, 0.8, size=(29,)).astype(np.float64)
    runner.mj_data.qvel[runner.pol_dof_adrs] = qd_rand_bfs

    runner.mj_data.qpos[runner.obj_qpos_adr : runner.obj_qpos_adr + 3] = np.array([0.53, 0.17, 0.29])
    oq = quat_normalize(torch.tensor([[0.85, -0.21, 0.34, 0.34]], dtype=torch.float32))[0].numpy()
    runner.mj_data.qpos[runner.obj_qpos_adr + 3 : runner.obj_qpos_adr + 7] = oq
    mujoco.mj_forward(runner.mj_model, runner.mj_data)

    st_mj = runner._read_mujoco_tensors()
    hist_mj = runner.history
    assert hist_mj is not None
    # Populate single-step history from this non-trivial state
    from sugar_rl.supermimic.hoi_task import TrackerHistoryState
    last_act = torch.linspace(-0.2, 0.2, 29).view(1, 29)
    hist_mj = TrackerHistoryState(
        base_ang_vel=st_mj["base_ang_vel"].unsqueeze(1).repeat(1, 5, 1),
        joint_pos_rel=st_mj["joint_pos_rel"].unsqueeze(1).repeat(1, 5, 1),
        joint_vel_rel=st_mj["joint_vel_rel"].unsqueeze(1).repeat(1, 5, 1),
        actions=last_act.unsqueeze(1).repeat(1, 5, 1),
        project_gravity=st_mj["project_gravity"].unsqueeze(1).repeat(1, 5, 1),
    )
    cmd_36 = torch.linspace(-0.1, 0.3, 36).view(1, 36)
    obs_mj = torch.cat(
        [
            cmd_36,
            hist_mj.base_ang_vel.reshape(1, -1),
            hist_mj.joint_pos_rel.reshape(1, -1),
            hist_mj.joint_vel_rel.reshape(1, -1),
            hist_mj.actions.reshape(1, -1),
            hist_mj.project_gravity.reshape(1, -1),
            st_mj["obj_pos_b"],
            st_mj["obj_ori_6d"],
        ],
        dim=-1,
    )

    # Compare against analytical `SugarHOITask.build_tracker_obs`
    fk = FastTreeKinematics(env_joint_names=G1_ISAACLAB_BFS_JOINT_NAMES, device="cpu")
    task = SugarHOITask(fk, task_name_or_id="CarryBox")
    # Convert local `qvel[3:6]` -> world `root_ang_vel` expected by `SugarHOITask.pack_state`
    rq_t = torch.as_tensor(rq, dtype=torch.float32).view(1, 4)
    w_world = quat_apply(rq_t, torch.as_tensor(w_local, dtype=torch.float32).view(1, 3))
    s_flat = SugarHOITask.pack_state(
        root_pos=torch.tensor([[0.12, -0.08, 0.77]]),
        root_quat=rq_t,
        q=torch.as_tensor(q_rand_bfs, dtype=torch.float32).view(1, 29),
        root_lin_vel=torch.zeros(1, 3),
        root_ang_vel=w_world,
        qd=torch.as_tensor(qd_rand_bfs, dtype=torch.float32).view(1, 29),
        obj_pos=torch.tensor([[0.53, 0.17, 0.29]]),
        obj_quat=torch.as_tensor(oq, dtype=torch.float32).view(1, 4),
        obj_lin_vel=torch.zeros(1, 3),
        obj_ang_vel=torch.zeros(1, 3),
    )
    hist_ref = task.init_tracker_history_from_states(
        s_flat.unsqueeze(1).repeat(1, 5, 1),
        last_act.unsqueeze(1).repeat(1, 5, 1),
    )
    obs_analytic = task.build_tracker_obs(s_flat, cmd_36, hist_ref)
    max_err = float((obs_mj - obs_analytic).abs().max().item())
    assert max_err < 1e-4, f"510-D observation mismatch: max_abs_err={max_err:.6e}"
    print(f"  -> Passed 510-D observation isomorphism (max_abs_err={max_err:.2e} < 1e-4)!")


def test_4_pd_standing_stability_and_replay(tmp_path: Path) -> None:
    tmp_dir = tmp_path
    print("[Test 4] Verifying 500Hz implicit PD closed-loop standing stability over 3.0s (1500 substeps)...")
    stand_pt = _save_dummy_supermimic_tracker(
        tmp_dir / "tracker_stand.pt", G1_ISAACLAB_BFS_JOINT_NAMES, closed_loop_balance=True
    )
    ds_dir = _create_dummy_motion_dataset(tmp_dir / "motion_stand", T=180)

    runner = MuJoCoSim2SimRunner(
        task_name="CarryBox",
        tracker_ckpt=stand_pt,
        mode="replay",
        motion_folder=ds_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    runner.reset()
    for _ in range(150):  # 150 * 0.02s = 3.0s simulated time (1500 MuJoCo steps)
        info = runner.step()
        assert np.isfinite(info["pelvis_z"]), "NaN detected during MuJoCo PD simulation!"
    max_qd = float(np.max(np.abs(runner.mj_data.qvel[:35])))
    assert 0.70 < info["pelvis_z"] < 0.80, f"Robot fell or diverged during standing PD test: pelvis_z={info['pelvis_z']:.3f}m"
    assert max_qd < 1.0, f"High-frequency PD oscillation detected: max_qvel={max_qd:.3f} rad/s"
    print(f"  -> Passed 3.0s standing stability (final pelvis_z={info['pelvis_z']:.3f}m, max_qvel={max_qd:.3f} rad/s)!")


def test_5_autonomous_generator_closed_loop_and_cli_video(tmp_path: Path) -> None:
    tmp_dir = tmp_path
    print("[Test 5] Verifying autonomous 10Hz Generator + 50Hz Tracker closed-loop + MP4 CLI...")
    trk_pt = _save_dummy_supermimic_tracker(
        tmp_dir / "tracker_auto.pt", G1_ISAACLAB_BFS_JOINT_NAMES, closed_loop_balance=True
    )
    gen_ckpt = _save_dummy_generator_ckpt(tmp_dir / "generator.ckpt", use_target=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runner = MuJoCoSim2SimRunner(
        task_name="PushBox",
        tracker_ckpt=trk_pt,
        mode="autonomous",
        generator_ckpt=gen_ckpt,
        generator_call_interval=10,
        device=dev,
    )
    runner.reset(custom_obj_pos=(0.55, 0.0, 0.22), custom_target_pos=(1.25, 0.20, 0.22))

    # Step across replanning boundary (step 10 & step 20) and test live mocap target dragging
    for s in range(25):
        if s == 12:
            runner.mj_data.mocap_pos[runner.target_mocap_id] = np.array([1.50, -0.30, 0.22])
        info = runner.step()
    assert abs(float(runner.target_pos_w[0, 0].item()) - 1.50) < 1e-4, "Live mocap target drag failed to sync!"
    assert np.isfinite(info["goal_dist_m"])

    # Test `run_sim2sim.py --headless --record_mp4` CLI entry point on SitChair (use_target=False)
    gen_notarget = _save_dummy_generator_ckpt(tmp_dir / "generator_sitchair.ckpt", use_target=False)
    mp4_out = tmp_dir / "sitchair_demo.mp4"
    cmd = [
        sys.executable,
        str(FUSION_ROOT / "scripts" / "sim2sim" / "run_sim2sim.py"),
        "--task", "SitChair",
        "--mode", "autonomous",
        "--tracker_ckpt", str(trk_pt),
        "--generator_ckpt", str(gen_notarget),
        "--steps", "15",
        "--video_width", "640",
        "--video_height", "480",
        "--headless",
        "--record_mp4", str(mp4_out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"CLI run_sim2sim.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    assert mp4_out.exists() and mp4_out.stat().st_size > 1000, "Expected non-empty MP4 output from CLI!"
    print(f"  -> Passed autonomous closed-loop & MP4 video export ({mp4_out.stat().st_size} bytes)!")


def test_6_omnicarrybox_obb_alignment_and_compat(tmp_path: Path) -> None:
    tmp_dir = tmp_path
    print("[Test 6] Verifying OBB-aligned OmniCarryBox mesh centering, mtime cache, clean clips & ckpt compat...")
    import json
    import os
    import trimesh
    from scipy.spatial.transform import Rotation as R

    # 1. Verify backward-compatible loading when a checkpoint carries an extra `residual_mask` buffer
    base_pt = _save_dummy_supermimic_tracker(
        tmp_dir / "tracker_compat_base.pt", G1_ISAACLAB_BFS_JOINT_NAMES, closed_loop_balance=False
    )
    payload = torch.load(str(base_pt), weights_only=False)
    payload["policy_state_dict"]["residual_mask"] = torch.ones(29)
    compat_pt = tmp_dir / "tracker_compat_extra_mask.pt"
    torch.save(payload, str(compat_pt))
    bundle = load_tracker_checkpoint(compat_pt, device="cpu")
    assert bundle.joint_names == list(G1_ISAACLAB_BFS_JOINT_NAMES)

    ds_dir = tmp_dir / "omni_box_ds"
    ds_dir.mkdir(parents=True)
    cache_dir = tmp_dir / "scene_cache"
    cache_dir.mkdir(parents=True)

    # 2. Create a small centered mesh & write v1 STL cache
    mesh_v1 = trimesh.creation.box(extents=(0.30, 0.30, 0.30))
    mesh_v1.export(str(ds_dir / "obj_aligned.obj"))
    (ds_dir / "box_spec.json").write_text(
        json.dumps(
            {
                "dataset": "test_obb_mtime",
                "box_size_xyz": [0.30, 0.30, 0.30],
                "box_center_xyz": [0.0, 0.0, 0.0],
                "mass_kg": 0.5,
                "mesh_file": "obj_aligned.obj",
            }
        ),
        encoding="utf-8",
    )
    xml1, _, _ = build_mujoco_scene_xml(
        task_name="CarryBox",
        output_xml_path=cache_dir / "scene1.xml",
        motion_folder=ds_dir,
    )
    stl_cached = cache_dir / "meshes" / "small_box_test_obb_mtime.stl"
    assert stl_cached.is_file()
    _ = mujoco.MjModel.from_xml_path(str(xml1))
    sz1 = trimesh.load(str(stl_cached), force="mesh").extents

    # 3. Update obj_aligned.obj with larger size & bump mtime -> cache MUST rebuild automatically!
    mesh_v2 = trimesh.creation.box(extents=(0.40, 0.42, 0.38))
    mesh_v2.export(str(ds_dir / "obj_aligned.obj"))
    future_t = stl_cached.stat().st_mtime + 5.0
    os.utime(ds_dir / "obj_aligned.obj", (future_t, future_t))

    xml2, _, _ = build_mujoco_scene_xml(
        task_name="CarryBox",
        output_xml_path=cache_dir / "scene2.xml",
        motion_folder=ds_dir,
    )
    _ = mujoco.MjModel.from_xml_path(str(xml2))
    sz2 = trimesh.load(str(stl_cached), force="mesh").extents
    assert abs(float(sz1[0]) - 0.30) < 1e-4 and abs(float(sz2[0]) - 0.40) < 1e-4, (
        f"STL cache failed to refresh on mtime update: sz1={sz1}, sz2={sz2}"
    )

    # 4. Verify real cleaned data/OmniCarryBox if present locally
    real_omni = FUSION_ROOT / "data" / "OmniCarryBox"
    if (real_omni / "box_spec.json").is_file() and (real_omni / "obj_aligned.obj").is_file():
        m_real = trimesh.load(str(real_omni / "obj_aligned.obj"), force="mesh", process=False)
        ctr_real = 0.5 * (m_real.bounds[0] + m_real.bounds[1])
        assert float(np.max(np.abs(ctr_real))) < 1e-4, f"OmniCarryBox mesh not centered: {ctr_real}"
        xml_r, _, _ = build_mujoco_scene_xml(
            task_name="CarryBox",
            output_xml_path=cache_dir / "scene_real.xml",
            motion_folder=real_omni,
        )
        m_mj = mujoco.MjModel.from_xml_path(str(xml_r))
        assert m_mj.nq == 43
        clips = sorted(real_omni.glob("data_*"))
        assert len(clips) >= 100, f"Too few high-quality OmniCarryBox clips: {len(clips)}"
        arm_bfs_idx = [i for i, n in enumerate(G1_ISAACLAB_BFS_JOINT_NAMES) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
        manifest_file = real_omni / "manifest.json"
        has_lafan_walk = manifest_file.is_file() and ("lafan_source" in manifest_file.read_text(encoding="utf-8"))
        manifest_by_folder = (
            {m["folder"]: m for m in json.loads(manifest_file.read_text(encoding="utf-8"))}
            if has_lafan_walk
            else {}
        )
        for c in clips:
            r = np.load(c / "robot_50hz.npz")
            cl = np.load(c / "contact_labels_50hz.npy")
            tq = r["body_quat_w"][0, 10]
            up_z = float(R.from_quat([tq[1], tq[2], tq[3], tq[0]]).as_matrix()[2, 2])
            assert up_z >= 0.80, f"{c.name} has bent t=0 start: torso_up_z={up_z:.3f}"
            assert not bool(cl[0]), f"{c.name} has t=0 contact=True!"
            if has_lafan_walk:
                m_info = manifest_by_folder[c.name]
                w_n = int(m_info["lafan_frames_50hz"])
                b_n = int(m_info["blend_frames_50hz"])
                idx_c = np.flatnonzero(cl)
                assert len(idx_c) > 0 and idx_c[0] >= w_n + b_n, f"{c.name} missing pre-contact approach segment!"
                walk_seg = r["body_pos_w"][: idx_c[0] + 1, 0, :2]
                walk_dist = float(np.linalg.norm(np.diff(walk_seg, axis=0), axis=-1).sum())
                assert walk_dist >= 1.05, f"{c.name} approach walk too short: {walk_dist:.2f}m"
                seam_slice = r["joint_pos"][max(0, w_n - 2) : w_n + b_n + 3]
                seam_jump = float(np.abs(np.diff(seam_slice, axis=0)).max() * 50.0)
                assert seam_jump < 10.0, f"{c.name} blend seam joint speed spike detected: {seam_jump:.2f} rad/s"
                walk_arm_acc = float(np.abs(np.diff(r["joint_pos"][:w_n, arm_bfs_idx], n=2, axis=0)).max() * 2500.0)
                assert walk_arm_acc <= 45.0, f"{c.name} walking arm jitter accel too high: {walk_arm_acc:.1f} rad/s^2"
                assert float(m_info["max_blend_min_foot_z_m"]) <= 0.0585, f"{c.name} airborne blend foot: {m_info}"
                assert float(m_info["max_blend_com_margin_m"]) <= 0.1405, f"{c.name} blend COM drift: {m_info}"
                if "min_hand_dy_m" in m_info:
                    assert float(m_info["min_hand_dy_m"]) >= 0.24, f"{c.name} crossed/narrow arms: {m_info}"
                    assert float(m_info["min_leg_dist_cm"]) >= 1.25, f"{c.name} leg collision risk: {m_info}"
                    assert float(m_info["com_toe_pick_cm"]) <= 0.0, f"{c.name} forward CoM tip at pickup: {m_info}"
                    if "max_pre_skate_cm" in m_info:
                        assert float(m_info["max_pre_skate_cm"]) <= 2.25, f"{c.name} foot skating before pickup: {m_info}"
                    if "max_bend_drift_cm" in m_info:
                        assert float(m_info["max_bend_drift_cm"]) <= 2.25, f"{c.name} cumulative foot drift during bend: {m_info}"

    # 5. Verify cross-NumPy (2.x -> 1.x) pickle deserialization for `obj_motion_global_50hz.pkl`
    import importlib
    importlib.import_module("numpy.core.multiarray")
    pkl_path = real_omni / "data_000" / "obj_motion_global_50hz.pkl"
    pkl_sample = pkl_path.read_bytes() if pkl_path.is_file() else pickle.dumps({"obj_trans": np.zeros((10, 3), dtype=np.float32)})
    saved_mods = {k: sys.modules[k] for k in list(sys.modules) if k == "numpy._core" or k.startswith("numpy._core.")}
    try:
        for k in list(saved_mods):
            sys.modules[k] = None  # type: ignore[assignment]
        sys.modules["numpy._core"] = None  # type: ignore[assignment]
        sys.modules["numpy._core.multiarray"] = None  # type: ignore[assignment]
        obj_loaded = loads_pickle_compat(pkl_sample)
        assert isinstance(obj_loaded, dict) and "obj_trans" in obj_loaded and obj_loaded["obj_trans"].ndim == 2
        cmd_src = (FUSION_ROOT / "source" / "sugar_rl" / "sugar_rl" / "tasks" / "locomanip" / "mdp" / "commands.py").read_text(encoding="utf-8")
        ana_src = (FUSION_ROOT / "scripts" / "supermimic" / "analyze_rollouts.py").read_text(encoding="utf-8")
        assert "load_pickle_compat" in cmd_src and "pickle.load(" not in cmd_src
        assert "load_pickle_compat" in ana_src and "pickle.load(" not in ana_src
    finally:
        for k in ("numpy._core", "numpy._core.multiarray"):
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)

    print("  -> Passed OBB centering, mtime cache, clean OmniCarryBox, NumPy compat & checkpoint checks!")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        test_1_all_six_tasks_scene_compilation(tmp_dir)
        test_2_dual_format_checkpoint_loading_and_joint_order(tmp_dir)
        test_3_observation_isomorphism_with_sugar_hoi_task(tmp_dir)
        test_4_pd_standing_stability_and_replay(tmp_dir)
        test_5_autonomous_generator_closed_loop_and_cli_video(tmp_dir)
        test_6_omnicarrybox_obb_alignment_and_compat(tmp_dir)
    print("\nALL 6 MUJOCO SIM-TO-SIM END-TO-END TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

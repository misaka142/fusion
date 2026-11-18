"""MuJoCo MJCF Scene Builder for all 6 SUGAR Whole-Body Loco-Manipulation Tasks.

Constructs a physics-aligned MuJoCo scene containing:
1. Unitree G1 29-DoF humanoid with floating base (`pelvis`), `torso_link` anchor,
   `left_rubber_hand` & `right_rubber_hand` high-friction collision geometry,
   exact CAD link inertias, SUGAR joint ranges, rotor `armature`, and 29 torque `<motor>` actuators.
2. Free-floating interactive object (`small_box`, `big_box`, `bottle`, `chair`) matching
   `sugar_rl/assets/objects/objects.py` across all 6 tasks:
   `CarryBox`, `PushBox`, `KickBox`, `PickBottle`, `StandBottle`, `SitChair`.
3. Semi-transparent green mocap body `target_vis` for visualizing (and interactively dragging
   in `mujoco.viewer` via Ctrl + Mouse!) the target object pose `target_obj_pos_w` / `target_obj_quat_w`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np

# Ensure local pure-PyTorch `sugar_rl.supermimic` is importable without installing IsaacLab
_FUSION_ROOT = Path(__file__).resolve().parents[2]
_SUGAR_RL_SRC = _FUSION_ROOT / "source" / "sugar_rl"
if str(_SUGAR_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_SUGAR_RL_SRC))

from sugar_rl.supermimic.urdf_kinematics import (  # noqa: E402
    G1_29DOF_SDK_JOINT_NAMES,
    _BUILTIN_G1_29DOF_TREE,
)


def _compute_isaaclab_bfs_joint_names() -> list[str]:
    """Compute exact PhysX / IsaacLab breadth-first joint ordering on G1 29-DoF kinematic tree."""
    children: dict[str | None, list[tuple]] = {}
    for row in _BUILTIN_G1_29DOF_TREE:
        children.setdefault(row[0], []).append(row)

    bfs_joints: list[str] = []
    q: deque[str | None] = deque([None])
    while q:
        parent = q.popleft()
        for row in children.get(parent, []):
            link_name, joint_name = row[1], row[2]
            if joint_name is not None:
                bfs_joints.append(joint_name)
            q.append(link_name)
    assert len(bfs_joints) == 29, f"Expected 29 BFS joints, got {len(bfs_joints)}"
    return bfs_joints


# Canonical 29-DoF joint orderings:
# - `G1_ISAACLAB_BFS_JOINT_NAMES`: IsaacLab PhysX articulation order (default in SUGAR datasets & checkpoints)
# - `G1_29DOF_SDK_JOINT_NAMES`: Depth-first tree / Unitree SDK order (default in MuJoCo `qpos[7:36]`)
G1_ISAACLAB_BFS_JOINT_NAMES: list[str] = _compute_isaaclab_bfs_joint_names()


# -----------------------------------------------------------------------------
# Exact G1 29-DoF CAD link inertias & motor rotor armature (`unitree.py`)
# -----------------------------------------------------------------------------

# (mass, com_pos_xyz, com_quat_wxyz, diaginertia_xyz) from Unitree G1 rev_1_0 CAD
_G1_CAD_INERTIALS: dict[str, tuple[float, tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]] = {
    "pelvis": (3.813, (0.0, 0.0, -0.07605), (1.0, 0.0, -0.000399148, 0.0), (0.010549, 0.0093089, 0.0079184)),
    "left_hip_pitch_link": (1.35, (0.002741, 0.047791, -0.02606), (0.954862, 0.293964, 0.0302556, 0.030122), (0.00181517, 0.00153422, 0.00116212)),
    "left_hip_roll_link": (1.52, (0.029812, -0.001045, -0.087934), (0.977808, -1.97119e-05, 0.205576, -0.0403793), (0.00254986, 0.00241169, 0.00148755)),
    "left_hip_yaw_link": (1.702, (-0.057709, -0.010981, -0.15078), (0.600598, 0.15832, 0.223482, 0.751181), (0.00776166, 0.00717575, 0.00160139)),
    "left_knee_link": (1.932, (0.005457, 0.003964, -0.12074), (0.923418, -0.0327699, 0.0158246, 0.382067), (0.0113804, 0.0112778, 0.00146458)),
    "left_ankle_pitch_link": (0.074, (-0.007269, 0.0, 0.011137), (0.603053, 0.369225, 0.369225, 0.603053), (1.89e-05, 1.40805e-05, 6.9195e-06)),
    "left_ankle_roll_link": (0.608, (0.026505, 0.0, -0.016425), (-0.000481092, 0.728482, -0.000618967, 0.685065), (0.00167218, 0.0016161, 0.000217621)),
    "right_hip_pitch_link": (1.35, (0.002741, -0.047791, -0.02606), (0.954862, -0.293964, 0.0302556, -0.030122), (0.00181517, 0.00153422, 0.00116212)),
    "right_hip_roll_link": (1.52, (0.029812, 0.001045, -0.087934), (0.977808, 1.97119e-05, 0.205576, 0.0403793), (0.00254986, 0.00241169, 0.00148755)),
    "right_hip_yaw_link": (1.702, (-0.057709, 0.010981, -0.15078), (0.751181, 0.223482, 0.15832, 0.600598), (0.00776166, 0.00717575, 0.00160139)),
    "right_knee_link": (1.932, (0.005457, -0.003964, -0.12074), (0.923439, 0.0345276, 0.0116333, -0.382012), (0.011374, 0.0112843, 0.00146452)),
    "right_ankle_pitch_link": (0.074, (-0.007269, 0.0, 0.011137), (0.603053, 0.369225, 0.369225, 0.603053), (1.89e-05, 1.40805e-05, 6.9195e-06)),
    "right_ankle_roll_link": (0.608, (0.026505, 0.0, -0.016425), (0.000481092, 0.728482, 0.000618967, 0.685065), (0.00167218, 0.0016161, 0.000217621)),
    "waist_yaw_link": (0.214, (0.003494, 0.000233, 0.018034), (0.289697, 0.591001, -0.337795, 0.672821), (0.000163531, 0.000107714, 0.000102205)),
    "waist_roll_link": (0.086, (0.0, 2.3e-05, 0.0), (0.5, 0.5, -0.5, 0.5), (8.245e-06, 7.079e-06, 6.339e-06)),
    "torso_link": (7.818, (0.00203158, 0.000339683, 0.184568), (0.999803, -6.03319e-05, 0.0198256, 0.00131986), (0.121847, 0.109825, 0.0273735)),
    "head_link": (1.036, (0.0001, 0.0001, 0.0001), (1.0, 0.0, 0.0, 0.0), (0.004, 0.004, 0.002)),
    "left_shoulder_pitch_link": (0.718, (0.0, 0.035892, -0.011628), (0.654152, 0.0130458, -0.326267, 0.68225), (0.000465864, 0.000432842, 0.000406394)),
    "left_shoulder_roll_link": (0.643, (-0.000227, 0.00727, -0.063243), (0.701256, -0.0196223, -0.00710317, 0.712604), (0.000691311, 0.000618011, 0.000388977)),
    "left_shoulder_yaw_link": (0.734, (0.010773, -0.002949, -0.072009), (0.716879, -0.0964829, -0.0679942, 0.687134), (0.00106187, 0.00103217, 0.000400661)),
    "left_elbow_link": (0.6, (0.064956, 0.004454, -0.010062), (0.541765, 0.636132, 0.388821, 0.388129), (0.000443035, 0.000421612, 0.000259353)),
    "left_wrist_roll_link": (0.085445, (0.0171394, 0.000537591, 4.8864e-07), (0.575338, 0.411667, -0.574906, 0.411094), (5.48211e-05, 4.96646e-05, 3.57798e-05)),
    "left_wrist_pitch_link": (0.48405, (0.0229999, -0.00111685, -0.00111658), (0.249998, 0.661363, 0.293036, 0.643608), (0.000430353, 0.000429873, 0.000164648)),
    "left_wrist_yaw_link": (0.254576, (0.0708244, 0.000191745, 0.00161742), (0.510571, 0.526295, 0.468078, 0.493188), (0.000646113, 0.000559993, 0.000147566)),
    "left_rubber_hand": (0.175, (0.055, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.00015, 0.00015, 0.00010)),
    "right_shoulder_pitch_link": (0.718, (0.0, -0.035892, -0.011628), (0.68225, -0.326267, 0.0130458, 0.654152), (0.000465864, 0.000432842, 0.000406394)),
    "right_shoulder_roll_link": (0.643, (-0.000227, -0.00727, -0.063243), (0.712604, -0.00710317, -0.0196223, 0.701256), (0.000691311, 0.000618011, 0.000388977)),
    "right_shoulder_yaw_link": (0.734, (0.010773, 0.002949, -0.072009), (0.687134, -0.0679942, -0.0964829, 0.716879), (0.00106187, 0.00103217, 0.000400661)),
    "right_elbow_link": (0.6, (0.064956, -0.004454, -0.010062), (0.388129, 0.388821, 0.636132, 0.541765), (0.000443035, 0.000421612, 0.000259353)),
    "right_wrist_roll_link": (0.085445, (0.0171394, -0.000537591, 4.8864e-07), (0.411667, 0.575338, -0.411094, 0.574906), (5.48211e-05, 4.96646e-05, 3.57798e-05)),
    "right_wrist_pitch_link": (0.48405, (0.0229999, 0.00111685, -0.00111658), (0.643608, 0.293036, 0.661363, 0.249998), (0.000430353, 0.000429873, 0.000164648)),
    "right_wrist_yaw_link": (0.254576, (0.0708244, -0.000191745, 0.00161742), (0.493188, 0.468078, 0.526295, 0.510571), (0.000646113, 0.000559993, 0.000147566)),
    "right_rubber_hand": (0.175, (0.055, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.00015, 0.00015, 0.00010)),
}

_G1_MESH_FILES: dict[str, str] = {
    "pelvis": "pelvis.STL",
    "pelvis_contour_link": "pelvis_contour_link.STL",
    "left_hip_pitch_link": "left_hip_pitch_link.STL",
    "left_hip_roll_link": "left_hip_roll_link.STL",
    "left_hip_yaw_link": "left_hip_yaw_link.STL",
    "left_knee_link": "left_knee_link.STL",
    "left_ankle_pitch_link": "left_ankle_pitch_link.STL",
    "left_ankle_roll_link": "left_ankle_roll_link.STL",
    "right_hip_pitch_link": "right_hip_pitch_link.STL",
    "right_hip_roll_link": "right_hip_roll_link.STL",
    "right_hip_yaw_link": "right_hip_yaw_link.STL",
    "right_knee_link": "right_knee_link.STL",
    "right_ankle_pitch_link": "right_ankle_pitch_link.STL",
    "right_ankle_roll_link": "right_ankle_roll_link.STL",
    "waist_yaw_link": "waist_yaw_link_rev_1_0.STL",
    "waist_roll_link": "waist_roll_link_rev_1_0.STL",
    "torso_link": "torso_link_rev_1_0.STL",
    "head_link": "head_link.STL",
    "left_shoulder_pitch_link": "left_shoulder_pitch_link.STL",
    "left_shoulder_roll_link": "left_shoulder_roll_link.STL",
    "left_shoulder_yaw_link": "left_shoulder_yaw_link.STL",
    "left_elbow_link": "left_elbow_link.STL",
    "left_wrist_roll_link": "left_wrist_roll_link.STL",
    "left_wrist_pitch_link": "left_wrist_pitch_link.STL",
    "left_wrist_yaw_link": "left_wrist_yaw_link.STL",
    "left_rubber_hand": "left_rubber_hand.STL",
    "right_shoulder_pitch_link": "right_shoulder_pitch_link.STL",
    "right_shoulder_roll_link": "right_shoulder_roll_link.STL",
    "right_shoulder_yaw_link": "right_shoulder_yaw_link.STL",
    "right_elbow_link": "right_elbow_link.STL",
    "right_wrist_roll_link": "right_wrist_roll_link.STL",
    "right_wrist_pitch_link": "right_wrist_pitch_link.STL",
    "right_wrist_yaw_link": "right_wrist_yaw_link.STL",
    "right_rubber_hand": "right_rubber_hand.STL",
}


def g1_joint_armature_and_effort(joint_name: str) -> tuple[float, float]:
    """Return `(armature_kgm2, effort_limit_nm)` matching `sugar_rl/assets/robots/unitree.py`."""
    arm, _, _, eff = g1_joint_actuator_spec(joint_name)
    return arm, eff


def g1_joint_actuator_spec(joint_name: str) -> tuple[float, float, float, float]:
    """Return `(armature_kgm2, kp_nm_per_rad, kd_nms_per_rad, effort_limit_nm)` matching `unitree.py`."""
    arm_5020 = 0.003609725
    arm_7520_14 = 0.010177520
    arm_7520_22 = 0.025101925
    arm_4010 = 0.00425
    nat_freq = 10.0 * 2.0 * 3.1415926535
    damp_ratio = 2.0

    def _from_arm(arm: float, eff: float) -> tuple[float, float, float, float]:
        kp = arm * nat_freq**2
        kd = 2.0 * damp_ratio * arm * nat_freq
        return arm, kp, kd, eff

    if re.fullmatch(r".*_hip_(pitch|yaw)_joint", joint_name) or joint_name == "waist_yaw_joint":
        return _from_arm(arm_7520_14, 88.0)
    if re.fullmatch(r".*_(hip_roll|knee)_joint", joint_name):
        return _from_arm(arm_7520_22, 139.0)
    if re.fullmatch(r".*_ankle_(pitch|roll)_joint", joint_name) or joint_name in ("waist_roll_joint", "waist_pitch_joint"):
        return _from_arm(2.0 * arm_5020, 50.0)
    if re.fullmatch(r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint", joint_name):
        return _from_arm(arm_5020, 25.0)
    if re.fullmatch(r".*_wrist_(pitch|yaw)_joint", joint_name):
        return _from_arm(arm_4010, 5.0)
    raise ValueError(f"Unknown G1 29-DoF joint name: {joint_name}")


# -----------------------------------------------------------------------------
# 6 SUGAR Task Specifications (`objects.py` + `train.sh`)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Sim2SimTaskSpec:
    """Task-level object physics & default poses for MuJoCo Sim-to-Sim."""

    task_name: str
    object_subdir: str                                    # "small_box" | "big_box" | "bottle" | "chair"
    object_mass: float                                    # kg (from `sugar_rl/assets/objects/objects.py`)
    angular_damping: float                                # N*m*s/rad
    friction: tuple[float, float, float]                  # (sliding, torsional, rolling)
    use_target_default: bool                              # Matches `train.sh` USE_TARGET
    default_half_extents: tuple[float, float, float]      # Fallback (hx, hy, hz) in meters
    default_obj_pos: tuple[float, float, float]
    default_obj_quat: tuple[float, float, float, float]   # wxyz
    default_target_pos: tuple[float, float, float]
    default_target_quat: tuple[float, float, float, float]
    rgba: tuple[float, float, float, float]


SIM2SIM_TASK_SPECS: dict[str, Sim2SimTaskSpec] = {
    "CarryBox": Sim2SimTaskSpec(
        task_name="CarryBox",
        object_subdir="small_box",
        object_mass=0.5,
        angular_damping=0.01,
        friction=(0.75, 0.008, 0.0005),
        use_target_default=True,
        default_half_extents=(0.15, 0.15, 0.15),
        default_obj_pos=(0.45, 0.0, 0.15),
        default_obj_quat=(1.0, 0.0, 0.0, 0.0),
        default_target_pos=(1.15, 0.0, 0.15),
        default_target_quat=(1.0, 0.0, 0.0, 0.0),
        rgba=(0.82, 0.58, 0.34, 1.0),
    ),
    "PushBox": Sim2SimTaskSpec(
        task_name="PushBox",
        object_subdir="big_box",
        object_mass=0.75,
        angular_damping=0.01,
        friction=(0.55, 0.005, 0.0005),
        use_target_default=True,
        default_half_extents=(0.22, 0.22, 0.22),
        default_obj_pos=(0.55, 0.0, 0.22),
        default_obj_quat=(1.0, 0.0, 0.0, 0.0),
        default_target_pos=(1.25, 0.0, 0.22),
        default_target_quat=(1.0, 0.0, 0.0, 0.0),
        rgba=(0.76, 0.50, 0.28, 1.0),
    ),
    "KickBox": Sim2SimTaskSpec(
        task_name="KickBox",
        object_subdir="big_box",
        object_mass=0.75,
        angular_damping=0.01,
        friction=(0.55, 0.005, 0.0005),
        use_target_default=False,
        default_half_extents=(0.22, 0.22, 0.22),
        default_obj_pos=(0.52, 0.0, 0.22),
        default_obj_quat=(1.0, 0.0, 0.0, 0.0),
        default_target_pos=(1.35, 0.0, 0.22),
        default_target_quat=(1.0, 0.0, 0.0, 0.0),
        rgba=(0.76, 0.50, 0.28, 1.0),
    ),
    "PickBottle": Sim2SimTaskSpec(
        task_name="PickBottle",
        object_subdir="bottle",
        object_mass=0.75,
        angular_damping=0.20,
        friction=(0.85, 0.010, 0.0005),
        use_target_default=False,
        default_half_extents=(0.045, 0.045, 0.13),
        default_obj_pos=(0.42, 0.0, 0.13),
        default_obj_quat=(1.0, 0.0, 0.0, 0.0),
        default_target_pos=(0.42, 0.0, 0.85),
        default_target_quat=(1.0, 0.0, 0.0, 0.0),
        rgba=(0.22, 0.62, 0.86, 1.0),
    ),
    "StandBottle": Sim2SimTaskSpec(
        task_name="StandBottle",
        object_subdir="bottle",
        object_mass=0.75,
        angular_damping=0.20,
        friction=(0.85, 0.010, 0.0005),
        use_target_default=False,
        default_half_extents=(0.045, 0.045, 0.13),
        default_obj_pos=(0.45, 0.0, 0.045),
        default_obj_quat=(0.70710678, 0.0, 0.70710678, 0.0),
        default_target_pos=(0.45, 0.0, 0.13),
        default_target_quat=(1.0, 0.0, 0.0, 0.0),
        rgba=(0.22, 0.62, 0.86, 1.0),
    ),
    "SitChair": Sim2SimTaskSpec(
        task_name="SitChair",
        object_subdir="chair",
        object_mass=3.0,
        angular_damping=0.05,
        friction=(0.85, 0.010, 0.0010),
        use_target_default=False,
        default_half_extents=(0.23, 0.23, 0.41),
        default_obj_pos=(-0.48, 0.0, 0.41),
        default_obj_quat=(1.0, 0.0, 0.0, 0.0),
        default_target_pos=(-0.48, 0.0, 0.41),
        default_target_quat=(1.0, 0.0, 0.0, 0.0),
        rgba=(0.45, 0.48, 0.54, 1.0),
    ),
}


def resolve_task_spec(task_name_or_id: str) -> Sim2SimTaskSpec:
    """Resolve canonical `Sim2SimTaskSpec` from short name (`CarryBox`) or Gym ID (`Sugar-G129dof-CarryBox-...`)."""
    for key, spec in SIM2SIM_TASK_SPECS.items():
        if key.lower() in task_name_or_id.lower():
            return spec
    valid = ", ".join(SIM2SIM_TASK_SPECS.keys())
    raise ValueError(f"Unknown SUGAR task '{task_name_or_id}'. Expected one of: {valid}")


# -----------------------------------------------------------------------------
# Mesh discovery & optional USD -> STL extraction
# -----------------------------------------------------------------------------

def _fmt_vec(vals: Sequence[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def _discover_g1_mesh_dir(descriptions_dir: Path | None) -> Path | None:
    """Locate directory containing Unitree G1 `.STL` link meshes if available locally."""
    candidates: list[Path] = []
    if descriptions_dir is not None:
        candidates.extend([
            descriptions_dir / "robots" / "g1" / "meshes",
            descriptions_dir / "g1" / "meshes",
            descriptions_dir / "meshes",
        ])
    candidates.extend([
        _FUSION_ROOT / "descriptions" / "robots" / "g1" / "meshes",
        _FUSION_ROOT.parent / "SUGAR-main" / "descriptions" / "robots" / "g1" / "meshes",
        Path(r"D:\robot\papers\gmr\code-readonly\assets\unitree_g1\meshes"),
    ])
    for cand in candidates:
        if cand.is_dir() and (cand / "pelvis.STL").exists():
            return cand.resolve()
    return None


def _try_extract_usd_to_stl(usd_path: Path, out_stl_path: Path) -> Path | None:
    """Extract mesh geometry from `obj_aligned.usd` to `.stl` if `usd-core` (`pxr`) is installed."""
    if out_stl_path.exists():
        return out_stl_path
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        import trimesh
    except Exception:
        return None

    try:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            return None
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        sub_meshes = []
        traverse_iter = (
            stage.Traverse(Usd.TraverseInstanceProxies())
            if hasattr(Usd, "TraverseInstanceProxies")
            else stage.Traverse()
        )
        for prim in traverse_iter:
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh_geom = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh_geom.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mesh_geom.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if pts.size == 0 or counts.size == 0:
                continue
            mat = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
            # USD uses row-vector convention: p_world = [p_local, 1] @ mat
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=-1)
            pts_w = (pts_h @ mat)[:, :3]

            faces = []
            cursor = 0
            for c in counts:
                poly = indices[cursor : cursor + c]
                cursor += c
                for k in range(1, len(poly) - 1):
                    faces.append([int(poly[0]), int(poly[k]), int(poly[k + 1])])
            if faces:
                sub_meshes.append(trimesh.Trimesh(vertices=pts_w, faces=np.asarray(faces, dtype=np.int64), process=False))

        if not sub_meshes:
            return None
        merged = trimesh.util.concatenate(sub_meshes)
        out_stl_path.parent.mkdir(parents=True, exist_ok=True)
        merged.export(str(out_stl_path))
        return out_stl_path
    except Exception:
        return None


def _discover_object_mesh(
    spec: Sim2SimTaskSpec,
    descriptions_dir: Path | None,
    cache_dir: Path,
    motion_folder: Path | None = None,
) -> tuple[Path | None, np.ndarray | None, np.ndarray | None, str]:
    """Find object `.obj`/`.stl`/`.ply` (or extract from `.usd`) and return `(mesh_path, center_xyz, half_extents, tag)`."""
    # Priority 0: Dataset-local object override (`box_spec.json` / `obj_aligned.obj` inside `motion_folder`, e.g. OmniRetarget)
    if motion_folder is not None:
        m_dir = Path(motion_folder).resolve()
        for cand_dir in (m_dir, m_dir.parent):
            spec_json = cand_dir / "box_spec.json"
            if spec_json.is_file():
                import json

                meta = json.loads(spec_json.read_text(encoding="utf-8"))
                mesh_rel = meta.get("mesh_file", "obj_aligned.obj")
                local_mesh = cand_dir / mesh_rel
                if not local_mesh.is_file():
                    raise FileNotFoundError(
                        f"[Scene Builder] Found `{spec_json}`, but missing dataset mesh `{local_mesh}`. "
                        f"Refusing silent box fallback; please re-run `convert_omniretarget_to_sugar.py`."
                    )
                import trimesh

                loaded = trimesh.load(str(local_mesh), force="mesh")
                # Use convex hull for clean fast MuJoCo collision/visual (<2k verts)
                hull = loaded.convex_hull
                bounds = np.asarray(hull.bounds, dtype=np.float64)
                center = 0.5 * (bounds[0] + bounds[1])
                half_ext = np.maximum(0.5 * (bounds[1] - bounds[0]), 0.01)
                ds_tag = str(meta.get("dataset", "custom")).lower()
                stl_cache = cache_dir / f"{spec.object_subdir}_{ds_tag}.stl"
                if (not stl_cache.exists()) or (stl_cache.stat().st_mtime < local_mesh.stat().st_mtime):
                    stl_cache.parent.mkdir(parents=True, exist_ok=True)
                    hull.export(str(stl_cache))
                return stl_cache.resolve(), center, half_ext, ds_tag

    roots: list[Path] = []
    if descriptions_dir is not None:
        roots.append(descriptions_dir)
    roots.extend([
        _FUSION_ROOT / "descriptions",
        _FUSION_ROOT.parent / "SUGAR-main" / "descriptions",
    ])

    mesh_file: Path | None = None
    for root in roots:
        obj_dir = root / "objects" / spec.object_subdir
        if not obj_dir.is_dir():
            continue
        for pat in ("obj_aligned.obj", "obj_aligned.stl", "obj_aligned.ply", "*.obj", "*.stl", "*.ply"):
            matches = sorted(obj_dir.glob(pat))
            if matches:
                mesh_file = matches[0].resolve()
                break
        if mesh_file is not None:
            break
        usd_cand = obj_dir / "obj_aligned.usd"
        if usd_cand.exists():
            extracted = _try_extract_usd_to_stl(usd_cand, cache_dir / f"{spec.object_subdir}_aligned.stl")
            if extracted is not None:
                mesh_file = extracted.resolve()
                break

    if mesh_file is None:
        return None, None, None, "default"

    try:
        import trimesh

        loaded = trimesh.load(str(mesh_file), force="mesh")
        bounds = np.asarray(loaded.bounds, dtype=np.float64)  # (2, 3)
        center = 0.5 * (bounds[0] + bounds[1])
        half_ext = np.maximum(0.5 * (bounds[1] - bounds[0]), 0.01)

        # MuJoCo `<mesh>` supports `.stl` and `.obj` natively; convert `.ply` or multi-material `.obj` to binary `.stl`
        if mesh_file.suffix.lower() not in (".stl",):
            stl_cache = cache_dir / f"{spec.object_subdir}_visual.stl"
            if not stl_cache.exists():
                stl_cache.parent.mkdir(parents=True, exist_ok=True)
                loaded.export(str(stl_cache))
            mesh_file = stl_cache.resolve()
        return mesh_file, center, half_ext, "sugar"
    except Exception:
        return None, None, None, "default"


# -----------------------------------------------------------------------------
# Robot & Object XML Subtree Builders
# -----------------------------------------------------------------------------

def _append_fallback_link_geoms(body_el: ET.Element, link_name: str, child_rows: list[tuple]) -> None:
    """Attach clean primitive visual + collision geoms when STL meshes are not present."""
    if link_name == "pelvis":
        ET.SubElement(body_el, "geom", type="box", size="0.08 0.11 0.06", pos="0 0 -0.05", rgba="0.25 0.27 0.30 1")
        return
    if link_name == "torso_link":
        ET.SubElement(body_el, "geom", type="box", size="0.085 0.125 0.16", pos="0 0 0.16", rgba="0.32 0.35 0.40 1")
        return
    if link_name == "head_link":
        ET.SubElement(body_el, "geom", type="sphere", size="0.075", pos="0 0 0.40", contype="0", conaffinity="0", rgba="0.22 0.24 0.28 1")
        return
    if link_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        # Flat foot sole box + 4 contact pads matching G1 foot geometry (-0.05..+0.12 x, +-0.03 y, z=-0.03)
        ET.SubElement(
            body_el, "geom", name=f"{link_name}_sole", type="box",
            size="0.088 0.032 0.008", pos="0.035 0 -0.027",
            friction="1.0 0.005 0.0001", rgba="0.18 0.18 0.20 1",
        )
        return
    if link_name in ("left_rubber_hand", "right_rubber_hand"):
        sign = 1.0 if "left" in link_name else -1.0
        # High-friction palm + curved finger pad for reliable box / bottle / chair manipulation
        ET.SubElement(
            body_el, "geom", name=f"{link_name}_palm", type="box",
            size="0.048 0.016 0.040", pos=f"0.045 {0.006 * sign:.4f} 0.0",
            friction="1.2 0.02 0.001", rgba="0.15 0.15 0.16 1",
        )
        ET.SubElement(
            body_el, "geom", name=f"{link_name}_fingers", type="box",
            size="0.032 0.014 0.034", pos=f"0.105 {-0.008 * sign:.4f} 0.0",
            friction="1.2 0.02 0.001", rgba="0.22 0.22 0.24 1",
        )
        return

    # Connect parent link origin to primary child joint origin via capsule
    for crow in child_rows:
        c_pos = crow[3]
        dist = math.sqrt(sum(v * v for v in c_pos))
        if dist > 0.035:
            ET.SubElement(
                body_el, "geom", type="capsule", size="0.032",
                fromto=f"0 0 0 {_fmt_vec(c_pos)}", rgba="0.65 0.67 0.70 1",
            )
            break


def _append_mesh_link_geoms(body_el: ET.Element, link_name: str) -> None:
    """Attach STL visual mesh + stable collision geoms for Unitree G1 with rubber hands."""
    mesh_key = link_name if link_name in _G1_MESH_FILES else None
    if mesh_key is not None:
        dark = link_name in ("pelvis", "head_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_rubber_hand", "right_rubber_hand")
        rgba = "0.22 0.23 0.25 1" if dark else "0.72 0.74 0.76 1"
        ET.SubElement(
            body_el, "geom", type="mesh", mesh=mesh_key,
            contype="0", conaffinity="0", group="1", density="0", rgba=rgba,
        )

    if link_name == "pelvis":
        ET.SubElement(
            body_el, "geom", type="mesh", mesh="pelvis_contour_link",
            contype="0", conaffinity="0", group="1", density="0", rgba="0.72 0.74 0.76 1",
        )
        ET.SubElement(body_el, "geom", type="mesh", mesh="pelvis_contour_link", rgba="0.72 0.74 0.76 1")
    elif link_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        # 4 contact spheres + thin sole box matching Unitree G1 MJCF foot contact pads
        for px, py in ((-0.05, 0.025), (-0.05, -0.025), (0.12, 0.03), (0.12, -0.03)):
            ET.SubElement(
                body_el, "geom", type="sphere", size="0.006",
                pos=f"{px} {py} -0.03", friction="1.0 0.005 0.0001", rgba="0.2 0.2 0.2 1",
            )
        ET.SubElement(
            body_el, "geom", name=f"{link_name}_sole", type="box",
            size="0.082 0.028 0.005", pos="0.035 0 -0.029",
            friction="1.0 0.005 0.0001", rgba="0.2 0.2 0.2 0.0",
        )
    elif link_name in ("left_rubber_hand", "right_rubber_hand"):
        # CRITICAL FIX over mocap XML: enable rubber hand collision mesh + high-friction inner palm pad!
        ET.SubElement(
            body_el, "geom", name=f"{link_name}_col_mesh", type="mesh", mesh=link_name,
            friction="1.2 0.02 0.001", rgba="0.22 0.23 0.25 1",
        )
        sign = 1.0 if "left" in link_name else -1.0
        ET.SubElement(
            body_el, "geom", name=f"{link_name}_palm_pad", type="box",
            size="0.042 0.012 0.034", pos=f"0.048 {0.005 * sign:.4f} 0.0",
            friction="1.2 0.02 0.001", rgba="0.2 0.2 0.2 0.0",
        )
    elif mesh_key is not None and link_name not in ("waist_yaw_link", "waist_roll_link", "pelvis_contour_link"):
        ET.SubElement(body_el, "geom", type="mesh", mesh=mesh_key, rgba="0.72 0.74 0.76 1")


def _build_g1_robot_tree(
    worldbody_el: ET.Element,
    actuator_el: ET.Element,
    use_stl_meshes: bool,
) -> None:
    """Populate G1 29-DoF kinematic hierarchy with exact CAD inertias, SUGAR joint limits & rotor armature."""
    children_map: dict[str | None, list[tuple]] = {}
    for row in _BUILTIN_G1_29DOF_TREE:
        children_map.setdefault(row[0], []).append(row)

    def _add_node(parent_xml: ET.Element, row: tuple) -> None:
        _, link_name, joint_name, pos_xyz, quat_wxyz, axis_xyz, lower, upper = row
        if link_name == "pelvis_contour_link":
            # Pure visual/collision shell already attached directly onto `pelvis`
            return
        body_attrib = {"name": link_name, "pos": _fmt_vec(pos_xyz)}
        if abs(quat_wxyz[0] - 1.0) > 1e-6 or sum(abs(c) for c in quat_wxyz[1:]) > 1e-6:
            body_attrib["quat"] = _fmt_vec(quat_wxyz)
        body_el = ET.SubElement(parent_xml, "body", **body_attrib)

        if link_name == "pelvis":
            ET.SubElement(body_el, "freejoint", name="pelvis_freejoint")

        if link_name in _G1_CAD_INERTIALS:
            mass, ipos, iquat, idiag = _G1_CAD_INERTIALS[link_name]
            ET.SubElement(
                body_el,
                "inertial",
                pos=_fmt_vec(ipos),
                quat=_fmt_vec(iquat),
                mass=f"{mass:.6g}",
                diaginertia=_fmt_vec(idiag),
            )

        if joint_name is not None:
            armature, effort = g1_joint_armature_and_effort(joint_name)
            ET.SubElement(
                body_el,
                "joint",
                name=joint_name,
                type="hinge",
                pos="0 0 0",
                axis=_fmt_vec(axis_xyz),
                range=f"{lower:.6g} {upper:.6g}",
                armature=f"{armature:.9g}",
                actuatorfrclimited="true",
                actuatorfrcrange=f"{-effort:.6g} {effort:.6g}",
            )

        c_rows = children_map.get(link_name, [])
        if use_stl_meshes:
            _append_mesh_link_geoms(body_el, link_name)
        else:
            _append_fallback_link_geoms(body_el, link_name, c_rows)

        for crow in c_rows:
            _add_node(body_el, crow)

    for root_row in children_map[None]:
        _add_node(worldbody_el, root_row)

    # Create 29 implicit PD position actuators matching IsaacLab `ImplicitActuatorCfg` (`kp`, `kv`, `forcerange`)
    for j_name in G1_29DOF_SDK_JOINT_NAMES:
        _, kp, kd, effort = g1_joint_actuator_spec(j_name)
        ET.SubElement(
            actuator_el,
            "position",
            name=j_name,
            joint=j_name,
            kp=f"{kp:.8g}",
            kv=f"{kd:.8g}",
            forcelimited="true",
            forcerange=f"{-effort:.6g} {effort:.6g}",
        )


def _append_g1_intra_joint_excludes(contact_el: ET.Element, max_tree_hops: int = 3) -> None:
    """Exclude convex-hull overlaps within compound 3-DoF ball joints (hip, waist, shoulder, wrist, ankle).

    Counts active hinge joints along the tree (`max_tree_hops=3`), treating fixed welded child bodies
    (`head_link`, `left_rubber_hand`, `right_rubber_hand`) as 0 additional joint hops while preserving
    leg-vs-leg, arm-vs-torso/thigh, and 100% of robot-vs-object / robot-vs-ground contacts.
    """
    parent_of: dict[str, str] = {}
    has_joint: dict[str, bool] = {}
    valid_links: list[str] = []
    for parent_lk, lk, j_name, *_ in _BUILTIN_G1_29DOF_TREE:
        if lk == "pelvis_contour_link":
            continue
        valid_links.append(lk)
        has_joint[lk] = j_name is not None
        if parent_lk is not None:
            parent_of[lk] = parent_lk

    seen: set[tuple[str, str]] = set()
    for lk in valid_links:
        cur = lk
        joint_hops = 0
        while True:
            step_cost = 1 if has_joint.get(cur, False) else 0
            if joint_hops + step_cost > max_tree_hops:
                break
            anc = parent_of.get(cur)
            if anc is None:
                break
            joint_hops += step_cost
            pair = (anc, lk) if anc < lk else (lk, anc)
            if pair not in seen:
                seen.add(pair)
                ET.SubElement(contact_el, "exclude", body1=anc, body2=lk)
            cur = anc


def _add_object_collision_and_visual(
    parent_el: ET.Element,
    spec: Sim2SimTaskSpec,
    center: np.ndarray,
    half_ext: np.ndarray,
    visual_mesh_name: str | None,
    is_ghost: bool = False,
) -> None:
    """Populate collision/visual geoms for `obj` (physical) or `target_vis` (ghost mocap marker)."""
    rgba_str = "0.18 0.88 0.38 0.32" if is_ghost else _fmt_vec(spec.rgba)
    col_kwargs = (
        {"contype": "0", "conaffinity": "0", "mass": "0"}
        if is_ghost
        else {
            "contype": "1",
            "conaffinity": "1",
            "friction": _fmt_vec(spec.friction),
        }
    )

    if visual_mesh_name is not None:
        ET.SubElement(
            parent_el,
            "geom",
            type="mesh",
            mesh=visual_mesh_name,
            contype="0",
            conaffinity="0",
            mass="0",
            rgba=rgba_str,
        )

    hx, hy, hz = float(half_ext[0]), float(half_ext[1]), float(half_ext[2])
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    if is_ghost and visual_mesh_name is not None:
        return

    # When visual mesh is present, hide primitive collision proxies (alpha=0) so visuals stay crisp
    prim_rgba = rgba_str if visual_mesh_name is None else "0.5 0.5 0.5 0.0"

    if spec.object_subdir in ("small_box", "big_box"):
        if visual_mesh_name is not None:
            ET.SubElement(
                parent_el,
                "geom",
                type="mesh",
                mesh=visual_mesh_name,
                rgba=prim_rgba,
                **({"mass": f"{spec.object_mass:.6g}"} if not is_ghost else {}),
                **col_kwargs,
            )
        else:
            ET.SubElement(
                parent_el,
                "geom",
                type="box",
                pos=_fmt_vec((cx, cy, cz)),
                size=_fmt_vec((hx, hy, hz)),
                rgba=prim_rgba,
                **({"mass": f"{spec.object_mass:.6g}"} if not is_ghost else {}),
                **col_kwargs,
            )
    elif spec.object_subdir == "bottle":
        radius = 0.5 * (hx + hy)
        body_hz = hz * 0.72
        neck_hz = hz * 0.28
        body_z = cz - hz + body_hz
        neck_z = cz + hz - neck_hz
        ET.SubElement(
            parent_el,
            "geom",
            type="cylinder",
            pos=_fmt_vec((cx, cy, body_z)),
            size=_fmt_vec((radius, body_hz)),
            rgba=prim_rgba,
            **({"mass": f"{spec.object_mass * 0.85:.6g}"} if not is_ghost else {}),
            **col_kwargs,
        )
        ET.SubElement(
            parent_el,
            "geom",
            type="cylinder",
            pos=_fmt_vec((cx, cy, neck_z)),
            size=_fmt_vec((radius * 0.42, neck_hz)),
            rgba=prim_rgba,
            **({"mass": f"{spec.object_mass * 0.15:.6g}"} if not is_ghost else {}),
            **col_kwargs,
        )
    elif spec.object_subdir == "chair":
        # Multi-box L-shaped decomposition so `SitChair` preserves concave seat space (never sealed by convex hull!)
        min_z = cz - hz
        total_h = 2.0 * hz
        seat_top = min_z + 0.52 * total_h
        seat_thick = max(0.025, 0.06 * total_h)
        seat_z = seat_top - seat_thick
        leg_hz = max(0.02, 0.5 * (seat_top - 2.0 * seat_thick - min_z))
        leg_z = min_z + leg_hz
        back_hz = max(0.05, 0.5 * (cz + hz - seat_top))
        back_z = seat_top + back_hz
        back_thick = max(0.025, 0.12 * hx)

        # Seat plate
        ET.SubElement(
            parent_el, "geom", type="box",
            pos=_fmt_vec((cx, cy, seat_z)),
            size=_fmt_vec((hx * 0.95, hy * 0.95, seat_thick)),
            rgba=prim_rgba,
            **({"mass": f"{spec.object_mass * 0.45:.6g}"} if not is_ghost else {}),
            **col_kwargs,
        )
        # Backrest (at -x side of chair local frame; symmetric enough on legs while leaving +x seat open)
        ET.SubElement(
            parent_el, "geom", type="box",
            pos=_fmt_vec((cx - hx * 0.85, cy, back_z)),
            size=_fmt_vec((back_thick, hy * 0.92, back_hz)),
            rgba=prim_rgba,
            **({"mass": f"{spec.object_mass * 0.25:.6g}"} if not is_ghost else {}),
            **col_kwargs,
        )
        # 4 chair legs
        leg_r = max(0.02, 0.12 * min(hx, hy))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                ET.SubElement(
                    parent_el, "geom", type="box",
                    pos=_fmt_vec((cx + sx * (hx * 0.80), cy + sy * (hy * 0.80), leg_z)),
                    size=_fmt_vec((leg_r, leg_r, leg_hz)),
                    rgba=prim_rgba,
                    **({"mass": f"{spec.object_mass * 0.075:.6g}"} if not is_ghost else {}),
                    **col_kwargs,
                )
    else:
        raise ValueError(f"Unsupported object_subdir: {spec.object_subdir}")


def build_mujoco_scene_xml(
    task_name: str,
    descriptions_dir: str | Path | None = None,
    sim_dt: float = 0.002,
    output_xml_path: str | Path | None = None,
    motion_folder: str | Path | None = None,
) -> tuple[Path, Sim2SimTaskSpec, dict[str, str]]:
    """Build a complete, self-contained MuJoCo scene XML for any of the 6 SUGAR tasks.

    Returns:
        `(xml_path, task_spec, asset_report)` where `asset_report` summarizes resolved robot/object meshes.
    """
    spec = resolve_task_spec(task_name)
    desc_path = Path(descriptions_dir).resolve() if descriptions_dir is not None else None
    motion_path = Path(motion_folder).resolve() if motion_folder is not None else None
    cache_dir = (
        Path(output_xml_path).resolve().parent
        if output_xml_path is not None
        else (_FUSION_ROOT / "outputs" / "sim2sim_scenes")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    g1_mesh_dir = _discover_g1_mesh_dir(desc_path)
    obj_mesh_path, obj_center, obj_half_ext, obj_tag = _discover_object_mesh(
        spec, desc_path, cache_dir / "meshes", motion_folder=motion_path
    )

    if obj_center is None or obj_half_ext is None:
        obj_half_ext = np.asarray(spec.default_half_extents, dtype=np.float64)
        obj_center = np.zeros(3, dtype=np.float64)

    mj_el = ET.Element("mujoco", model=f"sugar_g1_{spec.task_name.lower()}_sim2sim")
    ET.SubElement(mj_el, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        mj_el,
        "option",
        timestep=f"{sim_dt:.6g}",
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
    )

    vis_el = ET.SubElement(mj_el, "visual")
    ET.SubElement(vis_el, "headlight", diffuse="0.65 0.65 0.65", ambient="0.25 0.25 0.28", specular="0.4 0.4 0.4")
    ET.SubElement(vis_el, "rgba", haze="0.18 0.24 0.32 1")
    ET.SubElement(vis_el, "global", azimuth="-135", elevation="-18", offwidth="1920", offheight="1080")

    asset_el = ET.SubElement(mj_el, "asset")
    ET.SubElement(
        asset_el, "texture", type="skybox", builtin="gradient",
        rgb1="0.88 0.92 0.96", rgb2="0.22 0.28 0.36", width="512", height="512",
    )
    ET.SubElement(
        asset_el, "texture", name="grid_tex", type="2d", builtin="checker",
        rgb1="0.86 0.88 0.90", rgb2="0.74 0.77 0.80", mark="edge", markrgb="0.58 0.62 0.66",
        width="512", height="512",
    )
    ET.SubElement(
        asset_el, "material", name="grid_mat", texture="grid_tex",
        texrepeat="4 4", texuniform="true", reflectance="0.08",
    )

    use_stl = g1_mesh_dir is not None
    if use_stl:
        for mesh_name, stl_filename in _G1_MESH_FILES.items():
            stl_full = g1_mesh_dir / stl_filename
            if not stl_full.exists():
                alt = g1_mesh_dir / f"{mesh_name}.STL"
                if alt.exists():
                    stl_full = alt
            ET.SubElement(asset_el, "mesh", name=mesh_name, file=stl_full.resolve().as_posix())

    vis_obj_mesh_name: str | None = None
    if obj_mesh_path is not None:
        vis_obj_mesh_name = f"{spec.object_subdir}_visual_mesh"
        ET.SubElement(asset_el, "mesh", name=vis_obj_mesh_name, file=obj_mesh_path.as_posix())

    worldbody_el = ET.SubElement(mj_el, "worldbody")
    ET.SubElement(worldbody_el, "light", pos="1.5 -2.0 3.5", dir="-0.3 0.4 -0.85", diffuse="0.75 0.75 0.75", castshadow="true")
    ET.SubElement(worldbody_el, "light", pos="-2.0 2.0 3.0", dir="0.4 -0.4 -0.8", diffuse="0.35 0.38 0.42", castshadow="false")
    ET.SubElement(
        worldbody_el,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        material="grid_mat",
        contype="1",
        conaffinity="1",
        friction="1.0 0.005 0.0001",
        condim="3",
    )

    actuator_el = ET.SubElement(mj_el, "actuator")
    contact_el = ET.SubElement(mj_el, "contact")
    _build_g1_robot_tree(worldbody_el, actuator_el, use_stl_meshes=use_stl)
    _append_g1_intra_joint_excludes(contact_el, max_tree_hops=3)

    # Free-floating interactive object (`obj`)
    obj_body = ET.SubElement(
        worldbody_el,
        "body",
        name="obj",
        pos=_fmt_vec(spec.default_obj_pos),
        quat=_fmt_vec(spec.default_obj_quat),
    )
    ET.SubElement(
        obj_body,
        "freejoint",
        name="obj_freejoint",
    )
    _add_object_collision_and_visual(
        obj_body, spec, obj_center, obj_half_ext, vis_obj_mesh_name, is_ghost=False
    )

    # Draggable semi-transparent green target pose marker (`target_vis`, mocap=true)
    target_body = ET.SubElement(
        worldbody_el,
        "body",
        name="target_vis",
        mocap="true",
        pos=_fmt_vec(spec.default_target_pos),
        quat=_fmt_vec(spec.default_target_quat),
    )
    _add_object_collision_and_visual(
        target_body, spec, obj_center, obj_half_ext, vis_obj_mesh_name, is_ghost=True
    )
    # Coordinate axes triad on `target_vis` so target orientation is clear at a glance
    ET.SubElement(target_body, "site", name="target_axis_x", type="cylinder", fromto="0 0 0 0.14 0 0", size="0.006", rgba="0.95 0.2 0.2 0.75")
    ET.SubElement(target_body, "site", name="target_axis_y", type="cylinder", fromto="0 0 0 0 0.14 0", size="0.006", rgba="0.2 0.95 0.2 0.75")
    ET.SubElement(target_body, "site", name="target_axis_z", type="cylinder", fromto="0 0 0 0 0 0.14", size="0.006", rgba="0.2 0.45 0.98 0.75")

    suffix = "" if obj_tag in ("sugar", "default") else f"_{obj_tag}"
    xml_path = (
        Path(output_xml_path).resolve()
        if output_xml_path is not None
        else (cache_dir / f"g1_{spec.task_name.lower()}{suffix}_scene.xml").resolve()
    )
    ET.indent(mj_el, space="  ")
    xml_path.write_text(ET.tostring(mj_el, encoding="unicode"), encoding="utf-8")

    report = {
        "task": spec.task_name,
        "object_type": spec.object_subdir,
        "robot_geometry": f"STL meshes ({g1_mesh_dir})" if use_stl else "Built-in CAD-inertia primitives (with rubber hand pads)",
        "object_geometry": f"Mesh ({obj_mesh_path})" if obj_mesh_path is not None else f"Parametric {spec.object_subdir} (mass={spec.object_mass}kg)",
        "scene_xml": str(xml_path),
    }
    return xml_path, spec, report

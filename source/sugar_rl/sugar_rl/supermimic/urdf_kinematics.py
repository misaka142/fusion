"""MuJoCo-free differentiable forward kinematics & G1 29-DoF actuator specification.

Supports:
1. Direct parsing of SUGAR's official URDF (`./descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf`)
2. Parsing of G1 MJCF XML (`g1_mocap_29dof.xml`)
3. Built-in exact Unitree G1 29-DoF (rev_1_0 with rubber hands) kinematic tree fallback when running
   unit tests or standalone scripts without `descriptions/` extracted.

Automatically maps arbitrary environment joint orderings (e.g., IsaacLab PhysX BFS order)
to topological kinematic tree order in pure differentiable PyTorch.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rotations import (
    axis_unit_angle_to_rotmat,
    mat_mul,
    mat_vec,
    quat_from_euler_xyz,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
)

# -----------------------------------------------------------------------------
# SUGAR G1 29-DoF canonical body & joint definitions
# -----------------------------------------------------------------------------

SUGAR_TRACKED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

SUGAR_KEY_BODY_NAMES: list[str] = [
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

SUGAR_EE_TERMINATION_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

SUGAR_CONTACT_BODY_NAMES: list[str] = [
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "pelvis_contour_link",
]

G1_29DOF_SDK_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Built-in exact G1 29-DoF rev_1_0 kinematic tree (parent, link, joint_or_None, pos_xyz, quat_wxyz, axis_xyz, lower, upper)
_BUILTIN_G1_29DOF_TREE: list[tuple] = [
    (None, "pelvis", None, (0.0, 0.0, 0.793), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0),
    ("pelvis", "pelvis_contour_link", None, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0),
    # Left leg (6 DoF)
    ("pelvis", "left_hip_pitch_link", "left_hip_pitch_joint", (0.0, 0.064452, -0.1027), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.5307, 2.8798),
    ("left_hip_pitch_link", "left_hip_roll_link", "left_hip_roll_joint", (0.0, 0.052, -0.030465), (0.996179, 0.0, -0.0873386, 0.0), (1.0, 0.0, 0.0), -0.5236, 2.9671),
    ("left_hip_roll_link", "left_hip_yaw_link", "left_hip_yaw_joint", (0.025001, 0.0, -0.12412), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.7576, 2.7576),
    ("left_hip_yaw_link", "left_knee_link", "left_knee_joint", (-0.078273, 0.0021489, -0.17734), (0.996179, 0.0, 0.0873386, 0.0), (0.0, 1.0, 0.0), -0.087267, 2.8798),
    ("left_knee_link", "left_ankle_pitch_link", "left_ankle_pitch_joint", (0.0, -9.4445e-05, -0.30001), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -0.87267, 0.5236),
    ("left_ankle_pitch_link", "left_ankle_roll_link", "left_ankle_roll_joint", (0.0, 0.0, -0.017558), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.2618, 0.2618),
    # Right leg (6 DoF)
    ("pelvis", "right_hip_pitch_link", "right_hip_pitch_joint", (0.0, -0.064452, -0.1027), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.5307, 2.8798),
    ("right_hip_pitch_link", "right_hip_roll_link", "right_hip_roll_joint", (0.0, -0.052, -0.030465), (0.996179, 0.0, -0.0873386, 0.0), (1.0, 0.0, 0.0), -2.9671, 0.5236),
    ("right_hip_roll_link", "right_hip_yaw_link", "right_hip_yaw_joint", (0.025001, 0.0, -0.12412), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.7576, 2.7576),
    ("right_hip_yaw_link", "right_knee_link", "right_knee_joint", (-0.078273, -0.0021489, -0.17734), (0.996179, 0.0, 0.0873386, 0.0), (0.0, 1.0, 0.0), -0.087267, 2.8798),
    ("right_knee_link", "right_ankle_pitch_link", "right_ankle_pitch_joint", (0.0, 9.4445e-05, -0.30001), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -0.87267, 0.5236),
    ("right_ankle_pitch_link", "right_ankle_roll_link", "right_ankle_roll_joint", (0.0, 0.0, -0.017558), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.2618, 0.2618),
    # Waist & Torso (3 DoF)
    ("pelvis", "waist_yaw_link", "waist_yaw_joint", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.618, 2.618),
    ("waist_yaw_link", "waist_roll_link", "waist_roll_joint", (-0.0039635, 0.0, 0.044), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.52, 0.52),
    ("waist_roll_link", "torso_link", "waist_pitch_joint", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -0.52, 0.52),
    ("torso_link", "head_link", None, (0.0039635, 0.0, -0.044), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0),
    # Left arm (7 DoF + fixed rubber hand)
    ("torso_link", "left_shoulder_pitch_link", "left_shoulder_pitch_joint", (0.0039563, 0.10022, 0.24778), (0.990264, 0.139201, 1.38722e-05, -9.86868e-05), (0.0, 1.0, 0.0), -3.0892, 2.6704),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_roll_joint", (0.0, 0.038, -0.013831), (0.990268, -0.139172, 0.0, 0.0), (1.0, 0.0, 0.0), -1.5882, 2.2515),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link", "left_shoulder_yaw_joint", (0.0, 0.00624, -0.1032), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.618, 2.618),
    ("left_shoulder_yaw_link", "left_elbow_link", "left_elbow_joint", (0.015783, 0.0, -0.080518), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.0472, 2.0944),
    ("left_elbow_link", "left_wrist_roll_link", "left_wrist_roll_joint", (0.1, 0.00188791, -0.01), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.97222, 1.97222),
    ("left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_pitch_joint", (0.038, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.61443, 1.61443),
    ("left_wrist_pitch_link", "left_wrist_yaw_link", "left_wrist_yaw_joint", (0.046, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -1.61443, 1.61443),
    ("left_wrist_yaw_link", "left_rubber_hand", None, (0.0415, 0.003, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0),
    # Right arm (7 DoF + fixed rubber hand)
    ("torso_link", "right_shoulder_pitch_link", "right_shoulder_pitch_joint", (0.0039563, -0.10021, 0.24778), (0.990264, -0.139201, 1.38722e-05, 9.86868e-05), (0.0, 1.0, 0.0), -3.0892, 2.6704),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_roll_joint", (0.0, -0.038, -0.013831), (0.990268, 0.139172, 0.0, 0.0), (1.0, 0.0, 0.0), -2.2515, 1.5882),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link", "right_shoulder_yaw_joint", (0.0, -0.00624, -0.1032), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.618, 2.618),
    ("right_shoulder_yaw_link", "right_elbow_link", "right_elbow_joint", (0.015783, 0.0, -0.080518), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.0472, 2.0944),
    ("right_elbow_link", "right_wrist_roll_link", "right_wrist_roll_joint", (0.1, -0.00188791, -0.01), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.97222, 1.97222),
    ("right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_pitch_joint", (0.038, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.61443, 1.61443),
    ("right_wrist_pitch_link", "right_wrist_yaw_link", "right_wrist_yaw_joint", (0.046, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -1.61443, 1.61443),
    ("right_wrist_yaw_link", "right_rubber_hand", None, (0.0415, -0.003, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0),
]


# -----------------------------------------------------------------------------
# Actuator & joint parameter helper matching `sugar_rl/assets/robots/unitree.py`
# -----------------------------------------------------------------------------

@dataclass
class G1ActuatorParams:
    """Per-joint PD gains, limits, default pose, and action scales for a given joint order."""

    joint_names: list[str]
    default_joint_pos: Tensor      # (29,)
    action_scale: Tensor           # (29,)
    stiffness: Tensor              # (29,)
    damping: Tensor                # (29,)
    effort_limit: Tensor           # (29,)
    soft_lower: Tensor             # (29,)
    soft_upper: Tensor             # (29,)


def build_g1_actuator_params(
    joint_names: Sequence[str],
    joint_limits_from_tree: dict[str, tuple[float, float]] | None = None,
    soft_limit_factor: float = 0.9,
    device: torch.device | str = "cpu",
) -> G1ActuatorParams:
    """Compute exact SUGAR G1 29-DoF actuator parameters for `joint_names`."""
    arm_5020 = 0.003609725
    arm_7520_14 = 0.010177520
    arm_7520_22 = 0.025101925
    arm_4010 = 0.00425
    nat_freq = 10.0 * 2.0 * 3.1415926535
    damp_ratio = 2.0

    kp_5020 = arm_5020 * nat_freq**2
    kp_7520_14 = arm_7520_14 * nat_freq**2
    kp_7520_22 = arm_7520_22 * nat_freq**2
    kp_4010 = arm_4010 * nat_freq**2

    kd_5020 = 2.0 * damp_ratio * arm_5020 * nat_freq
    kd_7520_14 = 2.0 * damp_ratio * arm_7520_14 * nat_freq
    kd_7520_22 = 2.0 * damp_ratio * arm_7520_22 * nat_freq
    kd_4010 = 2.0 * damp_ratio * arm_4010 * nat_freq

    default_pos_rules = [
        (r".*_hip_pitch_joint", -0.312),
        (r".*_knee_joint", 0.669),
        (r".*_ankle_pitch_joint", -0.363),
        (r".*_elbow_joint", 0.6),
        (r"left_shoulder_roll_joint", 0.2),
        (r"left_shoulder_pitch_joint", 0.2),
        (r"right_shoulder_roll_joint", -0.2),
        (r"right_shoulder_pitch_joint", 0.2),
    ]

    builtin_limits = {
        row[2]: (row[6], row[7]) for row in _BUILTIN_G1_29DOF_TREE if row[2] is not None
    }
    if joint_limits_from_tree:
        builtin_limits.update(joint_limits_from_tree)

    q_def, scales, kps, kds, efforts, lows, highs = [], [], [], [], [], [], []
    for name in joint_names:
        # Default joint pos
        val = 0.0
        for pat, v in default_pos_rules:
            if re.fullmatch(pat, name):
                val = v
                break
        q_def.append(val)

        # Stiffness, damping, effort limit
        if re.fullmatch(r".*_hip_(pitch|yaw)_joint", name) or name == "waist_yaw_joint":
            kp, kd, eff = kp_7520_14, kd_7520_14, 88.0
        elif re.fullmatch(r".*_(hip_roll|knee)_joint", name):
            kp, kd, eff = kp_7520_22, kd_7520_22, 139.0
        elif re.fullmatch(r".*_ankle_(pitch|roll)_joint", name) or name in ("waist_roll_joint", "waist_pitch_joint"):
            kp, kd, eff = 2.0 * kp_5020, 2.0 * kd_5020, 50.0
        elif re.fullmatch(r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint", name):
            kp, kd, eff = kp_5020, kd_5020, 25.0
        elif re.fullmatch(r".*_wrist_(pitch|yaw)_joint", name):
            kp, kd, eff = kp_4010, kd_4010, 5.0
        else:
            raise ValueError(f"Unknown G1 joint name: {name}")

        kps.append(kp)
        kds.append(kd)
        efforts.append(eff)
        scales.append(0.25 * eff / kp)

        lo, hi = builtin_limits.get(name, (-math.pi, math.pi))
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * soft_limit_factor
        lows.append(mid - half)
        highs.append(mid + half)

    t_kwargs = {"dtype": torch.float32, "device": device}
    return G1ActuatorParams(
        joint_names=list(joint_names),
        default_joint_pos=torch.tensor(q_def, **t_kwargs),
        action_scale=torch.tensor(scales, **t_kwargs),
        stiffness=torch.tensor(kps, **t_kwargs),
        damping=torch.tensor(kds, **t_kwargs),
        effort_limit=torch.tensor(efforts, **t_kwargs),
        soft_lower=torch.tensor(lows, **t_kwargs),
        soft_upper=torch.tensor(highs, **t_kwargs),
    )


# -----------------------------------------------------------------------------
# Pure-Python URDF & MJCF parsers
# -----------------------------------------------------------------------------

def _parse_floats(text: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if not text:
        return tuple(float(x) for x in default)
    return tuple(float(x) for x in text.strip().split())


def parse_urdf_tree(urdf_path: str | Path) -> list[tuple]:
    """Parse standard ROS/IsaacLab URDF XML into topological link/joint tuples."""
    root_el = ET.parse(str(urdf_path)).getroot()
    links = {el.attrib["name"] for el in root_el.findall("link")}
    child_to_joint: dict[str, tuple] = {}
    children_map: dict[str, list[str]] = {lk: [] for lk in links}

    for j_el in root_el.findall("joint"):
        j_name = j_el.attrib["name"]
        j_type = j_el.attrib.get("type", "fixed")
        parent_lk = j_el.find("parent").attrib["link"]
        child_lk = j_el.find("child").attrib["link"]

        orig_el = j_el.find("origin")
        xyz = _parse_floats(orig_el.attrib.get("xyz") if orig_el is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_floats(orig_el.attrib.get("rpy") if orig_el is not None else None, (0.0, 0.0, 0.0))
        r_t = torch.tensor(rpy[0], dtype=torch.float64)
        p_t = torch.tensor(rpy[1], dtype=torch.float64)
        y_t = torch.tensor(rpy[2], dtype=torch.float64)
        quat = tuple(float(v) for v in quat_from_euler_xyz(r_t, p_t, y_t).tolist())

        if j_type in ("revolute", "continuous"):
            axis_el = j_el.find("axis")
            axis = _parse_floats(axis_el.attrib.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0))
            norm = math.sqrt(sum(a * a for a in axis)) or 1.0
            axis = tuple(a / norm for a in axis)
            lim_el = j_el.find("limit")
            lo = float(lim_el.attrib.get("lower", -math.pi)) if lim_el is not None else -math.pi
            hi = float(lim_el.attrib.get("upper", math.pi)) if lim_el is not None else math.pi
            active_j_name = j_name
        else:
            axis = (0.0, 0.0, 0.0)
            lo, hi = 0.0, 0.0
            active_j_name = None

        child_to_joint[child_lk] = (parent_lk, child_lk, active_j_name, xyz, quat, axis, lo, hi)
        children_map.setdefault(parent_lk, []).append(child_lk)

    roots = [lk for lk in links if lk not in child_to_joint]
    root_link = "pelvis" if "pelvis" in roots else sorted(roots)[0]

    ordered: list[tuple] = [
        (None, root_link, None, (0.0, 0.0, 0.793), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0)
    ]
    queue = list(children_map.get(root_link, []))
    while queue:
        lk = queue.pop(0)
        ordered.append(child_to_joint[lk])
        queue.extend(children_map.get(lk, []))
    return ordered


def parse_mjcf_tree(xml_path: str | Path) -> list[tuple]:
    """Parse MuJoCo MJCF XML (`g1_mocap_29dof.xml`) into topological link/joint tuples."""
    root_el = ET.parse(str(xml_path)).getroot()
    worldbody = root_el.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in MJCF: {xml_path}")

    ordered: list[tuple] = []

    def _visit(body_el: ET.Element, parent_name: str | None) -> None:
        b_name = body_el.attrib["name"]
        pos = _parse_floats(body_el.attrib.get("pos"), (0.0, 0.0, 0.0))
        quat = _parse_floats(body_el.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0))
        q_norm = math.sqrt(sum(c * c for c in quat)) or 1.0
        quat = tuple(c / q_norm for c in quat)

        hinge_el = None
        for j_el in body_el.findall("joint"):
            if j_el.attrib.get("type", "hinge") == "hinge":
                hinge_el = j_el
                break

        if parent_name is None or hinge_el is None:
            j_name = None
            axis = (0.0, 0.0, 0.0)
            lo, hi = 0.0, 0.0
        else:
            j_name = hinge_el.attrib["name"]
            axis = _parse_floats(hinge_el.attrib.get("axis"), (0.0, 0.0, 1.0))
            a_norm = math.sqrt(sum(a * a for a in axis)) or 1.0
            axis = tuple(a / a_norm for a in axis)
            rng = _parse_floats(hinge_el.attrib.get("range"), (-math.pi, math.pi))
            lo, hi = rng[0], rng[1]

        ordered.append((parent_name, b_name, j_name, pos, quat, axis, lo, hi))
        for child_el in body_el.findall("body"):
            _visit(child_el, b_name)

    root_body = worldbody.find("body")
    if root_body is None:
        raise ValueError(f"No root <body> under <worldbody> in {xml_path}")
    _visit(root_body, None)
    return ordered


# -----------------------------------------------------------------------------
# Differentiable Forward Kinematics Engine
# -----------------------------------------------------------------------------

@dataclass
class FKResult:
    """World-frame poses and velocities of tracked bodies and contact links."""

    tracked_pos_w: Tensor             # (B, 14, 3)
    tracked_rotmat_w: Tensor          # (B, 14, 3, 3)
    tracked_quat_w: Tensor | None     # (B, 14, 4) or None when `with_quat=False`
    tracked_lin_vel_w: Tensor         # (B, 14, 3)
    tracked_ang_vel_w: Tensor         # (B, 14, 3)

    contact_pos_w: Tensor             # (B, 7, 3)  in SUGAR_CONTACT_BODY_NAMES order
    contact_lin_vel_w: Tensor         # (B, 7, 3)

    anchor_pos_w: Tensor              # (B, 3)     torso_link world pos
    anchor_quat_w: Tensor             # (B, 4)     torso_link world quat
    anchor_rotmat_w: Tensor           # (B, 3, 3)  torso_link world rotmat
    anchor_lin_vel_w: Tensor          # (B, 3)
    anchor_ang_vel_w: Tensor          # (B, 3)
    root_rotmat_w: Tensor | None = None  # (B, 3, 3) pelvis world rotmat (reused to skip redundant quat_to_rotmat)

    def scale_grad(self, alpha: float, anchor_idx: int = 7) -> "FKResult":
        """Same values, gradients scaled by `alpha`, preserving `anchor_*` slice strides on `tracked_*`."""
        if alpha == 1.0:
            return self
        b = 1.0 - alpha
        t_pos = self.tracked_pos_w * alpha + self.tracked_pos_w.detach() * b
        t_mat = self.tracked_rotmat_w * alpha + self.tracked_rotmat_w.detach() * b
        t_quat = (
            self.tracked_quat_w * alpha + self.tracked_quat_w.detach() * b
            if self.tracked_quat_w is not None
            else None
        )
        t_v = self.tracked_lin_vel_w * alpha + self.tracked_lin_vel_w.detach() * b
        t_w = self.tracked_ang_vel_w * alpha + self.tracked_ang_vel_w.detach() * b
        c_pos = self.contact_pos_w * alpha + self.contact_pos_w.detach() * b
        c_v = self.contact_lin_vel_w * alpha + self.contact_lin_vel_w.detach() * b
        a_quat = (
            t_quat[:, anchor_idx]
            if t_quat is not None
            else self.anchor_quat_w * alpha + self.anchor_quat_w.detach() * b
        )
        r_mat = (
            self.root_rotmat_w * alpha + self.root_rotmat_w.detach() * b
            if self.root_rotmat_w is not None
            else None
        )
        return FKResult(
            tracked_pos_w=t_pos,
            tracked_rotmat_w=t_mat,
            tracked_quat_w=t_quat,
            tracked_lin_vel_w=t_v,
            tracked_ang_vel_w=t_w,
            contact_pos_w=c_pos,
            contact_lin_vel_w=c_v,
            anchor_pos_w=t_pos[:, anchor_idx],
            anchor_quat_w=a_quat,
            anchor_rotmat_w=t_mat[:, anchor_idx],
            anchor_lin_vel_w=t_v[:, anchor_idx],
            anchor_ang_vel_w=t_w[:, anchor_idx],
            root_rotmat_w=r_mat,
        )


class FastTreeKinematics(nn.Module):
    """Differentiable tree forward kinematics for Unitree G1 29-DoF in pure PyTorch."""

    DEFAULT_URDF_REL_PATH = "./descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"

    def __init__(
        self,
        urdf_or_xml_path: str | Path | None = None,
        env_joint_names: Sequence[str] | None = None,
        tracked_body_names: Sequence[str] = SUGAR_TRACKED_BODY_NAMES,
        contact_body_names: Sequence[str] = SUGAR_CONTACT_BODY_NAMES,
        anchor_body_name: str = "torso_link",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        tree = self._load_tree(urdf_or_xml_path)
        self.num_links = len(tree)
        self.link_names: list[str] = [row[1] for row in tree]
        self.link_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.link_names)}

        self.tree_joint_names: list[str] = [row[2] for row in tree if row[2] is not None]
        assert len(self.tree_joint_names) == 29, f"Expected 29 hinge joints, got {len(self.tree_joint_names)}"

        if env_joint_names is None:
            env_joint_names = G1_29DOF_SDK_JOINT_NAMES
        self.env_joint_names: list[str] = list(env_joint_names)
        assert len(self.env_joint_names) == 29, f"Expected 29 env joints, got {len(self.env_joint_names)}"

        # Map each link (1..num_links-1) to its parent link index and env joint index (-1 if fixed)
        self.parent_indices: list[int] = [-1]
        self.link_env_joint_idx: list[int] = [-1]
        env_j_map = {name: i for i, name in enumerate(self.env_joint_names)}

        rel_pos_list = [[0.0, 0.0, 0.0]]
        rel_rot_list = [torch.eye(3, dtype=torch.float32)]
        axis_list = [[0.0, 0.0, 1.0]]
        joint_limits: dict[str, tuple[float, float]] = {}

        for row in tree[1:]:
            parent_name, _, j_name, pos, quat, axis, lo, hi = row
            self.parent_indices.append(self.link_to_idx[parent_name])
            if j_name is not None:
                if j_name not in env_j_map:
                    raise KeyError(f"Joint '{j_name}' from URDF not found in env_joint_names")
                self.link_env_joint_idx.append(env_j_map[j_name])
                joint_limits[j_name] = (lo, hi)
                axis_list.append(list(axis))
            else:
                self.link_env_joint_idx.append(-1)
                axis_list.append([0.0, 0.0, 1.0])

            rel_pos_list.append(list(pos))
            q_t = quat_normalize(torch.tensor(quat, dtype=torch.float32))
            rel_rot_list.append(quat_to_rotmat(q_t))

        # Active link indices & corresponding env joint indices for vectorized Rodrigues computation
        active_links = [i for i in range(1, self.num_links) if self.link_env_joint_idx[i] >= 0]
        active_env_joints = [self.link_env_joint_idx[i] for i in active_links]
        self._link_to_active_slot: list[int] = [-1] * self.num_links
        for slot, lk_idx in enumerate(active_links):
            self._link_to_active_slot[lk_idx] = slot

        self.register_buffer("rel_pos", torch.tensor(rel_pos_list, dtype=torch.float32), persistent=False)
        self.register_buffer("rel_rot", torch.stack(rel_rot_list, dim=0), persistent=False)
        self.register_buffer(
            "active_axes",
            torch.tensor([axis_list[i] for i in active_links], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "active_env_joint_idx",
            torch.tensor(active_env_joints, dtype=torch.long),
            persistent=False,
        )

        # Output index buffers
        self.tracked_body_names = list(tracked_body_names)
        self.contact_body_names = list(contact_body_names)
        self.anchor_body_name = anchor_body_name
        self.anchor_tracked_idx = self.tracked_body_names.index(anchor_body_name)

        self.register_buffer(
            "tracked_link_indices",
            torch.tensor([self.link_to_idx[n] for n in self.tracked_body_names], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "contact_link_indices",
            torch.tensor([self.link_to_idx[n] for n in self.contact_body_names], dtype=torch.long),
            persistent=False,
        )
        self.anchor_link_idx: int = self.link_to_idx[anchor_body_name]
        self.register_buffer(
            "link_com_pos_b",
            torch.zeros((self.num_links, 3), dtype=torch.float32),
            persistent=False,
        )

        # Precompute topological depth layers so `forward()` runs in ~11 batched depth steps instead of 33 serial links.
        # In any rooted tree, depth(u) = depth(parent(u)) + 1, so every link at depth `d` has its parent in `d - 1`.
        depths = [0] * self.num_links
        for i in range(1, self.num_links):
            depths[i] = depths[self.parent_indices[i]] + 1
        self.max_depth: int = max(depths) if self.num_links > 1 else 0
        self._num_active_joints: int = len(active_links)  # 29; sentinel slot `29` holds identity rot & zero qd

        layer_links: list[list[int]] = [
            [i for i in range(self.num_links) if depths[i] == d] for d in range(self.max_depth + 1)
        ]
        bfs_order: list[int] = [lk for grp in layer_links for lk in grp]
        bfs_pos_of_link: dict[int, int] = {lk: pos for pos, lk in enumerate(bfs_order)}

        for d in range(1, self.max_depth + 1):
            prev_local = {lk: idx for idx, lk in enumerate(layer_links[d - 1])}
            cur_links = layer_links[d]
            p_local = [prev_local[self.parent_indices[lk]] for lk in cur_links]
            slots = [
                self._link_to_active_slot[lk] if self._link_to_active_slot[lk] >= 0 else self._num_active_joints
                for lk in cur_links
            ]
            axes_d = [
                axis_list[lk] if self._link_to_active_slot[lk] >= 0 else [0.0, 0.0, 0.0]
                for lk in cur_links
            ]
            self.register_buffer(f"_d{d}_p_local", torch.tensor(p_local, dtype=torch.long), persistent=False)
            self.register_buffer(f"_d{d}_slot", torch.tensor(slots, dtype=torch.long), persistent=False)
            self.register_buffer(
                f"_d{d}_rel_pos",
                torch.tensor([rel_pos_list[lk] for lk in cur_links], dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                f"_d{d}_rel_rot", torch.stack([rel_rot_list[lk] for lk in cur_links], dim=0), persistent=False
            )
            self.register_buffer(f"_d{d}_axis", torch.tensor(axes_d, dtype=torch.float32), persistent=False)

        self.register_buffer("_eye3_slot", torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3), persistent=False)
        self.register_buffer("_bfs_order", torch.tensor(bfs_order, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_tracked_bfs_indices",
            torch.tensor([bfs_pos_of_link[self.link_to_idx[n]] for n in self.tracked_body_names], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_contact_bfs_indices",
            torch.tensor([bfs_pos_of_link[self.link_to_idx[n]] for n in self.contact_body_names], dtype=torch.long),
            persistent=False,
        )

        # Actuator parameters in `env_joint_names` order
        act = build_g1_actuator_params(self.env_joint_names, joint_limits_from_tree=joint_limits, device="cpu")
        self.register_buffer("default_joint_pos", act.default_joint_pos, persistent=False)
        self.register_buffer("action_scale", act.action_scale, persistent=False)
        self.register_buffer("stiffness", act.stiffness, persistent=False)
        self.register_buffer("damping", act.damping, persistent=False)
        self.register_buffer("effort_limit", act.effort_limit, persistent=False)
        self.register_buffer("soft_joint_lower", act.soft_lower, persistent=False)
        self.register_buffer("soft_joint_upper", act.soft_upper, persistent=False)

        self.to(device)

    @classmethod
    def _load_tree(cls, urdf_or_xml_path: str | Path | None) -> list[tuple]:
        candidates: list[Path] = []
        if urdf_or_xml_path is not None:
            candidates.append(Path(urdf_or_xml_path))
        else:
            candidates.append(Path(cls.DEFAULT_URDF_REL_PATH))

        for path in candidates:
            if path.is_file():
                suffix = path.suffix.lower()
                if suffix == ".urdf":
                    return parse_urdf_tree(path)
                if suffix == ".xml":
                    return parse_mjcf_tree(path)
                # Auto-detect by root tag
                tag = ET.parse(str(path)).getroot().tag.lower()
                return parse_urdf_tree(path) if tag == "robot" else parse_mjcf_tree(path)

        if urdf_or_xml_path is not None:
            raise FileNotFoundError(f"Specified robot description file not found: {urdf_or_xml_path}")
        return list(_BUILTIN_G1_29DOF_TREE)

    def sync_from_isaaclab_robot(self, robot, action_scale: Tensor | None = None) -> None:
        """Synchronize joint ordering, PD gains, default positions, COMs, and soft limits from IsaacLab."""
        live_names = list(robot.joint_names)
        if live_names != self.env_joint_names:
            env_j_map = {name: i for i, name in enumerate(live_names)}
            active_links = [i for i in range(1, self.num_links) if self.link_env_joint_idx[i] >= 0]
            new_active_env_joints = []
            for lk_idx in active_links:
                old_env_idx = self.link_env_joint_idx[lk_idx]
                j_name = self.env_joint_names[old_env_idx]
                new_idx = env_j_map[j_name]
                self.link_env_joint_idx[lk_idx] = new_idx
                new_active_env_joints.append(new_idx)
            self.env_joint_names = live_names
            self.active_env_joint_idx.copy_(
                torch.tensor(new_active_env_joints, dtype=torch.long, device=self.active_env_joint_idx.device)
            )

        # Copy live tensors from robot.data (prefer unrandomized nominal joint defaults when present)
        dev = self.default_joint_pos.device
        nominal_jpos = getattr(robot.data, "default_joint_pos_nominal", None)
        if nominal_jpos is not None:
            self.default_joint_pos.copy_(nominal_jpos.reshape(-1).to(dev))
        elif hasattr(robot.data, "default_joint_pos"):
            self.default_joint_pos.copy_(robot.data.default_joint_pos[0].to(dev))

        if hasattr(robot.data, "soft_joint_pos_limits"):
            self.soft_joint_lower.copy_(robot.data.soft_joint_pos_limits[0, :, 0].to(dev))
            self.soft_joint_upper.copy_(robot.data.soft_joint_pos_limits[0, :, 1].to(dev))

        # Sync per-link local COM offsets so FK reproduces IsaacLab 2.x body COM linear velocities
        try:
            if hasattr(robot.data, "body_com_pos_b"):
                coms_b = robot.data.body_com_pos_b[..., :3].float().mean(dim=0).to(dev)
            else:
                coms_b = robot.root_physx_view.get_coms()[..., :3].float().mean(dim=0).to(dev)
            live_body_map = {n: idx for idx, n in enumerate(robot.body_names)}
            com_buf = torch.zeros_like(self.link_com_pos_b)
            for lk_name, lk_idx in self.link_to_idx.items():
                if lk_name in live_body_map:
                    com_buf[lk_idx] = coms_b[live_body_map[lk_name]]
            self.link_com_pos_b.copy_(com_buf)
        except Exception:
            pass

        # Recompute exact action scales for live joint order (or override from live ActionTerm if provided)
        act = build_g1_actuator_params(self.env_joint_names, device=dev)
        if action_scale is not None:
            self.action_scale.copy_(action_scale.reshape(-1).to(dev))
        else:
            self.action_scale.copy_(act.action_scale)
        self.stiffness.copy_(act.stiffness)
        self.damping.copy_(act.damping)
        self.effort_limit.copy_(act.effort_limit)

    def forward(
        self,
        root_pos_w: Tensor,
        root_quat_w: Tensor,
        q_env: Tensor,
        root_lin_vel_w: Tensor | None = None,
        root_ang_vel_w: Tensor | None = None,
        qd_env: Tensor | None = None,
        ang_vel_in_world: bool = True,
        with_quat: bool = True,
    ) -> FKResult:
        """Compute differentiable world-frame poses and velocities for all tracked & contact bodies.

        Args:
            root_pos_w: (B, 3) pelvis position in env/world frame.
            root_quat_w: (B, 4) pelvis wxyz quaternion in world frame.
            q_env: (B, 29) joint positions in `self.env_joint_names` order.
            root_lin_vel_w: (B, 3) pelvis linear velocity in world frame (defaults to zeros).
            root_ang_vel_w: (B, 3) pelvis angular velocity in world frame if `ang_vel_in_world=True`,
                else in pelvis local frame (defaults to zeros).
            qd_env: (B, 29) joint velocities in `self.env_joint_names` order (defaults to zeros).
            ang_vel_in_world: True for IsaacLab states (`root_ang_vel_w`), False if local body frame.
            with_quat: Compute full 14-body `tracked_quat_w`; set False inside compiled BPTT
                to skip 13 unnecessary `rotmat_to_quat` branches (only `anchor_quat_w` is computed).
        """
        B = root_pos_w.shape[0]
        if root_lin_vel_w is None:
            root_lin_vel_w = torch.zeros_like(root_pos_w)
        if root_ang_vel_w is None:
            root_ang_vel_w = torch.zeros_like(root_pos_w)
        if qd_env is None:
            qd_env = torch.zeros_like(q_env)

        R_root = quat_to_rotmat(root_quat_w)
        w_root = root_ang_vel_w if ang_vel_in_world else mat_vec(R_root, root_ang_vel_w)

        # Vectorized joint rotation matrices for all 29 active joints + 1 fixed-joint identity sentinel slot
        q_active = q_env[:, self.active_env_joint_idx]     # (B, 29)
        qd_active = qd_env[:, self.active_env_joint_idx]   # (B, 29)
        R_joints = axis_unit_angle_to_rotmat(self.active_axes.unsqueeze(0), q_active)  # (B, 29, 3, 3)
        R_slots = torch.cat([R_joints, self._eye3_slot.expand(B, 1, 3, 3)], dim=1)     # (B, 30, 3, 3)
        qd_slots = F.pad(qd_active, (0, 1))                                            # (B, 30), last slot = 0

        # Depth 0: pelvis root, converting COM velocity (`root_lin_vel_w` in IsaacLab 2.x) to link-origin velocity
        root_com_w = mat_vec(R_root, self.link_com_pos_b[0].unsqueeze(0))
        v0 = root_lin_vel_w - torch.cross(w_root, root_com_w, dim=-1)

        pos_prev = root_pos_w.unsqueeze(1)   # (B, 1, 3)
        R_prev = R_root.unsqueeze(1)         # (B, 1, 3, 3)
        v_prev = v0.unsqueeze(1)             # (B, 1, 3)
        w_prev = w_root.unsqueeze(1)         # (B, 1, 3)

        pos_layers = [pos_prev]
        R_layers = [R_prev]
        v_layers = [v_prev]
        w_layers = [w_prev]

        for d in range(1, self.max_depth + 1):
            p_loc: Tensor = getattr(self, f"_d{d}_p_local")
            slot_d: Tensor = getattr(self, f"_d{d}_slot")
            r_pos_d: Tensor = getattr(self, f"_d{d}_rel_pos")
            r_rot_d: Tensor = getattr(self, f"_d{d}_rel_rot")
            axis_d: Tensor = getattr(self, f"_d{d}_axis")

            R_p = R_prev[:, p_loc]           # (B, K, 3, 3)
            pos_p = pos_prev[:, p_loc]       # (B, K, 3)
            v_p = v_prev[:, p_loc]           # (B, K, 3)
            w_p = w_prev[:, p_loc]           # (B, K, 3)

            offset_w = mat_vec(R_p, r_pos_d.unsqueeze(0))          # (B, K, 3)
            pos_d = pos_p + offset_w
            R_base = mat_mul(R_p, r_rot_d.unsqueeze(0))            # (B, K, 3, 3)
            R_d = mat_mul(R_base, R_slots[:, slot_d])              # (B, K, 3, 3)
            axis_w = mat_vec(R_base, axis_d.unsqueeze(0))          # (B, K, 3)
            w_d = w_p + qd_slots[:, slot_d].unsqueeze(-1) * axis_w
            v_d = v_p + torch.cross(w_p, offset_w, dim=-1)

            pos_layers.append(pos_d)
            R_layers.append(R_d)
            v_layers.append(v_d)
            w_layers.append(w_d)

            pos_prev, R_prev, v_prev, w_prev = pos_d, R_d, v_d, w_d

        bfs_pos = torch.cat(pos_layers, dim=1)      # (B, L, 3)
        bfs_mat = torch.cat(R_layers, dim=1)        # (B, L, 3, 3)
        bfs_v_orig = torch.cat(v_layers, dim=1)     # (B, L, 3)
        bfs_w = torch.cat(w_layers, dim=1)          # (B, L, 3)

        # Convert link-origin linear velocities back to per-link COM velocities matching `robot.data.body_lin_vel_w`
        com_bfs = self.link_com_pos_b[self._bfs_order]
        bfs_com_w = mat_vec(bfs_mat, com_bfs.unsqueeze(0))
        bfs_v = bfs_v_orig + torch.cross(bfs_w, bfs_com_w, dim=-1)

        t_idx = self._tracked_bfs_indices
        tracked_pos = bfs_pos[:, t_idx]
        tracked_mat = bfs_mat[:, t_idx]
        tracked_v = bfs_v[:, t_idx]
        tracked_w = bfs_w[:, t_idx]

        c_idx = self._contact_bfs_indices
        contact_pos = bfs_pos[:, c_idx]
        contact_v = bfs_v[:, c_idx]

        a_idx = self.anchor_tracked_idx
        if with_quat:
            tracked_quat: Tensor | None = rotmat_to_quat(tracked_mat)
            anchor_quat = tracked_quat[:, a_idx]
        else:
            tracked_quat = None
            anchor_quat = rotmat_to_quat(tracked_mat[:, a_idx])

        return FKResult(
            tracked_pos_w=tracked_pos,
            tracked_rotmat_w=tracked_mat,
            tracked_quat_w=tracked_quat,
            tracked_lin_vel_w=tracked_v,
            tracked_ang_vel_w=tracked_w,
            contact_pos_w=contact_pos,
            contact_lin_vel_w=contact_v,
            anchor_pos_w=tracked_pos[:, a_idx],
            anchor_quat_w=anchor_quat,
            anchor_rotmat_w=tracked_mat[:, a_idx],
            anchor_lin_vel_w=tracked_v[:, a_idx],
            anchor_ang_vel_w=tracked_w[:, a_idx],
            root_rotmat_w=R_root,
        )

    @torch.no_grad()
    def compute_joint_weights(self, eps: float = 1e-4) -> Tensor:
        """Compute error-propagation importance weights for all 29 joints (mean = 1.0).

        Following SuperMimic (`joint_weights.py`), measures how much a small perturbation `eps`
        on joint `j` displaces all tracked bodies in 3D space at the default standing pose.
        Root/hip/waist joints naturally receive higher weights than wrist joints.
        """
        dev = self.default_joint_pos.device
        root_pos = torch.tensor([[0.0, 0.0, 0.76]], dtype=torch.float32, device=dev)
        root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
        q0 = self.default_joint_pos.unsqueeze(0)  # (1, 29)

        base_fk = self.forward(root_pos, root_quat, q0)
        base_pos = base_fk.tracked_pos_w[0]       # (14, 3)

        impacts = torch.zeros(29, dtype=torch.float32, device=dev)
        for j in range(29):
            q_pert = q0.clone()
            q_pert[0, j] += eps
            pert_fk = self.forward(root_pos, root_quat, q_pert)
            disp = (pert_fk.tracked_pos_w[0] - base_pos).norm(dim=-1) / eps  # (14,)
            # Root-mean-square displacement across all 14 tracked bodies + floor to keep wrist active
            impacts[j] = torch.sqrt((disp * disp).mean()).clamp_min(0.02)

        return impacts / impacts.mean()

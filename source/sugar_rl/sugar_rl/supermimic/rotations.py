"""Pure-PyTorch differentiable 3D rotation utilities (wxyz quaternion convention).

Strictly aligned with IsaacLab (`isaaclab.utils.math`) and SUGAR (`sugar_rl/.../observations.py`)
while remaining smooth and NaN-free for 32-step BPTT through the world model.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quat_normalize(q: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize quaternion in wxyz format."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conj(q: Tensor) -> Tensor:
    """Quaternion conjugate for wxyz format: (w, -x, -y, -z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_inv(q: Tensor) -> Tensor:
    """Inverse of unit (or non-unit) quaternion in wxyz format."""
    norm_sq = (q * q).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return quat_conj(q) / norm_sq


def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two wxyz quaternions (broadcastable)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3D vector `v` by unit wxyz quaternion `q`."""
    q_u = quat_normalize(q)
    w = q_u[..., :1]
    xyz = q_u[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3D vector `v` by the inverse of unit wxyz quaternion `q`."""
    return quat_apply(quat_conj(q), v)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert wxyz quaternion (..., 4) to rotation matrix (..., 3, 3).

    Matches `isaaclab.utils.math.matrix_from_quat` exactly.
    """
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix (..., 3, 3) to unit wxyz quaternion (..., 4) via Shepperd's method."""
    batch_shape = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    trace = m00 + m11 + m22

    s0 = torch.sqrt((trace + 1.0).clamp_min(1e-10)) * 2.0
    q0 = torch.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], dim=-1)

    s1 = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-10)) * 2.0
    q1 = torch.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], dim=-1)

    s2 = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-10)) * 2.0
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], dim=-1)

    s3 = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-10)) * 2.0
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], dim=-1)

    cond0 = trace > 0.0
    cond1 = (m00 > m11) & (m00 > m22)
    cond2 = m11 > m22

    q = torch.where(
        cond0.unsqueeze(-1),
        q0,
        torch.where(cond1.unsqueeze(-1), q1, torch.where(cond2.unsqueeze(-1), q2, q3)),
    )
    # Canonicalize hemisphere (w >= 0 when possible) for continuity
    sign = torch.where(q[:, :1] < 0.0, -torch.ones_like(q[:, :1]), torch.ones_like(q[:, :1]))
    q = quat_normalize(q * sign)
    return q.reshape(*batch_shape, 4)


def rot6d_col(R: Tensor) -> Tensor:
    """Column-major 6D rotation representation `[c0, c1]` used inside world model features."""
    return R[..., :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def sugar_rot6d_from_mat(R: Tensor) -> Tensor:
    """SUGAR's exact 6D rotation representation: `mat[..., :2].reshape(..., 6)`.

    Note: In `sugar_rl/tasks/locomanip/mdp/observations.py`, `mat[..., :2]` of shape `(..., 3, 2)`
    is reshaped directly without transpose, yielding `[R00, R01, R10, R11, R20, R21]`.
    """
    return R[..., :2].reshape(*R.shape[:-2], 6)


def sugar_rot6d_from_quat(q: Tensor) -> Tensor:
    """SUGAR's exact 6D rotation representation from wxyz quaternion."""
    return sugar_rot6d_from_mat(quat_to_rotmat(q))


def subtract_frame_transforms(
    t01: Tensor, q01: Tensor, t02: Tensor, q02: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """Match `isaaclab.utils.math.subtract_frame_transforms` in pure differentiable PyTorch.

    Computes pose of frame 2 in frame 1 given both in frame 0:
      t12 = R(q01)^T (t02 - t01)
      q12 = q01^{-1} * q02
    """
    q10 = quat_conj(quat_normalize(q01))
    t12 = quat_apply(q10, t02 - t01)
    if q02 is None:
        return t12, None
    q12 = quat_normalize(quat_mul(q10, quat_normalize(q02)))
    return t12, q12


def sugar_anchor_vel_rotmat(anchor_quat_w: Tensor) -> Tensor:
    """Exact reproduction of SUGAR's `observations.py` velocity projection matrix.

    In `observations.py` (lines 118-121, 531-539), SUGAR computes:
        quat_inv = robot_anchor_quat_w.clone()
        quat_inv[:, :3] = -quat_inv[:, :3]
        rot_mat = matrix_from_quat(quat_inv)
    Reproducing this identically guarantees zero discrepancy between IsaacLab's
    `PrivilegedCfg` / `TeacherCfg` observations and world-model imagination observations.
    """
    q_quirk = torch.cat([-anchor_quat_w[..., :3], anchor_quat_w[..., 3:]], dim=-1)
    return quat_to_rotmat(q_quirk)


def axis_angle_to_quat(aa: Tensor) -> Tensor:
    """Exponential map from 3D axis-angle vector (..., 3) to wxyz unit quaternion (..., 4).

    Uses Taylor expansion around zero angle for smooth 2nd-order gradients.
    """
    angle_sq = (aa * aa).sum(dim=-1, keepdim=True)
    angle = torch.sqrt(angle_sq + 1e-12)
    half = 0.5 * angle
    # sin(half) / angle = 0.5 * sinc(half / pi); Taylor near 0: 0.5 - angle^2 / 48
    k_small = 0.5 - angle_sq / 48.0
    k_large = torch.sin(half) / angle
    k = torch.where(angle_sq < 1e-8, k_small, k_large)
    w = torch.where(angle_sq < 1e-8, 1.0 - angle_sq / 8.0, torch.cos(half))
    return quat_normalize(torch.cat([w, aa * k], dim=-1))


def axis_angle_to_rotmat(aa: Tensor) -> Tensor:
    """Rodrigues formula from 3D axis-angle vector (..., 3) to rotation matrix (..., 3, 3)."""
    return quat_to_rotmat(axis_angle_to_quat(aa))


def axis_unit_angle_to_rotmat(axis: Tensor, angle: Tensor) -> Tensor:
    """Rodrigues rotation matrix for unit axes `axis` (..., 3) and scalar angles `angle` (...).

    Vectorized for batched forward kinematics across all joints simultaneously.
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_minus_c = 1.0 - c
    ux, uy, uz = axis.unbind(-1)

    r00 = c + ux * ux * one_minus_c
    r01 = ux * uy * one_minus_c - uz * s
    r02 = ux * uz * one_minus_c + uy * s
    r10 = uy * ux * one_minus_c + uz * s
    r11 = c + uy * uy * one_minus_c
    r12 = uy * uz * one_minus_c - ux * s
    r20 = uz * ux * one_minus_c - uy * s
    r21 = uz * uy * one_minus_c + ux * s
    r22 = c + uz * uz * one_minus_c

    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).unflatten(-1, (3, 3))


def quat_to_axis_angle(q: Tensor) -> Tensor:
    """Smooth logarithmic map from wxyz quaternion (..., 4) to 3D axis-angle vector (..., 3)."""
    q = quat_normalize(q)
    # Choose shortest arc hemisphere (w >= 0)
    sign = torch.where(q[..., :1] < 0.0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half_sq = (xyz * xyz).sum(dim=-1, keepdim=True)
    sin_half = torch.sqrt(sin_half_sq + 1e-12)
    half_angle = torch.atan2(sin_half, w)
    scale = torch.where(sin_half_sq < 1e-8, 2.0 * (1.0 + sin_half_sq / 6.0), 2.0 * half_angle / sin_half)
    return xyz * scale


def quat_error_magnitude(q1: Tensor, q2: Tensor) -> Tensor:
    """Differentiable geodesic angle error in radians [0, pi], matching IsaacLab's `quat_error_magnitude`.

    Uses `2 * atan2(||xyz||_safe, |w|)` instead of `acos` so gradients never blow up at 0 error.
    """
    dq = quat_mul(quat_normalize(q1), quat_conj(quat_normalize(q2)))
    w_abs = dq[..., 0].abs().clamp_max(1.0)
    xyz_norm = torch.sqrt((dq[..., 1:] * dq[..., 1:]).sum(dim=-1) + 1e-12)
    return 2.0 * torch.atan2(xyz_norm, w_abs)


def yaw_from_quat(q: Tensor) -> Tensor:
    """Extract yaw angle around world +Z from wxyz quaternion (matches `isaaclab.utils.math.yaw_quat`)."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(q: Tensor) -> Tensor:
    """Pure-yaw wxyz quaternion extracted from `q`, matching `isaaclab.utils.math.yaw_quat`."""
    half_yaw = 0.5 * yaw_from_quat(q)
    zeros = torch.zeros_like(half_yaw)
    return quat_normalize(torch.stack([torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=-1))


def heading_rotmat(q: Tensor) -> Tensor:
    """2D/3D yaw rotation matrix R_z(yaw) from wxyz quaternion."""
    yaw = yaw_from_quat(q)
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([c, -s, z, s, c, z, z, z, o], dim=-1).unflatten(-1, (3, 3))


def heading_inv_rotmat(q: Tensor) -> Tensor:
    """Inverse yaw rotation matrix R_z(-yaw) = R_z(yaw)^T from wxyz quaternion."""
    return heading_rotmat(q).transpose(-1, -2)


def quat_from_euler_xyz(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """Convert extrinsic XYZ (roll-pitch-yaw, URDF standard) Euler angles to wxyz quaternion."""
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_normalize(torch.stack([w, x, y, z], dim=-1))

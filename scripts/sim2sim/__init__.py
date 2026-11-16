"""Local MuJoCo Sim-to-Sim deployment suite for SuperMimic x SUGAR (all 6 HOI tasks)."""

from .scene_builder import (
    G1_ISAACLAB_BFS_JOINT_NAMES,
    SIM2SIM_TASK_SPECS,
    Sim2SimTaskSpec,
    build_mujoco_scene_xml,
    resolve_task_spec,
)

__all__ = [
    "G1_ISAACLAB_BFS_JOINT_NAMES",
    "SIM2SIM_TASK_SPECS",
    "Sim2SimTaskSpec",
    "build_mujoco_scene_xml",
    "resolve_task_spec",
]

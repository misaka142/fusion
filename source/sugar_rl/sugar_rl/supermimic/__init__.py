"""SuperMimic differentiable world-model BPTT package fused into SUGAR."""

from .env_bridge import SugarEnvBridge, assert_privileged_obs_aligned
from .hoi_task import (
    CONTACT_DIM,
    DELTA_DIM,
    FUTURE_FRAMES,
    HISTORY_LEN,
    PRIVILEGED_OBS_DIM,
    PRIVILEGED_OBS_SLICES,
    RefFrame,
    RefFutureWindow,
    SDIM,
    SugarHOITask,
    TASK_PROFILES,
    TRACKER_CMD_DIM,
    TRACKER_OBS_DIM,
    TrackerHistoryState,
    WM_STATE_FEAT_DIM,
    XI_DIM,
    resolve_task_profile,
)
from .networks import JointWorldModel, RunningMeanStd, SuperMimicCritic, SuperMimicPolicy
from .replay import HOIReplayBuffer, ImaginationStartBatch, WMSequenceBatch
from .trainer import SuperMimicTrainCfg, SuperMimicTrainer, load_supermimic_policy
from .urdf_kinematics import (
    FKResult,
    FastTreeKinematics,
    G1_29DOF_SDK_JOINT_NAMES,
    SUGAR_CONTACT_BODY_NAMES,
    SUGAR_TRACKED_BODY_NAMES,
)

__all__ = [
    "CONTACT_DIM",
    "DELTA_DIM",
    "FKResult",
    "FUTURE_FRAMES",
    "FastTreeKinematics",
    "G1_29DOF_SDK_JOINT_NAMES",
    "HISTORY_LEN",
    "HOIReplayBuffer",
    "ImaginationStartBatch",
    "JointWorldModel",
    "PRIVILEGED_OBS_DIM",
    "PRIVILEGED_OBS_SLICES",
    "RefFrame",
    "RefFutureWindow",
    "RunningMeanStd",
    "SDIM",
    "SUGAR_CONTACT_BODY_NAMES",
    "SUGAR_TRACKED_BODY_NAMES",
    "SugarEnvBridge",
    "SugarHOITask",
    "SuperMimicCritic",
    "SuperMimicPolicy",
    "SuperMimicTrainCfg",
    "SuperMimicTrainer",
    "TASK_PROFILES",
    "TRACKER_CMD_DIM",
    "TRACKER_OBS_DIM",
    "TrackerHistoryState",
    "WMSequenceBatch",
    "WM_STATE_FEAT_DIM",
    "XI_DIM",
    "assert_privileged_obs_aligned",
    "load_supermimic_policy",
    "resolve_task_profile",
]

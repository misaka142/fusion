#!/usr/bin/env bash
# =============================================================================
# SuperMimic x SUGAR Closed-Loop Inference & Video Recording
#
# Usage:
#   bash inference_fusion.sh <TASK_NAME> <EXP_NAME> [NUM_ENVS] [VIDEO_LEN]
#   (Optional env override: MOTION_DIR=data/OmniCarryBox bash inference_fusion.sh CarryBox sm_v2_diplo_lafan)
# =============================================================================
set -uo pipefail

TASK_NAME="${1:-CarryBox}"
EXP_NAME="${2:-sm_v2_diplo_lafan}"
NUM_ENVS="${3:-4}"
VIDEO_LEN="${4:-200}"

if [[ -z "${MOTION_DIR:-}" ]]; then
    if [[ "$TASK_NAME" == "CarryBox" && -d "data/OmniCarryBox" ]]; then
        MOTION_DIR="data/OmniCarryBox"
    else
        MOTION_DIR="data/${TASK_NAME}"
    fi
fi

OUT_DIR="outputs/${TASK_NAME}_${EXP_NAME}"
TRACKER_CKPT="${OUT_DIR}/ckpts/tracker.pt"
GENERATOR_CKPT="${OUT_DIR}/ckpts/generator.ckpt"
VIDEO_DIR="$(dirname "$(realpath "${TRACKER_CKPT}")")/videos/play"

python -u scripts/supermimic/play.py \
    --task "Sugar-G129dof-${TASK_NAME}-Inference" \
    --checkpoint "${TRACKER_CKPT}" \
    --generator_checkpoint "${GENERATOR_CKPT}" \
    --motion_folder "${MOTION_DIR}" \
    --num_envs "${NUM_ENVS}" \
    --eval_random_motion \
    --headless \
    --video \
    --video_length "${VIDEO_LEN}"
rc=$?
if [[ $rc -ne 0 && $rc -ne 139 ]]; then
    exit $rc
fi
echo "[SuperMimic Inference] Completed! Videos saved under: ${VIDEO_DIR}"

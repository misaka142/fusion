#!/usr/bin/env bash
# =============================================================================
# SuperMimic x SUGAR Full 5-Stage Training Pipeline
# Replaces BOTH Stage 1 Refiner (PPO) and Stage 2 Tracker (BCPPO) with SuperMimic!
#
# Remote server usage (requires real-time quota check + interactive `trainreg run --`):
#   cd /home/xiaoji/YXM/gowork/fusion
#   trainreg balance --name-id yexiaoming --json
#   trainreg run -- bash train_fusion.sh <TASK_NAME> <EXP_NAME> [RL_ITERS] [NUM_ENVS]
# Example:
#   trainreg run -- bash train_fusion.sh CarryBox sm_run1 5001 2048
# =============================================================================
set -uo pipefail

# Enforce `trainreg run --` ancestry when `trainreg` is installed on this Linux server
if command -v trainreg >/dev/null 2>&1; then
    _has_trainreg=0
    _pid=$$
    while [[ $_pid -gt 1 ]]; do
        _comm="$(cat "/proc/${_pid}/comm" 2>/dev/null || true)"
        _cmd="$(tr '\0' ' ' < "/proc/${_pid}/cmdline" 2>/dev/null || true)"
        if [[ "$_comm" == *"trainreg"* || "$_cmd" == *"trainreg"* ]]; then
            _has_trainreg=1
            break
        fi
        _pid="$(awk '{print $4}' "/proc/${_pid}/stat" 2>/dev/null || echo 1)"
    done
    if [[ $_has_trainreg -ne 1 ]]; then
        echo "[trainreg Compliance Error] 'trainreg' is installed on this server." >&2
        echo "Direct execution without 'trainreg run --' is forbidden." >&2
        echo "  1) Check live balance: trainreg balance --name-id yexiaoming --json" >&2
        echo "  2) Launch pipeline:    trainreg run -- bash $0 $*" >&2
        exit 2
    fi
fi

# Helper: tolerate Omniverse C++ teardown exit code 139 when expected output exists
run_isaac_stage() {
    local expected_path="$1"
    shift
    set +e
    "$@"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        if [[ $rc -eq 139 && -e "$expected_path" ]]; then
            echo "[WARN] Isaac Sim exited with teardown code 139, but output '${expected_path}' is intact. Continuing..."
        else
            echo "[ERROR] Command failed with exit code ${rc} (expected '${expected_path}')."
            exit $rc
        fi
    fi
}
set -e

TASK_NAME="${1:-CarryBox}"
EXP_NAME="${2:-sm_run1}"
RL_ITERS="${3:-5001}"
NUM_ENVS="${4:-2048}"
ROLLOUT_ENVS="${5:-1000}"
IL_EPOCHS="${6:-1001}"

LAST_ITER=$((RL_ITERS - 1))
LAST_EPOCH=$((IL_EPOCHS - 1))

if [[ "$TASK_NAME" == "CarryBox" || "$TASK_NAME" == "PickBox" || "$TASK_NAME" == "PushBox" ]]; then
    USE_TARGET="True"
else
    USE_TARGET="False"
fi

OUT_DIR="outputs/${TASK_NAME}_${EXP_NAME}"
mkdir -p "${OUT_DIR}/ckpts"

echo "================================================================"
echo "[1/5] Stage 1: SuperMimic Refiner Training (${TASK_NAME})"
echo "================================================================"
run_isaac_stage "${OUT_DIR}/logs/refiner/model_${LAST_ITER}.pt" \
    python -u scripts/supermimic/train.py \
    --task "Sugar-G129dof-${TASK_NAME}-Refiner" \
    --num_envs "${NUM_ENVS}" \
    --log_dir "${OUT_DIR}/logs/refiner" \
    --max_iterations "${RL_ITERS}" \
    --motion_folder "data/${TASK_NAME}" \
    --auto_resume \
    --headless

cp "${OUT_DIR}/logs/refiner/model_${LAST_ITER}.pt" "${OUT_DIR}/ckpts/refiner.pt"

echo "================================================================"
echo "[2/5] Stage 2: SuperMimic Refiner Rollout & Dataset Processing"
echo "================================================================"
run_isaac_stage "${OUT_DIR}/rollout_datasets/refiner/raw_npz" \
    python -u scripts/supermimic/play.py \
    --task "Sugar-G129dof-${TASK_NAME}-Refiner-Rollout" \
    --num_envs "${ROLLOUT_ENVS}" \
    --checkpoint "${OUT_DIR}/ckpts/refiner.pt" \
    --rollout_dir "${OUT_DIR}/rollout_datasets/refiner/raw_npz" \
    --motion_folder "data/${TASK_NAME}" \
    --headless

python -u scripts/sugar_rl/process_refiner_rollout.py \
    --data_dir "${OUT_DIR}/rollout_datasets/refiner" \
    --task_name "${TASK_NAME}"

echo "================================================================"
echo "[3/5] Stage 3: SuperMimic Tracker Training (WM-BPTT + Teacher BC)"
echo "================================================================"
run_isaac_stage "${OUT_DIR}/logs/tracker/model_${LAST_ITER}.pt" \
    python -u scripts/supermimic/train.py \
    --task "Sugar-G129dof-${TASK_NAME}-Tracker" \
    --num_envs "${NUM_ENVS}" \
    --teacher_ckpt "${OUT_DIR}/ckpts/refiner.pt" \
    --motion_folder "${OUT_DIR}/rollout_datasets/refiner/rl_dataset" \
    --teacher_motion_folder "data/${TASK_NAME}" \
    --log_dir "${OUT_DIR}/logs/tracker" \
    --max_iterations "${RL_ITERS}" \
    --auto_resume \
    --headless

cp "${OUT_DIR}/logs/tracker/model_${LAST_ITER}.pt" "${OUT_DIR}/ckpts/tracker.pt"

echo "================================================================"
echo "[4/5] Stage 4: SuperMimic Tracker Rollout & Zarr IL Processing"
echo "================================================================"
run_isaac_stage "${OUT_DIR}/rollout_datasets/tracker/raw_npz" \
    python -u scripts/supermimic/play.py \
    --task "Sugar-G129dof-${TASK_NAME}-Tracker-Rollout" \
    --checkpoint "${OUT_DIR}/ckpts/tracker.pt" \
    --num_envs "${ROLLOUT_ENVS}" \
    --rollout_dir "${OUT_DIR}/rollout_datasets/tracker/raw_npz" \
    --motion_folder "${OUT_DIR}/rollout_datasets/refiner/rl_dataset" \
    --teacher_motion_folder "data/${TASK_NAME}" \
    --teacher_ckpt "${OUT_DIR}/ckpts/refiner.pt" \
    --headless

python -u scripts/sugar_rl/process_tracker_rollout.py \
    --data_dir "${OUT_DIR}/rollout_datasets/tracker"

echo "================================================================"
echo "[5/5] Stage 5: Diffusion Command Generator Training (sugar_il)"
echo "================================================================"
python -u scripts/sugar_il/train.py \
    --config-name train_generator_workspace.yaml \
    task="${TASK_NAME}" \
    use_target="${USE_TARGET}" \
    num_epochs="${IL_EPOCHS}" \
    log_path="${OUT_DIR}/logs/generator" \
    dataset_path="${OUT_DIR}/rollout_datasets/tracker/il_dataset"

cp "${OUT_DIR}/logs/generator/epoch_checkpoints/epoch=${LAST_EPOCH}.ckpt" "${OUT_DIR}/ckpts/generator.ckpt"

echo "================================================================"
echo "All 5 stages complete! Checkpoints stored in: ${OUT_DIR}/ckpts"
echo "================================================================"

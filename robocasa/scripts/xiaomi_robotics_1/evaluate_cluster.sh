#!/usr/bin/env bash
set -euo pipefail

# Run Xiaomi's released RoboCasa365 evaluator with cluster-safe storage,
# pinned artifacts, server health checks, and strict result validation.

XR1_COMMIT=${XR1_COMMIT:-4da1db0a4deefa6de7ebb4ef0b8754017290f5f7}
XR1_HF_REVISION=${XR1_HF_REVISION:-0d1aa76d0d82debc9b611e4d1e231096434d5be4}

PROJECT_FS=${PROJECT_FS:-/gs/fs/tga-shinoda/felid}
STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
ROBOCASA_REPO=${ROBOCASA_REPO:-${PROJECT_FS}/robocasa}
XR1_REPO=${XR1_REPO:-${PROJECT_FS}/robocasa_benchmark_repos/Xiaomi-Robotics-1}
XR1_SERVER_ENV=${XR1_SERVER_ENV:-${STORAGE_BS}/envs/xiaomi_robotics_1_server}
XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-${STORAGE_BS}/envs/xiaomi_robotics_1_robocasa365}
XR1_CHECKPOINT=${XR1_CHECKPOINT:-${STORAGE_BS}/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365}
HF_HOME=${HF_HOME:-${STORAGE_BS}/hf_home}
ROBOCASA_LOG_ROOT=${ROBOCASA_LOG_ROOT:-${STORAGE_BS}/robocasa_logs}
XR1_ROLLOUT_ROOT=${XR1_ROLLOUT_ROOT:-${STORAGE_BS}/robocasa_rollouts/xiaomi_robotics_1}

SERVER_PYTHON=${SERVER_PYTHON:-${XR1_SERVER_ENV}/bin/python}
CLIENT_PYTHON=${CLIENT_PYTHON:-${XR1_CLIENT_ENV}/bin/python}
BASE_PORT=${BASE_PORT:-10086}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}

MODE=${1:-}
if [[ -n "${MODE}" ]]; then
    shift
fi

GPU_IDS=${GPU_IDS:-0}
NUM_TRIALS=
SEED=${SEED:-7}
SMOKE_TASK=${SMOKE_TASK:-CloseBlenderLid}
SELECTED_TASK=
MAX_TASKS=
HORIZON=
DRY_RUN=0
SAVE_VIDEOS=0
SAVE_FAILURE_VIDEOS=0
RUN_TAG=${RUN_TAG:-xiaomi_robotics_1_$(date +%Y%m%d_%H%M%S)}

usage() {
    cat <<'EOF'
Usage:
  evaluate_cluster.sh preflight [options]
  evaluate_cluster.sh smoke [options]
  evaluate_cluster.sh run [options]
  evaluate_cluster.sh summarize [options]

Modes:
  preflight   Verify pinned source, environments, checkpoint, ports, and GPUs.
  smoke       Load one server and run one short CloseBlenderLid rollout.
  run         Evaluate target50; defaults to the official 50 trials per task.
  summarize   Revalidate an existing run without loading the model.

Options:
  --gpus IDS             Comma-separated physical GPU ids (default: 0).
  --episodes N           Trials per selected task (run default: 50).
  --task NAME            Evaluate one target50 task.
  --max-tasks N          Evaluate the first N target50 tasks.
  --horizon N            Override the official task horizon.
  --seed N               Environment base seed (official default: 7).
  --base-port N          First model-server port (default: 10086).
  --run-tag TAG          Stable result, scheduler, and log identifier.
  --save-videos          Save every rollout video.
  --save-failure-videos  Save only failed rollout videos.
  --dry-run              Print the launch without starting processes.
  -h, --help             Show this help.

The official full protocol is fixed to split=pretrain, task_set=target50,
obs_history=4, obs_interval=2, replan_steps=16, and crop_ratio=0.95.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPU_IDS=$2; shift ;;
        --episodes) NUM_TRIALS=$2; shift ;;
        --task) SELECTED_TASK=$2; shift ;;
        --max-tasks) MAX_TASKS=$2; shift ;;
        --horizon) HORIZON=$2; shift ;;
        --seed) SEED=$2; shift ;;
        --base-port) BASE_PORT=$2; shift ;;
        --run-tag) RUN_TAG=$2; shift ;;
        --save-videos) SAVE_VIDEOS=1 ;;
        --save-failure-videos) SAVE_FAILURE_VIDEOS=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

case "${MODE}" in
    preflight|smoke|run|summarize) ;;
    *) usage >&2; exit 2 ;;
esac

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "${BASE_PORT}" || (( BASE_PORT > 65535 )); then
    echo "--base-port must be an integer from 1 to 65535." >&2
    exit 2
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "${NUM_TRIALS}" ]] && ! is_positive_integer "${NUM_TRIALS}"; then
    echo "--episodes must be a positive integer." >&2
    exit 2
fi
if [[ -n "${MAX_TASKS}" ]] && ! is_positive_integer "${MAX_TASKS}"; then
    echo "--max-tasks must be a positive integer." >&2
    exit 2
fi
if [[ -n "${HORIZON}" ]] && ! is_positive_integer "${HORIZON}"; then
    echo "--horizon must be a positive integer." >&2
    exit 2
fi
if [[ -n "${SELECTED_TASK}" && -n "${MAX_TASKS}" ]]; then
    echo "Use either --task or --max-tasks, not both." >&2
    exit 2
fi
if [[ "${MODE}" == "smoke" ]]; then
    NUM_TRIALS=${NUM_TRIALS:-1}
    SELECTED_TASK=${SELECTED_TASK:-${SMOKE_TASK}}
    HORIZON=${HORIZON:-20}
elif [[ "${MODE}" == "run" ]]; then
    NUM_TRIALS=${NUM_TRIALS:-50}
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "At least one GPU id is required." >&2
    exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
    if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU id: ${gpu}" >&2
        exit 2
    fi
done
NUM_WORKERS=${#GPU_LIST[@]}
if (( BASE_PORT + NUM_WORKERS - 1 > 65535 )); then
    echo "Port range exceeds 65535." >&2
    exit 2
fi

RUN_ROOT=${XR1_ROLLOUT_ROOT}/${RUN_TAG}
QUEUE_DIR=${XR1_ROLLOUT_ROOT}/scheduler/${RUN_TAG}
CLIENT_LOG=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_client.log
PID_FILE=${RUN_ROOT}/server_pids.txt
SUMMARY_TOOL=${ROBOCASA_REPO}/robocasa/scripts/xiaomi_robotics_1/summarize_results.py

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file is missing: $1" >&2
        exit 1
    fi
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "Required executable is missing: $1" >&2
        exit 1
    fi
}

verify_checkpoint() {
    local filename
    for filename in \
        config.json processor_config.json model.safetensors.index.json \
        model-00001-of-00003.safetensors \
        model-00002-of-00003.safetensors \
        model-00003-of-00003.safetensors; do
        require_file "${XR1_CHECKPOINT}/${filename}"
    done
}

port_is_listening() {
    "${CLIENT_PYTHON}" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
    connection.settimeout(1.0)
    raise SystemExit(connection.connect_ex(("127.0.0.1", port)))
PY
}

verify_ports_free() {
    local index port
    for ((index = 0; index < NUM_WORKERS; index++)); do
        port=$((BASE_PORT + index))
        if port_is_listening "${port}"; then
            echo "Port ${port} is already listening; choose another --base-port." >&2
            exit 1
        fi
    done
}

preflight() {
    local current_commit
    require_executable "${SERVER_PYTHON}"
    require_executable "${CLIENT_PYTHON}"
    require_file "${XR1_REPO}/deploy/server.py"
    require_file "${XR1_REPO}/scripts/launch_robocasa365.sh"
    require_file "${XR1_REPO}/eval_robocasa365/dynamic_eval.py"
    require_file "${SUMMARY_TOOL}"
    verify_checkpoint

    current_commit=$(git -C "${XR1_REPO}" rev-parse HEAD)
    if [[ "${current_commit}" != "${XR1_COMMIT}" ]]; then
        echo "Xiaomi source commit mismatch: ${current_commit}" >&2
        echo "Expected: ${XR1_COMMIT}" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${XR1_REPO}" status --porcelain)" ]]; then
        echo "Xiaomi source checkout is dirty: ${XR1_REPO}" >&2
        exit 1
    fi

    "${SERVER_PYTHON}" -c \
        "import flash_attn, torch, transformers; assert torch.cuda.is_available(); assert torch.__version__.startswith('2.8.0'); assert transformers.__version__ == '4.57.1'; print('server:', torch.__version__, transformers.__version__, flash_attn.__version__)"
    "${CLIENT_PYTHON}" - "${XR1_CHECKPOINT}" "${ROBOCASA_REPO}" <<'PY'
import pathlib
import sys

import robocasa
import transformers
from transformers import AutoProcessor

checkpoint = pathlib.Path(sys.argv[1])
expected_repo = pathlib.Path(sys.argv[2]).resolve()
actual_repo = pathlib.Path(robocasa.__file__).resolve().parents[1]
assert robocasa.__version__ == "1.0.1", robocasa.__version__
assert transformers.__version__ == "4.57.1", transformers.__version__
assert actual_repo == expected_repo, (actual_repo, expected_repo)
processor = AutoProcessor.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
)
assert "robocasa365" in processor.list_robot_types()
print("client: robocasa", robocasa.__version__, "transformers", transformers.__version__)
print("processor robot types:", processor.list_robot_types())
PY

    verify_ports_free
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo "Pinned Xiaomi source: ${XR1_COMMIT}"
    echo "Pinned checkpoint revision: ${XR1_HF_REVISION}"
    echo "Preflight passed."
}

server_pids=()
cleanup_servers() {
    if (( ${#server_pids[@]} > 0 )); then
        echo "Stopping ${#server_pids[@]} Xiaomi model server(s)."
        kill "${server_pids[@]}" 2>/dev/null || true
        wait "${server_pids[@]}" 2>/dev/null || true
    fi
}

start_servers() {
    local index gpu port server_log pid deadline
    mkdir -p "${RUN_ROOT}" "${ROBOCASA_LOG_ROOT}/eval"
    : > "${PID_FILE}"
    trap cleanup_servers EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    for ((index = 0; index < NUM_WORKERS; index++)); do
        gpu=${GPU_LIST[$index]}
        port=$((BASE_PORT + index))
        server_log=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_server_${port}.log
        echo "Starting server ${index}: gpu=${gpu} port=${port} log=${server_log}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        HF_HOME="${HF_HOME}" \
        TOKENIZERS_PARALLELISM=false \
        nohup "${SERVER_PYTHON}" -u "${XR1_REPO}/deploy/server.py" \
            --model "${XR1_CHECKPOINT}" \
            --host 127.0.0.1 \
            --port "${port}" \
            > "${server_log}" 2>&1 &
        pid=$!
        server_pids+=("${pid}")
        printf '%s %s %s %s\n' "${pid}" "${gpu}" "${port}" "${server_log}" >> "${PID_FILE}"
    done

    deadline=$((SECONDS + SERVER_START_TIMEOUT))
    for ((index = 0; index < NUM_WORKERS; index++)); do
        port=$((BASE_PORT + index))
        server_log=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_server_${port}.log
        pid=${server_pids[$index]}
        until port_is_listening "${port}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "Server exited before listening on ${port}." >&2
                tail -100 "${server_log}" >&2 || true
                exit 1
            fi
            if (( SECONDS >= deadline )); then
                echo "Timed out waiting for server port ${port}." >&2
                tail -100 "${server_log}" >&2 || true
                exit 1
            fi
            echo "Waiting for Xiaomi model server on ${port}..."
            tail -5 "${server_log}" 2>/dev/null || true
            sleep 10
        done
        if ! grep -q "Model loaded" "${server_log}"; then
            echo "Port ${port} opened without the expected model-load marker." >&2
            tail -100 "${server_log}" >&2 || true
            exit 1
        fi
        echo "Server ready on ${port}."
    done
}

summarize_run() {
    local summary_path=${RUN_ROOT}/summary.json
    local args=(
        "${CLIENT_PYTHON}" -u "${SUMMARY_TOOL}"
        "${summary_path}"
        --queue-dir "${QUEUE_DIR}"
        --expected-episodes "${NUM_TRIALS:-50}"
        --expected-seed "${SEED}"
        --source-commit "${XR1_COMMIT}"
        --checkpoint-revision "${XR1_HF_REVISION}"
        --output "${RUN_ROOT}/reproduction_summary.json"
    )
    if [[ -n "${SELECTED_TASK}" || -n "${MAX_TASKS}" || "${NUM_TRIALS:-50}" != "50" ]]; then
        args+=(--allow-partial)
    fi
    "${args[@]}"
}

if [[ "${MODE}" == "summarize" ]]; then
    if [[ -z "${NUM_TRIALS}" ]]; then
        NUM_TRIALS=50
    fi
    summarize_run
    exit $?
fi

echo "Xiaomi-Robotics-1 RoboCasa365 evaluation"
echo "  mode                 : ${MODE}"
echo "  GPUs                 : ${GPU_IDS}"
echo "  workers              : ${NUM_WORKERS}"
echo "  ports                : ${BASE_PORT}-$((BASE_PORT + NUM_WORKERS - 1))"
echo "  source               : ${XR1_REPO}"
echo "  source commit        : ${XR1_COMMIT}"
echo "  checkpoint           : ${XR1_CHECKPOINT}"
echo "  checkpoint revision  : ${XR1_HF_REVISION}"
echo "  split / task set     : pretrain / target50"
echo "  episodes per task    : ${NUM_TRIALS:-n/a}"
echo "  seed                 : ${SEED}"
echo "  task                 : ${SELECTED_TASK:-all target50}"
echo "  max tasks            : ${MAX_TASKS:-none}"
echo "  horizon override     : ${HORIZON:-none}"
echo "  run root             : ${RUN_ROOT}"
echo "  scheduler            : ${QUEUE_DIR}"
echo "  client log           : ${CLIENT_LOG}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${MODE}" == "preflight" ]]; then
        echo "Would verify pinned source, checkpoint, environments, ports, and GPUs."
        exit 0
    fi
    for ((index = 0; index < NUM_WORKERS; index++)); do
        print_command env \
            CUDA_VISIBLE_DEVICES="${GPU_LIST[$index]}" \
            HF_HOME="${HF_HOME}" \
            "${SERVER_PYTHON}" -u "${XR1_REPO}/deploy/server.py" \
            --model "${XR1_CHECKPOINT}" \
            --host 127.0.0.1 \
            --port "$((BASE_PORT + index))"
    done
else
    preflight
fi

if [[ "${MODE}" == "preflight" ]]; then
    exit 0
fi

extra_args=()
if [[ -n "${SELECTED_TASK}" ]]; then
    extra_args+=(--task-name "${SELECTED_TASK}")
fi
if [[ -n "${MAX_TASKS}" ]]; then
    extra_args+=(--max-tasks "${MAX_TASKS}")
fi
if [[ -n "${HORIZON}" ]]; then
    extra_args+=(--horizon "${HORIZON}")
fi
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
    extra_args+=(--save-videos)
fi
if [[ "${SAVE_FAILURE_VIDEOS}" == "1" ]]; then
    extra_args+=(--save-failure-videos)
fi

if [[ "${DRY_RUN}" == "0" ]]; then
    if [[ -e "${RUN_ROOT}" || -e "${QUEUE_DIR}" || -e "${CLIENT_LOG}" ]]; then
        echo "Run tag already has result or scheduler state: ${RUN_TAG}" >&2
        echo "Use a new --run-tag; existing partial evidence was left untouched." >&2
        exit 1
    fi
    verify_ports_free
    start_servers
fi

launch_command=(
    bash "${XR1_REPO}/scripts/launch_robocasa365.sh"
    "${NUM_WORKERS}"
    "${XR1_ROLLOUT_ROOT}"
    "${XR1_CHECKPOINT}"
)
if (( ${#extra_args[@]} > 0 )); then
    launch_command+=("${extra_args[@]}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    print_command env \
        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        HF_HOME="${HF_HOME}" \
        WANDB_MODE=disabled \
        WANDB_DISABLED=true \
        WANDB_ENABLED=0 \
        BASE_PORT="${BASE_PORT}" \
        SERVER_ADDR=127.0.0.1 \
        PYTHON="${CLIENT_PYTHON}" \
        RUN_ID="${RUN_TAG}" \
        QUEUE_DIR="${QUEUE_DIR}" \
        SPLIT=pretrain \
        TASK_SET=target50 \
        NUM_TRIALS="${NUM_TRIALS}" \
        REPLAN_STEPS=16 \
        OBS_HISTORY=4 \
        OBS_INTERVAL=2 \
        SEED="${SEED}" \
        CROP_RATIO=0.95 \
        "${launch_command[@]}"
    echo "Dry run complete; no processes were started."
    exit 0
fi

echo "Starting simulator workers after all model servers passed health checks."
set +e
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
HF_HOME="${HF_HOME}" \
WANDB_MODE=disabled \
WANDB_DISABLED=true \
WANDB_ENABLED=0 \
BASE_PORT="${BASE_PORT}" \
SERVER_ADDR=127.0.0.1 \
PYTHON="${CLIENT_PYTHON}" \
RUN_ID="${RUN_TAG}" \
QUEUE_DIR="${QUEUE_DIR}" \
SPLIT=pretrain \
TASK_SET=target50 \
NUM_TRIALS="${NUM_TRIALS}" \
REPLAN_STEPS=16 \
OBS_HISTORY=4 \
OBS_INTERVAL=2 \
SEED="${SEED}" \
CROP_RATIO=0.95 \
"${launch_command[@]}" 2>&1 | tee "${CLIENT_LOG}"
launcher_status=${PIPESTATUS[0]}
set -e

if (( launcher_status != 0 )); then
    echo "Xiaomi evaluator exited with status ${launcher_status}." >&2
    echo "Partial scheduler state and logs were preserved under ${QUEUE_DIR}." >&2
    exit "${launcher_status}"
fi

summarize_run
echo "Evaluation complete: ${RUN_ROOT}/reproduction_summary.json"

#!/usr/bin/env bash
set -euo pipefail

# Run the released ABot-M0.5 RoboCasa365 evaluator with cluster-safe paths.
# The ABot model server runs from its own environment; the simulator client
# runs from the existing RoboCasa/OpenPI environment.

ABOT_COMMIT=${ABOT_COMMIT:-7642747ed2817b241dde5df06e17ee80192718ad}
PROJECT_FS=${PROJECT_FS:-/gs/fs/tga-shinoda/felid}
STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
USER_ID=${USER_ID:-ut06746}

ROBOCASA_REPO=${ROBOCASA_REPO:-${PROJECT_FS}/robocasa}
ABOT_REPO=${ABOT_REPO:-${PROJECT_FS}/robocasa_benchmark_repos/ABot-Manipulation}
ABOT_ENV=${ABOT_ENV:-${STORAGE_BS}/envs/abot_m05}
ROBOCASA_CLIENT_ENV=${ROBOCASA_CLIENT_ENV:-${STORAGE_BS}/envs/robocasa_openpi}
CLIENT_PYTHON=${CLIENT_PYTHON:-${ROBOCASA_CLIENT_ENV}/bin/python}

ABOT_CHECKPOINT_ROOT=${ABOT_CHECKPOINT_ROOT:-${STORAGE_BS}/robocasa_checkpoints/abot_m05/ABot-M0.5-RoboCasa365}
ABOT_LOCAL_CHECKPOINT_ROOT=${ABOT_LOCAL_CHECKPOINT_ROOT:-/tmp/${USER_ID}/abot_m05/ABot-M0.5-RoboCasa365}
HF_HOME=${HF_HOME:-${STORAGE_BS}/hf_home}
ROBOCASA_LOG_ROOT=${ROBOCASA_LOG_ROOT:-${STORAGE_BS}/robocasa_logs}
ROBOCASA_ROLLOUT_ROOT=${ROBOCASA_ROLLOUT_ROOT:-${STORAGE_BS}/robocasa_rollouts}

MODE=${1:-}
if [[ -n "${MODE}" ]]; then
    shift
fi

GPU_IDS=${GPU_IDS:-0}
NUM_EPISODES=
SEED=${SEED:-0}
ATTN_MODE=${ATTN_MODE:-torch}
SPLIT=${SPLIT:-pretrain}
SMOKE_TASK=${SMOKE_TASK:-CloseFridge}
BACKGROUND=0
DRY_RUN=0
COPY_LOCAL=1
TRACK_SUBTASK_PROGRESS=0
RUN_TAG=${RUN_TAG:-abot_m05_$(date +%Y%m%d_%H%M%S)}
MIN_LOCAL_FREE_GIB=${MIN_LOCAL_FREE_GIB:-40}

usage() {
    cat <<'EOF'
Usage:
  evaluate_cluster.sh preflight [options]
  evaluate_cluster.sh smoke [options]
  evaluate_cluster.sh atomic_seen [options]
  evaluate_cluster.sh composite_seen [options]
  evaluate_cluster.sh composite_unseen [options]
  evaluate_cluster.sh all [options]

Options:
  --gpus IDS             Comma-separated physical GPU ids (default: 0).
  --episodes N           Episodes per task (default: 1 for smoke, 50 otherwise).
  --seed N               Base evaluation seed (default: 0).
  --task NAME            Smoke-test task (default: CloseFridge).
  --attention MODE       torch or flashattn (default: torch).
  --run-tag TAG          Stable output tag for logs, rollouts, and resume.
  --subtask-progress     Save ordered subtask progress for every rollout.
  --background           Detach a single split through the official launcher.
  --no-local-copy        Load the durable checkpoint directly from /gs/bs.
  --dry-run              Print commands without checking paths or running them.
  -h, --help             Show this help.

Full protocol:
  Split is fixed to "pretrain". Use 50 episodes per task to match the official
  RoboCasa365 benchmark. The "all" mode runs the three splits sequentially and
  cannot be combined with --background; wrap it in nohup instead.
EOF
}

case "${MODE}" in
    preflight|smoke|atomic_seen|composite_seen|composite_unseen|all)
        ;;
    -h|--help|"")
        usage
        [[ -n "${MODE}" ]] && exit 0 || exit 2
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        usage >&2
        exit 2
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_IDS=${2:?--gpus requires a value}
            shift
            ;;
        --episodes)
            NUM_EPISODES=${2:?--episodes requires a value}
            shift
            ;;
        --seed)
            SEED=${2:?--seed requires a value}
            shift
            ;;
        --task)
            SMOKE_TASK=${2:?--task requires a value}
            shift
            ;;
        --attention)
            ATTN_MODE=${2:?--attention requires a value}
            shift
            ;;
        --run-tag)
            RUN_TAG=${2:?--run-tag requires a value}
            shift
            ;;
        --subtask-progress)
            TRACK_SUBTASK_PROGRESS=1
            ;;
        --background)
            BACKGROUND=1
            ;;
        --no-local-copy)
            COPY_LOCAL=0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ -z "${NUM_EPISODES}" ]]; then
    if [[ "${MODE}" == "smoke" ]]; then
        NUM_EPISODES=1
    else
        NUM_EPISODES=50
    fi
fi

if [[ ! "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--episodes must be a positive integer: ${NUM_EPISODES}" >&2
    exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer: ${SEED}" >&2
    exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "--gpus must be a comma-separated list of integer GPU ids: ${GPU_IDS}" >&2
    exit 2
fi
if [[ "${ATTN_MODE}" != "torch" && "${ATTN_MODE}" != "flashattn" ]]; then
    echo "--attention must be torch or flashattn: ${ATTN_MODE}" >&2
    exit 2
fi
if [[ "${SPLIT}" != "pretrain" ]]; then
    echo "ABot-M0.5 leaderboard reproduction requires SPLIT=pretrain." >&2
    exit 2
fi
if [[ "${MODE}" == "all" && "${BACKGROUND}" == "1" ]]; then
    echo "The all mode runs splits sequentially; use nohup instead of --background." >&2
    exit 2
fi
if [[ "${MODE}" == "smoke" && "${BACKGROUND}" == "1" ]]; then
    echo "Smoke evaluation is intentionally foreground so startup failures are visible." >&2
    exit 2
fi

ABOT_PYTHON=${ABOT_ENV}/bin/python
SERVER_CHECKPOINT_ROOT=${ABOT_CHECKPOINT_ROOT}
if [[ "${COPY_LOCAL}" == "1" ]]; then
    SERVER_CHECKPOINT_ROOT=${ABOT_LOCAL_CHECKPOINT_ROOT}
fi
POSTTRAIN_CHECKPOINT=${SERVER_CHECKPOINT_ROOT}/checkpoint_step
BASE_CHECKPOINT=${SERVER_CHECKPOINT_ROOT}/base_checkpoint
CLIENT_PYTHONPATH=${ABOT_REPO}:${ROBOCASA_REPO}
if [[ "${TRACK_SUBTASK_PROGRESS}" == "1" ]]; then
    CLIENT_PYTHONPATH=${ROBOCASA_REPO}/robocasa/scripts/abot_m05/python_startup:${CLIENT_PYTHONPATH}
fi
WORKER_COUNT=$(awk -F, '{print NF}' <<<"${GPU_IDS}")

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command "$@"
    else
        "$@"
    fi
}

run_in_abot_repo() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  (cd %q && ' "${ABOT_REPO}"
        printf '%q ' "$@"
        printf ')\n'
    else
        (
            cd "${ABOT_REPO}"
            "$@"
        )
    fi
}

required_checkpoint_files() {
    cat <<'EOF'
base_checkpoint/vae/diffusion_pytorch_model.safetensors
base_checkpoint/text_encoder/model.safetensors.index.json
base_checkpoint/tokenizer/tokenizer.json
checkpoint_step/transformer/config.json
checkpoint_step/transformer/diffusion_pytorch_model.safetensors
EOF
}

verify_checkpoint_root() {
    local root=$1
    local relative_path
    while IFS= read -r relative_path; do
        if [[ ! -f "${root}/${relative_path}" ]]; then
            echo "Missing checkpoint artifact: ${root}/${relative_path}" >&2
            exit 1
        fi
    done < <(required_checkpoint_files)
}

preflight() {
    local current_commit
    if [[ ! -d "${ROBOCASA_REPO}/.git" ]]; then
        echo "RoboCasa checkout is missing: ${ROBOCASA_REPO}" >&2
        exit 1
    fi
    if [[ ! -d "${ABOT_REPO}/.git" ]]; then
        echo "ABot checkout is missing: ${ABOT_REPO}" >&2
        exit 1
    fi
    current_commit=$(git -C "${ABOT_REPO}" rev-parse HEAD)
    if [[ "${current_commit}" != "${ABOT_COMMIT}" ]]; then
        echo "ABot commit mismatch: expected ${ABOT_COMMIT}, found ${current_commit}" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${ABOT_REPO}" status --porcelain)" ]]; then
        echo "ABot checkout is dirty: ${ABOT_REPO}" >&2
        exit 1
    fi
    if [[ ! -x "${ABOT_PYTHON}" ]]; then
        echo "ABot Python is missing: ${ABOT_PYTHON}" >&2
        exit 1
    fi
    if [[ ! -x "${CLIENT_PYTHON}" ]]; then
        echo "RoboCasa client Python is missing: ${CLIENT_PYTHON}" >&2
        exit 1
    fi
    verify_checkpoint_root "${ABOT_CHECKPOINT_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONPATH="${ABOT_REPO}" \
        "${ABOT_PYTHON}" -c \
        "import torch, diffusers, transformers, wam; assert torch.cuda.is_available(), 'CUDA is unavailable'; assert torch.cuda.device_count() == ${WORKER_COUNT}, (torch.cuda.device_count(), ${WORKER_COUNT}); print('ABot server imports and GPUs: OK'); print('torch', torch.__version__, 'visible_gpus', torch.cuda.device_count())"
    if [[ "${ATTN_MODE}" == "flashattn" ]]; then
        PYTHONPATH="${ABOT_REPO}" "${ABOT_PYTHON}" -c \
            "from wam.modules.attention_ops import flash_attn_func; assert flash_attn_func is not None, 'flash-attn is not installed'; print('flash-attn: OK')"
    fi
    PYTHONPATH="${CLIENT_PYTHONPATH}" "${CLIENT_PYTHON}" -c \
        "import gymnasium, msgpack, scipy, websockets, robocasa; assert robocasa.__version__ == '1.0.1', robocasa.__version__; print('RoboCasa client imports: OK'); print('robocasa', robocasa.__version__)"
    echo "ABot-M0.5 preflight: OK"
    echo "Durable checkpoint: ${ABOT_CHECKPOINT_ROOT}"
    echo "Requested GPUs: ${GPU_IDS}"
}

prepare_local_checkpoint() {
    local available_kib
    local required_kib
    if [[ "${COPY_LOCAL}" != "1" ]]; then
        verify_checkpoint_root "${ABOT_CHECKPOINT_ROOT}"
        return
    fi
    run mkdir -p "${ABOT_LOCAL_CHECKPOINT_ROOT}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        available_kib=$(df -Pk "${ABOT_LOCAL_CHECKPOINT_ROOT}" | awk 'NR == 2 {print $4}')
        required_kib=$((MIN_LOCAL_FREE_GIB * 1024 * 1024))
        if [[ -z "${available_kib}" || ! "${available_kib}" =~ ^[0-9]+$ ]]; then
            echo "Could not determine node-local free space." >&2
            exit 1
        fi
        if (( available_kib < required_kib )); then
            echo "Node-local /tmp has less than ${MIN_LOCAL_FREE_GIB} GiB free." >&2
            echo "Use --no-local-copy only if loading the model directly from /gs/bs is acceptable." >&2
            exit 1
        fi
    fi
    run rsync -a --info=progress2 \
        "${ABOT_CHECKPOINT_ROOT}/" \
        "${ABOT_LOCAL_CHECKPOINT_ROOT}/"
    if [[ "${DRY_RUN}" == "0" ]]; then
        verify_checkpoint_root "${ABOT_LOCAL_CHECKPOINT_ROOT}"
    fi
}

common_env=(
    env
    HF_HOME="${HF_HOME}"
    TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    WANDB_MODE=disabled
    WANDB_DISABLED=true
    WANDB_ENABLED=0
    MUJOCO_GL=egl
    PYOPENGL_PLATFORM=egl
    TOKENIZERS_PARALLELISM=false
    CKPT_PATH="${POSTTRAIN_CHECKPOINT}"
    WAN22_PRETRAINED_PATH="${BASE_CHECKPOINT}"
    WAN22_PRETRAINED_MODEL_NAME_OR_PATH="${BASE_CHECKPOINT}"
    WRAPPER_PYTHON="${ABOT_PYTHON}"
    SERVER_PYTHON="${ABOT_PYTHON}"
    CLIENT_PYTHON="${CLIENT_PYTHON}"
    ROBOCASA_CLIENT_PYTHONPATH="${CLIENT_PYTHONPATH}"
    ROBOCASA_TRACK_SUBTASK_PROGRESS="${TRACK_SUBTASK_PROGRESS}"
    GPU_IDS="${GPU_IDS}"
    WORKER_COUNT="${WORKER_COUNT}"
    SPLIT=pretrain
    NUM_EPISODES="${NUM_EPISODES}"
    SEED="${SEED}"
    ATTN_MODE="${ATTN_MODE}"
    HORIZON_MULTIPLIER=1
    SKIP_MESA_INSTALL=1
    AUTO_AGGREGATE=1
)

run_smoke() {
    local run_root=${ROBOCASA_ROLLOUT_ROOT}/abot_m05/${RUN_TAG}/smoke_${SMOKE_TASK}
    local log_path=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_smoke_${SMOKE_TASK}.log
    local server_log=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_smoke_${SMOKE_TASK}_server.log
    run mkdir -p "${run_root}" "$(dirname "${log_path}")"
    echo "Smoke result root: ${run_root}"
    echo "Smoke log: ${log_path}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_in_abot_repo \
            "${common_env[@]}" \
            GPU_ID="${GPU_IDS%%,*}" \
            ENV_NAME="${SMOKE_TASK}" \
            RUN_ROOT="${run_root}" \
            SAVE_ROOT="${run_root}/server_predictions" \
            RESULT_DIR="${run_root}/results" \
            SAVE_VIDEO_DIR="${run_root}/videos" \
            SAVE_JSON="${run_root}/results/robocasa_eval.json" \
            SERVER_LOG="${server_log}" \
            START_PORT=31056 \
            MASTER_PORT=31100 \
            bash evaluation/robocasa/launch_server_env_sweep.sh
    else
        (
            cd "${ABOT_REPO}"
            "${common_env[@]}" \
                GPU_ID="${GPU_IDS%%,*}" \
                ENV_NAME="${SMOKE_TASK}" \
                RUN_ROOT="${run_root}" \
                SAVE_ROOT="${run_root}/server_predictions" \
                RESULT_DIR="${run_root}/results" \
                SAVE_VIDEO_DIR="${run_root}/videos" \
                SAVE_JSON="${run_root}/results/robocasa_eval.json" \
                SERVER_LOG="${server_log}" \
                START_PORT=31056 \
                MASTER_PORT=31100 \
                bash evaluation/robocasa/launch_server_env_sweep.sh
        ) 2>&1 | tee "${log_path}"
    fi
}

split_script() {
    case "$1" in
        atomic_seen)
            printf '%s\n' script/eval/local/eval_atomic_seen.sh
            ;;
        composite_seen)
            printf '%s\n' script/eval/local/eval_composite_seen.sh
            ;;
        composite_unseen)
            printf '%s\n' script/eval/local/eval_composite_unseen.sh
            ;;
        *)
            return 1
            ;;
    esac
}

split_port_base() {
    case "$1" in
        atomic_seen)
            printf '%s\n' 32000
            ;;
        composite_seen)
            printf '%s\n' 33000
            ;;
        composite_unseen)
            printf '%s\n' 34000
            ;;
        *)
            return 1
            ;;
    esac
}

run_split() {
    local split_name=$1
    local script_path
    local start_port
    local run_root
    local launcher_log
    local worker_log_dir
    script_path=$(split_script "${split_name}")
    start_port=$(split_port_base "${split_name}")
    run_root=${ROBOCASA_ROLLOUT_ROOT}/abot_m05/${RUN_TAG}/${split_name}
    launcher_log=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_${split_name}_launcher.log
    worker_log_dir=${ROBOCASA_LOG_ROOT}/eval/${RUN_TAG}_${split_name}_workers
    run mkdir -p "${run_root}" "${ROBOCASA_LOG_ROOT}/eval" "${worker_log_dir}"
    echo "${split_name} result root: ${run_root}"
    run_in_abot_repo \
        "${common_env[@]}" \
        BACKGROUND="${BACKGROUND}" \
        RUN_TAG="${RUN_TAG}_${split_name}" \
        RUN_ROOT="${run_root}" \
        LOCAL_LAUNCH_LOG_DIR="${worker_log_dir}" \
        BACKGROUND_LOG="${launcher_log}" \
        START_PORT="${start_port}" \
        MASTER_PORT_BASE="$((start_port + 100))" \
        NUMBA_CACHE_DIR="/tmp/${USER_ID}/numba_abot_m05_${split_name}" \
        bash "${script_path}"
}

echo "ABot-M0.5 evaluation"
echo "  mode                  : ${MODE}"
echo "  ABot repository       : ${ABOT_REPO}"
echo "  ABot environment      : ${ABOT_ENV}"
echo "  RoboCasa client       : ${CLIENT_PYTHON}"
echo "  durable checkpoint    : ${ABOT_CHECKPOINT_ROOT}"
echo "  server checkpoint     : ${SERVER_CHECKPOINT_ROOT}"
echo "  GPUs                   : ${GPU_IDS}"
echo "  episodes per task     : ${NUM_EPISODES}"
echo "  split                  : pretrain"
echo "  attention              : ${ATTN_MODE}"
echo "  subtask progress       : ${TRACK_SUBTASK_PROGRESS}"
echo "  run tag                : ${RUN_TAG}"
echo

if [[ "${DRY_RUN}" == "0" ]]; then
    preflight
elif [[ "${MODE}" == "preflight" ]]; then
    echo "Dry-run preflight: checks skipped."
fi

if [[ "${MODE}" == "preflight" ]]; then
    exit 0
fi

prepare_local_checkpoint

case "${MODE}" in
    smoke)
        run_smoke
        ;;
    atomic_seen|composite_seen|composite_unseen)
        run_split "${MODE}"
        ;;
    all)
        run_split atomic_seen
        run_split composite_seen
        run_split composite_unseen
        run "${CLIENT_PYTHON}" \
            "${ROBOCASA_REPO}/robocasa/scripts/abot_m05/summarize_results.py" \
            "${ROBOCASA_ROLLOUT_ROOT}/abot_m05/${RUN_TAG}" \
            --expected-episodes "${NUM_EPISODES}"
        if [[ "${TRACK_SUBTASK_PROGRESS}" == "1" ]]; then
            run "${CLIENT_PYTHON}" \
                "${ROBOCASA_REPO}/robocasa/scripts/abot_m05/summarize_subtask_progress.py" \
                "${ROBOCASA_ROLLOUT_ROOT}/abot_m05/${RUN_TAG}" \
                --expected-episodes "${NUM_EPISODES}"
        fi
        ;;
esac

echo
echo "Launch complete."
echo "Results: ${ROBOCASA_ROLLOUT_ROOT}/abot_m05/${RUN_TAG}"
echo "Logs: ${ROBOCASA_LOG_ROOT}/eval"

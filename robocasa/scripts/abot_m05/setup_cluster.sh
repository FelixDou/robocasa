#!/usr/bin/env bash
set -euo pipefail

# Install the public ABot-M0.5 inference stack and download its gated
# RoboCasa365 checkpoint using the Shinoda-lab cluster storage contract.
#
# Source code stays on /gs/fs. Environments, Hugging Face cache, and model
# weights stay on /gs/bs. This script never accepts or stores a token itself;
# authenticate first with `hf auth login` or export HF_TOKEN in the shell.

ABOT_REPOSITORY_URL=${ABOT_REPOSITORY_URL:-https://github.com/amap-cvlab/ABot-Manipulation.git}
ABOT_COMMIT=${ABOT_COMMIT:-7642747ed2817b241dde5df06e17ee80192718ad}
ABOT_HF_REPOSITORY=${ABOT_HF_REPOSITORY:-acvlab/ABot-M0.5-RoboCasa365}

PROJECT_FS=${PROJECT_FS:-/gs/fs/tga-shinoda/felid}
STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
ABOT_REPO=${ABOT_REPO:-${PROJECT_FS}/robocasa_benchmark_repos/ABot-Manipulation}
ABOT_ENV=${ABOT_ENV:-${STORAGE_BS}/envs/abot_m05}
ABOT_CHECKPOINT_ROOT=${ABOT_CHECKPOINT_ROOT:-${STORAGE_BS}/robocasa_checkpoints/abot_m05/ABot-M0.5-RoboCasa365}
HF_HOME=${HF_HOME:-${STORAGE_BS}/hf_home}
CONDA_EXE=${CONDA_EXE:-conda}

DRY_RUN=0
SKIP_ENV=0
SKIP_DOWNLOAD=0
WITH_FLASH_ATTN=0
MIN_FREE_GIB=${MIN_FREE_GIB:-60}

usage() {
    cat <<'EOF'
Usage: setup_cluster.sh [options]

Options:
  --dry-run            Print the planned operations without changing anything.
  --skip-env           Reuse an existing ABot environment.
  --skip-download      Do not download the gated checkpoint.
  --with-flash-attn    Install flash-attn 2.8.3 (torch attention is the default).
  -h, --help           Show this help.

Environment overrides:
  PROJECT_FS, STORAGE_BS, ABOT_REPO, ABOT_ENV, ABOT_CHECKPOINT_ROOT,
  HF_HOME, ABOT_REPOSITORY_URL, ABOT_COMMIT, ABOT_HF_REPOSITORY, CONDA_EXE.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --skip-env)
            SKIP_ENV=1
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            ;;
        --with-flash-attn)
            WITH_FLASH_ATTN=1
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

run_in_repo() {
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

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command is unavailable: $1" >&2
        exit 1
    fi
}

check_free_space() {
    local available_kib
    local required_kib
    available_kib=$(df -Pk "${STORAGE_BS}" | awk 'NR == 2 {print $4}')
    required_kib=$((MIN_FREE_GIB * 1024 * 1024))
    if [[ -z "${available_kib}" || ! "${available_kib}" =~ ^[0-9]+$ ]]; then
        echo "Could not determine free space for ${STORAGE_BS}" >&2
        exit 1
    fi
    if (( available_kib < required_kib )); then
        echo "Insufficient free space under ${STORAGE_BS}." >&2
        echo "Need at least ${MIN_FREE_GIB} GiB before installing and downloading." >&2
        echo "Run df -h and lfs quota -h; do not retry into a quota error." >&2
        exit 1
    fi
}

verify_source_checkout() {
    local current_commit
    current_commit=$(git -C "${ABOT_REPO}" rev-parse HEAD)
    if [[ "${current_commit}" != "${ABOT_COMMIT}" ]]; then
        echo "ABot checkout is not pinned to ${ABOT_COMMIT}: ${current_commit}" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${ABOT_REPO}" status --porcelain)" ]]; then
        echo "ABot checkout is dirty; refusing to change or evaluate it." >&2
        exit 1
    fi
}

verify_checkpoint() {
    local required_files=(
        "base_checkpoint/vae/diffusion_pytorch_model.safetensors"
        "base_checkpoint/text_encoder/model.safetensors.index.json"
        "base_checkpoint/tokenizer/tokenizer.json"
        "checkpoint_step/transformer/config.json"
        "checkpoint_step/transformer/diffusion_pytorch_model.safetensors"
    )
    local relative_path
    for relative_path in "${required_files[@]}"; do
        if [[ ! -f "${ABOT_CHECKPOINT_ROOT}/${relative_path}" ]]; then
            echo "Missing checkpoint artifact: ${ABOT_CHECKPOINT_ROOT}/${relative_path}" >&2
            exit 1
        fi
    done
}

echo "ABot-M0.5 cluster setup"
echo "  source repository : ${ABOT_REPO}"
echo "  source commit     : ${ABOT_COMMIT}"
echo "  conda environment : ${ABOT_ENV}"
echo "  checkpoint root   : ${ABOT_CHECKPOINT_ROOT}"
echo "  Hugging Face repo : ${ABOT_HF_REPOSITORY}"
echo "  Hugging Face home : ${HF_HOME}"
echo "  attention default : torch"
echo

if [[ "${DRY_RUN}" == "0" ]]; then
    require_command git
    require_command "${CONDA_EXE}"
    require_command df
    check_free_space
fi

run mkdir -p "$(dirname "${ABOT_REPO}")" "${HF_HOME}" "$(dirname "${ABOT_CHECKPOINT_ROOT}")"

if [[ "${DRY_RUN}" == "1" ]]; then
    print_command git clone "${ABOT_REPOSITORY_URL}" "${ABOT_REPO}"
    print_command git -C "${ABOT_REPO}" fetch origin "${ABOT_COMMIT}"
    print_command git -C "${ABOT_REPO}" checkout --detach "${ABOT_COMMIT}"
elif [[ ! -e "${ABOT_REPO}" ]]; then
    git clone "${ABOT_REPOSITORY_URL}" "${ABOT_REPO}"
    git -C "${ABOT_REPO}" fetch origin "${ABOT_COMMIT}"
    git -C "${ABOT_REPO}" checkout --detach "${ABOT_COMMIT}"
elif [[ -d "${ABOT_REPO}/.git" ]]; then
    if [[ -n "$(git -C "${ABOT_REPO}" status --porcelain)" ]]; then
        echo "Existing ABot checkout is dirty; refusing to overwrite it: ${ABOT_REPO}" >&2
        exit 1
    fi
    git -C "${ABOT_REPO}" fetch origin "${ABOT_COMMIT}"
    git -C "${ABOT_REPO}" checkout --detach "${ABOT_COMMIT}"
else
    echo "ABOT_REPO exists but is not a git checkout: ${ABOT_REPO}" >&2
    exit 1
fi

if [[ "${SKIP_ENV}" == "0" ]]; then
    if [[ "${DRY_RUN}" == "1" || ! -x "${ABOT_ENV}/bin/python" ]]; then
        run "${CONDA_EXE}" create -y -p "${ABOT_ENV}" python=3.10
    fi
    run "${ABOT_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
    run "${ABOT_ENV}/bin/python" -m pip install \
        torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
    run "${ABOT_ENV}/bin/python" -m pip install --editable "${ABOT_REPO}"
    # Transformers 4.55.2 and Tokenizers 0.21.4 both require Hub < 1.0.
    # Pin the known-compatible release instead of allowing pip to install a
    # newer 1.x version that makes Transformers fail during import.
    run "${ABOT_ENV}/bin/python" -m pip install huggingface_hub==0.36.2
    if [[ "${WITH_FLASH_ATTN}" == "1" ]]; then
        run env MAX_JOBS="${MAX_JOBS:-8}" "${ABOT_ENV}/bin/python" -m pip install \
            flash-attn==2.8.3 --no-build-isolation
    fi
fi

if [[ "${DRY_RUN}" == "0" && ! -x "${ABOT_ENV}/bin/python" ]]; then
    echo "ABot Python is missing: ${ABOT_ENV}/bin/python" >&2
    exit 1
fi

if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env HF_HOME="${HF_HOME}" "${ABOT_ENV}/bin/hf" auth whoami
        print_command env HF_HOME="${HF_HOME}" "${ABOT_ENV}/bin/hf" download \
            "${ABOT_HF_REPOSITORY}" \
            --repo-type model \
            --include "base_checkpoint/**" \
            --include "checkpoint_step/**" \
            --local-dir "${ABOT_CHECKPOINT_ROOT}"
    else
        if ! env HF_HOME="${HF_HOME}" "${ABOT_ENV}/bin/hf" auth whoami >/dev/null 2>&1; then
            cat >&2 <<EOF
Hugging Face authentication is required.
1. Accept the access conditions at:
   https://huggingface.co/${ABOT_HF_REPOSITORY}
2. Run:
   HF_HOME=${HF_HOME} ${ABOT_ENV}/bin/hf auth login
3. Re-run this setup script.
EOF
            exit 1
        fi
        env HF_HOME="${HF_HOME}" "${ABOT_ENV}/bin/hf" download \
            "${ABOT_HF_REPOSITORY}" \
            --repo-type model \
            --include "base_checkpoint/**" \
            --include "checkpoint_step/**" \
            --local-dir "${ABOT_CHECKPOINT_ROOT}"
    fi
fi

if [[ "${DRY_RUN}" == "0" ]]; then
    verify_source_checkout
    if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
        verify_checkpoint
    fi
    run_in_repo "${ABOT_ENV}/bin/python" -c \
        "import torch, diffusers, transformers, wam; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__)"
fi

echo
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete; no changes were made."
else
    echo "ABot-M0.5 setup is ready."
    echo "Next: run robocasa/scripts/abot_m05/evaluate_cluster.sh preflight"
fi

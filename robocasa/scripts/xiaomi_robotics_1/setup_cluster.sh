#!/usr/bin/env bash
set -euo pipefail

# Install the pinned Xiaomi-Robotics-1 evaluation stack on the Shinoda cluster.
# Source code stays on /gs/fs. Environments, caches, and checkpoint artifacts
# stay on /gs/bs. The public checkpoint does not require authentication.

XR1_REPOSITORY_URL=${XR1_REPOSITORY_URL:-https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git}
XR1_COMMIT=${XR1_COMMIT:-4da1db0a4deefa6de7ebb4ef0b8754017290f5f7}
XR1_HF_REPOSITORY=${XR1_HF_REPOSITORY:-XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365}
XR1_HF_REVISION=${XR1_HF_REVISION:-0d1aa76d0d82debc9b611e4d1e231096434d5be4}

PROJECT_FS=${PROJECT_FS:-/gs/fs/tga-shinoda/felid}
STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
ROBOCASA_REPO=${ROBOCASA_REPO:-${PROJECT_FS}/robocasa}
XR1_REPO=${XR1_REPO:-${PROJECT_FS}/robocasa_benchmark_repos/Xiaomi-Robotics-1}
XR1_SERVER_ENV=${XR1_SERVER_ENV:-${STORAGE_BS}/envs/xiaomi_robotics_1_server}
XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-${STORAGE_BS}/envs/xiaomi_robotics_1_robocasa365}
ROBOCASA_BASE_ENV=${ROBOCASA_BASE_ENV:-${STORAGE_BS}/envs/robocasa_openpi}
XR1_CHECKPOINT=${XR1_CHECKPOINT:-${STORAGE_BS}/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365}
HF_HOME=${HF_HOME:-${STORAGE_BS}/hf_home}
CONDA_EXE=${CONDA_EXE:-conda}
MIN_FREE_GIB=${MIN_FREE_GIB:-45}

TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}

DRY_RUN=0
SKIP_SOURCE=0
SKIP_SERVER_ENV=0
SKIP_CLIENT_ENV=0
SKIP_DOWNLOAD=0

usage() {
    cat <<'EOF'
Usage: setup_cluster.sh [options]

Options:
  --dry-run            Print planned operations without changing anything.
  --skip-source        Reuse an existing pinned Xiaomi source checkout.
  --skip-server-env    Reuse the Xiaomi model-server environment.
  --skip-client-env    Reuse the dedicated RoboCasa simulator environment.
  --skip-download      Reuse the downloaded RoboCasa365 checkpoint.
  -h, --help           Show this help.

Environment overrides:
  PROJECT_FS, STORAGE_BS, ROBOCASA_REPO, ROBOCASA_BASE_ENV, XR1_REPO,
  XR1_SERVER_ENV, XR1_CLIENT_ENV, XR1_CHECKPOINT, HF_HOME, CONDA_EXE,
  XR1_REPOSITORY_URL, XR1_COMMIT, XR1_HF_REPOSITORY, XR1_HF_REVISION.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --skip-source) SKIP_SOURCE=1 ;;
        --skip-server-env) SKIP_SERVER_ENV=1 ;;
        --skip-client-env) SKIP_CLIENT_ENV=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        -h|--help) usage; exit 0 ;;
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
        echo "Could not determine free space under ${STORAGE_BS}." >&2
        exit 1
    fi
    if (( available_kib < required_kib )); then
        echo "Insufficient free space under ${STORAGE_BS}." >&2
        echo "Need at least ${MIN_FREE_GIB} GiB before setup." >&2
        echo "Inspect df -h, df -ih, and lfs quota -h; do not retry into a quota error." >&2
        exit 1
    fi
}

verify_source() {
    local current_commit
    current_commit=$(git -C "${XR1_REPO}" rev-parse HEAD)
    if [[ "${current_commit}" != "${XR1_COMMIT}" ]]; then
        echo "Xiaomi checkout is not pinned to ${XR1_COMMIT}: ${current_commit}" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${XR1_REPO}" status --porcelain)" ]]; then
        echo "Xiaomi checkout is dirty; refusing to use it: ${XR1_REPO}" >&2
        exit 1
    fi
}

verify_checkpoint() {
    local required_files=(
        config.json
        configuration_mibot.py
        modeling_mibot.py
        processing_mibot.py
        processor_config.json
        model.safetensors.index.json
        model-00001-of-00003.safetensors
        model-00002-of-00003.safetensors
        model-00003-of-00003.safetensors
    )
    local filename
    for filename in "${required_files[@]}"; do
        if [[ ! -s "${XR1_CHECKPOINT}/${filename}" ]]; then
            echo "Missing or empty checkpoint artifact: ${XR1_CHECKPOINT}/${filename}" >&2
            exit 1
        fi
    done

    "${XR1_CLIENT_ENV}/bin/python" - "${XR1_CHECKPOINT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
assert config["architectures"] == ["MiBoTForActionGeneration"]
assert config["transformers_version"] == "4.57.1"
assert config["state_length"] == 4
assert config["state_dim"] == 60
assert config["action_dim"] == 60
assert index["metadata"]["total_parameters"] == 5_442_105_856
assert index["metadata"]["total_size"] == 10_106_299_392
shards = sorted(root.glob("model-*-of-*.safetensors"))
assert len(shards) == 3, shards
assert sum(path.stat().st_size for path in shards) >= index["metadata"]["total_size"]
print("checkpoint manifest verified")
PY
}

echo "Xiaomi-Robotics-1 RoboCasa365 cluster setup"
echo "  source repository : ${XR1_REPO}"
echo "  source commit     : ${XR1_COMMIT}"
echo "  server environment: ${XR1_SERVER_ENV}"
echo "  client environment: ${XR1_CLIENT_ENV}"
echo "  client base env   : ${ROBOCASA_BASE_ENV}"
echo "  checkpoint        : ${XR1_CHECKPOINT}"
echo "  checkpoint repo   : ${XR1_HF_REPOSITORY}"
echo "  checkpoint revision: ${XR1_HF_REVISION}"
echo "  Hugging Face home : ${HF_HOME}"
echo

if [[ "${DRY_RUN}" == "0" ]]; then
    require_command git
    require_command "${CONDA_EXE}"
    require_command df
    check_free_space
    if [[ ! -d "${ROBOCASA_REPO}/.git" ]]; then
        echo "RoboCasa checkout is missing: ${ROBOCASA_REPO}" >&2
        exit 1
    fi
fi

run mkdir -p \
    "$(dirname "${XR1_REPO}")" \
    "$(dirname "${XR1_CHECKPOINT}")" \
    "${HF_HOME}" \
    "${STORAGE_BS}/robocasa_logs/eval" \
    "${STORAGE_BS}/robocasa_rollouts/xiaomi_robotics_1"

if [[ "${SKIP_SOURCE}" == "0" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command git clone "${XR1_REPOSITORY_URL}" "${XR1_REPO}"
        print_command git -C "${XR1_REPO}" fetch origin "${XR1_COMMIT}"
        print_command git -C "${XR1_REPO}" checkout --detach "${XR1_COMMIT}"
    elif [[ ! -e "${XR1_REPO}" ]]; then
        git clone "${XR1_REPOSITORY_URL}" "${XR1_REPO}"
        git -C "${XR1_REPO}" fetch origin "${XR1_COMMIT}"
        git -C "${XR1_REPO}" checkout --detach "${XR1_COMMIT}"
    elif [[ -d "${XR1_REPO}/.git" ]]; then
        if [[ -n "$(git -C "${XR1_REPO}" status --porcelain)" ]]; then
            echo "Existing Xiaomi checkout is dirty; refusing to overwrite it." >&2
            exit 1
        fi
        git -C "${XR1_REPO}" fetch origin "${XR1_COMMIT}"
        git -C "${XR1_REPO}" checkout --detach "${XR1_COMMIT}"
    else
        echo "XR1_REPO exists but is not a git checkout: ${XR1_REPO}" >&2
        exit 1
    fi
fi

if [[ "${SKIP_SERVER_ENV}" == "0" ]]; then
    if [[ "${DRY_RUN}" == "1" || ! -x "${XR1_SERVER_ENV}/bin/python" ]]; then
        run "${CONDA_EXE}" create -y -p "${XR1_SERVER_ENV}" python=3.12
    fi
    run "${XR1_SERVER_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
    run "${XR1_SERVER_ENV}/bin/python" -m pip install \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url "${TORCH_INDEX_URL}"
    run "${XR1_SERVER_ENV}/bin/python" -m pip install \
        transformers==4.57.1 huggingface_hub==0.36.2 tqdm
    run "${XR1_SERVER_ENV}/bin/python" -m pip install "${FLASH_ATTN_WHEEL}"
fi

if [[ "${SKIP_CLIENT_ENV}" == "0" ]]; then
    if [[ "${DRY_RUN}" == "1" || ! -x "${XR1_CLIENT_ENV}/bin/python" ]]; then
        run "${CONDA_EXE}" create -y -p "${XR1_CLIENT_ENV}" --clone "${ROBOCASA_BASE_ENV}"
    fi
    run "${XR1_CLIENT_ENV}/bin/python" -m pip install \
        transformers==4.57.1 huggingface_hub==0.36.2 \
        'imageio[ffmpeg]' tqdm scipy
    run "${XR1_CLIENT_ENV}/bin/python" -m pip install --no-deps --editable "${ROBOCASA_REPO}"
fi

if [[ "${DRY_RUN}" == "0" ]]; then
    if [[ ! -x "${XR1_SERVER_ENV}/bin/python" ]]; then
        echo "Server Python is missing: ${XR1_SERVER_ENV}/bin/python" >&2
        exit 1
    fi
    if [[ ! -x "${XR1_CLIENT_ENV}/bin/python" ]]; then
        echo "Client Python is missing: ${XR1_CLIENT_ENV}/bin/python" >&2
        exit 1
    fi
fi

if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
    run env HF_HOME="${HF_HOME}" "${XR1_SERVER_ENV}/bin/hf" download \
        "${XR1_HF_REPOSITORY}" \
        --revision "${XR1_HF_REVISION}" \
        --local-dir "${XR1_CHECKPOINT}"
fi

if [[ "${DRY_RUN}" == "0" ]]; then
    verify_source
    verify_checkpoint
    "${XR1_SERVER_ENV}/bin/python" -c \
        "import flash_attn, torch, transformers; assert torch.__version__.startswith('2.8.0'); assert transformers.__version__ == '4.57.1'; print('server torch', torch.__version__, 'transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)"
    "${XR1_CLIENT_ENV}/bin/python" -c \
        "import robocasa, transformers; assert robocasa.__version__ == '1.0.1'; assert transformers.__version__ == '4.57.1'; print('client robocasa', robocasa.__version__, 'transformers', transformers.__version__)"
fi

echo
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete; no changes were made."
else
    echo "Xiaomi-Robotics-1 evaluation setup is ready."
    echo "Next: bash robocasa/scripts/xiaomi_robotics_1/evaluate_cluster.sh preflight"
fi

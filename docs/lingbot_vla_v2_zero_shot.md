# LingBot-VLA-v2 zero-shot RoboCasa evaluation

This runbook evaluates the released `robbyant/lingbot-vla-v2-6b` checkpoint on
the 50 RoboCasa target tasks **without post-training, fine-tuning, or fitting
normalization statistics on RoboCasa demonstrations**.

The primary protocol is deliberately strict:

- released pretrained weights are used unchanged;
- a fixed, hand-written PandaOmron feature map selects LingBot's canonical
  end-effector, gripper, base, and waist slots;
- identity statistics map the controller's nominal `[-1, 1]` range to itself;
- no RoboCasa observation or action samples are used before evaluation;
- task success, per-subtask progress, errors, actions, and smoke-test videos are
  recorded.

Dataset-derived normalization is not part of this protocol. If it is tested
later, report it separately as **zero-shot with target-domain calibration**.

## 1. Architecture

LingBot and RoboCasa require different Python stacks, so run them as separate
processes on the same allocated node:

```text
GPU 0: LingBot-VLA-v2 Python 3.12 / PyTorch 2.8 server, port 9330
                         ^ MessagePack WebSocket
                         |
GPU 1: RoboCasa simulator in robocasa_openpi environment
```

The simulator can share GPU 0 for a one-GPU smoke test. Two GPUs are preferred
for the full evaluation.

## 2. Canonical cluster exports

Start in a fresh allocated terminal and run:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export USER_ID=ut06746
export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid

export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export ROBOCASA_LOG_ROOT="$STORAGE_BS/robocasa_logs"
export ROBOCASA_ROLLOUT_ROOT="$STORAGE_BS/robocasa_rollouts"
export ROBOCASA_CKPT_ROOT="$STORAGE_BS/robocasa_checkpoints"

export LINGBOT_REPO="$PROJECT_FS/robocasa_benchmark_repos/lingbot-vla-v2"
export LINGBOT_ENV="$STORAGE_BS/envs/lingbot_vla_v2"
export LINGBOT_CKPT_ROOT="$ROBOCASA_CKPT_ROOT/lingbot_vla_v2"
export LINGBOT_DOWNLOAD="$LINGBOT_CKPT_ROOT/downloads/lingbot-vla-v2-6b"
export QWEN3VL_PATH="$LINGBOT_CKPT_ROOT/downloads/Qwen3-VL-4B-Instruct"
export LINGBOT_RUNTIME="$LINGBOT_CKPT_ROOT/runtime_pretrained"
export LINGBOT_NORM="$LINGBOT_REPO/assets/norm_stats/robocasa_zero_shot_identity.json"
export LINGBOT_PORT=9330

export HF_HOME="$STORAGE_BS/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$STORAGE_BS/xdg_cache"
export PIP_CACHE_DIR="$STORAGE_BS/pip_cache"
export CONDA_PKGS_DIRS="$STORAGE_BS/conda_pkgs"
export TRITON_CACHE_DIR="/tmp/$USER_ID/lingbot_triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER_ID/lingbot_torchinductor_cache"
export CUDA_CACHE_PATH="/tmp/$USER_ID/lingbot_cuda_cache"
export TMPDIR="/tmp/$USER_ID/lingbot_tmp"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

mkdir -p \
  "$PROJECT_FS/robocasa_benchmark_repos" \
  "$LINGBOT_CKPT_ROOT/downloads" \
  "$ROBOCASA_LOG_ROOT/eval" \
  "$ROBOCASA_ROLLOUT_ROOT" \
  "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
```

## 3. Install LingBot in a durable environment

Clone source under `/gs/fs`; keep the environment, caches, and weights under
`/gs/bs`:

```bash
test -d "$LINGBOT_REPO/.git" || \
  git clone https://github.com/Robbyant/lingbot-vla-v2.git "$LINGBOT_REPO"

module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

test -x "$LINGBOT_ENV/bin/python" || \
  conda create --prefix "$LINGBOT_ENV" python=3.12 pip -y
conda activate "$LINGBOT_ENV"

cd "$LINGBOT_REPO"
python -m pip install -U pip setuptools wheel
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  torchdata==0.11.0 torchcodec==0.6.0
python -m pip install -r requirements.txt
python -m pip install numpydantic==1.9.0 --no-deps
MAX_JOBS=8 python -m pip install --no-build-isolation flash-attn==2.8.3
python -m pip install --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
python -m pip install -e . --no-deps
python -m pip install -r requirements-depth.txt
python -m pip install -r requirements.txt
python -m pip install numpydantic==1.9.0 --no-deps
python -m pip install -e \
  lingbotvla/models/vla/vision_models/lingbot-depth --no-deps
python -m pip install -e lingbotvla/models/vla/vision_models/MoGe
python -m pip install huggingface_hub==0.34.0

python - <<'PY'
import site
from pathlib import Path

repo = Path.cwd()
site_packages = Path(site.getsitepackages()[0])
(site_packages / "stablevla_local_depth.pth").write_text(
    str(repo / "lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d") + "\n"
)
PY

python - <<'PY'
import torch
import flash_attn
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
assert torch.__version__.split("+", 1)[0] == "2.8.0"
assert torch.cuda.is_available()
PY
```

If a matching `flash-attn` wheel is available, install it with `--no-deps`
instead of compiling from source.

## 4. Download and verify the released models

Use the current `hf` command, not the deprecated `huggingface-cli`:

```bash
conda activate "$LINGBOT_ENV"
hf version

hf download robbyant/lingbot-vla-v2-6b \
  --local-dir "$LINGBOT_DOWNLOAD" \
  --cache-dir "$HF_HUB_CACHE"

hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$QWEN3VL_PATH" \
  --cache-dir "$HF_HUB_CACHE"

hf cache verify robbyant/lingbot-vla-v2-6b \
  --local-dir "$LINGBOT_DOWNLOAD"
hf cache verify Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$QWEN3VL_PATH"
```

The released Hugging Face snapshot contains the pretrained weights but does not
contain `lingbotvla_cli.yaml`. The upstream loader nevertheless requires that
file three levels above the weight directory. Create a runtime view using the
upstream native-depth `real_robot.yaml` as the architecture description. Only
runtime paths and the dataset label are changed; this does not copy or alter
model weights and does not use RoboCasa training data:

```bash
export LINGBOT_WEIGHT_SOURCE="$(
  find "$LINGBOT_DOWNLOAD" -type f -name '*.safetensors' -printf '%h\n' \
    | sort -u | head -n 1
)"
export LINGBOT_CONFIG_SOURCE="$LINGBOT_REPO/configs/vla/real_robot/real_robot.yaml"

test -n "$LINGBOT_WEIGHT_SOURCE" || { echo "No safetensors found"; exit 1; }
test -f "$LINGBOT_CONFIG_SOURCE" || { echo "No upstream real_robot.yaml found"; exit 1; }

mkdir -p "$LINGBOT_RUNTIME/checkpoints/global_step_0"
test -e "$LINGBOT_RUNTIME/checkpoints/global_step_0/hf_ckpt" || \
  ln -s "$LINGBOT_WEIGHT_SOURCE" \
    "$LINGBOT_RUNTIME/checkpoints/global_step_0/hf_ckpt"
export LINGBOT_MODEL_PATH="$LINGBOT_RUNTIME/checkpoints/global_step_0/hf_ckpt"

python - \
  "$LINGBOT_CONFIG_SOURCE" \
  "$LINGBOT_RUNTIME/lingbotvla_cli.yaml" \
  "$LINGBOT_MODEL_PATH" \
  "$QWEN3VL_PATH" \
  "$LINGBOT_NORM" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst, model_path, qwen_path, norm_path = map(Path, sys.argv[1:])
with src.open() as stream:
    config = yaml.safe_load(stream)

config["model"]["model_path"] = str(model_path)
config["model"]["tokenizer_path"] = str(qwen_path)
config["data"]["data_name"] = "robocasa"
config["data"]["norm_stats_file"] = str(norm_path)

# The deployment-time FeatureTransform calls ast.literal_eval() on these
# values. Training CLI snapshots serialize them as strings even though the
# source configuration expresses them as YAML mappings.
for key in ("joints", "norm_type"):
    config["data"][key] = [
        repr(item) if isinstance(item, dict) else item
        for item in config["data"][key]
    ]

with dst.open("w") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

test -n "$(find -L "$LINGBOT_MODEL_PATH" -maxdepth 1 -name '*.safetensors' -print -quit)"
test -f "$LINGBOT_RUNTIME/lingbotvla_cli.yaml"
```

If the model repository contains more than one directory of safetensors, stop
and inspect the candidates instead of accepting `head -n 1` silently.

## 5. Install the strict zero-shot interface assets

First pull the local integration commit onto the cluster. Then copy the static
feature map and identity statistics into the LingBot source tree:

```bash
cd "$ROBOCASA_REPO"
git fetch origin
git switch codex/lingbot-vla-zero-shot
git pull --ff-only origin codex/lingbot-vla-zero-shot

mkdir -p \
  "$LINGBOT_REPO/configs/robot_configs" \
  "$LINGBOT_REPO/assets/norm_stats"

cp robocasa/models/assets/lingbot_vla_v2/robocasa.yaml \
  "$LINGBOT_REPO/configs/robot_configs/robocasa.yaml"
cp robocasa/models/assets/lingbot_vla_v2/robocasa_zero_shot_identity.json \
  "$LINGBOT_REPO/assets/norm_stats/robocasa_zero_shot_identity.json"

python -m json.tool \
  "$LINGBOT_REPO/assets/norm_stats/robocasa_zero_shot_identity.json" >/dev/null
```

No RoboCasa dataset is read in this step.

## 6. Start and verify the LingBot server

Use the LingBot environment for the server. The launcher passes the identity
statistics explicitly, avoiding the checkpoint's post-training normalization
path.

```bash
conda activate "$LINGBOT_ENV"
export QWEN3VL_PATH
export XFORMERS_DISABLED=1
export TRITON_CACHE_DIR="/tmp/$USER_ID/lingbot_triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER_ID/lingbot_torchinductor_cache"
export CUDA_CACHE_PATH="/tmp/$USER_ID/lingbot_cuda_cache"
export TMPDIR="/tmp/$USER_ID/lingbot_tmp"
mkdir -p \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TMPDIR"
export SERVER_TAG=lingbot_vla_v2_pretrained_zero_shot_$(date +%Y%m%d_%H%M%S)
export SERVER_LOG="$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_${LINGBOT_PORT}.log"

cd "$LINGBOT_REPO"
CUDA_VISIBLE_DEVICES=0 \
nohup python -u "$ROBOCASA_REPO/robocasa/recovery/serve_lingbot_vla.py" \
  --lingbot-repo "$LINGBOT_REPO" \
  --model-path "$LINGBOT_MODEL_PATH" \
  --robot-norm-path "$LINGBOT_NORM" \
  --port "$LINGBOT_PORT" \
  --use-length 8 \
  --use-compile false \
  > "$SERVER_LOG" 2>&1 &

echo "server_pid=$!"
echo "server_log=$SERVER_LOG"

until curl -fsS "http://127.0.0.1:${LINGBOT_PORT}/healthz"; do
  date
  tail -40 "$SERVER_LOG" 2>/dev/null || true
  sleep 10
done

ss -ltnp | grep ":${LINGBOT_PORT}"
nvidia-smi
tail -80 "$SERVER_LOG"
```

The health endpoint only proves that the socket is open. Do not start the full
benchmark until the log confirms that weights loaded and the smoke client
returns finite actions.

## 7. One-task smoke evaluation

Switch to the normal RoboCasa environment. This is a standard closed-loop
rollout; the existing failure-dataset collector is used with
`--include-successes`, so every attempted rollout is retained as evaluation
evidence.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi

python - <<'PY'
import msgpack
import websockets
print("msgpack", msgpack.__version__)
print("websockets", websockets.__version__)
PY

cd "$ROBOCASA_REPO"
export RUN_TAG=lingbot_vla_v2_pretrained_zero_shot_smoke_$(date +%Y%m%d_%H%M%S)
export EVAL_DIR="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
export EVAL_LOG="$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}.log"

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
python -u robocasa/recovery/create_recovery_failure_dataset.py \
  --output-dir "$EVAL_DIR" \
  --output "$EVAL_DIR/results.json" \
  --dataset-type policy_evaluation \
  --policy-module robocasa.recovery.lingbot_vla_websocket_policy:make_policy \
  --policy-name lingbot_vla_v2_6b_pretrained_zero_shot \
  --policy-arg host=127.0.0.1 \
  --policy-arg port="$LINGBOT_PORT" \
  --policy-arg robo_name=robocasa \
  --policy-arg replan_steps=8 \
  --envs OpenDrawer \
  --split target \
  --num-rollouts 1 \
  --seed 0 \
  --include-successes \
  --include-trace \
  --record-actions \
  --record-videos \
  --video-render-source obs \
  2>&1 | tee "$EVAL_LOG"
```

For a one-GPU allocation, change both GPU indices above to `0` and monitor
memory closely.

Validate the smoke artifact before scaling up:

```bash
python - "$EVAL_DIR/results.json" <<'PY'
import json
import pathlib
import sys
import numpy as np

p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
print("partial:", d.get("partial"))
print("summary:", d.get("summary"))
print("errors:", len(d.get("errors", [])))
assert d.get("partial") is False
assert len(d.get("errors", [])) == 0
assert d["summary"]["num_attempted_rollouts"] == 1
assert len(d.get("samples", [])) == 1

sample = d["samples"][0]
action_path = pathlib.Path(sample["action_trajectory_path"])
video_path = pathlib.Path(sample["video_path"])
assert action_path.is_file() and action_path.stat().st_size > 0
assert video_path.is_file() and video_path.stat().st_size > 0
with np.load(action_path) as actions:
    for key in actions.files:
        value = actions[key]
        if not np.issubdtype(value.dtype, np.number):
            print(f"skipping non-numeric metadata: {key} ({value.dtype})")
            continue
        assert np.all(np.isfinite(value)), key
print("sample success:", sample["success"])
print("actions:", action_path)
print("video:", video_path)
PY
```

Success is not required for the plumbing smoke test. Required signals are a
complete manifest, zero errors, finite actions, a non-empty video, and available
subtask diagnostics.

## 8. Full 50-task zero-shot evaluation

Run the three official target groups separately so their success rates cannot
be accidentally mixed. Five episodes per task gives 90 atomic-seen, 80
composite-seen, and 80 composite-unseen rollouts.

```bash
cd "$ROBOCASA_REPO"
export RUN_TAG=lingbot_vla_v2_pretrained_zero_shot_target50_$(date +%Y%m%d_%H%M%S)
export EVAL_ROOT="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
mkdir -p "$EVAL_ROOT"

for TASK_SET in atomic_seen composite_seen composite_unseen; do
  OUT="$EVAL_ROOT/$TASK_SET"
  LOG="$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_${TASK_SET}.log"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
  python -u robocasa/recovery/create_recovery_failure_dataset.py \
    --output-dir "$OUT" \
    --output "$OUT/results.json" \
    --dataset-type policy_evaluation \
    --policy-module robocasa.recovery.lingbot_vla_websocket_policy:make_policy \
    --policy-name lingbot_vla_v2_6b_pretrained_zero_shot \
    --policy-arg host=127.0.0.1 \
    --policy-arg port="$LINGBOT_PORT" \
    --policy-arg robo_name=robocasa \
    --policy-arg replan_steps=8 \
    --task-set "$TASK_SET" \
    --split target \
    --num-rollouts 5 \
    --seed 0 \
    --include-successes \
    --include-trace \
    --record-actions \
    --no-record-videos \
    > "$LOG" 2>&1 || exit 1
done

echo "EVAL_ROOT=$EVAL_ROOT"
```

This loop is intentionally sequential because the LingBot server caches action
chunks and is stateful. Do not point concurrent simulator clients at one server.

## 9. Report results and integrity checks

```bash
python - "$EVAL_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
total_attempted = total_success = total_errors = 0
for group in ("atomic_seen", "composite_seen", "composite_unseen"):
    path = root / group / "results.json"
    data = json.loads(path.read_text())
    summary = data["summary"]
    attempted = summary["num_attempted_rollouts"]
    successes = summary["num_rollout_successes"]
    errors = summary["num_errors"]
    rate = successes / attempted if attempted else 0.0
    print(
        f"{group:20s} {successes:3d}/{attempted:3d} "
        f"success={rate:.1%} errors={errors} partial={data.get('partial')}"
    )
    assert data.get("partial") is False
    assert summary["num_recorded_samples"] == attempted
    assert summary["num_rollout_successes"] + summary["num_rollout_failures"] == attempted
    total_attempted += attempted
    total_success += successes
    total_errors += errors

print(
    f"episode-weighted overall {total_success}/{total_attempted} "
    f"success={total_success / total_attempted:.1%} errors={total_errors}"
)
assert total_attempted == 250
assert total_errors == 0
PY
```

Also verify duplicate sample IDs, action artifacts, task coverage, and per-task
balance before declaring the run healthy:

```bash
python - "$EVAL_ROOT" <<'PY'
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
samples = []
for path in root.glob("*/results.json"):
    samples.extend(json.loads(path.read_text()).get("samples", []))

ids = [sample["sample_id"] for sample in samples]
counts = collections.Counter(sample["task_name"] for sample in samples)
missing_actions = [
    sample["sample_id"]
    for sample in samples
    if not pathlib.Path(sample["action_trajectory_path"]).is_file()
]
print("samples:", len(samples))
print("tasks:", len(counts))
print("per-task counts:", collections.Counter(counts.values()))
print("duplicate IDs:", len(ids) - len(set(ids)))
print("missing actions:", len(missing_actions))
assert len(samples) == 250
assert len(counts) == 50
assert set(counts.values()) == {5}
assert len(ids) == len(set(ids))
assert not missing_actions
PY
```

Record the RoboCasa commit, LingBot commit, Hugging Face verification result,
model path, Qwen path, server log, run root, GPU mapping, and the exact success
table with the final report.

## Interpretation

This experiment tests whether the pretrained generalist can control RoboCasa
through a new embodiment adapter without learning from RoboCasa. A low score is
still a valid result. It may reflect one or more of:

- generalization limits of the released checkpoint;
- the gap between LingBot's canonical action semantics and RoboCasa's delta EEF
  controller;
- identity normalization versus embodiment-specific training statistics;
- camera/domain differences;
- control-frequency or action-chunk mismatch.

Do not describe this as a native supported benchmark: upstream publishes
RoboTwin deployment, not a RoboCasa recipe. Keep strict zero-shot results
separate from any later normalization calibration or post-training experiment.

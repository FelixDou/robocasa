# Policy Learning Algorithms

We provide official support for benchmarking the following policy learning algorithms: [Diffusion Policy](https://github.com/robocasa-benchmark/diffusion_policy), [Openpi](https://github.com/robocasa-benchmark/openpi), and [GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T).

RLDX-1 is evaluated through the official [RLDX-1](https://github.com/RLWRLD/RLDX-1) codebase. For recovery/subtask experiments in this RoboCasa checkout, run the RLDX model as a ZeroMQ policy server and pass `robocasa.recovery.rldx_zmq_policy:make_policy` to the recovery evaluator.

-------
## Diffusion Policy
We fork the official Diffusion Policy code base, hosted at [https://github.com/robocasa-benchmark/diffusion_policy](https://github.com/robocasa-benchmark/diffusion_policy).
### Recommended system specs
For training we recommend a GPU with at least 24 GB of memory, but 48 GB+ is prefered.
For inference we recommend a GPU with at least 8 GB of memory.

### Installation
```
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy
pip install -e .
```

### Key files
- Training: [train.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/train.py)
- Evaluation: [eval_robocasa.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/eval_robocasa.py)

### Experiment workflow
```
# train model
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/<dataset-soup>

# Evaluate model
python eval_robocasa.py \
--checkpoint <checkpoint-path> \
--task_set <task-set> \
--split <split>

# Report evaluation results
python diffusion_policy/scripts/get_eval_stats.py \
--dir <outputs-dir>
```


-------
## Openpi
We fork the official Openpi code base, hosted at [https://github.com/robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi). Our fork support training for **pi0**.

### Recommended system specs
For training we recommend a GPU with at least 100 GB of memory (B100, H200, etc).
For inference we recommend a GPU with at least 8 GB of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/openpi
cd openpi
pip install -e .
pip install -e packages/openpi-client/
```

### Key files
- Training: [scripts/train.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/train.py)
- Evaluation: [scripts/serve_policy.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/serve_policy.py) and [examples/robocasa/main.py](https://github.com/robocasa-benchmark/openpi/blob/main/examples/robocasa/main.py)
- Setting up configs: [src/openpi/training/config.py](https://github.com/robocasa-benchmark/openpi/blob/main/src/openpi/training/config.py)

### Experiment workflow
```
# train model
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
<dataset-soup> \
--exp-name=<exp-name>

# evaluate model
# part a: start inference server
python scripts/serve_policy.py \
--port=8000 policy:checkpoint \
--policy.config=<dataset-soup> \
--policy.dir=<checkpoint-path>

# part b: run evals on server
python examples/robocasa/main.py \
--args.port 8000 \
--args.task_set <task-set> \
--args.split <split> \
--args.log_dir <checkpoint-path>

# report evaluation results
python examples/robocasa/get_eval_stats.py \
--dir <checkpoint-path>
```

-------
## GR00T
We fork the official GR00T code base, hosted at [https://github.com/robocasa-benchmark/Isaac-GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T). Our fork supports training for **GR00T N1.5**.

### Recommended system specs
For training we recommend a GPU with at least 100 GB of memory (B100, H200, etc).
For inference we recommend a GPU with at least 8 GB of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd groot
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### Key files
- Training: [scripts/gr00t_finetune.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py)
- Evaluation: [scripts/run_eval.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/run_eval.py)

### Experiment workflow
```
# train model
python scripts/gr00t_finetune.py \
--output-dir <experiment-path> \
--dataset_soup <dataset-soup> \
--max_steps <num-training-steps>

# evaluate model
python scripts/run_eval.py \
--model_path <checkpoint-path> \
--task_set <task-set> \
--split <split>

# report evaluation results
python gr00t/eval/get_eval_stats.py \
--dir <checkpoint-path>
```

-------
## RLDX-1
RLDX-1 is hosted at [https://github.com/RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1). It is not vendored into this RoboCasa repository; keep the model server in the RLDX environment and run RoboCasa rollouts from the RoboCasa environment.

### Key files
- Model server: [rldx/eval/run_rldx_server.py](https://github.com/RLWRLD/RLDX-1/blob/main/rldx/eval/run_rldx_server.py)
- Official simulator evaluation: [rldx/eval/rollout_policy.py](https://github.com/RLWRLD/RLDX-1/blob/main/rldx/eval/rollout_policy.py)
- Recovery adapter in this repo: `robocasa/recovery/rldx_zmq_policy.py`
- Recovery evaluator in this repo: `robocasa/recovery/evaluate_recovery_benchmark.py`

### Official RLDX RoboCasa evaluation
Start the RLDX server from the RLDX-1 checkout:
```
python rldx/eval/run_rldx_server.py \
  --model-path <rldx-checkpoint-or-hf-model> \
  --embodiment-tag general_embodiment \
  --port 5555 \
  --use-sim-policy-wrapper
```

Then run the official RLDX simulator evaluator:
```
python rldx/eval/rollout_policy.py \
  --env-name robocasa/<task-name> \
  --robocasa-split pretrain \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --n-action-steps 8 \
  --n-episodes <num-rollouts>
```

### RLDX recovery/subtask evaluation
Start the same RLDX server, then run this repository's recovery evaluator:
```
python robocasa/recovery/evaluate_recovery_benchmark.py \
  --output <output-dir>/results.json \
  --policy-module robocasa.recovery.rldx_zmq_policy:make_policy \
  --policy-arg host=127.0.0.1 \
  --policy-arg port=5555 \
  --policy-arg execution_horizon=8 \
  --task-set all_target \
  --split pretrain \
  --num-rollouts 5 \
  --modes env_to_last_good eef_to_last_good continue_from_failure \
  --match-recovery-horizon-to-no-progress \
  --video-dir <output-dir>/videos
```

The RLDX adapter maps RoboCasa Gym observations to the RLDX simulator-wrapper keys (`video.res256_image_side_0`, `video.res256_image_side_1`, `video.res256_image_wrist_0`, and `annotation.human.action.task_description`) and converts returned action chunks back to RoboCasa Gym action dictionaries.

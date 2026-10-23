# Per-seed standalone detection metrics (RSS Table I, mean ± std over seeds)

Purpose: report Table I as Hide-and-Seek does (mean ± std over 3 seeds) for every detector
with training randomness, on the August validation cohorts (calibration 114/114, test 380/380).
Analysis only; no rollout is collected and no policy runs. Code:
`scripts/xr1_lead_time/per_seed_detection.py`.

| Table I row | Per-seed source | Action |
|---|---|---|
| SAFE-MLP | `per_seed_normalized_running_max` in `$XR1_PS_INPUTS` | run A |
| Hide-and-Seek | Amendment 2 scores (`score_cohorts`, `per_seed_raw`) | run A |
| ActProbe | only the seed average was saved (`run_baselines`) | rescore 3 seeds (B1), then B2 |
| Mahalanobis, cosine k-NN | deterministic: no training randomness | none (std 0) |
| PCA-KMeans, RND | section 15.10 trained one model seed | not covered here |
| Time risk, global timer, P | deterministic | none |

Gate: each run first evaluates the seed-averaged score and must reproduce its Table I row
(`reference ... -> OK` and `REFERENCE OK`); otherwise stop and report. Local check on SAFE
(2026-10-24): reproduced TPR 0.566 / FPR 0.039; per seed bACC 0.741 ± 0.010, TWA 0.571 ± 0.005.

## Commands (TSUBAME, CPU only)

`PSEED_SHA` is the full SHA of the pushed commit that contains this file.

### 0. Environment and pinned worktree

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env

export PSEED_SHA=<PSEED_SHA>
export PSEED_REPO=$PROJECT_FS/robocasa_perseed_${PSEED_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/time-aware-fusion-paper
test -e $PSEED_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $PSEED_REPO $PSEED_SHA
test "$(git -C $PSEED_REPO rev-parse HEAD)" = "$PSEED_SHA" || echo "STOP: worktree HEAD is not PSEED_SHA"

export PSEED_TAG=xr1_per_seed_detection_$(date +%Y%m%d_%H%M%S)
export PSEED_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$PSEED_TAG
export PSEED_LOG=$STORAGE_BS/robocasa_logs/eval/$PSEED_TAG
mkdir -p $PSEED_ROOT $PSEED_LOG
echo "AMEND2_ROOT=$AMEND2_ROOT XR1_PS_INPUTS=$XR1_PS_INPUTS XR1_FIRST_RUN=$XR1_FIRST_RUN"
for f in $XR1_PS_INPUTS/calibration_predictions.jsonl $XR1_PS_INPUTS/event_predictions.jsonl \
         $AMEND2_ROOT/scores/august_calibration_predictions.jsonl $AMEND2_ROOT/scores/august_test_predictions.jsonl \
         $AMEND2_ROOT/scores/normalizations.json; do
  test -s $f && echo "OK $f" || echo "STOP: missing $f"
done
sha256sum $XR1_PS_INPUTS/calibration_predictions.jsonl $XR1_PS_INPUTS/event_predictions.jsonl \
  $AMEND2_ROOT/scores/august_*_predictions.jsonl $AMEND2_ROOT/scores/normalizations.json \
  | tee $PSEED_LOG/frozen_inputs.sha256
```

### A. SAFE-MLP and Hide-and-Seek (existing scores; minutes)

```bash
cd $PSEED_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/per_seed_detection.py \
  --base-calibration $XR1_PS_INPUTS/calibration_predictions.jsonl --base-test $XR1_PS_INPUTS/event_predictions.jsonl \
  --safe \
  --scores-calibration $AMEND2_ROOT/scores/august_calibration_predictions.jsonl \
  --scores-test $AMEND2_ROOT/scores/august_test_predictions.jsonl \
  --normalizations $AMEND2_ROOT/scores/normalizations.json \
  --detectors hide_and_seek \
  --expect safe_mlp=0.566,0.039 hide_and_seek=0.584,0.061 \
  --output $PSEED_ROOT/per_seed_safe_hide_and_seek.json 2>&1 | tee $PSEED_LOG/run_a.log
```

### B1. Rescore ActProbe with per-seed outputs (CPU, about 1–2 hours)

The `--reference` checks make the run fail unless the seed-averaged trajectories equal those
of the published-baselines run behind Table I.

```bash
cd $PSEED_REPO
test -s $XR1_FIRST_RUN/calibration_predictions.jsonl -a -s $XR1_FIRST_RUN/evaluation_predictions.jsonl \
  && echo "reference run OK" || echo "STOP: set XR1_FIRST_RUN to the published-baselines run with ActProbe"
SCORE_ARGS=(
  --dataset-dir "${XR1_RAW_DIRS[@]}"
  --runtime-bundle $XR1_RUNTIME_BUNDLE --train-ids $XR1_TRAIN_IDS
  --cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json
  --cohort august_test $XR1_EVAL_IDS
  --detectors actprobe --feature-aggregation last --seeds 0 1 2
  --reference august_calibration $XR1_FIRST_RUN/calibration_predictions.jsonl
  --reference august_test $XR1_FIRST_RUN/evaluation_predictions.jsonl
  --output-dir $PSEED_ROOT/actprobe_scores
)
PYTHONPATH=$PSEED_REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 \
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${SCORE_ARGS[@]}" > $PSEED_LOG/actprobe_score.log 2>&1 &
echo "score PID: $! on $(hostname)"
```

Monitor with `tail -f $PSEED_LOG/actprobe_score.log`. It must end with two
`reference ... -> OK` lines and `REFERENCE OK`; otherwise stop.

### B2. ActProbe per-seed metrics

```bash
cd $PSEED_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/per_seed_detection.py \
  --base-calibration $XR1_PS_INPUTS/calibration_predictions.jsonl --base-test $XR1_PS_INPUTS/event_predictions.jsonl \
  --scores-calibration $PSEED_ROOT/actprobe_scores/august_calibration_predictions.jsonl \
  --scores-test $PSEED_ROOT/actprobe_scores/august_test_predictions.jsonl \
  --normalizations $PSEED_ROOT/actprobe_scores/normalizations.json \
  --detectors actprobe --expect actprobe=0.403,0.063 \
  --output $PSEED_ROOT/per_seed_actprobe.json 2>&1 | tee $PSEED_LOG/run_b2.log
sha256sum -c $PSEED_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
```

Report back: the `reference` lines, the per-detector `mean+-std` lines and `REFERENCE OK` from
`run_a.log` and `run_b2.log`, and `INPUTS UNCHANGED`.

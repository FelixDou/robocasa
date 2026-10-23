# RLDX-1 offline replication of Hide-and-Seek + P (plan)

**Status: exploratory, second policy, no new rollouts.**
- Question: does the preregistered XR-1 result (report §15.21, Amendment 2: Hide-and-Seek
  + P adds +0.071 [+0.022, +0.157] of H_k in warning lead over the per-task timer P)
  hold on a second policy, RLDX-1?
- This plan was committed **before** any Hide-and-Seek or control score was computed on
  RLDX-1 data. "Computed" means any trajectory, threshold, alarm or metric.
- The RLDX-1 cohorts are already open: SAFE was trained and evaluated on them (§8.3) and
  SAFE + P was compared with P on them (§15.15). The result is therefore exploratory,
  whatever it is. It supports, but cannot replace, the preregistered E2.

**Result (recorded after the run):** R1 met (Hide-and-Seek + P +0.083 [+0.033, +0.138]
over P at 4 of 100 false alarms; +0.052 at matched FPR). The prediction below was wrong.
See report Section 15.22. This file is otherwise unchanged since commit `6f617d51`.

## What has already been seen on these data

- SAFE (MLP and LSTM) discrimination and conformal operating points (§8.3).
- SAFE + P against P, the global timer, noise + P and the atomic/composite strata (§15.15):
  SAFE (LSTM) + P gains +0.034 [+0.008, +0.067] over P (lead 0.271), all of it on atomic
  tasks.
- **Not seen:** no Hide-and-Seek, ActProbe or control quantity has ever been computed on
  RLDX-1.

## Data (unchanged from §15.15)

- **Rollouts.** The RLDX-1 SAFE20 natural-rate run: 20 tasks, 1,000 rollouts.
- **Splits**, as in §15.15, from `<SAFE20 root>/calibration/lstm/lstm_seed{0,1,2}/normalized_scores.jsonl`:
  - training: 740 rollouts (402 successes, 338 failures);
  - calibration: 60 held-out successes, three per task;
  - test: 200 held-out rollouts, five successes and five failures per task.
- **H_k:** the longest training failure of task k, in inferences. In §15.15 every training
  and test failure had L = H_k.
- **Raw features.** The official SAFE export (`official_safe_supported20_natural50`) was
  written from a raw `save_rollout` dataset. Its `conversion_report.json` names that source
  (`source_dataset`) and keeps the rollout IDs. Each raw tensor stores `features` with shape
  (inferences, denoising steps, action horizon, MSAT dim) (`docs/safe_rldx1.md`), which is
  what `score_cohorts` reads. Step 1 of the runbook checks this before anything is scored:
  every one of the 1,000 IDs must be in the raw manifest with
  `valid_sequence_length` equal to SAFE's trajectory length.
  - If the raw source is missing or does not match, **stop**. Rebuilding features from the
    export's per-inference pickles would be a new step, written up before scoring.

## Adapter (the only new code)

- `scripts/xr1_lead_time/rldx_hide_and_seek.py prepare` writes what `score_cohorts` needs
  from the §15.15 files, and computes no score:
  - an RLDX "runtime bundle": `task_horizons` = H_k, `development_rollout_ids` = the
    training IDs, and a zero time-risk curve that nothing uses;
  - the training, calibration and test ID lists.
- `score_cohorts --score-train` also writes the training rollouts' (in-sample) trajectories.
  §15.15 chooses the fused path on training failures, so their scores are needed. The flag
  is off by default; nothing else in `score_cohorts` changes.
- `rldx_hide_and_seek.py run` is the analysis below. It reuses the §15.15 row assembly
  (`rldx_replication.py`) and the Amendment 2 helpers (`confirm_hide_and_seek.py`).

## Detectors (frozen, the Amendment 2 settings)

- `hide_and_seek`, and its three controls: `hide_and_seek_first` (S1, every inference gets
  the rollout's own first-inference features), `hide_and_seek_zero` and
  `hide_and_seek_task_shuffle` (S2). The shuffle control is included, as in Amendment 2,
  so both learned-timer controls are replicated.
- Settings: PCA to 128 dimensions, LSTM hidden size 128, peak margin 0.2, onset weight 1.0,
  AdamW (learning rate 1e-3, weight decay 1e-4, batch size 32), 200 epochs, seeds 0, 1
  and 2, CPU. All are the code defaults at this commit; none is tuned on RLDX-1.
- **Input:** feature aggregation `last` (last denoising step, last action position), as
  on XR-1.
  - Difference from XR-1: there, this was the exact vector the frozen SAFE MLP read. The
    RLDX SAFE models read other selections (LSTM: last denoising step, first action
    position; MLP: `concat-2`). `last` is kept so the detector is the same across
    policies.
- **Training set:** the 740 training rollouts, the same set SAFE was trained on.
- **Scores:** per-seed running maximum, normalized per task from training maxima over the
  first H_k inferences (median and max(MAD, SD)), averaged over the three seeds, as in
  `run_baselines` and Amendment 2.

## Analysis (the §15.15 protocol, with the E1 fit)

- **P:** the per-task q = 0.95 quantile of training success durations L/H_k, plus a
  conformal margin set on the 60 calibration successes at alpha = 0.05.
- **Score + P:** the OR rule on the nested path (u, s), fitted with `confirm_lead.fit_fast`
  (the registered E1 grid): u over the 0, 0.05, ..., 1 quantiles of the path-choice
  failures' maximum scores, s in {0, 1, 2, 5, 10, 20, 50}. The FPR margin comes only from
  the 60 held-out calibration successes.
- **Path-choice failures:** the 338 training failures, as in §15.15. For Hide-and-Seek these
  are more in-sample than they were for SAFE, since its loss directly pushes their scores
  up. This can only make the chosen path worse on test; it does not bias the test estimate,
  which uses held-out rollouts only.
- **Endpoint:** mean lead (H_k − T)/H_k over the 100 test failures, with a miss counted as 0
  and an alarm at the final inference counted.
- **Scores fitted:** SAFE (LSTM, the §15.15 primary), Hide-and-Seek and the three controls.
- **Intervals:**
  - for reading: the §15.15 test-only task-then-rollout bootstrap (2,000 replicates,
    seed 0) of each gain over P and of each paired contrast;
  - reported alongside: the E1 decision bootstrap (tasks drawn once for both cohorts,
    rollouts within task and class, P and the fused path refitted per replicate). It is
    not used for reading: with three calibration successes per task it is unstable and
    skewed upward (Amendment 2 dry-run note).
- **Paired contrasts:** Hide-and-Seek + P minus each of first + P, zero + P, shuffle + P and
  SAFE + P.
- **Nulls,** fitted exactly like Hide-and-Seek + P, 20 draws, seed 1: noise + P (running max
  of i.i.d. N(0, 1)) and within-task resampled Hide-and-Seek + P (Amendment 2's H2 null).
- **Descriptive:** atomic and composite strata; P tuned on the test cohort to
  Hide-and-Seek + P's FPR; how often the score alarms before the timer part; C1 (do
  failures run to H_k).

## Prerequisite (no reading without it)

The analysis recomputes SAFE (LSTM) + P with the same code and must reproduce the stored
§15.15 result (`rldx_lead_lstm.json`) exactly: P lead, SAFE + P lead, gain and test-only
CI (0.271, 0.305, +0.034, [+0.008, +0.067] to the stored rounding). Otherwise the run
prints `REPRODUCTION FAILED` and nothing is read.

## Reading rules, written before scoring

- **R1, the XR-1 result replicates at its margin**, if all of:
  1. the test-only 95% CI of Hide-and-Seek + P's gain over P lies above 0;
  2. the point gain is at least 0.05 H_k;
  3. Hide-and-Seek + P has a test FPR of at most 0.075;
  4. its Wilson 95% upper bound is at most 0.10. With 100 test successes this allows at
     most four false alarms (P had three in §15.15).
- **Direction only:** if the CI lies above 0 but the point gain is below 0.05, the result
  is reported as "same direction, margin not reached".
- **R2, the warning develops during the rollout:** the CI of Hide-and-Seek − first lies
  above 0.
- **R3, not elapsed time or task identity:** the CIs of Hide-and-Seek − zero and
  Hide-and-Seek − shuffle both lie above 0.
- **Null flag:** if either null's mean gain reaches 0.05, or reaches Hide-and-Seek's gain,
  the fit itself produces gain on this cohort, and R1 is not read as evidence.
- Mechanism claims (R2, R3) are made only if the gain CI lies above 0.

## Expected outcome, written before scoring

- **A smaller gain than on XR-1 (+0.071).** Reasons:
  - RLDX-1's internal features were weak for SAFE (normalized LSTM ROC-AUC 0.61, §8.3);
  - fewer training rollouts (740, about 37 per task, against 1,824 on XR-1);
  - the fused path is chosen on in-sample failures;
  - only 100 test failures.
- **Point gain:** more likely positive than not, and most likely between 0 and 0.05, so R1
  is expected **not** to be met. The study does not predict whether the CI clears 0.
- **Strata:** the gain is expected to be larger on atomic than on composite tasks, as for
  SAFE on both policies.
- **Controls:** zero inputs and first inference are expected to gain under 0.02 over P.

## How the outcome will be reported

- **R1 met:** exploratory second-policy support for the XR-1 claim; E2 is still the test.
- **Direction only:** consistent with XR-1 but not at its margin on this cohort.
- **No gain (CI includes 0, point near 0):** the XR-1 advantage was not seen on RLDX-1. It
  may be policy-specific, or limited by RLDX's weaker features and smaller training set.

## Stop conditions

Any of these stops the run. Do not edit inputs to make it pass:
- the raw source is missing, or IDs, labels or lengths differ from the SAFE files;
- a scoring error, or a detector producing non-finite scores;
- the SAFE reproduction failing;
- a checksum change in the frozen inputs.

## Runbook

Run on the cluster from a pinned worktree, CPU only, following
[`cluster_experiment_runbook.md`](cluster_experiment_runbook.md). Never pull in the shared
checkout. `RLDXHS_SHA` is the full SHA of the pushed commit that contains this file.

### 1. Read-only discovery

```bash
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export RLDX_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530
export RLDX_EXPORT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_original_safe35_natural50_20260808_203158/official_safe_supported20_natural50
export RLDX_CAL=$RLDX_ROOT/calibration/lstm
export RLDX_LEAD_OUT=$(dirname "$(ls -d $STORAGE_BS/robocasa_checkpoints/safe/rldx1_lead_replication_*/rldx_lead_lstm.json | sort | tail -n 1)")
echo "15.15 run: $RLDX_LEAD_OUT"
"$SAFE_PY" - "$RLDX_EXPORT/conversion_report.json" "$RLDX_CAL/lstm_seed0/normalized_scores.jsonl" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
src = Path(json.load(open(sys.argv[1]))["source_dataset"])
print("RLDX_RAW=" + str(src), "| manifest exists:", (src / "manifest.jsonl").exists())
safe = {}
for line in open(sys.argv[2]):
    if line.strip():
        r = json.loads(line); safe[r["rollout_id"]] = (len(r["full_scores"]), bool(r["failed"]))
man = {}
for line in open(src / "manifest.jsonl"):
    if line.strip():
        r = json.loads(line); man[r["rollout_id"]] = r
missing = [i for i in safe if i not in man]
bad = [i for i in safe if i in man and (man[i]["valid_sequence_length"], bool(man[i]["failed"])) != safe[i]]
print("SAFE rollouts", len(safe), "| missing from raw", len(missing), missing[:3], "| length/label mismatches", len(bad), bad[:3])
r = man[sorted(safe)[0]]
with np.load(src / r["tensor_path"]) as p:
    f = p["features"] if "features" in p.files else p["safe_features"]
    print("keys", sorted(p.files), "| features", f.shape, f.dtype, "| model", r.get("model_family"))
    ok = not missing and not bad and f.ndim == 4 and "failed" in p.files
print("DISCOVERY OK" if ok else "STOP: raw data do not match the SAFE files or the score_cohorts format")
PY
```

Continue only with `DISCOVERY OK`.

### 2. Pinned code, inputs, checksums, tests

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export RLDX_RAW=<path printed by step 1>
export RLDXHS_SHA=<full SHA>
export RLDXHS_REPO=$PROJECT_FS/robocasa_rldx_hs_${RLDXHS_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin main
test -e $RLDXHS_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $RLDXHS_REPO $RLDXHS_SHA
test "$(git -C $RLDXHS_REPO rev-parse HEAD)" = "$RLDXHS_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export RLDXHS_TAG=rldx1_hide_and_seek_$(date +%Y%m%d_%H%M%S)
export RLDXHS_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$RLDXHS_TAG
export RLDXHS_LOG=$STORAGE_BS/robocasa_logs/eval/$RLDXHS_TAG
mkdir -p $RLDXHS_ROOT $RLDXHS_LOG
printf 'export %s=%q\n' RLDXHS_SHA "$RLDXHS_SHA" RLDXHS_REPO "$RLDXHS_REPO" RLDXHS_ROOT "$RLDXHS_ROOT" \
  RLDXHS_LOG "$RLDXHS_LOG" RLDX_ROOT "$RLDX_ROOT" RLDX_CAL "$RLDX_CAL" RLDX_RAW "$RLDX_RAW" \
  RLDX_EXPORT "$RLDX_EXPORT" RLDX_LEAD_OUT "$RLDX_LEAD_OUT" SAFE_PY "$SAFE_PY" > $RLDXHS_LOG/paths.env
sha256sum $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl $RLDX_CAL/split_manifest.json \
  $RLDX_LEAD_OUT/rldx_lead_lstm.json $RLDX_RAW/manifest.jsonl $RLDX_EXPORT/conversion_report.json \
  | tee $RLDXHS_LOG/frozen_inputs.sha256
cd $RLDXHS_REPO
PYTHONPATH=$RLDXHS_REPO CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_rldx_hide_and_seek \
  tests.test_safe_score_cohorts tests.test_rldx_lead_replication tests.test_safe_baselines 2>&1 | grep -E "^(Ran|OK|FAILED)"
"$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_hide_and_seek.py prepare --calibration-dir $RLDX_CAL \
  --output-dir $RLDXHS_ROOT/inputs | tee $RLDXHS_LOG/prepare.log
```

### 3. Score (no metrics; CPU, background)

```bash
cd $RLDXHS_REPO
SCORE_ARGS=(
  --dataset-dir $RLDX_RAW
  --runtime-bundle $RLDXHS_ROOT/inputs/rldx_bundle.json --train-ids $RLDXHS_ROOT/inputs/train_ids.json
  --cohort calibration $RLDXHS_ROOT/inputs/calibration_ids.json
  --cohort test $RLDXHS_ROOT/inputs/test_ids.json
  --detectors hide_and_seek hide_and_seek_first hide_and_seek_zero hide_and_seek_task_shuffle
  --feature-aggregation last --seeds 0 1 2 --score-train
  --output-dir $RLDXHS_ROOT/scores
)
PYTHONPATH=$RLDXHS_REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 \
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${SCORE_ARGS[@]}" > $RLDXHS_LOG/score.log 2>&1 &
echo "score PID: $! on $(hostname)"
```

It prints `train 740; calibration 60, test 200`, then one `<detector> seed <s> done` line per
detector and seed (12 in all).

### 4. Analysis

```bash
source "$(ls -d $STORAGE_BS/robocasa_logs/eval/rldx1_hide_and_seek_* | sort | tail -n 1)/paths.env"
cd $RLDXHS_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_hide_and_seek.py run --calibration-dir $RLDX_CAL \
  --inputs-dir $RLDXHS_ROOT/inputs --scores-dir $RLDXHS_ROOT/scores \
  --reference-result $RLDX_LEAD_OUT/rldx_lead_lstm.json \
  --output $RLDXHS_ROOT/rldx_hide_and_seek.json 2>&1 | tee $RLDXHS_LOG/run.log
sha256sum -c $RLDXHS_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
```

Report the whole `run.log` and `INPUTS UNCHANGED`.

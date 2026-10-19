# Oracle stage-overdue detector vs the per-task timer P: go/no-go plan

Written 2026-10-19, before any stage-overdue alarm, lead or FPR was computed on
any cohort. Exploratory: it decides whether a progress-aware detector is worth
developing. It is not a confirmatory test and makes no paper claim.

## Question

Can a detector that knows the robot's progress through the task give more
warning lead than the per-task quantile timer P (the registered comparator in
amendment 1) at the same 5% false-positive budget? The detector here is an
*oracle*: it reads the simulator's ordered subtask predicates, which a deployed
robot does not have. If even the oracle cannot beat P by a useful margin, a
learned progress estimator will not either, and the progress direction is
dropped.

## Data the analysis can use

- **Not the August cohorts.** The development (1,824), calibration (228) and
  prospective test (760) cohorts were collected with
  `--no-record-subtask-trace` (docs/xr1_prospective_matched_fpr.md,
  docs/safe_xiaomi_robotics_1.md). They hold no per-inference stage
  predicates. Annotating them by replaying stored actions was tried for XR-1
  and rejected: replay was slower than collecting and sometimes changed the
  terminal outcome (experiment report §15.7). Keeping only the rollouts whose
  replay matches would select on replay stability, so it is not used here.
- **Used: XR-1 cohorts recorded with live predicates** (`--record-subtask-trace`):
  - primary: `xr1_dense_subtask_live_5tasks_50each_20260817_195216/subtask_training/merged_5tasks_50each`,
    250 natural parents, 50 per task, no outcome quota;
  - optional, pooled if it exists and validates: the context cohort
    `xr1_stage_context_5tasks_50each_*/merged_250` (seed base 900007).
- **Tasks:** ArrangeBreadBasket, ArrangeTea, BreadSelection,
  CuttingToolSelection and GarnishPancake. All five are composite and all are in
  the 38-task August set. There are no atomic tasks with stage traces, so the
  atomic/composite split cannot be reported. For an atomic task the oracle
  reduces to P by construction (one stage, whose duration is the rollout
  duration). Results are reported overall and per task.
- **Never touched:** the confirmatory cohort (base seed 1300007). The script
  refuses any rollout whose environment seed lies in [1300007, 1325007).
- These parents were already used to develop stage-aware SAFE (§15.7–15.8),
  but no timer or stage-duration detector has been evaluated on them.

## Definitions (shared with lead_time.py and quantile_timer.py)

- Inference index t = 1..L, with L the number of genuine policy inferences.
  H_k = `task_horizons[k]` from the frozen August `runtime_bundle.json`.
- Lead of a detected failure = (L − T)/H_k, where T is the first alarm
  inference; a miss counts as 0. An alarm at the final inference counts.
- **P (comparator, registered):** rho_k = 95th percentile (numpy linear) of the
  August development success durations L/H_k, read from the time-risk curves'
  active counts. Alarm at t ≥ ceil(min(1, rho_k + delta)·H_k).
- **Oracle stage-overdue + P (primary detector):** at inference t the current
  stage s_t is the next ordered semantic subtask not yet completed
  (`inference_records[].segment_index` of the recorded subtask-SAFE trace). When
  every subtask is complete but the task has not yet succeeded, s_t is a final
  "post-completion" stage. Time in stage tau_t = the number of inferences since
  the current stage began, including t. Q_{k,j} = the 95th percentile, over
  the development-fold successes of task k, of the number of inferences spent
  in stage j divided by H_k. A success that passed through a stage between two
  inferences contributes 0. The detector alarms at the first t at which
  t ≥ ceil(min(1, rho_k + delta)·H_k) or tau_t/H_k ≥ Q_{k,s_t} + delta.
  One shared margin delta.
- **Margin, for every detector:** split-conformal rule of Proposition 1 at
  alpha = 0.05 on calibration-fold successes. Success i is silent at
  V_i = max(L_i/H − rho_k, max_t(tau_t/H − Q_{k,s_t}), 0) (plus 1e-9), and
  infinite if L_i ≥ H_k. delta = the ceil((n+1)(1−alpha))-th smallest V_i.
- **Folds:** every parent is scored once, by 5-fold cross-fitting stratified by
  (task, outcome), with seed 0. For test fold f, folds f+1 and f+2 (mod 5) are
  calibration and the other two are development. Development is used only for
  Q_{k,j}, and for rho in the local-P sensitivity check.

## Endpoint and decision rule

- **Primary endpoint:** paired difference in mean lead over failures,
  (oracle stage + P) − P, pooled over the five test folds, at q = 0.95.
- **Bootstrap:** task then rollout, 2,000 replicates, seed 0, as in
  quantile_timer.py. With five tasks the task-level resampling is coarse.
  A task-stratified rollout bootstrap is reported next to it as a sensitivity
  check.
- **GO** (the progress direction is worth pursuing) only if:
  1. the point gain is at least 0.05 H_k; and
  2. the 95% CI (task-then-rollout bootstrap) lies entirely above 0.
- **NO-GO** otherwise.
- **INCONCLUSIVE** instead of GO if the oracle passes both conditions against
  the registered P but its point gain over the local P (rho_k refit on the
  development-fold successes of these cohorts) is below 0.05. In that case the
  gain could come from a shift in success durations between the August and the
  live collections, not from progress information.
- **Descriptive only:** oracle stage-only (no timer), oracle + local P, and the
  quantile sweep q ∈ {0.75, 0.90, 1.0}. Also reported: recall, FPR with
  Wilson intervals, the fraction of failures alarmed by 0.5 H_k and 0.75 H_k,
  per-task values, and P's lead on these five tasks in the August test cohort,
  as a check that the live cohorts are representative.

## Known limits, stated before running

- The oracle uses privileged simulator state. A GO says the information is
  worth estimating, not that a deployable detector exists.
- Five composite tasks out of 38. A GO here does not extend to the August task
  set without a broader traced collection.
- The regression of a completed stage is not used as an alarm signal. This
  rule tests only stage-overdue.

## Local check (done before the cluster run)

`python3 -I scripts/xr1_lead_time/stage_overdue_oracle.py reproduce-p <review_data> <quantile_timer.json>`
re-fits P from this file's copy of the definitions on the opened August cohorts and
matches the registered dry run at every quantile (q = 0.95: delta 0.1067, 17 FP,
FPR 0.0447, mean lead 0.1952). On the August test, P's mean lead on these five tasks is
0.186 (FPR 0.02), close to its 0.195 over all 38 tasks.

## Cluster run (CPU only, about a minute)

Run it from a pinned detached worktree so the shared checkout used by the recovery runs
is not touched. The script needs only numpy. The run needs this file and the script
committed and pushed on `codex/safe-xiaomi-robotics-1`.

```bash
export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export PY=$STORAGE_BS/envs/robocasa_openpi/bin/python
git -C $PROJECT_FS/robocasa fetch origin codex/safe-xiaomi-robotics-1
export SHA=$(git -C $PROJECT_FS/robocasa rev-parse origin/codex/safe-xiaomi-robotics-1)
export WT=$PROJECT_FS/robocasa_stage_oracle_${SHA:0:8}
test -d $WT || git -C $PROJECT_FS/robocasa worktree add --detach $WT $SHA
test -f $WT/scripts/xr1_lead_time/stage_overdue_oracle.py

export BUNDLE=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json
export LIVE=$STORAGE_BS/robocasa_rollouts/safe/xr1_dense_subtask_live_5tasks_50each_20260817_195216/subtask_training/merged_5tasks_50each
export OUT=$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_overdue_oracle_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUT
test -f $BUNDLE && test -f $LIVE/manifest.jsonl
wc -l $LIVE/manifest.jsonl
# Optional second traced cohort (seed base 900007): pool it only if it exists and validates.
ls -d $STORAGE_BS/robocasa_rollouts/safe/xr1_stage_context_5tasks_50each_*/merged_250 2>/dev/null
sha256sum $BUNDLE $LIVE/manifest.jsonl | tee $OUT/inputs_sha256_before.txt

cd $WT
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
nice -n 10 $PY -I -u scripts/xr1_lead_time/stage_overdue_oracle.py run \
  --runtime-bundle $BUNDLE --dataset-dir $LIVE \
  --output $OUT/stage_overdue_oracle.json 2>&1 | tee $STORAGE_BS/robocasa_logs/eval/$(basename $OUT).log

sha256sum $BUNDLE $LIVE/manifest.jsonl | diff - $OUT/inputs_sha256_before.txt && echo INPUTS UNCHANGED
```

To pool the context cohort, add its `merged_250` path after `$LIVE` in `--dataset-dir`,
and add it to both `sha256sum` lines. Paste back the printed summary and the
`VERDICT:` line.

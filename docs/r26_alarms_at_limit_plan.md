# R26: Table I with an alarm at the limit counted as a miss

Branch `claude/r22-fixed-paths`. Written 2026-11-22, before the run. Descriptive: no hypothesis is
added, no threshold, false alarm or lead changes, and the reporting below is fixed now.

## Why

Table I (XR-1, monitors and timers alone, 5% FPR target) counts any alarm on a failure as a detection. An alarm
at the failure's final inference counts too, with lead 0.

Every timer's alarm is capped at the final inference (`hk_free.timer_idx`: `ceil(min(H_k, …))`). For the
elapsed-time timer E this cap is its only use of H_k, and it adds detections that carry no warning:
- E alarms only at the final inference on 190 of 380 failures, on unseen tasks (R19) and on seen tasks
  (inferred in draft v6's README);
- its bACC, 0.975, is 0.725 with those alarms as misses.

A reviewer of draft v5 asks to label or change this convention. Draft v6 labels it in Table I's note. R26
recounts every row of Table I with an alarm at a failure's final inference as a miss, so the table itself can
use the stricter rule. The rule then matches the lead, which already gives such an alarm zero.

## What R26 computes (XR-1, the rows of Table I)

- **Rule.** A failure counts as detected only if the monitor alarms before its final inference. Successes are
  unchanged: any alarm on a success is a false alarm, as before.
- **Unchanged:** thresholds, false alarms, FPR and mean lead.
- **Recounted, per row, side (seen, unseen) and seed:**
  - `alarms_at_limit`: failures whose first alarm is at their final inference;
  - `tpr_before_limit = (detections − alarms_at_limit) / failures`;
  - `bacc_before_limit = (tpr_before_limit + 1 − FPR) / 2`.
- **How.** A failure's alarm is at its final inference exactly when it is a hit with lead 0, because
  lead = (L − a)/H_k with a ≤ L, and a failure runs to L = H_k. Every Table I code path already returns each
  failure's hit and lead, so the recount reuses it unchanged:
  - **OOD scores, SAFE-MLP, SAFE-LSTM and Hide-and-Seek** (constant threshold): `per_seed_detection.threshold`
    and `outcomes`, per seed and pooled over folds on unseen tasks.
  - **G:** the same path, on the score t/H_k.
  - **P:** `confirm_lead.fit_fast` and `cross_task_lead.raw_eval`, as `per_task_timer_seen`.
  - **E:** `hk_free`'s timer alone (`fit`, `raw_eval`, `pool`), on seen tasks and per fold on unseen tasks, as
    R14's `timers_alone`.
  - **Own bands:** R21 already saved each band's `alarms_at_limit` (`r19_extras.outcome`), so these rows are
    recounted from R21's output, with no new computation.
- **Code:**
  - `scripts/xr1_lead_time/r26_alarms_at_limit.py` (the recount and the gates);
  - `seen_unseen_tables.py run --r26 --r26-r14 … --r26-r21 …`;
  - tests: `tests/test_r26_alarms_at_limit.py` and `RunTests.test_r21_original_rules`.

## Gates

- **R0:** every recounted row's ordinary values (TPR, FPR, bACC, mean lead, false alarms) equal this run's
  table1 (≤ 1e-12).
- **R1:** they equal R14's table1 and, for E, R14's `timers_alone` (FPR, mean lead, false alarms; ≤ 1e-12).
  R14's output is Table I's source.
- **R2:** P on seen tasks and E on unseen tasks equal R21's `timer_alone`: false alarms, TPR, mean lead and
  alarms at the limit (≤ 1e-12).
- The run's own T6 gate (`--r6-result`) also holds: every published R6 row is reproduced.

Any failure prints `R26 FAILED`, and no R26 number is used.

## Expectations (written before running)

- **P (seen):** 20 alarms at the limit; bACC 0.972 → 0.946 (R19).
- **E:** 190 on unseen tasks, bACC 0.975 → 0.725 (R19). 190 on seen tasks, bACC 0.974 → 0.724 (inferred:
  R18's E at the same 20 false alarms has recall 0.5 before the limit, and no XR-1 limit lies between the
  two operating points).
- **G:** none at the limit, on either side; bACC unchanged (0.988).
- **Constant-threshold monitors:** no expectation is fixed. A running maximum that first crosses at the
  final inference should be rare, so their bACC should drop by less than 0.01.
- **Own bands (from R21, already known):** mean alarms at the limit per seed are 0 to 2.3, and bACC drops
  by at most 0.003. For example, SAFE-MLP's band on unseen tasks goes from 0.687 ± 0.003 to 0.684 ± 0.005.

## Reporting (fixed now)

- **Table I** reports `bacc_before_limit` in its bACC columns (mean ± std over seeds). Its note becomes one
  line: "A failure alarmed only at its final inference counts as missed." The ranking marks are recomputed.
- **Sec. VI-B:** the learned detectors' recall ("at most 69%") is recomputed under the same rule. The draft
  v6 phrases "even with alarms at the limit counted as misses" and "its high bACC comes from alarms at the
  final inference" go. E's bACC is stated as 0.72.
- **Conclusion and limitations:** the qualifier on timers that warn before the limit stays. E's alarm cap
  stays in the limitations.

## Runbook (CPU; one run)

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=2:00:00
```

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export STORAGE_BS=/gs/bs/tga-shinoda/felid PROJECT_FS=/gs/fs/tga-shinoda/felid
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r11_rows_20261030_220636/paths.env
source $STORAGE_BS/robocasa_logs/eval/rldx1_unseen_20261029_135501/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r14_hk_free_20261106_191531/paths.env
source $STORAGE_BS/robocasa_checkpoints/safe/rldx1_e1b_test_table_20261113_145031/session.env
for v in E1_OUT XR1_RUNTIME_BUNDLE RS_ROOT XT_ROOT R6_ROOT XR1_BASELINE_ROOT AMEND2_ROOT OOD_ROOT PSEED_ROOT R11_ROOT; do
  test -n "${!v}" || echo "STOP: $v is unset"; done
export SAFE_PY=${SAFE_PY:-$STORAGE_BS/envs/vla_safe/bin/python}
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R26_SHA=<full SHA of the pushed commit containing this plan>
export R26_REPO=$PROJECT_FS/robocasa_r26_${R26_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R26_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R26_REPO $R26_SHA
test "$(git -C $R26_REPO rev-parse HEAD)" = "$R26_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R26_R14=$R26_REPO/reports/safe_robocasa365_research_plan_20260823/evidence/xr1_r14_hk_free_20261106_191531/seen_unseen_tables_r14.json
export R26_R21=$R26_REPO/reports/r21_native_bands_evidence/seen_unseen_tables_r21.json
export R26_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r26_alarms_at_limit_$(date +%Y%m%d_%H%M%S)
export R26_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R26_ROOT)
mkdir -p $R26_ROOT $R26_LOG
export XR1_FOLDS=$R26_REPO/scripts/xr1_lead_time/cross_task_folds.json
sha256sum $R26_R14 $R26_R21 | cut -c1-16
cd $R26_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r26_alarms_at_limit \
  tests.test_xr1_seen_unseen_tables 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- no `STOP` line prints (the same environment files as R25, whose XR-1 run used these variables);
- `HEAD OK` prints;
- the hashes start with `1578f312` (R14) and `53599170` (R21);
- the tests end with `OK`.

```bash
cd $R26_REPO
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice "$SAFE_PY" -I -u scripts/xr1_lead_time/seen_unseen_tables.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --folds $XR1_FOLDS \
  --safe-fold-dirs $RS_ROOT/folds/fold_{0,1,2,3} --xt-fold-dirs $XT_ROOT/fold_{0,1,2,3} \
  --actprobe-fold-dirs $R6_ROOT/actprobe/fold_{0,1,2,3} $R6_ROOT/actprobe/fold_all_b1ids \
  --ood-fold-dirs $R6_ROOT/ood/fold_{0,1,2,3} \
  --ood-source-bundle $XR1_BASELINE_ROOT/develop/baseline_bundle.json \
  --amend2-scores $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl \
  --amend2-normalizations $AMEND2_ROOT/scores/normalizations.json \
  --ood-seen 0 $OOD_ROOT/seed0/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed0/heldout/test_with_baselines.jsonl \
  --ood-seen 1 $OOD_ROOT/seed1/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed1/heldout/test_with_baselines.jsonl \
  --ood-seen 2 $OOD_ROOT/seed2/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed2/heldout/test_with_baselines.jsonl \
  --run-c $PSEED_ROOT/per_seed_fused_heldout.json --ood-fused $OOD_ROOT/ood_per_seed_fused_heldout.json \
  --cross-task-result $XT_ROOT/cross_task_lead.json --cross-task-safe-result $RS_ROOT/cross_task_safe.json \
  --actprobe-reference-normalizations $PSEED_ROOT/actprobe_scores/normalizations.json \
  --actprobe-t-fold-dirs $R11_ROOT/actprobe_t/fold_{0,1,2,3,all} \
  --safe-lstm-fold-dirs $R11_ROOT/safe_lstm/fold_{0,1,2,3,all} \
  --r6-result $R6_ROOT/seen_unseen_tables_b1ids.json \
  --r26 --r26-r14 $R26_R14 --r26-r21 $R26_R21 \
  --output $R26_ROOT/seen_unseen_tables_r26.json' > $R26_LOG/r26_xr1.log 2>&1 &
echo "started; R26_ROOT=$R26_ROOT"
```

**Runtime:** loading the fold directories dominates, as in R25's XR-1 run; there is no bootstrap. Expect
well under an hour.

**When the log ends:**

```bash
grep -E "R26 OK|R26 FAILED|REPRODUCTION FAILED|Traceback" $R26_LOG/r26_xr1.log
sed -n '/^R26 XR-1/,$p' $R26_LOG/r26_xr1.log
mkdir -p $R26_ROOT/evidence && cp $R26_ROOT/*.json $R26_LOG/*.log $R26_ROOT/evidence/ && ls $R26_ROOT/evidence
```

The log must end with `R26 OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r26_alarms_at_limit_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r26_alarms_at_limit_evidence/
```

## Result (2026-11-22, from `c51b0939`): `R26 OK`

The log ends with `R26 OK`. All three gates hold with max |diff| 0:
- R0: 59 values against this run's table1;
- R1: 61 values against R14's table1 and timers_alone;
- R2: 2 timers against R21's timer_alone.

The local test suites passed too (27 tests). Evidence is in `reports/r26_alarms_at_limit_evidence/`:
- `seen_unseen_tables_r26.json`, sha256 `4700d436…`;
- `r26_xr1.log`, gitignored and kept locally.

**Result** (XR-1; bACC as printed → with an alarm at the limit as a miss; recall before the limit; alarms at
the limit per seed; monitors: mean ± std over seeds):

| Row | Seen | Unseen |
| --- | --- | --- |
| P | 0.972 → 0.946; 0.947; 20 | – |
| E | 0.974 → 0.724; 0.500; 190 | 0.975 → 0.725; 0.500; 190 |
| G | 0.988 → 0.988; 1.000; 0 | 0.988 → 0.988; 1.000; 0 |
| Mahalanobis | 0.818 → 0.814 ± 0.002; 0.681; 3.0 | 0.726 → 0.726; 0.518; 0 |
| Cosine k-NN | 0.788 → 0.784; 0.629; 3.0 | 0.729 → 0.729; 0.526; 0 |
| PCA-KMeans | 0.796 → 0.786 ± 0.002; 0.635; 8.3 | 0.728 → 0.728; 0.523; 0 |
| RND | 0.793 → 0.788 ± 0.001; 0.640; 3.7 | 0.728 → 0.728; 0.525; 0 |
| SAFE-MLP | 0.763 → 0.761 ± 0.005; 0.566; 1.0 | 0.713 → 0.712; 0.471; 0.3 |
| SAFE-LSTM | 0.610 → 0.608 ± 0.013; 0.252; 1.7 | 0.686 → 0.682 ± 0.020; 0.404; 3.3 |
| Hide-and-Seek | 0.742 → 0.739 ± 0.029; 0.527; 2.0 | 0.790 → 0.786 ± 0.002; 0.633; 3.3 |
| SAFE-MLP band | 0.713 → 0.713; 0.482; 0 | 0.687 → 0.684 ± 0.005; 0.398; 2.3 |
| SAFE-LSTM band | 0.601 → 0.600 ± 0.016; 0.262; 0.7 | 0.568 → 0.567 ± 0.010; 0.192; 0.3 |
| Hide-and-Seek band | 0.751 → 0.751; 0.581; 0 | 0.692 → 0.692; 0.468; 0.3 |

**Against the expectations:** all hold.
- P, E and G match R19 and the inference for seen E: 190 alarms at the limit, bACC 0.724.
- The constant-threshold monitors drop by at most 0.010 (PCA-KMeans, seen tasks).
- The own bands drop by at most 0.003.

**Reporting (as fixed before the run):**
- Draft v6 (paper commit `26cc63e7`) reports `bacc_before_limit` in Table I, with a one-line note.
- The learned detectors' seen-task recall before the limit is at most 68% (Mahalanobis 0.681).
- E's bACC is 0.72.

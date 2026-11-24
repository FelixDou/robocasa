# R27b: why did the own-band gain move under the published split?

Branch `claude/r22-fixed-paths`. Written 2026-11-24, after R27 and before this run. Descriptive: no hypothesis is
added, and the reading and the reporting below are fixed now.

## Why

R27 refit Hide-and-Seek's own band on 20 random assignments of the published 30/70 split. The direction held in all 20
on both policies, but XR-1's gain over matched E fell from 0.147 (our fixed 50/50 split) to a median of 0.064, and
0.147 is above all 20 assignments. Two explanations lead to different text:

- the **ratio**: fitting the mean and modulation on 30% of the successes works worse than on 50%;
- the **assignment**: our fixed split (per task, rollouts sorted by ID alternate) happened to be favorable.

The paper's opening figure should also show the published-split result with its spread, which needs each assignment's
curve (R27 saved only a summary of it), and its illustrated rollout should use the alarms of a representative
published-split band.

## What R27b computes (unseen tasks, both policies)

- **Published:** R27's 20 random 30/70 assignments (seeds 27 to 46), now also saving each assignment's curve: the mean
  of the three models' FPR and lead at each calibration target.
- **Half:** 20 random 50/50 assignments, stratified by task (`numpy.random.default_rng(127 + j)`, rollouts sorted by ID
  first), with the same quantities: the curve, the ensemble's gain over matched E, the first-step control.
- **Example:** each failed test rollout's alarm under the ensemble band and E, for the *representative* 30/70
  assignment: the one whose gain over matched E is closest to the median (ties: the first). It is fixed by this rule,
  so the illustrated band is not chosen for looking good.
- **Unchanged:** scores, folds, SAFE's routine, E, and the test rollouts.
- **Code:**
  - `scripts/xr1_lead_time/r27b_split_controls.py`;
  - `seen_unseen_tables.py run --r27b` and `rldx_test_table.py run --r27b`;
  - tests: `tests/test_r27b_split_controls.py`, and the R21 and R25 end-to-end tests.

## Gates

- **M1:** the 20 published assignments reproduce R27's saved rows (false alarms, FPR, lead, gain, first-step Δ and
  target, curve crossing; ≤ 1e-12).
- **M2:** every random 50/50 assignment is a partition of the calibration successes, with half of each task in the
  reference set.
- **M3:** the example's saved alarms give its ensemble's and E's mean lead (≤ 1e-12).

Any failure prints `R27b FAILED`, and no R27b number is used.

## Reading (fixed now)

Per policy, for the gain over matched E and for the first-step control:

- **The fixed assignment was favorable** if the fixed 50/50 value is above at least 18 of the 20 random 50/50
  assignments.
- **Otherwise**, the fixed value is typical of 50/50 splits, and any difference from 30/70 is attributed to the ratio.

## Reporting (fixed now, whatever the reading)

- **The headline is the published 30/70 split:** the median and range over its 20 assignments, in the abstract,
  Sec. VI-B and the conclusion. The fixed 50/50 numbers (0.147, 0.143, 0.273, 0.146) no longer lead.
- **Fig. 1** shows the published-split curves (median and range over assignments) against E, and the illustrated
  rollout uses the representative assignment's alarms.
- **The text** states the reading above in one sentence.
- **Shinoda sensei** is told in the message, with the 50/50 and 30/70 numbers side by side.

## Expectations (written before running)

- The 30/70 rows reproduce R27.
- No expectation is fixed for the random 50/50 assignments.

## Runbook (CPU; both runs in parallel)

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
for v in E1_OUT XR1_RUNTIME_BUNDLE RS_ROOT XT_ROOT R6_ROOT XR1_BASELINE_ROOT AMEND2_ROOT OOD_ROOT PSEED_ROOT R11_ROOT \
  RLDX_ROOT RLDXHS_ROOT R16_ROOT E1B_ROOT E1_FREEZE R7_ROOT R7S RLDXPS_ROOT FOLDS; do
  test -n "${!v}" || echo "STOP: $v is unset"; done
export SAFE_PY=${SAFE_PY:-$STORAGE_BS/envs/vla_safe/bin/python}
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R27B_SHA=<full SHA of the pushed commit containing this plan>
export R27B_REPO=$PROJECT_FS/robocasa_r27b_${R27B_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R27B_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R27B_REPO $R27B_SHA
test "$(git -C $R27B_REPO rev-parse HEAD)" = "$R27B_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R27B_EV=$R27B_REPO/reports
export R27B_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r27b_split_controls_$(date +%Y%m%d_%H%M%S)
export R27B_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R27B_ROOT)
mkdir -p $R27B_ROOT $R27B_LOG
export XR1_FOLDS=$R27B_REPO/scripts/xr1_lead_time/cross_task_folds.json
echo "FOLDS=$FOLDS"
sha256sum $R27B_EV/r18_lead_fpr_evidence/seen_unseen_tables_r18.json $R27B_EV/r18_lead_fpr_evidence/rldx_e1b_r18.json \
  $R27B_EV/r27_band_split_evidence/seen_unseen_tables_r27.json $R27B_EV/r27_band_split_evidence/rldx_e1b_r27.json | cut -c1-8
cd $R27B_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r27b_split_controls tests.test_r27_band_split \
  tests.test_xr1_seen_unseen_tables tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- no `STOP` line prints;
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes are, in order: `624ee65a`, `2162a01d` (R18), `431b93c8`, `52f8b5d4` (R27);
- the tests end with `OK`.

```bash
cd $R27B_REPO
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
  --r27b --r27b-r18 $R27B_EV/r18_lead_fpr_evidence/seen_unseen_tables_r18.json \
  --r27b-r27 $R27B_EV/r27_band_split_evidence/seen_unseen_tables_r27.json \
  --output $R27B_ROOT/seen_unseen_tables_r27b.json' > $R27B_LOG/r27b_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r27b --r27b-r18 $R27B_EV/r18_lead_fpr_evidence/rldx_e1b_r18.json \
  --r27b-r27 $R27B_EV/r27_band_split_evidence/rldx_e1b_r27.json \
  --output $R27B_ROOT/rldx_e1b_r27b.json' > $R27B_LOG/r27b_rldx.log 2>&1 &
echo "started; R27B_ROOT=$R27B_ROOT"
```

**Runtime:** 40 evaluations per policy, with no bootstrap. R27's 21 took under ten minutes, so expect under half an
hour.

**Progress:** `grep "R27b .* split" $R27B_LOG/r27b_*.log | tail -n 2`. When both logs end:

```bash
grep -E "R27b OK|R27b FAILED|REPRODUCTION FAILED|Traceback" $R27B_LOG/r27b_*.log
sed -n '/^R27b XR-1/,$p' $R27B_LOG/r27b_xr1.log; sed -n '/^R27b RLDX-1/,$p' $R27B_LOG/r27b_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R27B_ROOT/evidence && cp $R27B_ROOT/*.json $R27B_LOG/*.log $R27B_ROOT/evidence/ && ls $R27B_ROOT/evidence
```

Both logs must end with `R27b OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r27b_split_controls_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r27b_split_controls_evidence/
```

## Result (2026-11-24, run `r27b_split_controls_20261124_020457`, from `b103e21e`): `R27b OK`, the fixed assignment was favorable

Both runs ended with `R27b OK` and `INPUTS UNCHANGED`. Gates M1 to M3 pass on both policies. Evidence is in
`reports/r27b_split_controls_evidence/`:
- `seen_unseen_tables_r27b.json`, sha256 `c0f430d1…`;
- `rldx_e1b_r27b.json`, `31d82f21…`;
- logs, gitignored and kept locally.

**Result** (unseen tasks; fractions of H_k; random splits: median [min, max] over 20 assignments):

| | Fixed 50/50 | Random 50/50 | Published 30/70 |
| --- | --- | --- | --- |
| XR-1, gain over matched E | 0.147 | 0.094 [0.027, 0.142] | 0.064 [0.023, 0.098] |
| RLDX-1, gain over matched E | 0.143 | 0.106 [0.043, 0.155] | 0.101 [0.053, 0.148] |
| XR-1, full − first step | 0.273 | 0.213 [0.135, 0.272] | 0.178 [0.129, 0.222] |
| RLDX-1, full − first step | 0.146 | 0.095 [0.035, 0.157] | 0.091 [0.035, 0.157] |

**Reading.** The fixed assignment was favorable for both quantities on both policies: it is above 20 of 20 random
50/50 assignments on XR-1 and 18 of 20 on RLDX-1 (the rule needs at least 18). The direction is robust: the gain and
the first-step difference are positive in all 80 random assignments. On XR-1 the 30/70 ratio also lowers the medians
(0.094 to 0.064 and 0.213 to 0.178). On RLDX-1 the two ratios are close.

**Example.** The representative 30/70 assignment is 6 on XR-1 (gain 0.0646) and 2 on RLDX-1 (0.1015). Fig. 1 uses XR-1's
failed DeliverStraw rollout `74717434c269675fe5e00a6f` under assignment 6: band alarm at inference 64, E at 127, limit
160, 0.8 s per inference, so warnings of 76.8 s and 26.4 s, a difference of 50.4 s. It was chosen for illustration.

**Reporting.** As fixed above, the paper's headline is now the published 30/70 split, and Fig. 1 shows its curves.

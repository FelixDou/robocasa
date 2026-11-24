# R27c: every own band on R27's random 30/70 assignments, against E, G and P

Status: descriptive, written before the run (2026-11-24). Follows R27 (`docs/r27_band_split_plan.md`) and R27b
(`docs/r27b_split_controls_plan.md`).

## Why

The paper now reports Hide-and-Seek's own band over R27's 20 random 30/70 assignments of the calibration successes
(median [range]). Three of its numbers still come from the single fixed 50/50 split of R19 to R25:

- Table II's SAFE-MLP and SAFE-LSTM rows (gain over matched E, with a 95% CI);
- Sec. VI-B's "even with it, the ensemble is not clearly earlier than G (0.070, 0.019) or P on seen tasks (0.047,
  0.086)", which the conclusion's "no monitor alone clearly beats a timer told the limit" rests on;
- Sec. VI-B's "SAFE-MLP's [band] warns later than P on RLDX-1's seen tasks (−0.126, [−0.232, −0.037])".

R27b showed that the fixed 50/50 assignment was a favorable draw for Hide-and-Seek. R27c puts every head and every
comparison on the same random assignments, so that Table II and the text use one protocol. SAFE's paper states no
split of its calibration successes, so SAFE's bands use the same 30/70 assignments as Hide-and-Seek's.

## What R27c computes (both policies)

- **Assignments.** R27's 20 random 30/70 splits (seeds 27 to 46, `r27_band_split.split_30_70`), stratified by task.
  On seen tasks the calibration successes of every task are split with the same seed, which gives the same
  assignment (gate Q2 checks it).
- **Heads.** Three-seed ensembles (average of the normalized raw scores, as in R21): SAFE-MLP (`safe_mlp` on XR-1,
  `safe_indep` on RLDX-1), SAFE-LSTM and Hide-and-Seek, each under SAFE's routine at alpha = 0.05.
- **Comparisons.** Unseen tasks: E (`G_abs`) and G, each matched to the band's test false alarms (R21's
  `evaluate`: the timer at the smallest margin with at most as many). Seen tasks: P, matched the same way.
- **Per assignment, head, side and timer:** the band's false alarms, FPR and mean lead, and the gain in mean lead over
  the matched timer.
- **Summary:** median [min, max] over the 20 assignments, and the number of assignments with a positive and with a
  negative gain.
- **Unchanged:** scores, folds, SAFE's routine, the timers, the test rollouts. No bootstrap.

## Gates

- **Q0:** at R19's fixed 50/50 split, every head's ensemble on every side and timer equals R21's saved point
  (`r21.bootstrap.{side}.timers.{timer}.{head}.standalone`): false alarms equal, matched gain within 1e-12. A missing
  reference entry fails the gate.
- **Q1:** Hide-and-Seek's 30/70 rows against E equal R27's saved rows (false alarms, matched gain; 1e-12).
- **Q2:** every assignment is a 30/70 partition of the calibration successes on both sides, with the same reference
  rollouts on both.

Any failure prints `R27c FAILED`, and no R27c number is used.

## Reading (fixed now)

"Clearly" uses R27's convention: in at least 18 of the 20 assignments.

1. **Table II.** The SAFE rows report the median [range] of the gain over matched E, the median FPR and the median
   false alarms over the 20 assignments, like Hide-and-Seek's rows. The fixed 50/50 split leaves Table II.
2. **"SAFE's bands show no clear gain over the timers"** (abstract, contribution, conclusion) stays if, for each SAFE
   head on each policy, the gain over matched E (unseen) and over matched P (seen) is positive in fewer than 18 of 20
   assignments. Otherwise the paper states the gain that is positive in at least 18.
3. **Hide-and-Seek against G (unseen) and P (seen).** "Not clearly earlier than G or P" stays if each gain is positive
   in fewer than 18 of 20 assignments; the numbers become the 30/70 medians [ranges]. If a gain is positive in at
   least 18, the paper says Hide-and-Seek warns earlier than that timer, and the conclusion's "no monitor alone
   clearly beats a timer told the limit" is revised.
4. **SAFE-MLP against P on RLDX-1's seen tasks.** "Warns later than P" stays if the gain is negative in at least 18
   of 20 assignments; otherwise the sentence says it is not clearly different from P.
5. **The fixed 50/50 split** then remains only in Fig. 2's own-band curves; Sec. VI-B keeps one clause saying that
   this split beat 20 and 18 of 20 random 50/50 splits (R27b).

## Code

- `scripts/xr1_lead_time/r27c_split_all_bands.py`;
- hooks: `seen_unseen_tables.py --r27c --r27c-r21 --r27c-r27 [--r27c-splits]` (XR-1; needs `--safe-lstm-fold-dirs`)
  and `rldx_test_table.py --r27c --r27c-r21 --r27c-r27 [--r27c-splits]` (RLDX-1);
- `tests/test_r27c_split_all_bands.py` (synthetic folds: shared assignment, `point` equals R21's `evaluate`, gates
  pass on consistent references and fail on perturbed or missing ones, summary counts).

## Runbook (CPU; both runs in parallel)

Read `docs/cluster_experiment_runbook.md` first.

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
export R27C_SHA=<full SHA of the pushed commit containing this plan>
export R27C_REPO=$PROJECT_FS/robocasa_r27c_${R27C_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R27C_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R27C_REPO $R27C_SHA
test "$(git -C $R27C_REPO rev-parse HEAD)" = "$R27C_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R27C_EV=$R27C_REPO/reports
export R27C_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r27c_split_all_bands_$(date +%Y%m%d_%H%M%S)
export R27C_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R27C_ROOT)
mkdir -p $R27C_ROOT $R27C_LOG
export XR1_FOLDS=$R27C_REPO/scripts/xr1_lead_time/cross_task_folds.json
echo "FOLDS=$FOLDS"
sha256sum $R27C_EV/r21_native_bands_evidence/seen_unseen_tables_r21.json $R27C_EV/r21_native_bands_evidence/rldx_e1b_r21.json \
  $R27C_EV/r27_band_split_evidence/seen_unseen_tables_r27.json $R27C_EV/r27_band_split_evidence/rldx_e1b_r27.json | cut -c1-8
cd $R27C_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r27c_split_all_bands tests.test_r27_band_split \
  tests.test_r27b_split_controls tests.test_xr1_seen_unseen_tables tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- no `STOP` line prints;
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes are, in order: `53599170`, `8f9dd45c` (R21), `431b93c8`, `52f8b5d4` (R27);
- the tests end with `OK`.

```bash
cd $R27C_REPO
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
  --r27c --r27c-r21 $R27C_EV/r21_native_bands_evidence/seen_unseen_tables_r21.json \
  --r27c-r27 $R27C_EV/r27_band_split_evidence/seen_unseen_tables_r27.json \
  --output $R27C_ROOT/seen_unseen_tables_r27c.json' > $R27C_LOG/r27c_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r27c --r27c-r21 $R27C_EV/r21_native_bands_evidence/rldx_e1b_r21.json \
  --r27c-r27 $R27C_EV/r27_band_split_evidence/rldx_e1b_r27.json \
  --output $R27C_ROOT/rldx_e1b_r27c.json' > $R27C_LOG/r27c_rldx.log 2>&1 &
echo "started; R27C_ROOT=$R27C_ROOT"
```

**Runtime:** 21 evaluations per policy (the fixed split and 20 assignments), each with three heads on both sides and
no bootstrap or curves. R27's 21 single-head evaluations, with curves, took under ten minutes, so expect under half an
hour.

**Progress:** `grep "R27c split" $R27C_LOG/r27c_*.log | tail -n 2`. When both logs end:

```bash
grep -E "R27c OK|R27c FAILED|REPRODUCTION FAILED|Traceback" $R27C_LOG/r27c_*.log
sed -n '/^R27c XR-1/,$p' $R27C_LOG/r27c_xr1.log; sed -n '/^R27c RLDX-1/,$p' $R27C_LOG/r27c_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R27C_ROOT/evidence && cp $R27C_ROOT/*.json $R27C_LOG/*.log $R27C_ROOT/evidence/ && ls $R27C_ROOT/evidence
```

Both logs must end with `R27c OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r27c_split_all_bands_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r27c_split_all_bands_evidence/
```

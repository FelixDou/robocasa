# R27: does Hide-and-Seek's own-band result hold under the published 30/70 split?

Branch `claude/r22-fixed-paths`. Written 2026-11-24, before the run. A sensitivity check: no hypothesis is
added, and the reading below is fixed now.

## Why

Shinoda sensei's final directions (v5) make this the paper's central claim: Hide-and-Seek, under its original
alarm rule, warns earlier than the elapsed-time timer on unseen tasks. They ask us to say whether our rule differs
mathematically from the published one.

We compared Appendix B.2 of Hide-and-Seek (arXiv 2605.30834) with SAFE's released routine
(`failure_prob/utils/conformal/functional_predictor.py`) and our copy (`robocasa/recovery/safe/conformal.py`).

- **The same:**
  - the center: the mean of the reference successes;
  - the modulation: the largest absolute residual over the reference trajectories kept at the
    ⌈(n+1)(1−α)⌉-th smallest maximum residual;
  - the one-sided nonconformity score;
  - the width: the (1−α) empirical quantile, with no finite-sample correction;
  - padding by the last score;
  - the alarm at the first score ≥ the band.
- **Different:** Hide-and-Seek fits the mean and modulation on 30% of the calibration successes and the width on
  the other 70%, without saying how rollouts are assigned. R19 to R25 split 50/50 (per task, rollouts sorted by
  ID alternate).
- **Not a difference for single models:** we read normalized scores, but on unseen tasks the normalization is one
  shift and scale per fold, which leaves a single model's band alarms unchanged (unit test). The three-seed
  ensemble averages normalized scores, an adaptation the paper already states.
- **Not checkable:** Hide-and-Seek's repository holds only a README ("Code coming soon!"), so the published
  equations are the reference.

## What R27 computes (unseen tasks, both policies)

- **Assignments.** 20 random 30/70 splits of the calibration successes, stratified by task (XR-1: 3 + 7 per task;
  RLDX-1: 6 + 14), from `numpy.random.default_rng(27 + j)`, j = 0…19, with rollouts sorted by ID first.
- **Unchanged:** scores, folds, SAFE's routine, E, and the test rollouts.
- **Per assignment:**
  1. **Curve (Fig. 2):** each of the three models' band over R21's calibration targets, averaged over models,
     against E's curve (R18). We record the FPR above which the band stays above E, on their observed overlap up
     to 15%, with linear interpolation between sampled points.
  2. **Matched gain (Table II):** the three-seed ensemble's band at α = 0.05 against E matched to its test false
     alarms (Δℓ̄), with its false alarms and FPR.
  3. **First-step control (Table II):** the full ensemble band minus the band on the first-step retraining,
     matched to the full band's false alarms, with the control at its best-lead target within that count
     (R25's rule, which favors the control).
- **Saved for the paper's opening figure:** each failed test rollout's alarm under the 50/50 ensemble band and
  under E at α = 0.05 (`failure_alarms`), so that an illustrated rollout shows the real alarms of the primary
  comparison.
- **Reported:** the 50/50 values, and the median, minimum and maximum over the 20 assignments. No CI: the
  spread over assignments is what this check measures.
- **Code:**
  - `scripts/xr1_lead_time/r27_band_split.py`;
  - `r21_native_bands.Design(..., split=)`, whose default is unchanged;
  - `seen_unseen_tables.py run --r27` and `rldx_test_table.py run --r27`;
  - tests: `tests/test_r27_band_split.py`, and the R21 and R25 end-to-end tests.

## Gates

- **K0:** the 50/50 ensemble equals R21's (false alarms, Δℓ̄; ≤ 1e-12).
- **K1:** the 50/50 single-model curves equal R21's at every target (FPR, lead; ≤ 1e-12).
- **K2:** the 50/50 first-step comparison equals R25's (Δ, matched false alarms, α; ≤ 1e-12).
- **K3:** every assignment is a partition of the calibration successes, with 30% of each task in the reference
  half.
- **K4:** the saved failure alarms give the ensemble's and E's mean lead (≤ 1e-12).

Any failure prints `R27 FAILED`, and no R27 number is used.

## Reading (fixed now)

On each policy, per quantity, the result **holds under the published split** if, in at least 18 of the 20
assignments:

- **Curve:** the averaged band is above E at every observed FPR above 7%.
- **Matched gain:** Δℓ̄ over matched E is positive.
- **First-step control:** the full band's Δ over the first-step band is positive.

Then:

- **All hold on both policies:** the paper keeps its 50/50 numbers as the main result, states the split difference
  in Sec. IV-A, and adds one sentence with the 30/70 range.
- **Any does not hold:** that claim is reworded as sensitive to the calibration split, in the abstract, Sec. VI-B and
  the conclusion, and the author and Shinoda sensei are told before the draft is sent.

## Expectations (written before running)

- The 50/50 values reproduce R21 and R25:
  - XR-1: 29 false alarms, Δℓ̄ 0.147, first-step Δ 0.273;
  - RLDX-1: 40 false alarms, Δℓ̄ 0.143, first-step Δ 0.146.
- No expectation is fixed for the 30/70 values. A smaller reference half makes the modulation noisier, so the
  realized FPR and its spread may change.

## Runbook (CPU; both runs in parallel)

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=3:00:00
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
export R27_SHA=<full SHA of the pushed commit containing this plan>
export R27_REPO=$PROJECT_FS/robocasa_r27_${R27_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R27_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R27_REPO $R27_SHA
test "$(git -C $R27_REPO rev-parse HEAD)" = "$R27_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R27_EV=$R27_REPO/reports
export R27_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r27_band_split_$(date +%Y%m%d_%H%M%S)
export R27_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R27_ROOT)
mkdir -p $R27_ROOT $R27_LOG
export XR1_FOLDS=$R27_REPO/scripts/xr1_lead_time/cross_task_folds.json
echo "FOLDS=$FOLDS"
sha256sum $R27_EV/r18_lead_fpr_evidence/seen_unseen_tables_r18.json $R27_EV/r18_lead_fpr_evidence/rldx_e1b_r18.json \
  $R27_EV/r21_native_bands_evidence/seen_unseen_tables_r21.json $R27_EV/r21_native_bands_evidence/rldx_e1b_r21.json \
  $R27_EV/r25_band_ablation_evidence/seen_unseen_tables_r25.json $R27_EV/r25_band_ablation_evidence/rldx_e1b_r25.json | cut -c1-8
cd $R27_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r27_band_split tests.test_r25_band_ablation \
  tests.test_r21_native_bands tests.test_xr1_seen_unseen_tables tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- no `STOP` line prints;
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes are, in order: `624ee65a`, `2162a01d` (R18), `53599170`, `8f9dd45c` (R21), `a887c7ff`, `5b0cdb44` (R25);
- the tests end with `OK`.

```bash
cd $R27_REPO
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
  --r27 --r27-r18 $R27_EV/r18_lead_fpr_evidence/seen_unseen_tables_r18.json \
  --r27-r21 $R27_EV/r21_native_bands_evidence/seen_unseen_tables_r21.json \
  --r27-r25 $R27_EV/r25_band_ablation_evidence/seen_unseen_tables_r25.json \
  --output $R27_ROOT/seen_unseen_tables_r27.json' > $R27_LOG/r27_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r27 --r27-r18 $R27_EV/r18_lead_fpr_evidence/rldx_e1b_r18.json \
  --r27-r21 $R27_EV/r21_native_bands_evidence/rldx_e1b_r21.json \
  --r27-r25 $R27_EV/r25_band_ablation_evidence/rldx_e1b_r25.json \
  --output $R27_ROOT/rldx_e1b_r27.json' > $R27_LOG/r27_rldx.log 2>&1 &
echo "started; R27_ROOT=$R27_ROOT"
```

**Runtime:** 21 evaluations per policy (the 50/50 reference and 20 assignments), with no bootstrap. Expect under
an hour.

**Progress:** `grep "R27 split" $R27_LOG/r27_*.log | tail -n 2`. When both logs end:

```bash
grep -E "R27 OK|R27 FAILED|REPRODUCTION FAILED|Traceback" $R27_LOG/r27_*.log
sed -n '/^R27 XR-1/,$p' $R27_LOG/r27_xr1.log; sed -n '/^R27 RLDX-1/,$p' $R27_LOG/r27_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R27_ROOT/evidence && cp $R27_ROOT/*.json $R27_LOG/*.log $R27_ROOT/evidence/ && ls $R27_ROOT/evidence
```

Both logs must end with `R27 OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r27_band_split_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r27_band_split_evidence/
```

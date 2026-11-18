# R21b. Why R21's band CIs are lopsided

Written 2026-11-18, after R21 and before any R21b run. Branch `claude/r21-native-bands`. Exploratory
and diagnostic. Asked by the author before the paper adopts the bands as the main alarm rule.

## Why

- **The pattern:** R21's full-pipeline bootstrap puts several point estimates near the top of their 95%
  intervals. Examples, Hide-and-Seek's band OR timer, matched gains:
  - XR-1 over G: 0.013 [−0.153, 0.076];
  - RLDX-1 over E: 0.096 [−0.164, 0.187].
- **The consequence:** most resampled replicates do worse than the real data, and every Hide-and-Seek
  CI includes zero. R19's constant-threshold CIs, with the same kind of draws, did not look like this.
- **Two readings:**
  - **Real:** one band is pooled over tasks with limits from 29 to 272 inferences, so it may be
    fragile when the mix of tasks changes. That would be a property of the original rule, to report.
  - **Artifact:** a band recalibrated on a bootstrap sample sees only about 63% distinct successes,
    some of them duplicated. If that alone distorts the band, the full-pipeline CIs understate it,
    and a CI conditional on the real calibration set is the fair one.
- **What R21 lacks:** it saved only the percentiles, not the replicate values, so neither reading can
  be checked from its output.

## Design (fixed now)

Code: `scripts/xr1_lead_time/r21_native_bands.py` (`diagnose`, `Design.draw_test`,
`Design.draw_calibration`), called with `--r21b` by `seen_unseen_tables.py` and `rldx_test_table.py`,
after R21 itself. The inputs are R21's. No new rollout, model or score is used.

- **Scope:** R21's ensembles (seed-averaged score). It covers every head, both sides and every
  comparator, for the band alone and for the band at α/2 OR the timer at α/2.
- **Six resampling variants,** each with 1,000 replicates and seed 21:

  | Variant | Calibration successes | Test rollouts |
  |---|---|---|
  | `full` | R21's draws, replayed: tasks with replacement, then within task and half with replacement | the same tasks, then within class |
  | `test` | as they are | tasks with replacement, then within task and class |
  | `calibration` | as in `full` | as they are |
  | `calibration_rollouts` | every task once; within task and half, with replacement | as they are |
  | `calibration_distinct` | every task once; per task and half, m without replacement, with m = round(n(1 − (1 − 1/n)^n)), the expected number of distinct rollouts in a bootstrap sample of n | as they are |
  | `calibration_tasks` | tasks with replacement; each drawn task's successes as they are | as they are |

  Every band, timer margin and matched timer is refitted on each replicate, as in R21.
- **Saved per replicate:**
  - gain and matched gain of the band alone and of the fused rule;
  - the false alarms of the band alone, the fused rule, the timer at α, and each matched timer.

  They go to a compressed `.npz` file next to the JSON.
- **Summary per variant:**
  - the point estimate;
  - the replicate mean, median, 2.5th and 97.5th percentiles, and share below zero;
  - the Spearman correlation, across replicates, between a rule's false alarms and its matched gain.

## Gates

- **D0:** variant `full` reproduces R21's bootstrap from its output (`--r21b-reference`): every point
  estimate and every 95% CI within 1e-12, with the same number of replicates.
- **R21's gates** all pass in the same run, with `--r21-boot 0`, since `full` replays the bootstrap.
- **The scripts' own gates,** and the inputs are unchanged at the end.

## Interpretation (fixed now)

These rules apply to Hide-and-Seek's matched gains (band alone and band OR timer), per policy, side
and comparator.

- **Shift:** a variant shifts when its point estimate exceeds its replicate median by more than 0.02.
- **Where the shift comes from:**
  - If `full` shifts, `test` does not, and `calibration` does: the lopsided CIs come from recalibrating
    the band.
  - If `calibration_rollouts` and `calibration_distinct` both shift: the band depends on having all the
    distinct calibration successes. Resampling removes about a third of them, so this is an artifact,
    and the CI to report is `test`'s: conditional on the real calibration set, like Table III's.
  - If only `calibration_rollouts` shifts: duplicated successes distort the band. This is also an
    artifact, and `test`'s CI is reported.
  - If `calibration_tasks` shifts and the two rollout-level variants do not: the pooled band is
    sensitive to the mix of tasks. That is a real property of the original rule. `full`'s CI stays,
    and the paper says that the band is unstable across task mixes.
  - If `test` shifts too: the skew is in the test data, not in the band, and `full`'s CI stays.
- **Supporting evidence:** a negative correlation between the band's false alarms and its matched
  gain in `calibration` supports the artifact readings, since an over-alarming band would be a
  distorted one.
- **Mixed outcomes** (none of the patterns above) are reported as they are, with `full`'s CI kept.

## Expectations (written before running)

- **D0:** holds exactly.
- **Shifts:** `full` and `calibration` shift for Hide-and-Seek on unseen tasks (the most lopsided
  intervals); `test` does not.
- **Cause:** fewer distinct calibration successes (`calibration_distinct` shifts). On unseen tasks,
  each held-out fold's band calibrates on about 280 successes on XR-1 and about 300 on RLDX-1, so
  losing a third of them changes the `tfunc` envelope.

## Runbook

CPU only. Six variants of about 5 minutes each, plus R21 without its bootstrap: well under two hours
per run. The two runs go in parallel.

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=3:00:00
```

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export STORAGE_BS=/gs/bs/tga-shinoda/felid
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r11_rows_20261030_220636/paths.env
source $STORAGE_BS/robocasa_logs/eval/rldx1_unseen_20261029_135501/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r14_hk_free_20261106_191531/paths.env
source $STORAGE_BS/robocasa_checkpoints/safe/rldx1_e1b_test_table_20261113_145031/session.env
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R21B_SHA=<full SHA of the pushed commit containing this plan>
export R21B_REPO=$PROJECT_FS/robocasa_r21b_${R21B_SHA:0:8}
export R21B_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r21b_band_ci_diagnostic_$(date +%Y%m%d_%H%M%S)
export R21B_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R21B_ROOT)
export R21_RUN=$STORAGE_BS/robocasa_checkpoints/safe/r21_native_bands_20261118_172324
export R19_XR1=$(ls $STORAGE_BS/robocasa_checkpoints/safe/r19_triage_b_*/seen_unseen_tables_r19.json | tail -n 1)
export R19_RLDX=$(ls $STORAGE_BS/robocasa_checkpoints/safe/r19_triage_b_*/rldx_e1b_r19.json | tail -n 1)
mkdir -p $R21B_ROOT $R21B_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r21-native-bands
test -e $R21B_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R21B_REPO $R21B_SHA
test "$(git -C $R21B_REPO rev-parse HEAD)" = "$R21B_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export XR1_FOLDS=$R21B_REPO/scripts/xr1_lead_time/cross_task_folds.json
export R12_LISTB=$R21B_REPO/reports/safe_robocasa365_research_plan_20260823/evidence/xr1_list_b_20261030_212017/list_b.json
cmp $XR1_FOLDS $R14_REPO/scripts/xr1_lead_time/cross_task_folds.json && echo "XR-1 FOLDS SAME AS R14"
sha256sum $R19_XR1 $R19_RLDX $R12_LISTB $R21_RUN/seen_unseen_tables_r21.json $R21_RUN/rldx_e1b_r21.json | cut -c1-16
cd $R21B_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r21_native_bands tests.test_xr1_seen_unseen_tables \
  tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- `HEAD OK` and `XR-1 FOLDS SAME AS R14` print.
- The hashes are, in order: `d4d8176f…`, `81d0beec…`, `0e755875…`, `53599170…`, `8f9dd45c…`.
- The tests end with `OK`.

Then start both runs in the background:

```bash
cd $R21B_REPO
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/seen_unseen_tables.py run \
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
  --r21 --r21-reference $R19_XR1 --r21-r12-reference $R12_LISTB --r21-boot 0 \
  --r21b --r21b-reference $R21_RUN/seen_unseen_tables_r21.json --r21b-boot 1000 \
  --output $R21B_ROOT/seen_unseen_tables_r21b.json' > $R21B_LOG/r21b_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r21 --r21-reference $R19_RLDX --r21-boot 0 \
  --r21b --r21b-reference $R21_RUN/rldx_e1b_r21.json --r21b-boot 1000 \
  --output $R21B_ROOT/rldx_e1b_r21b.json' > $R21B_LOG/r21b_rldx.log 2>&1 &
echo "started; R21B_ROOT=$R21B_ROOT"
```

**Progress:** `tail -n 2 $R21B_LOG/r21b_*.log`. Each variant prints every 250 replicates.

When both logs end:

```bash
grep -E "R21 OK|R21b OK|FAILED|STOP|Traceback" $R21B_LOG/r21b_*.log
sed -n '/^R21b /,$p' $R21B_LOG/r21b_xr1.log; sed -n '/^R21b /,$p' $R21B_LOG/r21b_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R21B_ROOT/evidence && cp $R21B_ROOT/*.json $R21B_ROOT/*.npz $R21B_LOG/*.log $R21B_ROOT/evidence/ && ls $R21B_ROOT/evidence
```

Both logs must end with `R21b OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r21b_band_ci_diagnostic_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r21-native-bands/reports/r21b_band_ci_diagnostic_evidence/
```

## Result

(to be written after the run)

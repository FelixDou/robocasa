# R24: Table III's missing intervals, and H3's gain by task type in seconds

Branch `claude/r22-fixed-paths`. Written 2026-11-19, before the run. Exploratory: no hypothesis
changes, and every point estimate already appears in the paper.

## Why

A review of draft v4 asks Table III to show, for every ensemble, its realized false alarms, the
matched timer's, and a 95% CI. It also asks for one current-protocol result in seconds, split by
atomic and composite tasks:
- XR-1 has 10 atomic tasks of 38, RLDX-1 14 of 20.
- RLDX-1's larger pooled gain is hard to read without that split.

What exists already:
- **Intervals.** R22 and R23 give intervals for nine of Table III's cells. The cells filled from
  R22's Table II seed averages have none:
  - XR-1 seen + P: SAFE-LSTM;
  - XR-1 unseen + E: Mahalanobis, first step, zero input;
  - RLDX-1 seen + P: SAFE-MLP, SAFE-LSTM.
- **Seconds.** XR-1's split exists: R22's `per_type`, Hide-and-Seek + P, Δℓ̄ in fractions of
  H_k and in seconds. RLDX-1 has only E1b's same-target split, which is not the paper's statistic.

## What R24 computes

- **XR-1:** R22's XR-1 run again, with two additions. SAFE-LSTM joins the seen-task ensemble loop
  (`ensemble_row`: task-then-failure 95% CIs, as for the other seen rows). Mahalanobis, first step
  and zero input join the unseen full-pipeline bootstrap (`boot_unseen`: 1,000 replicates, seed 22,
  paths frozen). The replicate draws depend only on the seed, so the added rows leave R22's
  intervals unchanged.
- **RLDX-1:** on R23's run, three additions:
  - SAFE-MLP and SAFE-LSTM + P as ensembles on R22's fixed paths (chosen on the recorded validation
    rollouts), with the same CIs.
  - Hide-and-Seek + P on H3's registered path (R23's row), with Δℓ̄ by task type, in fractions of
    H_k and in seconds. RLDX-1 runs 8 control steps per inference at 20 Hz, so 0.4 s per inference
    (XR-1: 0.8 s).
  - Matched timer false alarms for every row.

## Gates

- **XR-1, R0:** every entry of R22's output is unchanged: Table II, ensembles, per type, the
  bootstrap, nulls, the budget timer and the teaser (max |diff| ≤ 1e-12). The added rows only add
  keys.
- **XR-1, R1:** each added row's false alarms, gain and Δℓ̄ equal R22's Table II seed average for
  the same detector and path (≤ 1e-12). For the seen row, the matched timer's false alarms must
  also be equal.
- **RLDX-1, K1:** SAFE-MLP's and SAFE-LSTM's false alarms, matched timer false alarms, gain and Δℓ̄
  equal R22's seed average (≤ 1e-12).
- **RLDX-1, K2:** H3's row equals R23's (≤ 1e-12).
- **Both:** R22's and R20's own gates pass again.

Any failure prints `R24 FAILED`, and no R24 number is used.

## Expectations (written before running)

The points equal Table III's:
- XR-1 seen SAFE-LSTM: 20 false alarms, 0.037.
- XR-1 unseen, all at 19 false alarms: Mahalanobis 21, −0.013; first step and zero input −0.006.
- RLDX-1 seen: SAFE-MLP 27, 0.000; SAFE-LSTM 28, 0.024.

The intervals are new. The disabled rows (score unused on the validation-chosen path) have intervals
centred near their small negative points.

For H3 by type, the gain should come from atomic tasks: E1b's same-target split was atomic 0.092,
composite −0.036.

## Reporting (fixed now)

- **Table III:** every cell gets its 95% CI and the matched timer's false alarms.
- **Results:** one sentence or a small table on Hide-and-Seek + P by task type in seconds, on both
  VLAs. XR-1's values come from R22's `per_type`, RLDX-1's from R24.
- **RLDX-1 paragraph:** its "composite: −0.036" (E1b's same-target split) is replaced by R24's Δℓ̄
  split.

## Runbook (CPU; both runs in parallel)

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=4:00:00
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
export SAFE_PY=${SAFE_PY:-$STORAGE_BS/envs/vla_safe/bin/python}
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $(ls -d $STORAGE_BS/robocasa_logs/eval/r22a_validation_scores_* | tail -n 1)/paths.env
export R24_SHA=<full SHA of the pushed commit containing this plan>
export R24_REPO=$PROJECT_FS/robocasa_r24_${R24_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R24_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R24_REPO $R24_SHA
test "$(git -C $R24_REPO rev-parse HEAD)" = "$R24_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R22B=$(ls -d $STORAGE_BS/robocasa_checkpoints/safe/r22b_fixed_paths_* | tail -n 1)
export R23_JSON=$(ls $STORAGE_BS/robocasa_checkpoints/safe/r23_h3_h1_rule_*/rldx_e1b_r23.json | tail -n 1)
export R24_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r24_table3_ci_seconds_$(date +%Y%m%d_%H%M%S)
export R24_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R24_ROOT)
mkdir -p $R24_ROOT $R24_LOG
export XR1_FOLDS=$R24_REPO/scripts/xr1_lead_time/cross_task_folds.json
export R17_JSON=$(ls $STORAGE_BS/robocasa_checkpoints/safe/rldx1_r17_elapsed_*/rldx_e1b_r17.json | tail -n 1)
export R8_JSON=$R24_REPO/reports/safe_robocasa365_research_plan_20260823/evidence/xr1_fixed_path_20261029_164116/fixed_path.json
echo "R22B=$R22B R23_JSON=$R23_JSON FOLDS=$FOLDS"
sha256sum $R22B/seen_unseen_tables_r22.json $R22B/rldx_e1b_r22.json $R23_JSON $R14_ROOT/seen_unseen_tables_r14.json \
  $R17_JSON $R8_JSON $E1B_ROOT/e1b_result.json | cut -c1-16
cd $R24_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r22_fixed_paths tests.test_xr1_seen_unseen_tables \
  tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes start with, in order: `76280d1b` (R22 XR-1), `fc776307` (R22 RLDX-1), `48ecc74f` (R23),
  `1578f312` (R14), `f531fd5a` (R17), `5b4c7b6a` (R8) and `bd0aab11` (E1b);
- the tests end with `OK`.

```bash
cd $R24_REPO
OODV=()
for S in 0 1 2; do OODV+=(--r22-ood-val $S $OOD_ROOT/seed$S/calibrate/calibration_predictions.jsonl $OOD_ROOT/seed$S/evaluate/event_predictions.jsonl); done
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
  --r22 --r22-scores-dir $R22A_ROOT \
  --r22-val-calibration $XR1_PS_INPUTS/calibration_predictions.jsonl --r22-val-test $XR1_PS_INPUTS/event_predictions.jsonl \
  --r22-val-ids $AMEND2_ROOT/august_calibration_ids.json $PSEED_ROOT/august_test_balanced_ids.json \
  --r22-val-hs-scores $AMEND2_ROOT/scores/august_calibration_predictions.jsonl $AMEND2_ROOT/scores/august_test_predictions.jsonl \
  '"${OODV[*]}"' --r22-reference $R14_ROOT/seen_unseen_tables_r14.json --r22-r8 $R8_JSON \
  --r22-boot 1000 --r22-null-draws 100 --r24 --r24-reference $R22B/seen_unseen_tables_r22.json \
  --output $R24_ROOT/seen_unseen_tables_r24.json' > $R24_LOG/r24_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r24 --r24-reference $R22B/rldx_e1b_r22.json --r24-r23 $R23_JSON \
  --output $R24_ROOT/rldx_e1b_r24.json' > $R24_LOG/r24_rldx.log 2>&1 &
echo "started; R24_ROOT=$R24_ROOT"
```

The RLDX-1 run takes minutes, the XR-1 run up to a few hours (R22's run plus three detectors in the
bootstrap). **Progress:** `tail -n 3 $R24_LOG/r24_*.log`. When both logs end:

```bash
grep -E "R24 OK|R24 FAILED|R22 OK|R22 FAILED|Traceback" $R24_LOG/r24_*.log
sed -n '/^R24 /,$p' $R24_LOG/r24_xr1.log; sed -n '/^R24 /,$p' $R24_LOG/r24_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R24_ROOT/evidence && cp $R24_ROOT/*.json $R24_LOG/*.log $R24_ROOT/evidence/ && ls $R24_ROOT/evidence
```

Both logs must end with `R24 OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r24_table3_ci_seconds_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r24_table3_ci_seconds_evidence/
```

## Result (2026-11-19, run `r24_table3_ci_seconds_20261119_144619`, from `4fbfbeef`): `R24 OK`

Both runs ended with `R24 OK`, the XR-1 run also with `R22 OK`, and `INPUTS UNCHANGED`. Evidence is
in `reports/r24_table3_ci_seconds_evidence/`:
- `seen_unseen_tables_r24.json`, sha256 `ed942e22…`;
- `rldx_e1b_r24.json`, `7c9e0d9c…`;
- both logs, gitignored and kept locally.

The recorded inputs are R22's `76280d1b…` (XR-1) and `fc776307…` (RLDX-1), and R23's `48ecc74f…`.

**Gates.**
- **XR-1, R0:** R22's output is unchanged (max |diff| 0).
- **XR-1, R1:** all four added rows OK (max |diff| 0).
- **RLDX-1, K1:** both SAFE rows OK (max |diff| 0).
- **RLDX-1, K2:** H3 equals R23 (max |diff| 0).

**Table III's missing intervals.** The points are as expected.

| Cell | FA (monitor / matched timer) | Δℓ̄ | 95% CI |
| --- | --- | --- | --- |
| XR-1 seen + P, SAFE-LSTM | 20 / 20 | 0.037 | [0.008, 0.073] |
| XR-1 unseen + E, Mahalanobis | 21 / 20 | −0.013 | [−0.099, 0.014] |
| XR-1 unseen + E, first step | 19 / 19 | −0.006 | [−0.057, 0.011] |
| XR-1 unseen + E, zero input | 19 / 19 | −0.006 | [−0.057, 0.011] |
| RLDX-1 seen + P, SAFE-MLP (score unused) | 27 / 27 | 0.000 | [0.000, 0.000] |
| RLDX-1 seen + P, SAFE-LSTM | 28 / 28 | 0.024 | [−0.001, 0.056] |

One result is new: **SAFE-LSTM's seen-task gain on XR-1 has a 95% CI above zero**, like SAFE-MLP's
and Mahalanobis's. It does not repeat on RLDX-1, where the CI touches zero. On unseen tasks, first
step and zero input equal SAFE-MLP's row: all three paths leave the score unused, or it never fires
first.

**H3 by task type (RLDX-1, Hide-and-Seek + P, 0.4 s per inference).**

| Tasks | Failures | P's lead | Δℓ̄ | Δℓ̄ (s) |
| --- | --- | --- | --- | --- |
| Atomic (14) | 280 | 10.9 s | 0.153 | 5.3 |
| Composite (6) | 120 | 8.5 s | 0.012 | 0.4 |
| All (20) | 400 | 10.2 s | 0.111 | 3.9 |

Per task type, the gain is about the same on both VLAs (XR-1, R22: atomic 0.141 = 5.1 s,
composite 0.026 = −0.1 s). RLDX-1's pooled gain is larger because 14 of its 20 tasks are atomic,
against 10 of XR-1's 38. At XR-1's mix of 26% atomic tasks, RLDX-1's per-type gains would pool
to about 0.049.

# R27d: Fig. 2's own-band curves on R27's random 30/70 assignments

Status: descriptive, written before the run (2026-11-24). Follows R27c (`docs/r27c_split_all_bands_plan.md`).

## Why

After R27c, Table II and the text report every own band (SAFE-MLP, SAFE-LSTM, Hide-and-Seek) over R27's 20 random
30/70 assignments of the calibration successes, and Fig. 1 draws Hide-and-Seek's unseen-task curve as the median over
those assignments (R27b). Fig. 2 is the last place that uses the fixed 50/50 split of R19 to R25: its own-band curves
(dotted, both policies, seen and unseen tasks) are R21's. Sec. IV-A therefore needs "Fig. 2 uses one fixed 50/50
split", and Fig. 2's Hide-and-Seek curve on unseen tasks differs from Fig. 1's. R27d computes Fig. 2's own-band curves
on the same 20 assignments, so that the paper uses one split protocol throughout.

## What R27d computes (both policies)

- **Assignments.** R27's 20 random 30/70 splits (seeds 27 to 46, `r27_band_split.split_30_70`), stratified by task; on
  seen tasks the same split per task, hence the same assignment (gate D2).
- **Curves.** For each head (SAFE-MLP: `safe_mlp` on XR-1, `safe_indep` on RLDX-1; SAFE-LSTM; Hide-and-Seek), side
  (seen, unseen) and assignment: each single model's band at every calibration target (`r21_native_bands.curve`, as
  Fig. 2 draws now), and their mean at each target (`r27_band_split.mean_curve`).
- **Summary, as Fig. 1 does it** (`analysis/unseen_lead_figure.published_curve`): each assignment's mean curve
  interpolated on a grid of 301 FPRs from 0 to 15% within its observed range; the median, minimum and maximum where at
  least 15 of the 20 assignments reach the FPR; the target point, the median FPR and lead over the assignments of the
  band calibrated for 5%.
- **Gaps to the timers** (R18's curves): the median curve against P on seen tasks and against E and G on unseen tasks,
  on their overlap up to 15% FPR: smallest and largest gap, and the FPR above which the curve stays above the timer
  (`r27_band_split.versus_timer`). The same gaps for the fixed 50/50 split's mean curve (Fig. 2 as drawn now).
- **Unchanged:** scores, folds, SAFE's routine, the calibration targets, the timers, the test rollouts. No bootstrap.

## Gates

- **D0:** at R19's fixed 50/50 split, every head's single-model curves on both sides equal R21's saved curves
  (`r21.curves.{head}.{side}.seed_{s}`: target, FPR, mean lead within 1e-12; false alarms equal). A missing reference
  fails the gate.
- **D1:** Hide-and-Seek's unseen-task mean curve on each 30/70 assignment equals R27b's saved curve
  (`r27b.published[j].mean_curve`, Fig. 1's data; 1e-12).
- **D2:** every assignment is a 30/70 partition of the calibration successes on both sides, with the same reference
  rollouts on both.

Any failure prints `R27d FAILED`, and no R27d curve is used.

## Reading (fixed now)

"Near" a timer's curve means within 0.02 of its lead (the fixed-split curves the text describes now: SAFE-MLP against E
within about ±0.015, the SAFE bands against P at most +0.018 above it).

1. **Fig. 2.** Each own-band curve becomes the median over the 20 assignments, drawn where at least 15 reach the FPR,
   with its dot at the median target point, as in Fig. 1. The caption's "Own bands: one fixed 50/50 calibration split"
   becomes "Own bands: median over 20 random 30/70 calibration splits". Sec. IV-A drops "Fig. 2 uses one fixed 50/50
   split". The fixed 50/50 split then appears only in Sec. VI-B's favorable-draw clause.
2. **Sec. VI-B's description of Fig. 2** ("on seen tasks the monitors' curves stay near or below P's, except
   Hide-and-Seek's; on unseen tasks SAFE-MLP's curve lies on E's") stays for the own bands if, on both policies, the
   median curves of SAFE-MLP and SAFE-LSTM are never more than 0.02 above P's on seen tasks, Hide-and-Seek's is more
   than 0.02 above P's somewhere, and SAFE-MLP's unseen-task curve stays within 0.02 of E's. Otherwise the sentence
   states what the median curves show (which band, against which timer, from which FPR). The constant-threshold
   curves do not change.
3. **Fig. 1 and Sec. VI-B's first paragraph** are unchanged: by D1, Fig. 2's Hide-and-Seek curve on unseen tasks is
   Fig. 1's median curve.

## Code

- `scripts/xr1_lead_time/r27d_split_curves.py`;
- hooks: `seen_unseen_tables.py --r27d --r27d-r18 --r27d-r21 --r27d-r27b [--r27d-splits]` (XR-1; needs
  `--safe-lstm-fold-dirs`) and `rldx_test_table.py --r27d --r27d-r18 --r27d-r21 --r27d-r27b [--r27d-splits]` (RLDX-1;
  keeps each seed's normalized raw score of the three heads, as `--r21` does);
- `tests/test_r27d_split_curves.py` (synthetic folds: the mean curve is R27's, Fig. 1's median and target rules, the
  gates pass on consistent references and fail on perturbed or missing ones, the gap summary);
- after the run, `analysis/lead_fpr_figure.py` (paper worktree) reads the R27d outputs for the own bands.

## Runbook (CPU; both runs in parallel)

Read `docs/cluster_experiment_runbook.md` first.

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
for v in E1_OUT XR1_RUNTIME_BUNDLE RS_ROOT XT_ROOT R6_ROOT XR1_BASELINE_ROOT AMEND2_ROOT OOD_ROOT PSEED_ROOT R11_ROOT \
  RLDX_ROOT RLDXHS_ROOT R16_ROOT E1B_ROOT E1_FREEZE R7_ROOT R7S RLDXPS_ROOT FOLDS; do
  test -n "${!v}" || echo "STOP: $v is unset"; done
export SAFE_PY=${SAFE_PY:-$STORAGE_BS/envs/vla_safe/bin/python}
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R27D_SHA=<full SHA of the pushed commit containing this plan>
export R27D_REPO=$PROJECT_FS/robocasa_r27d_${R27D_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R27D_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R27D_REPO $R27D_SHA
test "$(git -C $R27D_REPO rev-parse HEAD)" = "$R27D_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R27D_EV=$R27D_REPO/reports
export R27D_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r27d_split_curves_$(date +%Y%m%d_%H%M%S)
export R27D_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R27D_ROOT)
mkdir -p $R27D_ROOT $R27D_LOG
export XR1_FOLDS=$R27D_REPO/scripts/xr1_lead_time/cross_task_folds.json
echo "FOLDS=$FOLDS"
sha256sum $R27D_EV/r18_lead_fpr_evidence/seen_unseen_tables_r18.json $R27D_EV/r18_lead_fpr_evidence/rldx_e1b_r18.json \
  $R27D_EV/r21_native_bands_evidence/seen_unseen_tables_r21.json $R27D_EV/r21_native_bands_evidence/rldx_e1b_r21.json \
  $R27D_EV/r27b_split_controls_evidence/seen_unseen_tables_r27b.json $R27D_EV/r27b_split_controls_evidence/rldx_e1b_r27b.json | cut -c1-8
cd $R27D_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r27d_split_curves tests.test_r27c_split_all_bands tests.test_r27_band_split \
  tests.test_r27b_split_controls tests.test_xr1_seen_unseen_tables tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- no `STOP` line prints;
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes are, in order: `624ee65a`, `2162a01d` (R18), `53599170`, `8f9dd45c` (R21), `c0f430d1`, `31d82f21` (R27b);
- the tests end with `OK`.

```bash
cd $R27D_REPO
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
  --r27d --r27d-r18 $R27D_EV/r18_lead_fpr_evidence/seen_unseen_tables_r18.json \
  --r27d-r21 $R27D_EV/r21_native_bands_evidence/seen_unseen_tables_r21.json \
  --r27d-r27b $R27D_EV/r27b_split_controls_evidence/seen_unseen_tables_r27b.json \
  --output $R27D_ROOT/seen_unseen_tables_r27d.json' > $R27D_LOG/r27d_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r27d --r27d-r18 $R27D_EV/r18_lead_fpr_evidence/rldx_e1b_r18.json \
  --r27d-r21 $R27D_EV/r21_native_bands_evidence/rldx_e1b_r21.json \
  --r27d-r27b $R27D_EV/r27b_split_controls_evidence/rldx_e1b_r27b.json \
  --output $R27D_ROOT/rldx_e1b_r27d.json' > $R27D_LOG/r27d_rldx.log 2>&1 &
echo "started; R27D_ROOT=$R27D_ROOT"
```

**Runtime:** 21 designs per policy (the fixed split and 20 assignments), each with three heads, two sides and three
single models, so 378 curves per policy and no bootstrap: about three times R27b's curve work. Expect one to two
hours; the allocation allows four.

**Progress:** `grep "R27d split" $R27D_LOG/r27d_*.log | tail -n 2`. When both logs end:

```bash
grep -E "R27d OK|R27d FAILED|REPRODUCTION FAILED|Traceback" $R27D_LOG/r27d_*.log
sed -n '/^R27d XR-1/,$p' $R27D_LOG/r27d_xr1.log; sed -n '/^R27d RLDX-1/,$p' $R27D_LOG/r27d_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R27D_ROOT/evidence && cp $R27D_ROOT/*.json $R27D_LOG/*.log $R27D_ROOT/evidence/ && ls $R27D_ROOT/evidence
```

Both logs must end with `R27d OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r27d_split_curves_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r27d_split_curves_evidence/
```

## Result (2026-11-24, run `r27d_split_curves_20261124_174901`, from `d64bed13`): `R27d OK`; Fig. 2's description holds on the 30/70 splits

Both runs ended with `R27d OK` and `INPUTS UNCHANGED`. Gates D0 to D2 pass on both policies with max |diff| 0.0.
Evidence is in `reports/r27d_split_curves_evidence/`:
- `seen_unseen_tables_r27d.json`, sha256 `11abe1ab…`;
- `rldx_e1b_r27d.json`, `64d33238…`;
- logs, gitignored and kept locally.

**Result** (median curve over the 20 assignments, where at least 15 reach the FPR; gap = band minus timer, in
fractions of H_k, on the overlap up to 15% FPR; fixed 50/50 for reference):

| Policy | Head | Side, timer | 30/70 median: gap [min, max] | above from | Fixed 50/50: gap [min, max] |
| --- | --- | --- | --- | --- | --- |
| XR-1 | SAFE-MLP | seen, P | [−0.142, −0.001] | never | [−0.137, −0.016] |
| XR-1 | SAFE-MLP | unseen, E | [−0.009, +0.016] | 12.5% | [−0.010, +0.014] |
| XR-1 | SAFE-LSTM | seen, P | [−0.123, −0.024] | never | [−0.127, −0.024] |
| XR-1 | SAFE-LSTM | unseen, E | [−0.037, +0.041] | 11.6% | [−0.044, +0.049] |
| XR-1 | Hide-and-Seek | seen, P | [−0.015, +0.051] | — | [−0.020, +0.036] |
| XR-1 | Hide-and-Seek | unseen, E | [−0.032, +0.150] | 6.2% | [−0.027, +0.178] |
| XR-1 | Hide-and-Seek | unseen, G | [−0.114, −0.003] | never | [−0.098, +0.025] |
| RLDX-1 | SAFE-MLP | seen, P | [−0.143, −0.063] | never | [−0.126, −0.036] |
| RLDX-1 | SAFE-MLP | unseen, E | [−0.005, +0.011] | 5.6% | [−0.008, +0.017] |
| RLDX-1 | SAFE-LSTM | seen, P | [−0.042, −0.012] | never | [−0.040, +0.021] |
| RLDX-1 | SAFE-LSTM | unseen, E | [−0.024, +0.109] | 5.0% | [−0.010, +0.106] |
| RLDX-1 | Hide-and-Seek | seen, P | [−0.004, +0.067] | — | [−0.005, +0.069] |
| RLDX-1 | Hide-and-Seek | unseen, E | [+0.015, +0.160] | 7.0% | [+0.011, +0.186] |
| RLDX-1 | Hide-and-Seek | unseen, G | [−0.085, −0.029] | never | [−0.078, −0.003] |

("never": the curve is not above the timer at 15% FPR; "—": above in places but not at 15%.) The SAFE bands stay below
G on unseen tasks except SAFE-MLP on XR-1, whose median curve is up to 0.028 above G at low FPR (also on the fixed
split). 5% target points (median FPR, lead): XR-1 SAFE-MLP 4.7%, 0.192 (seen) and 3.9%, 0.115 (unseen); SAFE-LSTM 4.4%,
0.139 and 5.7%, 0.112; Hide-and-Seek 5.0%, 0.241 and 8.0%, 0.197. RLDX-1 SAFE-MLP 5.8%, 0.113 and 7.1%, 0.078;
SAFE-LSTM 7.5%, 0.249 and 6.6%, 0.103; Hide-and-Seek 5.6%, 0.298 and 10.4%, 0.170.

**Reading** (rules fixed above):
1. Fig. 2's own bands become the 30/70 medians with the median target dots; the caption and Sec. IV-A drop the fixed
   50/50 split, which remains only in Sec. VI-B's favorable-draw clause.
2. Sec. VI-B's description stays: on seen tasks the SAFE bands' median curves never rise above P's (at most −0.001)
   and Hide-and-Seek's rises up to 0.051 and 0.067 above it; on unseen tasks SAFE-MLP's stays within 0.02 of E's
   (−0.009 to +0.016 and −0.005 to +0.011).
3. Fig. 1 is unchanged (D1). Its text agrees with the new Fig. 2: Hide-and-Seek's median curve is above E from 6.2% and
   7.0% FPR and below G throughout.

**Correction to the rationale above.** The parenthetical under "Reading" quoted the fixed-split gaps from a local check
that wrongly averaged the ensemble curve with the three single models. The fixed-split values above are the correct
ones: SAFE-MLP against E −0.010 to +0.014 and −0.008 to +0.017, and SAFE-LSTM on RLDX-1 up to +0.021 above P (not
+0.018). The 0.02 rule itself was fixed before the run and is applied unchanged.

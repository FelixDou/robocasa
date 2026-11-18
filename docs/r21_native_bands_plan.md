# R21. The original alarm rules as main results

Written 2026-11-18, before any run. Branch `claude/r21-native-bands`, cut from R20
(`claude/r20-timer-g`, 28fd08a3). Exploratory. Asked by the author after Shinoda sensei's revision
directions v2 (2026-11-18). Their first priority is to compare each monitor, under its own scoring and
alarm rule, with a time-only timer at the same false-alarm budget, and to move those results into the
main results.

## Why

- **The deviation:**
  - In Tables II and III, SAFE-MLP, SAFE-LSTM and Hide-and-Seek alarm when their running maximum
    crosses a constant threshold.
  - Both papers instead use SAFE's functional conformal band, a time-varying threshold.
  - The OOD scores have no published band, and FIPER already runs unchanged.
- **What exists (R12 B2, R19):**
  - Covered: the band on the seed-averaged score (an ensemble), alone at α = 0.05 and at α/2 OR the
    timer at α/2.
  - Not covered:
    - per-seed values (Table II's ±);
    - CIs;
    - curves for Fig. 2;
    - a matched comparison of the band alone with the timer alone.
  - XR-1's seen tasks come from R12, whose matching used a grid of margins.

  | Hide-and-Seek, band OR timer, matched gain | Seen + P | Unseen + E | Unseen + G |
  |---|---|---|---|
  | XR-1 | 0.068 (R12) | 0.091 | 0.013 |
  | RLDX-1 | 0.083 | 0.096 | −0.016 |

  SAFE-MLP and SAFE-LSTM gain at most 0.012.

## Design (fixed now)

Code: `scripts/xr1_lead_time/r21_native_bands.py`, called with `--r21` by `seen_unseen_tables.py`
(XR-1) and `rldx_test_table.py` (RLDX-1). The inputs are R14's and R16's, as in R19 and R20. No new
rollout, model or score is used.

- **The band is R19's:**
  - SAFE's official one-sided functional upper band (`robocasa/recovery/safe/conformal.py`:
    modulation `tfunc`, alignment `extend`, alarm at the first score ≥ the band).
  - It reads the normalized raw score, with no running maximum.
  - It is calibrated on the fold's calibration successes. Per task, those are sorted by rollout ID
    and alternate between the reference and nonconformity halves.
- **Heads:** SAFE-MLP, SAFE-LSTM and Hide-and-Seek on both policies.
  - XR-1: `safe_mlp`, `safe_lstm`, `hide_and_seek`.
  - RLDX-1: `safe_indep`, `safe_lstm`, `hide_and_seek`.
- **Sides and comparators:**
  - Seen tasks (fold `all`, per-task normalization): against P.
  - Unseen tasks (the four folds, pooled normalization, held-out tests pooled): against E (`G_abs`)
    and G.
- **Scores:**
  - Each training seed's band: single models, the ± of Table II.
  - The band on the seed-averaged score: the ensemble, R19's.
- **At α = 0.05, per head, side, comparator and score:**
  1. **Band alone:** FPR, false alarms, TPR, mean lead.
  2. **Band alone against the timer alone (new):** the direct monitor-against-timer comparison.
     - Same target: the timer at α.
     - Matched: the timer at the smallest margin with at most the band's test false alarms
       (`hk_free.matched_timer`, one margin over the pooled folds), as in Table II.
  3. **Band at α/2 OR the timer at α/2 (R19's):** FPR, lead, gain over the timer at α, matched gain.

  Single models are summarized by mean ± std over the three seeds.
- **Curves:**
  - The band alone at each target in R18's grid except 0, since the band needs α > 0.
  - Each point gives test FPR, mean lead and TPR, per seed (with mean ± std) and for the ensemble.
  - The timers' curves are R18's.
- **Bootstrap (ensembles):** 1,000 replicates, seed 21, percentile 95% CIs, on both sides.
  - Each replicate draws tasks with replacement. Within each drawn task it draws:
    - calibration successes within their half (reference or nonconformity), so a resampled rollout
      never lands in both halves;
    - test successes and failures within class.
  - It refits every band (α and α/2) and every timer, then re-matches the timer on the pooled
    resampled tests.
  - Calibration failures are not drawn, because neither the band nor the timer reads them.
  - Statistics:
    - the gain and matched gain of 2 and of 3;
    - the matched-gain contrasts Hide-and-Seek − SAFE-MLP, for 2 and for 3.
  - The seen-task CIs refit the pipeline. Table III's seen-task CIs do not: they are conditional on
    the fitted thresholds.

## Gates

- **R0:** for every head, side, seed and rollout, the running maximum of the normalized raw score
  equals the score Tables I and II use (≤ 1e-12). The band therefore reads the same normalization.
- **N0:** every timer at α equals `hk_free.fit`'s, as in R19.
- **N1:** the ensemble values equal R19's `native_bands`:
  - FP exactly, every other value within 1e-12;
  - XR-1's unseen tasks (+ E, + G); RLDX-1's seen (+ P) and unseen tasks;
  - covering the band alone (FPR, FP, lead), the fused rule (FPR, FP, lead, gain, matched gain and
    the matched timer's false alarms) and the timer alone.
- **N2:** XR-1's seen-task ensemble values for SAFE-MLP and Hide-and-Seek equal R12 B2's
  (`list_b.json`, sha256 `0e755875…`, in the repository's evidence folder):
  - at its 4 decimals (≤ 5e-5), with FP exact;
  - covering the band alone (FPR, FP, lead) and the fused rule (FPR, FP, lead, gain).
  - The matched gain is compared for information only: R12 matched P on a margin grid, while R21
    uses `hk_free.matched_timer`, like Table II.
- **C0:** each curve's α = 0.05 point equals item 1 (≤ 1e-12).
- **B0:** R21's evaluation, which the bootstrap reuses (explicit halves, cohort subsets), equals
  `r19_extras.native_band` run on the same rows in the same run. It covers every head, side,
  comparator and score (each seed and the ensemble), all values, matched gains included
  (≤ 1e-12; FP exactly). The bootstrap's point estimates equal the same values.
- **Tests:** `tests/test_r21_native_bands.py` also reproduces R12's own band code
  (`review_analyses_b.band_analysis`) on synthetic seen tasks at R12's 4 decimals.
- **The scripts' own gates:**
  - XR-1: T1–T6;
  - RLDX-1: G0–G2 and K0.
- **Inputs:** R16's frozen inputs are unchanged at the end.

## Expectations (written before running)

- **Gates:** all pass, and the ensemble's fused values equal the table above.
- **Band alone against the matched timer:** rough values, read off R18's curves and R19's
  band-alone points.
  - Hide-and-Seek:
    - seen tasks: above P (XR-1 about +0.05, RLDX-1 about +0.08);
    - unseen tasks: above E (about +0.15 on both) and near G (XR-1 about +0.07, RLDX-1 about +0.01).
  - SAFE-MLP: at or below the timers (RLDX-1 seen about −0.12).
  - SAFE-LSTM: in between.
- **Over-alarming:** the bands over-alarm on unseen tasks. In R19, Hide-and-Seek's band alone reached
  FPR 0.076 on XR-1 and 0.100 on RLDX-1, so only matched values are comparable there.
- **Single models:** they gain less than the ensemble, as with the constant threshold (0.036 against
  0.056 on XR-1's seen tasks).
- **CIs (Hide-and-Seek, band OR timer):**
  - over E: above zero on both policies;
  - over G: including zero on both;
  - seen tasks: including zero on XR-1 and above zero on RLDX-1.
  - SAFE-MLP's include zero or lie below it.
- **Curves:** Hide-and-Seek's band lies above P on seen tasks, above E on unseen tasks, and near G.

## Reporting (fixed now)

- **Source:** every original-rule value the paper quotes comes from R21, with its CI (ensembles) or
  its ± (single models).
- **Both comparisons are reported:**
  - the band alone against the timer: sensei's direct comparison;
  - the band OR the timer against the timer: whether the monitor adds early warning beyond it.
- **Placement:** the author chooses after the result whether these go in Table II rows, Fig. 2
  curves or the text. No head, side or target is chosen by its result.
- **Completeness:** R21 is reported whatever it shows.

## Runbook

CPU only. Each run takes well under an hour, about as long as R20, plus the bootstrap. Timed locally
at XR-1's size, a replicate takes about 0.1 s per head, so 1,000 replicates take about 5 minutes. The
two runs go in parallel.

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=2:00:00
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
export R21_SHA=<full SHA of the pushed commit containing this plan>
export R21_REPO=$PROJECT_FS/robocasa_r21_${R21_SHA:0:8}
export R21_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r21_native_bands_$(date +%Y%m%d_%H%M%S)
export R21_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R21_ROOT)
export R19_XR1=$(ls $STORAGE_BS/robocasa_checkpoints/safe/r19_triage_b_*/seen_unseen_tables_r19.json | tail -n 1)
export R19_RLDX=$(ls $STORAGE_BS/robocasa_checkpoints/safe/r19_triage_b_*/rldx_e1b_r19.json | tail -n 1)
mkdir -p $R21_ROOT $R21_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r21-native-bands
test -e $R21_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R21_REPO $R21_SHA
test "$(git -C $R21_REPO rev-parse HEAD)" = "$R21_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export XR1_FOLDS=$R21_REPO/scripts/xr1_lead_time/cross_task_folds.json
export R12_LISTB=$R21_REPO/reports/safe_robocasa365_research_plan_20260823/evidence/xr1_list_b_20261030_212017/list_b.json
cmp $XR1_FOLDS $R14_REPO/scripts/xr1_lead_time/cross_task_folds.json && echo "XR-1 FOLDS SAME AS R14"
echo "R19_XR1=$R19_XR1"; echo "R19_RLDX=$R19_RLDX"
sha256sum $R19_XR1 $R19_RLDX $R12_LISTB $R16_ROOT/rldx_e1b_test_table.json | cut -c1-16
cd $R21_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r21_native_bands tests.test_xr1_seen_unseen_tables \
  tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- `HEAD OK` and `XR-1 FOLDS SAME AS R14` print.
- `R19_XR1` and `R19_RLDX` each name one file.
- The hashes are, in order: `d4d8176f…`, `81d0beec…` (R19's evidence copies), `0e755875…` (R12),
  and R16's table.
- The tests end with `OK`.

Then start both runs in the background:

```bash
cd $R21_REPO
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
  --r21 --r21-reference $R19_XR1 --r21-r12-reference $R12_LISTB --r21-boot 1000 \
  --output $R21_ROOT/seen_unseen_tables_r21.json' > $R21_LOG/r21_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r21 --r21-reference $R19_RLDX --r21-boot 1000 --output $R21_ROOT/rldx_e1b_r21.json' > $R21_LOG/r21_rldx.log 2>&1 &
echo "started; R21_ROOT=$R21_ROOT"
```

**Progress:** `tail -n 3 $R21_LOG/r21_*.log`. The bootstrap prints every 100 replicates.

When both logs end:

```bash
grep -E "R21 OK|R21 FAILED|FAILED|STOP|Traceback" $R21_LOG/r21_*.log
sed -n '/^R21 /,$p' $R21_LOG/r21_xr1.log; sed -n '/^R21 /,$p' $R21_LOG/r21_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R21_ROOT/evidence && cp $R21_ROOT/*.json $R21_LOG/*.log $R21_ROOT/evidence/ && ls $R21_ROOT/evidence
```

Both logs must end with `R21 OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r21_native_bands_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r21-native-bands/reports/r21_native_bands_evidence/
```

## Result

Run `r21_native_bands_20261118_172324` at b571774b, on 2026-11-18. Evidence is in
`reports/r21_native_bands_evidence/`: `seen_unseen_tables_r21.json` (sha256 `53599170…`),
`rldx_e1b_r21.json` (`8f9dd45c…`) and both logs.

- **Gates:**
  - Both logs end with `R21 OK`, and the frozen inputs are unchanged.
  - R0, N0 and C0 hold exactly (max |diff| 0).
  - B0 holds exactly: 45 comparisons per policy.
  - N1 holds exactly: 6 bands on XR-1, 9 on RLDX-1.
  - N2 holds within R12's rounding (max 4.9e-5).
  - The scripts' own gates pass.
- **Matching (N2, information):** R12's grid matching gave a matched gain over P of 0.0677
  (Hide-and-Seek) and 0.0055 (SAFE-MLP). R21's exact matching gives 0.0656 and 0.0053, with the same
  timer false alarms (26 and 21).
- **Ensembles, matched gain over the timer, with the full-pipeline 95% CI:**

  | Head, policy | Seen: alone vs P | Seen: OR P | Unseen: alone vs E | Unseen: OR E | Unseen: alone vs G | Unseen: OR G |
  |---|---|---|---|---|---|---|
  | Hide-and-Seek, XR-1 | +0.047 [−0.038, 0.128] | +0.066 [−0.020, 0.136] | +0.147 [−0.030, 0.208] | +0.091 [−0.048, 0.165] | +0.070 [−0.149, 0.091] | +0.013 [−0.153, 0.076] |
  | Hide-and-Seek, RLDX-1 | +0.086 [−0.034, 0.166] | +0.083 [−0.015, 0.135] | +0.143 [−0.066, 0.259] | +0.096 [−0.164, 0.187] | +0.019 [−0.162, 0.068] | −0.016 [−0.148, 0.042] |
  | SAFE-MLP, XR-1 | −0.016 [−0.122, 0.054] | +0.005 [−0.081, 0.067] | −0.012 [−0.065, 0.030] | −0.027 [−0.068, 0.016] | −0.075 [−0.187, 0.059] | −0.036 [−0.124, 0.036] |
  | SAFE-MLP, RLDX-1 | −0.126 [−0.232, −0.037] | +0.001 [−0.102, 0.028] | +0.004 [−0.044, 0.033] | −0.006 [−0.038, 0.028] | −0.119 [−0.261, −0.028] | −0.073 [−0.179, −0.003] |
  | SAFE-LSTM, XR-1 | −0.016 [−0.140, 0.053] | +0.035 [−0.067, 0.078] | −0.021 [−0.122, 0.063] | −0.011 [−0.085, 0.060] | −0.090 [−0.231, 0.007] | −0.061 [−0.170, 0.029] |
  | SAFE-LSTM, RLDX-1 | −0.021 [−0.093, 0.067] | +0.009 [−0.053, 0.078] | +0.033 [−0.103, 0.148] | +0.012 [−0.159, 0.132] | −0.075 [−0.204, 0.049] | −0.071 [−0.159, 0.028] |

  - The bands alone over-alarm on unseen tasks. Hide-and-Seek reaches FPR 0.076 (XR-1) and 0.100
    (RLDX-1); its fused rule reaches 0.074 and 0.118.
  - Hide-and-Seek − SAFE-MLP (matched) is above zero only on RLDX-1's seen tasks: +0.212
    [0.052, 0.354] for the bands alone, and +0.082 [0.005, 0.188] with the timer.
- **Single models, matched gain, mean ± std over seeds (Hide-and-Seek):**

  | Policy | Seen: alone vs P | Seen: OR P | Unseen: alone vs E | Unseen: OR E | Unseen: alone vs G | Unseen: OR G |
  |---|---|---|---|---|---|---|
  | XR-1 | +0.026 ± 0.039 | +0.065 ± 0.017 | +0.079 ± 0.027 | +0.057 ± 0.026 | −0.021 ± 0.041 | −0.025 ± 0.031 |
  | RLDX-1 | +0.069 ± 0.027 | +0.082 ± 0.022 | +0.083 ± 0.037 | +0.061 ± 0.010 | −0.066 ± 0.020 | −0.047 ± 0.019 |

- **Curves (single models, mean over seeds), against R18's timer curves:**
  - Hide-and-Seek, seen tasks: above P up to an FPR of about 0.10, level with it around 0.11 to 0.12,
    and slightly below it beyond. For example, on XR-1 at FPR 0.079: lead 0.303, against P's 0.27.
  - Hide-and-Seek, unseen tasks: above E everywhere. Against G:
    - XR-1: near it, and above it from FPR 0.15.
    - RLDX-1: below it up to FPR 0.11, level around 0.15, above it beyond.
  - SAFE-MLP: far below P on RLDX-1's seen tasks, and on or below the timers elsewhere.
- **Expectations:**
  - **Held:**
    - the gates;
    - every ensemble point estimate within a few hundredths of the rough values;
    - over-alarming on unseen tasks;
    - CIs over G including zero;
    - SAFE-MLP's CIs including zero or below it;
    - XR-1's seen-task CI including zero.
  - **Not held:**
    - Hide-and-Seek's gain over E has CIs that include zero on both policies (expected above zero).
    - RLDX-1's seen-task CI includes zero (expected above zero).
  - **Partly held:**
    - Single models gain less than the ensemble on unseen tasks, but about as much on seen tasks.
    - The curves lie above P only at low FPR.
- **The CIs are lopsided:** several point estimates lie near the top of their interval (XR-1 over G:
  0.013 in [−0.153, 0.076]; RLDX-1 over E: 0.096 in [−0.164, 0.187]). Resampled replicates mostly do
  worse than the real data, which R19's constant-threshold CIs did not show. R21b diagnoses this
  before the paper reports these CIs (`docs/r21b_band_ci_diagnostic_plan.md`).
- **Reporting:** pending R21b. Under the fixed rules, the point estimates and the single-model values
  above are what the paper quotes for the original alarm rules. The CI to quote depends on R21b's
  interpretation rules.

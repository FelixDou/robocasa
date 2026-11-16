# R19. The draft v3 mock review's offline analyses (triage B)

Written 2026-11-16, before any run. Branch `claude/r19-triage-b`, cut from R18
(`claude/r18-lead-fpr-curves`). Exploratory. Source:
`reports/paper_time_aware_fusion/rss2027_stage1/mock_review_20261116.md` (paper-drafts), triage B.

## Why

The three mock reviewers of draft v3 asked for analyses that need no new rollouts:
1. **Monitors under their original time-varying alarm bands**, not only our constant threshold:
   SAFE-MLP, SAFE-LSTM and Hide-and-Seek.
   - R12 B2 did this on XR-1's seen tasks: Hide-and-Seek's band + P gains +0.068 matched; SAFE-MLP's
     +0.006.
   - Missing: unseen tasks (against E and G) on both policies, and RLDX-1's seen tasks.
2. **CIs for the unseen-task gains in the text** (XR-1 0.097 and 0.048; RLDX-1 0.136/0.140), from a
   bootstrap that resamples successes as well as failures, refits the folds and re-matches the
   timer.
3. **The timers' detections at the limit:** a timer "flags" a failure at the final inference with
   lead 0, which inflates its bACC.
4. **Lead in seconds and per-task gains** on unseen tasks (R12 B5 covers XR-1's seen tasks).
5. **The zero-input model's plateau** in Fig. 2 (RLDX-1 seen tasks), which may be saturation.

## Design (fixed now)

Code: `scripts/xr1_lead_time/r19_extras.py`, called with `--r19` by `seen_unseen_tables.py` (XR-1)
and `rldx_test_table.py` (RLDX-1). The inputs are R14's (XR-1) and R16's (RLDX-1); no new rollout,
model or score.

- **Native bands (`native_band`):**
  - The band is SAFE's official functional conformal upper band (`conformal.py`; "tfunc"
    modulation, "extend" alignment, as R12 B2), on the seed-averaged normalized raw score, with no
    running maximum.
  - Per fold, the band is calibrated on that fold's calibration successes, split per task into
    reference and nonconformity halves.
  - Reported: the band alone at α = 0.05, and the band at α/2 OR the timer at α/2 (Bonferroni, as
    R12). Gains are over the timer at α, at the same target and matched.
  - Unseen tasks: against E (`G_abs`) and G. RLDX-1's seen tasks: against P.
  - Heads: SAFE-MLP, SAFE-LSTM (XR-1 only if its fold directories are given, as in R14) and
    Hide-and-Seek.
- **Bootstrap (`boot_unseen`):**
  - Each replicate draws tasks with replacement, then calibration and test rollouts within task
    and class.
  - It refits every fold (`hk_free.fit`), pools the held-out tests and re-matches the timer.
  - 1,000 replicates (seed 19); percentile 95% CIs.
  - Ensembles of Hide-and-Seek, SAFE-MLP, SAFE-LSTM and the zero input; timers E and G.
  - Matched contrasts: Hide-and-Seek − SAFE-MLP and Hide-and-Seek − zero input.
- **Leads (`leads_detail`):** per failure, for Hide-and-Seek, SAFE-MLP and SAFE-LSTM
  (ensembles):
  - seen tasks + P and unseen tasks + E;
  - lead in seconds (XR-1 0.8 s, RLDX-1 0.4 s per inference), split atomic and composite;
  - per-task gains, with the number of tasks above and below zero and the top-5 share;
  - the timers P, E and G alone: failures flagged only at the final inference, and bACC if those
    count as misses.
- **Saturation (`saturation`):** distinct values and largest tie among the calibration running
  maxima, for the zero input and Hide-and-Seek (seed average and each seed), on seen tasks.

## Gates

- **B0:** the bootstrap's point estimates, with no resampling, equal `hk_free.block`'s seed
  averages in R14 (XR-1) and R17 (RLDX-1), ≤ 1e-12. They are passed as `--r19-reference`.
- **N0:** in every band fit, the timer at α equals `hk_free.fit`'s.
- **The scripts' own gates:**
  - XR-1: T1–T6 and R6's rows unchanged.
  - RLDX-1: R16's G0–G2, and K0 (Table II equals R16's).
- **Inputs:** R16's frozen inputs are unchanged at the end.

## Expectations (written before running)

- **Native bands, unseen tasks:**
  - Hide-and-Seek's band + E gains over E, by about as much as its constant-threshold fusion
    (ensemble: 0.097 matched on XR-1). On seen XR-1 tasks the band did better than the constant
    threshold (0.068 vs 0.056).
  - SAFE-MLP's band + E gains at most about 0.02. If the band removed SAFE-MLP's time trend, it
    could gain more; that would change the paper's SAFE-MLP sentence.
  - SAFE-LSTM: a few hundredths.
- **Bootstrap:**
  - Hide-and-Seek's matched gain over E has a 95% CI above zero on both policies.
  - Over G, the CI may include zero (XR-1 point 0.048).
  - SAFE-MLP's CI includes zero.
- **Timers at the limit:**
  - P flags at most about 10% of failures only at the limit.
  - E flags more, on short tasks whose limit is below its cutoff.
- **Seconds:** Hide-and-Seek's unseen-task gain is larger on atomic tasks.
- **Saturation:** the zero input on RLDX-1 seen tasks has very few distinct calibration maxima;
  Hide-and-Seek has many.

## Reporting (fixed now)

- Every unseen-task gain the paper quotes gets its R19 CI.
- The native-band gains enter the text in one sentence, whatever they show.
- The timers' share of failures flagged only at the limit goes into Table I's footnote.
- If the zero input is saturated on RLDX-1, Fig. 2's caption says so.

## Runbook

CPU only. Each run takes about an hour, mostly the bootstrap; the two run in parallel.

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
export R19_SHA=<full SHA of the pushed commit containing this plan>
export R19_REPO=$PROJECT_FS/robocasa_r19_${R19_SHA:0:8}
export R19_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r19_triage_b_$(date +%Y%m%d_%H%M%S)
export R19_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R19_ROOT)
export R17_JSON=$(ls $STORAGE_BS/robocasa_checkpoints/safe/rldx1_r17_elapsed_*/rldx_e1b_r17.json | tail -n 1)
mkdir -p $R19_ROOT $R19_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r19-triage-b
test -e $R19_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R19_REPO $R19_SHA
test "$(git -C $R19_REPO rev-parse HEAD)" = "$R19_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export XR1_FOLDS=$R19_REPO/scripts/xr1_lead_time/cross_task_folds.json
cmp $XR1_FOLDS $R14_REPO/scripts/xr1_lead_time/cross_task_folds.json && echo "XR-1 FOLDS SAME AS R14"
echo "R17_JSON=$R17_JSON"; sha256sum $R17_JSON $R14_ROOT/seen_unseen_tables_r14.json | cut -c1-16
cd $R19_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r19_extras tests.test_xr1_seen_unseen_tables \
  tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if these hold:
- `HEAD OK` and `XR-1 FOLDS SAME AS R14` print;
- `R17_JSON` names one file, with hash `f531fd5a…`, and R14's output hashes `1578f312…` (the
  evidence copy of `seen_unseen_tables_r14.json`);
- the tests end with `OK`.

Then both runs, in the background:

```bash
cd $R19_REPO
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
  --r19 --r19-reference $R14_ROOT/seen_unseen_tables_r14.json --r19-boot 1000 \
  --output $R19_ROOT/seen_unseen_tables_r19.json' > $R19_LOG/r19_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r19 --r19-reference $R17_JSON --r19-boot 1000 --output $R19_ROOT/rldx_e1b_r19.json' > $R19_LOG/r19_rldx.log 2>&1 &
echo "started; R19_ROOT=$R19_ROOT"
```

**Progress:** `tail -n 3 $R19_LOG/r19_*.log`.

When both logs end:

```bash
grep -E "R19 OK|R19 FAILED|FAILED|STOP|Traceback" $R19_LOG/r19_*.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R19_ROOT/evidence && cp $R19_ROOT/*.json $R19_LOG/*.log $R19_ROOT/evidence/ && ls $R19_ROOT/evidence
```

Both logs must end with `R19 OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r19_triage_b_*/evidence/' <local worktree>/reports/r19_triage_b_evidence/
```

## Result

(to be written after the run)

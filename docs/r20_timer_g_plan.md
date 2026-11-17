# R20. Timer G as a row of Table II

Written 2026-11-17, before any run. Branch `claude/r20-timer-g`, cut from R19 (`claude/r19-triage-b`,
7aec984f). Exploratory. Asked by the author for `draft_v3_final`
(`reports/paper_time_aware_fusion/rss2027_stage1/`, paper-drafts).

## Why

- **The gap:** Table II gives each monitor's gain over P (seen tasks) or E (unseen tasks). The abstract
  also compares Hide-and-Seek with G, the timer told each task's limit H_k, but G is not in Table II.
- **The request:** a row "Timer G" under "Timer P / E", with G's FPR and its gain over P (seen) and
  over E (unseen), at the same 5% target and at matched false alarms.
- **What exists:**
  - Same-target values from R14 (XR-1) and R16/R17 (RLDX-1):

    | Policy | Seen: G − P | Unseen: G − E |
    |---|---|---|
    | XR-1 | −0.102 (G 8 FP, P 21) | +0.004 (9 vs 19) |
    | RLDX-1 | −0.114 (23 vs 27) | +0.085 (26 vs 30) |

  - Matched values do not exist.
  - On XR-1, G has half of E's false alarms, so the same-target gain understates G.

## Design (fixed now)

Code: `scripts/xr1_lead_time/r20_timer_g.py`, called with `--r20` by `seen_unseen_tables.py` (XR-1)
and `rldx_test_table.py` (RLDX-1). The inputs are R14's and R16's, as in R19. No new rollout, model
or score is used, and no score is read: every timer alarms on elapsed time only.

- **G is hk_free's G** (time unit H_k, offset 0):
  - It is the timer of Table II's unseen + G comparisons and of the abstract's gains over G.
  - It is not Table I's standalone global timer, which thresholds t/H_k with Table I's rule. On
    XR-1's seen tasks that timer has 9 false alarms; hk_free's G has 8.
- **Seen tasks:** Table II's seen cohort (fold `all`), G against P. Each timer's margin is set by
  Proposition 1 on the calibration successes.
- **Unseen tasks:** Table II's four folds, G against E (`G_abs`). Calibration uses the other tasks;
  the held-out tests are pooled.
- **Gain:** the mean over failures of G's lead minus the comparator's (same failures, paired).
- **Matched:** the comparator at the smallest margin with at most G's test false alarms
  (`hk_free.matched_timer`, one margin over the pooled folds), as for every monitor in Table II.
- **No CI:** Table II has none. Its ± are over training seeds, and timers have no seed.

## Gates

- **T0:** each timer alone equals R14's (XR-1) or R17's (RLDX-1) `timers_alone`: the number of false
  alarms exactly, FPR and mean lead within 1e-12.
  - XR-1: P and G on seen tasks; E and G on unseen tasks.
  - RLDX-1: P on seen tasks; E and G on unseen tasks.
  - RLDX-1's seen G has no hk_free reference. For information only, it is compared with R16's rounded
    Table II value (`table2.calibration.timer_G.seen`: FPR 0.0575, lead 0.1544, gain over P −0.1142).
- **The scripts' own gates:**
  - XR-1: T1–T6;
  - RLDX-1: G0–G2 and K0.
- **Inputs:** R16's frozen inputs are unchanged at the end.

## Expectations (written before running)

- **Same target:** the values above, since T0 reproduces each timer.
- **Matched:** a comparator held to G's fewer false alarms loses lead.
  - XR-1 unseen: the gain rises above +0.004, with E held to 9 false alarms instead of 19.
  - XR-1 seen: the gain rises above −0.102, with P held to 8 instead of 21.
  - RLDX-1: smaller changes, since G's false alarms are close to the comparators'.
- **RLDX-1's seen G matches R16's rounded value** if R16 used hk_free's rule.

## Reporting (fixed now)

- **Table II:** a row "Timer G" under "Timer P / E" in each policy block. It gives G's FPR, gain and
  matched gain over P (seen) and over E (unseen), with no ±, whatever the numbers.
- **Ranking:** timers are not ranked (no bold or underline).
- **Caption:** it says that G is told H_k.

## Runbook

CPU only, about the size of R19 without its bootstrap (well under an hour). The two runs go in
parallel.

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
export R20_SHA=<full SHA of the pushed commit containing this plan>
export R20_REPO=$PROJECT_FS/robocasa_r20_${R20_SHA:0:8}
export R20_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r20_timer_g_$(date +%Y%m%d_%H%M%S)
export R20_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R20_ROOT)
export R17_JSON=$(ls $STORAGE_BS/robocasa_checkpoints/safe/rldx1_r17_elapsed_*/rldx_e1b_r17.json | tail -n 1)
mkdir -p $R20_ROOT $R20_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r20-timer-g
test -e $R20_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R20_REPO $R20_SHA
test "$(git -C $R20_REPO rev-parse HEAD)" = "$R20_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export XR1_FOLDS=$R20_REPO/scripts/xr1_lead_time/cross_task_folds.json
cmp $XR1_FOLDS $R14_REPO/scripts/xr1_lead_time/cross_task_folds.json && echo "XR-1 FOLDS SAME AS R14"
echo "R17_JSON=$R17_JSON"; sha256sum $R17_JSON $R14_ROOT/seen_unseen_tables_r14.json | cut -c1-16
cd $R20_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r20_timer_g tests.test_xr1_seen_unseen_tables \
  tests.test_rldx_test_table 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- `HEAD OK` and `XR-1 FOLDS SAME AS R14` print;
- `R17_JSON` names one file, with hash `f531fd5a…`, and R14's output hashes `1578f312…`;
- the tests end with `OK`.

Then start both runs in the background:

```bash
cd $R20_REPO
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
  --r20 --r20-reference $R14_ROOT/seen_unseen_tables_r14.json \
  --output $R20_ROOT/seen_unseen_tables_r20.json' > $R20_LOG/r20_xr1.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r20 --r20-reference $R17_JSON --output $R20_ROOT/rldx_e1b_r20.json' > $R20_LOG/r20_rldx.log 2>&1 &
echo "started; R20_ROOT=$R20_ROOT"
```

**Progress:** `tail -n 3 $R20_LOG/r20_*.log`.

When both logs end:

```bash
grep -E "R20 OK|R20 FAILED|FAILED|STOP|Traceback" $R20_LOG/r20_*.log
sed -n '/^R20 /,$p' $R20_LOG/r20_xr1.log; sed -n '/^R20 /,$p' $R20_LOG/r20_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R20_ROOT/evidence && cp $R20_ROOT/*.json $R20_LOG/*.log $R20_ROOT/evidence/ && ls $R20_ROOT/evidence
```

Both logs must end with `R20 OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r20_timer_g_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r20-timer-g/reports/r20_timer_g_evidence/
```

## Result

Run `r20_timer_g_20261117_174756` at 028742de, 2026-11-17. Evidence is in
`reports/r20_timer_g_evidence/`: `seen_unseen_tables_r20.json` (sha256 `41a52c2f…`),
`rldx_e1b_r20.json` (`39721a0e…`) and both logs.

- **Gates:**
  - Both logs end with `R20 OK`.
  - T0 holds exactly (max |diff| 0): 4 timers against R14 and 3 against R17.
  - The scripts' own gates pass, and the frozen inputs are unchanged.
- **G alone against the comparator alone** (G's FPR and false alarms; gain at the same target;
  matched gain, with the comparator's false alarms):

  | Policy | Seen tasks, against P | Unseen tasks, against E |
  |---|---|---|
  | XR-1 | 0.021 (8 FP); −0.1024; −0.0444 (P at 8 FP) | 0.024 (9 FP); +0.0043; +0.0222 (E at 9 FP) |
  | RLDX-1 | 0.0575 (23 FP); −0.1143; −0.1016 (P at 23 FP) | 0.065 (26 FP); +0.0855; +0.0826 (E at 25 FP) |

- **Expectations:**
  - Same target: as written.
  - Matched: as written. On XR-1, holding the comparator to G's fewer false alarms raises G's gain
    (seen −0.102 → −0.044, unseen +0.004 → +0.022). On RLDX-1 it barely moves.
  - RLDX-1's seen G uses R16's rule:
    - FPR is equal and the lead differs by 3.6e-5 (R16 stores 4 decimals);
    - R16's gain over P, −0.1142, is the difference of its rounded leads (0.1544 − 0.2686); the
      exact value is −0.1143.
    - The script's information check flagged this 8.2e-5 rounding difference as "differs".
- **Reporting:** Table II's G rows in the paper (paper-drafts, `main.tex`):
  - XR-1: 0.021, −0.102, −0.044 | 0.024, 0.004, 0.022;
  - RLDX-1: 0.058, −0.114, −0.102 | 0.065, 0.085, 0.083.
  - Timers are not ranked.

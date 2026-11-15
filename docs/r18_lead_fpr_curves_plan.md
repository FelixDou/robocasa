# R18. Lead–FPR curves of monitors and time-only timers

Written 2026-11-16, before any run. Branch `claude/r18-lead-fpr-curves`, cut from R17
(`claude/r17-elapsed-time`). Exploratory.

## Why

Sensei's revision directions for draft v3, points 3 and 5:
- compare each monitor's alarm times directly with the timer's;
- show a warning-lead-versus-FPR curve.

Draft v3's Fig. 2b plots only the gain of fused ensembles over a matched timer on XR-1's seen
tasks. R18 gives every monitor's own lead–FPR curve next to the timers', on both policies, seen
and unseen tasks.

## Design (fixed now)

- **One point per calibration target α** in `lead_fpr_curves.ALPHAS`: 0 to 0.30, 20 targets.
  - The threshold is set on calibration rollouts with Table I's rule
    (`per_seed_detection.threshold`): the best calibration lead among thresholds with calibration
    FPR ≤ α.
  - Each point is then evaluated on the test set (test FPR, mean lead). No threshold sees the
    test set.
- **Alarm:** the first inference whose running maximum reaches the threshold, as in Table I.
- **Monitors:** OOD (4), SAFE-MLP, SAFE-LSTM, Hide-and-Seek and its first-step and zero-input
  ablations, plus the shuffle on RLDX-1's seen tasks.
  - Seen tasks: per-task normalized scores, one threshold for all tasks.
  - Unseen tasks: per fold, the threshold comes from the fold's seen tasks' calibration rollouts,
    and the held-out tasks' test outcomes are pooled.
  - Per seed, the three-seed ensemble, and mean ± std over seeds at each α.
- **Timers:** each is a score with the same rule. G = t/H_k; E = t (inferences; not told H_k);
  P = t/H_k − ρ_k (seen tasks only).
- **Code:**
  - `scripts/xr1_lead_time/lead_fpr_curves.py`;
  - `seen_unseen_tables.py run --curves` (XR-1);
  - `rldx_test_table.py run --curves` (RLDX-1).
- **Inputs:** R14's for XR-1, R16's for RLDX-1. No new rollout, model or score.

## Gates

- **C1 (XR-1):** at α = 0.05, every monitor's point (seed average and each seed, seen and unseen)
  and G's equal Table I (FPR and lead, ≤ 1e-12).
- **XR-1:** `seen_unseen_tables.py` reruns its own gates (T1–T6), and `table1`/`table2` must equal
  R14's output.
- **RLDX-1:** R16's gates and K0 (Table II equals R16's output, sha256 `9f41e264…`). RLDX-1 has no
  Table I, so no C1; the curve code is the one C1 checks on XR-1.
- **Inputs:** R16's frozen inputs are unchanged at the end.

## Expectations (written before running)

- **XR-1, seen tasks:**
  - At α = 0.05, P leads every monitor except Hide-and-Seek, as in Table I (0.225 against 0.246).
  - E is far below P: it lacks the task's own duration.
- **XR-1, unseen tasks:**
  - Hide-and-Seek's curve lies above E's at every FPR.
  - The zero-input ablation's curve lies close to E's, since it is learned elapsed time.
  - SAFE-MLP's lies close to E's, since it adds 0.002 when fused with E.
  - G lies above E.
- **RLDX-1:** the same ordering. Hide-and-Seek has the highest curve on both sides.

## Reporting (fixed now)

- **The figure:** the curves become a figure in draft v3, whatever they show.
  - Content: mean lead against test FPR (mean over seeds), for Hide-and-Seek, SAFE, one OOD score
    (Mahalanobis), the zero-input ablation and the timers (P on seen tasks, E and G on unseen
    tasks).
  - XR-1 first; RLDX-1 if the page allows.
  - Where it goes (replacing Table I or Fig. 2b) is decided with the author once the figure
    exists. Table I's numbers stay quoted in the text.
- **Text:** α = 0.05 values come from these files.

## Runbook

CPU only. XR-1's step loads every export, as in R14 step 12.4, so allow about an hour.

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
source $STORAGE_BS/robocasa_logs/eval/xr1_r14_hk_free_20261106_191531/paths.env     # R14_ROOT ...
source $STORAGE_BS/robocasa_checkpoints/safe/rldx1_e1b_test_table_20261113_145031/session.env   # R16
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R18_SHA=<full SHA of the pushed commit containing this plan>
export R18_REPO=$PROJECT_FS/robocasa_r18_${R18_SHA:0:8}
export R18_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r18_lead_fpr_$(date +%Y%m%d_%H%M%S)
export R18_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R18_ROOT)
mkdir -p $R18_ROOT $R18_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r18-lead-fpr-curves
test -e $R18_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R18_REPO $R18_SHA
test "$(git -C $R18_REPO rev-parse HEAD)" = "$R18_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
cd $R18_REPO
export XR1_FOLDS=$R18_REPO/scripts/xr1_lead_time/cross_task_folds.json
cmp $XR1_FOLDS $R14_REPO/scripts/xr1_lead_time/cross_task_folds.json && echo "XR-1 FOLDS SAME AS R14"
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_xr1_seen_unseen_tables tests.test_rldx_test_table \
  tests.test_xr1_hk_free 2>&1 | grep -E "^(Ran|OK|FAILED)"
echo "R18_ROOT=$R18_ROOT"
```

Continue only if `HEAD OK`, `XR-1 FOLDS SAME AS R14` and `OK` print.

**XR-1** (R14's step 12.4 command, without `--r14-durations`, with `--curves`):

```bash
cd $R18_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/seen_unseen_tables.py run \
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
  --r6-result $R6_ROOT/seen_unseen_tables_b1ids.json --curves \
  --output $R18_ROOT/seen_unseen_tables_r18.json 2>&1 | tee $R18_LOG/r18_xr1.log | tail -n 40
"$SAFE_PY" -I - $R14_ROOT/seen_unseen_tables_r14.json $R18_ROOT/seen_unseen_tables_r18.json <<'PY'
import json, sys
x, y = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print("XR-1 TABLES SAME AS R14" if all(x[k] == y[k] for k in ("table1", "table2")) else "STOP: tables differ from R14")
PY
```

**RLDX-1** (R17's command, without `--elapsed-time`, with `--curves`):

```bash
cd $R18_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --curves --output $R18_ROOT/rldx_e1b_r18.json 2>&1 | tee $R18_LOG/r18_rldx.log | tail -n 40
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R18_ROOT/evidence && cp $R18_ROOT/*.json $R18_LOG/*.log $R18_ROOT/evidence/ && ls $R18_ROOT/evidence
```

**Continue only if all of these hold:**
- the XR-1 log ends with `R18 C1: every alpha = 0.05 point equals Table I`, the curve printout
  and `R18 CURVES OK`;
- `XR-1 TABLES SAME AS R14` prints;
- the RLDX-1 log shows R16's gates `OK`, `R17 K0: Table II equals …` and `R18 CURVES OK`;
- `INPUTS UNCHANGED` prints.

Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r18_lead_fpr_*/evidence/' <local worktree>/reports/r18_lead_fpr_evidence/
```

## Result

(to be written after the run)

# R27e: Table I's own-band rows on R27's random 30/70 assignments (XR-1)

Status: descriptive, written before the run (2026-11-24). Follows R27c and R27d (`docs/r27c_split_all_bands_plan.md`,
`docs/r27d_split_curves_plan.md`).

## Why

After R27c and R27d, Table II, Fig. 1 and Fig. 2 report every own band (SAFE-MLP, SAFE-LSTM, Hide-and-Seek) on R27's 20
random 30/70 assignments of the calibration successes, and Sec. IV-A says "We fit every band with Hide-and-Seek's
published 30/70 split". Table I (XR-1, monitors and timers alone at the 5% target) still takes its three own-band rows
from R21's fixed 50/50 split (via R26's recount, which counts an alarm at a failure's final inference as a miss). They
no longer match Fig. 2's own-band dots: Hide-and-Seek's lead is 0.303 (seen) and 0.263 (unseen) in Table I, against
0.241 and 0.197 at Fig. 2's dots. R27e computes Table I's own-band rows on the same assignments.

## What R27e computes (XR-1 only; Table I has no RLDX-1 part)

- **Assignments.** R27's 20 random 30/70 splits (seeds 27 to 46), stratified by task; seen tasks split per task with
  the same seed, hence the same assignment (gate E2).
- **Per assignment, head, side and single model:** the band alone at alpha = 0.05 (SAFE's routine, pooled over folds
  as `r21_native_bands.evaluate` computes its standalone part), recounted as R26 does
  (`r26_alarms_at_limit.from_counts`: recall and bACC with an alarm at the final inference counted as a miss; FPR,
  false alarms and mean lead), and the mean and std (ddof 1) over the three models, as Table I prints them.
- **Summary:** for every value, the median over the 20 assignments of the mean over models and the median of their
  std (the Table I cell), the range of the mean, and each model's median over the assignments.
- **Unchanged:** scores, folds, SAFE's routine, the 5% target, the test rollouts. The other Table I rows (timers, OOD
  scores, constant-threshold classifiers) do not use a calibration split and are not recomputed.

## Gates

- **E0:** at R19's fixed 50/50 split, every head's single-model rows on both sides equal R26's saved own-band rows
  (`r26.rows.{head}_own_band.{side}.seed_{s}`, every recounted value within 1e-12; Table I as printed). A missing
  reference fails the gate.
- **E1:** on each 30/70 assignment, the mean over the three models of FPR and mean lead equals R27d's saved mean
  curve at alpha = 0.05 (`r27d.splits[j].mean_curves`, Fig. 2's points; 1e-12).
- **E2:** every assignment is a 30/70 partition of the calibration successes on both sides, with the same reference
  rollouts on both.

Any failure prints `R27e FAILED`, and no R27e number is used.

## Reading (fixed now)

1. **Table I.** The three own-band rows become the R27e cells: the median over the 20 assignments of the mean over the
   three models, ± the median of their std, for bACC (alarm at the limit = miss) and mean lead, on seen and unseen
   tasks. By E1, each lead cell is Fig. 2's own-band dot. The caption adds "Own bands: medians over 20 random 30/70
   calibration splits", and bold / underline are recomputed from the printed values.
2. **Sec. VI-B, "learned detectors catch at most 68% of XR-1's seen-task failures before the limit; … (bACC 0.95 and
   0.99, against at most 0.81)".** Both numbers stay if no own band's cell for recall before the limit on seen tasks
   exceeds 0.681 (Mahalanobis, the largest now) and no own band's bACC cell exceeds 0.814 (Mahalanobis); otherwise
   they become the largest values.
3. **Sec. IV-A** points to "(Figs. 1 and 2, Tables I and II)". The fixed 50/50 split then appears nowhere in the paper.

Applied with these, independent of the run: Sec. VI-A's "own bands are held fixed (refitted on seen tasks)" is
deleted (it described own-band CIs the paper no longer reports), and FIPER's interval is labeled "95% CI".

## Code

- `scripts/xr1_lead_time/r27e_table1_bands.py`;
- hook: `seen_unseen_tables.py --r27e --r27e-r26 --r27e-r27d [--r27e-splits]` (needs `--safe-lstm-fold-dirs`); the
  single models' scores are attached by `own_band_seed_keys`, now shared with R27d's hook (the same code, moved);
- `tests/test_r27e_table1_bands.py` (synthetic folds: the band alone equals R21's evaluate, the gates pass on
  consistent references and fail on perturbed or missing ones, the summary takes medians).

## Runbook (CPU; one run, XR-1 only)

Read `docs/cluster_experiment_runbook.md` first.

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=1:00:00
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
export R27E_SHA=<full SHA of the pushed commit containing this plan>
export R27E_REPO=$PROJECT_FS/robocasa_r27e_${R27E_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R27E_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R27E_REPO $R27E_SHA
test "$(git -C $R27E_REPO rev-parse HEAD)" = "$R27E_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R27E_EV=$R27E_REPO/reports
export R27E_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r27e_table1_bands_$(date +%Y%m%d_%H%M%S)
export R27E_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R27E_ROOT)
mkdir -p $R27E_ROOT $R27E_LOG
export XR1_FOLDS=$R27E_REPO/scripts/xr1_lead_time/cross_task_folds.json
echo "FOLDS=$FOLDS"
sha256sum $R27E_EV/r26_alarms_at_limit_evidence/seen_unseen_tables_r26.json \
  $R27E_EV/r27d_split_curves_evidence/seen_unseen_tables_r27d.json | cut -c1-8
cd $R27E_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r27e_table1_bands tests.test_r27d_split_curves tests.test_r27c_split_all_bands \
  tests.test_xr1_seen_unseen_tables 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- no `STOP` line prints;
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes are, in order: `4700d436` (R26), `11abe1ab` (R27d);
- the tests end with `OK`.

```bash
cd $R27E_REPO
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
  --r27e --r27e-r26 $R27E_EV/r26_alarms_at_limit_evidence/seen_unseen_tables_r26.json \
  --r27e-r27d $R27E_EV/r27d_split_curves_evidence/seen_unseen_tables_r27d.json \
  --output $R27E_ROOT/seen_unseen_tables_r27e.json' > $R27E_LOG/r27e_xr1.log 2>&1 &
echo "started; R27E_ROOT=$R27E_ROOT"
```

**Runtime:** 21 designs (the fixed split and 20 assignments), each with three heads, two sides and three single
models at one target: 378 band fits, about 1/19 of R27d's curve work. Expect a few minutes after loading.

**Progress:** `grep "R27e split" $R27E_LOG/r27e_xr1.log | tail -n 1`. When the log ends:

```bash
grep -E "R27e OK|R27e FAILED|REPRODUCTION FAILED|Traceback" $R27E_LOG/r27e_xr1.log
sed -n '/^R27e XR-1/,$p' $R27E_LOG/r27e_xr1.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R27E_ROOT/evidence && cp $R27E_ROOT/*.json $R27E_LOG/*.log $R27E_ROOT/evidence/ && ls $R27E_ROOT/evidence
```

The log must end with `R27e OK`. Then, on the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r27e_table1_bands_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r27e_table1_bands_evidence/
```

# R23: the test of H3 under H1's rule

Branch `claude/r22-fixed-paths`. Written 2026-11-19, before the run.

## Why

The paper now judges H1 by one condition: the 95% CI of Δℓ̄ (the gain over the timer at matched false
alarms, matched exactly with `hk_free.matched_timer`) lies above zero. H3 was registered with a stricter
rule: 97.5% CIs at the same target and at matched false alarms, with P matched on a margin grid, plus
null-fusion and FPR checks. Its registered result is 0.113 (97.5% CI [0.049, 0.141]).

The author's decision (2026-11-19) is to judge H3 by H1's rule, for one rule across the paper. That
needs two numbers that E1b did not store:
- Δℓ̄ with exact matching. R22's gate K3 already computed the point estimate: 0.1108.
- Its 95% CI, from the bootstrap Table III uses for XR-1's ensembles. E1b saved only its 97.5%
  intervals, not the replicates.

Nothing else changes. The path is H3's registered frozen path (u 0.2301, s 5), and the margin is
certified on the new calibration successes, as in E1b. The model and the rollouts are the same.

## Rule (fixed now)

- Hide-and-Seek + P's three-seed ensemble on RLDX-1's test set, on E1b's frozen path
  (`r22_fixed_paths.frozen_row`).
- Δℓ̄ against P matched exactly to the ensemble's test false alarms.
- **CI:** `r22_fixed_paths.boot_levels`. It draws tasks, then failures within each drawn task, with
  2,000 paired resamples and seed 0, conditional on the calibrated thresholds and the matched timer.
  This is `quantile_timer.boot`, which gives Table III's XR-1 intervals; at the 95% level it reproduces
  `qt.boot` exactly (unit test). The 97.5% interval is reported too, for comparison with the
  registered one.
- **H1's rule:** H3 is supported if the 95% CI of Δℓ̄ lies above zero. The same-target gain and its
  intervals are recorded, but are not part of the rule.

## Gates

- **K1:** the false alarms (19) and the same-target gain (0.05354) equal E1b's registered result
  (≤ 1e-9, as R22's K3).
- **K2:** the false alarms, the gain and Δℓ̄ equal R22's `h3` in `rldx_e1b_r22.json` (≤ 1e-12).

Either failure prints `R23 FAILED`, and no R23 number is used.

## Expectations (written before running)

- Δℓ̄ = 0.1108, as R22's K3 information.
- The 95% CI's lower bound is above zero. The registered 97.5% interval with grid matching starts at
  0.049, and exact matching changes the point by only 0.002.

If the 95% CI included zero, the paper would report that H3 fails H1's rule, and say that it passes
its registered rule.

## Reporting (fixed now)

- **Paper:** H3's hypothesis drops "including"; the Metrics paragraph's separate H3 rule goes. The H3
  paragraph, Table III, the abstract, the contributions and the conclusion report Δℓ̄ with its 95% CI.
- **Message to sensei:** one line on the change.

## Runbook (CPU, minutes)

```bash
qrsh -g tga-shinoda -l cpu_4=1 -l h_rt=1:00:00
```

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export STORAGE_BS=/gs/bs/tga-shinoda/felid PROJECT_FS=/gs/fs/tga-shinoda/felid
source $STORAGE_BS/robocasa_logs/eval/rldx1_unseen_20261029_135501/paths.env
source $STORAGE_BS/robocasa_checkpoints/safe/rldx1_e1b_test_table_20261113_145031/session.env
export SAFE_PY=${SAFE_PY:-$STORAGE_BS/envs/vla_safe/bin/python}
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R23_SHA=<full SHA of the pushed commit containing this plan>
export R23_REPO=$PROJECT_FS/robocasa_r23_${R23_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R23_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R23_REPO $R23_SHA
test "$(git -C $R23_REPO rev-parse HEAD)" = "$R23_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
export R22_RLDX=$(ls $STORAGE_BS/robocasa_checkpoints/safe/r22b_fixed_paths_*/rldx_e1b_r22.json | tail -n 1)
export R23_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r23_h3_h1_rule_$(date +%Y%m%d_%H%M%S)
export R23_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R23_ROOT)
mkdir -p $R23_ROOT $R23_LOG
echo "R22_RLDX=$R22_RLDX FOLDS=$FOLDS"
sha256sum $R22_RLDX $E1B_ROOT/e1b_result.json | cut -c1-16
cd $R23_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_r22_fixed_paths tests.test_rldx_test_table \
  2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if all of these hold:
- `HEAD OK` prints;
- `FOLDS` ends in `rldx_folds.json`;
- the hashes are `fc776307…` (R22's RLDX-1 output; this line said `76280d1b…`, the XR-1 file's hash, until R23 ran) and `bd0aab11…` (E1b);
- the tests end with `OK`.

```bash
cd $R23_REPO
nohup bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --reference-table $R16_ROOT/rldx_e1b_test_table.json \
  --r23 --r23-reference $R22_RLDX --output $R23_ROOT/rldx_e1b_r23.json' > $R23_LOG/r23_rldx.log 2>&1 &
echo "started; R23_ROOT=$R23_ROOT"
```

When the log ends (`tail -n 3 $R23_LOG/r23_rldx.log`):

```bash
grep -E "R23 OK|R23 FAILED|Traceback" $R23_LOG/r23_rldx.log; sed -n '/^R23 RLDX-1/,$p' $R23_LOG/r23_rldx.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R23_ROOT/evidence && cp $R23_ROOT/*.json $R23_LOG/*.log $R23_ROOT/evidence/ && ls $R23_ROOT/evidence
```

On the Mac:

```bash
rsync -av 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/r23_h3_h1_rule_*/evidence/' \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/.claude/worktrees/r22-fixed-paths/reports/r23_h3_h1_rule_evidence/
```

## Result (2026-11-19, run `r23_h3_h1_rule_*`, from `e33cfedb`): `R23 OK`

The run ended with `R23 OK` and `INPUTS UNCHANGED`. Evidence is in `reports/r23_h3_h1_rule_evidence/`:
- `rldx_e1b_r23.json`, sha256 `48ecc74f…`;
- the log, gitignored and kept locally.

R23 recorded its inputs as `rldx_e1b_r22.json` `fc776307…` and E1b's `e1b_result.json` `bd0aab11…`. Both
equal the local evidence files. The runbook's hash check had named `76280d1b…` for R22's RLDX-1 file;
that is `seen_unseen_tables_r22.json`'s hash, swapped in R22's evidence list. Both lists are now
corrected. R23 read the right file (K2 below), so this does not affect the result.

**Gates.**
- **K1:** 19 false alarms and a same-target gain of 0.05354, equal to E1b's registered result
  (max |diff| 0).
- **K2:** the false alarms, the gain and Δℓ̄ equal R22's `h3` (max |diff| 0).
- R20's run gates (G1, G2a–c and the reference rows) passed as before.

**Result.** Hide-and-Seek + P (three-seed ensemble, H3's registered path) on RLDX-1's test set:

| | Point | 95% CI | 97.5% CI |
| --- | --- | --- | --- |
| Δℓ̄ (P matched exactly to 19 false alarms) | 0.1108 | [0.0553, 0.1689] | [0.0498, 0.1785] |
| Same-target gain (P alone, 27 false alarms) | 0.0535 | [−0.0027, 0.1112] | [−0.0095, 0.1207] |
| Registered (E1b, P matched on a grid) | 0.1126 | | [0.0491, 0.1410] |
| Registered same-target gain (E1b) | 0.0535 | | [0.0007, 0.1293] |

**Verdict.** Under H1's rule, H3 passes: the 95% CI of Δℓ̄ lies above zero. As expected, Δℓ̄ is
0.1108.

**One difference from the registered analysis.** The two bootstraps differ:
- E1b's resamples calibration and test rollouts within task and class, and re-sets the margin in
  each replicate.
- R23's resamples tasks, then failures, with the thresholds fixed.

Resampling tasks adds the spread between tasks. The gain comes from atomic tasks (composite:
−0.036), so the intervals widen. Δℓ̄'s intervals stay above zero either way. The same-target gain's
lower bound was 0.0007 in E1b and is −0.003 in R23 (95%), so its CI now includes zero. The rule
adopted before this run uses only Δℓ̄, and the paper no longer reports same-target gains, so the
verdict stands. The registered test, with its own bootstrap, still passes as registered.

**Paper.** Updated as planned:
- H3 reads "at matched false alarms";
- the Metrics paragraph drops H3's separate rule, and the seen-task CI sentence covers both VLAs;
- Table III, the H3 paragraph, the abstract and the contributions give 0.111 (95% CI [0.055, 0.169]).

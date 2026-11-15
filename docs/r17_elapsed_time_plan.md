# R17. The elapsed-time timer as the unseen-task comparator

Written 2026-11-15, before any run. Branch `claude/r17-elapsed-time`, cut from R16's pushed tip
(`claude/rldx-e1b-table`, `5295bf29`). Exploratory, like Table II.

## Why

Sensei's revision directions for draft v3 (2026-11-15), point 4, ask for the main unseen-task
comparator to be an elapsed-time baseline without H_k.
- **The problem with G:** on unseen tasks it is given the benchmark's time limit H_k, which the
  learned monitors are not.
- **XR-1 needs no run.** R14 (`seen_unseen_tables_r14.json`) already fits every Table II row
  against the elapsed-time timer `G_abs` on XR-1's test set. Its G rows reproduce Table II exactly.
- **RLDX-1 does.** R14's RLDX-1 rows use the recorded sets, which the paper now calls validation.
  The paper's RLDX-1 test set is E1b's, with Table II computed by R16.
- **R17** fits R16's rows against R14's timers on E1b's sets.

## Design (fixed now)

- **Code:** `rldx_test_table.py run --elapsed-time --reference-table <R16 output>`, at R17's SHA.
  It recomputes R16's table, then fits every row of `table2.calibration` (the paper's) with
  `hk_free.block`:
  - unseen tasks: G and `G_abs`, where `G_abs` alarms after one number of inferences for every task;
  - seen tasks: P and `P_mult` (P without H_k);
  - for each, single models (seeds 0–2, mean ± std) and the three-seed ensemble, at the same
    target and at matched false alarms.
- **Inputs:** R16's directories, unchanged: folds, SAFE folds, shuffle, IDs, scores and E1b's
  `e1b_result.json`.
- **No new rollout, model or score.**

## Gates (no R17 row without them)

- R16's gates G0, G1 and G2, recomputed as in R16.
- **K0:** `table2` equals R16's stored output (`rldx_e1b_test_table.json`, sha256 `9f41e264…`).
- **K1:** G through `hk_free` reproduces every unseen + G row (FPR, lead, gain) to 1e-12.
- **K5:** P through `hk_free` reproduces every seen + P row at its stored 4 decimals.
- The 54 frozen inputs of R16 are unchanged at the end.

## Expectations (written before running)

- **The elapsed-time timer alone:** less lead than G at a similar number of false alarms, as in
  R14 on both policies.
- **Hide-and-Seek** keeps the largest unseen-task gain. Its matched gain over elapsed time is at
  least its matched gain over G, as on XR-1 (0.097 against 0.048) and on RLDX-1's recorded sets
  (0.122 against 0.084).
- **SAFE-LSTM:** a few hundredths.
- **OOD rows and SAFE-MLP:** at most about 0.02.
- **Zero-input ablation:** about 0, as on XR-1 (0.000).
- **Seen tasks:** removing H_k from P (`P_mult`) changes each main row's gain by at most about
  0.01, as in R14 (≤ 0.009).

## Reporting (fixed now)

- **Table II, unseen columns:** they become "+ elapsed time" on both policies, single models
  (mean ± std), whatever they show.
  - XR-1: R14 `r14.unseen[d].G_abs.mean_std_over_seeds`.
  - RLDX-1: R17 `r17.unseen[d].G_abs.mean_std_over_seeds`.
- **Footnote:** the elapsed-time timer's lead, from `timers_alone.unseen.G_abs`.
- **Text:** the unseen-task ensembles, at the same target and matched, come from the same files
  (`seed_average`).
- **G:** still reported, in one sentence, so readers can compare.
- **Amendment (2026-11-15, still before the run):** draft v3's Table II also has a Matched column
  on each side. RLDX-1's come from R17 too:
  - seen tasks: `r17.seen[d].P.mean_std_over_seeds.matched_gain`;
  - unseen tasks: `r17.unseen[d].G_abs.mean_std_over_seeds.matched_gain`.

  The seen FPR and Δℓ stay R16's, and gate K5 ties them to `r17.seen[d].P`. The code is
  unchanged; this is a reporting rule only.

## Runbook

CPU only, a few minutes. Any allocation with 16 cores will do, e.g.
`qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=1:00:00`.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
source /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/rldx1_e1b_test_table_20261113_145031/session.env
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R17_SHA=<full SHA of the pushed commit containing this plan>
export R17_REPO=$PROJECT_FS/robocasa_r17_${R17_SHA:0:8}
export R17_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_r17_elapsed_$(date +%Y%m%d_%H%M%S)
export R17_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R17_ROOT)
mkdir -p $R17_ROOT $R17_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r17-elapsed-time
test -e $R17_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R17_REPO $R17_SHA
test "$(git -C $R17_REPO rev-parse HEAD)" = "$R17_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
cd $R17_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_rldx_test_table tests.test_xr1_hk_free 2>&1 | grep -E "^(Ran|OK|FAILED)"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --elapsed-time \
  --reference-table $R16_ROOT/rldx_e1b_test_table.json --output $R17_ROOT/rldx_e1b_r17.json 2>&1 | tee $R17_LOG/r17.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
sha256sum $R16_ROOT/rldx_e1b_test_table.json $R17_ROOT/rldx_e1b_r17.json
mkdir -p $R17_ROOT/evidence && cp $R17_ROOT/rldx_e1b_r17.json $R17_LOG/r17.log $R17_ROOT/evidence/
echo "R17_ROOT=$R17_ROOT"
```

**Continue only if all of these hold:**
- `HEAD OK` prints;
- the tests end with `OK`;
- the log shows R16's gates (G1, G0, G2a–c) `OK` and `RLDX-1 TEST TABLE OK`;
- it then shows `R17 K0: Table II equals …`, the K1/K5 line and `R17 ELAPSED-TIME OK`;
- `INPUTS UNCHANGED` prints;
- R16's table hash starts with `9f41e264`.

Then, on the Mac:

```bash
rsync -av tsubame:<R17_ROOT>/evidence/ <local worktree>/reports/r17_elapsed_time_evidence/
```

## Result

(to be written after the run)

# R22. Fixed paths everywhere, and one metric

Written 2026-11-18, before any run. Branch `claude/r22-fixed-paths`, cut from R21
(`claude/r21-native-bands`, 0f653cb0). Exploratory. This is the author's decision after sensei's
revision directions v2 and the discussion of draft v4.

## Why

- **Two designs.** The paper evaluates fused monitors in two ways:
  - H1, H2 and Table II use the *adaptive path*, chosen on the same calibration set that sets its
    margin. That voids Proposition 1 for the chosen monitor, and the expected test FPR rises to
    about 0.055.
  - H3 uses a path fixed on separate RLDX-1 validation rollouts, before its test.
- **The decision (author, 2026-11-18):**
  - Every fused monitor uses a path fixed on validation data, with its margin set on the
    calibration successes and one evaluation on the test set. Proposition 1 then covers every
    result.
  - The paper says nothing more about the adaptive path.
- **One metric (author, same day):** every gain is Δℓ̄, the gain in mean lead over the timer held
  to the monitor's test false alarms (today's "matched" gain). The same-target gain is kept only to
  restate H3's registered rule in one clause.
- **What R8 showed (XR-1, seen tasks, ensembles):**
  - Hide-and-Seek: unchanged (0.072 vs 0.071 at the same target, 0.057 vs 0.056 matched).
  - SAFE-MLP: its frozen path is conservative (16 false alarms against P's 21), and its matched gain
    rises to 0.039 (test-only CI [0.009, 0.072]).

## Rule (fixed now)

For each policy, monitor score (each seed, and the seed average for ensembles), side (seen tasks
against P; unseen tasks against E and G) and fold:

1. **Choose:** run `hk_free.fit` on the *validation* rows of that fold's calibration tasks. Its
   choices are unchanged: the path δ = λ, τ_s = u + sλ, the timer alone allowed, u from the
   validation failures' quantiles, and s from `confirm_lead.S_GRID`. For each candidate, the margin
   comes from Proposition 1 on the validation successes; the path with the largest mean lead on the
   validation failures is kept. The timer is the side's comparator (P or E). G is reported against
   E's paths.
2. **Freeze** (u, s).
3. **Certify:** λ̂ by Proposition 1 on the calibration successes, along the frozen path.
4. **Evaluate** once on the test set:
   - FPR;
   - Δℓ̄: the comparator at the smallest margin with at most the monitor's test false alarms
     (`hk_free.matched_timer`), as in Table II;
   - the same-target gain, kept for the record only.

**Validation sets:**
- **XR-1:** the paper's validation set, the August cohorts. August calibration has 3 + 3 rollouts
  per task (114 + 114) and August test 10 + 10 (380 + 380), 988 rollouts in all. R8 checked them
  disjoint from E1 (gate F2).
- **RLDX-1:** the recorded validation cohorts (R7: 100 + 100 test rollouts and 60 calibration
  successes). H3 stays exactly as registered, with its path frozen on E1's 1,600 further RLDX-1
  rollouts; it is reproduced here, not refitted.
- **Unseen folds:** the validation rows of the fold's seen tasks only, scored by that fold's models
  with pooled normalization. The held-out tasks' validation rows are never read.

**Unchanged** (no path): Table I, Fig. 2, the timer G rows, R21's band rows and FIPER.

## Scores needed, and what exists

| Policy, side | Score | Validation scores |
|---|---|---|
| XR-1 seen | SAFE-MLP (3 seeds) | exist: the frozen final models on the August cohorts (R8's inputs, `$XR1_PS_INPUTS`) |
| XR-1 seen | Hide-and-Seek, zero input, first step, shuffle (3 seeds) | exist: Amendment 2's `august_*_predictions.jsonl` (R8's inputs) |
| XR-1 seen | OOD ×4 (3 seeds, per-task) | exist as the OOD per-seed `calibrate` / `evaluate` outputs on `cache_calibration` / `cache_prospective_test` (the August cohorts); to be confirmed in A5 |
| XR-1 seen | SAFE-LSTM (fold all) | **A2** |
| XR-1 unseen | Hide-and-Seek, zero input, first step (folds 0–3) | **A1** |
| XR-1 unseen | SAFE-LSTM (folds 0–3) | **A2** |
| XR-1 unseen | OOD ×4 (folds 0–3) | **A3** |
| XR-1 unseen | SAFE-MLP (folds 0–3) | **A4** |
| RLDX-1 | every score, seen and unseen | exist: R7's recorded cohorts, reproduced by R16's fold directories (gate G1) |

## Stage A: score XR-1's validation set (cluster)

Every job adds the two August cohorts to an existing scoring run. Each one must also reproduce the
stored E1-cohort trajectories of that run; that is how the retrained or reloaded models are shown to
be the same.

- **A1 (CPU, several hours per fold):** `score_cross_task` for folds 0–3 with Hide-and-Seek, zero
  input and first step.
  - Cohorts: `--cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json` and
    `--cohort august_test $PSEED_ROOT/august_test_balanced_ids.json`, as R6 C2 did for ActProbe's
    fold all.
  - Gate: `--reference-raw` on both E1 cohorts against `$XT_ROOT/fold_F`.
- **A2 (CPU):** `score_cross_task --detectors safe_lstm` for folds 0–3 and all, with the same
  cohorts. Gate: `--reference-raw` against `$R11_ROOT/safe_lstm/fold_F`.
- **A3 (CPU, minutes):** `run_offline_detector_baselines score-fold` for folds 0–3, with R6's
  bundles (`$R6_ROOT/ood/develop/fold_F/seed{0,1,2}`).
  - Cohorts: the two E1 cohorts and the August ones (`cache_calibration`, `cache_prospective_test`).
  - Gate: the E1-cohort rows equal `$R6_ROOT/ood/fold_F`.
- **A4 (GPU):** `score_seen_checkpoint_external` with the saved fold checkpoints
  (`$RS_ROOT/train/fold_F/indep_seed{0,1,2}`) on the August SAFE exports
  (`$XR1_AUGUST_ROOT`, to be located).
  - `cross_task_safe assemble` is extended to accept the two August cohorts as optional extra
    cohorts.
  - Gate: the E1-cohort rows equal `$RS_ROOT/folds/fold_F`.
- **A5 (CPU, minutes):** confirm, or produce, the per-seed seen-task OOD scores on the August
  cohorts with the seed bundles, gated on reproducing the stored E1-cohort files.

## Stage B: the fixed-path analysis (cluster, CPU)

Code: `scripts/xr1_lead_time/r22_fixed_paths.py`, called with `--r22` by `seen_unseen_tables.py`
(XR-1, with the stage A directories) and `rldx_test_table.py` (RLDX-1, validation = recorded
cohorts).

**Outputs** (everything the paper quotes about fused monitors):
- **Table II:** every constant-threshold row on both policies and both sides (OOD ×4, SAFE-MLP,
  SAFE-LSTM, Hide-and-Seek, zero input, first step, shuffle on seen tasks only). Per seed: FPR, Δℓ̄
  and the chosen path; then mean ± std.
- **Table III (XR-1 rows):** ensembles + P (SAFE-MLP, Hide-and-Seek, Mahalanobis as one model, zero
  input, first step, shuffle).
  - False alarms and Δℓ̄, with Table III's CI (tasks, then failures; 2,000 paired resamples,
    conditional on the fitted thresholds and the matched timer).
  - The same-target gain, for the record.
- **Unseen ensembles:** Hide-and-Seek, SAFE-MLP and SAFE-LSTM against E and G, Δℓ̄ with R19's
  full-pipeline bootstrap.
  - Bootstrap: 1,000 replicates, seed 22; tasks, then calibration and test rollouts within task and
    class.
  - In each replicate the path stays frozen, while λ̂ and the timer are refit and the timer is
    re-matched.
- **H2 nulls:** 100 random-noise and 100 within-task resampled fusions per policy's seen side.
  - Each null is built on the validation rows the same way as on the test rows, gets its own frozen
    path, and is evaluated like the monitors.
  - Reported: Hide-and-Seek's and SAFE-MLP's ranks among them, at Δℓ̄.
- **Per task:** Δℓ̄ for atomic and composite tasks (SAFE-MLP, Hide-and-Seek, OOD), and the
  top-5 share.
- **Budget-allocated timer:** per-task margins allocated on the validation set, certified on the
  calibration successes; Hide-and-Seek + P's Δℓ̄ against it.
- **Fig. 1:** the frozen Hide-and-Seek + P path's τ_s and timer alarm on rollout
  `d5be410e4f86cbc95d554fe7`.

## Gates

- **A0 (stage A):** every rerun reproduces its stored E1-cohort trajectories (bit for bit, or
  ≤ 1e-12 where the stored run was compared that way). The August rows are new.
- **V0:** validation, calibration and test rollouts are disjoint (rollout IDs), on both policies.
- **K0:**
  - With the selection switched back to the calibration set (adaptive), the code reproduces every
    stored adaptive row: R14 (XR-1) and R17 (RLDX-1), Table II and III, ≤ 1e-12.
  - This shows that only the selection data change.
- **K8:** on XR-1's seen tasks, the fixed-path SAFE-MLP and Hide-and-Seek ensembles reproduce R8's
  chosen paths and same-target gains at R8's precision.
- **K3:** the RLDX-1 H3 row reproduces E1b's registered result with its frozen path (19 false
  alarms, 0.054, 0.113).
- **N0:** every certified λ̂ equals Proposition 1 on the calibration successes along the frozen path.
- **The scripts' own gates,** and the frozen inputs are unchanged.

## Expectations (written before running)

- **Hide-and-Seek:** about as before on both policies (R8: unchanged on XR-1).
- **SAFE-MLP:** its seen-task Δℓ̄ on XR-1 rises toward R8's 0.039, because its frozen path is
  conservative. Elsewhere, small changes.
- **FPRs:** test FPRs near or below 0.05, without the adaptive inflation to about 0.055.
- **Unseen ensembles:** Hide-and-Seek's gain over E stays positive. Its CIs may widen, since the
  path no longer adapts in each replicate.

## Reporting (fixed now)

- **Source:** every fused number in the paper comes from R22.
- **Method text:** the adaptive path is removed. Sec. IV-C says that each path is chosen on the
  validation set and frozen, and that its margin comes from Proposition 1 on the calibration
  successes.
- **Metric:** every gain is Δℓ̄ (matched), defined once. H3's same-target condition is restated in
  one clause.
- **Completeness:** reported whatever it shows, including SAFE-MLP's conservative-path effect,
  which the text explains.

## Runbook

Stage A first; stage B after its gates hold. A1 is the longest job (it retrains the Hide-and-Seek
family per fold, several hours on CPU) and its inputs are known, so it starts first. The discovery
block (D) prints what A2–A5 need. A2–A5 commands follow once D's output is back.

### Setup (one CPU allocation with at least 32 cores, for A1's four folds × 8 threads)

```bash
qrsh -g tga-shinoda -l cpu_80=1 -l h_rt=12:00:00
```

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export STORAGE_BS=/gs/bs/tga-shinoda/felid
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r11_rows_20261030_220636/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r14_hk_free_20261106_191531/paths.env
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export R22_SHA=<full SHA of the pushed commit containing this runbook>
export R22_REPO=$PROJECT_FS/robocasa_r22_${R22_SHA:0:8}
export R22A_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/r22a_validation_scores_$(date +%Y%m%d_%H%M%S)
export R22A_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R22A_ROOT)
mkdir -p $R22A_ROOT $R22A_LOG
git -C $PROJECT_FS/robocasa fetch origin claude/r22-fixed-paths
test -e $R22_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R22_REPO $R22_SHA
test "$(git -C $R22_REPO rev-parse HEAD)" = "$R22_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
printf 'export %s=%q\n' R22_SHA "$R22_SHA" R22_REPO "$R22_REPO" R22A_ROOT "$R22A_ROOT" R22A_LOG "$R22A_LOG" \
  > $R22A_LOG/paths.env
export FOLDS=$R22_REPO/scripts/xr1_lead_time/cross_task_folds.json
cmp $FOLDS $R14_REPO/scripts/xr1_lead_time/cross_task_folds.json && echo "FOLDS SAME AS R14"
for f in $E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl $XR1_RUNTIME_BUNDLE $XR1_TRAIN_IDS \
         $AMEND2_ROOT/august_calibration_ids.json $PSEED_ROOT/august_test_balanced_ids.json \
         $XT_ROOT/fold_{0,1,2,3}/lead_calibration_predictions.jsonl $XT_ROOT/fold_{0,1,2,3}/confirmatory_test_predictions.jsonl; do
  test -s $f && echo "OK $f" || echo "STOP: missing $f"
done | grep -c '^OK'
echo "XR1_RAW_DIRS: ${#XR1_RAW_DIRS[@]} directories"
cd $R22_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_safe_score_cross_task 2>&1 | grep -E "^(Ran|OK|FAILED)"
nproc; free -g | head -n 2
```

Continue only if `HEAD OK` and `FOLDS SAME AS R14` print, the count is 14, `XR1_RAW_DIRS` is
nonempty, and the tests end with `OK`.

### A1. Hide-and-Seek family on the August cohorts, folds 0–3 (CPU, background)

Each fold retrains its three detectors × three seeds exactly as the stored run did (same training-ID
order, seeds, epochs, 8 threads). It scores the two E1 cohorts and the two August cohorts, and
checks the E1 cohorts against the stored fold directory (`--reference-raw`, ≤ 1e-12).

```bash
cd $R22_REPO
A1_ARGS=(
  --dataset-dir "${XR1_RAW_DIRS[@]}" $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained
  --runtime-bundle $XR1_RUNTIME_BUNDLE --train-ids $XR1_TRAIN_IDS --folds $FOLDS
  --cohort lead_calibration $E1_OUT/calibration_predictions.jsonl
  --cohort confirmatory_test $E1_OUT/test_predictions.jsonl
  --cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json
  --cohort august_test $PSEED_ROOT/august_test_balanced_ids.json
  --detectors hide_and_seek hide_and_seek_zero hide_and_seek_first --feature-aggregation last --seeds 0 1 2
)
for F in 0 1 2 3; do
  PYTHONPATH=$R22_REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 \
    "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task "${A1_ARGS[@]}" --fold $F \
    --reference-raw lead_calibration $XT_ROOT/fold_$F/lead_calibration_predictions.jsonl \
    --reference-raw confirmatory_test $XT_ROOT/fold_$F/confirmatory_test_predictions.jsonl \
    --output-dir $R22A_ROOT/xt/fold_$F > $R22A_LOG/a1_fold_$F.log 2>&1 &
  echo "A1 fold $F PID $! on $(hostname)"
done
```

**Progress:** `tail -qn 1 $R22A_LOG/a1_fold_*.log`. Each log must end with six
`reference ... -> OK` lines (two cohorts × three detectors) and `REFERENCE OK`.
`REFERENCE MISMATCH`, `Traceback` or `error` is a stop condition.

### D. Discovery for A2–A5 (seconds; run while A1 runs, and paste the output back)

```bash
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
source $STORAGE_BS/robocasa_logs/eval/xr1_r11_rows_20261030_220636/paths.env
echo "== envs"; env | grep -E '^(XR1_[A-Z_]*(EXPORT|ROOT|INPUTS)|RS_ROOT|R11_ROOT|R6_ROOT|OOD_ROOT|XR1_BASELINE_ROOT|SAFE_REPO)=' | sort
echo "== August SAFE scores used by R8"; head -c 600 $XR1_PS_INPUTS/calibration_predictions.jsonl | tr ',' '\n' | grep -E 'rollout_id|task_name|per_seed_raw_safe' | head -3
wc -l $XR1_PS_INPUTS/calibration_predictions.jsonl $XR1_PS_INPUTS/event_predictions.jsonl
find $STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_20260815_163700* $STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700* \
  -maxdepth 4 -name provenance.json 2>/dev/null | head -10
for p in $(find $STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_20260815_163700* $STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700* -maxdepth 4 -name provenance.json 2>/dev/null | head -6); do
  echo "-- $p"; grep -oE '"(export_dir|group|training_run_dir)": "[^"]*"' $p | head -4; done
echo "== OOD caches"; for c in calibration prospective_test; do echo "-- cache_$c"; grep -oE '"(group|n_rollouts|rollouts)": [^,}]*' $XR1_BASELINE_ROOT/cache_$c/provenance.json | head -4; done
"$SAFE_PY" -I -c "import json,sys; ids=json.load(open(sys.argv[1])); ids=ids.get('rollout_ids', ids) if isinstance(ids, dict) else ids; print('august_test_balanced ids', len(ids))" $PSEED_ROOT/august_test_balanced_ids.json
"$SAFE_PY" -I -c "import json,sys; ids=json.load(open(sys.argv[1])); ids=ids.get('rollout_ids', ids) if isinstance(ids, dict) else ids; print('august_calibration ids', len(ids))" $AMEND2_ROOT/august_calibration_ids.json
echo "== OOD seen per-seed outputs"; ls $OOD_ROOT/seed0/ $OOD_ROOT/seed0/calibrate $OOD_ROOT/seed0/evaluate 2>&1 | head -30
echo "== SAFE fold checkpoints"; ls -d $RS_ROOT/train/fold_*/indep_seed* 2>&1 | head -12; ls -d $R11_ROOT/train/fold_*/* 2>&1 | head -16
ls $RS_ROOT/score/fold_0/ $R11_ROOT/score/fold_0/ 2>&1 | head
```

## Result

(to be written after the runs)

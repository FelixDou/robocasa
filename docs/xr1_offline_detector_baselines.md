# Offline baselines for the frozen Xiaomi staged SAFE + time detector

This runbook adds standard failure-detection baselines to the completed
prospective matched-FPR study in
[`xr1_prospective_matched_fpr.md`](xr1_prospective_matched_fpr.md). It uses
only data that was already recorded: no new rollouts, no policy server, and no
GPU. The frozen runtime bundle, its thresholds, and the reported prospective
numbers are read but never modified. The evaluator refuses to report anything
unless it first reproduces the frozen calibration thresholds and the published
prospective results exactly.

Read [`cluster_experiment_runbook.md`](cluster_experiment_runbook.md) first.

## 1. What is compared

Each detector produces one causal score per genuine XR-1 policy inference. The
score at inference `t` uses only inputs available through `t`.

| Detector | Score | Threshold(s) |
|---|---|---|
| `time_only`, `safe_only`, `staged_safe_time` | frozen (unchanged) | frozen (reproduced) |
| `fused_safe_time` | normalized SAFE OR time risk, both eligible for the whole rollout (staging ablation) | joint pair |
| `<baseline>_only` | the baseline alone over the whole rollout | single |
| `staged_<baseline>_time` | the baseline in the 0–25% window, time fallback from 50% (the frozen staged wrapper) | joint pair |
| `fused_<baseline>_time` | the baseline OR time risk, both eligible for the whole rollout (section 10 extension) | joint pair |

The `staged_<baseline>_time` rows separate the contribution of the SAFE
representation from that of the staging scheme: they keep the wrapper fixed
and swap only the early-window score.

Baselines, all fitted on **successful** development rollouts only:

| Baseline | Definition | Development grid |
|---|---|---|
| `mahalanobis` | per-task Gaussian (Ledoit-Wolf shrinkage) in a pooled PCA space of standardized features | PCA dim 16, 64, 256 |
| `knn_cosine` | mean cosine distance to the `k` nearest successful inferences | `k` 1, 5, 10, 25; per-task or pooled reference |
| `pca_kmeans` | distance to the nearest KMeans centroid in pooled PCA space | PCA dim 16, 64; 8 or 32 clusters; per-task or pooled |
| `rnd` | feature-space random network distillation: a trained MLP predicts a fixed random 2-layer network | hidden/output 128/32 or 512/128 |
| `action_inconsistency` | official SAFE `compute_stac_metrics_single`: mean L2 distance between the unexecuted tail of chunk `t-1` and the matching prefix of chunk `t` | none |

The embedding baselines follow the `use_success_only` variants of the official
SAFE `embed` model. `rnd` is a feature-only adaptation; the official SAFE RND
uses an action-conditioned 1-D UNet and is not reproduced here.
`action_inconsistency` requires `pred_horizon > exec_horizon`. The extractor
checks this and records it, and `develop` skips the baseline with the reason
when it fails. **For XR-1 it fails.** The released RoboCasa365 checkpoint
predicts 16-step chunks (`actions` `(16, 12)`, `pre_velocity` `(5, 16, 1024)`)
and replans every 16 steps. Each chunk is executed in full, so consecutive
chunks share no predicted step and STAC-Single is undefined. This was verified
on the calibration export on 17 October 2026. The XR-1 suite therefore
evaluates 12 detectors; the baseline remains available for policies whose
chunks overlap.

All embedding baselines read exactly the per-inference vector that the frozen
SAFE MLP reads: the official SAFE loader with the frozen configuration
(`horizon_idx_rel=1.0`, `diff_idx_rel=1.0`, i.e. the last flow step and the
last of the 16 action positions of the pre-`action_output_layer` DiT state).
The pinned loader selects that position from the feature tensor's own shape
and sets the configuration's `pred_horizon` from the exported `actions`; it
never truncates actions or features. Every baseline
also chooses a causal temporal aggregation (running maximum, running mean, or
the official cumulative sum) during development.

## 2. Protocol

1. **Development** reads only the 1,824 outer-training rollouts of
   `xr1_pooled_39tasks_final_refits_20260814_231508`. The frozen split manifest
   is taken from the runtime bundle's checksummed copy. Each task/outcome group
   is divided deterministically into a reference-fit partition (70%) and a
   meta-validation partition (30%) before any fitting. Baselines are fitted on
   reference-fit successes. Fit-partition failures are unused. Hyperparameters
   and aggregation maximize the mean task-macro at-risk ROC-AUC at the 10% and
   25% task-horizon landmarks on meta-validation (the early window in which a
   staged baseline acts). Ties break by task-macro event ROC-AUC, then by grid
   order. Task normalization (median and robust scale of each rollout's
   maximum, the same form as SAFE's) is fitted on meta-validation. Development
   never reads calibration or prospective outcomes.
2. **Calibration** uses the frozen 228-rollout primary calibration cohort, the
   frozen horizons and time-risk curves, the same event-level `FPR <= 0.05`
   cap, and the same ranking (recall, then earlier miss-adjusted detection,
   then the more conservative threshold). It first re-selects the three frozen
   detectors and stops unless their thresholds match the runtime bundle
   exactly.
3. **Evaluation** runs once on the 760-rollout primary prospective cohort,
   with the 1,643-rollout all-retained cohort as a secondary result. It stops
   unless the frozen `overall`, `all_retained_secondary`,
   `paired_staged_minus_time`, and `task_rollout_bootstrap` values are
   reproduced. It then reports, for every detector, TPR/FPR with Wilson
   intervals, balanced accuracy, ROC/AUPRC, miss-adjusted detection, recall at
   the 10/25/50% landmarks, atomic/composite strata, and calibration-to-test
   FPR drift. Each detector is also paired against both `time_only` and
   `staged_safe_time`, with a task/rollout bootstrap using the frozen
   resampling sequence (2,000 replicates, seed 0).

The prospective test was already opened for the frozen detectors. Baseline
results are a single post-hoc evaluation of detectors frozen before scoring
it. They do not change the frozen SAFE result.

## 3. Fresh session, code, and preflight

The suite is CPU-only. Use one allocation with ample RAM; extraction is
I/O-bound and reads the materialized official exports.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export SAFE_REPO="$PROJECT_FS/SAFE"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"
export SAFE_PY="$SAFE_ENV/bin/python"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0

export XR1_MATCHED_FPR_TAG=xr1_prospective_matched_fpr_20260815_163700
export XR1_CAL_ANALYSIS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_MATCHED_FPR_TAG}_calibration"
export XR1_TEST_ANALYSIS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_MATCHED_FPR_TAG}_prospective_test"
export XR1_RUNTIME_BUNDLE="$XR1_CAL_ANALYSIS/runtime_bundle.json"
export XR1_FROZEN_ANALYSIS="$XR1_TEST_ANALYSIS/analysis.json"

cd "$ROBOCASA_REPO"
git pull origin main
echo "code commit: $(git rev-parse HEAD)"
test -f robocasa/recovery/safe/run_offline_detector_baselines.py
test -f robocasa/recovery/safe/offline_detector_baselines.py
test -x "$SAFE_PY"
test "$(git -C "$SAFE_REPO" rev-parse HEAD)" = b6036abe07b2b2bb9996afb2c07f13d6a9f507c0
test -f "$XR1_RUNTIME_BUNDLE"
test -f "$XR1_CAL_ANALYSIS/calibration_predictions.jsonl"
test -f "$XR1_FROZEN_ANALYSIS"
test -f "$XR1_TEST_ANALYSIS/event_predictions.jsonl"
df -h "$STORAGE_BS"
df -ih "$STORAGE_BS"
```

Resolve the frozen SAFE configuration and the three official exports from the
frozen provenance rather than retyping them:

```bash
export XR1_BASELINE_TAG=xr1_offline_baselines_$(date +%Y%m%d_%H%M%S)
export XR1_BASELINE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$XR1_BASELINE_TAG"
export XR1_BASELINE_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval/$XR1_BASELINE_TAG"
mkdir -p "$XR1_BASELINE_ROOT" "$XR1_BASELINE_LOG_ROOT"

"$SAFE_PY" - "$XR1_RUNTIME_BUNDLE" "$XR1_FROZEN_ANALYSIS" > "$XR1_BASELINE_ROOT/paths.env" <<'PY'
import json
import pathlib
import shlex
import sys

from omegaconf import OmegaConf

runtime = json.loads(pathlib.Path(sys.argv[1]).read_text())
analysis = json.loads(pathlib.Path(sys.argv[2]).read_text())
seed = str(sorted(int(value) for value in runtime["model_seeds"])[0])
config = runtime["checkpoint_bundle"][seed]["config.yaml"]["path"]
cfg = OmegaConf.load(config)
prefix = str(cfg.dataset.get("data_path_prefix") or "")
values = {
    "XR1_SAFE_CONFIG": config,
    "XR1_DEV_EXPORT": str(pathlib.Path(prefix) / str(cfg.dataset.data_path)),
    "XR1_CAL_EXPORT": runtime["external_score_provenance"][seed]["export_dir"],
    "XR1_TEST_EXPORT": analysis["external_score_provenance"][seed]["export_dir"],
}
assert str(cfg.dataset.horizon_idx_rel) == "1.0" and str(cfg.dataset.diff_idx_rel) == "1.0"
for key, value in values.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY

source "$XR1_BASELINE_ROOT/paths.env"
cat "$XR1_BASELINE_ROOT/paths.env"
for export_dir in "$XR1_DEV_EXPORT" "$XR1_CAL_EXPORT" "$XR1_TEST_EXPORT"; do
  test -f "$export_dir/conversion_report.json" || echo "STOP: missing export $export_dir"
done

printf 'export XR1_BASELINE_TAG=%q\nexport XR1_BASELINE_ROOT=%q\nexport XR1_BASELINE_LOG_ROOT=%q\nsource %q\n' \
  "$XR1_BASELINE_TAG" "$XR1_BASELINE_ROOT" "$XR1_BASELINE_LOG_ROOT" "$XR1_BASELINE_ROOT/paths.env" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_offline_baselines_latest.env"
```

In a later terminal, rerun the export block above (through
`XR1_FROZEN_ANALYSIS`) and then
`source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_offline_baselines_latest.env"`.

## 4. Smoke tests (run first)

The unit tests and the synthetic end-to-end smoke need no recorded data. The
synthetic smoke builds a miniature frozen protocol with the real
`run_prospective_matched_fpr` phases, then runs `develop`, `calibrate`, and
`evaluate`, including the exact-reproduction checks:

```bash
cd "$ROBOCASA_REPO"
"$SAFE_PY" -m unittest tests.test_safe_offline_detector_baselines tests.test_safe_prospective_matched_fpr

"$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines synthetic-smoke \
  --output-dir "$XR1_BASELINE_ROOT/synthetic_smoke"
```

Continue only if all tests pass and the smoke prints
`"frozen_results_reproduced": true` with 14 detectors. The synthetic data has
overlapping chunks, so it includes the two action-baseline detectors that the
XR-1 run skips.

The first real-data smoke is the calibration extraction, the smallest export
(454 rollouts):

```bash
cd "$ROBOCASA_REPO"
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines extract \
  --export-dir "$XR1_CAL_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --safe-config "$XR1_SAFE_CONFIG" \
  --group calibration \
  --output-dir "$XR1_BASELINE_ROOT/cache_calibration" \
  > "$XR1_BASELINE_LOG_ROOT/extract_calibration.log" 2>&1 &
echo "extract calibration PID: $!"
```

When it exits, inspect its provenance:

```bash
tail -n 40 "$XR1_BASELINE_LOG_ROOT/extract_calibration.log"
"$SAFE_PY" - "$XR1_BASELINE_ROOT/cache_calibration/provenance.json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
for key in ("group", "rollouts", "successes", "failures", "inferences", "feature_dim",
            "pred_horizon", "dim_action", "exec_horizon", "stac_overlap_positions",
            "stac_supported", "irregular_inference_step_gaps", "horizon_selector",
            "diffusion_selector"):
    print(f"{key}: {d[key]}")
print("tasks:", len(d["tasks"]))
PY
```

Expected: 454 rollouts (254 successes, 200 failures), 38 tasks, 38,403
inferences, `feature_dim` 1024, `pred_horizon` 16, `dim_action` 12,
`exec_horizon` 16, `stac_overlap_positions` 0, `stac_supported: False`, and
`irregular_inference_step_gaps` 0 (all observed on 17 October 2026).
`develop` will then report `action_inconsistency` under `skipped`.

## 5. Extract the development and prospective caches

```bash
cd "$ROBOCASA_REPO"
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines extract \
  --export-dir "$XR1_DEV_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --safe-config "$XR1_SAFE_CONFIG" \
  --group development \
  --output-dir "$XR1_BASELINE_ROOT/cache_development" \
  > "$XR1_BASELINE_LOG_ROOT/extract_development.log" 2>&1 &
echo "extract development PID: $!"

nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines extract \
  --export-dir "$XR1_TEST_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --safe-config "$XR1_SAFE_CONFIG" \
  --group prospective_test \
  --output-dir "$XR1_BASELINE_ROOT/cache_prospective_test" \
  > "$XR1_BASELINE_LOG_ROOT/extract_prospective_test.log" 2>&1 &
echo "extract prospective PID: $!"
```

Expected rollout counts: 1,950 development rollouts (all 39 exported tasks;
`develop` keeps only the frozen 1,824 outer-training rollouts) and 1,643
prospective rollouts. Each cache is roughly 0.3–1 GB.

## 6. Development: smoke grid, then full grid

The smoke grid fits one configuration per baseline. It reads development
outcomes only:

```bash
cd "$ROBOCASA_REPO"
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop \
  --development-cache "$XR1_BASELINE_ROOT/cache_development" \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" \
  --output-dir "$XR1_BASELINE_ROOT/develop_smoke" \
  --grid smoke \
  > "$XR1_BASELINE_LOG_ROOT/develop_smoke.log" 2>&1 &
echo "develop smoke PID: $!"
```

Optionally check the real calibration inputs end to end with the smoke bundle.
This reproduces the frozen thresholds on the real cohort and opens no
prospective outcome:

```bash
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines calibrate \
  --baseline-bundle "$XR1_BASELINE_ROOT/develop_smoke/baseline_bundle.json" \
  --calibration-cache "$XR1_BASELINE_ROOT/cache_calibration" \
  --output-dir "$XR1_BASELINE_ROOT/calibrate_smoke" \
  --workers 8 \
  > "$XR1_BASELINE_LOG_ROOT/calibrate_smoke.log" 2>&1 &
echo "calibrate smoke PID: $!"
```

Never evaluate a smoke bundle. Then run the full development grid:

```bash
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop \
  --development-cache "$XR1_BASELINE_ROOT/cache_development" \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" \
  --output-dir "$XR1_BASELINE_ROOT/develop" \
  --grid full \
  > "$XR1_BASELINE_LOG_ROOT/develop.log" 2>&1 &
echo "develop PID: $!"
```

Inspect `develop/status.json` and `develop/selection_audit.csv`. The bundle
records the fit/validation partitions, the selected configuration and
aggregation, the meta-validation selection score, task normalization, and model
checksums. A missing baseline in `selected` must be explained in `skipped`.

## 7. Calibration

```bash
cd "$ROBOCASA_REPO"
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines calibrate \
  --baseline-bundle "$XR1_BASELINE_ROOT/develop/baseline_bundle.json" \
  --calibration-cache "$XR1_BASELINE_ROOT/cache_calibration" \
  --output-dir "$XR1_BASELINE_ROOT/calibrate" \
  --workers 8 \
  > "$XR1_BASELINE_LOG_ROOT/calibrate.log" 2>&1 &
echo "calibrate PID: $!"
```

The joint searches dominate run time. Continue only if `calibrate/status.json`
reports `frozen_thresholds_reproduced: true` and every detector has
calibration FPR at most 0.05. Do not edit thresholds afterwards.

## 8. One prospective evaluation

```bash
cd "$ROBOCASA_REPO"
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines evaluate \
  --baseline-calibration "$XR1_BASELINE_ROOT/calibrate/baseline_calibration.json" \
  --test-cache "$XR1_BASELINE_ROOT/cache_prospective_test" \
  --frozen-analysis "$XR1_FROZEN_ANALYSIS" \
  --output-dir "$XR1_BASELINE_ROOT/evaluate" \
  --test-per-class 10 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 0 \
  > "$XR1_BASELINE_LOG_ROOT/evaluate.log" 2>&1 &
echo "evaluate PID: $!"
```

A `Frozen prospective results were not reproduced` error is a stop condition:
inspect the inputs and do not change the frozen arguments to make it pass.

Compact result audit:

```bash
"$SAFE_PY" - "$XR1_BASELINE_ROOT/evaluate/analysis.json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("status:", d["status"], "| frozen reproduced:", d["frozen_results_reproduced"])
print("counts:", d["counts"]["primary_balanced"], "| skipped:", d["skipped_baselines"])
print("selected:", json.dumps(d["baseline_selection"], indent=1, sort_keys=True))
header = f"{'detector':34s} {'TPR':>6s} {'FPR':>6s} {'BA':>6s} {'adj':>6s} {'r@25':>6s}  d_adj vs staged SAFE [95% CI]"
print(header)
for name, row in d["overall"].items():
    boot = d["task_rollout_bootstrap"].get(f"{name}__vs__staged_safe_time")
    delta = "" if boot is None else (
        f"{boot['adjusted_detection_delta']['mean']:+.3f} "
        f"[{boot['adjusted_detection_delta']['ci95'][0]:+.3f}, {boot['adjusted_detection_delta']['ci95'][1]:+.3f}]"
    )
    print(f"{name:34s} {row['true_positive_rate']:6.3f} {row['false_positive_rate']:6.3f} "
          f"{row['balanced_accuracy']:6.3f} {row['missed_failure_adjusted_detection_fraction']:6.3f} "
          f"{row['failure_recall_by_landmark']['0.25']:6.3f}  {delta}")
for task_type, rows in d["by_task_type"].items():
    print("\n", task_type)
    for name, row in rows.items():
        print(f"  {name:32s} TPR {row['true_positive_rate']:.3f} FPR {row['false_positive_rate']:.3f} adj {row['missed_failure_adjusted_detection_fraction']:.3f}")
PY
```

Outputs:

- `evaluate/analysis.json`: every metric, paired comparison, and bootstrap;
- `evaluate/detector_summary.csv`: one row per detector, for tables;
- `evaluate/event_predictions.jsonl`: per-rollout baseline trajectories and
  primary-cohort membership;
- `develop/selection_audit.csv` and `calibrate/threshold_curves.csv`: the full
  development and calibration searches.

Threshold keys follow the frozen module's slots: for a baseline detector,
`safe_only`/`early_safe` hold the baseline's threshold and `score_source` names
the score.

## 9. Interpretation rules

- Compare detectors only at the calibrated operating points. Observed test
  FPRs differ slightly; report them with their Wilson intervals.
- The representation question is `staged_<baseline>_time` versus
  `staged_safe_time` (same wrapper, same time fallback). The staging question
  is `fused_safe_time` versus `staged_safe_time`.
- The atomic/composite strata are secondary, as in the frozen study.
- Do not select a baseline, aggregation, or threshold from these test results,
  and do not rerun `evaluate` with a different bundle to improve a baseline.

## 10. Fused baseline extension

The completed run showed `fused_safe_time` ahead of `staged_safe_time` on
recall. This extension asks whether that gain needs SAFE in particular, or
whether any policy-internal score fused with elapsed time does as well. It
adds `fused_mahalanobis_time`, `fused_knn_cosine_time`,
`fused_pca_kmeans_time`, and `fused_rnd_time`. Each is the selected baseline
score OR the frozen time risk, both eligible for the whole rollout, with the
threshold pair searched exactly as for `fused_safe_time`. The run reuses the
existing caches and `develop/baseline_bundle.json` without change, and writes
only to new directories:

- `calibrate_fused/` must reproduce every threshold and calibration metric in
  `calibrate/` (`--reference-calibration`), or it stops;
- `evaluate_fused/` must reproduce every result in `evaluate/`
  (`--reference-evaluation`) and the frozen prospective analysis, or it stops.
  It also pairs every detector against `fused_safe_time`, `staged_safe_time`,
  and `time_only`.

This is a second post-hoc look at the already-opened prospective cohort. Write
the interpretation note's first part before calibration and its second part
before evaluation.

### 10.1 Session, code, and prior-output checksums

Run the export block in section 3 (through `XR1_FROZEN_ANALYSIS`; `SAFE_PY` is
the absolute `$SAFE_ENV/bin/python`), then point
at the completed run and move its dedicated worktree to the current main. Do
not touch the shared `$PROJECT_FS/robocasa` checkout. Every command is
CPU-only (`CUDA_VISIBLE_DEVICES=""`) and runs under `nice`.

```bash
export XR1_BASELINE_TAG=xr1_offline_baselines_20261017_100820
export XR1_BASELINE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$XR1_BASELINE_TAG"
export XR1_BASELINE_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval/$XR1_BASELINE_TAG"
export XR1_BASELINE_WORKTREE="$PROJECT_FS/robocasa_xr1_baselines_d71b6d27"
export XR1_FUSED_NOTE="$XR1_BASELINE_ROOT/interpretation_plan_fused.md"

test -z "$(git -C "$XR1_BASELINE_WORKTREE" status --porcelain --untracked-files=no)" \
  || echo "STOP: the baseline worktree has local changes"
git -C "$XR1_BASELINE_WORKTREE" fetch origin main
export XR1_FUSED_CODE_SHA="$(git -C "$XR1_BASELINE_WORKTREE" rev-parse FETCH_HEAD)"
git -C "$XR1_BASELINE_WORKTREE" checkout --detach "$XR1_FUSED_CODE_SHA"
test "$(git -C "$XR1_BASELINE_WORKTREE" rev-parse HEAD)" = "$XR1_FUSED_CODE_SHA" \
  || echo "STOP: worktree HEAD is not the fetched commit"
echo "fused code commit: $XR1_FUSED_CODE_SHA"
grep -q -- "--fused-baselines" \
  "$XR1_BASELINE_WORKTREE/robocasa/recovery/safe/run_offline_detector_baselines.py" \
  || echo "STOP: worktree lacks the fused extension"

cd "$XR1_BASELINE_WORKTREE"
CUDA_VISIBLE_DEVICES="" nice -n 10 "$SAFE_PY" -m unittest tests.test_safe_offline_detector_baselines tests.test_safe_prospective_matched_fpr
CUDA_VISIBLE_DEVICES="" nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines synthetic-smoke \
  --output-dir "$XR1_BASELINE_ROOT/synthetic_smoke_fused"

cd "$XR1_BASELINE_ROOT"
sha256sum -c --quiet frozen_inputs.sha256 && echo "frozen inputs OK"
find develop calibrate evaluate -type f -print0 | sort -z | xargs -0 sha256sum \
  > prior_outputs_before_fused.sha256
wc -l prior_outputs_before_fused.sha256
```

Continue only if the tests pass, the smoke reports five `fused_detectors` and
`reference_detectors_reproduced: 14`, and the frozen inputs verify.

### 10.2 Interpretation note, part i (before calibration)

```bash
if test -e "$XR1_FUSED_NOTE"; then
  echo "STOP: $XR1_FUSED_NOTE already exists; do not rewrite it"
else
  printf '# Interpretation plan: fused baseline detectors\n\nRun root: %s\nCode commit: %s\nPart i written (UTC): %s\n\n' \
    "$XR1_BASELINE_ROOT" "$XR1_FUSED_CODE_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$XR1_FUSED_NOTE"
  cat >> "$XR1_FUSED_NOTE" <<'NOTE'
## Part i: declared before calibration

Status of the cohort. The 760-rollout primary prospective cohort was opened
for the frozen detectors (17 August 2026) and again for the twelve offline
baseline detectors (17 October 2026, evaluate/). Evaluating the
fused_<baseline>_time detectors is a second post-hoc look at this
already-opened cohort. Its results are descriptive; a confirmatory claim
requires a fresh prospective collection.

Fixed before calibration. Baseline models, aggregations, and task
normalization come unchanged from develop/baseline_bundle.json. The four new
detectors (fused_mahalanobis_time, fused_knn_cosine_time,
fused_pca_kmeans_time, fused_rnd_time) take their threshold pairs only from
the 228-rollout primary calibration cohort, with the same joint search,
event-level FPR <= 0.05 cap, and tie-breaking as fused_safe_time.
calibrate_fused must reproduce every threshold in calibrate/, and
evaluate_fused must reproduce every result in evaluate/ and the frozen
prospective analysis, or the run stops. Nothing is selected or changed from
test results: no baseline, aggregation, threshold, stage window, or detector.
evaluate_fused runs once.

Decision rule. A SAFE-specific contribution to the fused detector is
supported only if fused_safe_time has higher failure recall than every
fused_<baseline>_time at the calibrated operating points, and, for every
baseline, the paired task/rollout bootstrap 95% CI of the recall difference
(fused_<baseline>_time minus fused_safe_time; 2,000 replicates, seed 0) lies
entirely below zero. Otherwise the fused gain is reported as coming from
fusing a policy-internal score with elapsed time, not from SAFE specifically.
Test FPR (with Wilson intervals) and miss-adjusted detection (with bootstrap
CIs) are reported alongside but do not change this rule.
NOTE
  cat "$XR1_FUSED_NOTE"
fi
```

### 10.3 Calibration

```bash
cd "$XR1_BASELINE_WORKTREE"
if grep -q '^## Part i:' "$XR1_FUSED_NOTE" && ! grep -q '^## Part ii' "$XR1_FUSED_NOTE"; then
  CUDA_VISIBLE_DEVICES="" nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines calibrate \
    --baseline-bundle "$XR1_BASELINE_ROOT/develop/baseline_bundle.json" \
    --calibration-cache "$XR1_BASELINE_ROOT/cache_calibration" \
    --output-dir "$XR1_BASELINE_ROOT/calibrate_fused" \
    --fused-baselines \
    --reference-calibration "$XR1_BASELINE_ROOT/calibrate/baseline_calibration.json" \
    --workers 8 \
    > "$XR1_BASELINE_LOG_ROOT/calibrate_fused.log" 2>&1 &
  echo "calibrate_fused PID: $!"
else
  echo "STOP: write part i of the interpretation note first"
fi
```

Continue only if `calibrate_fused/status.json` lists all 12 prior detectors
under `reference_detectors_reproduced` and every detector has calibration FPR
at most 0.05.

### 10.4 Interpretation note, part ii (before evaluation)

Append the in-sample calibration operating points. These chose the thresholds
and are not performance estimates.

```bash
"$SAFE_PY" - "$XR1_BASELINE_ROOT/calibrate_fused/baseline_calibration.json" "$XR1_FUSED_NOTE" <<'PY'
import datetime, hashlib, json, pathlib, sys

path, note = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if "## Part ii" in note.read_text():
    raise SystemExit("STOP: part ii already exists")
data = json.loads(path.read_text())
names = ["time_only", "staged_safe_time", "fused_safe_time"] + sorted(
    name for name, row in data["detectors"].items() if row["family"] == "fused_baseline"
)
lines = [
    "",
    "## Part ii: calibration operating points, appended before evaluation",
    "",
    f"Written (UTC): {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
    f"calibrate_fused/baseline_calibration.json sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}",
    f"Prior detectors reproduced: {len(data['reference_detectors_reproduced'])}",
    "",
    "| Detector | TP | FP | TPR | FPR | Miss-adjusted |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in names:
    metrics = data["detectors"][name]["calibration_metrics"]
    confusion = metrics["confusion"]
    lines.append(
        f"| {name} | {confusion['tp']}/{confusion['tp'] + confusion['fn']} | "
        f"{confusion['fp']}/{confusion['fp'] + confusion['tn']} | "
        f"{metrics['true_positive_rate']:.3f} | {metrics['false_positive_rate']:.3f} | "
        f"{metrics['missed_failure_adjusted_detection_fraction']:.3f} |"
    )
lines += ["", "These in-sample values selected the thresholds; they are not performance estimates.", ""]
with note.open("a") as stream:
    stream.write("\n".join(lines))
print(note.read_text())
PY
```

### 10.5 One evaluation

```bash
cd "$XR1_BASELINE_WORKTREE"
if grep -q '^## Part ii' "$XR1_FUSED_NOTE"; then
  CUDA_VISIBLE_DEVICES="" nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines evaluate \
    --baseline-calibration "$XR1_BASELINE_ROOT/calibrate_fused/baseline_calibration.json" \
    --test-cache "$XR1_BASELINE_ROOT/cache_prospective_test" \
    --frozen-analysis "$XR1_FROZEN_ANALYSIS" \
    --reference-evaluation "$XR1_BASELINE_ROOT/evaluate/analysis.json" \
    --interpretation-note "$XR1_FUSED_NOTE" \
    --output-dir "$XR1_BASELINE_ROOT/evaluate_fused" \
    --test-per-class 10 \
    --bootstrap-samples 2000 \
    --bootstrap-seed 0 \
    > "$XR1_BASELINE_LOG_ROOT/evaluate_fused.log" 2>&1 &
  echo "evaluate_fused PID: $!"
else
  echo "STOP: append part ii of the interpretation note first"
fi
```

After it exits, confirm that nothing earlier changed:

```bash
cd "$XR1_BASELINE_ROOT"
sha256sum -c --quiet prior_outputs_before_fused.sha256 && echo "prior outputs unchanged"
sha256sum -c --quiet frozen_inputs.sha256 && echo "frozen inputs unchanged"
```

### 10.6 Audit table and decision rule

```bash
"$SAFE_PY" - "$XR1_BASELINE_ROOT/evaluate_fused/analysis.json" <<'PY'
import json, pathlib, sys

d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("status:", d["status"], "| frozen reproduced:", d["frozen_results_reproduced"],
      "| prior detectors reproduced:", len(d["reference_detectors_reproduced"]))
print("note sha256:", d["interpretation_note_sha256"])
boot = d["task_rollout_bootstrap"]

def delta(name, reference, key):
    row = boot.get(f"{name}__vs__{reference}")
    if row is None:
        return f"{'-':>22s}"
    value = row[key]
    return f"{value['mean']:+.3f} [{value['ci95'][0]:+.3f},{value['ci95'][1]:+.3f}]"

print(f"{'detector':34s} {'TPR':>6s} {'FPR':>6s} {'BA':>6s} {'adj':>6s} {'r@25':>6s} "
      f"{'dTPR vs fused SAFE':>22s} {'d_adj vs fused SAFE':>22s} "
      f"{'dTPR vs staged':>22s} {'dTPR vs time':>22s}")
for name, row in sorted(d["overall"].items()):
    print(f"{name:34s} {row['true_positive_rate']:6.3f} {row['false_positive_rate']:6.3f} "
          f"{row['balanced_accuracy']:6.3f} {row['missed_failure_adjusted_detection_fraction']:6.3f} "
          f"{row['failure_recall_by_landmark']['0.25']:6.3f} "
          f"{delta(name, 'fused_safe_time', 'tpr_delta')} "
          f"{delta(name, 'fused_safe_time', 'adjusted_detection_delta')} "
          f"{delta(name, 'staged_safe_time', 'tpr_delta')} {delta(name, 'time_only', 'tpr_delta')}")

fused = sorted(name for name, row in d["detectors"].items() if row["family"] == "fused_baseline")
rule = {name: boot[f"{name}__vs__fused_safe_time"]["tpr_delta"]["ci95"][1] < 0.0 for name in fused}
print("\nSAFE-specific rule (every fused baseline recall CI below zero):", rule)
print("SAFE-specific contribution supported:", bool(rule) and all(rule.values()))
for task_type, rows in d["by_task_type"].items():
    print("\n", task_type)
    for name in ["time_only", "staged_safe_time", "fused_safe_time"] + fused:
        row = rows[name]
        print(f"  {name:32s} TPR {row['true_positive_rate']:.3f} FPR {row['false_positive_rate']:.3f} "
              f"adj {row['missed_failure_adjusted_detection_fraction']:.3f}")
PY
```

Report the decision rule's outcome as written in part i, whatever it is.

# Published baselines on the frozen Xiaomi matched-FPR protocol

`robocasa.recovery.safe.baselines` reimplements published failure detectors on
the same RoboCasa365 / Xiaomi-Robotics-1 rollouts that SAFE uses, and scores
them under the frozen August protocol (same training rollouts, task horizons,
elapsed-time risk curves, balanced calibration rollouts, 5% FPR target and
SAFE/time stage windows). No new collection or policy inference is needed: the
detectors read the stored DiT features and the recorded action chunks.

## Detectors

| Name | Follows | Inputs | Training labels |
| --- | --- | --- | --- |
| `action_consistency` | STAC-Single (SAFE baselines), ACC in VLA-FAIL | Overlap disagreement between consecutive action chunks, EMA | None |
| `actprobe` | ActProbe (Huang et al., 2026) | Chunk consistency, chunk magnitude, executed magnitude, task code; LSTM | Success/fail per rollout |
| `mahalanobis` | Embedding baseline in SAFE; LLMD in VLA-FAIL | Features, PCA + Ledoit-Wolf covariance | Successes only |
| `cosine_knn` | Embedding baseline in SAFE | Raw features, top-5 cosine distance | Successes only |
| `pca_kmeans` | Embedding baseline in SAFE and FAIL-Detect | Features, PCA + 32 k-means centroids | Successes only |
| `rnd` | OOD half of FIPER; RND in FAIL-Detect | Features, random network distillation | Successes only |
| `hide_and_seek` | Hide-and-Seek (Park et al., 2026) | Features (PCA 128); LSTM with peak-margin and proxy-onset losses | Success/fail per rollout |

Not reproducible from stored rollouts: full STAC, FIPER action-chunk entropy,
Tri-Info and ACE need several action samples per inference, and LogpZO needs a
flow density model. Those would need a new multi-sample collection.

Each detector is trained with seeds 0, 1 and 2. Scores are turned into a
running maximum, normalized per task from training maxima and averaged over
seeds, exactly as the SAFE scores in `run_prospective_matched_fpr`.

## Run on TSUBAME

Use the SAFE Python environment (needs numpy, scikit-learn and PyTorch). The
dataset directories must contain every raw rollout named by the training,
calibration and evaluation IDs: the original Xiaomi collections that fed the
final-refit export, plus the calibration and prospective test collections.

```bash
source "$STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_latest.env"

export XR1_FINAL_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508"
export XR1_PMF=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700
export XR1_PMF_ROLLOUTS=/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_prospective_matched_fpr_20260815_163700
export XR1_BASELINES_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_baselines_$(date +%Y%m%d_%H%M%S)"

# Bounded smoke: one seed, two cheap detectors, few epochs.
"$SAFE_PY" -u -m robocasa.recovery.safe.baselines.run_baselines \
  --dataset-dir "$XR1_SAFE_DATASET" <earlier-xiaomi-dataset-dirs> \
                "$XR1_PMF_ROLLOUTS/calibration_collection" \
                "$XR1_PMF_ROLLOUTS/prospective_test_collection" \
  --runtime-bundle "${XR1_PMF}_calibration/runtime_bundle.json" \
  --train-ids "$XR1_FINAL_ROOT/indep_seed0/scores.jsonl" \
  --evaluation-ids "${XR1_PMF}_prospective_test/event_predictions.jsonl" \
  --output-dir "${XR1_BASELINES_ROOT}_smoke" \
  --detectors action_consistency mahalanobis \
  --seeds 0

# Full comparison.
"$SAFE_PY" -u -m robocasa.recovery.safe.baselines.run_baselines \
  --dataset-dir "$XR1_SAFE_DATASET" <earlier-xiaomi-dataset-dirs> \
                "$XR1_PMF_ROLLOUTS/calibration_collection" \
                "$XR1_PMF_ROLLOUTS/prospective_test_collection" \
  --runtime-bundle "${XR1_PMF}_calibration/runtime_bundle.json" \
  --train-ids "$XR1_FINAL_ROOT/indep_seed0/scores.jsonl" \
  --evaluation-ids "${XR1_PMF}_prospective_test/event_predictions.jsonl" \
  --output-dir "$XR1_BASELINES_ROOT" \
  --seeds 0 1 2 \
  --device cuda
```

`--test-per-class 10` (the default) reproduces the 760-rollout balanced
primary cohort of the frozen evaluation. `--action-dims` restricts the
action-space baselines to chosen dimensions (for example the arm deltas without
the mobile base).

## Outputs

- `summary.md`: one row per detector variant with ROC-AUC, TPR, FPR, balanced
  accuracy, missed-failure-adjusted detection fraction and recall by 25% of the
  horizon, plus the time-only reference.
- `results.json`: configs, calibration thresholds, calibration and evaluation
  metrics, and paired detection-time comparisons against time-only.
- `evaluation_predictions.jsonl`: per-rollout normalized score and time-risk
  trajectories for every detector.

Each detector is reported twice: alone (`baseline only`) and in your staged
design (`staged baseline time`: the baseline may alarm before 25% of the
horizon, elapsed time after 50%). The second row answers whether the time
fallback helps every detector or only SAFE.

The evaluation cohort was already opened for SAFE, so these numbers compare
methods on a shared test set; they are not a new confirmatory test.

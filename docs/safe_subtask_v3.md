# Causal Subtask-SAFE v3

Subtask-SAFE v3 is the representation/objective experiment prescribed after the
v2 held-out continuation gate failed. It is backward compatible: without the new
flags, causal training still uses eventual segment outcomes, raw RLDX features,
and the pinned official SAFE objective.

## Scientific changes

### Finite-horizon target

For a causal prefix of length `p`, a source segment of length `L`, and prediction
horizon `H`, the v3 target is

```text
failure_within_H = source_segment_failed and (L - p <= H)
```

Thus an early prefix of an eventually failed segment is negative when the
terminal failure remains more than `H` policy inferences away. No future feature
is exposed to the model. The source outcome, source length, remaining inference
count, horizon, and causal target are written to every score record.

Use:

```text
--causal-label-mode within_horizon --causal-failure-horizon H
```

The command fails before training if the expanded training or test prefixes do
not contain both target classes. Inspect `target_counts_by_prefix` in completed
smoke metrics before choosing the preregistered selection prefix. Increasing `H`
or adding later prefixes is preferable to silently reverting to eventual labels.

### Causal temporal representations

`--temporal-representation` supports:

| Mode | Feature at inference `t` |
|---|---|
| `raw` | `x_t` |
| `delta` | `x_t - x_(t-1)`; zero at entry |
| `anchor_delta` | `x_t - x_1` |
| `raw_delta` | concatenated raw and consecutive delta |
| `raw_delta_mean_slope` | raw, delta, trailing-window mean, trailing-window slope |

The last mode uses `--temporal-window` (default `4`). Every transformation is
causal and preserves sequence length. Stage one-hot conditioning is appended
after the temporal transformation and its catalog remains training-only.

### Objectives

`--loss-mode` supports:

- `official`: the pinned SAFE margin/BCE objective;
- `bce`: binary cross-entropy on the instantaneous failure probability at each
  sampled causal prefix endpoint;
- `focal`: the same target with focal modulation, configured by
  `--focal-gamma` (default `2`).

Class weighting remains an independent choice through `--class-weighting`.
The endpoint-only loss avoids assigning a prefix label to earlier observations
that may have a different finite-horizon target. This permits scale-matched
unweighted BCE, inverse-frequency BCE, and focal ablations without changing the
outer split.

### Honest controls and calibration

The v3 gate evaluator reports two separate training-only controls:

1. **Stage prior:** smoothed target prevalence for each semantic stage and
   prefix. This is the control that v2 incorrectly named elapsed time.
2. **Duration progress:** prefix length divided by the median duration of
   successful training source segments for that stage.

Complete successful held-out parent rollouts are deterministically reserved for
threshold calibration. No segment from those parents remains in evaluation.
The final test ROC and parent bootstrap use only the remaining held-out parents.
The continuation gate requires SAFE to beat both controls.

## Recommended experiment order

Keep the original parent split and training-only stage-selection rule frozen. First use a
two-epoch, one-configuration smoke to find a failure horizon/prefix combination
with both labels. Candidate prefix lists should include later observations when
short horizons otherwise produce no failures, for example:

```text
--causal-prefix-horizons 1 2 4 8 16 32 64 128
```

Once target support is valid, freeze the previous best network hyperparameters
and screen representation/objective treatments with inner CV only:

```bash
python -u -m robocasa.recovery.safe.run_seen_cv_grid \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$TREATMENT_ROOT" \
  --model indep \
  --selection-manifest "$SUBTASK_PARENT_SPLIT" \
  --task-type composite \
  --class-weighting official_inverse_frequency \
  --loss-mode bce \
  --causal-prefix-mode fixed \
  --causal-prefix-horizons 1 2 4 8 16 32 64 128 \
  --causal-selection-prefix 32 \
  --causal-conditioning subtask_one_hot \
  --causal-label-mode within_horizon \
  --causal-failure-horizon 32 \
  --temporal-representation raw_delta \
  --temporal-window 4 \
  --min-stage-successes 9 \
  --min-stage-failures 5 \
  --horizon-selectors concat-2 \
  --diffusion-selectors 1.0 \
  --learning-rates 3e-4 \
  --regularization 1e-2 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume
```

The example horizon and selection prefix are candidates, not fixed scientific
choices. Confirm their target support in a smoke first. Select one treatment
from inner CV, perform three final refits, and evaluate exactly once on the
remaining held-out parents:

```bash
python -u -m robocasa.recovery.safe.evaluate_causal_subtask_gates \
  --final-root "$CAUSAL_FINAL_ROOT" \
  --output-dir "$CAUSAL_FINAL_ROOT/causal_gate_analysis" \
  --models indep \
  --seeds 0 1 2 \
  --prefixes 1 2 4 8 16 32 64 128 \
  --target-prefix 32 \
  --calibration-parent-fraction 0.30 \
  --calibration-seed 0 \
  --target-fpr 0.10 \
  --min-roc 0.65 \
  --min-delta 0.05 \
  --min-tpr 0.40 \
  --min-lead 0.25 \
  --bootstrap-replicates 2000
```

Do not collect additional rollouts or integrate recovery unless the final model
passes every gate. If all stored-feature treatments fail, the next experiment is
new feature capture from an earlier RLDX layer, not additional examples of the
same representation.

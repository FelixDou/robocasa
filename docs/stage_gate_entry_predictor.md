# Entry-time stall gate: offline analysis plan (trial-2 data)

Written 5 November 2026, before any trial-2 policy feature was loaded.
Motivation: [trial 2 results](stage_gated_reprompt_trial2_results_20261105.md).
Code: `robocasa/recovery/stage_gate_entry_predictor.py`; tests:
`tests/test_stage_gate_entry_predictor.py`.

## Question

At ArrangeTea's mug stage, re-prompting at stage entry helped the 58 seeds
whose stage later overran 617 steps (21 helped / 3 harmed) and harmed those
that finished on time (11 / 41). Does the state at stage entry predict the
overrun well enough that an entry-time gate beats baseline?

Reference values over the 200 ArrangeTea seeds:
- always re-prompt at entry: −0.060;
- stall gate (trial 2 arm): +0.005;
- oracle (re-prompt at entry exactly where the stage will overrun): +0.090;
- best arm per seed: +0.170.

## Data

- **Cohort:** the 173 ArrangeTea seeds whose stall-arm rollout entered the
  mug stage. Seeds that never entered it count with gain 0 in every policy,
  so values are over all 200 seeds.
- **Features:** the stall arm runs exactly like baseline until its switch,
  so its unswitched captures are baseline-state features. Each capture is
  reduced to the frozen SAFE selector (last flow step, last action token;
  1,024 dimensions).
- **Outcomes:** baseline and entry-gate (`gated_reprompt`) success of the
  same seed. Policy gain per seed: `entry − baseline` if the gate fires,
  else 0.
- **Label for the learned probes:** the stall gate switched (stage still
  running at 617 steps).

## Pre-specified gates

The decision is at the stage's first inference (inference 0). There the
entry-gate arm is the exact counterfactual.

| Gate | Score (higher = re-prompt) | Fitted in-fold |
|---|---|---|
| `safe` | frozen XR-1 SAFE ensemble, single-inference score | threshold only |
| `entry_step` | step at which the mug stage starts | threshold only |
| `probe` | L2 logistic on 16 PCA components of the token, trained on the stall label (C from {0.001, 0.01, 0.1, 1} by inner 5-fold log-loss) | PCA, logistic, threshold |
| `probe_plus_time` | `probe` plus entry step as one extra column after PCA | same |

- **Threshold:** in each training fold, the threshold maximizes the training
  seeds' summed gain; ties go to fewer re-prompts, and "never" is allowed.
- **Evaluation:** 10 repeats of stratified 5-fold cross-validation over the
  173 entered seeds; per-seed gain averaged over repeats.
- **Reported per gate:**
  - Δ vs baseline over 200 seeds, with a seed-bootstrap 95% CI;
  - permutation p: 200 permutations of the score inputs across seeds, with
    outcomes and labels fixed and the whole pipeline refit;
  - Holm correction across the four gates;
  - fraction of entered seeds gated, expected helped/harmed, and
    out-of-fold AUROC for the stall label.
- **Secondary, estimate only:** the same gates using captures up to
  inference 4 (about 64 steps after entry). That counterfactual is
  approximated by the entry-gate outcome, so it is optimistic.

**Decision rule for a third prospective trial:** at decision 0, some gate
has Δ ≥ +0.04 and Holm p < 0.05. If so, freeze that gate (fit on all 173
seeds, with its threshold) and run `baseline` vs `predicted_stall_entry_gate`
on fresh ArrangeTea seeds. If no gate passes, the paper reports that entry
state did not predict intervention value at this stage, and the stall
result stands as the negative timing result.

The 173 seeds give limited power: with 58 positives, only a strong
predictor (out-of-fold AUROC around 0.75 or more) can pass.

## Cluster step (CPU only, a few minutes)

The trial's `features/` directory stays on the cluster. This step writes one
small file inside the run directory, holding the 1,024-d selector tokens and
SAFE scores for the ArrangeTea stall arm:

```bash
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa
export SAFE_REPO=/gs/fs/tga-shinoda/felid/SAFE SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export XR1_RUNTIME_BUNDLE=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json
export GATE_ROOT=$STORAGE_BS/robocasa_rollouts/recovery/xr1_stage_gate_trial2_20261104_104336
cd "$ROBOCASA_REPO" && git pull origin main
"$SAFE_PY" -u -m robocasa.recovery.stage_gate_entry_predictor extract \
  --run-root "$GATE_ROOT" --output "$GATE_ROOT/entry_predictor/entry_features.npz" \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" --safe-repo "$SAFE_REPO"
```

Expect `"seeds": 173` and `"safe_scored": true`. If SAFE scoring fails, rerun
without `--runtime-bundle`/`--safe-repo` (into a new output path). The
`safe` gate is then dropped, and the others still run.

Copy to the Mac:

```bash
rsync -av tsubame:/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_stage_gate_trial2_20261104_104336/entry_predictor/ ~/Desktop/PhD/Shinoda_lab/Code/robocasa/outputs/xr1_stage_gate_trial2_20261104_104336/entry_predictor/
```

## Local analysis

```bash
python3 -m robocasa.recovery.stage_gate_entry_predictor analyze \
  --run-root outputs/xr1_stage_gate_trial2_20261104_104336 \
  --features outputs/xr1_stage_gate_trial2_20261104_104336/entry_predictor/entry_features.npz \
  --output-dir outputs/xr1_stage_gate_trial2_20261104_104336/entry_predictor
```

## Results (5 November 2026)

Run as pre-specified: 173 entered seeds (58 stalled), 1,564 unswitched
captures, all scored by the frozen SAFE ensemble. Output:
`outputs/xr1_stage_gate_trial2_20261104_104336/entry_predictor/entry_predictor.json`.

| Gate | Δ vs baseline at inference 0 [95% CI] | Permutation p | Holm p | Gated (of entered) | AUROC (stall) |
|---|---|---:|---:|---:|---:|
| `safe` | −0.018 [−0.057, +0.019] | 0.71 | 1.00 | 0.31 | 0.61 |
| `entry_step` | −0.011 [−0.042, +0.021] | 0.64 | 1.00 | 0.20 | 0.64 |
| `probe` | −0.009 [−0.049, +0.031] | 0.31 | 0.93 | 0.32 | 0.61 |
| `probe_plus_time` | −0.000 [−0.036, +0.038] | 0.23 | 0.92 | 0.32 | 0.62 |

At inference 4 (estimate only), all gates are between −0.011 and +0.019,
with no Holm p below 0.15.

**No gate passes the decision rule, so there is no third trial.**

Why entry-time gating cannot work at this stage:

- **Harm outweighs rescue.** The seeds that re-prompting harms outnumber
  those it rescues (115 on-time seeds, net −30; 58 stalled seeds, net +18).
  A gate therefore needs a very accurate stall predictor before it gains
  anything. Under a binormal ROC, the best achievable Δ is about +0.001 at
  AUROC 0.60, +0.020 at 0.75 and +0.038 at 0.85. Reaching +0.04 needs an
  AUROC of about 0.87.
- **The observed signals are near chance.** Stall AUROCs are 0.60–0.64.
  Without any fitting, SAFE and entry step separate helped from harmed seeds
  at AUROC 0.57 and 0.57 (32 helped / 44 harmed). The frozen SAFE score at
  entry predicts baseline failure at AUROC 0.53.

Conclusion: at ArrangeTea's mug stage, neither the entry state (frozen SAFE,
a probe on the XR-1 token, entry time) nor time in stage gives a usable
re-prompt gate. The stage's intervention value is real but not identifiable
in time. This complements the replicated positive at BreadSelection, where
re-prompting helps most rollouts that reach the stage, so no state signal is
needed.

# Stage-gated re-prompt trial 2: stall-gated re-prompting (XR-1)

> **Results (5 November 2026):** [stage_gated_reprompt_trial2_results_20261105.md](stage_gated_reprompt_trial2_results_20261105.md):
> primary not confirmed (+0.005, p = 1.00); the stall gate picks the right
> seeds but re-prompts too late.

Why: [trial 1 results](stage_gated_reprompt_results_20261030.md) and the
[offline analysis](stage_gate_state_signal_analysis_20261103.md). Re-prompting
ArrangeTea at mug-stage entry broke rollouts that were on the normal path;
time stuck in the stage separated the seeds it helped from those it harmed.
Code, protocol details and the trial-1 design:
[stage_gated_reprompt.md](stage_gated_reprompt.md). Read
[the cluster runbook](cluster_experiment_runbook.md) before cluster work.

## Design

Same machinery as trial 1: natural rollouts on fresh seeds, all arms of one
(task, seed) with the same environment seed and policy sampling-seed
schedule, arms identical until their first instruction switch.

| Arm | ArrangeTea mug stage | BreadSelection jam stage |
|---|---|---|
| `baseline` | task instruction | task instruction |
| `gated_reprompt` | stage instruction at stage entry (trial 1) | at stage entry |
| `stall_gated_reprompt` | stage instruction only after **617 steps** in the stage without placing the mug | at stage entry (delay 0) |

- **Delay 617** = 90th percentile of mug-stage durations in trial 1's 47
  baseline successes, frozen before any trial-2 rollout. A stage that
  completes earlier is never re-prompted (`completed_before_delay`).
- **Step cap** 1,280 steps per switch, as in trial 1.
- **Seeds** 410007–410206 (200 per task), unused by any earlier XR-1 run.
- **Features**: the XR-1 SAFE token at every 4th inference inside the
  whitelisted stage and at the first inference after each switch, saved per
  rollout under `features/`, for an offline SAFE-based gate afterwards.

## Pre-registered analysis

- **Primary:** full-task success, `stall_gated_reprompt − baseline` on
  **ArrangeTea**, paired by seed, two-sided exact McNemar, α = 0.05; paired
  bootstrap 95% CI.
- **Secondary:** the same contrast on BreadSelection (replication of trial 1),
  macro over both tasks, `gated_reprompt − baseline` (replication of trial 1's
  ArrangeTea harm) and `gated_reprompt − stall_gated_reprompt` (the value of
  waiting); help/harm counts, harm among baseline successes, rescue among
  baseline failures; how often the stall gate fired and how often stages
  completed before the delay.

The analyzer reads the primary from the config (`primary_contrast`,
`primary_tasks`) and prints it as the primary endpoint.

## Cost and power

`plan` with this config, 0.0585 s/step (trial-1 measured):

| | env steps | worker-h | wall-h on 4 GPUs (2 workers) |
|---|---:|---:|---:|
| expected | 2.20 M | 43.0 | **21.5** |
| upper | 2.52 M | 48.3 | 24.1 (one resume) |

Expected ArrangeTea effect: the stall gate touches about a quarter of the
rollouts, rescuing ~5–9% of all seeds and harming ~1–2%. Exact McNemar power
at 200 seeds: 0.85 for 7% help / 1% harm, 0.60 for 7% / 2%, 0.56 for 5% / 1%.

A cheaper alternative with more power would drop `gated_reprompt` (two arms,
300 seeds, about the same cost); it was not chosen, so that the value of
waiting is measured within one cohort.

## Cluster commands

Fresh terminal on a 4-GPU allocation:

```bash
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa
export XR1_CLIENT_ENV=$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export GATE_CONFIG=$ROBOCASA_REPO/scripts/stage_gated_reprompt_xr1_trial2.json
export GATE_RUN_PREFIX=xr1_stage_gate_trial2
cd "$ROBOCASA_REPO" && git pull origin main
```

1. Dry run (the planner prints a mimicgen warning to stdout first):

```bash
bash scripts/run_stage_gated_reprompt_xr1.sh plan 2>/dev/null | sed -n '/^{/,$p' | "$XR1_CLIENT_ENV/bin/python" -c 'import json,sys; p=json.load(sys.stdin); print(json.dumps({k: p[k] for k in ("seed_range","primary_contrast","primary_tasks","totals","warnings")}, indent=1))'
```

Expect seeds 410007–410206, primary `stall_gated_reprompt − baseline` on
ArrangeTea, no warnings.

2. Smoke (ArrangeTea, 2 seeds × 3 arms, about 25 min):

```bash
unset GATE_ROOT
GATE_GPU_PAIRS="0:1" bash scripts/run_stage_gated_reprompt_xr1.sh smoke
```

When `status` says `complete`, export the printed `GATE_ROOT` and check:

```bash
"$XR1_CLIENT_ENV/bin/python" - "$GATE_ROOT/shard0" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for line in (d / "rollouts.jsonl").read_text().splitlines():
    r = json.loads(line)
    gates = [(g["stage"], g.get("switched"), g.get("skip_reason"), g.get("entry_step"), g.get("switch_step"), g.get("return_reason")) for g in r["gate_stages"]]
    print(r["arm"], r["environment_seed"], "success", r["success"], r["seed_control"], "features", r.get("n_features"), gates)
print("feature files:", len(list((d / "features").glob("*.npz"))) if (d / "features").exists() else 0)
PY
```

Pass if:
- there are no errors and `initial_match` is true for both pairs;
- every rollout shows `policy_sampling_seed_base`;
- `gated_reprompt` switches at `mug_on_tray` at stage entry;
- `stall_gated_reprompt` either switches exactly 617 steps after `entry_step`
  or records `completed_before_delay`;
- every rollout that reached the mug stage has `features` > 0, and feature
  files exist.

3. Trial (4 GPUs):

```bash
unset GATE_ROOT
GATE_GPU_PAIRS="0:1 2:3" bash scripts/run_stage_gated_reprompt_xr1.sh start
```

Monitor with `status`; resume in a new allocation with the same
`GATE_CONFIG`, `GATE_RUN_PREFIX`, `GATE_ROOT` and two GPU pairs
(`bash scripts/run_stage_gated_reprompt_xr1.sh resume`). Analyze with
`analyze` when both shards are complete. Features add about 5 GB under
`/gs/bs`; copy the results without them:

```bash
rsync -av --exclude 'features/' tsubame:<GATE_ROOT>/ ~/Desktop/PhD/Shinoda_lab/Code/robocasa/outputs/<run name>/
```

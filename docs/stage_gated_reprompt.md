# Stage-gated re-prompting: prospective end-to-end trial (XR-1)

Code: `robocasa/recovery/stage_gated_reprompt.py` (collector, analysis, dry
run), `scripts/run_stage_gated_reprompt_xr1.sh` (cluster launcher),
`scripts/stage_gated_reprompt_xr1.json` (default config),
`tests/test_stage_gated_reprompt.py` (simulator-free tests).

## Why

Two counterfactual recoverability-ladder pilots
([recoverability_ladder.md](recoverability_ladder.md); v1:
[recoverability_ladder_pilot_20261016.md](recoverability_ladder_pilot_20261016.md);
v2: `recoverability_ladder_v2_pilot_20261017.md`, 2,020 branches) found that
switching XR-1 to the current stage's atomic instruction ("re-prompt", P)
changes success **with a stage-dependent sign**. Compared with resampling (R)
from the same states:

| task / stage | evidence |
|---|---|
| ArrangeTea / `mug_on_tray` | v1 0/16 → 4/16; v2 P − R = +0.20 to +0.35 from stage entry to +8 inferences |
| BreadSelection / `jam_on_cutting_board` | v1 0/20 → 19/20 (5 stalled parents); not retested in v2 |
| GarnishPancake / `strawberry_on_pancake` | v2 P − R = −0.25 to −0.46; v1 0/12 |
| ArrangeBreadBasket / `bread_in_basket` | v2 −0.25 at stage entry |
| all failures pooled | v2 −0.14 at stage entry |

Pooled re-prompting is harmful, so any deployable use must be gated by stage.
Those numbers are branch counterfactuals from failed parents' states. This
trial is the first prospective test: natural rollouts on fresh seeds, with
full-task success over **all** rollouts as the endpoint, so harm to rollouts
that would have succeeded anyway counts.

## Design

No branching and no snapshot restores. For every (task, seed) pair, each arm
runs one natural rollout from the same environment seed with the same policy
sampling-seed schedule:

| arm | instruction |
|---|---|
| `baseline` | task instruction throughout |
| `gated_reprompt` | when the ordered stage tracker **enters a whitelisted stage**, switch to that stage's atomic instruction until the stage predicate holds or `reprompt_step_cap` steps pass, then return to the task instruction |
| `always_reprompt` | the same at **every** stage (shows why gating matters) |

- **Whitelist** (frozen from the pilots, `reprompt_whitelist`):
  ArrangeTea `mug_on_tray` (positive in v1 and v2) and BreadSelection
  `jam_on_cutting_board` (positive in v1). **GarnishPancake** has no
  whitelisted stage. It is the control task: `gated_reprompt` cannot switch
  there and must equal `baseline`, while `always_reprompt` measures the harm
  of ungated switching on a task where the ladder found it.
- **Stage** = oracle stage from simulator predicates: the ladder's
  `OrderedProgress` over the env's `required_predicates`, with the same
  `exclude_predicates` / `required_subtask_overrides` mechanism (none are
  needed for these three tasks). The ordered lists are ArrangeTea `kettle_on_tray →
  mug_on_tray → cabinet_closed → gripper_released`, BreadSelection
  `croissant_on_cutting_board → jam_on_cutting_board → gripper_released`,
  GarnishPancake `fridge_open → pancake_on_plate → plate_on_table →
  strawberry_on_pancake → gripper_released`. A whitelisted name missing from
  the ordered list stops the rollout with an error (no silent no-op gate).
- **Stage instruction**: `_resolve_recovery_target` at `reprompt_level=atomic`,
  as in the ladder ("Move mug to tray.", "Move jam to cutting board.").
- **Switch timing**: the gate fires at the env step at which the previous
  stage latches; the stage counts as entered at that step. Each stage fires at most once per rollout. A switch ends when the target
  predicate holds (or the stage latches), after `reprompt_step_cap` steps, or
  at the end of the rollout. If the next stage is also eligible (only in
  `always_reprompt`), the next switch starts at the same step. Stages that
  latch together at one step (GarnishPancake's `pancake_on_plate`,
  `plate_on_table`) never become current and never fire.
- **Step cap: 1280 steps** (80 XR-1 inferences). In ladder re-prompt branches
  that went on to succeed, the target completed within 1280 steps in 96% (v2,
  ArrangeTea mug, 23 branches) and 100% (v1 ArrangeTea; v1 BreadSelection jam,
  22 branches, all within 960) of cases.
- **Policy context on a switch**: XR-1's client resets itself when the
  instruction changes (`xiaomi_robotics_1_policy.py`, `__call__`). It drops the
  action chunk and the image history, as the ladder's re-prompt branches did
  (`policy_context=reset`). The collector makes this explicit for any policy:
  every instruction change, including the return, resets the policy, so a switch takes effect
  at the step it fires. `policy_resets` records how many resets happened.
- **Seeds and pairing**: fresh base seed **400007** (unused by every earlier
  XR-1 run; smoke uses 499007). Before each arm's reset the global NumPy and
  Python RNGs are seeded from (task, seed), the env is reset with the seed, and
  the policy gets `policy_seed = stable_seed(task, seed)`. Because the policy
  reset also restarts XR-1's inference counter, the collector rebases
  `sampling_seed_base` to `policy_seed + global inference index` on every
  switch. The n-th inference of every arm therefore uses the same sampling
  seed, and arms are identical until the first switch if the simulator and
  server are deterministic. Arms of one seed run back to back on one worker
  in a counterbalanced order (stable permutation per seed). Each arm records
  the hash of its initial simulator state and a state hash every 16 steps;
  the pair row stores whether the initial states matched and the first step
  at which each arm's state diverged from baseline.
- **Sharding**: seeds round-robin over workers; each worker processes seed
  blocks of 10 through all three tasks, so a stopped run stays balanced across
  tasks. A seed is complete only when all its arms have run (its `pairs.jsonl`
  row); `resume` reruns incomplete seeds from scratch.

## Pre-registered analysis

- **Primary endpoint**: full-task success, paired by seed,
  `gated_reprompt − baseline`, **macro over the whitelisted tasks** (mean of
  ArrangeTea's and BreadSelection's paired differences). Test: two-sided
  **exact McNemar** on the pooled discordant pairs of those tasks, α = 0.05.
  CI: paired bootstrap, seeds resampled within task (2,000 resamples).
- **Per task** (secondary): the same contrast with its own exact McNemar and
  bootstrap CI; Holm-adjusted p across the whitelisted tasks.
- **Harm**: `help` (gated succeeded, baseline failed) and `harm` (the reverse),
  the harm rate among baseline successes and the rescue rate among baseline
  failures. The contrast among pairs where an arm switched (`Δ|switched`) is a
  valid subgroup when arms are identical until the switch, because whether the
  gate fires is decided before the arms diverge.
- **Secondaries**: stage entry and completion rates per stage and arm;
  time to success (median among successes, and the paired difference in
  restricted mean steps, with failures counted as H); how often the switch
  fired, its median start step and duration, how often the target completed
  while switched, and how often the cap was hit.
- **Other contrasts** (secondary): `always_reprompt − baseline` (expected
  ≤ 0, especially on GarnishPancake) and `gated_reprompt − always_reprompt`;
  macro over all tasks (the deployment view, which includes the control task).
- **Validity checks** (reported in `summary.md`): initial-state mismatches
  (must be 0); on the control task, `gated_reprompt` switches (must be 0),
  outcome concordance and identical state traces; discordant pairs in which
  neither arm switched (`disc|¬switched`), which measure the simulator and
  server nondeterminism floor.

## Dry-run estimate and power

**The shipped config uses 100 seeds per task** (900 rollouts: 29.8 worker-h
expected, 37.1 upper; about 15 / 18.6 h on two workers), chosen on 2026-10-19
before any trial rollout. The table below is the 50-seed reference.

`plan` with 3 tasks × 50 seeds × 3 arms = 450 rollouts,
0.055 s/step, 20 s per reset, 60 s per env creation, horizons ArrangeTea 2250,
BreadSelection 1950, GarnishPancake 2700, expected success 0.54 / 0.34 / 0.62.

| | env steps | worker-h | GPU-h | wall-h, 2 workers (4 GPUs) | wall-h, 1 worker (2 GPUs) |
|---|---:|---:|---:|---:|---:|
| expected (successes end at 0.6 / 0.7 / 0.45 H) | 0.79 M | 14.9 | 29.9 | **7.5** | 14.9 |
| upper (every rollout runs to H) | 1.04 M | 18.6 | 37.2 | **9.3** | 18.6 |

Both fit **one 24 h allocation** (23.5 usable hours), even on a single worker.
Two arms only (no `always_reprompt`) cost 10.1 / 12.5 worker-h. 100 seeds per
task cost 29.8 / 37.1 worker-h (14.9 / 18.6 h on two workers) and still fit one
allocation.

Exact McNemar power at α = 0.05 for a **15-point** success difference
(`p_help − p_harm = 0.15`; power falls as discordant pairs in both directions
grow):

| pairs | help / harm 0.15 / 0 | 0.20 / 0.05 | 0.25 / 0.10 | 10 points, 0.10 / 0 |
|---|---:|---:|---:|---:|
| per task, 50 | 0.78 | 0.48 | 0.36 | 0.38 |
| pooled whitelisted (primary), 100 | **1.00** | **0.84** | **0.67** | 0.94 |
| per task, 100 seeds | 1.00 | 0.84 | 0.67 | 0.94 |
| pooled, 100 seeds per task (200) | 1.00 | 0.99 | 0.95 | 1.00 |

Minimum difference with 80% power (no harm / harm one third of help): 0.16 /
0.34 per task and 0.08 / 0.17 pooled at 50 seeds; 0.08 / 0.17 per task at 100
seeds.

**What to expect.** The ladder effects are conditional on failed parents'
states. The unconditional full-task effect is diluted: it is roughly the
probability of reaching the stage and failing there times the rescue rate,
minus the probability of reaching it and succeeding times the harm rate. At
ArrangeTea that is plausibly +0.05 to +0.10. The primary pooled test is
adequately powered for 15 points unless harm is large; per-task tests at 50
seeds are not. If per-task claims matter, run 100 seeds (`seeds_per_task:
100`, same allocation); decide **before** looking at results.

## Outputs (per shard directory)

```text
plan.json        frozen config, seeds of this shard, horizons, whitelisted/control tasks, git commit
rollouts.jsonl   one row per (task, seed, arm): success, steps, progress events, stage latch steps,
                 step-resolution predicate trace, regression events, gate events and per-stage
                 switch records, instruction switches with sampling-seed rebase, initial-state
                 hash, state hashes every 16 steps
pairs.jsonl      one row per complete (task, seed): arm order, success/fired per arm, initial-state
                 match, first divergence step and first switch step per arm
errors.jsonl     per-seed tracebacks (collection stops at max_errors)
status.json      counts per task and arm, measured seconds per env step
```

`analyze` writes `summary.md`, `summary.json` and CSVs (`rollouts`,
`arms_by_task`, `contrasts`, `switches`, `stage_completion`,
`control_checks`).

## Cluster commands

Read `docs/cluster_experiment_runbook.md` first. XR-1 server/client setup and
the `sampling_seed` server patch are in `docs/safe_xiaomi_robotics_1.md` (the
launcher refuses an unpatched server). One worker = simulator GPU + XR-1
server GPU on its own port (`GATE_PORT_BASE`, default **10360**, so the
trial can run next to the ladder's 10340). Default `GATE_GPU_PAIRS="0:1 2:3"`
(two workers, four GPUs). Outputs go to
`$STORAGE_BS/robocasa_rollouts/recovery/xr1_stage_gate_*`, logs to
`$STORAGE_BS/robocasa_logs/eval`. Nothing is written under `/gs/fs`.

Local, after review (merge this branch into `main` first):

```bash
git push origin main
```

Cluster, fresh terminal (in every allocation):

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa
export XR1_CLIENT_ENV=$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
cd "$ROBOCASA_REPO" && git pull origin main
mkdir -p "$STORAGE_BS/robocasa_logs/eval"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
```

1. Dry run (no simulator, no GPU):

```bash
bash scripts/run_stage_gated_reprompt_xr1.sh plan | "$XR1_CLIENT_ENV/bin/python" -c 'import json,sys; p=json.load(sys.stdin); print(json.dumps({k: p[k] for k in ("seed_range","totals","power","warnings")}, indent=1))'
```

2. Smoke: one worker, ArrangeTea, 2 seeds × 3 arms (about 20 min with server
start-up):

```bash
GATE_GPU_PAIRS="0:1" bash scripts/run_stage_gated_reprompt_xr1.sh smoke
bash scripts/run_stage_gated_reprompt_xr1.sh status
tail -f "$STORAGE_BS"/robocasa_logs/eval/xr1_stage_gate_smoke_*_shard0_*.log
```

When `status.json` says `complete`, inspect:

```bash
source "$STORAGE_BS/robocasa_rollouts/recovery/xr1_stage_gate_latest.env" && export SMOKE_ROOT="$GATE_ROOT"
"$XR1_CLIENT_ENV/bin/python" - "$SMOKE_ROOT/shard0" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for line in (d / "pairs.jsonl").read_text().splitlines():
    p = json.loads(line)
    print(p["environment_seed"], p["arm_order"], "success", p["success"], "fired", p["fired"],
          "initial_match", p["initial_state_match"], "divergence", p["first_divergence_step"],
          "first_switch", p["first_switch_step"])
for line in (d / "rollouts.jsonl").read_text().splitlines():
    r = json.loads(line)
    print(r["arm"], r["environment_seed"], r["seed_control"], r["required_subtasks"],
          [(s["stage"], s.get("instruction"), s.get("switch_step"), s.get("return_reason")) for s in r["gate_stages"]])
PY
```

The smoke passes only if: the server log shows `Model loaded`; there is no
`errors.jsonl`; `status.json` has 2 pairs and `measured_seconds_per_env_step`;
every pair has `initial_match True`; every rollout has
`seed_control=policy_sampling_seed_base`; `gated_reprompt` switches only at
`mug_on_tray` with instruction `Move mug to tray.` (when the kettle stage
latched), and `always_reprompt` switches at its first stage at step 0. A
`gated_reprompt` divergence step at or before its first switch means the
server or simulator is nondeterministic. Record it but do not stop: the
analysis reports it, and pairing remains valid. An initial-state mismatch is a
stop-and-inspect condition (try `"seed_protocol": "fresh_env"` in a config
copy).

Optionally re-plan with the measured throughput:

```bash
GATE_CALIBRATION_DIR="$SMOKE_ROOT/shard0" bash scripts/run_stage_gated_reprompt_xr1.sh plan | grep -A8 '"upper"'
```

3. Trial (two workers, four GPUs; about 15–18.6 h at 100 seeds per task):

```bash
unset GATE_ROOT
GATE_GPU_PAIRS="0:1 2:3" bash scripts/run_stage_gated_reprompt_xr1.sh start
source "$STORAGE_BS/robocasa_rollouts/recovery/xr1_stage_gate_latest.env" && export TRIAL_ROOT="$GATE_ROOT" && echo "$TRIAL_ROOT"
```

With only two GPUs, use `GATE_GPU_PAIRS="0:1"` (one worker, about 30–37 h: one resume).

4. Monitor:

```bash
bash scripts/run_stage_gated_reprompt_xr1.sh status
tail -f "$STORAGE_BS"/robocasa_logs/eval/xr1_stage_gate_2*_shard*_*.log
```

5. If the allocation ended before `complete`: resume in a new allocation with
the **same number of GPU pairs** (seeds are sharded by worker count; the
launcher refuses a different count):

```bash
export GATE_ROOT=$STORAGE_BS/robocasa_rollouts/recovery/xr1_stage_gate_<timestamp>
GATE_GPU_PAIRS="0:1 2:3" bash scripts/run_stage_gated_reprompt_xr1.sh resume
```

6. Analyze:

```bash
export GATE_ROOT="$TRIAL_ROOT"
bash scripts/run_stage_gated_reprompt_xr1.sh analyze
cat "$GATE_ROOT/analysis/summary.md"
```

Do not delete partial shards or logs. A `worker_failure.json`, a nonempty
`errors.jsonl` or any initial-state mismatch is a stop-and-inspect condition.

## Design choices for review

1. **Switch resets the policy context.** The deployable intervention on XR-1
   includes the context reset (its client cannot change language without it),
   so the end-to-end contrast is valid as is. It does not separate the prompt
   from the reset. The ladder did: its R and P branches both reset context.
   A `gated_context_reset` arm (reset at stage entry, keep the task
   instruction) would separate them; it is not included.
2. **Oracle stage.** The gate reads simulator predicates, so this tests the
   value of the intervention given a perfect stage estimator. A learned stage
   estimator is the next step.
3. **Gate at stage entry, once per stage.** The ladder's effect was largest at
   stage entry to +8 inferences, and stage entry needs no failure detector.
   Re-firing after a cap or a regression is not done.
4. **Step cap 1280** from the ladder's re-prompt completion times (above).
5. **`always_reprompt` included by default.** It adds 50% compute but still
   fits one allocation; drop it with `"arms": ["baseline", "gated_reprompt"]`.
6. **Primary = pooled whitelisted macro**, not per task, because even 100 seeds
   per task are underpowered for small per-task effects.
7. **Repeated-reset seeding** (as in the ladder) with explicit global-RNG
   seeding before each reset and an initial-state hash check, instead of a
   fresh env per rollout (≈ 60 s more per rollout).

## Local tests

```bash
/path/to/python-with-numpy -m unittest tests.test_stage_gated_reprompt
```

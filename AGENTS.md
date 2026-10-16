# Codex Project Instructions

## Execution environment

The user develops this repository locally on macOS, but runs RoboCasa experiments
on a remote GPU cluster. A shell prompt such as
`[ut06746@r21n11 robocasa]` is output pasted from the cluster; it is not the
local Codex shell. Unless a remote-access tool is explicitly available, provide
copy-paste cluster commands and diagnose the output the user pastes back.

The local checkout and cluster checkout are separate:

- Local checkout: `/Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa`
- Cluster checkout: `/gs/fs/tga-shinoda/felid/robocasa`
- GitHub remote: `git@github.com:FelixDou/robocasa.git`

After local implementation is committed and pushed, the cluster must run
`git pull origin main`. Never assume an unpushed local file is visible there.

## Cluster storage contract

- `/gs/fs/tga-shinoda/felid`: source repositories and code only.
- `/gs/bs/tga-shinoda/felid`: durable environments, checkpoints, downloaded
  datasets, generated datasets, rollouts, videos, logs, caches, and reports.
- `/work/6/ut06746`: legacy/work-area staging only; do not treat it as the
  durable source of models or datasets.
- `/tmp/ut06746`: node-local checkpoint copy for fast policy-server loading;
  it is ephemeral and must be recreated in each new allocation.

Do not write experiment outputs, videos, model weights, or downloaded datasets
under `/gs/fs`. A cluster `Disk quota exceeded` error is a stop-and-inspect
condition, not a reason to retry immediately.

## Required runbook

For every cluster experiment, server startup, dataset generation, evaluation,
or result-inspection request, read
[`docs/cluster_experiment_runbook.md`](docs/cluster_experiment_runbook.md)
before proposing commands. It contains the canonical exports, paths, ports,
OpenPI and RLDX server architecture, one/two-GPU launch patterns, monitoring,
and result commands.

## Operational defaults

- The main conda environment is
  `/gs/bs/tga-shinoda/felid/envs/robocasa_openpi`.
- Start and verify the policy server before starting a simulator client.
- Use `python -u` under `nohup` so progress reaches logs promptly.
- Put every background process log in
  `/gs/bs/tga-shinoda/felid/robocasa_logs/eval`.
- Put generated failure datasets in
  `/gs/bs/tga-shinoda/felid/robocasa_failure_datasets`.
- Put rollout results and videos in
  `/gs/bs/tga-shinoda/felid/robocasa_rollouts`.
- Disable W&B unless the user explicitly requests logging.
- Use unique ports and output directories for concurrent jobs.
- Preserve partial manifests and logs when a job stops; inspect and resume or
  merge rather than deleting evidence.
- Before declaring a dataset healthy, verify sample count, task coverage,
  per-task balance, errors, duplicate sample IDs, artifact existence, and that
  `failed_subtask` does not overlap the temporal completed-subtask prefix.

Do not place credentials, access tokens, or private keys in repository files.

## Frozen research tracks

The exact-replay track (Phase 2/3/3B, operator screen, timing pilot) is frozen
as of 2026-10-16; see [`docs/exact_replay_frozen.md`](docs/exact_replay_frozen.md).
Do not extend `robocasa/recovery/exact_replay/` or `scripts/exact_replay/`, and
do not propose new bit-exact replay experiments. New recovery experiments use
approximate state branching (the recoverability-ladder study).

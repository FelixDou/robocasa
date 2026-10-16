# SAFE RoboCasa365 causal recoverability experiment plan

**Update 16 October 2026:** the exact-replay track (Phases 2-3B-T) is frozen; see
`docs/exact_replay_frozen.md`. The timing diagnostic is superseded by a scaled
recoverability-ladder study with approximate state branching. The LaTeX plan
has not yet been revised to reflect this.

This directory contains the native LaTeX source for the scientific experiment
plan requested on 23 August 2026 and updated through 12 October 2026. It now
includes the completed Phase 1 frozen confirmation, state-context follow-up,
passing Phase 2 exact snapshot-replay validation, negative Phase 3 short-horizon
screen, completed Phase 3B training-horizon continuation, and completed
160-suffix higher-level operator screen. The latter passed every engineering
gate but continuation reached 22.5% stage completion versus 17.5% for every
intervention, with zero oracle routing gain. The next proposed experiment is a
common-stage-entry timing diagnostic (200 unique scientific trajectories).
Its collector and analyzer are not implemented; broader recovery scaling and
model training remain blocked.

## Build

From the LaTeX plugin root, run:

```bash
python3 scripts/compile_latex.py \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/reports/safe_robocasa365_research_plan_20260823/main.tex \
  --output-directory \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/reports/safe_robocasa365_research_plan_20260823/build
```

The final verified PDF is also copied to:

```text
output/pdf/safe_robocasa365_research_experiment_plan_latex.pdf
```

## Structure

- `main.tex`: preamble, document assembly, and reusable scientific callouts.
- `sections/`: modular scientific content for the formal framework, Phases 0-9,
  statistics, implementation, venues, and references.
- `figures/`: evidence figures copied from the checksum-backed SAFE report.
- `evidence/xr1_operator_screen_20261012/`: user-supplied cluster engineering
  and scientific exports, their checksums, and explicit provenance limits.
- `sections/phase_3br.tex`: completed operator results and the next proposed
  timing diagnostic; see also `docs/xr1_earlier_intervention_timing_study.md`.
- `t1lmr.fd`, `t1lmtt.fd`, `omllmm.fd`: local scalable font maps that keep the
  bundled Tectonic build reproducible when its optional network font bundle is
  unavailable.
- `build/`: compiler output.

Numerical continuation gates are proposed planning decisions, not official
venue requirements. The final powered sample sizes must be frozen through
blinded hierarchical simulation using pilot variance.

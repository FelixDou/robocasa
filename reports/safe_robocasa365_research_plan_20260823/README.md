# SAFE RoboCasa365 causal recoverability experiment plan

This directory contains the native LaTeX source for the scientific experiment
plan requested on 23 August 2026 and updated on 26 August 2026 with the completed
Phase 1 frozen confirmation, state-context follow-up, passing Phase 2 exact
snapshot-replay validation, and active Phase 3 opportunity pilot.

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
- `t1lmr.fd`, `t1lmtt.fd`, `omllmm.fd`: local scalable font maps that keep the
  bundled Tectonic build reproducible when its optional network font bundle is
  unavailable.
- `build/`: compiler output.

Numerical continuation gates are proposed planning decisions, not official
venue requirements. The final powered sample sizes must be frozen through
blinded hierarchical simulation using pilot variance.

# RSS 2027 Stage 1 draft (due 4 Dec 2026, 11:59pm AoE)

Built from `../main.tex` (unchanged) on 2026-10-17, reframed after the mock review: every failed rollout ends at the time limit, so a calibrated timer catches all failures and the paper's endpoint is now **warning lead over the timer at matched FPR**.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (pdflatex gives Type-1 fonts only; checked with pdffonts).
Set `\notesfalse` in the preamble to hide the red/orange notes for the submission build.

## Before submitting
- **Template.** The official RSS package (roboticsconference.org/docs/paper-template-latex.tar.gz) was blocked by the network here. The preamble reproduces the RSS template from memory (IEEEtran conference, `times`, `natbib[numbers]`, `plainnat`, anonymous author line, `\pdfinfo`). Diff it against the official `root.tex` and use its files if they differ.
- **Preregistration (time-critical).** Amend the confirmatory preregistration (seed 1300007) with the timer as comparator and mean lead as primary endpoint *before* opening that cohort (paper Sec. VII, E1).
- **Control rate.** The seconds assume 16 steps x 20 Hz = 0.8 s per inference; confirm for the evaluation.
- **Bibliography.** Two `% CHECK` comments in refs.bib (Foresight author order, RoboCasa365 authors). RSS desk-rejects hallucinated citations.
- **Page limit.** 5 pages of content + references on page 6 (limit: 5 + 1).

## New numbers in this draft (exploratory, opened cohort)
From `analysis/lead_time.py` (run: `python3 -I analysis/lead_time.py review_data/`) and `analysis/extra.py` (noise+timer null, 20 draws; curves). The convention matches review_controls.py: an alarm at the final inference counts. `lead_time_strict.py` repeats it without that alarm (the JSAI/mock-review convention, timer c = 0.894).

| Detector | Recall | FPR | mean lead (frac. of limit) |
|---|---|---|---|
| Timer (c = 0.903) | 1.000 | 0.047 | 0.090 |
| Time risk + timer | 1.000 | 0.050 | 0.125 |
| Noise + timer (20 draws) | 1.000 | 0.066 | 0.101 |
| SAFE + time risk (frozen) | 0.955 | 0.058 | 0.184 |
| SAFE + timer | 1.000 | 0.058 | 0.207 (gain +0.117, CI 0.060-0.178) |


## Update 2026-10-17 (per-task timer, figure)

- **Per-task timer** (`analysis/per_task_timer.py`, `.json`). A timer with one cutoff per task (longest calibration success plus a shared margin) gets lead 0.248 at FPR 0.074. SAFE + per-task timer gets 0.271 at 0.076, a gain of 0.023 [0.002, 0.047], below the 0.05 margin. SAFE + global timer minus per-task timer is -0.041 [-0.111, 0.027]. All of SAFE's gain over the global timer sits in the 14 tasks where time risk stays silent. The paper and `preregistration_amendment.md` now use the per-task timer as the primary comparator.
- **Qualitative figure** (Fig. 1): `figures/qualitative/`, shared copy in `/mnt/project-files/figures/qualitative/`. Frames are placeholders until you run the TSUBAME command in `figures/qualitative/README.md`.
- The method diagram and the cohorts table were cut to make room; the text was tightened to about 4 pages of content.

## Update 2026-10-17 (second mock panel): one per-task timer

The per-task timer is now P. It uses the 95th percentile of each task's development success durations, with the margin set by Proposition 1 on the 114 calibration successes (`analysis/quantile_timer.py`, `.json`; the same file is in the repo under `scripts/xr1_lead_time/`).

- P: lead 0.195 at FPR 0.045.
- SAFE + P: 0.202 at 0.042, a gain of 0.007 (CI −0.003 to 0.019).
- Sensitivity by quantile (gain of SAFE + P over P):
  - q = 1.0: 0.057
  - q = 0.90: 0.028
  - q = 0.75: 0.064 (CI 0.029 to 0.099)

This replaces the 0.248 / 0.271 / 0.023 numbers above. Fig. 1 now shows P's alarm (t = 38). The amendment (PR #3) names P.

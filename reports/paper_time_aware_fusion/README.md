# Time-aware fusion paper drafts

- `main.tex`: the original full draft (2026-10-17, before the timer reframing).
- `rss2027_stage1/`: RSS 2027 Stage 1 extended abstract (due 2026-12-04 AoE).
- `jsai2027/`: JSAI 2027 short paper.
- `review_data/`: small evidence files and `review_controls.py`.

## Data not in git

`review_data/calibration_predictions.jsonl` (6 MB) and
`review_data/event_predictions.jsonl` (22 MB) are frozen August-protocol prediction
dumps that the analysis scripts read. They are ignored here. Copy them from the cluster
run folder of the offline baseline suite
(`/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820/`)
or from an existing local copy before re-running the analyses.

Confirmatory numbers (E1) come from `docs/safe_robocasa_experiment_report.md` section 15.13.

## Update 2026-10-21: §15.14–15.15 in `main.tex`

`main.tex` stays the older draft framed around recall. It received only small, labelled additions:
- the success-only baselines' lead gain over a per-task timer (0.020–0.034 against 0.030 for SAFE);
- the atomic/composite split across all five scores and on RLDX-1;
- limitation (iv): one policy prospectively, and RLDX-1 offline.

Its E1 `\pending` markers predate this update and were left unchanged.

All three `main.pdf` files were rebuilt on 2026-10-21 as tectonic (XeTeX) previews, with `\usepackage{times}` swapped for TeX Gyre Termes and a no-op `\pdfinfo`, because this Mac has no pdflatex. Rebuild them with pdflatex before submitting.

`main.tex` (2026-10-22): one exploratory paragraph on the point-of-no-return labels (report §15.17), placed after "No early warning". The draft is still 6 pages.

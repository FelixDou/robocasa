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

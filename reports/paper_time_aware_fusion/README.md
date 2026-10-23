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

`main.tex` (2026-10-22, b): one exploratory paragraph after the point-of-no-return paragraph covers the FIPER re-analysis, its caveats and the SAFE fixed-horizon boundary case (report §15.18–15.19). The draft is still 6 pages.

`main.tex` (2026-10-23): the FIPER paragraph now uses the paired re-run (report §15.23) instead of the published-versus-computed comparison of §15.18.

## Update 2026-10-22: Hide-and-Seek and Amendment 2 (RSS and JSAI drafts)

Both drafts now report Hide-and-Seek (report §15.20 exploratory, §15.21 preregistered: +0.071 [0.022, 0.157] over P, H1-HS and H2-HS supported) and the learned-timer and first-inference controls. The framing is now "time explains separation; an early-separation sequence model warns earlier". `main.tex` (the older recall-framed draft) is unchanged.

- **JSAI.** The abstract is 140 words and the body is still 4 pages. Cuts: the cumulative-lead figure, the sensitivity and "where the lead comes from" paragraphs, the video frame-rate sentence, a shorter atomic/composite paragraph, and the next-steps sentence of the conclusion.
- **Bibliography.** `refs.bib` lists Hide-and-Seek as an ICML 2026 workshop paper (arXiv:2605.30834). Verify the venue before submission: the project page cites only the arXiv version.

## Update 2026-10-23: RLDX-1 Hide-and-Seek replication (report §15.22)

- **RSS.** The RLDX-1 paragraph in the evidence section now includes Hide-and-Seek + P: +0.083 [0.033, 0.138] at 0.04 FPR, and +0.052 [0.002, 0.106] at matched false alarms. Its controls gain at most 0.018, and the gain is atomic only. The abstract and conclusion each gained a short clause. Content still ends on page 5.
- **JSAI.**
  - The abstract (147 words) and the discussion now include the RLDX-1 numbers.
  - To keep the body at 4 pages, the "Warnings remain late" paragraph was folded into limitation (iii), and the FIPER and conclusion sentences were shortened.

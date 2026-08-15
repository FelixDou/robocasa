# SAFE on RoboCasa full report

This directory is the source package for the complete July 13-August 15, 2026 SAFE/RoboCasa technical report.

- `main.tex`: LaTeX source.
- `figures/`: selected figures from the checksum-verified artifact bundle plus deterministic 6 August update figures.
- `tables/`: machine-readable CSV evidence for rollout SAFE, the V2 representation screen, and the frozen parent-aware analysis.
- `generate_update_figures.py`: regenerates the SAFE-paper comparison and final frozen-prefix figure from the report tables.
- `chart_map.md`: source and interpretation for every included figure.
- `build/`: LaTeX compilation output (generated).

Compile from the repository root with the bundled LaTeX plugin:

```bash
python3 /Users/felixdoublet/.codex/plugins/cache/openai-bundled/latex/0.2.5/scripts/compile_latex.py \
  /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/reports/safe_robocasa_full_report_20260804/main.tex \
  --output-directory /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/reports/safe_robocasa_full_report_20260804/build \
  --json
```

The final validated PDF is also copied to `output/pdf/safe_robocasa_complete_experimental_report.pdf`.

# Qualitative figure (shared by the RSS and JSAI drafts)

One failed OpenCabinet test rollout (`ad40212bc300d3f410d0cf4c`, limit 66 inferences). Four frames sit above its SAFE score, the threshold tau_s = 0.648 and the timer line at 0.903 of the limit. One success that stays below the threshold is drawn dashed. SAFE alarms at inference 30 and the timer at 60, giving a lead of 36 inferences against 6.

- `make_qualitative_data.py` rebuilds `rollout_curves.csv` and `frames.json` from `review_data/`.
- `fig_qualitative.tex` is the figure. Use it with `\providecommand{\qualdir}{<path to this folder>}` and `\input{\qualdir/fig_qualitative}`. It shows grey placeholders until `frames/f1.png` to `frames/f4.png` exist.
- `preview.tex` builds the figure on its own.

## Getting the frames (run on TSUBAME)

The videos live only on TSUBAME. This writes four PNGs to `/gs/bs` (never `/gs/fs`):

```bash
OUT=/gs/bs/tga-shinoda/felid/robocasa_rollouts/figures/qualitative/frames
mkdir -p "$OUT"
/gs/bs/tga-shinoda/felid/envs/robocasa_openpi/bin/python -u - <<'PY'
import imageio.v2 as iio
v = "/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_prospective_matched_fpr_20260815_163700/prospective_test_collection/official_safe_all_retained/env_records/rollout_00000589--ad40212bc300d3f410d0cf4c.mp4"
out = "/gs/bs/tga-shinoda/felid/robocasa_rollouts/figures/qualitative/frames"
frames = list(iio.get_reader(v))
n = len(frames)
print("video frames:", n)
for i, step in enumerate([144, 464, 944, 1040], 1):  # inferences 10, 30, 60, 66 of 1040 env steps
    k = min(n - 1, max(0, round(step / 1040 * n) - 1))
    iio.imwrite(f"{out}/f{i}.png", frames[k])
    print(f"f{i}.png <- frame {k}")
PY
ls -l "$OUT"
```

Then copy them to the Mac (run on the Mac):

```bash
cd /Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa/reports/paper_time_aware_fusion/rss2027_stage1/figures/qualitative
mkdir -p frames && scp 'tsubame:/gs/bs/tga-shinoda/felid/robocasa_rollouts/figures/qualitative/frames/f*.png' frames/
```

Replace `tsubame` with your ssh host alias. If the video has a side-by-side layout or several cameras, crop before saving; the figure expects roughly square frames.

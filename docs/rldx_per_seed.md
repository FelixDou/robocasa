# RLDX-1 per seed and OOD baselines (RSS Table II, RLDX-1 block)

Purpose: report Table II's RLDX-1 block as its XR-1 block: single-model mean ± std over 3 seeds,
plus the four OOD baselines fused with P. Exploratory: the RLDX-1 cohorts are already open
(report §8.3, §15.15, §15.22); no rollout is collected and no policy runs.

Protocol (unchanged from §15.15 and §15.22, `rldx_replication.py`, `rldx_hide_and_seek.py`):
- **Data.** The RLDX-1 SAFE20 natural-rate run: 20 tasks, 740 training rollouts, 60 held-out
  calibration successes (3 per task), 200 test rollouts (5 + 5 per task). H_k is the longest
  training failure of task k.
- **Fit.** P (q = 0.95) and each score + P with `confirm_lead.fit_fast`: the held-out
  calibration successes set the FPR, the training failures choose the fused path.
- **Per seed.** Each seed's normalized running maximum is fused with P on its own, instead of
  the seed average:
  - SAFE heads (LSTM, independent MLP): seed s's running maximum of its normalized trajectory;
  - `score_cohorts` detectors: (running max of `per_seed_raw[s]` − location) / scale, with
    seed s's task normalization.
- **Gates.** `scripts/xr1_lead_time/rldx_per_seed.py` stops before any mean unless
  - every stored seed-averaged trajectory is reproduced (max diff ≤ 1e-9), and
  - P and every seed-averaged score + P equal the stored §15.15 / §15.22 results exactly
    (SAFE-LSTM, SAFE-MLP, Hide-and-Seek and its three controls).
- **OOD baselines (new on RLDX-1).** Mahalanobis, cosine k-NN, PCA-KMeans and RND are trained by
  `score_cohorts` on the training successes with 3 seeds, the same inputs and feature
  aggregation as Hide-and-Seek. They use `score_cohorts`' fixed default settings: there is no
  development search on RLDX-1, unlike XR-1's tuned Section 15.10 configurations. They have no
  stored result, so only the trajectory check applies; the paper must say they are untuned.

Outputs: per detector, the seed average and each seed (FPR, mean lead, gain over P,
TWA = (lead + 1 − FPR)/2), and mean ± std (ddof 1) over seeds.

## Commands (TSUBAME, CPU only)

`RLDXPS_SHA` is the full SHA of the pushed commit that contains this file.

### 0. Allocation

Scoring loads the features of all 1,000 RLDX-1 rollouts. On XR-1, the comparable ActProbe
rescoring (1,824 training rollouts) peaked at 95 GB, so ask for ample memory and check the
peak:

```bash
qrsh -g tga-shinoda -l cpu_80=1 -l h_rt=6:00:00
```

### 1. Session, pinned worktree, inputs, tests

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source "$(ls -d $STORAGE_BS/robocasa_logs/eval/rldx1_hide_and_seek_* | sort | tail -n 1)/paths.env"
echo "15.22 run: $RLDXHS_ROOT | 15.15 run: $RLDX_LEAD_OUT | raw: $RLDX_RAW"

export RLDXPS_SHA=<RLDXPS_SHA>
export RLDXPS_REPO=$PROJECT_FS/robocasa_rldx_perseed_${RLDXPS_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/time-aware-fusion-paper
test -e $RLDXPS_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $RLDXPS_REPO $RLDXPS_SHA
test "$(git -C $RLDXPS_REPO rev-parse HEAD)" = "$RLDXPS_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export RLDXPS_TAG=rldx1_per_seed_$(date +%Y%m%d_%H%M%S)
export RLDXPS_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$RLDXPS_TAG
export RLDXPS_LOG=$STORAGE_BS/robocasa_logs/eval/$RLDXPS_TAG
mkdir -p $RLDXPS_ROOT $RLDXPS_LOG
printf 'export %s=%q\n' RLDXPS_SHA "$RLDXPS_SHA" RLDXPS_REPO "$RLDXPS_REPO" RLDXPS_ROOT "$RLDXPS_ROOT" \
  RLDXPS_LOG "$RLDXPS_LOG" RLDXHS_ROOT "$RLDXHS_ROOT" RLDX_ROOT "$RLDX_ROOT" RLDX_RAW "$RLDX_RAW" \
  RLDX_LEAD_OUT "$RLDX_LEAD_OUT" SAFE_PY "$SAFE_PY" > $RLDXPS_LOG/paths.env

for f in $RLDXHS_ROOT/inputs/rldx_bundle.json $RLDXHS_ROOT/inputs/{train,calibration,test}_ids.json \
         $RLDXHS_ROOT/scores/{train,calibration,test}_predictions.jsonl $RLDXHS_ROOT/scores/normalizations.json \
         $RLDXHS_ROOT/rldx_hide_and_seek.json $RLDX_LEAD_OUT/rldx_lead_lstm.json $RLDX_LEAD_OUT/rldx_lead_indep.json \
         $RLDX_ROOT/calibration/lstm/lstm_seed{0,1,2}/normalized_scores.jsonl \
         $RLDX_ROOT/calibration/indep/indep_seed{0,1,2}/normalized_scores.jsonl $RLDX_RAW/manifest.jsonl; do
  test -s $f && echo "OK $f" || echo "STOP: missing $f"
done
sha256sum $RLDXHS_ROOT/inputs/*.json $RLDXHS_ROOT/scores/*_predictions.jsonl $RLDXHS_ROOT/scores/normalizations.json \
  $RLDXHS_ROOT/rldx_hide_and_seek.json $RLDX_LEAD_OUT/rldx_lead_{lstm,indep}.json \
  $RLDX_ROOT/calibration/{lstm,indep}/*_seed?/normalized_scores.jsonl $RLDX_RAW/manifest.jsonl \
  > $RLDXPS_LOG/frozen_inputs.sha256

cd $RLDXPS_REPO
PYTHONPATH=$RLDXPS_REPO CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_rldx_per_seed \
  tests.test_rldx_hide_and_seek tests.test_rldx_lead_replication 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only if every input is `OK` and the tests end with `OK` (5 tests). If only
`rldx_lead_indep.json` is missing, drop `indep=...` from both `--safe-dir` and
`--reference-safe` below and say so in the report.

### 2. Score the OOD baselines (CPU, background)

```bash
cd $RLDXPS_REPO
SCORE_ARGS=(
  --dataset-dir $RLDX_RAW
  --runtime-bundle $RLDXHS_ROOT/inputs/rldx_bundle.json --train-ids $RLDXHS_ROOT/inputs/train_ids.json
  --cohort calibration $RLDXHS_ROOT/inputs/calibration_ids.json
  --cohort test $RLDXHS_ROOT/inputs/test_ids.json
  --detectors mahalanobis cosine_knn pca_kmeans rnd
  --feature-aggregation last --seeds 0 1 2 --score-train
  --output-dir $RLDXPS_ROOT/ood_scores
)
PYTHONPATH=$RLDXPS_REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 \
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${SCORE_ARGS[@]}" > $RLDXPS_LOG/ood_score.log 2>&1 &
echo "score PID: $! on $(hostname)"
```

Monitor until it ends (peak memory, then the log):

```bash
while pgrep -f "score_cohorts.*ood_scores" > /dev/null; do echo "$(date +%H:%M) $(qstat -j $JOB_ID | grep -o 'maxrss=[^,]*')"; tail -n 1 $RLDXPS_LOG/ood_score.log; sleep 120; done; tail -n 15 $RLDXPS_LOG/ood_score.log
```

It prints `train 740; calibration 60, test 200`, then one `<detector> seed <s> done` line per
detector and seed (12 in all).

### 3. Per-seed metrics (seconds to minutes)

```bash
cd $RLDXPS_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_per_seed.py \
  --safe-dir lstm=$RLDX_ROOT/calibration/lstm indep=$RLDX_ROOT/calibration/indep \
  --inputs-dir $RLDXHS_ROOT/inputs --scores-dir $RLDXHS_ROOT/scores $RLDXPS_ROOT/ood_scores \
  --reference-hs $RLDXHS_ROOT/rldx_hide_and_seek.json \
  --reference-safe lstm=$RLDX_LEAD_OUT/rldx_lead_lstm.json indep=$RLDX_LEAD_OUT/rldx_lead_indep.json \
  --output $RLDXPS_ROOT/rldx_per_seed.json 2>&1 | tee $RLDXPS_LOG/per_seed.log
sha256sum -c --quiet $RLDXPS_LOG/frozen_inputs.sha256 && echo "INPUTS UNCHANGED"
```

Report back: the two `stored seed averages reproduced` lines, `reference P ... OK`, the six
`reference ... -> OK` lines, the four `no stored result` lines (OOD), the per-seed and
`mean+-std` lines, `REFERENCE OK` and `INPUTS UNCHANGED`.

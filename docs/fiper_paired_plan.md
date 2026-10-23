# Paired re-run: FIPER's own code with an elapsed-time method (plan)

**Status: exploratory re-analysis.** It removes the main caveat of report §15.18, which
compared our computed timer with FIPER's *published* numbers.
- This plan was committed before FIPER's code was cloned or run.

## Design

- **Code:** FIPER's public repository (github.com/utiasDSL/fiper, MIT licence), pinned at
  commit `13d79c5c3069def843e454787ff128defc249838`.
- **Data:** FIPER's released rollouts, the same files as §15.18: `fiper_data.zip`, sha256
  `00922af4…`. They were already checked with the restricted unpickler, and every file
  holds only numpy arrays and built-in containers. That makes FIPER's own `pickle.load`
  of them safe.
- **Our additions, applied by `scripts/published_eval/fiper_paired/setup_fiper.py`.** It
  asserts every anchor it edits and changes nothing else.
  1. **A `time` method** (`evaluation/method_eval_classes/time_eval.py`,
     `configs/eval/time.yaml`). Its score at policy step t (0-based) is t + 1. It uses no
     features and no model. FIPER's pipeline treats it exactly like any other method,
     with the same:
     - window sizes {1, …, 50};
     - threshold styles (`ct_quantile`, `tvt_quantile`, `tvt_cp_band`);
     - quantiles 0.90–0.99;
     - seeds {0, 1, 2, 42, 43};
     - metrics code;
     - summary and selection code. FIPER chooses each method's window and threshold
       style by the best TWA, averaged over tasks and quantiles.
  2. **Methods run:** `rnd_oe`, `entropy` (ACE) and their FIPER combinations
     `rnd_oe_and_entropy` (FIPER) and `rnd_oe_or_entropy`, plus `time`. The baselines
     not needed here (`rnd_a`, `similarity`, `tc`, `logpzo`) are dropped from the method
     list; each method is evaluated independently, so this does not change FIPER's own
     numbers. `time` is added to the summary's method filter.
  3. **A per-seed dump** (`run_fiper.py`) of each method's per-rollout first-alarm index,
     for every style, quantile and window, plus success labels and lengths. FIPER
     overwrites its results each seed and keeps the AND-combination only in memory; the
     dump changes no computation. It is written only when `FIPER_FIRST_ALARM_DIR` is set.
- **Outputs:**
  - FIPER's own summary CSV, which now includes a `time` row;
  - our analysis of the dumps, `scripts/published_eval/fiper_paired/analyze_paired.py`.

## Quantities (FIPER's definitions)

- Detection is the first step whose score / threshold > 1, with detection patience 0.
- **DT** is the detection step / (T − 1), averaged over detected failures (1.0 if none),
  where T is the maximum episode length in the dataset.
- **TWA** = ½ (Σ over detected failures of (1 − DT)/N_fail + TN/N_success).
- Both balanced accuracy, ½ (TPR + TNR), and plain accuracy are reported.
- **Lead (ours):** (L − 1 − detection step)/(T − 1) per failure, 0 if missed, averaged
  over failures. This is a lead-time version of FIPER's DT.

## Analysis

1. **Reproduction check.** For `rnd_oe_and_entropy` with FIPER's selection rule, compare
   per-task TWA / accuracy / DT with FIPER's Table 1.
   - Reproduced if every task is within ±0.05 on TWA and accuracy.
   - Otherwise the gaps are reported, and the paired comparison is still valid, because
     both methods ran in the same pipeline, but it no longer reproduces the published
     table.
2. **Same-pipeline table.** For `time`, FIPER, `rnd_oe` and `entropy`: TWA, accuracy, DT,
   FPR and lead, per task and averaged, each at the configuration FIPER's rule selects.
3. **Lead at matched FPR.**
   - For each task and seed, the timer's lead-versus-FPR curve over its 10 quantiles (at
     its selected configuration) is interpolated at FIPER's test FPR for each quantile.
   - The difference is FIPER lead minus timer lead, averaged over quantiles and seeds.
   - CI: a bootstrap over test rollouts, resampled within task by outcome (1,000
     replicates, seed 0), recomputing both curves in each replicate.
   - FIPER FPRs outside the timer curve's range are clamped and counted.
4. **Same-quantile comparison** (no interpolation): FIPER minus timer lead and FPR at
   each nominal quantile.

## Expectations, written before running

- FIPER reproduces Table 1 within ±0.05 on most tasks; small gaps can come from hardware
  and library versions.
- `time` under FIPER's rule:
  - accuracy close to §15.18's 0.83 average;
  - DT clearly later than FIPER's.
  - FIPER's rule may select a window above 1 for `time`, because windowed sums of t are
    still monotone; `ct_quantile` is the expected style.
- Lead at matched FPR: FIPER ahead on Stacking, PushT and Sorting, about equal on
  PushChair (published DT 0.27 for both), uncertain on Pretzel (10 + 10 test rollouts).

## Runbook

Run on tsubame with one GPU (RND-OE training), from a pinned worktree of this repository
and a pinned clone of FIPER. Outputs go to `/gs/bs`. FIPER's `data/` directory is a
symlink to a `/gs/bs` run folder, the released rollouts are made read-only, and hydra's
run directory is redirected to `/gs/bs`.

```bash
export STORAGE_BS=/gs/bs/tga-shinoda/felid PROJECT_FS=/gs/fs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export PAIRED_SHA=<full SHA of the commit containing this runbook>
export PE_REPO=$PROJECT_FS/robocasa_fiper_paired_${PAIRED_SHA:0:8}
export FIPER_REPO=$PROJECT_FS/external/fiper_13d79c5c
export FIPER_DATA=$STORAGE_BS/robocasa_datasets/external/fiper/extracted/data_all
export FIPER_RUN=$STORAGE_BS/robocasa_checkpoints/safe/fiper_paired_$(date +%Y%m%d_%H%M%S)

# 1. Our pinned worktree and tests.
git -C $PROJECT_FS/robocasa fetch origin
test -e $PE_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $PE_REPO $PAIRED_SHA
cd $PE_REPO && CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_fiper_paired 2>&1 | tail -1

# 2. FIPER, pinned (source code only, under /gs/fs).
mkdir -p $PROJECT_FS/external
test -e $FIPER_REPO || git clone https://github.com/utiasDSL/fiper.git $FIPER_REPO
git -C $FIPER_REPO checkout --detach 13d79c5c3069def843e454787ff128defc249838

# 3. Environment check (FIPER needs python>=3.10, torch, hydra-core, omegaconf, zarr,
#    pandas, scikit-learn, scipy, matplotlib, tqdm).
$SAFE_PY -c "import sys, torch, hydra, omegaconf, zarr, pandas, sklearn, scipy, matplotlib, tqdm; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"

# 4. Apply the additions, protect the released data, link the run folder.
chmod -R a-w $FIPER_DATA
mkdir -p $FIPER_RUN
python3 -I $PE_REPO/scripts/published_eval/fiper_paired/setup_fiper.py \
  --repo $FIPER_REPO --rollouts-root $FIPER_DATA --run-dir $FIPER_RUN
git -C $FIPER_REPO diff > $FIPER_RUN/fiper_additions.diff

# 5. Run FIPER's pipeline (5 seeds x 5 tasks; one GPU). Use the environment that passed step 3.
cd $FIPER_REPO
FIPER_FIRST_ALARM_DIR=$FIPER_RUN/first_alarms CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  nohup $SAFE_PY -u scripts/run_fiper.py hydra.run.dir=$FIPER_RUN/hydra > $FIPER_RUN/run_fiper.log 2>&1 &
echo "PID $!"

# 6. When it finishes: FIPER's own summary, then our paired analysis.
ls $FIPER_RUN/data/results/ ; ls $FIPER_RUN/first_alarms | wc -l     # expect 25 dumps
$SAFE_PY -I $PE_REPO/scripts/published_eval/fiper_paired/analyze_paired.py $FIPER_RUN/first_alarms \
  --output $FIPER_RUN/paired.json 2>&1 | tee $FIPER_RUN/paired.log
```

If step 3 reports a missing module, stop: a separate environment for FIPER is created
only after approval. Do not install packages into `vla_safe`.

## Post hoc extension (2026-10-23, after the paired run, before this analysis)

**What happened.** The paired run (`fiper_paired_20261022_180548`, 25 dumps) reproduced FIPER's
Table 1 within ±0.05 on all five tasks. The planned matched-FPR comparison did not work as
designed: every FIPER operating point at FIPER's selected configuration (50 of 50 task ×
quantile points) had a test FPR above the highest FPR the timer reaches on FIPER's quantile
grid (0.90–0.99). FIPER's average test FPR was 0.35; the timer's was 0.09. The interpolation
therefore clamped, and the printed "+0.25 lead for FIPER" is an upper bound, not a matched
comparison.

**Extension, added after seeing the clamping.**
- The pure timer is FIPER's `ct_quantile`, window 1, with score step + 1. It is a
  deterministic function of rollout lengths, so its curve is recomputed offline over
  quantiles 0.30–0.99 (`scripts/published_eval/fiper_paired/extend_timer.py`).
- Calibration lengths come from the §15.18 index (successful calibration rollouts); test
  lengths and outcomes come from the dumps.
- **Gate:** at 0.90–0.99 the recomputed timer must reproduce, rollout by rollout, the
  `time` method's alarms from FIPER's own pipeline, for every task and seed. A mismatch stops
  the analysis.
- FIPER's points are unchanged: its selected configuration, per quantile, averaged over seeds.
- The matched-FPR lead difference and its bootstrap are computed as in the original plan
  (1,000 replicates, seed 0); clamped points are still counted.
- This comparison was not pre-declared. It is reported as post hoc, next to the
  same-pipeline table (which was pre-declared).

# Does elapsed time matter in published failure-detection evaluations? (plan)

**Status: exploratory re-analysis of data released by other groups.** No new rollouts.
- Purpose: show that the timer result is not specific to our RoboCasa setup. A published
  evaluation is affected wherever successful and failed rollouts differ in length.
- This plan was committed before any external data was downloaded.

## Data sources

| Source | File | Size | Contents (from the authors' README or paper) |
|---|---|---|---|
| FIPER (Römer et al., NeurIPS 2025), linked from github.com/utiasDSL/fiper | `fiper_data.zip`, Google Drive folder `1VuI3eQmFHT2QKCSYZGQ2pAuuhwNJkGCu` | 5.29 GB | Calibration and test rollouts for Sorting, Stacking, PushT, Pretzel and PushChair. One `.pkl` per rollout; success and rollout ID in the file name. |
| SAFE (Gu et al., NeurIPS 2025), linked from github.com/vla-safe/SAFE | `pi0fast_droid_0510_all.zip`, Google Drive file `13z_cdwnaJota2iHkZbhYgVALujZwtM3b` | not shown (check before download) | Real Franka, π0-FAST, 13 tasks × 30 successes + 30 failures. Every rollout is stopped at the same per-task T regardless of outcome. Failure times were annotated from video. |
| SAFE | `openvla_widowx.zip`, Google Drive file `1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp` | not shown | Real WidowX, OpenVLA. Lower priority. |

- **Storage:** downloads go only to `/gs/bs/tga-shinoda/felid/robocasa_datasets/external/<source>/`,
  never under `/gs/fs`.
- **Handling:**
  - The archives are untrusted, and `.pkl` files can execute code when loaded.
  - They are opened only on the cluster, with
    `scripts/published_eval/inspect_external.py`. Its restricted unpickler admits only
    numpy arrays, numpy scalars and built-in containers. It refuses anything else and
    reports the class it refused, so an archive that needs other classes (for example torch
    tensors) is a stop-and-report condition, not something to work around.
  - Scripts run with `python -I` from a directory outside the extracted data.

## Questions

**Q1 (FIPER): is duration informative, and how does a timer score on FIPER's own metrics?**
- **Durations.** Per task: rollout length by outcome (calibration and test), and the
  ROC-AUC of length alone for failure.
- **Timer under FIPER's protocol.**
  - FIPER calibrates a "CP constant" threshold on M successful calibration rollouts as the
    1 − δ quantile of each rollout's maximum score. It then averages the results over
    1 − δ ∈ {0.90, 0.91, …, 0.99}.
  - Applied to the score s_t = t (elapsed policy steps), the threshold is the 1 − δ quantile
    of calibration success lengths, and the timer alarms once t exceeds it.
  - Report, per task and averaged over δ exactly as FIPER does:
    - TPR and TNR;
    - accuracy (TPR + TNR) / 2;
    - detection time DT = alarm step / T, with T the task's maximum episode length,
      computed for correctly flagged failures;
    - time-weighted accuracy TWA = ((mean over failures of 1 − DT for detected ones, 0 if
      missed) + TNR) / 2.
- **Comparison.** Set these next to FIPER's published Table 1. Those are published numbers,
  not recomputed, so the comparison is descriptive. A paired re-run of FIPER's own code is a
  later option.

**Q2 (SAFE real robot): a case where duration carries no information, and when failures are
decided.**
- Check that all rollouts of a task share one length. Then duration alone has ROC-AUC 0.5,
  so the timer gives no information here; this is the honest boundary of our claim.
- Use the annotated failure times: report the distribution of annotated failure time / T
  for failed rollouts, overall and per task. This is real-robot evidence on when failures
  become decided, to set beside our XR-1 point-of-no-return labels.

## Expectations, written before downloading

- **FIPER tasks.** Successes end early in at least some tasks (FIPER notes that cumulative
  scores do better "on tasks with a large difference in duration"). On those tasks, length
  alone gives ROC-AUC ≥ 0.8. The timer's accuracy is within 0.1 of FIPER's on those tasks,
  but its DT is worse (close to the success-length quantile divided by T).
- **SAFE real robot.** Lengths are constant within each task, so the timer is
  uninformative. The median annotated failure time is well before T.

## Steps

1. **Download.** Only after the user approves each file by name and size (FIPER first).
2. **Inspect.** Run `inspect_external.py` on a sample of files, then report the structure:
   keys, array shapes, the file-name pattern, and where success, length and T live.
3. **Write the analysis.** Only after inspection, write the Q1/Q2 analysis against the
   observed structure. Its interpretation rules stay as stated above.

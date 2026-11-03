# Mock review, RSS 2027 Stage 1 — 3 November 2026

Draft reviewed: commit `80780d61` on `claude/time-aware-fusion-paper` (5 + 1 pages, `main.pdf` built 2026-10-31).
Three independent reviewers, each reading only the paper (PDF, LaTeX, figure CSVs), with the official RSS 2027
Stage 1 rules: reviewers "will evaluate the paper's hypothesis and methodology"; experiments may be preliminary;
outcome is reject or invite to rebuttal, and the metareview becomes the Extension Charter. None of them saw the
30 October mock review.

| Reviewer | Expertise | Score (1–5) | Confidence |
|---|---|---|---|
| R1 | Failure detection (SAFE, FIPER, Hide-and-Seek, STAC, ActProbe) | 4, invite | 4 |
| R2 | Statistics, conformal prediction, experimental design | 4, invite (3 if charter items 1–6, 10 are not adopted) | 4 |
| R3 | Robot learning practice, VLAs, deployment, retry | 4, invite | 4 |

## Area-chair meta-review

**Recommendation: invite to rebuttal.** All three reviewers find the duration-only comparator important ("if this
paper only made every future monitor paper report a timer row, it would already have impact", R3), the input
ablations well designed, Proposition 1 correct (R2 checked the proof), and the reporting candid. All TWA and
Δℓ̄ identities in Tables I–II and the Fig. 3 numbers recompute from the shipped CSVs (R1, R2).

Consensus concerns, roughly in order of weight:

1. **No visible decision rules or power for Stage 2 (R1, R2, R3).** For a Stage 1 paper judged on methodology,
   this is the main gap. E1 and E2 name data but not the test, α, multiplicity, smallest effect of interest or
   what counts as support or falsification. R1 and R2 guess E1 is under-powered (CI half-width about 0.065 against
   a 0.052 effect); R1 and R3 estimate E2's effect at about 1 point. R2 notes that the rules exist only as LaTeX
   comments.
2. **H1 is not supported at matched FPR; headlines use the adaptive path (R1, R2).** The abstract leads with
   same-target gains (0.030, 0.071) from a path Prop. 1 does not cover. R2 argues that calibration noise alone
   (Beta(19, 362), sd ≈ 0.011 in FPR) moves a detector by about ±0.03 in lead, so the same-target gains largely
   reflect where calibration landed. Asked for: matched and fixed-path numbers first, or Learn-then-Test.
3. **Key numbers are in no table; the ensemble effect is unexplained (R1, R2, R3).** Every ensemble, matched and
   fixed-path number is prose only. A single Hide-and-Seek model gains 0.039 ± 0.006; the ensemble 0.071. For SAFE
   on unseen tasks the ensemble is *worse* than single models (0.017 vs 0.028). Are the ablations ensembled too?
4. **E2's comparator is a timer tuned for FPR, not for success (R2, R3; R1 asks for an oracle).** The relevant
   time-only reset rule is the restart cutoff that maximizes success within 1.25 H_k (optimal restarts, Luby et al.
   1993). Also requested: oracle reset, budget sweep, reset cost as a parameter, common random numbers for retries,
   a precise scene-restore definition.
5. **The gain is small and uneven (R1, R2, R3).** Hide-and-Seek + P is later than P on 16 tasks (several at
   −0.21, from the looser shared δ); five tasks carry 78%; matched gain 0 s on composite tasks, about 1.3 s overall.
   Per-task FPR reaches 4/10. Requests: per-task FPR distribution, Mondrian calibration, a practical threshold in
   seconds.
6. **Lead to the limit may reward predicting slowness (R1, R3).** 79% of failures were recoverable at the limit.
   Requests: time-limit truncation sensitivity (relabel at 0.8 H_k), how many failures succeed if continued, lead to
   the point of no return as a measured outcome.
7. **No fresh XR-1 confirmatory test of H1/H2 (R1, R2; recurring from 30 Oct, list C2).** The paper's own
   Limitations ("tested on a test set already used for SAFE", every analysis beyond H1/H2 after scoring) make
   reviewers read H1/H2 as exploratory.
8. **Baseline fidelity and missing comparators (R1, R3).** SAFE-MLP accumulates its per-step outputs in SAFE's
   paper; which form is u_t? SAFE's own band is not reported; time-varying bands may hide a timer; the Hide-and-Seek
   reimplementation (PCA-128) is not validated; ActProbe not run; a hazard/survival time-only model; LIBERO or
   another policy family; real-robot scope.
9. **Clarity (R1, R2, R3).** About 40 numbers in Sec. VI prose; "matched" used before it is defined; "loosened"
   when P is sometimes tightened; notation clashes (ρ_k/ρ_i, s/s_t, u/u_t, T/T_k).

New relative to 30 October: the success-optimal restart comparator for E2, reset cost, the ensemble question,
SAFE-MLP accumulation, time-limit truncation, and "only Hide-and-Seek carries over" vs SAFE-LSTM's 0.058.
Points 1, 2, 7 and the per-task budget concern recur from the last review: the text changes since then have not
removed them.

### Reviewer claims checked against our evidence

| Claim | Check | Consequence |
|---|---|---|
| E1 is under-powered (R1, R2) | **Wrong on our numbers.** R10 (`e2_freeze_20261029_205743`): with every path frozen, n = 20, power 1.00 SAFE, 0.94 Hide-and-Seek (about 0.86 at gain 0.052); type I 0/2000 under both nulls. Not in the paper (removed for style). | One sentence on power and the rule answers the strongest common concern. |
| E2 effect ~1 point, no MDE (R1, R3) | **Right, and we know it.** R9: H&S+P vs P reset Δ ≈ +0.011, CI half-width 0.019 with branching at 1,000 scenes; a test needs ~5,700 scenes; so H4 is estimated, not tested. | State that E2 is an estimate with its expected half-width. |
| Failures continued past H_k (R1 W5, R3 Q2) | **Exists.** RLDX-1 continuation pilot (report §15.26): 18/444 = 4% succeed by 1.25 H_k. Trimmed out of E2's text. | Restore one clause. |
| ActProbe not run (R1) | **Ran (R11) and removed by the author.** XR-1 +P +0.025, unseen never fires; ActProbe-T (t/(H−1) input) changes nothing; RLDX-1 +P 0.050. | One clause in Sec. VI-A explains the omission. |
| SAFE's own band missing (R1) | **Ran (R12 B2).** SAFE with native band: matched +0.006 (H&S 0.068 is in the text). | Add SAFE's number next to H&S's. |
| Budget-spreading timer undescribed / overfit (R1, R2) | **Exists (R12 B1)**: A_fixed and A_adaptive both fail to beat P at matched FP (−0.013/+0.002, +0.007/−0.008). Construction is one sentence in the paper. | One more clause on how it is built; R2's "oracle allocation on test" is not run. |
| SAFE-MLP score is a cumulative sum in SAFE (R1) | **True for SAFE** (arXiv 2506.09937: f_MLP = Σ_τ σ(g(e_τ))). Our `SafeMLP` defaults to the per-step output (`cumulative_mlp=False`, `--cumulative-mlp` opt-in); whether any XR-1/RLDX-1 run passed the flag is **not checkable locally**. | Confirm the training config on the cluster; state the form in Sec. IV-A (deviation from SAFE if per-step). |
| H&S was tested on a reused test set, so H1 is post hoc (R1, R2, R3) | **Partly.** Amendment 2 (`7abfd271`) was registered before any H&S score on these test rollouts; the Limitations sentence omits this because of the vocabulary rule. | Author choice: one clause ("specified before the test set was scored with it") vs keeping the style rule. |
| "Only Hide-and-Seek carries over" conflicts with Table II (R1) | **True.** SAFE-LSTM gains 0.058 ± 0.020 over G at 0.040 FPR; the sentence predates the R11 rows. | Rephrase: "only Hide-and-Seek clearly carries over; SAFE-LSTM keeps half". |
| "All six policy scores" vs seven (R1, R2, R3) | **True.** The 6 are the Fig. 3 scores (4 OOD, SAFE-MLP, H&S), computed before SAFE-LSTM was added. | Name them, or add SAFE-LSTM's atomic/composite split. |
| "At most 0.012" vs Table II's 0.013 (R1, R2) | Ensemble (0.012) vs single model (0.013 ± 0.003); unseen zero-input ensemble is 0.033. | "on seen tasks, as ensembles". |
| 0.111 vs 0.127 − 0.017 (R1, R2, R3) | Rounding: 0.111 is the directly computed difference (cross-task report). | Keep, or print 0.110. |
| RLDX-1 "0.10 to 0.115" but RND 0.099 (R1, R2) | True; rounding. | "0.099 to 0.115". |
| "Loosened" P is tightened for fixed-path SAFE (R1, R2) | True (16 vs 21 FP). | Say "matched (loosened or tightened)". |
| XR-1's RoboCasa365 checkpoint source (R1) | XR-1 abstract only says checkpoints "will be released". | Cite the checkpoint URL/model card (anonymity permitting). |
| `\notestrue` in the source; Paper-ID placeholder (R1, R3) | True; `main_submission_pdflatex.pdf` (29 Oct) is stale. | Submission build with `\notesfalse`, real Paper-ID, strip internal comments (run IDs, branches) if sources go public. |

### Suggested triage

**A. Text fixes, no new runs (before 4 December).**
1. Sec. VII: one sentence per experiment with its rule and size, using results already in hand: E1 tests both
   gains with 97.5% CIs (same target and matched), the binomial FPR rule and the frozen null test, power ≥ 0.94 at
   n = 20; E2 estimates the paired difference (expected about +0.01, half-width about 0.02 at 1,000 scenes), not a
   test. This conflicts with the author's style rule ("no completion criteria"); all three reviewers ask for it.
2. Abstract/contributions/conclusion: put matched (and fixed-path) gains first; say H1 at matched FPR is not yet
   established; soften "a warning that develops during the rollout".
3. A small H1/H2 table (ensemble: realized FPR, same-target Δℓ̄, matched Δℓ̄ with CI; ablations and nulls), or at
   least an ensemble block in Table II. Say whether the ablations are ensembled.
4. One sentence on the ensemble effect, or flag it as open (see B1).
5. Restore the 4% continuation result in E2; define scene restore, retry seeding, whether the no-reset run goes
   to 1.25 H_k, and reset cost (zero in simulation, stated as a limitation).
6. ActProbe omission clause; SAFE native-band number; budget-spreading timer construction.
7. Fix: "only Hide-and-Seek carries over", "six policy scores", "at most 0.012", "0.10 to 0.115", "loosened",
   notation clashes (rename ρ_i → π_i or similar), "matched" defined in Sec. III, H3's parenthetical result
   moved out of the hypothesis, SAFE's "Franka" robot, RESCUE qualifier, Table I caption ("constant threshold, not
   native bands").
8. State SAFE-MLP's score form once the config is confirmed (B2).

**B. Analyses on existing data (cluster or local), optional before submission, strong for the rebuttal.**
1. Ensemble size 1/2/3 for H&S, SAFE and the ablations, same-target and matched (explains 0.039 → 0.071).
2. Confirm SAFE-MLP per-step vs cumulative in the XR-1/RLDX-1 training configs; if per-step, optionally score the
   cumulative variant.
3. Success-optimal time-only restart cutoff per task for a 1.25 H_k budget, from training success durations,
   compared with P's c_k; feed it into the R9 E2 simulation as an extra arm.
4. Time-limit truncation: relabel recorded rollouts at 0.8 and 0.9 H_k and recompute the matched gains.
5. Per-task FPR distribution (max, 90th percentile) for P and the fused detectors; R2's oracle budget allocation.

**C. Author decisions for Stage 2 / the charter.**
1. Add decision rules to the PDF (A1) despite the style preference.
2. A fresh XR-1 confirmatory test with fixed paths (recurring; cheapest experiment per R2).
3. E2 arms: success-optimal timer reset, oracle reset, budget sweep (1.25/1.5/2.0 H_k), reset cost.
4. Outside RoboCasa365: LIBERO-10 with π0 or OpenVLA (R3), or reframe as a simulation-protocol result.
5. Mondrian per-task calibration in Stage 2 (20 successes per task are enough).
6. Whether to mention that the Hide-and-Seek hypothesis was registered before scoring.

---

## Reviewer 1

**Paper:** "Time Is the Baseline to Beat: Lead Time over a Timer as the Test for VLA Failure Monitors"

**Reviewer expertise:** runtime failure detection for robot / VLA policies (SAFE, FIPER, Hide-and-Seek, STAC, ActProbe, conformal runtime monitors).

---

### Summary

The paper's starting point is that VLA benchmarks often end an episode at the first success or at a fixed time limit (RoboCasa365, LIBERO). Under that protocol every failure is a timeout, so elapsed time alone is a very strong failure detector. The paper backs this up with numbers:
- A global timer reaches bACC 0.988 and a per-task timer P reaches 0.972 on the XR-1 test set (Table I).
- The best learned or OOD detector reaches at most 0.818.
- An LSTM given only zero inputs, so seeing only elapsed time, matches the feature-fed LSTM's ROC-AUC (0.888 vs. 0.878).

The authors therefore propose a different measure: the *warning lead* a monitor adds over a per-task timer at the same FPR. They combine any score with the timer through an OR rule. The two thresholds are calibrated along a nested path by split conformal, which gives a finite-sample marginal FPR bound when the path does not depend on the calibration successes (Prop. 1). The setup is XR-1 on 38 RoboCasa365 tasks, with 760 test rollouts and an adaptive path that the bound does not cover. Results:
- At the 5% target (realized FPR 0.061 vs. 0.055 for P), three-seed ensembles of SAFE and Hide-and-Seek add 0.030 and 0.071 of the time limit over P.
- At matched false alarms the gains fall to 0.015 (95% CI [−0.007, 0.041]) and 0.056 ([−0.014, 0.120]).
- The zero-input, shuffle and first-step ablations gain −0.003, −0.011 and −0.003 at matched FPR.
- On unseen tasks, compared against a global timer, Hide-and-Seek gains 0.127 (0.051 matched). SAFE's ensemble gains 0.017, about the same as random scores.
- The same pattern appears on 100+100 recorded RLDX-1 rollouts.
- On FIPER's released benchmark, a timer has the best balanced accuracy, but FIPER warns 0.146 of the episode earlier at FIPER's own (high) FPR.
- In seconds, Hide-and-Seek + P warns 2.6 s earlier than P on average, and about 1.3 s earlier at matched false alarms (5.0 s on atomic tasks, 0 s on composite ones).

Stage 2 plans two experiments: E1 tests H3 on fresh RLDX-1 rollouts with fixed paths, and E2 tests in closed loop whether resetting at the earlier alarm raises success within a 1.25·H_k budget (H4).

---

### Strengths

1. **Important, uncomfortable observation for the field.** In success-terminated protocols, a duration-only comparator beats every published-style detector on bACC and stays competitive on TWA (Table I: P has TWA 0.585, above SAFE-MLP and every OOD detector). I checked the prior work this rests on:
   - SAFE (arXiv:2506.09937) does recognize the length shortcut. In its App. B.5 it scores LIBERO rollouts at the per-task minimum length, and its Franka rollouts run to a fixed per-task horizon. But SAFE fixes the shortcut only for ROC-AUC. Its timeliness/bACC evaluation has no timer comparator.
   - FIPER (arXiv:2510.09459) says that accumulated scores (STAC-style) flag failures "primarily due to their greater length". Its baselines are PCA-KMeans, logpZO, RND-A and STAC, with no duration baseline.
   - Hide-and-Seek's arXiv HTML (2605.30834) and ActProbe's (2606.08508) report no duration-only baseline. ActProbe even feeds the normalized timestep t/(T−1) as an input and never ablates it.

   So the claim in Sec. II that "none reports an elapsed-time detector" holds for every paper I checked. A short paper that makes this comparator standard would be useful.
2. **The metric is well chosen and positioned honestly.** The paper states that lead is just one minus FIPER's DT, and that it determines Hide-and-Seek's TWA at equal FPR. I checked Hide-and-Seek's App. B.3 TWA formula: TP earliness averaged over all |P| failures plus TNR. With failures running to H_k, this matches TWA = ½(ℓ̄ + 1 − FPR) in Sec. VI-A. The paper presents its novelty as the comparator, not the metric, which is the right framing.
3. **Ablations go after the main confound.** The zero-input, shuffle and first-step retrainings separate a model learning elapsed time or the task's typical trajectory from evidence specific to the rollout. That is the right control, and it is uncommon in this literature.
4. **Statistical reporting is candid.** The paper gives matched-FPR comparisons that favor the timer, paired task-level bootstrap CIs, a label-permutation-style null (100 noise and resampled-score fusions), and the 1–10% matched sweep (Fig. 3b). It says openly that both matched CIs include zero and that the adaptive path is outside Prop. 1. Its limitations also admit the test-set reuse (Hide-and-Seek was evaluated on a test set already used for SAFE). That transparency is exactly what Stage 1 is meant to reward.
5. **Prop. 1 is correct, stated with care, and limited correctly.** I checked the proof: the alarm sets are nested, V_i is defined as the first silent point on the path, the standard rank argument applies, and treating the disabled case as V_(r) = ∞ is handled properly. The text explains that the bound is marginal over tasks and that a per-task bound needs at least 19 calibration successes per task.
6. **Checks outside the main benchmark.** The paper re-runs FIPER's released code with a timer added and reports the result for FIPER even where it is favorable to FIPER. It also runs a second VLA and a small irreversibility annotation with a second annotator (κ = 0.80). These make the scope of the claim easier to judge.
7. **Stage 2 targets the right question.** H4 (does an earlier alarm raise success when acting on it?) is what makes "lead" matter at all. The E2 design branches from a shared no-reset run with matched accounting, which is a sound and efficient paired design.

---

### Weaknesses

**W1. H1 is currently not supported, and the headline numbers come from the condition the paper itself calls favorable to the fused detector.**
- The abstract, contribution 3 and the conclusion lead with 0.030 and 0.071. These come from adaptive-path calibration (Sec. IV-D, "Adaptive and fixed paths"), which Prop. 1 does not cover, at a realized FPR of 0.061 against P's 0.055.
- H1 as written (Sec. V) explicitly requires the gain "also when P is loosened to the fused detector's realized false alarms". At matched FPR both 95% CIs include zero: SAFE [−0.007, 0.041], Hide-and-Seek [−0.014, 0.120]. These CIs also exclude calibration and matching uncertainty (Sec. VI-A, last sentence), and "matched gains stay uncertain when calibration is also resampled" (Sec. VI-C).
- The honest Stage-1 verdict is therefore "H1 not confirmed at matched FPR for either detector". The paper says this in places ("neither clearly above zero"), but elsewhere it reads more strongly than the evidence:
  - Contribution 3 says "a warning that develops during the rollout".
  - The conclusion says the ablations "trace Hide-and-Seek's warning to the rollout".
  - The fixed-path SAFE result (0.007 at the same target) is not in the abstract.
- The abstract should lead with the matched, fixed-path numbers.

**W2. Matched gains, ensemble results and CIs, the numbers the claims rest on, are in no table.**
- Table II reports only same-target gains of *single* models, with realized FPRs ranging from 0.040 to 0.079 (unseen columns).
- Under the paper's own argument, Δℓ values at different FPRs are not comparable. Yet Table II bolds and underlines them as a ranking. For example, Mahalanobis at 0.079 FPR on unseen tasks is ranked against SAFE-LSTM at 0.040.
- The following appear only in running text:
  - every ensemble number (0.030, 0.071, 0.127, 0.017, 0.083, 0.034)
  - every matched number (0.015, 0.056, 0.051, −0.079, 0.052)
  - the few-shot-P unseen results (0.088–0.099)
  - the adaptive-timer comparator (−0.008, 0.063)
  - the fixed-path results
  - the time-varying-band result (0.068)
  - the seconds
- A reader cannot check the central claims against a table. A Stage-2 table should give, for every detector and ablation: realized FPR, same-target Δℓ, matched Δℓ with CI, single model and ensemble.

**W3. The large ensemble effect is unexplained, and it is applied unevenly.**
- Hide-and-Seek's gain nearly doubles from a single model (0.039 ± 0.006, Table II) to a three-seed ensemble (0.071). That is about 5 standard deviations above the single-model mean, from averaging three models.
- For SAFE on unseen tasks the ensemble is *worse* than the single-model mean (0.017 vs. 0.028 ± 0.009).
- The paper concludes "the ensemble is part of its warning" without a mechanism.
- Two candidate explanations should be checked:
  - **Normalization.** Eq. (1) averages per-seed z-scored running maxima. This can change the score's scale and therefore which point the (u, s) grid picks.
  - **Selection.** The adaptive path chooses (u, s) by largest calibration-failure lead, and ensembles may simply interact with that selection differently.
- It is also unclear whether the ablation numbers in Sec. VI-C (−0.003/−0.003/−0.011) are ensembles. If the ablations are single models and Hide-and-Seek is an ensemble, H2 compares unlike things.
- Eq. (1) defines the policy score as the ensemble, while Tables I–II report single models. The notation and the protocol disagree.

**W4. The gain is small in absolute terms and comes from net redistribution across tasks, not a uniform improvement.**
- Fig. 3a (data in `per_task_gains.csv`) shows Hide-and-Seek + P alarming *later* than P on 16 of 38 tasks. Seven of those lose about 0.21 of the limit, and three tasks show an identical −0.2128. This pattern fits a looser shared margin δ on tasks where the score never fires.
- Five tasks carry 78% of the net gain. At matched false alarms the composite-task gain in seconds is 0 s (Sec. VI-C), so the average matched warning is about 1.3 s, under two 0.8-s inferences.
- This matters for H4. On 16 tasks the fused monitor delays the alarm, and E2 will average over tasks.
- The shared δ is the culprit, but the "budget-spreading" timer alternative is described in only one sentence ("chosen as adaptively as the fused path"). The paper should state a practical-significance threshold in seconds and report the per-task distribution with CIs. It should also consider per-task or Mondrian allocation of the FPR budget (see also W8).

**W5. What counts as a failure depends on the arbitrary time limit, so lead may partly reward predicting slowness.**
- The irreversibility study (Sec. VI-D) finds that 79% of annotated failures reached no point of no return before the limit, "mostly slow or near-successful attempts".
- Those failure labels are therefore partly artifacts of H_k. A monitor that alarms early on a slow but progressing attempt earns lead credit, even though resetting might destroy an attempt that would succeed given slightly more time.
- The introduction turns this into an argument for lead over recall. The same fact also argues that lead is not obviously valuable. On the 10 irreversible failures, the fused monitors warn before the point of no return in only 2 cases, where P warns in 0.
- The paper needs:
  - **A time-limit sensitivity analysis.** Truncate the recorded rollouts at, e.g., 0.75·H_k and 0.9·H_k, relabel, and recompute every gain. Extend some failed rollouts beyond H_k to measure how many succeed.
  - **A pre-registered statement** of what result in E2's no-reset arm would falsify the premise.

**W6. The value hypothesis H4 has no pilot and no power analysis, and the size of E2 looks risky.**
- The expected effect of 1.3–2.6 s of extra lead (0.015–0.07·H_k) on retry success within a 1.25·H_k budget is probably a fraction of a percentage point. Only scenes where the two monitors alarm at different times contribute to the paired difference. Hide-and-Seek + P also adds false-alarm resets: 0.061 vs. 0.055 FPR at the same target.
- 1,000 paired scenes may be badly underpowered for "higher success rate".
- Missing from the E2 plan:
  - a minimum detectable effect (MDE) calculation;
  - the expected share of discordant scenes;
  - the source of stochasticity in a retry from the restored scene (policy noise seed? scene perturbation?);
  - whether the retry restores the initial state or the state at the alarm;
  - a budget sensitivity sweep (1.25 is arbitrary; at 1.25·H_k − 0.775·H_k ≈ 0.48·H_k, many tasks' successes cannot fit in the remaining budget);
  - an oracle upper bound (reset at the earliest inference for scenes known to fail) to show how much headroom any monitor has.
- Since H4 is the paper's main reason to care about lead, Stage 1 should at least size it credibly.

**W7. Test-set reuse, post-hoc hypotheses and multiplicity are acknowledged but not resolved by the Stage 2 plan.**
- The limitations say Hide-and-Seek "was tested on a test set already used for SAFE", and every analysis beyond H1/H2 came after the test was scored.
- H1 is phrased existentially ("a policy-internal score"), and the paper evaluates 7 learned scores × {same target, matched} × {seen, unseen}.
- H3 picks SAFE-*LSTM* for RLDX-1 because "SAFE's MLP head adds none there" (Sec. V). That is a hypothesis written after inspecting the recorded RLDX-1 data.
- E1 gives a fresh test only for RLDX-1. **No fresh confirmatory XR-1 test of H1/H2 is planned**, even though the XR-1 checkpoint is available and rollouts are cheap.
- The PDF does not state E1's decision rule: α, one- vs. two-sided, how the two detectors × two comparisons are corrected, or which outcome counts as confirming or falsifying H3. It also gives no power analysis. With 20 successes and 20 failures per task over 20 tasks, the CI half-width will probably be similar to the current ±0.065, while the RLDX-1 matched estimate is 0.052, so E1 may well be inconclusive by design.

**W8. Calibration choices drive the comparison, and per-task false alarms are heterogeneous.**
- The adaptive path picks (u, s) by calibration-failure lead with each grid point at its own λ̂. This inflated the expected test FPR from 0.047 to about 0.055 (Sec. IV-D), and the observed fused FPR is 0.061.
- The guarantee is marginal across tasks. One test task had 4 of 10 successes alarmed (40% FPR), which would be unacceptable in deployment.
- P's ρ_k is the 95th percentile of about 27 training successes per task on average. For tasks near the 5/50 success boundary it rests on roughly 5 successes, which makes it essentially the maximum.
- Missing: the per-task FPR distribution (max and quantiles) for P and the fused detectors, and the number of training successes per task. The paper should also discuss Mondrian (task-conditional) calibration or Learn-then-Test (which it cites) for choosing the path with a guarantee.

**W9. Fairness of the baseline implementations, and missing comparators that already use time or are action-based.**
- **SAFE-MLP's score is a cumulative sum.** I checked SAFE's paper (Sec. 4.2): SAFE-MLP *accumulates* per-step sigmoid outputs, so its raw score grows with t, and FIPER's remark about accumulated scores and length applies. It is unclear whether u_t in Eq. (1) is the per-step output or the accumulated score. If it is the accumulated score, "SAFE" is itself partly a timer, and the H1/H2 interpretation changes.
- **Constant thresholds replace both methods' own bands.** Every score is thresholded at a constant level instead of with SAFE's and Hide-and-Seek's time-varying functional-conformal bands (I confirmed both papers use time-varying bands). The paper reports the original band only for Hide-and-Seek (matched gain 0.068), not for SAFE.
- **The bands themselves may hide a timer.** A time-varying band calibrated on success-terminated rollouts has no calibration data beyond the longest success. Hide-and-Seek's App. B.2 pads calibration trajectories by repeating the final score. Such a band therefore contains an implicit timer, which may explain part of the timeliness these methods report. This is directly relevant to the thesis and should be analyzed.
- **The Hide-and-Seek reimplementation is unvalidated.** It runs on a 128-dimensional PCA of the hidden state and is not shown to reproduce the original results, unlike the FIPER re-run.
- **ActProbe is not run.** It is cited, runs on RoboCasa, is action-only (no multi-sample requirement), and feeds the normalized timestep directly. Its time input makes it the most natural "time-aware learned monitor" competitor, and evaluating it with and without that input would directly test the thesis.
- **Multi-sample action-uncertainty monitors are excluded.** FIPER's ACE, cluster entropy and STAC-single are left out because XR-1 executes full chunks. But a flow/diffusion VLA can be sampled several times per inference, so this is deferred (E3), not infeasible.
- **The time-only comparators are weak.** The strongest time-only comparators are a percentile timer and an adaptive budget-spread timer. A per-task hazard or survival model on elapsed time, or time plus initial-scene features, would be a stronger time-only baseline. Without it, "beats the timer" may overstate what the policy adds.

**W10. The unseen-task and second-policy claims are weaker than the text states.**
- **"Only Hide-and-Seek's gain carries over" conflicts with Table II.** Contribution 3 and Sec. VI-D both say this. Yet Table II shows SAFE-LSTM gaining 0.058 ± 0.020 over G at 0.040 FPR, *below* target, and the OOD detectors gaining 0.044–0.053 at 0.071–0.079 FPR. The text later concedes SAFE-LSTM carries over at "half of Hide-and-Seek's".
- **G is a weak comparator on unseen tasks.** For XR-1, G's lead is 0.126 vs. P's 0.225.
- **On RLDX-1, G is degenerate.** It has FPR 0.000 and lead 0.026, so "gains" of 0.10–0.20 over it are mostly what any detector would get.
- **The few-shot P comparator is the meaningful one but is underreported.** It is built from 1–5 successes of the new task, and Hide-and-Seek's 0.088–0.099 gain over it appears only in the text without matched values or CIs.
- **The RLDX-1 evidence is thin.** There are 5 successes and 5 failures per task over 20 tasks, with no CIs. The 20 tasks' selection criterion, the size of the calibration set and the origin of the "recorded" rollouts are not given. "The ordering holds" (abstract) needs CIs.

**W11. The FIPER-benchmark comparison is hard to interpret.**
- "the timer has the best balanced accuracy under FIPER's selection rule (0.81 against 0.78) at 0.09 FPR, against 0.35 for FIPER" is ambiguous. I read it as FIPER's FPR being 0.35.
- FIPER's own paper says it selects window and threshold type to maximize TWA, so "selection rule" should be defined.
- The favorable-to-FIPER comparison is made at FIPER's own FPR of 0.06–0.56, which is not a deployable operating point.
- Missing: lead over a timer at matched FPR on FIPER's tasks, how the timer was calibrated there (successes per task), and what "reproducing its results" means quantitatively.

**W12. The core numbers are densely packed, which hurts clarity.** Many key quantities are defined in one clause and then reused, e.g., "matched", "adaptive path", "ensemble" vs. "single", "same target". Sec. VI-C/D packs about 40 numbers into prose. A Stage-2 reader will need a results table per hypothesis and a short protocol box listing the data splits, which statistic tests which hypothesis, and the decision rules.

---

### Questions to the authors

- **Q1.** In Eq. (1), is u_t^(j) for SAFE-MLP the per-step sigmoid output or SAFE's accumulated score? If accumulated, how does the running max differ from a timer scaled by per-step probabilities, and do H1/H2 still hold with the per-step output?
- **Q2.** Are the ablation results in Sec. VI-C (−0.003, −0.003, −0.011 matched; 0.070/0.059/0.060 below Hide-and-Seek) three-seed ensembles like Hide-and-Seek? Which numbers in the abstract are ensembles and which are single models?
- **Q3.** Why does ensembling nearly double Hide-and-Seek's seen-task gain (0.039 → 0.071) but lower SAFE's unseen-task gain (0.028 → 0.017)? How do the gains change with 1, 2, 3 and 5 seeds, and with averaging raw scores vs. running maxima?
- **Q4.** How exactly is P "loosened" to match false alarms? Which δ is chosen when several give the same count? Is the matching global or per task?
- **Q5.** In Fig. 3a, why do three tasks have exactly −0.2128 and several others about −0.21? Is this the shift in the shared margin δ between P and the fused path? What does a per-task margin (or Mondrian calibration) do to the 16 tasks that lose?
- **Q6.** What is the zero-input vs. feature-LSTM ROC-AUC (0.888 vs. 0.878) in the introduction computed on: which split, which model (Hide-and-Seek's LSTM or SAFE-LSTM), and which score (final, running max)? Under SAFE's equal-length protocol (App. B.5), what are the ROC-AUCs?
- **Q7.** What per-task FPR distribution do P and the fused detectors have on the test set (max, 90th percentile)? How many tasks exceed 10%?
- **Q8.** For E2: does the retry restore the initial scene or the state at the alarm? Where does the retry's stochasticity come from? What success-rate difference do you expect, what share of scenes is discordant, and what is the MDE at 1,000 scenes?
- **Q9.** For E1: what decision rule, α, sidedness and multiplicity correction confirm or falsify H3? How were RLDX-1's 20 tasks chosen? How many rollouts per task are needed for the tasks with lower success rates (the "about 2,500" total looks low if some tasks have extreme success rates)?
- **Q10.** Does your Hide-and-Seek reimplementation on a 128-dimensional PCA reproduce the published LIBERO numbers? Why PCA rather than the full 1,024-dimensional features?
- **Q11.** With SAFE's own functional conformal band (not only Hide-and-Seek's), what are the fused matched gains? Do these time-varying bands, calibrated on early-terminating successes, behave like an implicit timer past the longest calibration success?
- **Q12.** On the FIPER benchmark, how was the timer calibrated (successes per task)? What is FIPER's lead over the timer at matched FPR per task?
- **Q13.** How sensitive are the conclusions to H_k? If you truncate rollouts at 0.8·H_k and relabel, do the gains over P keep their sign?
- **Q14.** What is the "budget-spreading" timer exactly, and how was it chosen "as adaptively as the fused path"?

---

### Required for the Extension Charter

1. **Fresh confirmatory XR-1 test of H1 and H2 with fixed paths.**
   - Use a new XR-1 calibration and test set (not reused).
   - Pre-specify the detector(s): Hide-and-Seek (primary) and SAFE (secondary). Fix the paths on existing data so Prop. 1 applies.
   - State the decision rule (α, sidedness, multiplicity correction across detectors and same-target/matched) before running.
   - Report matched gains with CIs that also resample calibration.
2. **E1 (H3) fully specified in the paper.**
   - Decision rule, power/MDE analysis and the RLDX-1 task-selection criterion.
   - Sample sizes per task justified by the CI widths observed in Stage 1.
   - A fixed-path random-score null.
3. **E2 (H4) sized and bracketed.**
   - Power/MDE with the expected share of discordant scenes.
   - Arms: no-reset; P reset; Hide-and-Seek + P reset; matched-FPR P reset; an oracle upper bound.
   - Retry protocol and source of stochasticity.
   - A sweep over the reset budget (e.g., 1.25, 1.5, 2.0·H_k) and a reset cost.
   - Separate accounting for false-alarm resets of would-be successes.
   - Pre-registered statement of which outcome falsifies H4, including "no-reset ≥ both".
4. **A main results table per hypothesis.** For every learned detector and every ablation, single model and ensemble, give: realized FPR, same-target Δℓ, matched Δℓ with CI, Δℓ in seconds and in inferences. Include full lead-vs-FPR curves with CI bands (extending Fig. 3b) and the area between curves.
5. **Ensemble analysis.** Gains vs. ensemble size (1–5) for Hide-and-Seek, SAFE and the ablations. Ablations must be ensembled exactly like the detector they control for.
6. **Additional comparators.**
   - ActProbe with and without its normalized-timestep input.
   - At least one multi-sample action-uncertainty monitor (FIPER-ACE or STAC-single with sampled chunks).
   - SAFE and Hide-and-Seek with their original time-varying conformal bands, fused and unfused.
   - A stronger time-only baseline: a per-task survival/hazard model on elapsed time, and elapsed time plus initial-scene features.
7. **SAFE-MLP score definition.** Clarify the score (per-step vs. accumulated) and report both.
8. **Time-limit sensitivity and label validity.** Truncation/relabeling at shorter limits. Extend a sample of failed rollouts beyond H_k to estimate how many would succeed. Report results separately for irreversible and recoverable failures.
9. **Per-task calibration analysis.** Per-task FPR distribution. A Mondrian/per-task calibrated variant (≥19 calibration successes per task) or LTT-chosen paths. Per-task allocation of the shared margin δ, with its effect on the 16 tasks that currently lose lead.
10. **Unseen-task results against the few-shot P comparator** (1–5 successes) in a table, with matched values and CIs. Treat G on RLDX-1 as degenerate and flag it as such.
11. **Irreversibility study scaled up.** At least about 150 failures, two annotators on the full set, and lead to the point of no return as a pre-specified secondary metric (currently E3).
12. **Validation of reimplementations.** Show that the Hide-and-Seek reimplementation (and its PCA input) reproduces published results on a public benchmark. Give quantitative FIPER reproduction numbers and a matched-FPR timer comparison per FIPER task.
13. **A pre-stated practical-significance threshold.** State in seconds what minimum lead is useful for the retry policy, and evaluate against it.
14. **Released evaluation protocol.** Code for the timer comparators and the lead/matched computation, so other monitors can be evaluated under the proposed test (the paper's main claimed contribution).

---

### Minor issues

1. **Numbers and wording that do not match:**
   - **"Only Hide-and-Seek carries over" vs. Table II.** Contribution 3 / Sec. VI-D say only Hide-and-Seek's gain carries over to unseen tasks, which conflicts with Table II (SAFE-LSTM 0.058 at 0.040 FPR) and with the text's own "SAFE's LSTM head carries over better … still half".
   - **0.111 vs. 0.110.** "0.111 more than SAFE": 0.127 − 0.017 = 0.110. Rounding is possible, but use consistent precision.
   - **"All six policy scores."** Sec. VI-C says "All six policy scores", but there are seven learned scores (4 OOD + SAFE-MLP + SAFE-LSTM + Hide-and-Seek). Say which six.
   - **"0.10 to 0.115."** The RLDX-1 unseen "SAFE-LSTM and the OOD detectors gain 0.10 to 0.115", but RND is 0.099.
   - **"At most 0.012."** The abstract's bound for input ablations holds for the ensembles. Table II's single zero-input model gains 0.013 ± 0.003 (seen) and 0.020 ± 0.008 (unseen), and the unseen zero-input ensemble gains 0.033. Say "on seen tasks, as ensembles".
   - **Where the arithmetic does check out:** I verified every TWA entry in Table II against ½(ℓ̄_timer + Δℓ + 1 − FPR) using the footnoted timer leads, and all are consistent. The Fig. 3 caption's 0.160/0.039 and "78% from five tasks" match `per_task_gains.csv`, and SAFE's 2–10% matched average (0.013) matches `matched_sweep.csv`.
2. **Results stated outside a table or in the wrong place:**
   - **H3 contains a result.** H3 includes "(SAFE's MLP head adds none there)". Results do not belong in a hypothesis statement.
   - **Cross-reference to the introduction for evidence.** Sec. VI-C ("Is the extra warning learned elapsed time?") cites "Sec. I" as evidence for the zero-input separation result. The number appears only in the introduction, with no table or setup.
3. **Notation:**
   - ρ_k (95th-percentile duration of task k) and ρ_i (normalized length of success i) share a symbol.
   - T is both the first alarm time and part of the control-step limit T_k.
   - "c" is used both for G's fraction and for a per-task c_k.
4. **Labels and naming:** Table row labels are inconsistent ("Class." in Table I vs. "Cl." in Table II), as are "Hide&Seek" vs. "Hide-and-Seek". Abstract: "SAFE failure score" should say which head (MLP).
5. **Fig. 1:**
   - The fused lane draws the fused timer at P's c_k (t = 38), though the caption admits the fused timer can be looser.
   - The example is a validation rollout of an atomic task, where gains concentrate. Label it as illustrative and selected.
   - The success curve, peaking just under τ_s at t = 33, looks hand-picked; say how it was chosen.
6. **Fig. 3:**
   - Tick and legend fonts are very small (`\tiny`).
   - The legend in (a) sits over the plotting area. In (b) the legend overlaps the region the Hide-and-Seek curve passes through, and the y-range is mostly empty.
   - (b) shows no CI bands, although the claims are about uncertainty. The data CSV extends to α = 0.2 but the plot stops at 0.10.
7. **Undefined terms:**
   - "Matched" for unseen tasks against G (−0.079 for SAFE) is not defined. Is G loosened on the unseen test set?
   - "Recorded RLDX-1 rollouts": say who recorded them and with what protocol (seeds, task selection, calibration split).
8. **Test-set composition:** The test sets are balanced (10 successes and 10 failures per task), so tasks are weighted equally in ℓ̄ and FPR rather than by natural failure frequency. Say so and give the naturally weighted numbers once.
9. **Fixed-path SAFE result:** "SAFE + P gains 0.007, or 0.039 matched, as its path uses fewer false alarms than P (16 against 21)". Here the matched gain exceeds the same-target gain. Explain in one more clause that "matched" can loosen *or tighten* P.
10. **Related work:** add survival/time-to-event modeling as the natural "time-only" model class. A Kaplan–Meier entry is already in refs.bib but uncited and so does not appear in the references. Also consider a sentence on conformal monitoring of sequences/streams.
11. **Anonymity and formatting:**
    - The author line still has the placeholder "Paper-ID [add your ID here]".
    - The PDF metadata are anonymized (Author empty), but CreationDate carries a JST timezone, which is harmless but a small locality hint.
    - hyperref draws colored link boxes (visible in the PDF viewer); consider `hidelinks`.
    - The page limit is respected: content on pp. 1–5, references on p. 6.
12. **LaTeX source** (not visible in the PDF, but relevant if sources are ever shared, e.g., on arXiv): the comments contain internal run IDs, branch names and dates ("review-analyses branch", "Author decisions 2026-10-29"). Strip them.
13. **Citations checked:**
    - SAFE: App. B.5 equal-length control and fixed-horizon Franka rollouts — accurate. SAFE has two real robots (Franka, WidowX), so "its real robot" should be "its Franka experiments".
    - FIPER: OR vs. AND alarm rule and DT computed over detected failures — accurate.
    - Hide-and-Seek: margin (inter-trajectory) and onset (intra-trajectory) losses, 3 seeds, random 3/10 unseen splits, time-varying band — accurate.
    - ActProbe: chunk consistency, magnitude, normalized timestep, task-conditioned LSTM(-MLP), RoboCasa — accurate.
    - RESCUE: the abstract says detection scores "do not establish" lower intervention cost on its primary strong-policy conditions. The paper's "do not guarantee" is a fair paraphrase but should keep that qualifier.
    - CheckVLA (arXiv:2607.26789), RLDX-1 (arXiv:2605.03269) and Xiaomi-Robotics-1 (arXiv:2607.15330) exist with the given titles. I did not find a RoboCasa365 checkpoint release statement in XR-1's abstract ("checkpoints will be released"); please cite the checkpoint source.
    - No citation looked hallucinated.

---

### Score

**4 — Invite to rebuttal.**

The central observation is important and well documented: under success-terminated protocols, a timer is a near-perfect detector, and monitors should be judged by lead over it at matched FPR. The input-ablation methodology and the conformal OR-fusion are sound. The paper is unusually honest about its uncertainty. Its current evidence does *not* support H1 at matched FPR (both CIs include zero), and the headline numbers come from a calibration outside its own guarantee. Those are acceptable at Stage 1 only if the Extension Charter enforces:
- a clean confirmatory test (items 1–2);
- a credibly powered value experiment (item 3);
- stronger and fairer comparators (items 6–7);
- an analysis of how much the result depends on the time limit (item 8).

Rigorously falsifying H1/H4 would itself be a valuable outcome.

### Confidence

**4** — I know this literature well and checked the cited works' key claims directly (SAFE, FIPER, Hide-and-Seek, ActProbe, RESCUE, CheckVLA). I did not have access to the authors' data beyond the figure CSVs shipped with the source.

---

## Reviewer 2

**Paper:** "Time Is the Baseline to Beat: Lead Time over a Timer as the Test for VLA Failure Monitors" (RSS 2027, Stage 1)

### Summary

The paper argues that in success-terminated, time-limited benchmarks every failure is a timeout. A timer that uses only elapsed time is therefore a strong failure detector: on RoboCasa365 with the XR-1 policy, G and P detect all test failures (bACC 0.988 and 0.972, Table I), and a zero-input LSTM reaches the same ROC-AUC as one fed policy features (0.888 vs 0.878). The authors propose to evaluate monitors by their *warning lead* over a per-task timer P at a matched false-positive rate. They fuse a monitor with the timer through an OR rule. A one-parameter conformal calibration along a monotone path sets both thresholds (Prop. 1), which bounds the marginal FPR when the path does not depend on the calibration successes. On 380 + 380 XR-1 test rollouts over 38 tasks, three-seed ensembles of SAFE and Hide-and-Seek fused with P gain 0.030 and 0.071 of the time limit over P at a 5% target (realized FPR 0.061 vs 0.055). At matched false alarms the gains drop to 0.015 (95% CI [−0.007, 0.041]) and 0.056 (95% CI [−0.014, 0.120]). The Stage 1 numbers use an *adaptive* path that the guarantee does not cover; this raises the expected FPR from 0.047 to about 0.055. With a fixed path, Hide-and-Seek + P gains 0.072, or 0.057 matched. Input ablations (zero input, shuffle, first step) gain ≤ 0.012 at the same target and ≤ 0 matched. On unseen tasks Hide-and-Seek gains 0.127 over the global timer G (0.051 matched), while SAFE is at noise level. On recorded rollouts of a second VLA (RLDX-1, 100 + 100 test rollouts), the ordering is the same (0.083; 0.052 matched). Stage 2 proposes two experiments. E1 tests H3 on fresh RLDX-1 data with fixed paths, 20 tasks × (20 + 20) for calibration and for test. E2 tests H4 with a closed-loop reset-and-retry experiment on 1,000 scenes.

### Strengths

1. **An important and under-appreciated confound.** The point that "failure = timeout" makes elapsed time a near-perfect offline detector under success-terminated protocols is clear and well supported (Table I; the zero-input LSTM; FIPER's benchmark, where length alone gives ROC-AUC 0.92–1.00 on 4 of 5 tasks). It should change how this subfield reports results. The paper presents the timer as a comparator every monitor paper should include, not as a "gotcha".
2. **The estimand fits the setting.** Lead over a timer at matched FPR is the right question once recall is saturated. The authors correctly separate "same target" from "matched realized FPR" comparisons and report both.
3. **Well-designed controls for "learned elapsed time".** The zero-input, shuffle (same-task, other rollout, same inference) and first-step ablations are retrained with the same pipeline. They form a graded hierarchy that isolates rollout-specific, time-varying evidence. Random-score and within-task-resampled-score fusions serve as nulls for the fusion mechanism. This is better design than most of the monitor literature.
4. **Prop. 1 is correct as stated, and its scope is stated honestly.** It is the standard split-conformal quantile argument applied to the scalar "first silent path index" V_i, which is valid because the alarm sets are nested. I checked the proof (details under W1). The paper says explicitly that the reported numbers use an adaptive path not covered by the proposition, quantifies the resulting FPR inflation (0.047 → ~0.055 over 200 re-splits), and gives fixed-path numbers. This level of disclosure is unusual and welcome.
5. **Careful handling of data splits and limitations.** Training, validation (development) and test sets are disjoint. The Limitations paragraph admits that Hide-and-Seek was added after the test set had been used for SAFE and that all analyses beyond H1/H2 are post hoc.
6. **H4 / E2 asks the question that matters.** It tests whether earlier alarms raise success. The branch-from-the-same-run design gives paired counterfactuals per scene, a good variance-reduction choice.

### Weaknesses

**W1. The guarantee does not cover any reported result, and the selection effect is larger than the headline "0.047 → 0.055" suggests.**
Prop. 1 holds as stated. Write V_i for the smallest path index at which success i is silent. Monotone coordinates make silence upward-closed in λ, so "alarm ⇔ V_{n+1} > V_(r)". Exchangeability then gives Pr ≤ 1 − r/(n+1) ≤ α. Random tie-breaking is not even needed for this direction. The disabled-monitor convention for V_(r) = ∞ is handled correctly.
The difficulty is scope. Every number in Tables II and the abstract comes from the adaptive path, where (u, s) is chosen to maximize calibration-failure lead *with each candidate at its own λ̂*, a function of the calibration successes. Three consequences follow:
- (a) The FPR inflation (≈10% relative to target) is reported for "the choice", averaged over detectors. It is not reported per detector, and the paper gives no distribution (e.g., the 90th percentile of the conditional FPR) or grid size |{(u,s)}|.
- (b) The selection also biases the *lead*, not just the FPR. Choosing the candidate that maximizes calibration lead at its own λ̂ favors candidates whose λ̂ happened to fall low, i.e. those that spend more false alarms. The realized FPRs of the fused detectors (0.055–0.062) all sit above P's (0.055). This is the mechanism by which "same-target" gains are inflated.
- (c) The paper presents Prop. 1 as a contribution (bullet 2). Its content is a direct application of split conformal / nested conformal prediction (Gupta et al. 2022). That is acceptable, but the contribution should be framed as "an application that makes the comparison fair", not a new guarantee.

The fix is standard and should be used. One option is Learn-then-Test over the full (u, s, λ) grid, e.g. Bonferroni over (u, s) combined with fixed-sequence testing along λ. The method may then choose the maximum-lead configuration among the certified ones, which is valid under FWER control. Another is to split the calibration successes into a path-selection half and a λ̂ half. LTT yields a PAC guarantee rather than a marginal one, and the paper should say which guarantee type it targets.

**W2. Matched-FPR comparisons are the only fair ones, and at matched FPR H1 is not supported for either score. The paper's narrative still leads with the same-target numbers.**
The data show how steep the timer's lead–FPR trade-off is: loosening P from 0.055 to 0.061 (≈ 2 extra false alarms out of 380 successes) raises its lead by 0.015 for both SAFE and Hide-and-Seek. That is half of SAFE's headline gain and a fifth of Hide-and-Seek's.
The sampling variability of a conformally calibrated FPR is large on this scale. With n = 380 and r = 362 the conditional FPR is Beta(19, 362): sd ≈ 0.011, 90th percentile ≈ 0.065. The test-set binomial sd adds another ≈ 0.011. Calibration noise therefore moves a detector along its own trade-off curve by ±0.03 in lead.
"Same target" gains (0.030, 0.071; the abstract, the contribution bullet, the conclusion) thus mostly measure *where calibration happened to land*. The matched gains are the estimand. Their CIs include zero for both scores: SAFE [−0.007, 0.041], Hide-and-Seek [−0.014, 0.120].
The paper says "neither clearly above zero", which is honest. It should go further: state that **H1 (matched) is not supported in Stage 1**, and report both numbers in each paired sentence with the matched one first, rather than giving the same-target gains as headlines (Sec. VI-C subsection titles "a small gain"/"a larger gain"; the conclusion).

**W3. What the CIs resample is too narrow for the claims made.**
The 95% CIs resample tasks, then failures within tasks, and hold fixed:
- the fitted thresholds (calibration draw)
- the matched timer, which was tuned on the test successes
- the test successes themselves (no FPR uncertainty at all)
- the training seeds and the ensemble

For a matched-FPR gain, the uncertainty in the matching point (test successes) and in calibration matters as much as the uncertainty over failures. "Matched gains stay uncertain when calibration is also resampled" is stated without numbers.
Further points:
- (i) Only two CIs are reported. Nothing for the HS-vs-SAFE difference (0.041), the ablation differences (0.070/0.059/0.060), the unseen-task matched gains (0.051, −0.079), the fixed-path numbers, or RLDX-1.
- (ii) A two-stage (task → failure) bootstrap with 38 heterogeneous clusters is reasonable. The per-task gains, however, are very heavy-tailed (five tasks carry 78% of HS's gain; Fig. 3a ranges from −0.21 to +0.55). Percentile intervals may under-cover with this skew. Please report a BCa or studentized interval, or a task-level t-interval as a check.
- (iii) Resampling tasks implies inference to a population of tasks. That population is "RoboCasa365 tasks with published success between 10% and 90%". This should be stated.

**W4. The gain is partly a re-allocation of the false-alarm budget across tasks, and P is not the strongest time-only comparator.**
Fig. 3a (and its data) shows that HS + P is *later* than P on 16 of 38 tasks, mostly by an identical ≈ 0.21 H_k. The fused path's timer margin δ is about 0.21 larger than P's. The fused detector gives up timer earliness on composite tasks and spends the marginal FPR budget on score alarms on atomic tasks. At matched FPR the composite-task gain is 0 s, so the matched effect rests on 10 atomic tasks.
The marginal guarantee allows this (one task had 4/10 successes alarmed). A per-task timer with a *uniform* margin, however, cannot re-allocate. For time-only rules, the posterior failure probability given survival to t is increasing in t, so per-task cutoffs are the optimal time-only family. The strongest time-only comparator is therefore the per-task-cutoff allocation that maximizes pooled lead at a pooled FPR budget (a Lagrangian/knapsack over tasks).
The paper reports one sentence on "a timer that spreads the false-alarm budget across tasks, chosen as adaptively as the fused path" (−0.008 vs P; HS + P gains 0.063 over it). The construction is not described. A data-optimal allocation doing *worse* than P suggests it was overfit to 10 calibration successes per task, which makes it a weak comparator, not a strong one. An oracle version (allocation optimized on the test set at the matched FPR, analogous to the oracle matched P) would bound what any time-only rule can do. This comparator should be primary, or at least co-primary, in Stage 2.

**W5. Confirmatory status of H1/H2: the XR-1 test set has been reused and Stage 2 adds no fresh XR-1 data.**
The Limitations paragraph states that Hide-and-Seek was "tested on a test set already used for SAFE". It also says all analyses beyond H1/H2 (the fixed-path numbers, stronger comparators, nulls, α-sweep, seconds, unseen tasks, irreversibility) came after the test set was scored. The paper gives no pre-registration or time-stamped analysis plan for H1/H2 themselves.
Stage 2's E1 tests H3 on RLDX-1 and E2 tests H4. H1 and H2 on XR-1 are therefore decided once and for all on a test set that has been extensively reused and whose primary matched CIs include zero. This is the main gap in the Stage 2 plan. A fresh XR-1 test set with fixed paths and a pre-specified rule would make H1/H2 confirmatory. With the existing pipeline it is the cheapest experiment in the plan.

**W6. The Stage 2 plan has no decision rules, no effect size of interest, and no power analysis, so it cannot be shown to decide H3 or H4.**
RSS Stage 1 is judged on whether the methodology can settle the hypotheses. For E1 and E2 the paper specifies the data but not the inference:
- **No primary endpoint or decision rule.** H3 is a conjunction of four claims: {SAFE-LSTM, HS} × {same target, matched}. The text says only "compared with P … and with random-score fusions". The paper does not say what supports H3, what falsifies it, or what happens if one conjunct fails. It also does not say whether the four tests are corrected for multiplicity (Bonferroni/Holm, or intersection-union for the conjunction, which needs no correction).
- **No power analysis.**
  - On XR-1 (38 tasks × 10 failures) the HS matched CI half-width is ≈ 0.067. E1 has only 20 tasks × 20 failures. Variance is dominated by between-task heterogeneity (W3, W4), so more failures per task helps little and fewer tasks hurts. The CI is likely to be at least as wide.
  - The recorded RLDX-1 matched estimate is 0.052, and SAFE-LSTM's matched gain on RLDX-1 is not reported at all.
  - On these numbers E1's power to exclude zero looks well below 50%. The authors should report a simulation-based power estimate, using a task-level resampling of the recorded RLDX-1 data, before committing to 2,500 rollouts.
- **No smallest effect size of interest and no equivalence test.** The CFP values rigorous falsification. That requires a pre-declared SESOI (e.g., 0.02 H_k ≈ 1–2 s) and a TOST-style rule, so that a null result can be read as "no meaningful gain" rather than "inconclusive". SAFE's Stage 1 matched CI [−0.007, 0.041] is exactly the inconclusive case.
- **E2 (H4) sizing.** The expected paired difference is small. HS + P alarms ≈ 0.07 H_k earlier on average, mostly on 10 atomic tasks. The retry window after a P alarm is only ≈ 0.45 H_k at a 1.25 H_k budget, and the retry must fit a new attempt in it. Under an i.i.d.-retry assumption the predicted effect can be computed *now* from the recorded success-time distributions: P(success in remaining budget) per task. The paper should compute it and derive the number of discordant pairs needed for a paired (McNemar or task-clustered) test. The comment in the LaTeX source mentions "97.5% CIs", a "binomial FPR rule" and a "frozen null test", but none of this appears in the paper, and the reviewers can only evaluate the paper.

**W7. H4's comparator is mis-specified: P is calibrated for FPR, not for retry success.**
For reset-and-retry, the relevant time-only baseline is the *restart cutoff that maximizes success within the 1.25 H_k budget*. This is the classical restart-strategy problem for Las Vegas algorithms (Luby, Sinclair & Zuckerman 1993). The optimal cutoff may sit at a much higher FPR than 5%, because resetting a slow success is cheap when retries are fast.
If HS + P beats P but a timer with a different cutoff beats both, H4 as worded is "supported" while the paper's thesis (time is the baseline to beat) is contradicted. E2 should include:
- the success-vs-cutoff curve for time-only restarts, with the cutoff chosen on development data
- HS + P at matched reset rate
- an oracle-restart upper bound

E2 should also use **common random numbers**: the same policy seed for the retry from the restored scene under every monitor. The retries then differ only in their start time and remaining budget, which sharply reduces variance and makes the comparison interpretable. The paper should also specify what "restore the scene" means (exact initial simulator state vs. a re-randomized layout) and how a no-reset run is scored beyond H_k.

**W8. Several results that carry the narrative appear only in prose, without tables, CIs or method details.**
- The headline ensemble numbers (0.030, 0.071, 0.127, 0.083) do not appear in any table. Table II reports single models, whose HS gain is 0.039 ± 0.006, almost half the ensemble value and five single-model SDs away from it.
- "The ensemble is part of its warning" is not explained, and part of the gap may be the FPR difference (0.061 vs 0.055; see W2). Please report single-model matched gains and ensemble rows with the same CIs.
- No variability of the ensemble itself is reported (one ensemble of three seeds).
- The H2 null test is under-specified. "Both scores beat all 100 fusions of noise and of their own scores resampled within the task" gives a randomization p ≈ 1/101 per comparison. The paper does not say:
  - whether the null fusions go through the same adaptive (u, s) selection as the real scores; if they do not, the null is too weak
  - how a resampled score trajectory from a rollout of different length is truncated or extended
  - whether the null is matched to its own realized FPR
  - The null distribution is centred below zero (−0.007 to −0.011), so beating it is evidence for H2. It is not evidence for H1, and the text should keep the two apart.
- The RLDX-1 section does not describe its training and calibration sets. With 100 + 100 recorded test rollouts on 20 tasks (5 per class per task), FPR has granularity 0.01 and the matched gain (0.052) has no CI.

**W9. The unseen-task results are outside any guarantee, and the calibration design is not suited to that setting.**
Thresholds are calibrated on seen-task successes and applied to held-out tasks, so the calibration and test successes are not exchangeable. Unseen-task FPRs reach 0.079 (Table II), and no matched CIs are reported.
If unseen-task deployment is a claim, the right tool is task-level (hierarchical) conformal calibration, which guarantees FPR marginally over a new task when tasks are exchangeable. Two references: Dunn, Wasserman & Ramdas, *JASA* 2023; Lee, Barber & Willett, 2023. At minimum the unseen-task comparisons should be matched-FPR and carry CIs, since G's FPR (0.024) is far below the fused detectors' (0.04–0.079).

**W10. The exchangeability premise of Prop. 1 does not literally match the stratified design.**
Calibration and test successes are sampled at exactly 10 per task. Under a fixed per-task count, a new success from task k is exchangeable only with the 10 task-k calibration points, not with all 380. Under a uniform task distribution I believe the pooled bound still holds approximately for balanced strata: small examples I worked through stay at ≈ α. It is, however, not what the proposition assumes. Please state the sampling model: i.i.d. task draw, or balanced strata with the test task uniform. Then either prove the bound for that model or switch to per-task (Mondrian) calibration. Per-task calibration becomes possible in Stage 2, which has 20 successes per task, and it would also remove the "4 of 10 successes alarmed on one task" pathology.

### Questions to the authors

- **Q1.** How large is the (u, s) grid, and what is the per-detector distribution of test FPR over the 200 re-splits for the adaptive path (mean, 90th percentile)? What exactly is re-split: calibration+test successes pooled, or calibration only?
- **Q2.** How is P "loosened" to the fused detector's false alarms? Is it a uniform δ shift chosen on test successes? In the SAFE fixed-path case (16 vs 21 false alarms) P is *tightened*. Is the matched gain then still conservative toward the timer?
- **Q3.** What are the matched gains (with CIs) of the *single* models, and of the ensembles at the *fixed* path? Why does ensembling nearly double HS's same-target gain?
- **Q4.** For the H2 nulls, do noise and resampled-score fusions go through the same adaptive path selection as the real scores? How are resampled score trajectories aligned when rollout lengths differ?
- **Q5.** How is the budget-spreading timer (W4) constructed? What is its matched gain when the allocation is chosen on the test set (oracle), as P's matching is?
- **Q6.** Does the zero-input Hide-and-Seek model receive any task identity, e.g. through per-task normalization (m_k, σ_k)? If yes, it is a per-task time-only rule, i.e. an alternative timer. That supports its role as the elapsed-time control and should be stated.
- **Q7.** In Sec. I, which LSTM gives the 0.888 vs 0.878 ROC-AUC (SAFE-LSTM or Hide-and-Seek), and on which split? Note that the zero-input model is *higher*.
- **Q8.** What are the RLDX-1 training/calibration splits behind Table II (bottom), and what is the CI on the 0.052 matched gain? What is SAFE-LSTM's matched gain on RLDX-1? H3 asserts it.
- **Q9.** For E1, what happens if some RLDX-1 tasks cannot reach 20 failures (or 20 successes) within the rollout budget? Is there a stopping rule, and are such tasks dropped before unblinding?
- **Q10.** In E2, does the no-reset run continue to 1.25 H_k? Is a scene that succeeds between H_k and 1.25 H_k a success for the no-reset arm? Is the retry seeded identically across monitors?
- **Q11.** For FIPER's benchmark, "at FIPER's own FPR (0.06 to 0.56 by task)", is the timer matched per task? What is the uncertainty on the 0.146 lead difference? An FPR of 0.56 is hardly an operating point.
- **Q12.** How were the 48 annotated failures selected (random, stratified by task)?

### Required for the Extension Charter

1. **A pre-specified statistical analysis plan in the Stage 2 paper**, frozen before unblinding. It must state:
   - the primary endpoint for each hypothesis (matched-FPR mean-lead gain for H1–H3; paired success-rate difference for H4)
   - the test, the α level and the multiplicity handling: intersection-union for the H3 conjunction; Holm or Bonferroni across scores
   - the decision rule for "supported", "falsified" and "inconclusive"
2. **A smallest effect size of interest and an equivalence test.** Declare a SESOI in units of H_k and in seconds, and apply TOST or a CI-inside-margin rule, so that a null result on SAFE or on composite tasks counts as falsification.
3. **A simulation-based power analysis for E1 and E2**, using a task-level resampling of the recorded RLDX-1 data and the recorded success-time distributions. If power is inadequate, enlarge the design, preferably with more tasks rather than more rollouts per task, given the between-task heterogeneity.
4. **A fresh XR-1 confirmatory test set for H1/H2** with fixed paths (Prop. 1 applies), analysed under item 1. Otherwise H1/H2 must be explicitly downgraded to exploratory.
5. **Fixed paths or LTT only.** Every fused number in Stage 2 must come from a path fixed before the calibration successes are seen, or be certified by Learn-then-Test over the full grid (with its PAC guarantee stated as such). Report realized test FPR for every row.
6. **Matched-FPR as the primary comparison throughout**, including unseen tasks and RLDX-1, and also as a curve. Report lead-vs-FPR trade-off curves for fused vs. P over, say, [0.02, 0.10], with simultaneous or pointwise bands, and the gap integrated over that range.
7. **CIs that resample the whole pipeline**, as a sensitivity analysis: calibration successes, test successes (hence the matching point), test failures, by task, and ideally training seeds. Report them for every claimed difference: HS vs SAFE, real vs ablations, real vs nulls, unseen-task and RLDX-1 matched gains. Use BCa, studentized, or task-level t intervals given the skew.
8. **The strongest time-only comparator.** Report the per-task-cutoff allocation optimal for pooled lead at the matched pooled FPR, both calibration-chosen and test-oracle, alongside P.
9. **Per-task FPR control.** With 20 calibration successes per task, report per-task (Mondrian) calibration as a secondary analysis, and give the maximum per-task test FPR for the pooled calibration.
10. **E2 design details:**
    - common random numbers for retries
    - a precise definition of the scene restore and of the no-reset arm
    - a time-only restart baseline with the cutoff optimized for success within 1.25 H_k on development data
    - HS + P vs P at matched reset rate on successes
    - an oracle-restart upper bound
    - the predicted effect size computed from recorded data before running
11. **Ensemble and single-model reporting.** Add ensemble rows to the main table, give single-model matched gains, and estimate ensemble-to-ensemble variability (e.g., with more seeds).
12. **Specification of the H2 null procedure** (selection, length alignment, matching), and use the same pipeline for nulls and real scores.
13. **Hierarchical (task-level) conformal calibration for unseen tasks** if unseen-task claims are retained, or else drop the unseen-task claims from the abstract.

### Minor issues

- **Numbers not in tables.** The abstract and contribution numbers (0.030, 0.071, 0.127, 0.083) are ensemble values that do not appear in Table II, which shows single models (0.024, 0.039, 0.122, 0.074). Readers will try to reconcile them.
- **Abstract vs. Table II.** The abstract says ablations "add at most 0.012". Table II gives zero input at 0.013 ± 0.003 (single model, seen). The 0.012 is the ensemble. Say which.
- **Rounding.** "0.111 more than SAFE": 0.127 − 0.017 = 0.110 from the reported values. "SAFE-LSTM and the OOD detectors gain 0.10 to 0.115", but RND on RLDX-1 unseen is 0.099 in Table II.
- **"All six policy scores"** (Sec. VI-C): the paper has seven (4 OOD + SAFE-MLP + SAFE-LSTM + HS). Specify which six.
- **Sec. I:** "an LSTM trained on zero inputs … matches the ROC-AUC of the same LSTM trained on the policy's features (0.888 against 0.878)". Name the LSTM and the split. The time-only model is *better*, which is worth stating.
- **"Loosened".** P is "loosened to the fused detector's false alarms", but in the SAFE fixed-path case (16 < 21) P is tightened. Use "matched".
- **Notation clashes.**
  - ρ_k (95th-percentile success duration of task k, Sec. IV-B) vs. ρ_i = L_i/H_{k_i} (Sec. IV-D)
  - s (path slope) vs. s_t (score)
  - u (path offset) vs. u_t^{(j)} (SAFE output)
  - T (first alarm) vs. T_k (control-step limit)
- **Prop. 1 statement.**
  - "Ties broken at random" is unnecessary for the ≤ direction. Drop it, or state that it is needed only for the uniform-rank claim.
  - Explicitly state that M_i is computed with a score function fitted without the calibration successes (it is implicit in "path does not depend on the calibration successes", but the score normalization m_k, σ_k must also be independent).
- **H1 wording.** "a policy-internal score" is existential. Name the scores (SAFE, Hide-and-Seek) and say whether the hypothesis is per score.
- **H3** includes a parenthetical result ("SAFE's MLP head adds none there") drawn from the same recorded RLDX-1 data. It is acceptable that SAFE-LSTM was chosen for H3 on that basis, since Stage 2 uses new rollouts, but say so.
- **Results section:** add a one-line verdict per hypothesis, e.g. "H1 (matched): not supported at 95%; H2 vs nulls: supported (p ≤ 1/101)".
- **Table I:** the standalone detectors use a constant threshold on the running max rather than their native time-varying conformal bands. The caption should say so, because "as in prior work" suggests the native protocol.
- **Table II.** Bold/underline rankings rest on differences well inside seed SDs (e.g., Mahalanobis 0.034 vs SAFE-LSTM 0.031). Std over three seeds is not an uncertainty estimate for the data; say so or drop the ranking marks.
- **"Positive at every target from 2% to 10%"** (Fig. 3b): the sweep points share data and are strongly dependent, so this is not additional evidence. At 1% the HS matched gain is negative (−0.008).
- **Fig. 3a:** tick labels and legends are barely legible at print size. The identical −0.21 bars deserve a sentence (W4).
- **Inter-rater agreement:** κ = 0.80 from 12 items (11/12 agreement) has a very wide CI. Report the CI or the raw agreement only.
- **Validation vs. test timer.** The intro's "timer at 90% of the limit flags all of them at 4.7% FPR" (validation) and Table II's G mean lead of 0.123 (test, implying c ≈ 0.88) come from different calibrations. Say which is quoted where.
- **E1 count.** "About 2,500 rollouts" for 1,600 kept rollouts depends on the per-task success rates. Give the per-task expected counts or the stopping rule.

### Score

**4 — invite to rebuttal** (with a substantial Extension Charter).

The central hypothesis is important, the controls are well conceived, and the paper is unusually candid about the scope of its guarantee and its post hoc analyses. The weaknesses are mostly statistical and fixable within Stage 2:
- the adaptive path outside the guarantee
- headline same-target numbers when only matched comparisons are fair
- CIs conditional on thresholds and matching
- an under-powered, rule-free Stage 2 plan

The Stage 2 plan as written cannot be shown to decide H3 or H4. If the charter items 1–6 and 10 are not adopted, I would move to 3.

### Confidence

**4.** I checked the proposition and its proof, recomputed the Fig. 3 summary statistics and the TWA/Δℓ̄ identities in Tables I–II (all consistent), and verified the internal number consistency described above. I did not have access to code or data beyond the paper.

---

## Reviewer 3

Reviewer expertise: robot learning practice (VLA policies, simulation benchmarks such as RoboCasa and LIBERO, real-robot deployment, recovery and retry).

### Summary

The paper argues that under success-terminated, time-limited evaluation protocols (RoboCasa365, LIBERO), a failure is by construction an episode that hits its time limit, so elapsed time alone is a near-perfect failure detector. On RoboCasa365 with the XR-1 VLA (38 tasks; test set of 380 successes + 380 failures), a per-task timer P (95th percentile of training success durations plus a calibrated margin) reaches bACC 0.972 and a global timer 0.988. Every learned detector stays at or below 0.818 bACC (Table I), and a zero-input LSTM matches the feature-based LSTM's ROC-AUC (0.888 vs 0.878). The authors therefore propose evaluating monitors by their *warning lead* over a timer: how much earlier, as a fraction of the time limit H_k, a monitor OR-fused with the timer alarms at the same false-positive rate. The fusion's split-conformal calibration bounds the FPR for a path fixed independently of the calibration successes (Proposition 1). In Stage 1, three-seed ensembles of SAFE and Hide-and-Seek fused with P lead P by 0.030 and 0.071 of H_k at 0.061 test FPR. At matched false alarms this drops to 0.015 (95% CI [−0.007, 0.041]) and 0.056 ([−0.014, 0.120]). Input ablations gain at most about 0.012. On average that is 2.6 s earlier than P, and 0 s on composite tasks at matched FPR. On unseen tasks, Hide-and-Seek gains 0.127 over a global timer, while SAFE does about as well as noise. The ordering holds on 100 + 100 recorded RLDX-1 rollouts. On FIPER's benchmark, rollout length alone gives ROC-AUC 0.92–1.00 on four of five tasks, yet FIPER still warns earlier than a timer at its own FPR. In 48 annotated failures, 79% never became irreversible before the limit. Stage 2 has two parts. E1 tests H3 on fresh RLDX-1 rollouts with fixed fusion paths. E2 tests H4 in closed loop on 1,000 RLDX-1 scenes: does resetting and retrying on Hide-and-Seek + P alarms, within a 1.25 H_k budget, beat resetting on P alarms?

### Strengths

1. **An important, under-reported baseline, demonstrated convincingly.** The observation that elapsed time dominates offline failure-detection metrics on success-terminated benchmarks is valuable to the community. Table I is striking: two deterministic timers with no policy access beat every published detector on bACC. Many recent monitor papers report on RoboCasa or LIBERO, and the FIPER re-analysis shows the problem is not specific to the authors' pipeline. If this paper only made every future monitor paper report a timer row, it would already have impact.
2. **The ablation logic is well designed.** The zero-input, shuffle and first-step ablations, plus random and within-task resampled score nulls, cleanly separate "learned task duration" from "rollout-specific evidence". This is exactly the control a skeptical practitioner would ask for, and it rules out the most obvious confound for the recurrent detector.
3. **Calibration is done carefully and reported honestly.** The authors state that the adaptive path used in Stage 1 is not covered by Proposition 1, quantify the resulting FPR inflation (0.047 → about 0.055), report fixed-path numbers, and commit to fixed paths in Stage 2. Matched-FPR comparisons that favor the timer are reported next to the same-target ones. CIs that include zero are stated plainly rather than hidden.
4. **The negative or weak findings are presented as findings.** SAFE's matched gain of about zero, SAFE matching noise on unseen tasks, and the per-task heterogeneity are all reported. This fits the Stage 1 spirit, where rigorous falsification is a contribution.
5. **The irreversibility annotation is the right instinct.** Asking whether failures had reached a point of no return is exactly the bridge between offline lead and deployment value. The answer (79% recoverable) is informative, even if it is under-exploited (see W2).
6. **The Stage 2 plan targets the right question.** E2 tests whether earlier alarms raise success when acted upon, rather than adding more offline metrics. The branching design (one shared no-reset run per scene, branched at each monitor's first alarm) is an efficient, paired way to do it.

### Weaknesses

**W1. The practical size of the effect is very small, and the paper's headline unit hides this.** Lead is expressed as a fraction of a benchmark-chosen time limit H_k. In seconds (Sec. VI-C, "Where the gain comes from"), Hide-and-Seek + P warns 2.6 s earlier than P on average. At matched false alarms it warns 5.0 s earlier on atomic tasks and **0 s earlier on composite tasks**, which are 28 of the 38 tasks and the long-horizon behaviors practitioners care about most. Fig. 3a and the per-task data show:
- The mean gain is a net of large gains on a few atomic or pick-and-place tasks (five tasks carry 78% of it).
- On 16 of 38 tasks the fused monitor alarms *later* than P, by up to about 0.21 H_k on several composite tasks.

For a deployed system, a monitor that is worse than a free timer on 42% of tasks is a regression unless the task is known to be favorable in advance. The headline 0.071 (or 0.056 matched) suggests a uniform small improvement; the real picture is "large on a handful of short atomic tasks, zero or negative elsewhere". The abstract and contributions should lead with seconds and with the per-task sign distribution.

**W2. "Lead over the time limit" is not yet tied to what a deployed system needs, and the paper's own evidence cuts against the implied use.** A deployed monitor's job is one of three things:
1. Stop before damage or an unrecoverable state.
2. Trigger recovery or retry.
3. Hand over to a human early enough for them to help.

Lead to H_k measures none of these directly:
- **Stopping before damage.** Of the 10 failures that became irreversible, P alarmed before the point of no return on none, and each fused monitor on only two (Sec. VI-D). On the safety-relevant notion, neither the timer nor the monitors work. RoboCasa does not model damage, collisions or unsafe states at all, so the benchmark cannot speak to this use. The title ("VLA Failure Monitors") and the introduction ("a deployed robot must notice when an attempt is going wrong") imply a broader scope than is tested.
- **Recovery or retry.** 79% of annotated failures were recoverable and "mostly slow or near-successful attempts". For such failures, the best action may be to *keep going*, not reset. An earlier alarm then has negative value. E2 must be able to reveal this (see W3).
- **Human handover.** A 2.6 s average gain is on the order of an operator's reaction time, so its value for handover is unclear without a model of operator latency.

The paper argues that "lead, not recall, is the quantity to report" because failures are recoverable (end of Sec. I). That argument justifies measuring *time*, but not measuring it *to an arbitrary benchmark deadline*.

**W3. E2 compares against the wrong timer, so it may not answer the practical question.** H4 compares resetting on Hide-and-Seek + P alarms with resetting on P alarms, where P is tuned for 5% FPR. For retry, false alarms are not the cost that matters. The objective is success within a budget, which trades wasted time on doomed attempts against killing attempts that would have succeeded. A time-only restart policy can be optimized directly for that objective from the success-duration distribution, with no failures and no policy features: restart at the cutoff that maximizes success probability within 1.25 H_k. This is the classical optimal-restart problem (e.g., Luby, Sinclair and Zuckerman, 1993, for Las Vegas algorithms, and restart strategies for heavy-tailed search). The per-task optimal cutoff may be much earlier than P's 95th-percentile-plus-δ, and could capture most or all of the retry benefit that earlier alarms are supposed to deliver. If so, E2 could show HS + P > P while the practitioner's correct conclusion is still "use a better timer". E2 needs three arms besides HS + P: no reset (run to 1.25 H_k), P reset, and a success-optimal timer reset chosen on development data. It also needs a fused monitor whose timer component is optimized the same way.

**W4. E2 has no stated power analysis or decision rule, and the expected effect looks small relative to its resolution.** A back-of-envelope estimate from the authors' own numbers suggests a small effect:
- About 44% of RLDX-1 rollouts fail.
- The fused monitor alarms earlier on only a minority of failures, concentrated in atomic tasks.
- The extra retry time is about 0.05–0.07 H_k of a 1.25 H_k budget.

That points to an overall success-rate difference of perhaps 1 percentage point. With 1,000 paired scenes and stochastic retries, the standard error of a paired success-rate difference is roughly 1 pp if about 10% of pairs are discordant, so a null or inconclusive outcome seems likely. Stage 1 papers are judged on methodology, so the paper must give:
- the minimum detectable effect, or a pilot-based power estimate;
- the pre-specified test, including whether it is one-sided and how multiplicity across the planned comparisons is handled;
- what result would count as falsifying H4.

The phrase "with a 95% CI" alone is not a decision rule. The same applies to E1: the paper does not say which CI must exclude zero, at what level, or whether same-target and matched comparisons both have to pass.

**W5. Reset and retry in simulation flatters the reset arm relative to real deployment.** "Restore the scene and retry" is free and exact in simulation. On a real robot, a reset needs a human, a learned reset policy, or a reversible task, and typically costs tens of seconds. It is also often impossible after a spill or a dropped object. Because the hypothesized advantage is a few seconds, any realistic reset cost could reverse it. E2 should treat reset cost as a parameter (0, 5, 15, 30 s, or a fraction of H_k) and report the break-even reset cost. Retrying from the identical initial scene also correlates the retry with the original attempt. The paper should report retry success conditional on first-attempt failure, and per-scene success variability, which bounds what any reset policy can gain.

**W6. Transfer beyond RoboCasa365 + XR-1 is asserted more than tested, and there is no real-robot plan.** The introduction states the issue for "RoboCasa365 and LIBERO", but LIBERO is never evaluated. The cheapest and most persuasive check would be SAFE's own LIBERO-10 setting with OpenVLA, π0 and π0-FAST, whose rollout-generation code is public. LIBERO also uses per-suite rather than per-task step limits, which tests whether the P vs G picture holds when limits are less tailored.
- **Second policy.** RLDX-1 is evaluated only on 100 + 100 recorded rollouts (5 + 5 per task). The paper does not say how architecturally different it is from XR-1 (both appear to be chunked-action VLAs on the same benchmark).
- **FIPER benchmark.** The FIPER result is encouraging, since length is again highly predictive. However, the two real-robot tasks are not broken out, and FIPER's FPR range (0.06–0.56) makes the "FIPER warns earlier" result hard to interpret practically.
- **Real-robot regimes.** Real deployments often lack an automatic success detector. Human operators also stop dangerous attempts early, so real failure-length distributions differ from timeouts. SAFE's real-robot protocol runs to a fixed horizon for this reason. The paper scopes this out in a single sentence in the Limitations.

For a robotics venue, the Stage 2 paper should either include a real-robot or real-data component, or explicitly reframe the contribution as an *evaluation-protocol* result for simulation benchmarks.

**W7. Practitioner cost and data requirements are not compared.** The learned monitors need a large labeled training set with failures (1,824 rollouts), plus 10 successes and 10 failures per task for calibration and an ensemble of three models. P needs only successful durations: the paper notes that 1–5 successes per task already allow a per-task timer. Mahalanobis needs only successes and matches or beats SAFE at matched FPR (Fig. 3b: about 0.030 vs 0.015 at 5%). A practitioner would want a cost-benefit table: data needed (successes vs failures), compute per step, and gain in seconds. Collecting failures on real robots is expensive and sometimes unsafe, so the success-only detectors and the timer deserve more emphasis than they get.

**W8. The ensemble effect is unexplained and carries the main H1 result.** A single Hide-and-Seek model gains 0.039 ± 0.006 (Table II), barely above deterministic Mahalanobis (0.034). The three-seed ensemble gains 0.071. With a seed std of 0.006, averaging three models should not nearly double the gain unless ensembling changes the score distribution, for example by lowering variance on successes and allowing a lower threshold. The paper's comment that "the ensemble is part of its warning" is not an explanation. The ensemble numbers that carry H1 also appear in no table, only in the text. The paper should explain this and show ensembles for all detectors, including OOD baselines with multiple fits where meaningful. It should also state whether ensembling was specified before the test set was scored.

**W9. Per-task FPR is uncontrolled, and that matters operationally.** The guarantee is marginal over tasks. On the test set, one task had 4 of 10 successes alarmed. In a retry deployment, a 40% false-alarm rate on a task means 40% of good attempts are interrupted. The fused path also shifts false-alarm budget across tasks (16 tasks get a looser timer). The paper should report the per-task FPR distribution for P and for the fused monitors. Stage 2 should either budget per task (the paper notes this needs 19 calibration successes per task) or report the worst-task FPR.

**W10. Researcher degrees of freedom and pilot-informed hypotheses.** The authors acknowledge three things: all analyses beyond H1/H2 came after the test set was scored, Hide-and-Seek was tested on a test set already used for SAFE, and H3 is phrased with knowledge of the recorded RLDX-1 results ("SAFE's MLP head adds none there"). This is acceptable for Stage 1, but it raises the bar for Stage 2. E1 and E2 must be fully frozen, including detectors, paths, metrics, decision rules and the analysis code.

### Questions to the authors

- **Q1.** How are RoboCasa365's per-task limits T_k set relative to demonstration or typical success durations? If H_k is generous (for example, several times the median success), a lead of 0.07 H_k means something different than if H_k is tight. Can you report lead in units of each task's median success duration as well as in seconds?
- **Q2.** In E2, does the no-reset arm run to 1.25 H_k, so that it benefits from extra time just as the reset arms do? Given that 79% of failures were "slow or near-successful", what fraction of failures at H_k succeed if simply allowed to continue to 1.25 H_k?
- **Q3.** Have you computed, from training success durations alone, the success-optimal restart cutoff per task for a 1.25 H_k budget, and how does it compare with P's c_k? If the optimal cutoff is much earlier than c_k, a simulation using the recorded rollouts could already show whether any monitor beats the optimal timer reset.
- **Q4.** What is the expected effect size for H4, and the minimum detectable difference with 1,000 paired scenes? What outcome would you treat as falsifying H4?
- **Q5.** Why does three-seed ensembling nearly double Hide-and-Seek's gain (0.039 → 0.071) when the seed std is 0.006? What happens with 5 or 10 seeds, and with ensembling applied to SAFE-LSTM and to the OOD detectors?
- **Q6.** On the 16 tasks where Hide-and-Seek + P alarms later than P, could a practitioner identify them in advance (for example from the validation set) and fall back to P? Would a per-task "use the monitor only where it helped on validation" rule keep the gain without the regressions?
- **Q7.** How different is RLDX-1 from XR-1 (backbone, action head, chunk length, execution of chunks)? Would you consider a policy from a different family, such as π0 or OpenVLA-OFT on LIBERO, instead of or in addition to RLDX-1?
- **Q8.** For FIPER's two real-robot tasks specifically, how well does length alone separate failures? How do the timer and FIPER compare at a practically usable FPR (≤ 0.1)?
- **Q9.** In real deployments without an automatic success check, how would the per-task timer be applied? Would a VLM success detector or a policy "done" signal be assumed? How sensitive are the conclusions to success-detection errors, such as a missed success that keeps the episode running?
- **Q10.** In the irreversibility study, what were the 10 irreversible failures (drops, spills, wrong object placed)? At what fraction of H_k did they reach the point of no return? Is the point of no return typically early enough that *any* online monitor could plausibly beat it?
- **Q11.** What latency do the monitors add per inference on the deployment hardware? E2 says latency counts against the budget; what are the values?

### Required for the Extension Charter

1. **E2 must include a success-optimal time-only reset comparator.** Use a per-task restart cutoff chosen on development data to maximize success within the budget. Also include a fused variant whose timer component is optimized the same way. Report all of these arms against the same paired scenes:
   - no reset (continue to 1.25 H_k);
   - P reset;
   - optimal-timer reset;
   - HS + P reset;
   - SAFE-LSTM + P reset.
2. **Pre-specified decision rules and power for E1 and E2.** For each hypothesis, state the test statistic, the CI level and sidedness, the multiplicity handling, the minimum detectable effect from a pilot or from the recorded rollouts, and what counts as support versus falsification. Freeze detectors, paths and analysis code before running new rollouts.
3. **Reset cost as an explicit parameter in E2.** Report success-rate differences as a function of per-reset cost (at least 0 s and one realistic real-robot cost such as 15–30 s) and the break-even cost. Report unnecessary resets, retry success conditional on first-attempt failure, and per-scene success variability.
4. **Report effects in physical time and per task.** Report seconds and the median-success-duration-normalized lead alongside fractions of H_k. Report the per-task sign distribution, meaning the number of tasks where the fused monitor alarms later than the timer and by how much. Report the per-task and worst-task FPR for every fused detector.
5. **A second benchmark or policy family.** Evaluate on at least one setting outside RoboCasa365. The preferred option is LIBERO-10 with π0 or OpenVLA, SAFE's own setting. The required content is the timer rows of Table I and the lead analysis with ablations, so it can be shown whether "time is the baseline to beat" generalizes beyond one benchmark's time-limit convention.
6. **Real-robot evidence or explicit scoping.** Either analyze at least one real-robot, success-terminated dataset with the full timer and lead pipeline (FIPER's two real tasks broken out at minimum), or retitle and reframe the contribution as an evaluation-protocol result for simulation benchmarks. A small real-robot retry pilot is strongly encouraged but not required.
7. **Make the irreversibility analysis concrete (E3 is currently one line).** Pre-specify a larger annotated sample on the new RLDX-1 test set, stratified by task type, with two annotators throughout. Report lead to the point of no return for timer, SAFE and Hide-and-Seek fusions. This is the quantity most relevant to "stop before damage" and should not be left vague.
8. **Report and explain ensembles.** Ensemble results that carry H1/H2 must appear in tables. Report ensembles for all learned detectors, together with an analysis of why ensembling changes lead.
9. **A cost-benefit summary.** For each monitor, report the data it needs (successes and failures per task), training and calibration set sizes, per-step latency, and lead gain in seconds. This should include the timer and the success-only detectors.

### Minor issues

- **Headline numbers absent from tables.** The H1/H2 headline numbers (0.030, 0.071, 0.015, 0.056, and their CIs) appear only in the text. Table II reports single models. Add an ensemble block or a small H1/H2 results table with the same-target and matched gains and CIs.
- **Seen vs unseen ensemble inconsistency.** On unseen tasks the SAFE ensemble (0.017) gains *less* than the single-model mean (0.028 ± 0.009, Table II). This deserves a sentence.
- **"Six policy scores".** Sec. VI-C says "All six policy scores gain more on the 10 atomic tasks…", but Table II lists seven learned detectors (four OOD, SAFE-MLP, SAFE-LSTM, Hide-and-Seek). Say which six.
- **"0.111 more than SAFE".** Sec. VI-D gives this difference, but 0.127 − 0.017 = 0.110. This is likely rounding; fine, but check.
- **Fig. 1.** The example uses SAFE, whose matched gain is about zero, on a hand-picked validation rollout. It also shows the success curve nearly reaching τ_s, which illustrates how scores also rise with time on successes. Consider either a representative example or an explicit note that it is illustrative. The tiny in-figure fonts ("success durations run out", "a success ends at t = 33") are illegible in print.
- **Fig. 3.** Panel (b) is vertically squashed, the tick labels are unreadable at print size, and no uncertainty is shown. Add a CI band for at least Hide-and-Seek and SAFE. Panel (a) needs a y-axis tick at the negative bars' magnitude so readers see that 16 tasks are negative. Its caption should say so explicitly, rather than only stating the atomic vs composite means.
- **Table II.** The "Abl." block reports single models of what is presumably the Hide-and-Seek architecture; say so in the caption. The RLDX-1 block has 5 + 5 test rollouts per task. Single-model stds are given, but the evaluation-sampling uncertainty, which is larger, is not.
- **Definitions.** "Matched" is defined in the Metrics paragraph, but it is used earlier (abstract, H1). A one-line definition in Sec. III would help robotics readers. Likewise, "adaptive path" vs "fixed path" is subtle; a one-sentence plain-language gloss ("we tuned the trade-off on the same successes we calibrate on") would help.
- **Odd cross-reference.** Sec. VI-C's reference "(Sec. I)" for the zero-input ROC-AUC claim points back to the introduction instead of to a table. Say which split the 0.888 vs 0.878 numbers come from.
- **Timeline confusion in "Data".** The paragraph says the validation calibration set "was used to develop the timers and the fused detectors". State clearly in the setup which choices (ensembling, PCA-128, running max, constant threshold) were made before the test set was first scored.
- **Composite-task significance.** The "0 s at matched false alarms on composite tasks" result is a key practitioner number buried in a parenthesis. Promote it.
- **Readability.** The writing is dense and number-heavy throughout. Many paragraphs carry five or more decimals with nested parentheticals. For a robotics audience, a short "practitioner takeaway" box (report a timer; expect seconds of extra warning on short atomic tasks, nothing on long-horizon ones; timers do not catch irreversible failures) would greatly improve clarity.
- **Anonymity and housekeeping.** The author line still reads "Paper-ID [add your ID here]". No anonymity violations were found: no self-identifying citations, links or acknowledgments, and the PDF author metadata is empty.
- **Bibliography.** `kaplan1958` and `kressgazit2024empirical` are in refs.bib but not cited. This is harmless but suggests leftover analyses (e.g., survival curves) were cut. Kress-Gazit et al. would in fact be a fitting citation for evaluation practice.
- **Missing related work.** Restart-strategy literature (Luby et al., 1993) is directly relevant to H4 and is not cited. Work on autonomous reset or reset-free learning is relevant to the cost of acting on alarms.

### Score

**4 — Invite to rebuttal.**

The core methodological point is correct, useful and well-controlled: monitors must be compared against elapsed time on success-terminated benchmarks, and the comparison should use lead at matched FPR. The calibration treatment is sound, and the honesty about weak and uncertain results is exemplary. The Stage 1 hypotheses and methodology are sufficiently developed. My reservations concern practical significance and the design of the one experiment that could demonstrate it. In Stage 1, the effect is a few seconds, confined to a handful of atomic tasks, and negative on many others. E2 compares against a timer tuned for the wrong objective, has no power analysis, and assumes free resets. These are fixable through the Extension Charter (items 1–4 especially). If they are not fixed, Stage 2 risks a result that is statistically inconclusive or that practitioners cannot act on.

### Confidence

**4** — I am familiar with VLA policies, RoboCasa/LIBERO evaluation protocols, and failure-detection and recovery work. I did not verify the conformal proof beyond a careful reading, and I could not independently check the two most recent cited preprints (Hide-and-Seek, RLDX-1).

---


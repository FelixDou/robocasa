# Mock review, RSS 2027 Stage 1 — 30 October 2026

Draft reviewed: commit `e4006e6d` on `claude/time-aware-fusion-paper` (5 + 1 pages).
Three independent reviewers, each reading only the paper (PDF and LaTeX), with the RSS 2027 Stage 1 rules.

| Reviewer | Expertise | Score (1–5) | Confidence |
|---|---|---|---|
| R1 | Failure detection (SAFE, FIPER, Hide-and-Seek, STAC) | 4, invite | 4 |
| R2 | Statistics, conformal prediction, experimental design | 4, invite | 4 |
| R3 | Robot learning practice, VLAs, deployment | 4, invite | 4 |

## Area-chair meta-review

**Recommendation: invite to Stage 2.** All three reviewers find the duration-only comparator new and useful,
the controls well designed (the shuffle ablation is "the correct null for learned elapsed time"), and the
reporting unusually candid (CIs that include zero, adaptive vs fixed path, post-hoc analyses, FIPER credited).
The FIPER result shows the test is not rigged against monitors.

The consensus concerns are about whether the Stage 1 evidence and the Stage 2 plan can establish the claims:

1. **No decision rules (R1, R2, R3).** The hypotheses name no primary detector, test, α, multiplicity
   correction or effect size, so H1–H3 cannot be judged supported or falsified. All three ask for a compact
   pre-specified analysis plan and a power analysis for E1 and E2.
2. **The headline numbers use the adaptive path (R1, R2).** Proposition 1 does not cover them, and with the
   fixed path SAFE + P falls to 0.007. Reviewers want fixed-path results, same-target and matched, as the
   headline, or a valid correction for path selection (split, Learn-then-Test).
3. **P may not be the strongest time-only comparator (R1, R2, R3).** The fused detector reallocates the
   false-alarm budget across tasks (later than P on 16 tasks); a time-only rule with per-task margins chosen
   under the same global FPR could capture part of the gain. "Richer time baselines" should move from E3
   to the primary comparison.
4. **E2 cannot show practical value as written (R1, R2, R3).** Requested arms: no reset to the full budget,
   random-time or random-score resets at a matched reset rate, an oracle reset; several budgets; operating
   points tuned for success, not 5% FPR; common random numbers; scene restoration is not available on a real
   robot.
5. **Lead to the time limit vs. irreversibility (R3, R1).** 79% of annotated failures were recoverable at the
   limit, "slow or nearly succeeded"; lead to the limit may reward predicting slowness. Make lead before the
   point of no return a primary measure, split slow from stuck failures, report lead in seconds.
6. **Uncertainty is incomplete (R2).** The matched CIs exclude calibration, matching, success resampling and
   ensemble variability; only two numbers have CIs; tables bold differences far smaller than test noise.
7. **Fidelity of baselines (R1).** SAFE and Hide-and-Seek use time-varying functional conformal bands; the
   paper uses a constant threshold on a per-task-normalized running maximum, 128-d PCA and three-seed
   ensembles. The native band, itself time-aware, is the missing competitor.
8. **Confirmation (R1, R2).** H1/H2 rest on a reused test set with post-hoc analyses; reviewers ask for a fresh
   XR-1 confirmatory test with fixed paths, otherwise an RLDX-1 null cannot be told from a Stage 1 false positive.

### Reviewer claims checked against the sources

| Claim | Check | Consequence |
|---|---|---|
| FIPER already noted a length effect (R1) | **True.** FIPER Sec. 6 / Fig. 5: accumulating scores (STAC) "detects failure rollouts very late, mostly due to their greater length". | Credit FIPER in Related Work; the novelty (no timer baseline, comparator at matched FPR) still holds. |
| FIPER's DT counts only flagged failures (R1) | **True.** FIPER: DT "is calculated only for failed rollouts correctly flagged as Fail". | Our "following [FIPER, H&S] … DT = 1 − ℓ̄, a miss counting as 1" must say it differs from FIPER. |
| Hide-and-Seek uses a time-varying band, no PCA (R1) | **True for the paper text**: threshold ζ_t = μ_t + b_t; only window average pooling is described. | Say PCA, running max, constant threshold and ensembles are our adaptation; add the native band (Stage 2 or before submission). Check whether PCA comes from the released code. |
| ActProbe feeds t/T (R1) | **True.** ActProbe appends τ_t = t/(T−1); default α = 0.15. | Related Work and Sec. IV-A should say so; check whether our ActProbe run included t/T, and explain its unseen-task bACC 0.500. |
| "Intro 4.7% FPR vs G's 0.024 in Table II" (R1, R2) | Both correct numbers on different sets: 4.7% is the validation set (c = 0.903), 0.024 the test set. | Say "on the validation set", or quote the test-set number. |
| E2 has no no-reset arm (R3) | **Partly wrong.** Each scene first runs without reset, so the no-reset outcome exists; the paper no longer says it is reported. | Restore one clause: reset vs no reset is reported. |
| Continuing may beat resetting (R2, R3) | Existing data: the RLDX-1 continuation pilot (report §15.26) found only 18/444 = 4% of failures succeed when continued to 1.25 H_k. | One sentence answers this; the point of no return question remains. |
| "What is the fixed-path matched gain?" (R2) | Exists: R8 gives matched +0.057 (Hide-and-Seek) and +0.039 (SAFE, whose path uses fewer false alarms than P), all CIs include 0. | Add to Sec. IV-D. |

### Suggested triage

**A. Text fixes for the Stage 1 submission (no new runs).** Applied 2026-10-30 (all 11). To stay at
5 + 1 pages, Table III (FIPER per task) became one sentence in the FIPER paragraph (the table is kept
in the session scratchpad and git history), Fig. 3 is slightly shorter, and the last Limitations
sentence ("Only 10 annotated failures became irreversible", stated in Sec. VI-F) was dropped. Also
found while checking: our ActProbe omits the published normalized-timestep input; the paper now says so.
1. Credit FIPER for the length observation and ActProbe for its t/T input (Related Work, Sec. IV-A).
2. DT: state that a miss counts as 1, unlike FIPER.
3. Describe PCA, running max, constant threshold and ensembling as our adaptation of SAFE and Hide-and-Seek.
4. Intro "4.7% FPR": say validation set, or use the test-set G (Table I).
5. Abstract unseen tasks: add the matched value (0.051) next to 0.127; disambiguate "G loosened to its 20 false alarms".
6. H3: say SAFE-LSTM, and why (the MLP head falls back to the timer on RLDX-1).
7. Proposition 1: allow per-task c_k(λ) and state it for nested alarm sets on a finite grid Λ fixed in advance; cite nested conformal prediction and Learn-then-Test, and soften "we extend the construction".
8. Sec. IV-D: add the fixed-path matched gains (Hide-and-Seek 0.057, SAFE 0.039).
9. E2: restore "reset vs no reset is also reported"; mention the 4% continuation success.
10. Section heading "Hide-and-Seek: an earlier warning" and "clear gain" (Sec. VI-E) read stronger than the matched CIs; consider "a larger gain".
11. Clarify the validation calibration set size (3 + 3 per task) vs the test calibration set (10 + 10).

**B. Analyses on existing data worth running before 4 December (cluster).**
1. A time-only detector with per-task margins chosen under the same global FPR (budget-optimal timer), fused-style selection, as a stronger comparator; at least report its lead vs P.
2. Hide-and-Seek and SAFE with their native time-varying conformal bands, standalone and fused.
3. Matched-FPR gains for random-noise and resampled-score fusions (H2 as stated), with paired CIs.
4. Full-pipeline bootstrap (resample calibration and successes, refit and re-match) for the two central matched gains.
5. Lead in seconds and by atomic/composite; lead before the point of no return on the 10 decided failures.

**C. Stage 2 / Extension Charter decisions (author).**
1. Whether to add a short decision rule per experiment (the reviewers' strongest common request) despite the
   preference for concise hypotheses; a one-line rule in Sec. VII would answer it without "pass/fail" wording
   in the hypotheses.
2. A fresh XR-1 confirmatory test with fixed paths (R1, R2).
3. E2 arms (no reset, random reset at matched rate, oracle), budgets 1.25/1.5/2.0 H_k, success-tuned operating points.
4. LIBERO and multi-sample / action-consistency monitors (STAC with mid-chunk queries, FIPER's ACE).

---

## Reviewer 1 — failure detection

**Summary.** In success-terminated, time-limited benchmarks every failure is a timeout, so a duration-only
timer is a strong detector, yet VLA monitoring papers do not report one. The paper scores monitors by their
warning lead over a per-task timer P at the same and at matched FPR, and OR-fuses any monitor with the timer,
calibrated by split conformal along a monotone path (Prop. 1). On XR-1 (38 tasks, 380 + 380 test rollouts),
ensembles of SAFE and Hide-and-Seek add 0.030 and 0.071 of H_k at the same target, 0.015 and 0.056 at
matched FPR, with matched CIs including zero.

**Strengths.** Real, actionable evaluation confound (both timers catch every failure at about 5% FPR);
conservative matched comparison; good controls; candid reporting; the test credits FIPER on its own
benchmark; H4 aims at value, not a proxy.

**Weaknesses.**
- W1. Stage 1 evidence for H1/H2 is exploratory (adaptive path, reused test set, matched CIs include 0;
  fixed-path SAFE + P only 0.007), and E1–E3 never re-test H1/H2 on fresh XR-1 rollouts with fixed paths.
- W2. No decision rules (sidedness, α, multiplicity over 2 detectors × 2 comparisons) and no power analysis;
  five tasks carry 78% of the gain and the CI is about 0.13 wide, so a 20-task E1 may be underpowered.
- W3. Published time-aware calibration missing and methods modified: SAFE and Hide-and-Seek use a functional
  conformal band (μ_t + h_t); SAFE-MLP's score is cumulative; the paper uses a constant threshold on a
  per-task-normalized running maximum, adds 128-d PCA and 3-seed ensembles. ActProbe already appends t/T; its
  unseen-task 0.500/0.500 suggests an implementation problem.
- W4. E2 design: 79% of failures are recoverable, so resets discard progress; missing arms (continue without
  reset, P at matched FPR, random-time reset at the same rate); with P at about 0.78 H_k only about 0.47 H_k
  is left, so 0.05 H_k of extra lead may change nothing; report several budgets; scene restoration is not
  real-robot feasible.
- W5. The time-only comparator may be too weak: one shared margin vs two extra fused parameters chosen to
  maximize lead; a fair comparator gets the same adaptive selection (per-task margins, hazard timer).
- W6. Evidence narrower than the title: LIBERO is named but not evaluated.
- W7. Multi-sample and action-consistency monitors absent on RoboCasa365 (STAC with mid-chunk queries,
  FIPER's ACE, FAIL-Detect).
- W8. Presentation: dense numbers; abstract mixes matched (seen) and unmatched (unseen) numbers; Paper-ID
  placeholder.

**Questions.** (1) Is u_t SAFE's per-step score or its cumulative sum; constant threshold or band in Table I?
(2) G's c, and on which data "90% … 4.7% FPR" holds given G's 0.024 in Table II? (3) Why 3 + 3 vs 10 + 10
calibration rollouts per task; was the path grid tuned at the smaller n? (4) Fixed-path matched gains and
CIs? (5) "G loosened to its 20 false alarms": whose FPR? (6) Why does ensembling nearly double Hide-and-Seek's
gain while SAFE's unseen ensemble (0.017) falls below single models (0.028); were ablations ensembled and
path-optimized the same way? (7) Are per-task gains stable between validation and test; are atomic gains
larger because P is weaker there? (8) Lead relative to the point of no return on the 10 irreversible
failures; κ on the binary label or the timestep? (9) RLDX-1 training-set size? (10) How to read FPR under the
natural task mix?

**Hypotheses and methodology.** Clear and falsifiable in principle, H2 controls well designed; but no
decision criterion, H1 framed as supportive though not rejected at matched FPR, H2's matched ablations exist
only for the Hide-and-Seek LSTM, H3 names SAFE but E1 uses SAFE-LSTM chosen after seeing RLDX-1 results, H4
confounds monitor quality with retry policy and budget; E1 is in effect the only confirmatory test.

**Correctness checks.** Recomputed and correct: P's TWA 0.585, RLDX-1 H&S TWA 0.648, atomic/composite average
0.071, Table III average 0.146, 19 calibration successes per task, H_k range. Problems: abstract claims an FPR
bound but its headline numbers use the adaptive path; unseen 0.127 is unmatched (matched about 0.051); DT
definition does not match FIPER (which counts only flagged failures; lead = 1 − DT holds per rollout, not on
the mean); FIPER already noted STAC detects failures "primarily due to their greater length" and ActProbe
uses t/T; Hide-and-Seek has no PCA and uses a band; intro 4.7% vs Table II 0.024; FIPER operates at FPR
0.30–0.56, far from the 5% regime; Table II lacks ActProbe; bootstrap resamples only failures.

**Novelty.** The duration-only comparator at matched FPR is new and useful; Prop. 1 is a standard nested
reduction (Gupta et al.); Learn-then-Test should be cited and used. Missing: progress-based monitors and
hazard-based timers (Kaplan–Meier in refs.bib but unused).

**Extension Charter.** (1) Fresh XR-1 confirmatory test of H1/H2 with fixed paths, matched gains with CIs.
(2) Pre-specified decision rules and power analysis from task-level variance. (3) SAFE and Hide-and-Seek with
their native bands, alone and fused; ablate ensembling, running max, PCA. (4) Stronger time-only comparators:
adaptive per-task margins, hazard timer, per-subtask timers. (5) E2 arms: no reset, P at matched FPR,
random-time resets; budgets 1.25/1.5/2.0 H_k; atomic vs composite. (6) LIBERO; FIPER's benchmark at 5% and 10%
FPR. (7) STAC and FIPER's ACE on RoboCasa365, or justify. (8) Lead from the point of no return on ≥ 250 doubly
annotated failures. (9) Clarify all set sizes, G's c, RLDX-1 data; fix or explain ActProbe; credit FIPER and
ActProbe; cite Learn-then-Test and nested conformal.

**Score: 4/5, confidence 4/5.** Important, well-controlled methodological point reported honestly; main
effects not yet confirmed at matched FPR; Stage 2 needs decision rules, native baselines and a stronger E2.

## Reviewer 2 — statistics and conformal calibration

**Summary.** A failure is by construction a timeout, so a timer detects every failure and a monitor can only
add an earlier alarm at the same FPR. The authors OR-fuse monitors with a per-task timer and calibrate both
thresholds along a one-parameter path (Prop. 1). Ensembles of SAFE and Hide-and-Seek gain 0.030 and 0.071 at
the same target and 0.015 and 0.056 matched, both CIs including zero; ablations gain nothing. Stage 2: E1
on fresh RLDX-1 rollouts, E2 closed-loop reset and retry.

**Strengths.** Important, actionable point; the shuffle ablation is the correct null for learned elapsed time;
unusually candid reporting; the FIPER cross-check shows the metric can credit a monitor; E1 has the right
confirmatory structure (frozen detectors, H_k, ρ_k and paths).

**Weaknesses.**
1. No decision rules, and H1's verdict is never stated; "a policy-internal score" is existential over six
   scores. Pre-register detector, estimand, test, α, smallest effect; Holm across H1/H3 conjunctions.
2. Headline numbers not covered by the paper's own guarantee (adaptive path); fixed-path SAFE + P 0.007;
   fixed-path matched gain not reported. Make fixed-path results the headline or correct for selection
   (split, Learn-then-Test, Bonferroni over the grid); report the grid size.
3. P is not the strongest time-only monitor: a budget-optimal per-task rule (maximize Σ_k(1 − c_k) s.t.
   mean FPR_k ≤ α) and a randomized boundary inference (P's discreteness hands the fused detector free
   budget) should be the primary comparator.
4. Incomplete uncertainty: only two numbers have CIs; they condition on path, λ̂, matched timer, test
   successes, training data and one 3-model ensemble (ensemble nearly doubles HS's gain); two-stage bootstrap
   fragile with five tasks carrying 78%; sign test p ≈ 0.5; bold rankings ignore test noise. Re-run the whole
   pipeline per replicate, resample successes, ≥ 9 seeds with disjoint ensembles, task-level sign-flip test,
   state the estimand.
5. H2 not tested as stated (matched FPR): random and resampled only at same target; no paired CIs for the
   matched ablation contrasts; use a within-task permutation null.
6. Garden of forking paths on a reused test set: the matched clause of H1 is post hoc; ensembling and the
   SAFE-MLP → SAFE-LSTM switch not stated as pre-specified; atomic/composite moderator post hoc. Stage 2 needs
   a fresh XR-1 confirmatory test.
7. E1 probably underpowered: effective n is tasks (20 vs 38; XR-1 half-width about 0.067 for 0.056); gain on
   RLDX-1 only from atomic tasks. Power analysis from XR-1 variance components; prefer more tasks.
8. E2 cannot isolate H4: weak reset rule at 5% monitoring FPR; missing no-reset, random-score + P and oracle
   arms; plausible effect only a few points, predictable offline from success-duration distributions; common
   random numbers; sensitivity to the 1.25 factor.

**Questions.** (1) Were H1/H2 including the matched clause written before scoring; was ensembling chosen on
validation? (2) Matching with a discrete timer; whose false alarms in "G loosened to its 20"? (3) Grid size;
which pool was re-split 200 times; fixed-path matched gain? (4) How are seeds combined; why does averaging
nearly double the gain? (5) Exchangeability under stratified calibration; FPR under the natural task mix?
(6) Gains in seconds; do atomic tasks have short H_k (timer granularity)? (7) RLDX-1 provenance, split sizes,
why SAFE-LSTM? (8) E2: no-reset to 1.25 H_k; full reset; shared retry seeds? (9) How were the 48 annotated
failures sampled?

**Correctness.** Prop. 1 correct for nested alarm sets; four caveats: stated for scalar c but P uses c_k;
{λ: ρ_i < c(λ)} is open so the minimum may not exist (use an infimum or a fixed finite grid); "we extend the
construction" overstates (nested conformal, Learn-then-Test); no guarantee for unseen tasks or per task (one of
38 tasks at 4/10 suggests heterogeneous per-task FPR). Abstract unseen 0.127 unmatched (matched 0.051);
abstract leads with adaptive-path numbers; "neither elapsed time nor the typical trajectory", "trace … to the
rollout", "an earlier warning", "clear gain" lack CIs; "which favors the timer" not shown; Fig. 3b point
estimates across α share data; H3 says SAFE + P, E1 uses SAFE-LSTM; intro 4.7% vs G 0.024. Arithmetic checks
pass (TWA, bACC, DT, atomic/composite 0.071, FIPER 0.146, κ 0.80).

**Extension Charter.** (1) Pre-registered analysis plan for H1–H4. (2) Fixed-path primary results, same
target and matched, with CIs. (3) Budget-optimal per-task time-only monitor with randomized boundary as primary
comparator in E1 and E2. (4) Full-pipeline CIs, ≥ 9 seeds, ensemble rows in Table II. (5) Fresh XR-1
confirmatory test. (6) H2 at matched FPR with paired CIs and a permutation null. (7) E1 power analysis,
atomic/composite stratification. (8) E2: no-reset, random + P and oracle arms; success-tuned operating
points; common random numbers; predicted effect size; budget sensitivity. (9) Per-task FPR distribution,
natural-mix FPR, hierarchical guarantee for unseen tasks. (10) Lead in seconds and to the point of no return,
two annotators on every failure.

**Score: 4, confidence 4.** The duration-only comparator is a lasting contribution; methodology sound in spirit
and fixable; Stage 2 must close the adaptive-path, timer-comparator, decision-rule and E2 gaps.

## Reviewer 3 — robot learning practice

**Summary.** On benchmarks where episodes end at success or a time limit, failures are timeouts, so elapsed
time is near-perfect under recall-based metrics. The paper tests how much earlier a monitor alarms than a
per-task timer at the same or matched FPR, and OR-fuses the two with split-conformal calibration. SAFE and
Hide-and-Seek ensembles add 0.030/0.071 (same target) and 0.015/0.056 (matched, CIs include zero); ablations
add ≤ 0.012. Stage 2: confirmatory RLDX-1 study and closed-loop reset and retry.

**Strengths.** Useful sanity check (zero-input LSTM matches ROC-AUC; length AUC 0.92–1.00 on FIPER's
benchmark); careful controls; honest reporting; monitor-agnostic, easy-to-deploy fusion with correctly scoped
guarantee; E1 is a real confirmatory design.

**Weaknesses.**
1. The paper's own data undercut lead to the limit: 79% recoverable, mostly slow or nearly succeeded; on the 10
   irreversible failures P alarmed after the point of no return; lead to the limit rewards predicting
   slowness. Run failures past H_k to split slow from stuck; make lead before the point of no return primary.
2. Small effects tied to an arbitrary horizon: 0.056 H_k is about 1–12 s; gains concentrate in 10 atomic
   tasks; composite (long-horizon) tasks gain 0.008–0.039; report seconds and sensitivity to rescaled limits.
3. E2 cannot show practical value: no no-reset comparison, scene restoration not real-robot feasible, no
   minimum detectable effect; about 6% FPR resets about 30 would-be successes.
4. P may not be the strongest duration-only comparator (shared margin; later than P on 16 tasks;
   reallocation not tested by the random-score control); richer time baselines belong in E1.
5. Real-robot relevance and scope: oracle success check; real failures include false completion, drops,
   freezing; FIPER beats the timer by 0.385 on the real Pretzel task; one benchmark, two VLAs; per-task timer
   needs many successes; FPR guarantee marginal over tasks.
6. Novelty partly obvious in hindsight (SAFE App. B.5; lead = 1 − DT; standard split conformal).
7. H1 not established in Stage 1; headings read stronger than the evidence; headline numbers use adaptive paths
   (test FPR 0.061 vs 0.05 target).

**Questions.** (1) On the 10 irreversible failures, did any fused monitor alarm before the point of no return?
(2) How are T_k set; what if doubled? (3) What fraction of failures succeed if continued to 1.25 or 2 H_k?
(4) Does the no-reset run get the full 1.25 H_k; expected effect and power? (5) How much matched gain survives a
per-task-optimized timer? (6) Why 3 + 3 vs 380 + 380 calibration? (7) Results under natural task and failure
frequencies? (8) Exact H3 rule?

**Hypotheses and methodology.** Clear and falsifiable in principle; H1 bundles two comparisons over six
monitors (forking paths); H2 needs paired tests of the difference; H3 well matched to E1 (freezing paths is the
strongest part); H4 decides practical significance but compares only with P resets; no hypothesis on
irreversibility.

**Clarity.** Headline ensemble numbers only in text (Table II shows single models); Table II mixes P and G;
Fig. 1 draws the same timer threshold in both lanes though the fused detector can alarm later than P; Eq. (2)
and Prop. 1 use scalar c vs per-task c_k; "matched FPR" used before defined; "the test credits FIPER" opaque;
Fig. 3a bars hard to read, 3b gray lines overlap; RLDX-1 baseline set changes without explanation.

**Extension Charter.** (1) Redesign E2: no-reset, fixed-fraction and oracle reset arms; retry without scene
restoration; pre-registered MDE, power, decision rule; resets of would-be successes and wasted time.
(2) Extended-horizon relabeling (slow vs stuck). (3) Lead before the point of no return as a pre-registered
hypothesis (≥ 250 failures, double annotation). (4) Strongest duration-only comparator in E1 (budget-optimized
per-task margins; demonstration-length timer for unseen tasks). (5) Lead in seconds, atomic/composite,
sensitivity to rescaled H_k. (6) Explicit decision rules and multiplicity; headline ensemble table; CIs with
calibration uncertainty. (7) Mondrian per-task calibration in E1; per-task FPR distribution. (8) Scope: small
real-robot or fixed-horizon experiment, or limit claims to oracle-terminated protocols; LIBERO if feasible.

**Score: 4 (conditional on Charter items 1–4), confidence 4.** Rigorous, honest, broadly useful sanity check;
the insight is close to obvious in hindsight and the Stage 1 positive claim is not established; practical value
needs a no-reset baseline in E2 and lead tied to irreversibility.

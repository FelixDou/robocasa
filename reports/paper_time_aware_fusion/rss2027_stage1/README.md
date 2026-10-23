# RSS 2027 Stage 1 draft (due 4 Dec 2026, 11:59pm AoE)

Built from `../main.tex` (unchanged) on 2026-10-17, reframed after the mock review: every failed rollout ends at the time limit, so a calibrated timer catches all failures and the paper's endpoint is now **warning lead over the timer at matched FPR**.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (pdflatex gives Type-1 fonts only; checked with pdffonts).
Set `\notesfalse` in the preamble to hide the red/orange notes for the submission build.

## Before submitting
- **Template.** Done 2026-10-23. The official RSS package is in `template/`, and its `IEEEtran.cls` (v1.7a) is in this folder; keep it, since a newer system class changes the layout. The preamble starts with the template's block word for word (no `fontenc`) and adds `\IEEEpeerreviewmaketitle` and anonymous `\hypersetup` metadata.
- **Preregistration.** Done: Amendment 1 (PR #3) was opened before the lead endpoint was computed, and E1 has been run (see the 2026-10-21 update below).
- **Stage 1 rules.** The draft now reports a completed confirmatory test (E1) and plans E2 to E5 for Stage 2. Check that the RSS 2027 Stage 1 rules allow completed confirmatory results.
- **Control rate.** The seconds assume 16 steps x 20 Hz = 0.8 s per inference; confirm for the evaluation.
- **Bibliography.** Re-checked 2026-10-23 against arXiv and the venue pages: 22 entries, "et al." for 4+ authors. Two `% CHECK` comments remain. Hide-and-Seek: the ICML 2026 workshop is confirmed, but arXiv v2 says NeurIPS 2026. VLA-FAIL: arXiv v2 says CoRL 2026. Update both if the venues confirm. RSS desk-rejects hallucinated citations.
- **Page limit.** 5 pages of content and 1 page of references. A `\clearpage` before the bibliography puts the references alone on page 6; the preview with notes on is 6 pages.

## New numbers in this draft (exploratory, opened cohort)
From `analysis/lead_time.py` (run: `python3 -I analysis/lead_time.py review_data/`) and `analysis/extra.py` (noise+timer null, 20 draws; curves). The convention matches review_controls.py: an alarm at the final inference counts. `lead_time_strict.py` repeats it without that alarm (the JSAI/mock-review convention, timer c = 0.894).

| Detector | Recall | FPR | mean lead (frac. of limit) |
|---|---|---|---|
| Timer (c = 0.903) | 1.000 | 0.047 | 0.090 |
| Time risk + timer | 1.000 | 0.050 | 0.125 |
| Noise + timer (20 draws) | 1.000 | 0.066 | 0.101 |
| SAFE + time risk (frozen) | 0.955 | 0.058 | 0.184 |
| SAFE + timer | 1.000 | 0.058 | 0.207 (gain +0.117, CI 0.060-0.178) |


## Update 2026-10-17 (per-task timer, figure)

- **Per-task timer** (`analysis/per_task_timer.py`, `.json`). A timer with one cutoff per task (longest calibration success plus a shared margin) gets lead 0.248 at FPR 0.074. SAFE + per-task timer gets 0.271 at 0.076, a gain of 0.023 [0.002, 0.047], below the 0.05 margin. SAFE + global timer minus per-task timer is -0.041 [-0.111, 0.027]. All of SAFE's gain over the global timer sits in the 14 tasks where time risk stays silent. The paper and `preregistration_amendment.md` now use the per-task timer as the primary comparator.
- **Qualitative figure** (Fig. 1): `figures/qualitative/`, shared copy in `/mnt/project-files/figures/qualitative/`. Frames are placeholders until you run the TSUBAME command in `figures/qualitative/README.md`.
- The method diagram and the cohorts table were cut to make room; the text was tightened to about 4 pages of content.

## Update 2026-10-17 (second mock panel): one per-task timer

The per-task timer is now P. It uses the 95th percentile of each task's development success durations, with the margin set by Proposition 1 on the 114 calibration successes (`analysis/quantile_timer.py`, `.json`; the same file is in the repo under `scripts/xr1_lead_time/`).

- P: lead 0.195 at FPR 0.045.
- SAFE + P: 0.202 at 0.042, a gain of 0.007 (CI −0.003 to 0.019).
- Sensitivity by quantile (gain of SAFE + P over P):
  - q = 1.0: 0.057
  - q = 0.90: 0.028
  - q = 0.75: 0.064 (CI 0.029 to 0.099)

This replaces the 0.248 / 0.271 / 0.023 numbers above. Fig. 1 now shows P's alarm (t = 38). The amendment (PR #3) names P.

## Update 2026-10-21: confirmatory test (E1) added

E1 is in a new Sec. VII with Table III. Its numbers are from `docs/safe_robocasa_experiment_report.md` §15.13 (Tables 15.13a–d). Run root: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108`.

- H1 is not supported. SAFE + P has lead 0.255 against 0.225 for P, a gain of +0.030 (decision CI [+0.007, +0.093]), at FPR 0.061 (Wilson upper bound 0.089). The CI is above zero, but the gain is below the 0.05 margin.
- H2 is supported: noise + P and resampled SAFE + P gain at most +0.009.
- Descriptive results:
  - P leads the global timer by 0.102.
  - Frozen SAFE + time risk is 0.035 below P.
  - A P timer tuned on test to the same FPR leaves +0.015 [−0.007, +0.041].
  - Atomic tasks gain +0.081, composite tasks +0.012.
- To keep the page limit, the exploratory section lost the calibration-bootstrap sentence and the q-sensitivity sentence; q-sensitivity is now reported on the confirmatory cohort.
- Updated: the abstract, contributions, hypotheses, setup, Stage 2 (E1 marked done), limitations and conclusion.
- Page check: with tectonic/XeTeX and TeX Gyre Termes, content ends on page 5 and references fit on page 6. Confirm with pdflatex and the official template.
- A red PENDING note in Stage 2 asks whether a registered atomic-only test (Amendment 2) belongs in the plan.

## Update 2026-10-21 (b): exploratory follow-up (§15.14) and RLDX-1 offline (§15.15)

Every new number is exploratory: §15.14 reuses the opened E1 cohorts, and §15.15 is offline on the existing RLDX-1 SAFE20 data. Sources: `docs/safe_robocasa_experiment_report.md` §15.14 (Tables 15.14a–b), §15.15 (Tables 15.15a–b) and §15.12 (oracle).

- **Sec. VI, new paragraph "A second policy, offline (RLDX-1)".** All 438 failures run to the limit. P has lead 0.271 at 0.03 FPR, against 0.032 for the global timer. SAFE (LSTM) + P gains +0.034 [0.008, 0.067] at the same FPR. Atomic +0.049 [0.013, 0.097], composite 0. The MLP head adds nothing. The paragraph states the caveats: the fused path was chosen in-sample, the FPR margin rests on 60 successes, and noise + P matched the gain in 2 of 20 draws, both at 0.08 FPR. It ends with a post hoc, untested note: SAFE fired only above z = 7.93, which suggests extreme-tail separation.
- **Sec. VII, "Atomic tasks carry the gain".**
  - The four success-only baselines + P gain 0.020–0.034 (SAFE 0.030).
  - All five scores gain more on atomic tasks (0.045–0.081) than on composite tasks (0.008–0.027), and RLDX-1 shows the same split.
  - The oracle stage-overdue result (+0.071, task CI −0.021 to 0.129) is linked as "P likely lacks progress".
  - Length-matched AUC is reported only as *not* explaining the gain (baselines 0.59–0.61 against SAFE 0.55).
- **Stage 2.**
  - E1 now points to Sec. VII.
  - New E1b (Amendment 2) runs on atomic tasks only, with inference conditional on the 10 tasks, about 40/40 per task per cohort, and about 4,000 rollouts. kNN is not a secondary. A red `\pending` waits for the sample-size justification from the separate Stage 2 sizing session.
  - E2 now names RLDX-1, run prospectively with held-out calibration failures. A `\pending` asks whether RLDX-1 can run the 38 tasks.
- **H3.** An orange `\todo` proposes a rewording, because H1 failed its margin on XR-1. The registered text is unchanged; the author decides.
- **Limitations.** They now read: one policy prospectively, and a second (RLDX-1) offline, on existing data with a small test set.
- **Cuts to stay at 5 + 1 pages.**
  - The Stage 2 items E3–E5 are shorter.
  - So are the joint-search paragraph, the captions of Tables II and III, and the conclusion.
  - Fig. 2 is 3.6 cm tall.
  - Removed: the OR/AND frontier numbers (0.208/0.211), the per-task split behind "the two effects cancel" (0.092 against 0.180), and the limitations clause "that access bought 0.030".
- **Page check (tectonic + TeX Gyre Termes).** With `\notesfalse` (the submission build), content ends on page 5 and references end on page 6, which is tight: about 1–2 spare lines. With notes on, the red and orange notes push the references about 7 lines onto page 7. The committed `main.pdf` is the notes-on preview.

## Update 2026-10-21 (c): Stage 2 rewritten with the author's design decisions

The numbers come from `docs/stage2_sizing.md`, section Results, on branch `claude/stage2-sizing` (commit `173a0274`, not merged to main). The draft is anonymized, so it cites that file only in a source comment.

- **E1b is gone as a separate experiment.**
  - The sizing found at most 0.55 power at the 0.05 margin up to 60 per class per task.
  - The 0.089 atomic estimate is optimistic: about 0.06 when all retained successes are used.
  - The atomic motivation becomes a descriptive stratum of E2: the 14 atomic RLDX-1 tasks.
- **E2 (RLDX-1, its 20 tasks).**
  - Fresh calibration and test cohorts of 20/20 per task, about 2,500 rollouts (p95 2,672).
  - Held-out calibration failures.
  - Primary: SAFE + P vs P, superiority (decision CI > 0), within-task inference conditional on the 20 tasks. Power 0.92 at +0.034; CI > 0 passes in every simulated study, so the FPR rule decides.
  - The 0.05 margin is dropped (power ≤ 0.09).
  - Secondary: the task-level CI, with no power claim (about 0.54).
  - Caveat in the text: the shared-pool estimate is optimistic, with 5 held-out failures per task.
- **FPR rule (new paragraph, marked `\todo` as proposed).**
  - Calibration target: conformal α = 0.05.
  - Test check: the false alarms must not reject "true FPR ≤ 0.075" in a one-sided exact binomial test at 0.05, i.e. at most the 95th percentile of Binomial(n_s, 0.075). That is 39 of 400 in E2, against 30 under the old fixed rule.
  - The rule is looser, so E2's power stays ≥ 0.92.
  - H1's registered rule in Sec. IV is unchanged (E1 already ran under it).
- **E3.**
  - Registered contrast: P vs no monitor at a 1.25 H_k budget, about 650 episodes per arm on XR-1 (about 480 on RLDX-1).
  - SAFE + P is a descriptive third arm. The one exception within reach is XR-1 atomic at 1.0 H_k, about 1,450 per arm, on an optimistic bound.
  - A red `\pending` says the E3 policy is not chosen (XR-1 assumed).
- **H3 and H4.** Orange `\todo`s now say both must be reworded to match E2 and E3, with proposals. The hypothesis text itself is unchanged.
- **Cuts.**
  - E4 and E5 are folded into one compact paragraph.
  - The E3 description is shorter. It no longer says "count successes aborted by false alarms" or "a null result would bound…".
  - The length-matched-AUC sentence in Sec. VII is removed.
  - The RLDX-1 post hoc note is shorter.
- **Page check (tectonic).** With `\notesfalse`, the content ends on page 5 and the references end on page 6. With notes on, the references spill to page 7.

## Update 2026-10-22: point-of-no-return labels (report §15.17, exploratory)

Source: `docs/safe_robocasa_experiment_report.md` §15.17 and `docs/no_return_labeling_plan.md`. Main's other new section, §15.16 (physical signals), is not used.

- **Sec. VI.** "No early warning yet" became "No early warning, and mostly no early failure":
  - 38 of 48 labeled confirmatory failures (79%) were still recoverable at the limit: atomic 17/20, composite 21/28.
  - A second labeler agreed on 11 of 12 videos, kappa 0.80.
  - So failure here mostly means slowness, and the paragraph answers the "trivial by construction" objection.
  - The 2/3 vs 0/7 pre-emptive split is given only as an illustration.
  - The rough 0.35 H_k "wasted time" figure is left out, because the t_u timing is unreliable.
- **E3.** New "more time, no reset" arm, which runs to 1.25 H_k, as the comparator for alarm-triggered reset. Its sample size is `\pending`: the existing 1.25 H_k sizing covers only P vs none.
- **E5.** Reframed: it targets the minority of failures that have an early point of no return, with several labelers, a written protocol and adjudication. The two labelers' t_u timings differed by 0.24–0.34 H_k.
- **Limitations.** One labeler, a 12-video reliability check, and only 10 decided failures.
- **Contributions and conclusion.** Each now mentions the finding in a few words.
- **Cuts to stay at 5 + 1.**
  - The abstract's hypothesis sentence, the "Time" paragraph (its video detail now lives only in Table I), and the confirmatory protocol paragraph are shorter.
  - The "Task duration" paragraph is tighter.
  - Removed:
    - the sentence on the deployed FPR's Beta distribution;
    - the per-task timer tuned on test (−0.003);
    - the OR/AND frontier sentence;
    - the RLDX-1 post hoc z = 7.93 note.
- **Page check.** With `\notesfalse` the build is 6 pages, with content ending on page 5. With notes on it is 7.

## Update 2026-10-22 (b): published evaluations (report §15.18 FIPER, §15.19 SAFE boundary; exploratory)

Sources: `docs/safe_robocasa_experiment_report.md` §15.18–15.19 and `docs/published_eval_reanalysis_plan.md`. The FIPER numbers are its published Table 1; the timer was computed by us.

- **Sec. VI, new paragraph "The confound in a published benchmark".**
  - On FIPER's released rollouts, rollout length alone separates failures with ROC-AUC 0.92–1.00 in 4 of 5 tasks (0.62 in Sorting).
  - The timer was calibrated like FIPER's thresholds. It reaches mean accuracy 0.83 against FIPER's 0.78, but DT 0.63 against 0.30, so FIPER wins on TWA (0.65 against 0.57).
  - All caveats are in the text: published against computed numbers, not a paired re-run, T is the longest released rollout, and the conformal rank gives about 0.54 on the real-robot tasks.
- **Sec. III, new "Scope" paragraph.** The termination rule decides whether time is a strong baseline:
  - when successes stop at completion, it is;
  - when every rollout runs to a fixed horizon (SAFE's real-robot release, where 98% of failures and ≥ 30 successes per task stop at one step, or SimplerEnv), it is uninformative, and early-warning claims need failure-time annotations, which SAFE did not release.
- **Related Work.**
  - SAFE also uses a fixed horizon on its real robot.
  - FIPER reports no elapsed-time baseline, although length separates most of its tasks.
- **E4.** Adds a re-run of FIPER's code with the timer, a few words.
- **Cuts to stay within 5 + 1.**
  - **Table I (units of time) is removed.** The "Time" paragraph defines every unit, and the figure labels its frames in inferences.
  - Related Work is tighter: the CheckVLA description, the Stage 2 baselines clause and the RESCUE sentence.
  - E3 and E5 are shorter, and E5 is merged into the E4 paragraph.
  - Limitations and the conclusion are shorter. The conclusion no longer lists "early-decided failures" as a Stage 2 item; E5 still has it.
- **Page check.** With `\notesfalse` the build is 6 pages, with content ending on page 5 and the references ending on page 6.

## Update 2026-10-23: paired FIPER re-run (report §15.23) replaces the §15.18 comparison

Source: `docs/safe_robocasa_experiment_report.md` §15.23 and `docs/fiper_paired_plan.md`. The re-run used FIPER's own code (pinned `13d79c5c`) with one added `time` method.

- **Sec. VI paragraph, now "The test on a published benchmark".**
  - Kept: length ROC-AUC 0.92–1.00 on 4 of 5 FIPER tasks.
  - Replaced the published-versus-computed numbers (0.83/0.78 accuracy, DT 0.63/0.30) and their caveats with the paired result:
    - FIPER reproduces its published results within 0.05;
    - the timer has the best balanced accuracy, 0.81 against 0.78, at test FPR 0.09 against 0.35;
    - at FIPER's FPR, FIPER warns 0.146 [0.104, 0.188] earlier, on 4 of 5 tasks (post hoc).
  - Framed as the test working both ways: it credits FIPER, and finds SAFE's gain on XR-1 small.
- **Related Work.** In FIPER's own pipeline, an elapsed-time method matches its accuracy, but FIPER warns earlier at a matched FPR.
- **E4.** The "re-run FIPER's code" item is removed; it is done.
- **Limitations.** The FIPER lead comparison is post hoc, with 10 + 10 test rollouts per real-robot task, and PushChair is a lower bound.
- **Cuts.**
  - The duplicate "Part of SAFE + P's gain is its higher test FPR" sentence is removed from Limitations; the same point remains in Sec. VII.
  - The conclusion's first sentence, E2's sentence on the replaced atomic test, the FPR-rule paragraph, E3 and E5 are tighter.
- **Not folded in (needs a decision).** Main's §15.20–15.22 report that Hide-and-Seek + P passed a preregistered "Amendment 2" test on the same E1 cohorts: +0.071, CI [0.022, 0.157]. That clashes with this draft's text, which still describes the dropped atomic test as an "Amendment 2"-style item and uses SAFE as the only policy score.
- **Page check.** With `\notesfalse` the build is 6 pages.

## Update 2026-10-23 (b): Stage 2 protocol pass (mock review concerns 1, 2, 5, 6)

Sources: `mock_review.md` on `origin/claude/rss-stage1-mock-review`, and `docs/stage2_sizing.md` on main, section "E2 co-primary sizing".

- **H3** now states the conditional superiority claim that E2 tests:
  - on RLDX-1's 20 tasks, each co-primary (SAFE + P, Hide-and-Seek + P) beats P;
  - 97.5% decision CI above zero, Bonferroni over the two;
  - no 0.05 margin;
  - generalization to other tasks is a secondary with no power claim.

  The TODO is removed.
- **H4** now states the contrast that E3 tests: reset and retry on a P alarm vs continuing to the same 1.25 H_k budget. The TODO is removed.
- **E2.**
  - Names its frozen components: SAFE's LSTM head (the MLP falls back to the timer), Hide-and-Seek with three seeds, H_k as the longest development failure, and ρ_k as the 95th percentile of development success durations.
  - The cohort is fresh, with held-out calibration failures.
  - Motivation: Hide-and-Seek + P gained +0.071 (§15.21) and +0.083 on RLDX-1 offline (§15.22).
  - Decision rule: the CI is above 0, and so is the gain against P tuned on test to the fused detector's own FPR. That second condition is proposed (Amendment 1's secondary).
  - Type I error, power with this condition, and the final n are `\pending`. Without the condition, power at n = 20 is ≥ 0.98 for each co-primary.
- **FPR rule.** Still the binomial rule, now marked with a red `\pending` until the author confirms it.
- **E3.**
  - Primary (proposed): P-triggered reset vs continuation to 1.25 H_k. Secondary: P vs no monitor.
  - Paired by environment seed; a retry reuses the scene with a new policy sampling seed; McNemar's test on discordant pairs.
  - SAFE + P is descriptive (at least about 1,450 episodes per arm).
  - The secondary needs "at least about 650 per arm" under r = p_k. A pilot of about 100 same-scene re-attempts sets the final size (`\pending`).
  - Policy: XR-1, proposed (`\pending`).
- **E4 and E5** are now explicitly exploratory, with no decision rules.
  - E4 is cut to one clause.
  - E5 is sized at about 250–500 labeled failures (50–100 decided), with third-labeler adjudication. It is reported only if a pilot of ≥ 20 doubly labeled decided failures reaches mean |Δt_u|/H_k ≤ 0.10.
  - Registering E5 instead is a `\todo` for the author.
- **Abstract and conclusion.** Both now describe Stage 2 as two monitors on a second policy plus reset vs continuation.
- **Cuts.**
  - The "SAFE adds lead over a single timer" and per-task-timer exploratory paragraphs are shorter. The 0.405/0.221 split and the 20.2/11.7-inference detail are gone.
  - The time-risk paragraph, two contribution items, Limitations and the conclusion are tighter.
- **Page check.** With `\notesfalse` the build is 6 pages.

**Open items in the file:** eight in Stage 2, all hidden by `\notesfalse`.

| Line | Kind | Text |
|---|---|---|
| 437 | `\pending` | checkpoint hashes |
| 441 | `\todo` | matched-FPR condition proposed; author to confirm |
| 443 | `\pending` | type I error, power, final n |
| 448 | `\pending` | author to confirm the FPR rule |
| 452 | `\pending` | policy: XR-1 proposed, author to confirm |
| 456 | `\todo` | primary contrast proposed; author to confirm |
| 462 | `\pending` | pilot result; final size |
| 468 | `\todo` | E5 exploratory proposed; registering it is the author's call |

Also still open: the two `% CHECK` comments in `refs.bib` (Foresight author order, RoboCasa365 authors), and the qualitative figure's frames, which stay grey placeholders until the frame images exist.

**Not yet updated (outside this brief).** §15.21 (Hide-and-Seek + P passed a preregistered test on XR-1, +0.071) appears only as E2 motivation. The Preliminary Evidence and Confirmatory sections, the contributions and the FIPER paragraph ("finds SAFE's gain on XR-1 small") still present SAFE as the only learned monitor tested.

## Update 2026-10-22: Hide-and-Seek and Amendment 2

Numbers are from `docs/safe_robocasa_experiment_report.md` §15.20 (exploratory, August cohort) and §15.21 (preregistered Amendment 2, E1 cohorts).

- **Framing.** Elapsed time explains how well monitors *separate* failures. Lead over a per-task timer shows which ones also warn *earlier*. SAFE adds 0.030 (below the margin). Hide-and-Seek, an LSTM trained for early separation on the same features, adds 0.071 [0.022, 0.157] and passes H1 and H2.
- **Edits:**
  - Abstract, contributions, hypotheses note, conclusion: new framing and the Amendment 2 result.
  - Method: new paragraph "Sequence score and controls" (zero-input, same-task shuffle and first-inference controls; ActProbe reduces to chunk magnitude).
  - Exploratory evidence: new paragraph "A sequence detector does warn earlier".
  - Table II: added the three controls and Hide-and-Seek + P.
  - Sec. VII: new paragraph "Amendment 2", including that Hide-and-Seek − SAFE is not significant and that the decision bootstrap skews upward.
  - Limitations: Hide-and-Seek's selection and the registration timing.
  - Related work: one clause pointing to the controls.
- **Cut for the page limit:** the cumulative-lead figure (old Fig. 2), the within-task ROC-AUC sentence, and shorter RLDX-1, FIPER, duration and atomic/composite paragraphs.
- **E2 (decided 2026-10-23).** SAFE + P and Hide-and-Seek + P are co-primary on RLDX-1, each with a 97.5% decision CI above zero (Bonferroni over the two), and Hide-and-Seek − first-inference control is the declared secondary. H3 was rewritten to match, and the E2 todo was removed. The power at 97.5% is now filled in from `docs/stage2_sizing.md` (Results, 2026-10-23): at n = 20, 1.00 for SAFE and ≥ 0.98 for Hide-and-Seek under the E2 FPR rule, and 0.89–0.92 under H1's fixed limit.
- **Pages (tectonic preview).** Content still ends on page 5. The references already ran to page 7 before this update (limit 5 + 1). Check with pdflatex and the official template, and trim the bibliography if needed.

## Merge 2026-10-23: Hide-and-Seek line + paired-FIPER/Stage 2 line

I merged `claude/time-aware-fusion-paper` (up to `dea166db`, the Hide-and-Seek commits) into the drafts branch at `f5fb27eb`. The author asked to keep:
- **from the Hide-and-Seek side:** the evidence, abstract and contributions;
- **from the drafts side:** the paired-FIPER text, the Stage 2 protocol, H3/H4 and the Limitations;
- **from the Hide-and-Seek side, inside the drafts side's E2:** the E2 power numbers.

How each part was combined:
- **Kept from the Hide-and-Seek side:**
  - the abstract and contributions;
  - the Method paragraph on the sequence score and its controls (tightened);
  - the exploratory paragraph on Hide-and-Seek on XR-1;
  - the RLDX-1 paragraph with Hide-and-Seek;
  - Table III with the Amendment 2 rows, and the Amendment 2 paragraph (two sentences tightened);
  - the conclusion (shortened);
  - the removal of the lead-curve figure.
- **Kept from the drafts side:**
  - the paired-FIPER paragraph, now ending "credits FIPER, and Hide-and-Seek on XR-1, but finds SAFE's gain small";
  - H3/H4, and E2/E3/E4–E5 as protocol;
  - Limitations, with one sentence added: Hide-and-Seek was chosen as the better of two supervised baselines on the opened cohort, and registered on cohorts already opened for SAFE.
- **E2 now also carries:**
  - the co-primary power without the matched-FPR condition (1.00 for SAFE; ≥ 0.98 for Hide-and-Seek, even at 0.052);
  - the task-level power (0.33, 0.34);
  - power under H1's fixed limit (0.89–0.92), moved to the FPR-rule paragraph;
  - the Hide-and-Seek side's declared secondary (Hide-and-Seek + P vs first-inference control + P).
- **Cuts to stay within 5 + 1.**
  - Scope paragraph.
  - The point-of-no-return illustration is reduced to "P alarmed after the point of no return on all 10".
  - Related Work: the CheckVLA "could be fused" clause and the chunk-overlap sentence (the Method paragraph covers ActProbe).
- **Page check.** With `\notesfalse` the build is 6 pages, and it has no undefined references.

**Open items now:**

| Line | Kind | Text |
|---|---|---|
| 425 | `\pending` | checkpoint hashes |
| 428 | `\todo` | matched-FPR condition |
| 432 | `\pending` | type I error, power, final n |
| 436 | `\pending` | FPR rule |
| 440 | `\pending` | E3 policy |
| 443 | `\todo` | E3 primary contrast |
| 448 | `\pending` | pilot result, final size |
| 455 | `\todo` | E5 registered or exploratory |

## Update 2026-10-23: official template, references, tables

- **Template.** The official RSS package and class are in place (see "Before submitting").
- **References.** Ten citations the argument does not need were cut: Tri-Info, VLA-Scope, Foresight, REFLECT, Sinha et al., Code-as-Monitor, AHA, Zollo & Zemel, Cox and FLARE. Long author lists use "et al.", and the references sit alone on page 6.
- **Guarantee wording.** `confirm_lead.py` (`fit_fast`) and `quantile_timer.py` choose `(u, s)` by lead at each path's lambda-hat, which comes from the calibration successes. Proposition 1 therefore bounds each candidate path, not the selected one. Sec. V-C now says so and points to the registered empirical FPR rules. The abstract and contribution 2 are qualified the same way. No number changed. A TODO asks whether E2 should choose its fused paths on held-out rollouts.
- **Tables.** Tables I and II have one-line captions, with notes under the table and labelled row groups. Table II keeps its row order, with explicit headers for test-only intervals and the decision bootstrap. Bold marks the H1 comparisons.
- **Figures.** New Fig. 2 shows alarm curves of the timer, SAFE + timer, P and SAFE + P on the exploratory cohort. It reproduces Table I's @.75 column (`analysis/lead_curves.py` writes `figures/lead_curves.csv`). In Fig. 1 each lead arrow now has its own label.
- **Not rebuilt.** The committed `main.pdf` was not rebuilt. Rebuild it with pdflatex (tectonic previews embed Type 1C fonts, and RSS wants Type 1).

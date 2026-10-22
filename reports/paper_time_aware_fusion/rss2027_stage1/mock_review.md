# Mock review: RSS 2027 Stage 1, "Time Is the Baseline to Beat"

- **Reviewed version:** `reports/paper_time_aware_fusion/rss2027_stage1/main.tex` and
  `sections/related_work.tex` at `origin/claude/time-aware-fusion-paper` = `36f21937`
  (includes report §15.17–15.19). Line numbers below refer to `main.tex` at that commit, or to
  `related_work.tex` where marked RW.
- **Checked against:** `docs/safe_robocasa_experiment_report.md` §15.10–15.19 (origin/main),
  `docs/stage2_sizing.md` (origin/claude/stage2-sizing, `173a0274`),
  `docs/xr1_confirmatory_amendment_1_lead_time.md`, `docs/xr1_lead_time_e1_implementation_note.md`,
  `refs.bib`, and arXiv / proceedings pages for every 2026 reference (fetched 2026-10-22).
- **Call:** RSS 2027 two-stage process (roboticsconference.org/information/cfp, read 2026-10-22).
  Stage 1 is 5 pages of content + 1 of references; anything beyond the 6 pages is not reviewed.
  Reviewers and ACs judge the hypothesis and methodology. Experiments may be missing or
  preliminary. The outcome is reject or an invitation to rebuttal, after which the AC writes
  an Extension Charter to be executed by 16 Apr 2027. Negative results are not grounds for
  rejection. Hallucinated references are desk-rejected after automated checks.
  The call does not list "feasibility" or "importance" as named criteria, but a Charter that
  must be executed in about 10 weeks makes feasibility a practical criterion.
- Written as a critical external reviewer would write it. The paper was not edited.

---

## 1. Summary and recommendation

**What the paper claims.** On RoboCasa365 every failed VLA rollout runs to its task time
limit, so a timer catches every failure at about 5% FPR and recall cannot separate a learned
monitor from elapsed time. The authors propose mean warning lead at a fixed FPR as the
endpoint and OR-fuse a policy-internal score (SAFE) with a timer. Calibration is a split-
conformal rule along a nested path with a finite-sample FPR bound (Prop. 1). On an opened
cohort, SAFE + a global timer doubles the lead, but a per-task quantile timer P nearly
matches it. A preregistered test on a fresh cohort (E1) finds SAFE + P ahead of P by 0.030 of
the time limit (CI 0.007–0.093), below the registered 0.05 margin, so H1 is not supported.
Noise controls gain at most 0.009 (H2 supported). Exploratory support includes an offline
second policy (RLDX-1, +0.034), point-of-no-return labels (79% of failures still recoverable at
the limit), and a timer re-analysis of FIPER's released data. Stage 2 proposes E2
(superiority on RLDX-1), E3 (closed-loop reset and retry, P vs no monitor), E4 (checks) and
E5 (lead from the point of no return).

**Assessment.** The question is worth asking. Monitors in this literature are rarely compared
with elapsed time, and the FIPER re-analysis suggests that the omission matters outside
RoboCasa. The E1 test is unusually careful for the venue: frozen artifacts, disjoint cohorts,
a refitting decision bootstrap, null controls, and a reported failure to reach the margin.
The writing is honest about H1 in the Confirmatory section and the Limitations.

The submission is weakest exactly where Stage 1 is judged: **the Stage 2 protocol.**
- Two of the four hypotheses (H3, H4) contradict the experiments meant to test them; the draft
  itself marks this with TODOs.
- E3 has no chosen policy and no size for its key comparator arm.
- The new FPR rule is marked "author to confirm".
- E4 and E5 have no hypotheses, sizes or decision rules.
- E2, the only Stage 2 test of a learned monitor, has a decision rule that a fused noise score
  could plausibly pass (concern 2).

Several sentences in the abstract, conclusion and evidence sections also say more than the
exploratory data allow.

**Recommendation: Revise. In RSS terms, invite to rebuttal, borderline.** Accept only if
the rebuttal fully specifies E2/E3/E5 and aligns H3/H4 with them. As written, I would expect at
least one reviewer to recommend rejection on concern 1 alone.

| Criterion | Score (1–5) | One-line reason |
|---|---|---|
| Importance of the question | 3 | Useful evaluation point; risk of "obvious by construction" (concern 3) |
| Soundness of the protocol (Stage 2) | 2 | Hypotheses and experiments inconsistent; E2 rule permissive; E3/E5 underspecified |
| Feasibility | 3 | E2 sized; E3 sized only on an optimistic retry model; E5 unsized; no timeline |
| Preliminary evidence | 4 | A completed confirmatory test plus three exploratory checks, honestly labeled in most places |
| Confidence | 4 | |

---

## 2. Major concerns, ranked by how likely each is to sink the paper

### 1. The Stage 2 protocol is not yet a protocol (most likely to sink)

The call says hypotheses and methodology "must be fully fleshed out" at Stage 1. In this draft:

- **H3 contradicts E2 (L152–156).** H3 says "H1 holds for a second VLA", which includes the
  0.05 margin. E2 drops the margin (L444–445) and tests superiority conditional on 20 tasks.
  An orange TODO proposes new wording. A reviewer reading the `\notesfalse` build sees H3 and E2
  disagree with no explanation.
- **H4 contradicts E3 (L158–162).** H4 says the *OR-fused* monitor beats the timer on success.
  E3 registers P vs no monitor, with SAFE + P descriptive (L459–461). As planned, Stage 2
  therefore never tests H4 as stated.
- **E3 is incomplete (L457–463).** The policy is "not chosen" (PENDING), and the sample size of
  the "more time, no reset" arm is PENDING. It is also not stated which contrast decides H4:
  P vs none, or P vs no-reset. The analysis for "paired by seed" (for example McNemar on
  discordant pairs) is not stated, and neither is how a retry is randomized (see concern 5).
- **The FPR rule is provisional (L451).** It is marked "Proposed rule; author to confirm".
- **E4 (L465–468) is a list of checks** with no hypotheses, sizes or decision rules. Either
  label it exploratory or drop it.
- **E5 (L468–471) has no size or rule.** It gives no number of labeled failures and no number
  of labelers. It has no reliability gate, endpoint definition or decision rule (concern 6).
- **E2 does not name the SAFE head.** Offline, only the LSTM head gains (+0.034); the MLP head
  falls back to the timer and gains 0 (§15.15a). E2 must name the head and the frozen
  checkpoint. It must also state how H_k and rho_k are defined for RLDX-1: H_k is the longest
  training failure (§15.15), not the official limit.

With `\notesfalse` every one of these notes disappears silently, so the risk is that the
submitted PDF *looks* complete while the design is open.

### 2. E2's decision rule can probably be passed by false-alarm budget alone

E2 is the only Stage 2 test of whether a *learned* monitor adds anything. Its rule is:
superiority with a within-task CI, under a looser FPR rule.

- **The within-task CI is very narrow.** At n = 20 the within-task SD is 0.007
  (sizing Table S4). Any true gain above about 0.015 passes "CI > 0" in essentially every study,
  so the FPR rule decides (the paper says so, L443–444).
- **The new FPR rule is permissive.** It passes unless an exact binomial test rejects
  "FPR ≤ 0.075". With 400 test successes that admits up to 39 false alarms (9.75%). For
  example:
  - a detector with true FPR 0.09 passes with probability 0.73;
  - one with true FPR 0.10, twice the conformal target, passes with probability 0.48.

  Under H1's rule (≤ 0.075 and Wilson ≤ 0.10) these probabilities are 0.09 and 0.02.
  (Computed here with scipy; "39 of 400" in L455 is correct.)
- **The authors' own data show that FPR budget buys lead.**
  - On XR-1, fused noise reached raw gains up to +0.028 (single score) and +0.042 (best of
    nine) by spending false alarms (§15.16b).
  - Half of E1's gain disappears at matched FPR (+0.015, CI −0.007 to 0.041; §15.13b).
  - On RLDX-1, two of 20 noise draws gained +0.038 and +0.049 at FPR 0.08 (§15.15). Both would
    pass the new binomial rule at E2's size (0.08 × 400 = 32 ≤ 39). With a within-task SD
    near 0.007, their CI would likely exclude zero too. This is an extrapolation from 100
    offline test rollouts, which is why the type I error must be simulated.
- **The type I error is unknown.** The sizing doc simulates power only. It never simulates how
  often a null fusion (noise + P, or within-task resampled SAFE + P) passes E2's rule. Without
  that number the E2 test cannot be judged.

**Required fix.** Make the null control part of the E2 decision, not a separate descriptive
check. Either:
- (a) the SAFE + P gain must exceed the 95th percentile of the noise + P and resampled
  SAFE + P gains, refitted the same way; or
- (b) the lead gain against P tuned to the fused detector's test FPR (already a registered
  secondary in Amendment 1) must have CI > 0.

Then report the rule's simulated type I error under both nulls next to its power. Keep the
H1 FPR rule, or bound the paired FPR *difference* to P instead of the absolute FPR.

A secondary point: within-task inference makes E2 a claim about these 20 tasks. The task-level
CI has power 0.54 at n = 20 and needs n ≈ 50, about 6,200 rollouts (sizing Table S4b). H3's
wording ("generality") must match that conditional claim.

### 3. "Trivial by construction": §15.17–15.19 answer half of the objection

The objection a reviewer will raise: *"If failure is defined as reaching the time limit, then
of course a timer catches every failure. The paper's headline is a tautology of the benchmark's
termination rule."* There are two versions:

- **(a) Tautology.** §15.17 does not refute this; it confirms it. 79% of failures were still
  recoverable at the limit (38/48; second labeler κ = 0.80 on 12), and the most common category
  is "slow or nearly succeeded". The paper says so directly (L348–349: "an early 'failure' alarm
  mostly predicts slowness"). That is an honest and interesting finding, but it **undercuts the
  primary endpoint for most failures**. Lead time rewards early alarms on failures that a reset
  may make worse. In the report's §18 operator screen, continuation scored highest of all
  interventions (22.5% vs 17.5%), though its CIs include zero. The paper
  needs a sentence that reconciles this. The lead endpoint is meaningful for decided failures
  (E5) and for a utility test (E3), not as an end in itself.
- **(b) Benchmark artifact.** §15.18 answers this best. In a published NeurIPS benchmark with two
  real-robot tasks, length alone gives ROC-AUC 0.92–1.00 on four of five tasks, and the method
  did not report a timer. §15.19 honestly marks the boundary: fixed-horizon evaluations such as
  SAFE's real-robot set carry no length signal. Together they turn "artifact" into a claim about
  evaluation design: the termination rule decides whether time is a baseline. That claim is
  defensible and is the paper's best contribution. The FIPER caveats (concern 7) weaken how far
  it can be pushed.

**Recommendation.** State the objection in the introduction and concede (a) explicitly: yes, a
timer is strong by construction. Then make the contribution the consequences of (a):
- published monitors are evaluated without the baseline that the construction makes strong (FIPER);
- the policy signal adds 0.03 of the limit beyond it;
- on this benchmark "failure" mostly means slowness, so evaluation should move to decided
  failures (E5) and closed-loop utility (E3).

Currently the paper raises the question (L343) and answers it obliquely.

### 4. Overclaims and claims stronger than the evidence

The Confirmatory section and the Limitations describe H1 accurately: "not supported", "below the
margin of 0.05 we declared in advance" (L64–65, L407–411). Elsewhere:

| Where | Text | Problem | Suggested wording |
|---|---|---|---|
| Abstract L61 | "finds a small **real** gain over this per-task timer" | Half of the gain is FPR budget. The registered matched-FPR secondary leaves +0.015 (CI includes 0). "Real" reads as confirmed. | "a gain whose CI excludes zero but which is below the registered margin, and half of which is extra false-alarm budget" |
| Abstract L56–58 | "more than doubles the mean warning lead (0.090 to 0.207 …, exploratory)" | This is the abstract's first result, yet it is the weakest: an opened cohort, a joint search with no guarantee, and fused FPR 0.058 against 0.047. | Lead with the confirmatory number. Cut or demote the doubling. |
| Intro L84–87 | Decision rules "fixed before the confirmatory data were analysed" for "four hypotheses" | H3/H4 rules are not fixed (TODOs). The cohort had already been scored for recall when the implementation note was committed (§15.13). It contradicts L376–377. | "H1 and H2 … registered before any lead quantity was computed on the confirmatory cohort" |
| L415 | "What SAFE adds is policy evidence, not an artifact of fusing any score with a timer." | H2 tests that the nulls stay below 0.05, not that SAFE beats them. Four label-free OOD scores gain 0.020–0.034, so the evidence is generic hidden-state novelty, not SAFE. | "Noise and resampled scores do not reproduce the gain (max 0.009 vs 0.030 in 20 draws each)." State the rank (empirical p < 1/21). |
| L429–430 | "The split is a property of atomic tasks under the per-task timer, not of SAFE, and RLDX-1 shows it too." | A descriptive subgroup on opened cohorts, with FPR not controlled per stratum. RLDX-1 has only 6 composite tasks. | Add "descriptively"; drop "is a property". |
| L430–432 | "On composite tasks P **likely** lacks progress: … an oracle … gained 0.071" | The oracle check was **NO-GO** under its own rule (task CI −0.021 to 0.129), on an opened cohort of 5 tasks. | "An oracle reading subtask predicates gained 0.071 on five composite tasks, but its task-level CI included zero." |
| L356–357 | "at the *same* FPR, so unlike on XR-1 none of it is extra FPR budget" | The FPR resolution is 1/100 (3 FP each), and the plain noise comparison gives p ≈ 0.14 (§15.15). The p-value is omitted. | Add "noise p ≈ 0.14". Keep the in-sample-path caveat. |
| L364–368 | Timer "reaches mean accuracy 0.83 against FIPER's published 0.78 … its advantage is earliness, the comparison we argue for" | See concern 7. Under the finite-sample rank, the mean is about 0.63–0.69. FIPER wins on accuracy on both real-robot tasks. FPR is not matched, so this is *not* the comparison the paper argues for. | See concern 7 |
| L349–351 | P after the point of no return on all 10; SAFE + P pre-emptive on 2/3 atomic | The two labelers' t_u differed by 0.24–0.34 H_k (§15.17), and this claim depends on t_u timing. The caveat appears only in E5 (L470). | Add "(timing labels unreliable; see E5)". |
| Conclusion L486–487 | "a learned monitor can only warn earlier than a timer …, here and on a published benchmark" | The published-benchmark part is an unpaired, exploratory re-analysis. | "…, here, and, in an exploratory re-analysis, on a published benchmark" |
| L98–100 | "monitors should … be evaluated on failures that end early" | Unclear: on RoboCasa no failure ends early. The likely meaning is failures *decided* early (E5). | "on failures that become unrecoverable early" |
| Scope L128–129 | "… or SimplerEnv, it is uninformative" | There is no citation and no data on SimplerEnv in the report. | Cite SimplerEnv and its termination rule, or drop it. |

### 5. E3 tests something close to true by construction, under an optimistic retry model

- **P vs none at 1.25 H_k is nearly guaranteed by design.**
  - In the no-monitor arm a failure is known at H_k, leaving 0.25 H_k: too little for most
    retries. In sizing Table S6 the no-monitor arm gains only 0–0.048 at 1.25 H_k, against
    0.075–0.182 for P.
  - P alarms at about 0.78 H_k on average (1 − 0.225), leaving about 0.47 H_k.
  - A positive result shows that stopping before the limit leaves time to retry. It says
    nothing about learned monitors, which is the paper's subject. SAFE + P is descriptive
    because the SAFE-specific contrast needs about 1,450 episodes per arm at best
    (XR-1 atomic, 1.0 H_k) and tens of thousands at 2 H_k.
- **The decisive comparator is the "more time, no reset" arm.** With 79% of failures
  recoverable at the limit, continuing to 1.25 H_k may beat reset; §18's operator screen found
  continuation was the best arm, with CIs including zero. This arm is currently unsized
  (PENDING) and its role is
  undefined. Register P-reset vs continue as the primary contrast, or as co-primary with
  multiplicity control.
- **The sample size rests on retry success r = p_k.** That assumes each retry is an independent
  fresh attempt (sizing doc, Limits). If a retry restarts the *same* scene and the policy is
  close to deterministic given the seed, r could be near 0 for hard scenes. The effect, and so
  the 650-per-arm figure, would then shrink sharply. "Paired by seed" also changes the analysis
  from the unpaired two-proportion formula used for 650. **Fix before Stage 1:** a pilot of
  about 100 re-attempts of failed XR-1 scenes to estimate r, and size E3 from that.
- The 650 figure is a lower bound, because the effect is an upper bound (sizing doc). L460 should
  say "at least about 650".

### 6. E5 is not sized and its main measurement is not yet reliable

- Only 10 of 48 labeled failures (21%) were decided before the limit. To get, say, 50–100
  decided failures for a lead-from-t_u estimate, E5 needs about 250–500 labeled failures,
  each with several labelers.
- The second labeler agreed on the recoverable call (κ = 0.80) but **not on timing**:
  t_u differed by 0.24–0.34 H_k on all three doubly-decided videos, which is about the size of
  the effects of interest.
- E5 needs:
  - a reliability gate before it runs, for example an ICC or mean absolute t_u difference
    below a fixed bound on a pilot;
  - an adjudication rule;
  - a primary endpoint (lead from t_u for SAFE + P vs P on decided failures);
  - a sample size.

  None of these is in L468–471.

### 7. FIPER comparison: the caveats the paper omits are the ones that flip the headline

L362–370 lists three caveats: published vs computed numbers, the horizon definition, and the
conformal rank on real-robot tasks. Missing:

- **The threshold rule decides the accuracy claim.** Under the finite-sample conformal rank,
  the rule the paper itself uses for P and advocates in Prop. 1:
  - Pretzel falls to 0.535 and PushChair to 0.55;
  - the three simulated tasks give 0.68–0.78 (§15.18);
  - so the five-task mean is about 0.63–0.69, *below* FIPER's 0.78.

  The 0.83 headline holds only under the linear-quantile rule. That is FIPER's own rule, so it
  is a fair like-for-like choice, but the paper must say both.
- **FIPER beats the timer on both real-robot tasks** on accuracy: 0.85 vs 0.80 and 0.96 vs 0.90.
  The timer's accuracy lead is entirely in simulation. L363 mentions "two on a real robot" in
  the confound sentence, which invites the opposite reading.
- **FPR is not matched.** FIPER's TNR at its operating point is not given in the paper. "Its
  advantage is earliness, the comparison we argue for" (L367–368) claims a matched-FPR
  earliness comparison that was not made.
- **There are no intervals.** The real-robot tasks have 10 successes and 10 failures each, and
  the paper gives no uncertainty for them.
- **DT and TWA depend on T,** here the longest released rollout. Check it against FIPER's
  configuration before submission. It is cheap and removes a caveat.

E4's "FIPER's code re-run with the timer" (L468) is the right fix. Make it a registered
Stage 2 item with a matched-FPR, paired endpoint.

### 8. The registration record is misdescribed, in the paper's disfavour

- **Timeline.** L376–378 and Limitations L477–478 say the amendment making P the comparator and
  lead the endpoint came "with its recall and FPR known". Amendment 1 is dated 2026-10-17 and
  states it was committed before any confirmatory rollout was opened or scored (PR #3, 2026-10-17
  22:06 JST). The cohort was first scored on 2026-10-19 20:51 JST (§15.13). Only the
  *implementation note* (`cb5ad443`) came after recall was known. It fixed the calibration seed
  range, the refitting decision bootstrap and the H2 mean-over-20-draws rule.
- **A deviation to disclose.** Amendment 1 required the decision bootstrap to be "added and
  committed before the confirmatory cohort is scored". It was committed after the recall scoring
  but before any lead quantity. Disclosing this in one line is cheap. Without it, a careful
  reviewer may find it.
- **The secondary check is not part of H1.** L144–146 says the fused detector "must also beat" a
  test-tuned timer, and L421 says "the secondary check of H1 also fails". In Amendment 1 this is
  a *secondary endpoint*; H1's rule has three criteria and "every other comparison is
  descriptive". The paper adds a fourth H1 criterion after the fact. It makes no difference to
  the verdict, but in a registered-report setting any mismatch with the registration costs
  credibility.
- **Verifiability.** "Preregistered" appears throughout, but reviewers cannot see the
  registration: RSS forbids external links in the blinded PDF. Offer an anonymized, timestamped
  registration (for example an anonymous OSF snapshot) to the AC through the submission system,
  or say it will be released at Stage 2.

### 9. Feasibility is asserted, not shown

The Charter must be executed between the Stage 1 decision (5 Feb 2027; rebuttal 12 Feb) and the
final paper (16 Apr 2027). Stage 2 as planned needs:
- E2: about 2,500 RLDX-1 rollouts, about 206k policy inferences (sizing Table S5);
- E3: at least about 650 episodes per arm × 2–3 arms, each up to 1.25 H_k;
- E5: labeling of several hundred failures by several people;
- E4.

The paper gives no compute estimate and no timeline. One line, for example "E1's 1,522-rollout
cohort took N GPU-hours over three allocations; E2 ≈ M GPU-hours", would answer the question.
Also state whether collection will start before the Stage 1 decision.

### 10. Smaller methodological points a statistically minded reviewer will raise

- **Prop. 1 is a lemma, not a contribution.** It is the standard split-conformal quantile lemma
  applied to a scalar. Present it as a lemma. Contribution 2 (L88–91) oversells it.
- **The guarantee is marginal and runs above nominal.** Test FPR runs above α: 0.055 for P and
  0.061 for SAFE + P in E1; 0.057–0.061 averaged in the sizing simulations, where the doc notes
  the cohorts "are not exchangeable in the success tail". Report per-task FPR. Say plainly that
  the guarantee is marginal over tasks and in expectation over calibration draws.
- **H2 is weak as a test.** Its rule (nulls < 0.05) is nearly certain to hold whenever the
  detector itself is below 0.05. A rank-based rule (SAFE gain above all 20 null draws) is more
  informative at no cost.
- **Unclear noise AUC (L333–334).** Within-task ROC-AUC "0.41–0.51 for noise": an AUC well below
  0.5 for pure noise needs a one-clause explanation (at-risk set, ties) or it looks like a bug.

---

## 3. Numbers audit

Every number in `main.tex` and RW was traced. Sources are the report (§15.10–15.19), the sizing
doc, Amendment 1 and the implementation note. Below are the mismatches first, then what matched,
then numbers whose source lies outside the checked documents.

### 3a. Mismatches and imprecisions

| # | Line | Draft says | Source says | Severity |
|---|---|---|---|---|
| N1 | L376–378, L477–478 | Endpoint amendment made "with its recall and FPR known" | Amendment 1 committed 2026-10-17, before any scoring. The cohort was first scored 2026-10-19 20:51 JST (§15.13). Only the implementation note `cb5ad443` followed the recall scoring. | High (misstates the registration) |
| N2 | L144–146, L421 | H1 "must also beat" a test-tuned timer; "the secondary check of H1 also fails" | Amendment 1: a secondary *endpoint*; the H1 rule has 3 criteria; "every other comparison is descriptive" | High (registration mismatch) |
| N3 | L84–87 | Decision rules for "four hypotheses" fixed "before the confirmatory data were analysed" | H3/H4 rules open (TODOs); recall already analysed (§15.13) | High |
| N4 | L446–447 | Atomic XR-1 test "with at most 0.55 power at the 0.05 margin up to 60 per class per task" | 0.55 is full-rule power at an *assumed gain* of 0.0695, n = 60. At a true gain of 0.05 it is 0.32 (Table S2). | Medium (reads as power at γ = 0.05) |
| N5 | L459–460 | "P against no monitor, paired by seed (about 650 episodes per arm)" | 650 comes from an *unpaired* two-proportion formula on an upper-bound effect with r = p_k, so it is a lower bound (sizing E3). A paired design was not sized. | Medium |
| N6 | L460–461 | Separating SAFE + P from P "needs thousands to over a million episodes per arm" | At 2 H_k: 73,749–1,520,098 ("tens of thousands" at minimum). "Thousands" applies only at 1.0 H_k (1,450 XR-1 atomic; 23,000 RLDX-1 atomic). | Low |
| N7 | L366, L370 | Timer mean accuracy 0.83 vs 0.78; real-robot conformal rank "about 0.54" | Correct as stated, but the same sensitivity line gives 0.68–0.78 for the simulated tasks, so the conformal-rank mean is about 0.63–0.69 (§15.18) | Medium (omission; see concern 7) |
| N8 | L111 | "(29 to 272, or 22.5 to 217.5 s)" with "0.8 s each" | 29 × 0.8 = 23.2 s. The seconds are T_k / 20 Hz (450 and 4,350 steps), not H_k × 0.8. Internally inconsistent by up to 0.7 s. | Low |
| N9 | L148–150, L413–414 | H2 "gain less than 0.05"; result "at most 0.009" | Registered rule is the **mean** over 20 draws (implementation note); 0.009 is the **max**. The table shows both, but the text conflates the rule's statistic with the reported max. | Low |
| N10 | L356–360 | RLDX noise caveat: "matched the gain in 2 of 20 draws, both at 0.08 FPR" | Correct, but §15.15 also gives the unconditioned p ≈ 0.14 and a mean noise gain of +0.009. Omitting p ≈ 0.14 overstates the evidence. | Low–medium |
| N11 | L347 | "most often slow or nearly successful" | 13 of 48 (27%) is the plurality category, not a majority | Low |
| N12 | L446 | "Secondary, no power claim: the CI with tasks resampled (power about 0.54)" | 0.54 matches Table S4b, but "no power claim" followed by a power figure is contradictory | Low (clarity) |

### 3b. Checked and matching (no action)

- **Abstract, intro.**
  - 380 failures; timer flags all at 0.047 FPR; about 12 inferences (mean 11.7, median 11.5).
  - 0.090 → 0.207; P nearly closes the gap (0.195); 760-rollout fresh cohort.
  - 0.255 vs 0.225, gain 0.030 [0.007, 0.093] at 0.061; nulls ≤ 0.009.
- **Setup.** 38 tasks (10 atomic / 28 composite); 29–272 inferences; development 1,824 rollouts.
  This is the outer-training split of §15.10/§15.13. §15.16's "1,900" also includes the 76-rollout
  outer test, so it is consistent. Calibration 3 + 3 per task, 454 retained.
- **Table I and Fig. 2 text** (§15.10, Amendment 1 dry run, paper README).
  - Timer c = 0.903, 1.000 / 0.047 / 0.090.
  - Time risk 0.632 / 0.047. SAFE only 0.566 / 0.039.
  - SAFE + time risk 0.955 / 0.058 / 0.184. Per-task timer 0.045 / 0.195.
  - SAFE + P 0.042 / 0.202. SAFE + timer 0.058 / 0.207. Noise + timer 0.066 / 0.101.
  - Time risk + timer 0.050 / 0.125.
  - 140 missed failures. Fused recalls 0.905–0.955 ("0.90–0.96"). 42% by 0.75 H_k.
  - Gain 0.007 [−0.003, 0.019]. Calibration vs test FPR 0.044 vs 0.058–0.068.
- **Fig. 1 annotations.** t = 30 / 38 / 60 of H_k = 66; leads 36 / 28 / 6; 36 × 0.8 ≈ 29 s.
- **Point of no return (§15.17).**
  - 48 failures (2 per atomic, 1 per composite); 38 (79%) recoverable; atomic 17/20, composite 21/28.
  - κ = 0.80, 11/12. 10 decided; 0/10 pre-emptive for P; 2/3 atomic and 0/7 composite for SAFE + P.
  - t_u differences 0.24–0.34 H_k.
- **RLDX-1 (§15.15).**
  - 20 tasks; 740 / 60 / 100 + 100; 438 failures at the limit.
  - P 0.271 at 0.03; global timer 0.032.
  - +0.034 [0.008, 0.067]; atomic +0.049 [0.013, 0.097]; composite 0; MLP 0.
  - 2/20 noise draws at 0.08.
- **FIPER (§15.18).** Five tasks, two real; AUC 0.92–1.00 (0.915–1.000) in four, 0.62 in Sorting.
  Accuracy 0.83 vs 0.78; DT 0.63 vs 0.30; TWA 0.57 vs 0.65; about 0.54 on the real-robot tasks
  under the conformal rank.
- **SAFE real robot (§15.19).** 98% of failures (887/906) and ≥ 30 successes per task at one
  fixed step; no failure-time annotations released.
- **Table II (§15.13a/b).** Every row: global timer, frozen SAFE + time risk, SAFE + global
  timer, P, noise + P, resampled SAFE + P, SAFE + P, with the CIs. Wilson upper bound 0.089.
- **Confirmatory prose.**
  - Tuned P +0.015 [−0.007, 0.041]; tuned global +0.024 [−0.022, 0.074].
  - q-sensitivity 0.026–0.030, 0.074 at q = 1 with P 0.164.
  - Atomic 0.081 (0.372 vs 0.291); composite 0.012.
  - Baselines + P 0.020–0.034; atomic 0.045–0.081; composite 0.008–0.027.
  - Oracle +0.071 [−0.021, 0.129]; opened-cohort 0.007.
- **Stage 2 (sizing doc).**
  - E2: about 2,500 rollouts (2,497); power 0.92; margin power ≤ 0.09; task-level power 0.54;
    14 atomic RLDX-1 tasks.
  - Atomic gain about 0.06 with all retained successes, 0.089 optimistic; 5 held-out failures
    per task.
  - "39 of 400" (exact one-sided binomial 95th percentile, recomputed).
  - E3: RLDX-1 about 480 per arm; 1,450 XR-1 atomic at 1.0 H_k.
- **Method.** 19 calibration successes for a per-task guarantee at α = 0.05; q-sensitivity set
  {0.75, 0.90, 1.0} as registered.

### 3c. Not traceable to the checked documents (verify before submission)

- L333–334: within-task ROC-AUC 0.55–0.60 (SAFE) vs 0.41–0.51 (noise). From `analysis/extra.py`
  per the README; not in the report. The 0.41 lower bound for noise needs a check (concern 10).
- L330: "20.2 against 11.7 inferences".
- L338: SAFE + timer vs per-task timer, 0.012 (CI −0.044 to 0.063).
- L340: "0.405 against 0.221" in the tasks where time risk never fires.
- L174–175: "1,024 dimensions" of the action-head hidden state.
- L381: 2,000 replicates for the *decision* bootstrap. The implementation note says 2,000 with
  seed 0, so this is consistent, but cite the note.
- L108: RoboCasa's 20 Hz control rate (open README item).

---

## 4. Minor issues

### Citations (RSS desk-rejects hallucinated references; automated checks)

Every 2026 entry was looked up on arXiv or the venue page. **None is fabricated**, but five
entries carry metadata that an automated checker could flag:

| Key | Problem | Fix |
|---|---|---|
| `wang2026foresight` | **Author order wrong.** arXiv 2606.23085 lists Haoran Zhang, Yifu Lu, Boyang Wang, Xuhui Kang, Yen-Ling Kuo, Zezhou Cheng, Mengdi Wang, Odest Chadwicke Jenkins. The bib starts "Wang, Boyang", as its own CHECK comment feared. | Reorder the authors; rename the key `zhang2026foresight` |
| `zollo2026calibration` | **Venue out of date.** arXiv 2507.17383 v3 (16 Oct 2026): "published in TMLR"; first posted July 2025. The bib says ICML 2026 workshop. | Cite as TMLR; check the year on the TMLR page |
| `park2026hideseek` | arXiv 2605.30834 v2 comment: "NeurIPS 2026"; the bib says ICML workshop. The arXiv author list gives the 4th author as "Samuel Yeh"; the bib "FIXED" it to "Min-Hsuan Yeh". | Cite NeurIPS 2026; use the name form of the cited version |
| `seligmann2026vlafail` | arXiv 2606.21386 v2 comment: "CoRL 2026"; the bib has `@misc`. | Cite CoRL 2026 |
| `robocasa365` | The authors are correct (Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, Yuke Zhu; arXiv 2603.04356, ICLR 2026). | Remove the CHECK comment; optionally add the arXiv id |

Verified as listed: `rajaprakash2026safecast`, `ma2026failurespot`, `yang2026triinfo`,
`zhu2026vlascope`, `huang2026actprobe`, `liu2026checkvla`, `ahmedt2026rescue`,
`xiaomi2026robotics` (team plus named authors on arXiv), `zhao2026flare` (CVPR 2026,
pp. 22391–22401), `soddy2026memoryvla` (NeurIPS 2026 PTA workshop poster, same five authors).
The pre-2026 entries are well-known papers with standard metadata. The `zhou2025codeasmonitor`
page numbers could not be checked (CVF returned 403).

**Missing citations.**
- **RLDX-1** is used throughout (L353, L439) and never cited. Add its paper or release.
- **SimplerEnv** (L128) is named without a citation.

**Claims about other papers** that their authors, who may be reviewers, will check. Re-verify
against the papers, not the abstracts:
- RW L15–17: SAFE "truncat[es] rollouts to equal length";
- RW L27–28: FIPER requires *both* scores to exceed their thresholds;
- RW L23: ActProbe "beating internal-feature baselines on RoboCasa";
- RW L33–35: CheckVLA's "conformal 5% rate of unnecessary interventions".

**Relevance.** Stage 1 judges citations on direct relevance. The five LLM/VLM-monitor citations
(RW L37) and Kaplan–Meier / Cox could shrink to save reference space if needed.

### Clarity, figures, format

- **Fig. 2 (L299–322)** shows the *opened* cohort with the *single* timer, a design the paper
  itself supersedes. It costs about 15 column lines. Cut it, or replace it with confirmatory
  P vs SAFE + P curves. It is the best source of space (see §5).
- **Fig. 1.**
  - It shows a hand-picked rollout with the exploratory single-timer threshold (τ_s = 0.648).
    The caption admits the selection, which is good.
  - The plotted success is "the longest of the 9 (out of 10) … that stay below the threshold",
    so the false alarm on the tenth is hidden. Show it, or say why not.
  - Use the confirmatory SAFE + P threshold if possible.
- **Table I** has nine rows from the opened cohort. "Time risk" alone, "SAFE only" and
  "Noise + timer" could go to the text (saves about 3 lines).
- **Two timers, three names.** "Timer", "global timer", "per-task timer" and "P" are used
  interchangeably across sections. Define "global timer G" and "per-task timer P" once and use
  only those.
- **L98.** "a small policy signal, not a useful one by our own standard" is good and honest.
  Keep it.
- **Template.** The preamble is reconstructed from memory (L3–6; README). The CFP page is
  reachable now; download the official template and diff it. Formatting changes and page
  numbers are explicitly disallowed.
- **Anonymity.** No issues found in the PDF text. Keep "Paper-ID" filled. The labeler description
  must not say "the author" (it currently does not).
- **The remaining notes** (2 TODO, 1 TODO on the FPR rule, 2 PENDING) must be resolved, not
  hidden by `\notesfalse` (concern 1).

---

## 5. Prioritised fix list with page-budget cost

The submission build has about 1–2 spare lines (drafts-session notes, tectonic preview).
Costs are in column lines (+ adds, − frees). Every addition needs a cut, so the list ends with
the cuts that pay for it.

| Priority | Fix | Concern | Cost |
|---|---|---|---|
| P0 | Rewrite H3 to match E2 (conditional, superiority, named head, FPR rule) and H4 to match E3 (named primary contrast). Remove the TODOs. | 1 | +3 |
| P0 | E2: add the null control (or matched-FPR gain) to the decision rule. Simulate and report the rule's type I error under noise + P and resampled SAFE + P (offline, same sizing code). Name the LSTM head and frozen checkpoint, and the H_k definition. | 1, 2 | +3 (plus one CPU run) |
| P0 | Settle the FPR rule. Either keep H1's rule, or bound the paired FPR difference to P. If the binomial rule stays, report its pass rates at true FPR 0.09 and 0.10. | 2 | +1 |
| P0 | E3: choose the policy; register the primary contrast (P-reset vs continue-to-1.25 H_k, or both with a multiplicity rule); define retry randomization; give the paired analysis and the n per arm for all arms. **Pilot r with about 100 re-attempts of failed scenes before 4 Dec.** | 1, 5 | +4 |
| P0 | Fix the Foresight author order and the Zollo venue. Update Hide-and-Seek and VLA-FAIL. Add RLDX-1 and SimplerEnv references, or drop SimplerEnv. | refs | 0 (references page) |
| P1 | Correct the registration timeline (N1); disclose the bootstrap-commit deviation; describe the matched-FPR check as a registered secondary, not an H1 criterion (N2, N3). Offer the AC an anonymized registration. | 8 | +1 (net; L376–378 and L477–478 are rewritten) |
| P1 | Remove or soften the overclaims in the concern 4 table: "real" gain, doubling-first abstract, L415, L429–432, the conclusion's "published benchmark", the SimplerEnv clause. | 4 | 0 to +1 |
| P1 | FIPER: add the conformal-rank mean (about 0.63–0.69), that FIPER wins on both real-robot tasks, that FPR is unmatched, and no intervals. Drop "the comparison we argue for". Promote "FIPER re-run" to a registered, paired, matched-FPR Stage 2 item. | 7 | +2 |
| P1 | E5: sample size, number of labelers, reliability gate on t_u, adjudication rule, primary endpoint. | 6 | +3 |
| P1 | State the "trivial by construction" objection and concede it in the intro. Make the contribution its consequences (concern 3). Replace L342–343's rhetorical question. | 3 | +2 |
| P2 | One feasibility line: compute per experiment and timeline to 16 Apr. | 9 | +1 |
| P2 | Prop. 1 → lemma; note it is marginal and that test FPR runs above α; per-task FPR. H2 as a rank rule. Explain the 0.41 noise AUC. | 10 | +1 |
| P2 | Fix N4–N12 wording (power at the assumed gain, "at least 650", seconds, mean vs max, noise p ≈ 0.14, "plurality", "no power claim"). | audit | 0 |
| P2 | Fig. 1: show or explain the hidden false alarm. Unify the timer names. | minor | 0 |
| **Cuts** | **Remove Fig. 2** (opened cohort, superseded single timer; keep "42% by 0.75 H_k" in the text) | | **−15** |
| | Drop three rows of Table I (time risk alone, SAFE only, noise + timer) | | −3 |
| | Collapse E4 to one exploratory sentence (FIPER re-run moves to its own item) | | −2 |
| | Shorten the time-risk paragraph in Method (L194–200) to two sentences | | −3 |
| | Trim the VLM/LLM monitor sentence and two citations in Related Work | | −1 |

**Net:** the additions total about +21 lines and the cuts free about −24, leaving about the
current 1–2 spare lines. Without the Fig. 2 cut the P0 items alone do not fit.

**Fastest path to an invitation to rebuttal:** all P0 items, plus the overclaim edits and N1–N3.
These are mostly writing, plus two cheap computations: the E2 type I error from the existing
sizing code, and the E3 retry pilot of about 100 rollouts. A reviewer who checks the protocol
against the hypotheses would otherwise stop reading at Section VIII.

# JSAI 2027 short paper

`main.tex` is a 4-page English general-session paper for JSAI 2027 (41st Annual Conference), derived from `../main.tex`, which is unchanged. It covers the Xiaomi-Robotics-1 results only. The confirmatory cohort is still marked PENDING in red.

**Build:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Template

The official English package (`jsaiac_tex_en-2026.zip` from https://conf.ai-gakkai.or.jp/jsai2026/en/author-guideline) could not be downloaded here, because the network blocks the JSAI site. The preamble reproduces the published rules instead: A4, two columns, a centred abstract of at most 150 words, and the contact address at the bottom of the first column. Before submitting, paste everything between `BEGIN BODY` and `END BODY` into the official template, and use the 2027 package once JSAI posts it. The submission is a PDF of at most 3 MB.

## Dual submission with RSS 2027 (needs a decision)

- **JSAI allows it.** JSAI's copyright rules (Art. 4(2), https://www.ai-gakkai.or.jp/pdf/national-convention/AC_copyright_jp.pdf) say JSAI will not object when authors later submit the work to another society. The author guidelines and CFP have no dual-submission clause.
- **RSS forbids it.** The RSS CFP says that submitting a paper that is currently under submission to another conference counts as a double submission and is "strictly forbidden". The only exemptions are venues "without published proceedings, or with only abstracts published", and preprints. JSAI publishes full 2–4 page PDFs, which go public on about 21 May.
- **The timing clashes.** The JSAI manuscript is due 17 Feb 2027 and the abstract 3 Feb. If RSS Stage 1 invites us on 5 Feb, the RSS paper stays under review until 30 Apr, so submitting this same content to JSAI on 17 Feb would break the RSS rule.
- **Options:**
  1. Register the JSAI abstract by 3 Feb. On 5 Feb, if RSS rejects, submit to JSAI on 17 Feb. If RSS invites, withdraw from JSAI or switch to a different topic. *(recommended)*
  2. Ask the RSS program chairs whether a non-peer-reviewed domestic paper in Japan counts as a dual submission.
  3. Narrow the JSAI paper to content that is not in the RSS submission.
- **IROS fallback (due 1 Mar 2027).** A JSAI paper submitted on 17 Feb would already be in the record when the work goes to IROS. Check the IEEE/IROS prior-publication rule before relying on both.

## Current version (2026-10-17, after the mock review)

- **Length:** 4 pages of body, with references starting on page 5 (`\clearpage` before the bibliography), as Félix asked. JSAI's limit is 2 pages plus at most 2 extra pages. If references count toward that limit, the body has to shrink to about 3.4 pages.
- **Framing:** every failure runs to the time limit, so a timer at c = 0.903 flags 380/380 failures at 0.047 FPR. The endpoint is the mean warning lead over the timer. SAFE + timer: 0.207 against 0.090 (gain 0.117, CI 0.060–0.178).
- **Numbers:** these match `../rss2027_stage1` (`analysis/lead_time.py` and `extra.py`, where an alarm at the last inference counts). The JSAI-only analyses are in `analysis/jsai_extras.py`: per task type, why the time risk misses, and where the lead comes from. Run it with `python3 -I analysis/jsai_extras.py ../review_data ../rss2027_stage1/analysis/lead_time.py`.
- **Kept out of JSAI** so that the two papers differ: the conformal proposition and the Stage-2 plan.
- **Bibliography:** taken from the RSS draft's `refs.bib`, with one fix. A `% CHECK` comment inside the `robocasa365` entry made BibTeX drop that entry. The RSS copy still has this bug.
- `analysis/timer_vs_detectors.py` is the earlier timer check. The paper no longer uses it.

**Check against main.tex:** results.json gives noise within-task ROC-AUC of 0.45 and 0.47 at 25% and 50% of the horizon. `../main.tex` says 0.52 and 0.53. This draft uses the results.json values.

## Update 2026-10-21: confirmatory test added

- **New Sec. 7 (Confirmatory Test) and Table 4.** Numbers are from `docs/safe_robocasa_experiment_report.md` §15.13. SAFE + P has lead 0.255 against 0.225 for P, a gain of +0.030 (decision CI [+0.007, +0.093]), at FPR 0.061. The gain is below the declared 0.05 margin. The noise and resampled controls gain at most +0.009.
- **The red PENDING note is gone.** The abstract (146 words), contributions, cohort table (new confirmatory-calibration row), protocol, discussion, limitations (i), (ii) and (vii), and conclusion now use the confirmatory result. The old Results section is now "Exploratory Results".
- **Cuts to stay within 4 body pages.**
  - The task-type table is folded into its paragraph.
  - The percentile-sensitivity paragraph is gone; the confirmatory section gives the q-sensitivity instead.
  - The sensitivity, time-risk, noise and conclusion paragraphs are shorter.
- **The PDFs are preview builds.** Both `main.pdf` files (here and in `../rss2027_stage1`) were built with tectonic (XeTeX) and TeX Gyre Termes, because this machine has no pdflatex. Rebuild with pdflatex for submission. In this preview, the body ends on page 4 and the references are on page 5.

## Update 2026-10-21 (b): §15.14 and §15.15 (exploratory)

- **Sec. 7, confirmatory paragraph.** The four success-only scorers fused with P gain 0.020–0.034. All five scores gain more on atomic tasks (0.045–0.081) than on composite tasks (0.008–0.027). The oracle stage-overdue result (+0.071, task CI −0.021 to 0.129) is linked to "P lacks progress on composites". All of this is labelled exploratory.
- **Discussion.** One sentence on the offline RLDX-1 check:
  - every failure runs to the limit;
  - P 0.271 against 0.032 for the global timer;
  - SAFE's **LSTM** head + P +0.034 [0.008, 0.067] at the same FPR, atomic only.
  - It also says the MLP head adds nothing, because this paper's SAFE is the MLP head.
- **Limitation (vi).** It now reads: one policy prospectively, and a second (RLDX-1) only offline. No Stage 2 material was added.
- **Cuts to stay within 4 body pages.**
  - Fig. 3 is 3.4 cm tall.
  - The captions of Tables 3 and 4 are shorter.
  - Limitation (vii) no longer repeats the q-sensitivity, which is still in Sec. 7.
  - Removed: the "233 failures, misses 17" sentence and the "high-success tasks such as opening a cabinet" sentence.
  - The conclusion and the sensitivity paragraph are tighter.
- **Page check (tectonic preview).** The body ends at the bottom of page 4, with no spare lines, and the references are on page 5.

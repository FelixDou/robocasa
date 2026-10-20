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

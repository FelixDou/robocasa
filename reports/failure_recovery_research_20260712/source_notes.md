# Source and validation notes

## Reporting job

- Question: What research direction should follow the initial natural-failure dataset, and should RoboCasa or the base VLA policy change?
- Audience: Technical robotics / VLA research audience.
- Decision: Select the benchmark, policy set, recovery architecture, data extensions, and first experiments.
- Snapshot time: 12 July 2026, JST.
- Decision-useful outcome: A staged experiment plan that can falsify the need for recovery demonstrations before committing to direct flow guidance.

## Required-structure mapping

- Technical summary: `technical_summary`.
- Key findings with visual evidence: `policy_strength_finding`, `policy_chart`, and `policy_chart_interpretation`.
- Scope, data, and metric definitions: `scope_definitions` and `experiment_design`.
- Methodology/model specification: `method_architecture` and `flow_guidance`.
- Limitations, uncertainty, and robustness: `limitations`.
- Recommended next steps: `next_steps`.
- Further questions: `further_questions`.

## Evidence boundaries

- Leaderboard values are copied from the official RoboCasa365 leaderboard as displayed on 12 July 2026; the page reports an update date of 9 July 2026.
- Paper method classifications are based on the six user-supplied PDFs and their primary public records.
- The user-supplied artifact URL resolves to an application shell rather than a portable report payload. Its content was therefore reconstructed from the current project report discussion and separately verified primary sources.
- No quantitative claims are made about the current failure dataset's progress distribution because no reviewed dataset-level aggregate was available in the supplied report content.
- Xiaomi-Robotics-1 is included only as closed upper-bound context.

## Chart map

| Report segment | Question | Family / type | Fields | Supported claim | Palette | Delivery |
|---|---|---|---|---|---|---|
| Policy selection | Which policy produces useful composite progress? | Comparison / grouped bar | policy, split, success_rate | RLDX-1 is a more useful higher-progress complement to pi0; pi0.5 is a small change | Blue, gold, olive plus neutrals | Native artifact chart in portable HTML; static SVG extracted for slides |

## Visual QA intent

- One full-width grouped bar chart because the evidence is a discrete category comparison, not a time series.
- The chart retains all three benchmark splits so composite performance is interpreted against atomic competence.
- The detailed policy and literature comparisons remain tables because exact method attributes are the point.

## Omitted analysis

- A chart of current failure depth was omitted because dataset-level progress counts were not available in the current report evidence.
- A causal claim that stronger policies necessarily produce better recovery data was avoided. The report proposes a matched pilot and useful-failure-yield metric instead.
- A second benchmark recommendation remains qualitative until compute and real-robot availability are known.

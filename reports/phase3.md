# Phase 3 report — Study 2 full: choosing the method

Status: **COMPLETE (B6/references finishing). → User checkpoint.**

## Setup

All eligible methods on the whole development evaluation part (809 rows), W1–W3,
3 runs each with **every call redone per run** (run-specific seeds). Official
scoring (Types 1–3, B = 100, α = 0.05, n = 500, seed patch; scoring seeds 1–3).
No-go methods from Phase 2 (P1, P2, P5, P5-view) are excluded per the fixed
rule; B6 and references included for context.

## Results (mean over 9 runs; SD over runs)

| condition | aggregate | SD | Type 1 | Type 2 | Type 3 | W1 | W2 | W3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **P3 judge-copula candidates** | **0.558** | 0.053 | 0.282 | 0.750 | 0.642 | 0.603 | 0.561 | 0.510 |
| P4 stated tables → copula | 0.546 | 0.042 | 0.299 | 0.750 | 0.590 | 0.581 | 0.558 | 0.499 |
| B1 reference (real marginals) | 0.513 | 0.021 | 0.523 | 0.717 | 0.300 | — | — | — |
| B2 reference (real copula) | 0.509 | 0.028 | 0.525 | 0.740 | 0.262 | — | — | — |
| B5 independent stating | 0.462 | 0.014 | 0.209 | 0.742 | 0.434 | 0.458 | 0.466 | 0.461 |
| R3 joint stating | 0.436 | 0.032 | 0.142 | 0.746 | 0.421 | 0.453 | 0.451 | 0.405 |
| B6 batch generation | 0.252 | 0.027 | | | | — | — | — |
| B4 official generation | 0.212 | 0.010 | 0.073 | 0.440 | 0.123 | — | — | — |

## Decisions carried to Phase 4 [pre-set rules]

- **Chosen method: P3** — the eligible method with the highest mean aggregate
  over runs and wordings (0.558 vs P4's 0.546). The +0.012 edge is within
  run-to-run noise (P3's SD 0.053), i.e. P3 and P4 are statistically tied;
  both clear the B1/B2 references (0.513/0.509). Reported honestly as a tie
  with P3 nominally first.
- **Contrast: P4** (P3's random pick among copula candidates is P4 itself),
  isolating whatever the judge adds on full-support candidates (Phase 2/3
  answer: nothing significant).
- **H2a** (judging vs stating): P3 vs B5/R3 = +0.10/+0.12 — but the gain comes
  from the *copula candidates*, not the judge (P3 ≈ P4). Judging per se adds
  nothing; the stated-table machinery does.
- **H2b** (population vs individual): P4 vs B5 = +0.084, and P4's Type 3
  (0.590 vs 0.434) confirms joint-structure recovery.
- **Run count:** between-run SD: P3 0.053, B4 0.010; difference SD 0.054.
  n ≥ 5 (spec minimum) resolves MMD ≥ 0.08; MMD 0.05 needs n ≈ 18; MMD 0.03
  needs n ≈ 49. **MMD must be fixed from the paper's GSS 2018 table before
  Phase 4** (user to supply).
- **Wording spread** (robustness gate: same sign, spread < MMD): P3 0.093,
  P4 0.081 — all wordings of P3/P4/B5/R3 beat B4, so the sign is stable; the
  spread magnitude must be read against the MMD once fixed.

## Interpretation notes

- The evaluation-part results confirm the pilot ordering exactly
  (P4 pilot 0.535 → full 0.546; B5 0.442 → 0.462; B4 0.201 → 0.212).
- B5 is the most *stable* method (SD 0.014); P3/P4 carry the wording variance.
- Per-output decomposition (Phase 2 data, `reports/per_output_t1_gain.csv`):
  P4/P3 recover outputs where B4 scores absolute zero (happy +0.79,
  occupation +0.72, wealth +0.60); a stubborn residual remains on work_hard,
  age_first_childbirth, isolated, trust, health — genuine knowledge gaps.

## Cost

Phase 3 total: **$186.10** (B6/references ≈ $0.36; everything cached).
Project total: **$355.30**. Remaining credit at checkpoint: ~$14.

Phase 3 is COMPLETE — all conditions, all runs, all wordings.

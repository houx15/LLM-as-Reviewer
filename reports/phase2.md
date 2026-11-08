# Phase 2 report — Study 2 pilot (whole profiles) + Study 3 (the judge)

Status: **finalizing** (P5-view + Study 3 results appended below; checkpoint at end)

## Setup

All conditions on the 500 development pilot rows, all 21 outputs, official
scoring (Types 1–3, B = 100, α = 0.05, n = 500, seed patch), 3 runs per
condition (scoring seeds 1–3). Judging rule = S4 (Phase 0 winner); candidates
and judge calls made once per wording and cached; selections redrawn per run.
Judge/test model `google/gemini-2.5-flash`; small generator `qwen/qwen3-32b`.
Aggregate = mean insignificant rate over Types 1–3 (higher = closer to real).

## Results (mean over 3 runs; wordings averaged where applicable)

| condition | cell | W1 | W2 | W3 | mean | Type 1 | Type 2 | Type 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B4 official generation | individual·sample | | | | **0.201** | 0.054 | 0.439 | 0.109 |
| B6 batch generation | population·sample | | | | 0.238 | 0.250 | 0.285 | 0.178 |
| B5 independent stating | individual·state | 0.455 | 0.444 | 0.427 | 0.442 | 0.226 | 0.696 | 0.405 |
| R3 joint stating | individual·state | 0.456 | 0.433 | 0.364 | 0.418 | 0.149 | 0.706 | 0.398 |
| **P4 stated tables → copula** | population·state | **0.627** | 0.545 | 0.431 | **0.535** | 0.335 | 0.724 | 0.545 |
| **P3 judge copula candidates** | individual·judge | 0.619 | 0.557 | 0.471 | **0.549** | 0.313 | 0.725 | 0.609 |
| P1 judge own candidates | individual·judge | 0.185 | 0.180 | 0.172 | 0.179 | 0.068 | 0.405 | 0.064 |
| P1-random contrast | | 0.174 | 0.202 | 0.175 | 0.184 | | | |
| P2 judge small-model candidates | individual·judge | 0.286 | 0.291 | 0.284 | 0.287 | 0.080 | 0.553 | 0.227 |
| P2-random contrast | | 0.292 | 0.282 | 0.278 | 0.284 | | | |
| P5 population builder λ=0 (=P1) | population·judge | | | | 0.178 | 0.093 | 0.367 | 0.026 |
| P5 λ=1 | | | | | 0.153 | | | |
| P5 λ=5 | | | | | 0.154 | | | |
| P5-view | population·judge | (pending) | | | | | | |
| B1 reference (real marginals) | | | | | 0.481 | 0.516 | 0.674 | 0.255 |
| **B2 reference (real copula)** | | | | | **0.508** | 0.553 | 0.697 | 0.272 |

## Go / no-go per judging method [fixed rule: mean(method − contrast) > 0 and > 2× contrast range]

| method | contrast | diff (W1/W2/W3) | verdict |
| --- | --- | --- | --- |
| P1 | P1-random | +.011 / −.021 / −.003 | **NO-GO** |
| P2 | P2-random | −.006 / +.008 / +.006 | **NO-GO** |
| P3 | P4 | −.008 / +.012 / +.039 | **NO-GO** |
| P5 λ=1 | P5 λ=0 | −.026 | **NO-GO** |
| P5 λ=5 | P5 λ=0 | −.024 | **NO-GO** |

**No judging method passes.** The judge's probabilities carry no signal over a
uniform pick among the same candidates, at any λ; the population term *hurts*
(picks changed vs λ=0: 90 at λ=1, 220 at λ=5 — the more the builder follows the
stated tables, the worse it gets, because it builds on uninformative j(c) and
over-fits small counts).

## What did pass

- **H0 at full scale: B5 (0.442) ≫ B4 (0.201)**; Type 1 0.226 vs 0.054. Stating
  beats sampling across all 21 outputs.
- **H2b at full scale: P4 (0.535) > B5 (0.442) and R3 (0.418)** — population-level
  stating wins, exactly as the Study 1 miniature suggested.
- **The strongest result class: P3 (0.549) and P4 (0.535) exceed B2 (0.508)** —
  a readout with no human data beats a Gaussian copula fitted on real
  respondents. P4/W1 alone reaches 0.627.
- Generator switch: **P2-random (0.284) > B4 (0.201)**, +0.083 from candidates
  alone (the more diverse qwen3 pool), no judging needed.
- H2a (judging vs stating): **fails** — everywhere. The paper's story becomes:
  *the knowledge is read out by asking for distributions, not by judging
  candidates*.
- Wording spread is large for P4 (0.627 → 0.431 W1→W3) and smaller for P3
  (0.619 → 0.471); wording means are the pre-registered estimand.
- Acceptance check: P1-random (0.184) within 2× B4's run range ✓; every
  condition has one valid profile per row (complete-rate 1.0 after official
  retries) ✓.

## Study 3 — why judging fails (gate: discrimination AUROC ≥ 0.578, calibration slope 0.5–1.5, typicality > 50%)

(pending — appended when the run lands)

## Cost

(pending final numbers; conditions so far ≈ $55)

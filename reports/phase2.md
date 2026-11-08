# Phase 2 report — Study 2 pilot (whole profiles) + Study 3 (the judge)

Status: **COMPLETE. → User checkpoint.**

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
| P5-view | population·judge | 0.189/0.222/0.188 | | | **0.200** | | | |
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

## Study 3 — why judging fails

Gate [fixed before running]: discrimination AUROC ≥ 0.578 (= 0.5 + half the GBM
reference's margin), calibration slope 0.5–1.5, typicality accuracy > 50%.

| test | result | gate | verdict |
| --- | --- | --- | --- |
| Discrimination (real vs one-output-swapped) | AUROC **0.587** vs GBM 0.589 (n=151) | ≥ 0.544 | **PASS** |
| Calibration (vs BN reference) | slope **0.312**, R² 0.20 (n=236) | 0.5–1.5 | **FAIL** (too flat) |
| Typicality trap | accuracy **0.410** (n=200; 204 trap groups) | > 0.5 | **FAIL** (prefers the stereotype) |
| Order stability | flips **15.2%** (n=99) | reported; >20% flagged | ok |
| Wording stability | flips **22.2%** | reported | flagged |
| Coherence | mean Bayes deviation **0.096** (50 pairs) | reported | moderate |
| Population discrimination | accuracy **100%** (n=18: B4, B5, P4) | reported | see caveat |

**Judge verdict: FAILS the gate** (1 of 3 required passes) — and the failure
profile explains the go/no-go results exactly:

- The judge **can discriminate** — it detects a swapped answer as well as a
  gradient-boosting classifier trained on real data (0.587 vs 0.589). The
  knowledge is there.
- But its **probabilities are nearly flat** (calibration slope 0.31): when
  asked "how likely is this value", it spreads mass too evenly — so S4-style
  selection weights barely differ from a uniform pick. Hence P1 ≈ P1-random,
  P2 ≈ P2-random, P3 ≈ P4.
- And it **shares generation's typicality bias** (41% accuracy on trap pairs):
  where the model's generation favours an answer that is actually uncommon,
  the judge favours it too. Judging cannot fix what generation gets wrong,
  because the judge wants the same wrong answer.
- Per-output discrimination (H3 material): child_number 0.81, marital_status
  0.65, occupation 0.62, age_first_childbirth 0.59, political_view 0.43 —
  the judge is worst exactly where its stereotypes are strongest.

Caveat on population discrimination (100%): the rendered summaries include
missing-data shares, which real data has (split-ballot NA up to 52%) and our
synthetic data never does — the judge can win by spotting the NA columns.
Treat as an artifact-detection upper bound; P5-view's failure (0.200) shows
the judge cannot *use* table-level knowledge to build a dataset.

## P5-view

0.189 / 0.222 / 0.188 (mean **0.200**) — seeing the population state in the
prompt does not help; consistent with the calibration failure.

## Cost (whole project so far)

| model | API calls | cost |
| --- | --- | --- |
| google/gemini-2.5-flash | 300,362 | $122.49 |
| qwen/qwen3-32b | 15,617 | $3.52 |
| **total** | 315,979 | **$126.01** |

(of which Phase 2 + Study 3 ≈ $105; everything is cached and reruns are free)

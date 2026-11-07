# Phase 0 report — Study 1: one question, which readout extracts the most

**Status: COMPLETE. → User checkpoint.**

## Setup

- Test model (always the judge): `google/gemini-2.5-flash` (paper-evaluated, still served).
- Small generator: `qwen/qwen3-32b` (Table S1 below).
- Questions: `political_view` (ordinal, 3 answers; known to collapse), `mental_health`
  (numeric 0–30, 31 answers; zero-inflated, 58% zeros in real data).
- Rows: 500 development pilot rows (from the 60% evaluation part, seed 2026; split frozen).
- Wordings W1–W3 per `docs/PROMPT-WORDINGS.md`; labels shuffled per call with logged
  seeds; judge calls in two label orders, averaged; up to 3 parse attempts, then uniform
  fallback (flagged).
- 3 runs (scoring seeds 1–3); candidates/judge calls/tables made once per wording and
  cached — generation variability understated (SPEC §3.4).
- Scoring: official Type 1 (question) + Type 2 (question × 8 inputs), restricted configs,
  master settings (B=100, α=0.05, n=500 per draw), seed patch.
- Aggregate = mean of Type 1 and Type 2 insignificant rates (higher = closer to real).

## Generator choice (Table S1)

50 pilot rows × 10 full profiles per model, official prompt and retry loop.

| model | valid rate | duplicate share | mean per-output entropy | mean pairwise Hamming |
| --- | --- | --- | --- | --- |
| **qwen/qwen3-32b** | 1.000 | 0.008 | **0.636** | 0.296 |
| google/gemini-2.5-flash | 0.9997 | 0.112 | 0.512 | 0.242 |

**Choice [default rule]: qwen/qwen3-32b.** Side observation: the test model duplicates
11.2% of profiles within 10 draws at temperature 1.0 — compression at generation.

## Main results

Mean over 3 runs × 3 wordings (range over the 9 runs in parentheses where informative).
Aggregate / Type 1 / Type 2 insignificant rates; spread ratio (model/real, 1 = real);
TV distance to the real marginal; signed association difference (model − real, mean over
8 inputs).

### political_view

| rule | agg (test gen) | agg (qwen3 gen) | T1 test / qwen3 | T2 test / qwen3 | spread | TV | assoc diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 sample (=generation) | 0.308 | 0.197 | .000 / .034 | .617 / .359 | 0.89–0.90 | 0.14–0.17 | +0.07 / +0.12 |
| S1 judge | 0.287 | 0.200 | .000 / .022 | .574 / .377 | 0.91 | 0.15 | +0.08 / +0.12 |
| S2 greedy | 0.267 | 0.403 | .013 / .428 | .521 / .378 | 0.92 | 0.09–0.15 | +0.10 / +0.11 |
| S3 filter | 0.294 | 0.192 | .002 / .021 | .587 / .362 | 0.91 | 0.15 | +0.08 / +0.12 |
| S4 freq-corrected | 0.337 | 0.306 | .000 / .131 | .674 / .481 | 0.92–0.93 | 0.11–0.15 | +0.05 / +0.09 |
| S5 pairwise | 0.310 | 0.199 | .000 / .027 | .620 / .372 | 0.90–0.91 | 0.15 | +0.07 / +0.12 |
| **S6 state, individual** | **0.785** | **0.792** | **.711 / .719** | .859 / .866 | **1.007** | **0.041** | −0.027 |
| **S7 state, population** | **0.857** | **0.905** | **.840 / .909** | .874 / .900 | **1.003** | **0.023–0.028** | +0.01 / .000 |

### mental_health

| rule | agg (test gen) | agg (qwen3 gen) | T1 test / qwen3 | T2 test / qwen3 | spread | TV | assoc diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.425 | 0.129 | .000 / .000 | .850 / .258 | 0.41 / 0.26 | 0.33 / 0.83 | ~0 |
| S1 | 0.451 | 0.142 | .000 | .903 / .284 | 0.25–0.26 | 0.37 / 0.82 | ~0 |
| S2 | 0.457 | 0.172 | .000 | .914 / .345 | 0.08 | 0.40 / 0.79 | ~0 |
| S3 | 0.410 | 0.107 | .000 | .821 / .214 | 0.43 / 0.26 | 0.32 / 0.82 | ~0 |
| S4 | 0.412 | 0.159 | .000 | .824 / .319 | 0.41 / 0.31 | 0.32 / 0.79 | ~0 |
| S5 | 0.407 | 0.119 | .000 | .813 / .238 | 0.47 / 0.26 | 0.32 / 0.82 | ~0 |
| **S6** | **0.468** | **0.471** | .000 | **.936 / .942** | 1.39 | 0.39 | ~0 |
| **S7** | **0.473** | 0.431 | **.214 / .084** | .731 / .778 | 1.43 | **0.27–0.28** | ~0 |

## The four headline comparisons

1. **H0 in miniature (S6 vs S0): HOLDS on both questions.**
   - political_view: 0.785/0.792 vs 0.308/0.197 — stating beats sampling decisively;
     within-group spread recovers to 1.007 (generation: 0.89); TV distance 0.04 vs 0.14–0.17.
   - mental_health: 0.468/0.471 vs 0.425/0.129 — gain driven by Type 2 (0.94 vs 0.26–0.85);
     Type 1 stays 0.000 for S6 (47.6% zeros stated vs 58.2% real — zero-inflation
     underestimated). Spread moves 0.41 → 1.39, i.e. from under- to over-dispersed
     (|1.39−1| = 0.39 < |0.41−1| = 0.59: closer in absolute distance, but overshooting).
   - **The model knows far more about these distributions than its sampling shows.**

2. **H2a in miniature (judging vs stating): FAILS at the value level.**
   All judging rules (S1–S5) cluster near S0 and far below S6. Mechanism is visible in
   the logs: the candidate support itself is collapsed — among 10 candidates, only
   **2.02 of 3** distinct values appear (political_view, test gen) and **3.15 of 31**
   (mental_health). A judge can reweight the offered values but cannot restore mass to
   values the generator never proposes. The judge is not uninformative (J entropy 0.65
   political / 0.85–0.92 mental); the pool starves it.
   - By the pre-set ranking, **S4 (frequency-corrected) is the judging rule** for both
     pools (test 0.375 vs S0 0.367; qwen3 0.233 vs 0.163) — a small edge, within
     run-range overlap on the test side.
   - This elevates **P3** (candidates drawn from P4's stated copula — full support) from
     "one more variant" to the design's direct answer to the support problem.

3. **H2b in miniature (S7 vs S6): population stating wins.**
   S7 > S6 on both questions (political 0.857–0.905 vs 0.785–0.792; mental 0.473 vs
   0.468). S7 is the **only** rule with any Type 1 credit on mental_health (0.084–0.214):
   stated two-way tables preserve zero-inflation better than person-level statements.
   Political_view marginals are near-perfect (T1 0.84–0.91).

4. **Collapse check (S2 vs S0): did not lose as predicted.**
   S2 ties or slightly beats S0 (0.267 vs 0.308 test; 0.403 vs 0.197 qwen3) — argmax over
   a row-dependent candidate set still varies across rows. The predicted collapse is
   real in spirit (spread 0.92, TV 0.08–0.15) but does not show up as a score drop
   against an already-collapsed S0.

## Other observations

- **H4 preview:** generation inflates input→attitude associations (+0.07 test, +0.12
  qwen3 on political_view); S6/S7 bring them to −0.03…+0.01. In this pilot, *stating*
  does not inflate stereotypes — *generation does*.
- **Wording effects are real:** S6 political_view W2 (frequency framing) = 0.593 vs
  W1/W3 = 0.871/0.891. S7 is stable across wordings (0.83–0.90). Wording spread must be
  reported beside means (as pre-registered).
- **S7 instability on mental_health:** run-to-run range up to 0.53 (aggregate), driven
  by borderline KS p-values near α and 5/32 table-row uniform fallbacks (31-key JSON is
  the hard parse case). Log-linear fit itself is tight (mean absolute deviation 0.009).
- **Judge diagnostics:** J entropy 0.65 (political) / 0.85–0.92 (mental); parse fallback
  after 3 attempts < 0.1% on judge calls; S6 renormalization common but within tolerance.
- **Missing rate:** 0.000 for every rule (valid-output rate 0.998 on candidates).

## Decisions carried to Phase 2 (per pre-set rules)

- **Judging rule: S4** (frequency-corrected) for P1, P3, P5 (test pool) and P2 (qwen3 pool).
- **Stating rule noted: S7-style population tables** — favors P4 over B5/R3 a priori.
- P3 and P4 gain priority given the support mechanism above.

## Deviations and defaults logged

- Missing inputs form their own group ("Missing") in S7 tables / log-linear design.
- Parser tolerates percentage-style answers in probability wordings; W2 totals 90–110.
- No existing B4 replication run exists (agent_* folders are extended-schema 500-row
  experiments); B4 will be generated fresh; reproduction check against the paper's
  published scores (user to supply before Phase 1).
- The first Study 1 execution crashed on a diagnostics type bug (str/float keys) after
  political_view completed; fixed, and everything re-run from cache deterministically.
  Diagnostics were recomputed from stored run files after a second key-format fix
  (TV distance for numeric questions); scores were never affected.
- Client concurrency 16; cache by request hash (reruns free).

## Calls and cost

| model | API calls | tokens in | tokens out | cost |
| --- | --- | --- | --- | --- |
| google/gemini-2.5-flash | 76,696 | 27.6M | 3.4M | $16.70 |
| qwen/qwen3-32b | 10,513 | 3.0M | 5.1M | $1.67 |
| **total** | 87,209 | | | **$18.37** |

(S5 pairwise and judge calls dominate the gemini count, as forecast.)

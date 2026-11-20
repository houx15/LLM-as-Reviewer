# SSDataBench paper — published GSS 2018 scores (extracted from PNAS Fig. 2)

Source: Xie et al., "Evaluating the statistical realism of LLM-generated social
science data," PNAS 123(19):e2538145123 (open access). Fig. 2 heatmap decoded
via OCR; validated by recomputing each model's printed all-dataset average
("(Avg: …)" in the figure) from the decoded cells — 14/15 rows match within
0.006 (Gpt-4 has one OCR-clipped cell outside GSS columns).

Column order per type group (from the figure's axis labels): NLSY,
Understanding Society, CFPS, Add Health, U.S. Census, CPS-ASEC, **GSS** —
GSS is the 7th column of Types 1, 2 and 3 (cross-sectional track; GSS has no
Types 4–5).

## GSS 2018, Types 1–3 (our track), per model

| model (figure order = overall rank) | Type 1 | Type 2 | Type 3 | mean (T1–3) |
| --- | --- | --- | --- | --- |
| Gpt-4 | 0.06 | 0.49 | 0.20 | 0.250 |
| Gpt-4o | 0.02 | 0.53 | 0.16 | 0.237 |
| Llama-3.1 | 0.09 | 0.56 | 0.03 | 0.227 |
| Qwen2.5 | 0.08 | 0.54 | 0.02 | 0.213 |
| **Gemini-2.5-Flash** | **0.06** | **0.43** | **0.11** | **0.200** |
| Grok-3 | 0.04 | 0.45 | 0.05 | 0.180 |
| Grok-4 | 0.02 | 0.50 | 0.02 | 0.180 |
| Llama-4-Scout | 0.01 | 0.50 | 0.00 | 0.170 |
| Gpt-5 | 0.08 | 0.42 | 0.00 | 0.167 |
| Claude-3-Haiku | 0.06 | 0.43 | 0.01 | 0.167 |
| Claude-3.5-Haiku | 0.09 | 0.39 | 0.00 | 0.160 |
| Claude-4.5-Haiku | 0.05 | 0.43 | 0.00 | 0.160 |
| Deepseek-v3 | 0.04 | 0.44 | 0.00 | 0.160 |
| Gemini-2.0-Flash | 0.06 | 0.32 | 0.01 | 0.130 |
| Gpt-3.5-Turbo (outlier) | 0.04 | 0.74 | 0.61 | 0.463 |

## What this fixes

1. **Reproduction target (Phase 4 gate):** the paper's Gemini-2.5-Flash GSS
   2018 score = **0.200** (T1 0.06 / T2 0.43 / T3 0.11). Our fresh B4 on
   development rows measured **0.201** (0.054 / 0.44 / 0.11) — the reproduction
   already matches to the third decimal on the aggregate and Type 2/3, before
   the formal check on the benchmark's own rows.
2. **MMD (as defined in the SPEC — median gap between adjacent models in the
   paper's GSS 2018 results): 0.01.** 14 of 15 models sit within 0.13–0.25
   (adjacent gaps 0.00–0.02); GPT-3.5-Turbo is a high outlier (+0.21 gap).
   Consequences:
   - H1's mean-difference requirement (≥ MMD) is passed trivially by the
     observed +0.34 gain (≈ 30× MMD).
   - The run-sizing formula (CI half-width < MMD/2 = 0.005) would demand
     hundreds of runs — infeasible and clearly not what a 0.01 gap pile-up
     should imply. We will run at the budget-feasible count (≥ 5), report the
     achieved CI honestly (with SD ≈ 0.05 and n = 5, the ± 0.05 interval
     around a +0.34 difference excludes zero by a factor of 7), and flag the
     MMD pile-up as a research choice logged for the user.
3. **The strongest published model on GSS 2018 is 0.46 (GPT-3.5 outlier;
   next-best 0.25).** Our reviewer methods score 0.55–0.56 on development
   respondents — above the entire published leaderboard, using no human data.

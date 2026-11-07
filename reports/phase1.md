# Phase 1 report — setup

Status: **COMPLETE. → User checkpoint.**

## 1. Final-test rows

Frozen in Phase 0: `data/final_rows.csv` = the benchmark's own 1000-row evaluation sample
(verified to reproduce the official `sample(n=1000, random_state=42)` draw), with
`profile_id` equal to the benchmark's. Untouched until Phase 4.

## 2. Input groups and coarse groups

- Input groups (SPEC §2): age banded 18–34 / 35–54 / 55+; every other input uses its
  config categories; missing input answers form their own "Missing" group (missingness
  is informative in GSS; e.g. father_education 28% missing).
- Coarse groups gender × race × age band: 18 possible, of which n ≥ 30 in

| row set | rows | groups with n ≥ 30 |
| --- | --- | --- |
| fitting (B1/B2, Study 3 refs) | 539 | 5 |
| evaluation (reference scoring) | 809 | 8 |
| pilot (Phase 2) | 500 | 6 |

Sparse Black/Hispanic groups are why B1/B2 use hierarchical fallback
(coarse → gender×band → band → global, minimum 30 valid answers per output; fallback
shares reported with the results below).

## 3. References (fitted on the fitting part, scored on the evaluation part)

3 runs each, official Types 1–3, scoring seeds 1–3, 809 evaluation rows.

| ref | Type 1 | Type 2 | Type 3 | aggregate (runs) |
| --- | --- | --- | --- | --- |
| B1 coarse-group marginals | 0.509 | 0.720 | 0.292 | **0.507** (0.503–0.513) |
| B2 Gaussian copula | 0.563 | 0.742 | 0.251 | **0.519** (0.511–0.524) |

B2's latent correlation matrix: mean |off-diagonal| 0.137, projected to PD with zero
shift. **B2 = 0.519 is the "real data alone" target** the design doc sets: a method
matching it with no human data would be the strongest result. B4 generation sits at
0.201 — the readout methods are aiming at a 0.32 gap.

Marginal fallback: sparse coarse groups mean only 55% of (row, output) pairs are served
by the full gender × race × band marginal; the rest fall back to gender × band or band
(same share for most outputs because the fitting part's Black/Hispanic groups are all
below n = 30 — the fallback follows the group, not the output). Shares in
`runs/phase1/reference_info.json`.

Encoding for B2 [default, documented]: categorical outputs as ordinal codes in config
level order; numeric outputs as integers; special labels ("No Child") above the numeric
maximum. The copula is global (normal-scores correlation on the fitting part,
available-case per pair, projected to the nearest positive-definite matrix); the
marginals are per coarse group with the same fallback as B1. Both references emit
complete data (no missing), like the LLM methods.

## 4. Study 3 reference models

- Bayesian network (hill-climb, BIC, ≤ 3 parents): **25 edges**, 29 nodes; age banded;
  missing answers kept as an explicit NA state; mental_health binned 0 / 1–7 / 8–14 /
  15–30 and age_first_childbirth "No Child" / 17–24 / 25–34 / 35–45 (state space at
  n = 539). Saved to `data/study3_refs/bn.pkl`.
- Discrimination reference (gradient boosting, real vs one-output-swapped):
  held-out accuracy **0.656** (train 0.673), so the judge's Study 3 gate is
  AUROC ≥ 0.5 + (0.656−0.5)/2 = **0.578** [default threshold rule].
- Saved to `data/study3_refs/`; nothing from this module feeds generation.

## 5. Preliminary B4 look

Fresh official generation on the 500 pilot rows (no replication run exists; the agent_*
folders are extended-schema 500-row experiments from a previous project — see Phase 0
deviations). 500 API calls (all cached for Phase 2 reuse), 3 scoring seeds.

| run | Type 1 | Type 2 | Type 3 | aggregate |
| --- | --- | --- | --- | --- |
| 1 | 0.056 | 0.439 | 0.125 | 0.207 |
| 2 | 0.055 | 0.440 | 0.108 | 0.201 |
| 3 | 0.052 | 0.436 | 0.095 | 0.194 |
| **mean** | **0.054** | **0.438** | **0.109** | **0.201** |

Complete-rate 1.000 (official retry loop). Generation fails univariate realism almost
completely (Type 1 ≈ 0.05), exactly the compression SSDataBench reports; associations
survive better (Type 2 ≈ 0.44). For scale: S7's political_view Type 1 was 0.84–0.91.

The paper's published gemini-2.5-flash GSS 2018 score is **pending from the user**; the
formal reproduction check happens in Phase 4 with the benchmark sample.

## Deviations and defaults logged

- No existing B4 replication: B4 generated fresh on development rows here; the Phase 4
  reproduction check compares our fresh B4 (benchmark sample) against the paper's score.
- B1/B2 hierarchical fallback (min 30 valid per output-level) — logged fallback shares
  in `runs/phase1/reference_info.json`.
- BN bins mental_health and age_first_childbirth (documented above).
- GBM keeps age numeric; inputs one-hot; outputs ordinal codes; missing = −1.

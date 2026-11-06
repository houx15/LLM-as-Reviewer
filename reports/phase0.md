# Phase 0 report — Study 1: one question, which readout extracts the most

Status: **in progress** (filled as results land; checkpoint at the end)

## Setup

- Test model (always the judge): `google/gemini-2.5-flash` (paper-evaluated, still served).
- Small generator: chosen in §Generator choice below.
- Questions: `political_view` (ordinal, 3 answers), `mental_health` (numeric, 0–30, 31 answers).
- Rows: 500 development pilot rows (drawn from the 60% evaluation part, seed 2026).
- Wordings W1–W3 per `docs/PROMPT-WORDINGS.md`; labels shuffled per call with logged seeds;
  judge calls run in two label orders, averaged.
- Runs: 3 (seeds 1–3); each redraws every rule's selection with per-cell RNG and a new
  scoring seed. Candidates, judge calls and stated tables are made once per wording and
  cached — this understates generation variability (noted in SPEC §3.4).
- Scoring: official Type 1 (this question) + Type 2 (this question × 8 inputs), restricted
  configs, master settings (B = 100, α = 0.05, 500 per draw), seed patch.

## Generator choice (Table S1, SPEC §4.1)

50 pilot rows × 10 full profiles per model, official prompt, official retry loop.

| model | valid rate | duplicate share | mean per-output entropy | mean pairwise Hamming |
| --- | --- | --- | --- | --- |
| (pending) | | | | |

Choice rule [default]: the most diverse model by mean entropy among those with ≥ 90%
valid outputs. Both candidates are run through Study 1 in any case (test model as
generator matters for S0's fidelity to B4; small generator feeds P2 later).

## Judge diagnostics

(pending: entropy of J probabilities, parse-failure / fallback rates, tokens)

## Results per question × generator × wording × rule

(pending: aggregate / Type 1 / Type 2 / spread ratio / TV distance / signed association
difference; mean and range over runs)

### Headline comparisons

- **H0 in miniature (S6 vs S0)**: pending
- **H2a in miniature (S1, S4 vs S6)**: pending
- **H2b in miniature (S7 vs S6)**: pending
- **Collapse check (S2 vs S0)**: pending

## Decisions carried to Phase 2

- Judging rule (test-model generator): pending
- Judging rule (small generator): pending
- Stating rule noted (S6 vs S7): pending

## Deviations and defaults logged

- Missing inputs form their own group ("Missing") in S7 tables and the log-linear design;
  missingness is informative in GSS (e.g. father_education 28% missing).
- Parser tolerates percentage-style answers (e.g. 60/40) in probability wordings.
- No existing B4 replication run was found (the agent_* folders are extended-schema
  500-row experiments, not the benchmark replication); B4 will be generated fresh and
  the reproduction check will use the paper's published scores (user to supply).
- Client concurrency 16; cache by request hash makes reruns free.

## Calls and cost

(pending)

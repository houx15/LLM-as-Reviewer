# Implementation spec: LLM as reviewer on SSDataBench (GSS 2018) — v2

**For:** a coding agent. **Source of truth for the research design:** the doc "LLM reviewer design v2" (English tab). This spec turns it into an implementation plan. Where they disagree, the design doc wins; ask the user. This spec supersedes the v1 spec and `SPEC-quick-pilot.md`.

**The question.** On SSDataBench's GSS 2018 track, with its inputs, outputs, sample size and scoring unchanged, does the same model produce more realistic synthetic data when asked to *state distributions* or *judge candidates* than when asked to *generate* answers, using no human data?

**The framework.** Every method asks the same model about the same people. Methods differ on two axes: the **readout form** (sample / state a distribution / judge candidates) and the **level** (one individual / the population). Judging methods differ further by **candidate source** (the model's own generation / a small open model / the model's own stated distribution).

| | Sample | State | Judge |
| --- | --- | --- | --- |
| **Individual** | B4 official generation | B5 (independent), R3 (joint, along a graph) | P1 (own candidates), P2 (small-model candidates), P3 (candidates from P4) |
| **Population** | B6 batch generation | P4 stated two-way tables → copula reconstruction | P5 population builder (λ links individual judging to P4's tables) |

**How the work proceeds.** In phases, each ending at a checkpoint where you stop, report, and wait for the user. The cheapest test of the riskiest assumption comes first.

| Phase | Design-doc study | What | Ends with |
| --- | --- | --- | --- |
| 0 | Study 1 | Infrastructure; development split; generator choice; one question × eight readout rules S0–S7 | H0 verdict in miniature, winning judging rule → **user checkpoint** |
| 1 | — | References (B1, B2), Study 3 reference models, preliminary B4 look | Setup report → **user checkpoint** |
| 2 | Study 2 pilot + Study 3 | All methods on 500 development rows; the judge measured directly | Go / no-go per judging method, judge pass / fail → **user checkpoint** |
| 3 | Study 2 full | All eligible methods on the whole development evaluation part | Chosen method, run count, MMD → **user checkpoint** |
| 4 | Study 4 | Final test on the benchmark sample; second model; wordings | H1–H4 verdicts → **user checkpoint** |
| 5 | Study 4 (last item) | Second SSDataBench dataset | Generalization results |

---

## 1. Ground rules (all phases)

1. **No human data in generation.** Every method under test sees only what SSDataBench's own generation sees: the official prompt framing with one real input row (9 fields, including birth year). Real *output* values are never shown to a model in any generation prompt, and nothing fitted on real outputs feeds a method under test. The exceptions are explicit and labelled: the references B1 and B2, and Study 3, where real profiles are *test items* for the judge.
2. **Official prompt and scoring.** Build every prompt on the official `build_prompt` framing (population context, reference year, field descriptions, allowed values). Score only with SSDataBench's evaluation code, with the seed patch (§3.3) as the sole change.
3. **Temperature 1.0 for every call.** No tuning.
4. **Stated probabilities.** Whenever a method needs a probability or a distribution, the model states it in JSON. Token logprobs are not used. API only; no local model is required.
5. **Pre-registered wordings and orders.** Every stating or judging prompt exists in 2–3 wordings (W1 default, W2, W3), fixed in `prompts.py` before Phase 0 runs and reviewed by the user. Answer options and candidate labels are shuffled per call with a logged seed and parsed by label. A method's score is defined as the **mean over wordings**; the spread across wordings is reported beside it. Which phases run which wordings is stated per phase.
6. **Everything is logged and cached:** prompt, raw response, parsed result, model id and version string, timestamp, seed, condition, wording, row, call purpose. Log as JSONL and cache responses by request hash, so reruns reproduce and cost nothing.
7. **The benchmark sample is untouched until Phase 4.** Phases 0–3 use only development respondents. References are fitted on the development fitting part and never scored on it.
8. **Research choices are not yours to make silently.** Every number and choice in this spec that does not appear in the design doc is a default, whether or not it is marked **[default]**. Changing one, or any prompt wording, gets logged and reported at the next checkpoint.
9. **Don't touch the user's existing replication project.** Read its outputs; work in a new repository (confirm with the user).

## 2. Inputs

- SSDataBench repo: https://github.com/lszshu/SSDataBench. Read first:
  - `simulation/generation_cs.py`: prompt, system message, JSON mode, validation, retries, input sampling with `random_state=42`
  - `simulation/configs/content/gss_2018.yaml`: fields and allowed values
  - `simulation/configs/param/*.yaml`: model ids
  - `evaluation/run_all_types.py`, `evaluation/code_by_type/type1.py`, `type2.py`, `type3.py`, `evaluation/config/gss_2018/*.yaml`: scoring
  - `scripts/evaluation/gss_2018.py`: scoring a run folder with `--single`
- SSDataBench's GSS 2018 data file in its format (`real_data/dataset/gss2018.csv`, supplied by the user). Respondents not drawn into the evaluation sample are the **development respondents**.
- The user's existing replication run: `sampled_inputs_<model>_temp1.0_top1.0.csv` (real sampled rows with `profile_id`) and `sim_profiles_<model>_temp1.0_top1.0.csv`. **Ask the user for its path and the model version.**
- API access (OpenRouter or provider) to the test model, to 2–3 small open models for the generator choice, and, from Phase 4, to a second model the paper evaluated.

**Fields.**
- **Inputs (9 in the prompt; the first 8 are scored):** age, gender, race, immigrant_status, mother_education, father_education, mother_occupation, father_occupation, plus birth_year in the prompt only.
- **Outputs (21):**
  - Marriage: marital_status, child_number, age_first_childbirth
  - SES: education, laborforce, occupation, spouse_occupation, income, wealth
  - Health: health, depress, isolated, lonely, mental_health
  - Attitude: gender_role_attitude, happy, political_view, satisfy_job, trust
  - Ability: vocabulary_test, work_hard
- **Numeric outputs** (child_number 0–8, age_first_childbirth 17–45 plus "No Child", mental_health 0–30, vocabulary_test 0–10): whenever a method asks for a distribution, it asks for a probability on **every allowed integer**. No bins.
- **Input categories for tables.** Age is banded (18–34, 35–54, 55+) **[default]**; every other input uses its own categories from the config. These categories define the "input groups" used by S7, P4, P5, B6 and the diagnostics.

## 3. Shared infrastructure (built in Phase 0, extended later)

### 3.1 Repository layout [default]

```
reviewer-sim/
  configs/            # one YAML per condition and per phase
  reviewer_sim/
    fields.py         # loads gss_2018.yaml: names, types, allowed values, descriptions, input groups
    llm.py            # client, JSON mode, retries, caching, logging
    prompts.py        # official framing + every stating / judging prompt in W1–W3
    parse.py          # JSON parsing, probability validation, renormalization
    tables.py         # stated two-way tables: calls, storage, distances
    loglinear.py      # p(y | inputs) fitted to stated tables (S7, P4 marginals)
    copula.py         # Gaussian copula from stated tables (P4, P3 candidates)
    conditions/       # b4.py, b5.py, b6.py, r3.py, p1.py, p2.py, p3.py, p4.py, p5.py, references.py
    rules.py          # selection rules S0–S7 for values and for whole profiles
    scoring.py        # wrapper around the patched official scorer
    diagnostics.py
    study3.py
    run.py            # CLI: run a condition on a row set; score a run folder
  patches/ssdatabench-seed.diff
  data/               # row lists, groups, stated tables; never real outputs inside prompts
  runs/<phase>/<condition>/<wording>/<run_id>/  # sampled_inputs_*.csv + sim_profiles_*.csv + logs
  reports/
```

Every condition writes one `sim_profiles_*.csv` per run, in the official format, keyed by `profile_id`, so the official scorer reads any run folder unchanged.

### 3.2 Stated-probability calls

- **Response:** JSON only, e.g. `{"<option>": p, ...}` or `{"A": p, "B": p}`.
- **Validation:** every required key present; values numeric and ≥ 0. If the sum is in 0.9–1.1, renormalize. Otherwise retry, up to 3 times. If it still fails, use a uniform distribution for that call and flag it.
- **Option order:** shuffle per call with a logged seed; parse by label, not position.
- **Judge calls on candidates:** always run twice with different label orders and average the two probability vectors.
- **Stated two-way tables:** one call per table. The prompt names the population and year, the input group (or the conditioning output's category), and asks for the share of each allowed answer. Each row of the table is a distribution and is validated as above.

### 3.3 Scoring patch

The official Types 1–3 create their random generators internally (`rng=None`), so scores vary from run to run. Patch them minimally so one seeded `numpy.random.Generator` is passed through. Keep the patch as `patches/ssdatabench-seed.diff`.

**Check:** the same seed gives identical scores twice, and different seeds give slightly different ones.

### 3.4 Run definition

A **run** is one independent pass over the row set, scored once with one scoring seed. In Phases 0 and 2 the expensive calls (candidates, judge probabilities, stated tables) are made once and only selections and draws are redrawn per run; the report says so, since this understates generation variability. In Phases 3 and 4 a run redoes every call.

### 3.5 Development data

1. Remove from `gss2018.csv` every respondent in the evaluation sample (the replication's `sampled_inputs_*.csv`). The rest are the development respondents. If fewer than 1,000 remain, stop and ask the user.
2. Split them once, at random (seed 2026), into a **fitting part** (40%) and an **evaluation part** (60%) **[default]**. Save both lists in `data/`. The fitting part fits B1, B2 and the Study 3 reference models. The evaluation part supplies all pilot rows, Study 3 items and Phase 3 rows.
3. Draw the **pilot rows**: 500 rows from the evaluation part **[default]**, or all of it if smaller. Write them in the official `sampled_inputs` format (with real outputs, used only for scoring).

---

## 4. Phase 0 — Study 1: one question, which readout extracts the most

**Purpose.** Test the premise (H0: the model states distributions better than it samples them) and pick the judging rule, at the smallest scale where the official score can still be computed.

### 4.1 Choosing the small generator

- **Candidates:** 2–3 open models plus the test model for comparison:
  - the Qwen2.5 model in SSDataBench's model list (it has a published score);
  - a newer mid-size open instruction model on OpenRouter (the user names it);
  - a base, non-instruction model, if one is served.
- **Procedure:** each model produces 10 full profiles for each of 50 pilot rows, with the official prompt and validation.
- **Report per model:** valid-output rate; share of duplicate candidates within a row; mean entropy of each output across a row's 10 candidates; mean pairwise Hamming distance between a row's candidates.
- **Choice [default rule]:** the most diverse model, by mean entropy, among those with at least 90% valid outputs. It is the **small generator** for P2. Report the table (Table S1); no stop is needed here.

### 4.2 Study 1 setup

- **Questions:** `political_view` and `mental_health` (numeric, 0–30), run separately.
- **Rows:** the 500 pilot rows.
- **Generators:** the test model and the small generator, each run separately. The test model is always the judge.
- **Candidates:** for each row, question and generator, 10 answers from a **one-question prompt**: the official framing, with only this question's description and allowed values. Store all 10.
- **Judge call J:** the judge sees the input row and the distinct values among the 10 candidates, in random order, and states for each "the probability that a real respondent with these inputs gives this answer"; probabilities sum to 1. Two orders, averaged. Write v for a distinct value, j(v) for its averaged probability, c(v) for how many of the 10 candidates gave it.
- **Wordings:** every call that has a wording (J, S3, S5, S6, S7 tables) runs in W1, W2 and W3. Each rule is scored per wording; the rule's score is the mean.

### 4.3 The rules

| Rule | Readout | Selection | Extra calls per row |
| --- | --- | --- | --- |
| S0 | Sample | Draw a candidate uniformly, i.e. value v with probability c(v)/10: generation itself | 0 |
| S1 | Judge | Draw v with probability ∝ c(v) · j(v) | 0 (uses J) |
| S2 | Judge, greedy | The v with the highest j(v); ties at random. Expected to lose | 0 |
| S3 | Judge, filter | One call: the judge marks each distinct value plausible / implausible. Draw uniformly among candidates with plausible values; if none, S0 | 1 |
| S4 | Judge, frequency-corrected | Draw v with probability ∝ j(v): the generator's frequency is divided out | 0 |
| S5 | Judge, pairwise | Shuffle candidates; the current pick is the first. For each later candidate with a different value, the judge states P(this one rather than the current pick fits a real respondent with these inputs), both orders averaged; replace with that probability | ≤ 18 |
| S6 | State, individual | One call: the judge states the probability of every allowed answer for this person (all 31 for mental_health); draw from it. No candidates | 1 |
| S7 | State, population | The model states 8 two-way tables, this question × each input (one call per input category, §3.2). A log-linear model (§4.4) fitted to them gives each person's distribution; draw from it | ≈ 50 in total |

### 4.4 The log-linear fit (S7, and P4's marginals)

Fit a multinomial logit p(y | x) with additive main effects of the 8 input categories **[default]**, choosing its parameters to minimize the KL divergence between the stated two-way tables and the tables implied by the model when averaged over the row set's actual input distribution (inputs are given to every method, so using their distribution is allowed). Report the fit (mean absolute deviation per table). No real outputs are used.

### 4.5 Scoring and decision

- **Scoring:** the official Type 1 test for this question, plus official Type 2 for this question paired with each of the 8 inputs, run on a config restricted to the inputs and this question with master settings (B = 100, α = 0.05, 500 per draw) and the seed patch. Type 3 does not apply.
- **Diagnostics:** within-group spread ratio (model / real, in input groups with at least 30 rows), collapse rate, total variation distance of the marginal, and the signed difference in association strength with each input (model minus real).
- **Runs:** candidates, judge calls and tables are made once per wording. There are 3 runs (seeds 1–3); each redraws every rule's selection and uses a new scoring seed.
- **What it shows.** S6 vs S0 is H0 directly. S1 and S4 vs S6 is H2a in miniature. S7 vs S6 is H2b in miniature. S2 vs S0 is the collapse check.
- **Decision [default rule]:** for each generator, rank S1, S3, S4, S5 by mean score (Type 1 and Type 2 combined as in the official aggregate), averaged over both questions and wordings. The test-model generator's winner is the **judging rule** for P1, P3 and P5; the small generator's winner is the rule for P2. The better of S6 and S7 is the **stating rule** noted for Study 2 (it informs B5/R3 vs P4, which are all run anyway). If no judging rule beats S0 on either question, Phase 2 still runs with S1, and the Study 1 result is reported as is.

**Phase 0 report** (`reports/phase0.md`): the generator-choice table; per question × generator × wording × rule: score, Type 1, Type 2, diagnostics, mean and range over runs; the winning rules; the H0 verdict in miniature (S6 vs S0 and spread ratio); judge diagnostics (entropy, correlation between orders, parse failures); calls and tokens. This is the material for Figure 2. **→ Stop. User checkpoint.**

---

## 5. Phase 1 — Setup

1. **Final-test rows.** The full benchmark sample. Freeze the list in `data/final_rows.csv` now.
2. **Input groups** (§2) and **coarse groups** gender × race × age band (18 groups) for diagnostics. Report how many coarse groups have at least 30 respondents in the evaluation part.
3. **References**, fitted on the development fitting part, never used by a method under test:
   - **B1:** samples each output from its marginal within the row's coarse group.
   - **B2:** a Gaussian copula of outputs given inputs **[default: categorical outputs as ordinal codes; document the encoding]**.
4. **Study 3 reference models**, fitted on the fitting part: a Bayesian network over inputs and outputs **[default: hill-climb, BIC, at most 3 parents]** for calibration and the typicality trap; a gradient-boosting classifier for discrimination (§7.1).
5. **Preliminary B4 look.** Score the user's existing B4 on the full benchmark sample and report it next to the paper's published score (user supplies it). The formal reproduction check is in Phase 4.
6. **Diagnostics module** (§9), tested on B4 and B1.

**Phase 1 report:** sample sizes, group counts, reference scores on the evaluation part, the preliminary B4 score next to the paper's. **→ Stop. User checkpoint.**

---

## 6. Phase 2 — Study 2 pilot: whole profiles across the matrix

All conditions on the 500 pilot rows, all 21 outputs, scored with the official scorer against those rows' real outputs. Study 3 (§7) runs alongside on the same rows.

### 6.1 Conditions

| ID | Cell | Procedure | Calls per row |
| --- | --- | --- | --- |
| B4 | Individual · sample | Fresh official generation, one profile (the replication covers only benchmark rows) | 1 |
| B6 | Population · sample | Rows grouped by coarse group in batches of 10 **[default]**; one call shows the 10 input rows and asks for 10 profiles "reflecting the real variation among such people"; validated as B4; a failed batch is retried, then split in two | ≈ 0.1 |
| B5 | Individual · state | For each output, one call stating its distribution given the inputs; outputs drawn independently | 21 |
| R3 | Individual · state, joint | §6.2 | 21 |
| P4 | Population · state | §6.3 | ≈ 400 in total |
| P1 | Individual · judge | 5 official generations by the test model; judge call on the 5 profiles (§6.4), two orders averaged; select by the judging rule applied to whole profiles | 7 |
| P2 | Individual · judge | 10 official generations by the small generator; test model judges them (source not revealed); same rule | 12, mostly to the small model |
| P3 | Individual · judge | 5 candidates drawn from P4's fitted copula for this row's inputs; same judge call and rule | 2 (+ P4's calls) |
| P5 | Population · judge | §6.5, with candidates from P1's pool **[default]**; λ ∈ {0, 1, 5} | as P1, plus bookkeeping |
| P5-view | Population · judge | §6.5 | as P5 |

**Contrasts, produced from the same calls:** P1-random, P2-random (uniform pick among the same candidates); P3-random is P4 itself; P5 at λ = 0 is P1. References B1 and B2 are scored alongside.

**Whole-profile rules.** Distinct profiles take the place of distinct values in S1/S3/S4/S5; S3 and S5 make their calls on profiles. Duplicate profiles stay separate entries; report the duplicate share per source.

**Wordings.** All methods run W1. B5, R3, P1, P2, P3 and P4 also run W2 and W3 (P3 and P5 inherit the judge wording; P4 the table wording). B4 and B6 have one official wording.

### 6.2 R3: joint stating along a graph

- **Graph:** one call lists the 21 outputs with descriptions and asks, for each output, up to 3 parent outputs, forming a DAG of how answers depend on each other. Validate acyclicity; re-ask up to 3 times. Fix the graph once per wording and log it.
- **Sampling:** per row, in topological order, the model sees the input row plus the sampled values of the output's parents and states the output's full distribution; draw from it.
- **Coherence check:** a full-chain variant (each output conditioned on all earlier outputs) in 3 random orders on 200 rows; report how far scores and pairwise associations differ. This feeds Study 3's coherence row.

### 6.3 P4: stated tables → copula reconstruction

- **Tables:** output × input group for all 21 outputs × 8 inputs (168 tables) and output × output for all 210 pairs, one call per table (§3.2). For an output × output table the conditioning output's categories are listed and the model states the distribution of the other output within each.
- **Marginals:** p(y_k | x) for each output from the log-linear fit (§4.4) to its 8 output × input tables.
- **Dependence:** convert each stated output × output table to a polychoric (ordinal) or, for nominal outputs, a Cramér's V-derived correlation **[default; document the mapping]**; assemble the 21 × 21 matrix; project to the nearest positive-definite matrix and report how far it moved.
- **Sampling:** per row, draw a 21-dimensional Gaussian with that correlation matrix and map each coordinate through the inverse CDF of p(y_k | x). This is also the candidate generator for P3.
- Stated tables are stored in `data/stated_tables/<wording>/` and reused by P3 and P5.

### 6.4 Whole-profile judge prompt (W1)

Official framing (population, year, input attributes, field descriptions and allowed values), then the k profiles as JSON labelled A, B, C …, then:

> "For people with exactly these input attributes in this population, how likely is each profile to be the answers of a real, randomly selected respondent? Give a probability for each profile; the probabilities must sum to 1. Judge each profile as a whole: whether its answers fit the inputs and fit each other."

Output is one key per label. W2 and W3 are paraphrases fixed before Phase 0; the user reviews all three.

### 6.5 P5: the population builder

- **Order:** rows added one at a time in a random order, reshuffled per run (order dependence is a reported quantity).
- **Candidates and judge:** the same k candidates and averaged judge probabilities j(c) as P1 (or P2, if the user chooses that pool).
- **Population term:** D(T) = mean over P4's stated tables of the total variation distance between the stated table and the same table computed from the rows added so far. For each candidate c, ΔD(c) = D(rows so far + c) − D(rows so far), computed by the program. For the first 50 rows **[default]** ΔD is set to 0 (tables are too noisy).
- **Selection:** draw c with probability ∝ j(c) · exp(−λ · ΔD(c) / s), where s is the median |ΔD| over the first 100 rows at λ = 0 **[default scale]**, logged. Never pick greedily.
- **λ:** 0, 1, 5 **[default]**. The score-against-λ curve is Figure 5.
- **P5-view:** the judge sees, next to the candidates, the current population's summary tables for this row's input groups and the stated target tables, and is asked directly which candidate to add; its stated probabilities replace j(c) · exp(−λΔD). One wording, W1.

### 6.6 Runs, decision rules and report

- **Runs:** candidates, judge probabilities and tables are made once per wording; 3 runs (seeds 1–3) redraw every selection and draw, with a new scoring seed each.
- **Go / no-go [fixed before running]:** for each judging method (P1, P2, P3, P5 at λ > 0), **go** if the mean of (method − its contrast) over the 3 paired runs is positive and larger than twice the contrast's range across runs. Contrasts: P1-random, P2-random, P4, P5 at λ = 0.
- **Also report:** P2-random − B4 (switching generators alone); B5 and R3 vs B4 on Type 1 and spread ratio (H0 at full scale); P4 vs B5 and R3 on output × output Type 2 and Type 3 (H2b in miniature); the share of P5's picks that changed relative to λ = 0.
- **If no judging method passes:** the Study 3 results on the same rows (§7) are the explanation; report whether judge probabilities were near-uniform or unstable across orders and wordings.
- **Acceptance checks:** P1-random's score within twice B4's range across scoring seeds; every condition has exactly one valid profile per row; an automatic check confirms no real output value appears in any logged prompt.

**Phase 2 report** (`reports/phase2.md`): results per condition × wording (aggregate, Types 1–3, mean and range), all contrasts, go / no-go per method, Study 3 results and gate (§7), diagnostics (§9), duplicate shares, missing rates, calls and tokens. **→ Stop. User checkpoint.**

---

## 7. Study 3 — why it works or fails (runs in Phase 2)

Items come from the development evaluation part (300 items per test **[default]**); reference models are fitted on the fitting part (§5.4). Nothing from Study 3 feeds generation.

### 7.1 Tests

| Test | What the judge sees | Measure | Fails if **[default thresholds; user confirms before running]** |
| --- | --- | --- | --- |
| Discrimination | A real profile and the same profile with one output swapped for a value drawn from that output's marginal in the person's coarse group; the judge states P(each is real) | AUROC; reference: a gradient-boosting classifier trained on the fitting part to separate real from altered | AUROC below 0.5 + half the reference classifier's margin above 0.5 |
| Calibration | Two partial profiles that differ in one output (the input row, two conditioning outputs, the differing output) | Slope of logit(judge P) on logit(Bayesian-network share) | Slope outside 0.5–1.5 |
| Typicality trap | Pairs where the answer the test model's own generation (Phase 2's B4) favours for this input profile is the less common one in the evaluation part | Accuracy on these pairs | At or below 50% |
| Order and wording stability | The same items in two label orders and in W1–W3 | Share of judgments that flip (argmax changes) | Reported; above 20% flagged |
| Coherence | 50 output pairs per item: P(X given Y), P(Y given X) and both marginals | Mean absolute deviation from Bayes' rule | Reported; large deviation puts a caveat on R3 and P4 |
| Population discrimination | Summary tables (marginals and the output × input and output × output cross-tabs for one coarse group) of the real evaluation part and of one method's synthetic data, labelled A and B; asked which is real and which cells give it away. Run for B4, B5, P4 and the best judging method, with labels swapped, 20 groups each **[default]** | Accuracy at naming the real set; overlap between the flagged cells and the 10 cells with the largest real deviation | Reported; accuracy at chance means P5-view is not trusted |

### 7.2 Per-output measures and the mechanism test (H3)

Discrimination and calibration are computed per output (the swapped or differing output), giving 21 values each. The per-output **gain** is the reviewer method's Type 1 score minus B4's on that output, from Phase 2 (and later Phase 4). Report the Spearman correlation between per-output calibration (and discrimination) and per-output gain with a bootstrap 95% interval. This is Figure 6; the second model doubles the points in Phase 4.

### 7.3 Gate

- The judge **passes** when discrimination, calibration and the typicality trap all pass.
- A failed judge does not stop the work: methods still run, and the failure profile becomes the explanation.
- The population-discrimination result decides whether P5-view is carried into Phase 3.

---

## 8. Phase 3 — Study 2 full: choosing the method

All conditions that were a **go** in Phase 2, plus B4, B5, R3, P4, B6 and the contrasts, on the whole development evaluation part **[default]**, scored against those respondents' real outputs. Run 3 times each; a run redoes every call. Wordings: W1–W3 for every stating or judging method.

- **Eligible for Phase 4:** any judging or stating method under test (B5, R3, P4, P1, P2, P3, P5, P5-view) that is not a reference. The chosen method is the eligible one with the highest mean aggregate score over runs and wordings.
- **Report each contrast:** P1 vs P1-random; P2 vs P2-random; P2-random vs B4; P3 vs P4; P5 (λ > 0) vs λ = 0; P4 and P5 vs the best individual method on output × output Type 2 and Type 3 (H2b); best judging vs best stating (H2a).
- **Run sizing.** From the between-run standard deviation of the chosen method and of B4, compute how many runs bring the 95% interval half-width of their difference below half the MMD (§9). At least 5.
- **MMD:** the median gap between adjacent models in the paper's GSS 2018 results, from the paper's numbers (user supplies). Fixed here.

**Phase 3 report:** results for all conditions, references and wordings; contrasts; diagnostics; R3 graph coherence; P5 order dependence; the chosen method, run count and MMD; calls and cost. **→ Stop. User checkpoint.**

---

## 9. Phase 4 — Study 4: final test and robustness

- **Rows:** `data/final_rows.csv`, the full benchmark sample. Real comparison data: the benchmark's `sampled_inputs`.
- **Conditions:** B4; B5; the chosen method; its contrast (random pick, P4, or λ = 0); references B1 and B2 for context. On the second model (another family the paper evaluated): B4, the chosen method and its contrast.
- **Runs:** as sized in Phase 3. Each run redoes every call and uses a fresh scoring seed.
- **Wordings:** the chosen method under each of W1–W3; the headline is the mean over wordings, with the spread beside it.
- **Reproduction first.** B4 reproduces the paper if the paper's score lies within B4's between-run interval widened by the MMD. Check this before scoring any other condition.
- **H1 holds if** the chosen method (mean over wordings) passes both: (1) against B4, the 95% interval of the mean difference excludes zero and the mean difference is at least the MMD; (2) its mean exceeds the paper's published score for the model by at least the MMD. If the paper's model is retired, H1 rests on (1), stated before running.
- **H1 null:** the difference interval lies within ±MMD (equivalence). An interval excluding zero with a mean below the MMD is a small, real gain, reported as such and not counted as support. Any other interval is inconclusive.
- **H2a, H2b:** read from Phase 3 (and Phase 4 where the conditions are present) with the same interval rule.
- **H3:** §7.2 on both models.
- **H4:** the signed input × output association difference (model minus real); clearly positive means stereotypes inflated. Check whether P5's population term reduces it.
- **Robustness:** the gain has the same sign under every wording and the spread across wordings is below the MMD.

**Phase 4 report:** every condition's score with intervals, per type; the H1–H4 verdicts read through the design doc's success/failure table; wording spread; second-model results; missing-value rates; diagnostics. This is the material for Figure 3, Figure 4 and Table 2. **→ Stop. User checkpoint.**

## 10. Diagnostics (computed for every scored run)

- Within-group spread ratio, model / real, in coarse groups with at least 30 real rows: standard deviation for ordinal and numeric outputs, entropy for nominal ones.
- Collapse rate: groups where one option exceeds 90% in the model's output but not in the real data.
- Type 2 split into input × output and output × output pairs; signed association difference (model minus real) for both.
- First-eigenvalue share of the outputs, model vs real.
- Respondent coverage gap: share of real respondents with no synthetic profile within Hamming distance d, for d = 2, 4, 6.
- Missing rate per output; stated-probability fallback rate; duplicate-candidate share.
- For P4: distance between stated tables and the tables of the generated data; for P5: the same against λ; for R3: distance between graphs across wordings.
- Calls and tokens per condition and wording.

## 11. Phase 5 — Second dataset

Apply the chosen method, with every setting unchanged (including wordings and λ), to a second SSDataBench dataset the user picks. Same protocol as Phase 4: B4, B5, the method and its contrast. Report without retuning.

---

## 12. Open items to settle with the user

| When | Item |
| --- | --- |
| Before Phase 0 | Path to the replication run and `gss2018.csv`; the test model and version, and whether it is still served; new repository or the existing project; which newer open model to include in the generator choice; review of W1–W3 for every prompt family |
| Before Phase 1 | The paper's GSS 2018 scores per model (for the B4 check and the MMD) |
| Before Phase 2 | Confirm or change the Study 3 thresholds; confirm λ ∈ {0, 1, 5}; P5 candidate pool (P1's or P2's) |
| Before Phase 3 | Second model; whether choosing the method on development data is acceptable (the alternative is fixing it in advance) |
| Before Phase 5 | Which second dataset |

## 13. Out of scope

Prompt optimization beyond the pre-registered wordings, temperature tuning, token-logprob methods, fine-tuning or any training, local model hosting, any use of real outputs inside generation, and the v1 methods R1a, R1b, R2, N2 and R3-theory. Before Phase 5, no dataset other than GSS 2018.

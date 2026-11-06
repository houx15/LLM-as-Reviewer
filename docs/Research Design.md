# LLM reviewer design v2

Nov 3, 2026 · @Evie Hou

Goal: show that an LLM knows more about a population than its generated answers reveal, and that asking it to state or judge reads that knowledge out better than asking it to generate, measured by the SSDataBench GSS 2018 score with only the process changed and no human data.

## Core question and hypotheses

**Core question.** On SSDataBench's GSS 2018 track, with its inputs, outputs, sample size and official scoring unchanged, can a different way of asking the same model (stating distributions or judging candidates, instead of generating answers) produce more realistic synthetic data, using no human data?

**Why it might work.** SSDataBench shows that generated data compress real variation. Jang, Lee and Kim (2026, preprint) report that instruction-tuned models can describe an opinion distribution far better than they can sample from it, and that post-training widens this gap. If the knowledge is there but generation does not use it, the remedy is to change how the knowledge is read out, not the model.

**The claim it would support.** "How you read out a model's population knowledge matters as much as which model you use": a readout based on judgment or stated distributions beats the same model's generation on a published benchmark, with no human data and no training.

Hypotheses, each with a pre-set test (criteria in the success/failure section):

1. **H0, premise.** The model states distributions better than it samples them on SSDataBench's variables: stated-distribution readouts beat generation on univariate realism (Type 1) and within-group spread.
2. **H1, main.** The best stating or judging readout, chosen on development data, scores higher than the same model's official generation (B4) and than the paper's published score for that model.
3. **H2, where the gain comes from.**
   - H2a, judging vs. stating: at the individual level, judging candidates beats stating a distribution.
   - H2b, population vs. individual: population-level readouts beat individual-level ones on joint structure (output × output associations and Type 3).
4. **H3, mechanism.** Across the 21 outputs, the better calibrated the judge is on an output, the larger the gain on that output.
5. **H4, limit.** Judging inflates input–output associations relative to real data (typicality bias).

H0 is the gate for everything else: if the model cannot state distributions better than it samples them, there is no hidden knowledge to read out, and the paper becomes a diagnostic of why.

## Framework: where the knowledge is read out

Every method in this study asks the same model about the same people; they differ only in what the model is asked to produce and whether it sees one person or the whole set.

&#91;embedded content: Figure 1 · readout form × level\]

SSDataBench's own method sits in the top-left corner. The study tests whether moving right (stating, judging) or down (reasoning about the population) recovers realism, and λ moves the population builder continuously between individual judging and population reconstruction. The two-way tables used by P4 and P5 are the formal description of the population: for this benchmark, one-way margins and pairwise associations are what decide realism (a maximum-entropy view), and individual judging adds what tables cannot capture, namely whether one person's answers fit together.

## Ground rules

Every comparison holds the benchmark fixed and changes only how the model is asked.

- **Protocol.** SSDataBench GSS 2018: 8 real inputs per row (birth year also shown in the prompt), 21 outputs, the benchmark's sample size, the official prompt framing and the official scoring (mean insignificance rate over Types 1–3; B = 100, α = 0.05, 500 per draw), with one patch that fixes the scoring seed.
- **No human data in generation.** A method sees only the input rows SSDataBench's own generation sees. Real answers are used only to score, to test the judge (Study 3) and to fit two labelled references (B1 independent marginals, B2 Gaussian copula) that show what real data alone would achieve.
- **Models.** API calls, temperature 1.0, no tuning. The test model is one the paper evaluated, so its published score is an external reference. A second model from another family repeats the final test.
- **Development vs. final test.** Methods are compared and chosen on respondents in the benchmark's GSS 2018 file who were not drawn into its evaluation sample (the development respondents). The final test then runs once on the full evaluation sample. Development respondents are split 40/60: the fitting part fits the references and the judge-test reference model, and the evaluation part scores everything during development.
- **What is estimated.** B4's score is defined on the official generation prompt. Every stating or judging method's score is defined on a pre-registered set of 2–3 wordings of its question, with answer options shown in at least two orders. The report gives the average and the spread across wordings, not one "true" answer. Format effects that hit all conditions alike cancel in the paired comparisons; differences in absolute level or spread are interpreted with that caveat.
- **Model memory.** GSS 2018 is likely in training data. That is not a confounder, since B4 shares the same memory, and the methods aim to use it.

## Study 1: one question, which readout extracts the most

Study 1 tests the premise (H0) and picks the readout rule, at the smallest scale where the score can still be computed. It is the first pilot: about 500 development rows, a few days, a few dollars.

**Setup.** Two questions, run separately: political\_view (an attitude known to collapse) and mental\_health (numeric, 0–30). For each row, a generator gives 10 candidate answers using a one-question version of the official prompt. Two generators are run: the test model, and a small open model chosen beforehand as the most diverse among 2–3 candidates (measured on 50 rows by valid-output rate, duplicate share and entropy). The test model is always the judge.

| Rule | Readout | How one answer is produced |
| --- | --- | --- |
| S0 | Sample | One candidate at random: generation itself |
| S1 | Judge | Draw a candidate weighted by the judge's probability for its value |
| S2 | Judge, greedy | The value the judge rates highest |
| S3 | Judge, filter | The judge marks values plausible or not; one plausible candidate at random |
| S4 | Judge, frequency-corrected | Draw a value in proportion to the judge's probability alone, dividing out how often the generator proposed it |
| S5 | Judge, pairwise | Compare candidates one by one with the current pick; replace with the judge's probability |
| S6 | State, individual | No candidates: the model states the distribution over all answers for this person; draw from it |
| S7 | State, population | The model states 8 two-way tables (this question by each input); a log-linear model fitted to them gives each person's distribution; draw from it |

**What it shows.**

- S6 against S0 tests H0 directly: does stating beat sampling for the same person?
- S1 and S4 against S6 is H2a in miniature: does judging candidates add anything over stating?
- S7 against S6 is H2b in miniature, without joint structure.
- S2 is expected to lose: choosing the most likely answer should collapse variation, which is why every judging rule samples instead of choosing.

**Scoring.** The official Type 1 test for the question and its official Type 2 associations with the 8 inputs, plus the within-group spread ratio and the signed difference in association strength. Each rule is scored on 3 redraws. The best judging rule and the better stating rule go to Study 2.

## Study 2: whole profiles across the matrix

Study 2 fills each cell of the framework with one method, on all 21 outputs, so that joint structure can be tested. It runs first as a pilot on 500 development rows (go / no-go), then on the whole development evaluation part to choose the method for the final test.

| ID | Cell | What the model does | How the profile is produced | Calls |
| --- | --- | --- | --- | --- |
| B4 | Individual · sample | Answers all outputs with the official prompt | As generated | 1 per row |
| B6 | Population · sample | Writes profiles for 10 input rows of one group at once, asked to reflect real variation | As generated | about 0.1 per row |
| B5 | Individual · state | States each output's distribution for this person | Outputs drawn independently | 21 per row |
| R3 | Individual · state, joint | States each output's distribution given the person and earlier outputs, along a dependency graph it proposes | Drawn in graph order | 21 per row |
| P4 | Population · state | States two-way tables: each output by input group, and output by output | A Gaussian copula fitted to the stated tables; each person's outputs drawn given their inputs | about 400 in total |
| P1 | Individual · judge | Judges 5 of its own candidate profiles | Study 1's best judging rule | 7 per row |
| P2 | Individual · judge | Judges 10 candidates from the small generator | Same rule | 12 per row, mostly to the small model |
| P3 | Individual · judge | Judges 5 candidates drawn from P4's fitted copula for this person's inputs; the candidates already match the stated tables, so the judge only adds individual coherence | Same rule | 2 per row, plus P4's calls |
| P5 | Population · judge | Judges candidates while the population is built row by row | Weight = judge's probability × exp(−λ · change in distance to P4's tables) | as P1 or P2, plus bookkeeping |

**P5, the population builder.** Rows are added in random order, like Lego bricks. For each row, every candidate gets two scores: how plausible the person is (the judge) and whether adding them moves the population's tables toward or away from the stated tables (computed by the program). The pick is drawn, never chosen greedily.

- λ = 0 is pure individual judging (P1 or P2).
- A large λ with uniform plausibility is pure reconstruction (close to P4).
- λ takes three pre-set values (0, 1, 5 \[default\]), and the curve across them is a result in itself.

One variant, P5-view, gives the judge the current population's summary tables next to the stated tables and lets it judge directly, which tests whether the model can reason about a population, not just a person. Row order is reshuffled across runs, so order dependence is measured.

**Contrasts.**

- Each judging method is compared with a random pick among the same candidates (P1-random, P2-random), which isolates the judge.
- P2-random against B4 isolates switching generators.
- P3 against P4 (a random pick among P3's candidates is P4 itself) isolates judging on top of stated population tables. With P1 and P2 it makes candidate source a third axis: own generation, a small model, or the stated distribution.
- P5 at λ > 0 against λ = 0 isolates the population term.

**Dropped from v1.** The two expensive chains (R1a, R1b) and the pairwise distributor (R2) are dropped. Their logic survives in cheaper form: pairwise judging in Study 1 (S5), population allocation in P5.

## Study 3: why it works or fails

Study 3 measures the judge directly, so that a gain (or its absence) can be explained rather than only reported. It runs on development respondents, alongside the Study 2 pilot. Nothing from it feeds generation.

| Test | What the judge sees | Measure |
| --- | --- | --- |
| Discrimination | A real profile, and the same profile with one output swapped for a value drawn from that output's marginal in the person's group | AUROC, compared with a classifier trained on the fitting part |
| Calibration | Two partial profiles that differ in one output | Slope of the judge's stated probability against a reference model's (a Bayesian network fitted on the fitting part) |
| Typicality trap | Pairs where the answer the model's own generation favours for this kind of person is the less common one in real data | Accuracy on these pairs alone |
| Order and wording stability | The same items in two orders and 2–3 wordings | Share of judgments that flip |
| Coherence | P(X given Y), P(Y given X) and both marginals | Deviation from Bayes' rule; needed to trust R3 and P4 |
| Population discrimination | Summary tables (marginals and cross-tabs) of the real development data and of one method's synthetic data, labelled A and B; asked which is real and which cells give it away | Accuracy at naming the real set over methods and relabellings; overlap between the flagged cells and the cells with the largest real deviation. Needed to trust P5-view, and the basis for any later method that lets the model audit a finished dataset |

**The mechanism test (H3).** Each of the 21 outputs gets its own calibration and discrimination scores, and its own gain (the reviewer method's Type 1 score minus B4's on that output). If judging works because the model knows the population, the outputs where it judges well should be the outputs where it gains most. That gives 21 points per model, and twice that with the second model.

**Thresholds.**

- The judge passes when discrimination, calibration and the typicality trap all pass. The proposed values (AUROC at least half the reference classifier's margin above 0.5, calibration slope 0.5–1.5, typicality accuracy above chance) are fixed before the test runs.
- A failed judge does not stop the work. Methods still run, and the failure profile becomes the explanation.

## Study 4: final test and robustness

Study 4 runs once, on the full SSDataBench evaluation sample, and produces the numbers the paper's claim rests on.

- **Conditions:**
  - B4, the official generation;
  - B5, the strongest no-review stating baseline for comparison;
  - the method chosen in Study 2: the eligible method with the highest mean score on the development evaluation part;
  - that method's contrast (its random pick, or λ = 0);
  - references B1 and B2 for context.
- **Runs.** Each run redoes every call. The number of runs is set from Study 2's between-run spread, so that the 95% interval of a difference is narrower than half the minimum meaningful difference; at least 5.
- **Minimum meaningful difference (MMD).** The median gap between adjacent models in the paper's GSS 2018 results, fixed before the test.
- **Reproduction first.** B4 must reproduce the paper's score for the same model, within its between-run interval widened by the MMD, before any other condition is scored.
- **Second model.** B4, the chosen method and its contrast are repeated on a model from another family that the paper also evaluated.
- **Wording robustness.** The chosen method is run under each pre-registered wording. The headline result is the average over wordings, with the spread reported beside it.
- **Second dataset.** The same method, unchanged, on another SSDataBench dataset, to show it is not tuned to one benchmark table.

## What counts as success or failure

The experiment succeeds if H0 and H1 hold, and it is informative even if they fail, because each failure points to a different explanation. All rules below are fixed before the data they apply to are seen.

| Question | Success | Failure | What a failure means |
| --- | --- | --- | --- |
| H0 premise (Study 1 and 2) | Stating beats sampling: S6 above S0 in Study 1, and B5 above B4 on Type 1, with within-group spread closer to real | No difference, or stating worse | The model does not know more than it generates for these variables; the reviewer idea has no basis here, and Study 3 explains why |
| Pilot go / no-go (Study 2 pilot) | A judging method beats its random pick by more than twice the random pick's range across runs | Not above the random pick | The judge adds nothing in this form; run Study 3 on the same rows before building more |
| H1 main (Study 4) | Against B4: 95% interval of the difference excludes zero and mean ≥ MMD; and mean ≥ the paper's score + MMD | Interval within ± MMD (equivalent) | Reading out differently does not raise the score for this model and benchmark |
| H2a judging vs. stating | Best judging method above best individual stating method (P1 or P2 vs. B5 and R3) | Stating equal or better | The value lies in asking for distributions, not in reviewing |
| H2b population vs. individual | P4 or P5 above the best individual method on output × output Type 2 and on Type 3 | No gain on joint structure | The model's knowledge is about individuals and marginals, not structure |
| H3 mechanism (Study 3) | Positive rank correlation between per-output judge calibration and per-output gain, interval excluding zero | No correlation | Gains, if any, do not come from judgment accuracy as measured |
| H4 limit | Signed input–output association difference near zero | Clearly positive: stereotypes inflated | Report as a cost of judging; check whether P5's population term reduces it |
| Robustness (Study 4) | Gain has the same sign under every wording, and the spread across wordings is below the MMD | Sign flips across wordings | The result is a property of one prompt, not of the readout |

Other outcomes:

- An interval that excludes zero but with a mean below the MMD is a small, real gain. It is reported as such and not counted as support for H1.
- An interval wider than ± MMD that includes zero is inconclusive.
- The strongest result would be a method reaching the B2 reference: with no human data, the model matching a model fitted on real respondents.

## Paper building blocks and figures

The paper tells one story: the knowledge is in the model, generation fails to use it, and the right readout recovers it. Each block answers one step of that story with one main exhibit.

| Block | The point it makes | Main exhibit |
| --- | --- | --- |
| 1. Motivation | SSDataBench shows generated data compress real variation; models describe distributions they cannot sample | Short text; prior results cited |
| 2. Framework | Readouts differ along two dimensions, form (sample, state, judge) and level (individual, population); the population builder links them through λ | **Figure 1**: the readout matrix, with each method placed in its cell and the λ axis drawn through the judge column |
| 3. Premise and readout rules (Study 1) | Stating beats sampling for the same person; judging adds (or does not add) on top; choosing the top answer collapses | **Figure 2**: score for each rule S0–S7, both questions, both generators, with intervals; inset: within-group spread ratio, where 1 = real |
| 4. Main result (Study 2 and 4) | Which cell recovers most realism on the benchmark itself | **Figure 3**: SSDataBench score per method, with reference lines for B4, the paper's score and B2; panels for Types 1, 2 and 3. **Table 2**: final test, with intervals, H1 verdict, second model and wording spread |
| 5. Where the gain lives | Gains on marginals vs. joint structure; whether judging inflates stereotypes | **Figure 4**: Type 2 split into input × output and output × output, plus the signed association difference (H4) |
| 6. Individual vs. population | How adding the population term changes realism | **Figure 5**: score against λ for P5, with P5-view as a separate point and the spread across row orders |
| 7. Why (Study 3) | Gains track judge calibration | **Figure 6**: per-output gain against per-output calibration, one point per output and model, with the rank correlation |
| 8. Discussion | Toward post-training that keeps distributions: training to report or sample in proportion, rather than to give the single best answer | Text |

In every figure, axis labels say which direction is better (for example "insignificance rate, higher = closer to real"), and reference lines carry their names. Supporting tables:

- **Table 1**: the methods, with their cell, what the model does and calls per row (the Study 2 table);
- **Table S1**: generator choice;
- **Table S2**: Study 3 results;
- **Table S3**: missing and fallback rates per condition.

## Order of work and decisions

The cheapest test of the riskiest assumption comes first, and each step ends with a stop for the user.

1. **Generator choice and Study 1** (development rows, about a week). Checkpoint: does stating beat sampling (H0), and which judging rule wins?
2. **Study 2 pilot and Study 3** (500 development rows). Checkpoint: go / no-go per judging method, and whether the judge passes.
3. **Study 2 full** (whole development evaluation part). Checkpoint: chosen method, number of final runs, MMD from the paper.
4. **Study 4** (full benchmark sample, second model, wordings). Checkpoint: H1–H4 verdicts.
5. **Second dataset.**

**Next project, not in this paper.** If judging or stating beats generating, the pipeline's outputs can be distilled back into the model, or the model trained to sample in proportion to plausibility (GFlowNet-style) or to report distributions under a proper scoring rule (as SubPOP does with human data). A model that improves its own generation through its own judgment, with no human data, would test the generation–verification gap view of self-improvement (Song et al., 2025).

**Open decisions.**

- [ ] The test model, its version, and the paper's GSS 2018 scores (for the reproduction check and the MMD).
- [ ] How many respondents the benchmark's evaluation sample uses, and so how many development respondents remain (the user supplies the data file).
- [ ] The newer open model to include in the generator choice.
- [ ] The λ values for P5 (proposed 0, 1, 5) and the 2–3 wordings, fixed before Study 2.
- [ ] Whether to pre-register H1–H4 and the criteria table.

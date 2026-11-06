# Pre-registered prompt wordings (W1–W3)

Companion to `SPEC.md` §1.5 and `prompts.py`. Every stating or judging call is built as:

```
[official framing, verbatim from build_prompt: population, reference year, the respondent's
 input attributes, and the description + allowed values of every field involved]

[one of the wordings below]

[fixed output instruction: JSON only, keys as stated, no other text]
```

Only the middle block varies across W1–W3. The framing and the output instruction are identical across wordings, so parsing never changes. W1 is the default. Once reviewed, these are frozen; any later change is logged and reported as a research choice.

The three wordings deliberately differ in stance, not just in synonyms:

- **W1 — probability.** Asks directly for the probability that a real respondent answers this way.
- **W2 — frequency.** Asks "out of 100 real respondents like this, how many…". Frequency framings are known to be read more literally than probabilities and often spread the mass more.
- **W3 — analyst.** Puts the model in the role of a survey researcher who has studied this population, and asks for their estimate. Tests whether the "expert" stance changes calibration.

Placeholders: `{N}` number of candidates or distinct values; `{Q}` the output's description from the config; `{G}` an input group such as "women aged 35–54"; `{Y}` the conditioning variable and `{y}` one of its categories.

---

## Family 1 — Judge call on candidate values (Study 1, call J)

The judge sees the input row and the `{N}` distinct values proposed for one question, labelled A, B, C …

**W1**
> Below are {N} possible answers to the question "{Q}" for a person with exactly the attributes above. For each answer, give the probability that a real, randomly selected respondent with these attributes in this population gave that answer. The probabilities over the listed answers must sum to 1. Do not favour an answer because it seems typical; favour it only if it is actually more common among such people.

**W2**
> Imagine 100 real respondents from this population who all have exactly the attributes above. Below are {N} possible answers to the question "{Q}". For each answer, say how many of the 100 would have given it. The numbers must sum to 100. Answers that only some of these people would give should get the share those people represent, not zero.

**W3**
> You are a survey researcher who has studied this population closely. For a respondent with exactly the attributes above, the question "{Q}" has {N} candidate answers listed below. Based on what you know about how such people actually answered, estimate for each candidate the probability that it is this respondent's real answer. The probabilities must sum to 1.

Output instruction (all wordings): `Return JSON only, one key per label, e.g. {"A": 0.4, "B": 0.6}.` (W2 returns counts; the parser divides by the total.)

## Family 2 — Judge call on whole candidate profiles (Study 2: P1, P2, P3, P5)

The judge sees the input row and `{N}` complete profiles (all 21 outputs) labelled A, B, C …

**W1**
> For people with exactly these input attributes in this population, how likely is each profile below to be the answers of a real, randomly selected respondent? Give a probability for each profile; the probabilities must sum to 1. Judge each profile as a whole: whether its answers fit the inputs and fit each other.

**W2**
> Imagine 100 real respondents from this population with exactly these input attributes. Each of the {N} profiles below is one possible full set of answers. For each profile, say how many of the 100 respondents would have a set of answers like it — considering both whether the answers fit the person's attributes and whether they are consistent with one another. The numbers must sum to 100.

**W3**
> You are a survey researcher checking synthetic records against what you know of this population. Each of the {N} profiles below is a candidate record for one respondent with exactly these input attributes. Estimate for each the probability that it is the real respondent's record rather than an implausible one, paying attention to internal consistency across answers as well as fit with the attributes. The probabilities must sum to 1.

Output instruction as Family 1.

## Family 3 — Plausibility filter (rule S3)

The judge sees the input row and the distinct values (or profiles).

**W1**
> For a person with exactly the attributes above, mark each listed answer as "plausible" if a real respondent with these attributes could reasonably have given it, or "implausible" if it would be very unusual or contradictory. Mark an answer implausible only when you are confident; uncommon is not the same as implausible.

**W2**
> Among 100 real respondents with exactly these attributes, which of the listed answers would at least a few of them have given? Mark those "plausible". Mark an answer "implausible" only if essentially none of the 100 would have given it.

**W3**
> As a survey researcher who knows this population, screen the listed answers for a respondent with exactly these attributes. Mark each "plausible" or "implausible", where implausible means you would flag it as a likely data error if it appeared in a real record.

Output instruction: `Return JSON only, one key per label with the value "plausible" or "implausible".`

## Family 4 — Pairwise judgment (rule S5)

The judge sees the input row and two candidates, X and Y.

**W1**
> For a person with exactly the attributes above, which of the two answers below is more likely to be a real respondent's answer to "{Q}"? Give the probability that X is the real one rather than Y.

**W2**
> Among 100 real respondents with exactly these attributes, suppose each gave either answer X or answer Y to "{Q}". How many gave X? Return a number from 0 to 100.

**W3**
> As a survey researcher who knows this population, compare the two candidate answers to "{Q}" for a respondent with exactly these attributes. Estimate the probability that X, rather than Y, is the respondent's actual answer.

Output instruction: `Return JSON only: {"X": p}` (W2: a count out of 100).

## Family 5 — Stated distribution for one person (S6, B5, R3)

The model sees the input row (and, for R3, the already-drawn parent outputs) and the full list of allowed answers to one question.

**W1**
> For a person with exactly the attributes above, give the probability of each allowed answer to "{Q}". The probabilities must sum to 1 and must cover every allowed answer, including those that are uncommon. The distribution should reflect how real respondents with these attributes actually answered, not the single most typical answer.

**W2**
> Imagine 100 real respondents from this population with exactly the attributes above. For "{Q}", say how many of the 100 gave each allowed answer. Every allowed answer must have a number (0 is allowed); the numbers must sum to 100.

**W3**
> You are a survey researcher who has studied this population. For respondents with exactly the attributes above, estimate the distribution of answers to "{Q}" as you would expect to find it in the actual survey data: a probability for every allowed answer, summing to 1.

Output instruction: `Return JSON only, one key per allowed answer, exactly as listed.` For numeric outputs every allowed integer (and "No Child" where applicable) is a key.

## Family 6 — Stated two-way table (S7, P4)

The model sees the population and year, the definition of the conditioning category, and the full list of allowed answers to the output. One call per row of the table.

**W1**
> Among {G} in this population in the reference year, what share gave each allowed answer to "{Q}"? Give a probability for every allowed answer, summing to 1. Base this on the actual distribution among such people, including minority answers.

**W2**
> Take 100 respondents from this population in the reference year who are {G}. For "{Q}", how many of the 100 gave each allowed answer? Every allowed answer needs a number (0 allowed); the numbers sum to 100.

**W3**
> You are a survey researcher reporting a cross-tabulation from this survey. For respondents who are {G}, report the row of the table for "{Q}": the proportion giving each allowed answer, summing to 1, as you expect the published cross-tab would show it.

For output × output tables, `{G}` reads "respondents who answered '{y}' to '{Y}'".

Output instruction as Family 5.

---

## Notes for the coding agent

- `{G}` for input groups is built from the config's category labels (age uses the three bands in SPEC §2). The group description names only the conditioning category, never other attributes.
- Labels for candidates (A, B, C …) and for pairs (X, Y) are assigned after shuffling, per call, with a logged seed; the two-order averaging in SPEC §3.2 is on top of this.
- W2 outputs are counts; convert to probabilities by dividing by the stated total and apply the same 0.9–1.1 tolerance on the total (90–110).
- Family 2's W1 is the wording already in SPEC §6.4; they must stay identical.
- If a wording clearly breaks JSON parsing in the first 50 calls (parse-failure rate above 10%), stop and report rather than editing it.

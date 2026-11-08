"""Prompt construction: official framing + pre-registered W1–W3 wordings.

The official framing replicates SSDataBench's ``build_prompt`` verbatim
(population context, reference year, conditioned input attributes, and the
description + allowed values of every field involved). Only the middle
wording block varies across W1–W3; see docs/PROMPT-WORDINGS.md (frozen).
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from .fields import FieldSpec, load_context, load_fields

WORDINGS = ("W1", "W2", "W3")

# --------------------------------------------------------------------------- #
# Official framing (verbatim from simulation/generation_cs.py build_prompt)
# --------------------------------------------------------------------------- #


def _stringify_allowed(fs: FieldSpec) -> str:
    if fs.allowed is not None:
        return "[" + ", ".join(f'"{v}"' for v in fs.allowed) + "]"
    assert fs.numeric is not None
    rng = f"[{fs.numeric.min}, {fs.numeric.max}]"
    if fs.numeric.special:
        special = list(fs.numeric.special)
        return f"number in {rng} OR one of {special}"
    return f"number in {rng}"


def _format_value(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def official_framing(
    inputs: dict[str, object] | None,
    fields: list[FieldSpec],
    context: str | None = None,
    reference_year: int | None = None,
) -> str:
    """The official prompt framing restricted to ``fields``.

    With all 21 outputs and the 9 inputs this reproduces the official
    generation prompt byte-for-byte (verified in tests).
    """
    ctx = load_context()
    context = context if context is not None else ctx["context"]
    reference_year = reference_year if reference_year is not None else ctx["reference_year"]

    desc_lines = [f"- {f.name} (static): {f.description}" for f in fields]
    allowed_lines = [f"- {f.name}: {_stringify_allowed(f)}" for f in fields]
    desc_block = "\n".join(desc_lines)
    allowed_block = "\n".join(allowed_lines)

    input_context = ""
    if inputs:
        ctx_parts = [f"{k}: {_format_value(v)}" for k, v in inputs.items()]
        input_context = (
            "Conditioned on the following input attributes:\n" + ", ".join(ctx_parts) + "\n"
        )

    keys = ", ".join(f.name for f in fields)
    return f"""
You are simulating one *randomly selected* synthetic individual sampled from {context}
The data represent a cross-sectional snapshot collected in the year {reference_year}.
{input_context}Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
{keys}.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
{desc_block}

Allowed values:
{allowed_block}
""".strip()


def official_generation_prompt(inputs: dict[str, object]) -> str:
    """The official B4 generation prompt, verbatim."""
    fields = [f for f in load_fields().values() if f.role == "y"]
    return official_framing(inputs, fields)


def one_question_generation_prompt(
    inputs: dict[str, object], question: str
) -> str:
    """One-question version of the official prompt (Study 1 candidates)."""
    fs = load_fields()[question]
    return official_framing(inputs, [fs])


# --------------------------------------------------------------------------- #
# Candidate labels (shuffled per call with a logged seed; SPEC §1.5)
# --------------------------------------------------------------------------- #


@dataclass
class Labelled:
    labels: list[str]                      # label -> candidate order after shuffle
    label_of: dict[int, str]               # original index -> label
    order: list[int]                       # shuffled original indices
    mapping_str: str                       # human-readable mapping for prompts


def label_candidates(n: int, seed: int) -> Labelled:
    order = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(order)
    letters = list(string.ascii_uppercase)
    labels = [letters[i] for i in range(n)]
    label_of = {orig: labels[pos] for pos, orig in enumerate(order)}
    mapping_str = ", ".join(f"{label_of[i]}={i}" for i in order)
    return Labelled(labels=labels, label_of=label_of, order=order, mapping_str=mapping_str)


def candidate_block(candidates: list[str], lab: Labelled) -> str:
    lines = [
        f"{lab.label_of[i]}: {candidates[i]}" for i in lab.order
    ]
    return "\n".join(lines)


def profile_block(
    profiles: list[dict[str, object]], keys: list[str], lab: Labelled
) -> str:
    import json

    lines = []
    for i in lab.order:
        obj = {k: _format_value(profiles[i][k]) for k in keys}
        lines.append(f'{lab.label_of[i]}: {json.dumps(obj, ensure_ascii=False)}')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Wording families (docs/PROMPT-WORDINGS.md, frozen)
# --------------------------------------------------------------------------- #

OUT_JSON_LABELS = 'Return JSON only, one key per label, e.g. {"A": 0.4, "B": 0.6}.'
OUT_JSON_ALLOWED = "Return JSON only, one key per allowed answer, exactly as listed."
OUT_JSON_PLAUSIBLE = (
    'Return JSON only, one key per label with the value "plausible" or "implausible".'
)
OUT_JSON_PAIR = 'Return JSON only: {"X": p} (W2: a count out of 100).'

# Family 1 — Judge call on candidate values (Study 1, call J)
FAMILY1 = {
    "W1": (
        "Below are {N} possible answers to the question \"{Q}\" for a person with exactly "
        "the attributes above. For each answer, give the probability that a real, randomly "
        "selected respondent with these attributes in this population gave that answer. The "
        "probabilities over the listed answers must sum to 1. Do not favour an answer because "
        "it seems typical; favour it only if it is actually more common among such people."
    ),
    "W2": (
        "Imagine 100 real respondents from this population who all have exactly the attributes "
        "above. Below are {N} possible answers to the question \"{Q}\". For each answer, say "
        "how many of the 100 would have given it. The numbers must sum to 100. Answers that "
        "only some of these people would give should get the share those people represent, "
        "not zero."
    ),
    "W3": (
        "You are a survey researcher who has studied this population closely. For a respondent "
        "with exactly the attributes above, the question \"{Q}\" has {N} candidate answers "
        "listed below. Based on what you know about how such people actually answered, "
        "estimate for each candidate the probability that it is this respondent's real answer. "
        "The probabilities must sum to 1."
    ),
}

# Family 2 — Judge call on whole candidate profiles (Study 2: P1, P2, P3, P5)
FAMILY2 = {
    "W1": (
        "For people with exactly these input attributes in this population, how likely is each "
        "profile below to be the answers of a real, randomly selected respondent? Give a "
        "probability for each profile; the probabilities must sum to 1. Judge each profile as a "
        "whole: whether its answers fit the inputs and fit each other."
    ),
    "W2": (
        "Imagine 100 real respondents from this population with exactly these input attributes. "
        "Each of the {N} profiles below is one possible full set of answers. For each profile, "
        "say how many of the 100 respondents would have a set of answers like it — considering "
        "both whether the answers fit the person's attributes and whether they are consistent "
        "with one another. The numbers must sum to 100."
    ),
    "W3": (
        "You are a survey researcher checking synthetic records against what you know of this "
        "population. Each of the {N} profiles below is a candidate record for one respondent "
        "with exactly these input attributes. Estimate for each the probability that it is the "
        "real respondent's record rather than an implausible one, paying attention to internal "
        "consistency across answers as well as fit with the attributes. The probabilities must "
        "sum to 1."
    ),
}

# Family 3 — Plausibility filter (rule S3)
FAMILY3 = {
    "W1": (
        "For a person with exactly the attributes above, mark each listed answer as "
        '"plausible" if a real respondent with these attributes could reasonably have given '
        'it, or "implausible" if it would be very unusual or contradictory. Mark an answer '
        "implausible only when you are confident; uncommon is not the same as implausible."
    ),
    "W2": (
        "Among 100 real respondents with exactly these attributes, which of the listed answers "
        'would at least a few of them have given? Mark those "plausible". Mark an answer '
        '"implausible" only if essentially none of the 100 would have given it.'
    ),
    "W3": (
        "As a survey researcher who knows this population, screen the listed answers for a "
        'respondent with exactly these attributes. Mark each "plausible" or "implausible", '
        "where implausible means you would flag it as a likely data error if it appeared in a "
        "real record."
    ),
}

# Family 4 — Pairwise judgment (rule S5)
FAMILY4 = {
    "W1": (
        "For a person with exactly the attributes above, which of the two answers below is "
        'more likely to be a real respondent\'s answer to "{Q}"? Give the probability that X '
        "is the real one rather than Y."
    ),
    "W2": (
        "Among 100 real respondents with exactly these attributes, suppose each gave either "
        'answer X or answer Y to "{Q}". How many gave X? Return a number from 0 to 100.'
    ),
    "W3": (
        "As a survey researcher who knows this population, compare the two candidate answers "
        'to "{Q}" for a respondent with exactly these attributes. Estimate the probability '
        "that X, rather than Y, is the respondent's actual answer."
    ),
}

# Family 5 — Stated distribution for one person (S6, B5, R3)
FAMILY5 = {
    "W1": (
        'For a person with exactly the attributes above, give the probability of each allowed '
        'answer to "{Q}". The probabilities must sum to 1 and must cover every allowed answer, '
        "including those that are uncommon. The distribution should reflect how real "
        "respondents with these attributes actually answered, not the single most typical "
        "answer."
    ),
    "W2": (
        "Imagine 100 real respondents from this population with exactly the attributes above. "
        'For "{Q}", say how many of the 100 gave each allowed answer. Every allowed answer '
        "must have a number (0 is allowed); the numbers must sum to 100."
    ),
    "W3": (
        "You are a survey researcher who has studied this population. For respondents with "
        'exactly the attributes above, estimate the distribution of answers to "{Q}" as you '
        "would expect to find it in the actual survey data: a probability for every allowed "
        "answer, summing to 1."
    ),
}

# Family 6 — Stated two-way table (S7, P4)
FAMILY6 = {
    "W1": (
        "Among {G} in this population in the reference year, what share gave each allowed "
        'answer to "{Q}"? Give a probability for every allowed answer, summing to 1. Base '
        "this on the actual distribution among such people, including minority answers."
    ),
    "W2": (
        "Take 100 respondents from this population in the reference year who are {G}. For "
        '"{Q}", how many of the 100 gave each allowed answer? Every allowed answer needs a '
        "number (0 allowed); the numbers sum to 100."
    ),
    "W3": (
        "You are a survey researcher reporting a cross-tabulation from this survey. For "
        'respondents who are {G}, report the row of the table for "{Q}": the proportion giving '
        "each allowed answer, summing to 1, as you expect the published cross-tab would show "
        "it."
    ),
}

FAMILIES = {
    1: FAMILY1, 2: FAMILY2, 3: FAMILY3, 4: FAMILY4, 5: FAMILY5, 6: FAMILY6,
}


def _join(framing: str, wording: str, candidates: str | None, output_instr: str) -> str:
    parts = [framing, wording]
    if candidates:
        parts.append(candidates)
    parts.append(output_instr)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Prompt builders. Each returns (prompt, meta) where meta records the label
# mapping and shuffle seed so nothing about the call is lost.
# --------------------------------------------------------------------------- #


def judge_values_prompt(
    inputs: dict[str, object],
    question: str,
    distinct_values: list[str],
    wording: str,
    seed: int,
) -> tuple[str, dict]:
    """Family 1: judge candidate values for one question (call J)."""
    fs = load_fields()[question]
    framing = official_framing(inputs, [fs])
    lab = label_candidates(len(distinct_values), seed)
    w = FAMILY1[wording].format(N=len(distinct_values), Q=fs.description)
    cand = "Candidate answers:\n" + candidate_block(distinct_values, lab)
    instr = OUT_JSON_LABELS if wording != "W2" else (
        'Return JSON only, one key per label with a count out of 100, e.g. {"A": 40, "B": 60}.'
    )
    prompt = _join(framing, w, cand, instr)
    meta = {"family": 1, "wording": wording, "label_of": lab.label_of, "seed": seed}
    return prompt, meta


def judge_profiles_prompt(
    inputs: dict[str, object],
    profiles: list[dict[str, object]],
    wording: str,
    seed: int,
    output_fields: list[str] | None = None,
) -> tuple[str, dict]:
    """Family 2: judge whole candidate profiles (P1, P2, P3, P5)."""
    fields = load_fields()
    keys = output_fields or [f.name for f in fields.values() if f.role == "y"]
    fs = [fields[k] for k in keys]
    framing = official_framing(inputs, fs)
    lab = label_candidates(len(profiles), seed)
    w = FAMILY2[wording].format(N=len(profiles))
    cand = "Candidate profiles:\n" + profile_block(profiles, keys, lab)
    instr = OUT_JSON_LABELS if wording != "W2" else (
        'Return JSON only, one key per label with a count out of 100, e.g. {"A": 40, "B": 60}.'
    )
    prompt = _join(framing, w, cand, instr)
    meta = {"family": 2, "wording": wording, "label_of": lab.label_of, "seed": seed}
    return prompt, meta


def plausibility_prompt(
    inputs: dict[str, object],
    question: str,
    distinct_values: list[str],
    wording: str,
    seed: int,
) -> tuple[str, dict]:
    """Family 3: plausible / implausible filter (rule S3)."""
    fs = load_fields()[question]
    framing = official_framing(inputs, [fs])
    lab = label_candidates(len(distinct_values), seed)
    w = FAMILY3[wording].format(Q=fs.description)
    cand = "Candidate answers:\n" + candidate_block(distinct_values, lab)
    prompt = _join(framing, w, cand, OUT_JSON_PLAUSIBLE)
    meta = {"family": 3, "wording": wording, "label_of": lab.label_of, "seed": seed}
    return prompt, meta


def pairwise_prompt(
    inputs: dict[str, object],
    question: str,
    x_value: str,
    y_value: str,
    wording: str,
    seed: int,
) -> tuple[str, dict]:
    """Family 4: pairwise judgment (rule S5). X/Y labels shuffled by seed."""
    fs = load_fields()[question]
    framing = official_framing(inputs, [fs])
    rng = random.Random(int(seed))
    xy = ["X", "Y"] if rng.random() < 0.5 else ["Y", "X"]
    vals = {"X": x_value, "Y": y_value}
    block = f"{xy[0]}: {vals[xy[0]]}\n{xy[1]}: {vals[xy[1]]}"
    w = FAMILY4[wording].format(Q=fs.description)
    prompt = _join(framing, w, block, OUT_JSON_PAIR)
    meta = {"family": 4, "wording": wording, "xy_order": xy, "seed": seed}
    return prompt, meta


def state_distribution_prompt(
    inputs: dict[str, object],
    question: str,
    wording: str,
    given: dict[str, object] | None = None,
) -> tuple[str, dict]:
    """Family 5: state the full distribution for one person (S6, B5, R3).

    ``given`` (R3): already-drawn parent outputs, appended to the input
    attributes block in framing order. Allowed answers are enumerated
    explicitly (every integer for numeric outputs, SPEC §2).
    """
    fs = load_fields()[question]
    merged: dict[str, object] = dict(inputs)
    if given:
        merged.update(given)
    framing = official_framing(merged, [fs])
    w = FAMILY5[wording].format(Q=fs.description)
    answers = answers_block(fs)
    instr = OUT_JSON_ALLOWED
    if fs.numeric is not None and fs.numeric.special:
        instr += ' Numeric answers are the integers as listed; include the special labels where applicable.'
    prompt = _join(framing, w, answers, instr)
    meta = {"family": 5, "wording": wording, "question": question}
    return prompt, meta


def answers_block(fs: FieldSpec) -> str:
    """Explicit enumeration of every allowed answer, to be used as JSON keys."""
    items = ", ".join(f'"{v}"' for v in fs.answers)
    return f'Allowed answers (use exactly these as the JSON keys): [{items}]'


def state_table_prompt(
    question: str,
    group_description: str,
    wording: str,
) -> tuple[str, dict]:
    """Family 6: one row of a stated two-way table (S7, P4).

    ``group_description``: e.g. "women aged 35-54" (input groups) or
    "respondents who answered 'Agree' to 'gender_role_attitude'" (output
    conditioning). Names only the conditioning category (PROMPT-WORDINGS notes).
    """
    fs = load_fields()[question]
    framing = official_framing(None, [fs])
    w = FAMILY6[wording].format(G=group_description, Q=fs.description)
    prompt = _join(framing, w, answers_block(fs), OUT_JSON_ALLOWED)
    meta = {"family": 6, "wording": wording, "question": question, "group": group_description}
    return prompt, meta


# --------------------------------------------------------------------------- #
# B6: population-level batch generation (single official wording, SPEC §6.1)
# --------------------------------------------------------------------------- #


def b6_batch_prompt(input_rows: list[dict[str, object]]) -> tuple[str, dict]:
    """Official framing for 10 input rows; asks for 10 profiles in one JSON
    array, reflecting the real variation among such people."""
    fields = [f for f in load_fields().values() if f.role == "y"]
    framing = official_framing(None, fields)
    lines = [
        f"{i + 1}. " + ", ".join(f"{k}: {_format_value(v)}" for k, v in row.items())
        for i, row in enumerate(input_rows)
    ]
    block = "Here are the input attributes of 10 respondents:\n" + "\n".join(lines)
    instr = (
        "Simulate one *randomly selected* synthetic individual for EACH of the 10 "
        "respondents listed above. Return a JSON array of EXACTLY 10 objects, in the "
        "same order as the respondents; each object contains EXACTLY these keys:\n"
        f"{', '.join(f.name for f in fields)}.\n"
        "The 10 individuals should reflect the real variation among such people: "
        "differ across respondents and differ from each other in the way real "
        "respondents in this population differ, not 10 copies of the most typical "
        "person."
    )
    prompt = framing + "\n\n" + block + "\n\n" + instr
    return prompt, {"family": "b6", "n_rows": len(input_rows)}


# --------------------------------------------------------------------------- #
# R3: dependency graph proposal (single structural call, SPEC §6.2)
# --------------------------------------------------------------------------- #


def judge_profiles_with_tables(
    inputs: dict[str, object],
    profiles: list[dict[str, object]],
    current_summary: str,
    stated_target: str,
    seed: int,
) -> tuple[str, dict]:
    """P5-view judge (single wording W1): candidates + current population
    tables + stated target tables; asked which candidate to add next."""
    from .prompts import label_candidates, profile_block  # noqa: PLC0415

    fields = [f for f in load_fields().values() if f.role == "y"]
    framing = official_framing(inputs, fields)
    lab = label_candidates(len(profiles), seed)
    keys = [f.name for f in fields]
    block = profile_block(profiles, keys, lab)
    w = (
        "You are adding respondents to a synthetic dataset one at a time. Below are "
        "candidate profiles for this respondent, the dataset built so far (current "
        "distributions within this respondent's groups), and the target distributions "
        "the finished dataset should match. Judge each candidate both as a plausible "
        "real respondent and by whether adding it moves the dataset toward the target. "
        "Give a probability for each candidate; they must sum to 1."
    )
    cur = "Dataset built so far, within this respondent's input groups:\n" + current_summary
    tgt = "Target distributions (stated):\n" + stated_target
    instr = 'Return JSON only, one key per label, e.g. {"A": 0.4, "B": 0.6}.'
    prompt = framing + "\n\n" + w + "\n\n" + cur + "\n\n" + tgt + "\n\nCandidate profiles:\n" + block + "\n\n" + instr
    meta = {"family": "p5view", "label_of": lab.label_of, "seed": seed}
    return prompt, meta


def r3_graph_prompt() -> tuple[str, dict]:
    """List the 21 outputs; ask for up to 3 parent outputs each."""
    fields = [f for f in load_fields().values() if f.role == "y"]
    names = [f.name for f in fields]
    desc_lines = [f"- {f.name}: {f.description}" for f in fields]
    prompt = (
        "You are describing how answers in a survey interview depend on each other "
        "for one respondent. The survey covers the following questions:\n"
        + "\n".join(desc_lines)
        + "\n\nFor each question, list up to 3 OTHER questions whose answers "
        "directly influence it for the same person (e.g. spouse_occupation depends "
        "on marital_status). Use only questions from the list. Prefer the fewest, "
        "strongest dependencies; use an empty list when an answer does not depend "
        "on other answers. Do not create circular dependencies.\n"
        "Return JSON only, one key per question, e.g.\n"
        '{"marital_status": [], "spouse_occupation": ["marital_status"]}'
    )
    return prompt, {"family": "r3_graph", "outputs": names}

"""Study 1 readout rules S0–S7 on one question (SPEC §4).

Everything expensive (candidates, judge calls, tables, stated distributions)
is made once per wording and cached; selections are redrawn per run seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .fields import input_group_of, input_group_values, load_fields
from .llm import LLMClient
from .loglinear import build_design_matrix, fit_loglinear
from .parse import DistResult, ParseError, parse_distribution, parse_labels, parse_single_probability, uniform
from .prompts import (
    judge_values_prompt,
    pairwise_prompt,
    plausibility_prompt,
    one_question_generation_prompt,
    state_distribution_prompt,
)
from .tables import stated_input_tables

SYSTEM_JSON = "You are a strict JSON generator. Output valid JSON only."


# --------------------------------------------------------------------------- #
# Calls (cached; once per wording/seed)
# --------------------------------------------------------------------------- #


def _row_inputs(row: pd.Series, input_names: list[str]) -> dict[str, object]:
    return {k: row[k] for k in input_names}


def generate_candidate_values(
    client: LLMClient,
    model: str,
    rows: pd.DataFrame,
    question: str,
    n_candidates: int,
    seed: int,
    input_names: list[str],
) -> list[list[str]]:
    """n_candidates one-question generations per row (official framing)."""
    from .llm import run_parallel

    fs = load_fields()[question]

    def one_row(_i, item):
        _, row = item
        prompt = one_question_generation_prompt(_row_inputs(row, input_names), question)
        cands: list[str] = []
        for j in range(n_candidates):
            res = client.chat(
                [
                    {"role": "system", "content": SYSTEM_JSON},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                json_mode=True,
                seed=seed + 10_000 * int(row["profile_id"]) + j,
                meta={"row": int(row["profile_id"]), "question": question, "j": j},
                purpose="candidate_value",
            )
            cands.append(_validate_answer(res.content, fs))
        return cands

    items = list(rows.iterrows())
    return run_parallel(one_row, items, max_workers=client.max_workers, desc="candidates")


def _validate_answer(raw: str, fs) -> str:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    v = obj.get(fs.name)
    if v is None:
        return ""
    s = str(v).strip()
    if fs.allowed is not None:
        low = {a.lower(): a for a in fs.allowed}
        return low.get(s.lower(), "")
    # numeric: integers or special labels
    try:
        iv = int(round(float(s)))
        if fs.numeric.min <= iv <= fs.numeric.max:
            return str(iv)
    except ValueError:
        pass
    if fs.numeric is not None:
        low = {a.lower(): a for a in fs.numeric.special}
        return low.get(s.lower(), "")
    return ""


def judge_values(
    client: LLMClient,
    judge_model: str,
    row: pd.Series,
    question: str,
    distinct_values: list[str],
    wording: str,
    seed: int,
    input_names: list[str],
    n_orders: int = 2,
    max_attempts: int = 3,
) -> dict[str, float]:
    """Call J: judge probabilities over distinct values, two orders averaged.

    Each order is asked up to ``max_attempts`` times (fresh shuffle each) and
    falls back to uniform over the labels on persistent failure.
    """
    probs: dict[str, list[float]] = {v: [] for v in distinct_values}
    for o in range(n_orders):
        dist = None
        last_err = ""
        for attempt in range(max_attempts):
            a_seed = seed + 97 * o + 977 * attempt
            prompt, meta = judge_values_prompt(
                _row_inputs(row, input_names), question, distinct_values, wording,
                seed=a_seed,
            )
            res = client.chat(
                [
                    {"role": "system", "content": SYSTEM_JSON},
                    {"role": "user", "content": prompt},
                ],
                model=judge_model,
                json_mode=True,
                seed=a_seed,
                meta={**meta, "row": int(row["profile_id"]), "order": o, "attempt": attempt},
                purpose="judge_values",
            )
            labels = list(meta["label_of"].values())
            try:
                dist = parse_distribution(res.content, labels, counts_mode=(wording == "W2"))
                break
            except ParseError as e:
                last_err = str(e)
        if dist is None:
            labels_now = list(meta["label_of"].values())
            dist = uniform(labels_now)
        label_of = meta["label_of"]  # original index -> label
        for idx_str, label in label_of.items():
            v = distinct_values[int(idx_str)]
            probs[v].append(dist.probs[label])
    return {v: float(np.mean(ps)) for v, ps in probs.items()}


def plausibility_filter(
    client: LLMClient,
    judge_model: str,
    row: pd.Series,
    question: str,
    distinct_values: list[str],
    wording: str,
    seed: int,
    input_names: list[str],
) -> dict[str, str]:
    prompt, meta = plausibility_prompt(
        _row_inputs(row, input_names), question, distinct_values, wording, seed
    )
    res = client.chat(
        [
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": prompt},
        ],
        model=judge_model,
        json_mode=True,
        seed=seed,
        meta={**meta, "row": int(row["profile_id"])},
        purpose="plausibility",
    )
    try:
        parsed = parse_labels(res.content, list(meta["label_of"].values()),
                              ("plausible", "implausible"))
        inv = {lab: int(i) for i, lab in meta["label_of"].items()}
        return {distinct_values[inv[lab]]: verdict for lab, verdict in parsed.items()}
    except (ParseError, KeyError):
        return {v: "plausible" for v in distinct_values}  # filter no-op on failure


def pairwise_probability(
    client: LLMClient,
    judge_model: str,
    row: pd.Series,
    question: str,
    x_value: str,
    y_value: str,
    wording: str,
    seed: int,
    input_names: list[str],
) -> float:
    """P(X rather than Y fits a real respondent), both XY/YX orders averaged."""
    probs = []
    for o in range(2):
        prompt, meta = pairwise_prompt(
            _row_inputs(row, input_names), question, x_value, y_value, wording,
            seed=seed + 131 * o,
        )
        res = client.chat(
            [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": prompt},
            ],
            model=judge_model,
            json_mode=True,
            seed=seed + 131 * o,
            meta={**meta, "row": int(row["profile_id"])},
            purpose="pairwise",
        )
        try:
            p = parse_single_probability(res.content, "X", counts_mode=(wording == "W2"))
        except ParseError:
            p = 0.5
        if meta["xy_order"][0] == "Y":
            p = 1.0 - p
        probs.append(p)
    return float(np.clip(np.mean(probs), 0.0, 1.0))


def state_distribution(
    client: LLMClient,
    model: str,
    row: pd.Series,
    question: str,
    wording: str,
    seed: int,
    input_names: list[str],
    given: dict[str, object] | None = None,
    max_attempts: int = 3,
) -> DistResult:
    fs = load_fields()[question]
    prompt, meta = state_distribution_prompt(
        _row_inputs(row, input_names), question, wording, given=given
    )
    last = ""
    for attempt in range(1, max_attempts + 1):
        res = client.chat(
            [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": prompt},
            ],
            model=model,
            json_mode=True,
            seed=seed + attempt if attempt > 1 else seed,
            meta={**meta, "row": int(row["profile_id"]), "attempt": attempt},
            purpose="state_distribution",
        )
        try:
            return parse_distribution(res.content, fs.answers, counts_mode=(wording == "W2"))
        except ParseError as e:
            last = str(e)
    return uniform(fs.answers)


def s7_row_distributions(
    client: LLMClient,
    model: str,
    question: str,
    rows: pd.DataFrame,
    input_names: list[str],
    wording: str,
    seed: int,
) -> tuple[list[dict[str, float]], dict]:
    """S7: stated question × input tables -> log-linear fit -> p(y | x_row)."""
    fs = load_fields()[question]
    tables, flags = stated_input_tables(
        client, model, question, rows, input_names, wording, seed
    )
    # design matrix over the row set
    rows_inputs = [
        {k: input_group_of(k, row[k]) for k in input_names} for _, row in rows.iterrows()
    ]
    from .fields import input_group_values_with_missing  # noqa: PLC0415

    cats = {k: input_group_values_with_missing(k, rows) for k in input_names}
    X, features = build_design_matrix(rows_inputs, input_names, cats)

    memberships = []
    for iname in input_names:
        for cat, dist in tables[iname].items():
            idx = np.array(
                [i for i, r in enumerate(rows_inputs) if r[iname] == cat], dtype=int
            )
            if len(idx) == 0:
                continue
            w = len(idx) / len(rows)
            s = np.array([dist[a] for a in fs.answers])
            memberships.append((s, idx, w))
    fit = fit_loglinear(X, memberships, fs.answers)
    P = fit.predict_proba(X)
    dists = [
        {a: float(p) for a, p in zip(fs.answers, P[i], strict=True)}
        for i in range(len(rows))
    ]
    info = {**flags, "mad": fit.mad, "loss": fit.loss, "n_iter": fit.n_iter}
    return dists, info


# --------------------------------------------------------------------------- #
# Rule application (pure selection; redrawn per run)
# --------------------------------------------------------------------------- #


@dataclass
class RuleInputs:
    """Cached expensive pieces for one row, question, generator, wording."""
    candidates: list[str]                       # 10 raw (may repeat / include "")
    j_probs: dict[str, float]                   # averaged judge prob per distinct value
    plausible: dict[str, str]                   # S3 verdicts
    pairwise: dict[tuple[str, str], float]      # S5 cached comparisons (x,y)->p
    stated: dict[str, float]                    # S6 distribution
    s7_dist: dict[str, float]                   # S7 distribution
    distinct: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.distinct = [v for v in dict.fromkeys(self.candidates) if v != ""]


def _weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    vals = [v for v, _ in items]
    ws = np.array([max(w, 0.0) for _, w in items])
    if ws.sum() <= 0:
        return rng.choice(vals)
    probs = ws / ws.sum()
    return str(rng.choices(vals, weights=probs.tolist(), k=1)[0])


def apply_rule(
    rule: str,
    ri: RuleInputs,
    rng: random.Random,
    pair_fetch=None,
) -> str:
    """One answer for one row under a rule; rng makes runs independent.

    ``pair_fetch(x, y) -> float`` lazily obtains (and caches) the judge's
    P(x rather than y) for rule S5; without it, cached pairs or 0.5 are used.
    """
    cands = [c for c in ri.candidates if c != ""]
    if not cands:
        cands = [""]  # generator produced nothing valid; propagate
    counts: dict[str, int] = {}
    for c in ri.candidates:
        if c != "":
            counts[c] = counts.get(c, 0) + 1
    dist = ri.distinct if ri.distinct else cands

    if rule == "S0":
        return rng.choice(cands)

    if rule == "S1":
        items = [(v, counts.get(v, 0) * ri.j_probs.get(v, 0.0)) for v in dist]
        return _weighted_choice(rng, items)

    if rule == "S2":
        best = max(dist, key=lambda v: ri.j_probs.get(v, 0.0))
        ties = [v for v in dist if abs(ri.j_probs.get(v, 0.0) - ri.j_probs.get(best, 0.0)) < 1e-12]
        return rng.choice(ties)

    if rule == "S3":
        plaus = [v for v in dist if ri.plausible.get(v, "plausible") == "plausible"]
        pool = [c for c in cands if c in plaus]
        return rng.choice(pool) if pool else rng.choice(cands)

    if rule == "S4":
        items = [(v, ri.j_probs.get(v, 0.0)) for v in dist]
        return _weighted_choice(rng, items)

    if rule == "S5":
        # walk the 10 candidates in this cell's shuffle; skip same-value; the
        # current pick is replaced with the judge's pairwise probability.
        order = [c for c in ri.candidates if c != ""]
        rng.shuffle(order)
        if not order:
            return ""
        current = order[0]
        for nxt in order[1:]:
            if nxt == current:
                continue
            p = _pair_prob(current, nxt, ri, pair_fetch)
            if rng.random() < p:
                current = nxt
        return current

    if rule == "S6":
        return _weighted_choice(rng, list(ri.stated.items()))

    if rule == "S7":
        return _weighted_choice(rng, list(ri.s7_dist.items()))

    raise ValueError(f"unknown rule {rule}")


def _pair_prob(x: str, y: str, ri: RuleInputs, pair_fetch) -> float:
    """P(x rather than y), from the cache, the reverse, a call, or 0.5."""
    if (x, y) in ri.pairwise:
        return ri.pairwise[(x, y)]
    if (y, x) in ri.pairwise:
        return 1.0 - ri.pairwise[(y, x)]
    if pair_fetch is not None:
        p = pair_fetch(x, y)
        ri.pairwise[(x, y)] = p
        return p
    return 0.5


def cell_rng(seed: int, rule: str, row_i: int) -> random.Random:
    """Deterministic per-(run-seed, rule, row) generator, so every draw is
    reproducible and independent of execution order."""
    r_i = RULES.index(rule)
    h = (seed * 1_000_003 + r_i * 10_007 + row_i * 7_919) % 2**31
    return random.Random(h)


RULES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")

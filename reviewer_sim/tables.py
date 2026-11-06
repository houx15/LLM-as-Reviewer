"""Stated two-way tables: calls, storage, distances (SPEC §3.2, §4.2 S7).

One call per row of a table: the prompt names the population and year, the
input group (or conditioning output's category), and asks for the share of
each allowed answer. Rows are validated as distributions (parse.py) and stored
as JSON under data/stated_tables/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fields import FieldSpec, input_group_of, input_group_values, load_fields
from .llm import LLMClient
from .parse import DistResult, ParseError, parse_distribution, uniform
from .prompts import state_table_prompt


def group_phrase(input_name: str, category: str) -> str:
    """Human phrase for an input group category, e.g. 'aged 35-54', 'women'."""
    if input_name == "age":
        return f"respondents aged {category}"
    if input_name == "gender":
        return f"{category.lower()} respondents"
    return f"respondents whose {input_name.replace('_', ' ')} is {category}"


def output_group_phrase(cond_question: str, cond_answer: str) -> str:
    fs = load_fields()[cond_question]
    return f"respondents who answered '{cond_answer}' to '{fs.description}'"


def ask_table_row(
    client: LLMClient,
    model: str,
    question: str,
    group_description: str,
    wording: str,
    seed: int,
    max_attempts: int = 3,
) -> tuple[DistResult, dict]:
    """One row of a stated two-way table, with retries and uniform fallback."""
    fs = load_fields()[question]
    prompt, meta = state_table_prompt(question, group_description, wording)
    messages = [
        {"role": "system", "content": "You are a strict JSON generator. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        res = client.chat(
            messages,
            model=model,
            temperature=1.0,
            json_mode=True,
            seed=seed + attempt if attempt > 1 else seed,
            meta={**meta, "group": group_description, "attempt": attempt},
            purpose="table_row",
        )
        try:
            dist = parse_distribution(
                res.content, fs.answers, counts_mode=(wording == "W2")
            )
            dist.raw_sum = dist.raw_sum
            return dist, {**meta, "attempt": attempt, "fallback": False}
        except ParseError as e:
            last_err = str(e)
    return uniform(fs.answers), {**meta, "fallback": True, "error": last_err}


def stated_input_tables(
    client: LLMClient,
    model: str,
    question: str,
    rows: "object",  # DataFrame with input columns
    input_names: list[str],
    wording: str,
    seed: int,
    only_categories: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict]:
    """question × each input: one call per (input, category present in rows)."""
    tables: dict[str, dict[str, dict[str, float]]] = {}
    flags = {"fallback": 0, "renormalized": 0, "calls": 0}
    for k_i, iname in enumerate(input_names):
        if only_categories:
            cats = only_categories[iname]
        else:
            present = {input_group_of(iname, v) for v in rows[iname]}
            # keep config order, then Missing last
            ordered = [c for c in input_group_values(iname) if c in present]
            if "Missing" in present:
                ordered.append("Missing")
            cats = ordered
        tables[iname] = {}
        for c_i, cat in enumerate(cats):
            seed_k = seed + 1000 * k_i + c_i
            dist, meta = ask_table_row(
                client, model, question,
                "respondents whose answer is missing" if cat == "Missing"
                else group_phrase(iname, cat),
                wording, seed_k,
            )
            tables[iname][cat] = dist.probs
            flags["calls"] += 1
            flags["fallback"] += int(dist.fallback)
            flags["renormalized"] += int(dist.renormalized)
    return tables, flags


def save_tables(tables: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, indent=1, ensure_ascii=False))


def load_tables(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def table_distance(t1: dict[str, float], t2: dict[str, float]) -> float:
    """Total variation distance between two distributions over the same keys."""
    keys = sorted(set(t1) | set(t2))
    a = np.array([t1.get(k, 0.0) for k in keys])
    b = np.array([t2.get(k, 0.0) for k in keys])
    return 0.5 * float(np.abs(a - b).sum())

"""B4: the official generation condition (SPEC §6.1).

One official prompt per row, temperature 1.0, JSON mode, official validation
and retry-on-empty loop. Writes sim_profiles in the official format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..fields import input_names, load_fields, output_names
from ..llm import LLMClient, run_parallel
from ..prompts import official_generation_prompt
from ..rules import SYSTEM_JSON, _validate_answer


def generate_b4(
    client: LLMClient,
    model: str,
    rows: pd.DataFrame,
    seed: int,
    max_retries: int = 3,
    n_profiles: int = 1,
) -> pd.DataFrame:
    """n_profiles validated profiles per row (default 1 = official B4).

    Returns one row per (row, draw) with a ``draw`` column when n_profiles > 1.
    """
    fs = load_fields()
    in_names = input_names()

    def one_row(_i, item):
        _, row = item
        inputs = {k: row[k] for k in in_names if pd.notna(row[k])}
        prompt = official_generation_prompt(inputs)
        draws = []
        for j in range(n_profiles):
            rec: dict[str, str] = {}
            for attempt in range(max_retries + 1):
                a_seed = seed + 10_000 * int(row["profile_id"]) + j * 100 + attempt
                res = client.chat(
                    [
                        {"role": "system", "content": SYSTEM_JSON},
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    json_mode=True,
                    seed=a_seed,
                    meta={"row": int(row["profile_id"]), "j": j, "attempt": attempt},
                    purpose="b4_profile",
                )
                try:
                    obj = json.loads(res.content)
                except json.JSONDecodeError:
                    obj = {}
                rec = {}
                for q in output_names():
                    v = obj.get(q)
                    rec[q] = (
                        _validate_answer(json.dumps({q: v}), fs[q]) if v is not None else ""
                    )
                if all(v != "" for v in rec.values()):
                    break
            draws.append(rec)
        return draws

    all_rows = run_parallel(one_row, list(rows.iterrows()),
                            max_workers=client.max_workers, desc=f"b4/{model}")
    in_cols = [c for c in rows.columns if c in set(input_names()) | {"resp_id"}]
    out = []
    for i, draws in enumerate(all_rows):
        for j, rec in enumerate(draws):
            base = {c: rows.iloc[i][c] for c in in_cols}
            base["profile_id"] = rows.iloc[i]["profile_id"]
            if n_profiles > 1:
                base["draw"] = j
            out.append({**base, **rec})
    return pd.DataFrame(out)

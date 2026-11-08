"""Finish Phase 2: P5-view + Study 3, appending to the existing results.

Reconstructs P1's pool and judge probabilities from the cache (deterministic
seeds), then runs the sequential P5-view builder and Study 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import devdata
from ..conditions import phase2 as p2
from ..conditions.study3 import run_study3
from ..fields import input_group_of, scored_input_names
from ..llm import LLMClient

REPO = Path(__file__).resolve().parents[2]


def finish_phase2(
    judge_model: str,
    out_root: str | Path = REPO / "runs" / "phase2",
    cache_root: str | Path = REPO / "runs" / "phase0" / "cache",
) -> dict:
    out_root = Path(out_root)
    rows = devdata.load_rows("pilot")
    real_csv = devdata.build_dev_split()["pilot"]
    client = LLMClient(
        cache_path=Path(cache_root) / "cache.jsonl",
        log_path=Path(cache_root) / "calls.jsonl",
    )

    own_pool = p2._generate_pool(client, judge_model, rows, 5, 930_000, "p1_own_candidate")
    j_own = p2._judge_pool(client, judge_model, rows, own_pool, "W1", 960_000)
    stated = json.loads(
        (REPO / "data" / "stated_tables" / "W1" / "tables.json").read_text()
    )["stated_tables"]
    rows_inputs = [
        {k: input_group_of(k, rows.iloc[i][k]) for k in scored_input_names()}
        for i in range(len(rows))
    ]

    resv = p2.run_p5_view(
        client, judge_model, rows, own_pool, j_own, stated, rows_inputs,
        real_csv, out_root / "P5-view", "W1", 991_000,
    )

    b4 = pd.read_csv(REPO / "runs" / "phase1" / "b4" / "b4_profiles.csv")
    s3 = run_study3(client, judge_model, out_root / "study3", b4)

    prev = pd.read_csv(out_root / "phase2_results.csv")
    combined = pd.concat([prev, resv], ignore_index=True)
    combined.to_csv(out_root / "phase2_results.csv", index=False)
    return {"p5view": resv, "study3": s3, "results": combined}

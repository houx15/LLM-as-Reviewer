"""Mock smoke test for Phase 2 conditions (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reviewer_sim import devdata  # noqa: E402
from reviewer_sim.conditions import phase2 as p2  # noqa: E402
from reviewer_sim.fields import load_fields, output_names, scored_input_names  # noqa: E402


class MockClient:
    max_workers = 2

    class _R:
        def __init__(self, content):
            self.content = content
            self.model_version = "mock"
            self.cached = False
            self.usage = {}

    def chat(self, messages, *, model, meta=None, seed=None, purpose="", **_):
        import random  # noqa: PLC0415

        rng = random.Random(seed)
        fs = load_fields()
        if purpose in ("b6_batch", "p1_own_candidate", "p2_small_candidate"):
            n = 10 if purpose == "b6_batch" else 1
            arr = []
            for _ in range(n):
                obj = {q: rng.choice(fs[q].answers) for q in output_names()}
                arr.append(obj)
            return self._R(json.dumps(arr if purpose == "b6_batch" else arr[0]))
        if purpose == "r3_graph":
            return self._R(json.dumps({"spouse_occupation": ["marital_status"]}))
        if purpose in ("judge_profiles", "p5view"):
            n = meta["label_of"] and len(meta["label_of"])
            labs = list(meta["label_of"].values())
            return self._R(json.dumps({lab: round(1 / n, 4) for lab in labs}))
        if purpose == "s3_discrimination":
            return self._R(json.dumps({"A": 0.5}))
        if purpose == "s3_calibration":
            return self._R(json.dumps({"p": 0.5}))
        if purpose == "s3_population":
            return self._R(json.dumps({"A": 0.5, "cells": ["health=Good"]}))
        # state_distribution / table_row / pairwise via generic JSON
        if meta and "question" in meta:
            f = fs[meta["question"]]
            if len(f.answers) <= 12:
                return self._R(json.dumps({a: round(1 / len(f.answers), 4) for a in f.answers}))
            top = f.answers[:5]
            return self._R(json.dumps({a: 0.2 for a in top}))
        return self._R(json.dumps({"A": 0.5}))


@pytest.fixture(scope="module", autouse=True)
def fast_scoring():
    p2.SCORING_B = 2
    yield
    p2.SCORING_B = 100


@pytest.fixture(scope="module")
def small_rows():
    return devdata.load_rows("pilot").head(12).reset_index(drop=True)


def test_b6_mock(tmp_path, small_rows):
    df, base = p2.run_b6(MockClient(), "mock/m", small_rows, devdata.build_dev_split()["pilot"],
                         tmp_path, seed=1, batch_size=6)
    assert len(df) == 3
    assert df["aggregate"].notna().all()


def test_b5_r3_mock(tmp_path, small_rows):
    real = devdata.build_dev_split()["pilot"]
    df5 = p2.run_b5(MockClient(), "mock/m", small_rows, real, tmp_path / "b5", "W1", seed=1)
    assert len(df5) == 3 and df5["aggregate"].notna().all()
    graph = p2.propose_graph(MockClient(), "mock/m", seed=1)
    assert "spouse_occupation" in graph
    df3 = p2.run_r3(MockClient(), "mock/m", small_rows, real, tmp_path / "r3", "W1", seed=1, graph=graph)
    assert len(df3) == 3 and df3["aggregate"].notna().all()


def test_pools_judge_p5_mock(tmp_path, small_rows):
    real = devdata.build_dev_split()["pilot"]
    client = MockClient()
    pools = p2._generate_pool(client, "mock/m", small_rows, 3, 7, "p1_own_candidate")
    assert len(pools) == len(small_rows)
    assert all(isinstance(p, list) for p in pools)
    j = p2._judge_pool(client, "mock/m", small_rows, pools, "W1", 11)
    assert all(abs(sum(x) - 1) < 0.05 for x in j if x)

    df = p2.run_judged_pool(client, "mock/m", small_rows, real, tmp_path, "P1", "W1", 5, pools, j)
    assert len(df) == 3
    dfr = p2.pool_random_contrast(small_rows, pools, real, tmp_path, "P1-random", "W1", 5)
    assert len(dfr) == 3

    # P5 on synthetic stated tables
    fs = load_fields()
    stated = {
        q: {
            iname: {cat: {a: round(1 / len(fs[q].answers), 4) for a in fs[q].answers}
                    for cat in ["g1", "g2"]}
            for iname in scored_input_names()
        }
        for q in output_names()
    }
    rows_inputs = [
        {k: ("g1" if i % 2 else "g2") for k in scored_input_names()}
        for i in range(len(small_rows))
    ]
    df5, info = p2.run_p5(small_rows, pools, j, stated, rows_inputs, real,
                          tmp_path / "p5", "W1", 3, scale_rows=5)
    assert len(df5) == 9  # 3 lambdas x 3 runs
    assert "s_scale" in info
    lam0 = df5[df5["lambda"] == 0]["aggregate"].mean()
    lam5 = df5[df5["lambda"] == 5]["aggregate"].mean()
    assert lam0 == lam0 and lam5 == lam5

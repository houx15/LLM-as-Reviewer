"""Tests: log-linear fit recovery, rule selection mechanics, mock Study 1."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reviewer_sim import devdata, diagnostics, rules  # noqa: E402
from reviewer_sim.conditions.study1 import Study1Config, run_study1  # noqa: E402
from reviewer_sim.fields import load_fields, scored_input_names  # noqa: E402
from reviewer_sim.llm import run_parallel  # noqa: E402
from reviewer_sim.loglinear import build_design_matrix, fit_loglinear  # noqa: E402
from reviewer_sim.parse import DistResult  # noqa: E402


# --------------------------------------------------------------------------- #
# Log-linear fit
# --------------------------------------------------------------------------- #

def test_loglinear_recovers_main_effects():
    """Stated tables generated from a known main-effects model are recovered."""
    rng = np.random.default_rng(0)
    inputs = ["gender", "race"]
    cats = {"gender": ["F", "M"], "race": ["A", "B", "C"]}
    classes = ["Liberal", "Middle", "Conservative"]

    def true_theta(row):
        th = np.zeros((4, 3))
        th[0] = [0.0, 0.3, 0.0]  # intercept
        if row["gender"] == "M":
            th[1] = [0.4, 0.0, -0.2]
        if row["race"] == "B":
            th[2] = [-0.3, 0.1, 0.2]
        if row["race"] == "C":
            th[3] = [0.1, -0.4, 0.5]
        return th

    n = 900
    rows_inputs = [
        {"gender": rng.choice(cats["gender"]), "race": rng.choice(cats["race"])}
        for _ in range(n)
    ]
    X, features = build_design_matrix(rows_inputs, inputs, cats)
    theta_true = np.array([true_theta(r) for r in rows_inputs])  # (n, d, k)
    logits = np.einsum("ndk,nd->nk", theta_true, X)
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(len(classes), p=p) for p in P])

    # stated tables = true model's implied distributions (no noise)
    memberships = []
    for iname in inputs:
        for c in cats[iname]:
            idx = np.array([i for i, r in enumerate(rows_inputs) if r[iname] == c])
            s = P[idx].mean(axis=0)
            memberships.append((s, idx, len(idx) / n))

    fit = fit_loglinear(X, memberships, classes)
    assert fit.mad < 0.01, f"fit MAD too large: {fit.mad}"
    # per-row predictions close to ground truth
    Ph = fit.predict_proba(X)
    assert np.abs(Ph - P).max() < 0.02


# --------------------------------------------------------------------------- #
# Rule mechanics
# --------------------------------------------------------------------------- #

def _ri(cands, j, stated, s7=None):
    return rules.RuleInputs(
        candidates=cands, j_probs=j, plausible={}, pairwise={},
        stated=stated, s7_dist=s7 or stated,
    )


def test_s0_mirrors_candidate_frequencies():
    ri = _ri(["A"] * 8 + ["B", "C"], {}, {})
    rng = random.Random(0)
    draws = [rules.apply_rule("S0", ri, rng) for _ in range(4000)]
    a = draws.count("A") / len(draws)
    assert abs(a - 0.8) < 0.03


def test_s2_greedy_and_s4_frequency_corrected():
    ri = _ri(["A"] * 8 + ["B", "B"], j={"A": 0.3, "B": 0.7}, stated={})
    assert rules.apply_rule("S2", ri, random.Random(1)) == "B"
    rng = random.Random(0)
    draws = [rules.apply_rule("S4", ri, rng) for _ in range(4000)]
    b = draws.count("B") / len(draws)
    assert abs(b - 0.7) < 0.03  # generator frequency divided out


def test_s1_weighted_by_count_times_judge():
    ri = _ri(["A"] * 8 + ["B", "B"], j={"A": 0.5, "B": 0.5}, stated={})
    rng = random.Random(0)
    draws = [rules.apply_rule("S1", ri, rng) for _ in range(4000)]
    b = draws.count("B") / len(draws)
    assert abs(b - 0.2) < 0.03  # (2*0.5)/(8*0.5+2*0.5)


def test_s5_chain_uses_pairwise_cache():
    ri = _ri(["A", "B", "C"], j={}, stated={})
    ri.pairwise = {("A", "B"): 0.9, ("B", "C"): 0.9, ("A", "C"): 0.9}
    rng = random.Random(4)
    # with p=0.9 the chain should usually end on the last distinct challenger
    draws = [rules.apply_rule("S5", ri, rng) for _ in range(500)]
    assert draws.count("A") / len(draws) < 0.15


def test_s6_draws_from_stated():
    ri = _ri(["A"], j={}, stated={"X": 0.25, "Y": 0.75})
    rng = random.Random(0)
    draws = [rules.apply_rule("S6", ri, rng) for _ in range(4000)]
    assert abs(draws.count("Y") / len(draws) - 0.75) < 0.03


def test_cell_rng_reproducible():
    a = rules.cell_rng(123, "S5", 7).random()
    b = rules.cell_rng(123, "S5", 7).random()
    assert a == b
    assert rules.cell_rng(123, "S4", 7).random() != a


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def test_study1_diagnostics_sane():
    rows = devdata.load_rows("pilot").head(150).reset_index(drop=True)
    fs = load_fields()["political_view"]
    from reviewer_sim.fields import input_group_of  # noqa: PLC0415

    in_names = scored_input_names()
    group_map = {}
    for iname in in_names:
        for v in rows[iname].dropna().unique():
            group_map[(iname, v)] = input_group_of(iname, v)

    real = rows[["profile_id", *in_names, "political_view"]].copy()
    rng = np.random.default_rng(0)
    sim = real.copy()
    # a collapsed simulation: everyone Conservative
    sim["political_view"] = "Conservative"
    d = diagnostics.study1_diagnostics(real, sim, "political_view", in_names, group_map)
    assert d["tv_distance"] > 0.3
    assert d["collapse_rate"] > 0
    assert 0 <= d["missing_rate"] < 1e-9


# --------------------------------------------------------------------------- #
# Mock end-to-end Study 1 (no network)
# --------------------------------------------------------------------------- #

class MockClient:
    """Returns canned JSON for every call type; never touches the network."""

    max_workers = 2

    class _R:
        def __init__(self, content):
            self.content = content
            self.model_version = "mock"
            self.cached = False
            self.usage = {}

    def chat(self, messages, *, model, meta=None, seed=None, purpose="", **_):
        prompt = messages[1]["content"]
        fs = load_fields()
        if purpose == "candidate_value":
            q = [k for k in meta["question"].split()]  # placeholder
            qname = meta["question"]
            f = fs[qname]
            vals = f.answers if not f.is_numeric else [str(v) for v in range(0, 31, 10)]
            v = vals[seed % len(vals)]
            return self._R(json.dumps({qname: v}))
        if purpose in ("judge_values",):
            # parse the candidate labels out of the prompt block
            block = prompt.split("Candidate answers:")[-1].split("Return JSON")[0]
            labels = [ln.split(":")[0].strip() for ln in block.strip().splitlines() if ":" in ln]
            n = len(labels)
            probs = {lab: round(1.0 / n, 4) for lab in labels}
            return self._R(json.dumps(probs))
        if purpose == "plausibility":
            block = prompt.split("Candidate answers:")[-1].split("Return JSON")[0]
            labels = [ln.split(":")[0].strip() for ln in block.strip().splitlines() if ":" in ln]
            return self._R(json.dumps({lab: "plausible" for lab in labels}))
        if purpose == "state_distribution":
            qname = meta["question"]
            f = fs[qname]
            if len(f.answers) <= 12:
                probs = {a: round(1 / len(f.answers), 4) for a in f.answers}
            else:
                probs = {a: 0.0 for a in f.answers}
                for a in f.answers[:5]:
                    probs[a] = 0.2
            return self._R(json.dumps(probs))
        if purpose == "pairwise":
            return self._R(json.dumps({"X": 0.5}))
        if purpose == "table_row":
            # find the question from its description in the prompt
            qname = meta["question"]
            f = fs[qname]
            probs = {a: round(1 / len(f.answers), 4) for a in f.answers}
            return self._R(json.dumps(probs))
        raise ValueError(f"unexpected purpose {purpose}")


@pytest.mark.slow
def test_study1_end_to_end_mock(tmp_path):
    cfg = Study1Config(
        judge_model="mock/model",
        generators={"test": "mock/model"},
        questions=["political_view"],
        wordings=("W1",),
        max_rows=60,
    )
    out = run_study1(cfg, tmp_path / "out", tmp_path / "cache", client=MockClient())
    df = out["results"]
    assert len(df) == 8 * 3  # 8 rules x 3 runs
    assert set(df["rule"]) == set(rules.RULES)
    # uniform judge => S1..S4 all near S0's score; just check they all scored
    assert df["aggregate"].notna().all()
    # sim files exist in official layout
    assert (tmp_path / "out" / "political_view" / "test" / "W1" / "run1" / "sim_S6.csv").exists()


def test_run_parallel_preserves_order():
    items = list(range(50))
    out = run_parallel(lambda i, x: x * 2, items, max_workers=4)
    assert out == [x * 2 for x in items]

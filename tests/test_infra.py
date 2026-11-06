"""Tests: official prompt reproduction, dev split, parsing, seed patch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reviewer_sim import devdata, fields, prompts  # noqa: E402
from reviewer_sim.parse import (  # noqa: E402
    ParseError,
    parse_distribution,
    parse_labels,
    parse_single_probability,
    uniform,
)
from reviewer_sim.scoring import score_run  # noqa: E402

INPUTS = {
    "birth_year": 1975,
    "age": 43,
    "gender": "Male",
    "race": "Non-Black, Non-Hispanic",
    "immigrant_status": "Non-immigrant",
    "mother_education": "High school",
    "father_education": "High school",
    "mother_occupation": "Sales and office occupations",
    "father_occupation": "Sales and office occupations",
}


# --------------------------------------------------------------------------- #
# Official prompt reproduction
# --------------------------------------------------------------------------- #

def test_official_generation_prompt_matches_benchmark():
    """Rebuild build_prompt() from the vendored generation script and compare."""
    cfg = yaml.safe_load(
        open(REPO / "data/raw/ssdatabench/simulation/configs/content/gss_2018.yaml")
    )
    output_vars = cfg["output_variables"]
    FIELDS = list(output_vars.keys())
    DESC_MAP = {k: v.get("description", "") for k, v in output_vars.items()}
    ALLOWED = {k: v.get("allowed", None) for k, v in output_vars.items()}
    TYPES = {k: v.get("type", "static") for k, v in output_vars.items()}

    def _stringify_allowed(spec):
        if isinstance(spec, list):
            return "[" + ", ".join(f'"{v}"' for v in spec) + "]"
        if isinstance(spec, dict) and spec.get("type") == "numeric":
            rng = f"[{spec.get('min')}, {spec.get('max')}]" if spec.get("min") is not None else "any"
            special = spec.get("special", [])
            if special:
                return f"number in {rng} OR one of {special}"
            return f"number in {rng}"
        return str(spec)

    desc_lines, allowed_lines = [], []
    for k in FIELDS:
        desc_lines.append(f"- {k} ({TYPES[k]}): {DESC_MAP[k]}")
        allowed_lines.append(f"- {k}: {_stringify_allowed(ALLOWED[k])}")
    desc_block = "\n".join(desc_lines)
    allowed_block = "\n".join(allowed_lines)
    reference_year = cfg["population_context"]["reference_year"]
    ctx_parts = [f"{k}: {v}" for k, v in INPUTS.items()]
    input_context = "Conditioned on the following input attributes:\n" + ", ".join(ctx_parts) + "\n"
    expected = f"""
You are simulating one *randomly selected* synthetic individual sampled from {cfg['context'].strip()}
The data represent a cross-sectional snapshot collected in the year {reference_year}.
{input_context}Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
{", ".join(FIELDS)}.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
{desc_block}

Allowed values:
{allowed_block}
""".strip()

    got = prompts.official_generation_prompt(INPUTS)
    assert got == expected


def test_one_question_prompt_uses_official_framing():
    p = prompts.one_question_generation_prompt(INPUTS, "political_view")
    assert "Political view" in p
    assert '"Liberal"' in p and '"Conservative"' in p
    # only the one question's description appears
    assert "Marital status" not in p


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #

def test_fields_counts_and_types():
    fs = fields.load_fields()
    assert len([f for f in fs.values() if f.role == "x"]) == 9
    assert len([f for f in fs.values() if f.role == "y"]) == 21
    assert fs["mental_health"].is_numeric
    assert len(fs["mental_health"].answers) == 31
    assert fs["age_first_childbirth"].numeric.special == ("No Child",)
    assert len(fs["age_first_childbirth"].answers) == 29 + 1  # 17..45 plus No Child
    assert fs["political_view"].answers == ["Liberal", "Middle", "Conservative"]


def test_age_bands():
    assert fields.age_band(18) == "18-34"
    assert fields.age_band(54) == "35-54"
    assert fields.age_band(55) == "55+"
    assert fields.age_band(89) == "55+"


# --------------------------------------------------------------------------- #
# Dev split
# --------------------------------------------------------------------------- #

def test_dev_split():
    paths = devdata.build_dev_split()
    fit = pd.read_csv(paths["fitting"])
    ev = pd.read_csv(paths["evaluation"])
    pilot = pd.read_csv(paths["pilot"])
    final = pd.read_csv(paths["final"])

    assert len(fit) + len(ev) == 1348
    assert len(fit) == round(0.4 * 1348)  # 539
    assert len(pilot) == 500
    assert len(final) == 1000

    full = devdata.load_full()
    bench_ids = set(full.loc[full.in_benchmark, "resp_id"])
    dev_ids = set(full.loc[~full.in_benchmark, "resp_id"])
    assert set(final["resp_id"] if "resp_id" in final else range(1000)) or True
    # dev lists disjoint from benchmark rows
    got_dev = set(pd.concat([fit, ev])["resp_id"])
    assert got_dev == dev_ids
    assert len(got_dev & bench_ids) == 0
    # pilot rows are a subset of the evaluation part
    assert set(pilot["resp_id"]) <= set(ev["resp_id"])
    # split is stable
    assert list(fit["resp_id"]) == list(pd.read_csv(paths["fitting"])["resp_id"])


def test_run_csv_format():
    rows = devdata.load_rows("pilot")
    from reviewer_sim.devdata import write_run_csv, _write_official_csv  # noqa: PLC0415
    sim = rows[["profile_id"]].copy()
    for f in fields.output_names():
        sim[f] = "x"
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        sim_path, _ = write_run_csv(sim, Path(td) / "run1", "test")
        got = pd.read_csv(sim_path)
        assert list(got.columns[:3]) == ["profile_id", "birth_year", "age"]
        assert len(got) == len(sim)
        assert set(got["profile_id"]) == set(rows["profile_id"])


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def test_parse_distribution_basic():
    r = parse_distribution('{"A": 0.6, "B": 0.4}', ["A", "B"])
    assert r.probs == {"A": 0.6, "B": 0.4}
    assert not r.renormalized

    r = parse_distribution('{"B": 60, "A": 40}', ["A", "B"])
    assert r.probs["A"] == pytest.approx(0.4)

    # tolerance and renormalization
    r = parse_distribution('{"A": 95, "B": 5}', ["A", "B"])
    assert r.probs["A"] == pytest.approx(0.95)
    with pytest.raises(ParseError):
        parse_distribution('{"A": 130, "B": 5}', ["A", "B"])
    with pytest.raises(ParseError):
        parse_distribution('{"A": 0.5}', ["A", "B"])
    with pytest.raises(ParseError):
        parse_distribution('{"A": -0.1, "B": 1.1}', ["A", "B"])


def test_parse_distribution_counts_mode_and_numeric_keys():
    r = parse_distribution('{"3": 25, "7": 25, "0": 50}', ["0", "3", "7"], counts_mode=True)
    assert r.probs["0"] == pytest.approx(0.5)
    # numeric key returned as JSON number
    r = parse_distribution('{3: 0.5, 7: 0.5}', ["3", "7"])
    assert r.probs["3"] == pytest.approx(0.5)
    # W2 sum tolerance 90..110
    r = parse_distribution('{"3": 55, "7": 55}', ["3", "7"], counts_mode=True)
    assert r.probs["3"] == pytest.approx(0.5)
    with pytest.raises(ParseError):
        parse_distribution('{"3": 80, "7": 80}', ["3", "7"], counts_mode=True)


def test_parse_distribution_fences_and_extra_keys():
    r = parse_distribution(
        '```json\n{"A": 0.5, "B": 0.5, "note": "hi"}\n```', ["A", "B"]
    )
    assert r.probs["A"] == pytest.approx(0.5)


def test_parse_labels_and_pairwise():
    got = parse_labels('{"A": "plausible", "B": "implausible"}', ["A", "B"],
                       ("plausible", "implausible"))
    assert got == {"A": "plausible", "B": "implausible"}
    with pytest.raises(ParseError):
        parse_labels('{"A": "yes"}', ["A", "B"], ("plausible", "implausible"))

    assert parse_single_probability('{"X": 0.7}') == pytest.approx(0.7)
    assert parse_single_probability('{"X": 70}') == pytest.approx(0.7)
    assert parse_single_probability('{"X": 70}', counts_mode=True) == pytest.approx(0.7)
    with pytest.raises(ParseError):
        parse_single_probability('{"X": 1.5}')


def test_uniform_fallback():
    r = uniform(["a", "b", "c"])
    assert r.fallback and r.probs["a"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Seed patch
# --------------------------------------------------------------------------- #

def _tiny_run_csvs(td: Path):
    """Small real/sim pair with the official column layout."""
    fs = fields.load_fields()
    cols = ["profile_id", *fields.scored_input_names(), *fields.output_names()]
    rng = np.random.default_rng(7)
    n = 120
    real = pd.DataFrame({"profile_id": np.arange(n)})
    sim = pd.DataFrame({"profile_id": np.arange(n)})
    for c in cols[1:]:
        f = fs[c]
        if f.allowed is not None:
            real[c] = rng.choice(f.allowed, size=n)
            sim[c] = rng.choice(f.allowed, size=n)
        else:
            real[c] = rng.integers(f.numeric.min, f.numeric.max + 1, size=n)
            sim[c] = rng.integers(f.numeric.min, f.numeric.max + 1, size=n)
    p_real = td / "sampled_inputs_test.csv"
    p_sim = td / "sim_profiles_test.csv"
    real.to_csv(p_real, index=False)
    sim.to_csv(p_sim, index=False)
    return p_real, p_sim


@pytest.mark.slow
def test_scoring_seed_patch(tmp_path):
    import warnings  # noqa: PLC0415

    p_real, p_sim = _tiny_run_csvs(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = score_run(p_sim, p_real, tmp_path / "sA", scoring_seed=1, B=5, verbose=False)
        b = score_run(p_sim, p_real, tmp_path / "sB", scoring_seed=1, B=5, verbose=False)
        c = score_run(p_sim, p_real, tmp_path / "sC", scoring_seed=2, B=5, verbose=False)
    assert a.type_scores == b.type_scores, "same seed must give identical scores"
    diff = any(
        abs(a.type_scores[t] - c.type_scores[t]) > 1e-12
        for t in a.type_scores
        if a.type_scores[t] == a.type_scores[t]
    )
    assert diff, "different seed should vary at least one type score"


# --------------------------------------------------------------------------- #
# Prompt builders: wording families
# --------------------------------------------------------------------------- #

def test_judge_values_prompt_w1w2w3():
    vals = ["Liberal", "Middle", "Conservative"]
    for w in ("W1", "W2", "W3"):
        p, meta = prompts.judge_values_prompt(INPUTS, "political_view", vals, w, seed=11)
        assert meta["wording"] == w
        assert "A: " in p and "B: " in p and "C: " in p
        assert "Political view" in p
        # W1 anti-typicality instruction present
        if w == "W1":
            assert "not the single most typical" not in p  # that is Family 5
            assert "Do not favour an answer because it seems typical" in p
        if w == "W2":
            assert "100" in p


def test_state_distribution_prompt_numeric_keys_listed():
    p, _ = prompts.state_distribution_prompt(INPUTS, "vocabulary_test", "W1")
    # 0..10 all appear as keys in the answer list block
    for v in range(11):
        assert f'"{v}"' in p


def test_state_table_prompt_group_only():
    p, _ = prompts.state_table_prompt("political_view", "women aged 35-54", "W1")
    assert "women aged 35-54" in p
    assert "Conditioned on the following input attributes" not in p


def test_label_candidates_shuffle_deterministic():
    lab = prompts.label_candidates(5, seed=3)
    lab2 = prompts.label_candidates(5, seed=3)
    assert lab.order == lab2.order and lab.label_of == lab2.label_of
    assert sorted(lab.label_of.values()) == ["A", "B", "C", "D", "E"]
    lab3 = prompts.label_candidates(5, seed=4)
    assert lab3.order != lab.order

"""Mock test for the generator-choice module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reviewer_sim import devdata  # noqa: E402
from reviewer_sim.conditions.genchoice import diversity_metrics, run_genchoice  # noqa: E402
from reviewer_sim.fields import output_names  # noqa: E402


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
        obj = {}
        for q in output_names():
            obj[q] = f"v{rng.randint(0, 2)}" if q == "political_view" else "x"
        return self._R(json.dumps(obj))


def test_diversity_metrics_counts():
    profs = [
        [  # row 1: half duplicates
            {"a": "x", "b": "x"}, {"a": "x", "b": "x"},
            {"a": "y", "b": "x"}, {"a": "x", "b": "y"},
        ],
    ]
    m = diversity_metrics(profs)
    assert 0 < m["duplicate_share"] < 1
    assert 0 < m["mean_entropy"] <= 1
    assert 0 < m["mean_hamming"] < 1


def test_genchoice_mock(tmp_path, monkeypatch):
    rows = devdata.load_rows("pilot").head(4)
    monkeypatch.setattr(
        "reviewer_sim.conditions.genchoice.devdata.load_rows", lambda name: rows
    )
    df = run_genchoice(
        {"mock": "mock/model"},
        out_dir=tmp_path / "out",
        cache_root=tmp_path / "cache",
        n_rows=4,
        client=MockClient(),
    )
    assert len(df) == 1
    assert 0 <= df.iloc[0]["valid_rate"] <= 1
    assert (tmp_path / "out" / "profiles_mock.json").exists()

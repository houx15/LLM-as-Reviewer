"""Wrapper around the patched official SSDataBench scorer.

Scores a run folder's ``sim_profiles_*.csv`` against its real rows using the
official Types 1–3 with master settings (B = 100, α = 0.05, 500 per draw) and
the seed patch (``patches/ssdatabench-seed.diff``) as the sole change: the same
scoring seed gives identical scores, different seeds vary slightly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .scoring_ssd import type1, type2, type3

SSD_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ssdatabench"
TYPE_CONFIG_DIR = SSD_ROOT / "evaluation" / "config" / "gss_2018"

MASTER_B = 100
MASTER_ALPHA = 0.05
MASTER_SAMPLE_N = 500


@dataclass
class ScoreResult:
    type_scores: dict[str, float]           # {"type1": ..., "type2": ..., "type3": ...}
    aggregate: float
    summaries: dict[str, pd.DataFrame] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)

    def per_variable(self, type_name: str) -> pd.DataFrame:
        """Per-variable insignificant rates (rows before the trailing 'avg')."""
        df = self.summaries[type_name]
        return df[~df["variable"].isin(["avg", ""])].copy() if "variable" in df.columns else df


def score_run(
    sim_csv: str | Path,
    real_csv: str | Path,
    out_dir: str | Path,
    scoring_seed: int,
    types: tuple[str, ...] = ("type1", "type2", "type3"),
    B: int = MASTER_B,
    alpha: float = MASTER_ALPHA,
    sample_n: int = MASTER_SAMPLE_N,
    verbose: bool = False,
) -> ScoreResult:
    """Score one run with the patched official scorer.

    The real CSV must contain ``profile_id`` and the real output columns; the
    sim CSV the same columns with one row per profile_id (official format).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    funcs = {"type1": type1, "type2": type2, "type3": type3}
    type_scores: dict[str, float] = {}
    summaries: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}

    for t in types:
        module = funcs[t]
        config_path = str(TYPE_CONFIG_DIR / f"{t}.yaml")
        run_fn = getattr(module, f"run_{t}_eval")
        res = run_fn(
            config=config_path,
            real_csv=str(real_csv),
            sim_csv=str(sim_csv),
            out_dir=str(out_dir / t),
            bootstrap_B=B,
            alpha=alpha,
            bootstrap_sample_n=sample_n,
            verbose=verbose,
            seed=scoring_seed,
        )
        if res is None:  # type3 can bail when no real model fits
            type_scores[t] = float("nan")
            continue
        type_scores[t] = float(res["avg_insignificant_rate"])
        summaries[t] = res.get("summary_df", pd.DataFrame())
        paths[t] = Path(res.get("summary_path", ""))

    vals = [v for v in type_scores.values() if v == v]
    aggregate = sum(vals) / len(vals) if vals else float("nan")
    return ScoreResult(
        type_scores=type_scores, aggregate=aggregate, summaries=summaries, paths=paths
    )


def _load_type_config(t: str) -> dict:
    with open(TYPE_CONFIG_DIR / f"{t}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_study1(
    question: str,
    sim_csv: str | Path,
    real_csv: str | Path,
    out_dir: str | Path,
    scoring_seed: int,
    input_names: list[str],
    B: int = MASTER_B,
    alpha: float = MASTER_ALPHA,
    sample_n: int = MASTER_SAMPLE_N,
    verbose: bool = False,
) -> ScoreResult:
    """Score one Study 1 question: official Type 1 for the question and
    official Type 2 for this question paired with each input (SPEC §4.5),
    on configs restricted to those variables, master settings, seed patch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg1 = _load_type_config("type1")
    cfg1["variables"] = {question: cfg1["variables"][question]}
    cfg2 = _load_type_config("type2")
    keep = {k for k in cfg2["variables"] if k == question or k in input_names}
    cfg2["variables"] = {k: v for k, v in cfg2["variables"].items() if k in keep}

    type_scores: dict[str, float] = {}
    summaries: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for t, module, cfg in (("type1", type1, cfg1), ("type2", type2, cfg2)):
        res = getattr(module, f"run_{t}_eval")(
            config=cfg,
            real_csv=str(real_csv),
            sim_csv=str(sim_csv),
            out_dir=str(out_dir / t),
            bootstrap_B=B,
            alpha=alpha,
            bootstrap_sample_n=sample_n,
            verbose=verbose,
            seed=scoring_seed,
        )
        type_scores[t] = float(res["avg_insignificant_rate"])
        summaries[t] = res.get("summary_df", pd.DataFrame())
        paths[t] = Path(res.get("summary_path", ""))

    agg = (type_scores["type1"] + type_scores["type2"]) / 2
    return ScoreResult(type_scores=type_scores, aggregate=agg, summaries=summaries, paths=paths)

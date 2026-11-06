"""CLI: run conditions and score runs.

Usage:
  uv run python -m reviewer_sim.run study1 --judge-model M --generator name=id [--question q]*
  uv run python -m reviewer_sim.run split        # build/freeze the dev split
  uv run python -m reviewer_sim.run models       # list candidate models + prices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import devdata

REPO = Path(__file__).resolve().parents[1]


def cmd_split(_args) -> None:
    paths = devdata.build_dev_split()
    for name, p in paths.items():
        import pandas as pd  # noqa: PLC0415

        print(f"{name:10s} {p}  ({len(pd.read_csv(p))} rows)")


def cmd_models(_args) -> None:
    import os  # noqa: PLC0415

    import requests  # noqa: PLC0415

    key = os.getenv("OPENROUTER_API_KEY")
    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    r.raise_for_status()
    models = r.json()["data"]
    interest = [
        "google/gemini-2.5-flash", "openai/gpt-4o", "openai/gpt-5",
        "anthropic/claude-3.5-sonnet", "deepseek/deepseek-chat",
        "qwen/qwen2.5-vl-32b-instruct:free", "x-ai/grok-4",
        "qwen/qwen3-32b", "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it", "mistralai/mistral-small-3.2-24b-instruct",
        "openai/gpt-4o-mini",
    ]
    by_id = {m["id"]: m for m in models}
    for mid in interest:
        m = by_id.get(mid)
        if not m:
            print(f"{mid:45s} NOT SERVED")
            continue
        p = m.get("pricing", {})
        ki = float(p.get("prompt", 0)) * 1e6
        ko = float(p.get("completion", 0)) * 1e6
        print(f"{mid:45s} in=${ki:.2f}/M out=${ko:.2f}/M  ctx={m.get('context_length')}")


def cmd_study1(args) -> None:
    from .conditions.study1 import Study1Config, run_study1  # noqa: PLC0415

    generators = dict(g.split("=", 1) for g in args.generator)
    cfg = Study1Config(
        judge_model=args.judge_model,
        generators=generators,
        questions=args.question or ["political_view", "mental_health"],
        wordings=tuple(args.wordings) if args.wordings else ("W1", "W2", "W3"),
        n_candidates=args.n_candidates,
        rows_name=args.rows,
        max_rows=args.max_rows,
    )
    tag = args.tag or "study1"
    out_root = REPO / "runs" / "phase0" / tag
    cache_root = REPO / "runs" / "phase0" / "cache"
    result = run_study1(cfg, out_root, cache_root)
    df = result["results"]
    print("\n==== Study 1 summary (mean over runs and wordings) ====")
    agg = (
        df.groupby(["question", "generator", "rule"])
        .agg(
            aggregate=("aggregate", "mean"),
            spread=("spread_ratio", "mean"),
            tv=("tv_distance", "mean"),
            assoc_diff=("assoc_diff", "mean"),
        )
        .round(3)
    )
    print(agg.to_string())
    print(f"\nfull results: {out_root / 'study1_results.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="reviewer_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("split").set_defaults(func=cmd_split)
    sub.add_parser("models").set_defaults(func=cmd_models)

    s1 = sub.add_parser("study1")
    s1.add_argument("--judge-model", required=True)
    s1.add_argument("--generator", action="append", required=True,
                    help="name=model_id, repeatable")
    s1.add_argument("--question", action="append",
                    choices=["political_view", "mental_health"])
    s1.add_argument("--wordings", nargs="+")
    s1.add_argument("--n-candidates", type=int, default=10)
    s1.add_argument("--rows", default="pilot")
    s1.add_argument("--max-rows", type=int, default=None)
    s1.add_argument("--tag", default=None)
    s1.set_defaults(func=cmd_study1)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

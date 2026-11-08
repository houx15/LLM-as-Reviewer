"""Phase 3 (Study 2 full): eligible methods on the whole evaluation part.

Eligible per SPEC §8 (methods under test; no-go judging methods dropped):
B5, R3, P4, P3, plus B4/B6 baselines and references B1/B2. 3 runs; a run
redoes every call (run-specific seeds). A budget guard pauses the command
before exceeding the weekly cap; rerunning resumes at the first missing
(condition, wording, run) combination — results are deduplicated on those keys.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import devdata
from ..conditions import phase2 as p2
from ..conditions.phase2 import propose_graph
from ..copula import sample_stated_copula
from ..fields import output_names
from ..llm import LLMClient
from ..references import fit_copula, fit_marginals, sample_b1, sample_b2

REPO = Path(__file__).resolve().parents[2]
WORDINGS = ("W1", "W2", "W3")
RUNS = (1, 2, 3)
PRICE = {"gemini": (0.30, 2.50), "qwen": (0.08, 0.28)}


def _cache_cost(cache_path: Path) -> float:
    cost = 0.0
    if not cache_path.exists():
        return 0.0
    for line in open(cache_path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = "gemini" if "gemini" in r.get("model_version", "") else "qwen"
        u = r.get("usage", {})
        p = PRICE[m]
        cost += u.get("prompt_tokens", 0) / 1e6 * p[0]
        cost += u.get("completion_tokens", 0) / 1e6 * p[1]
    return cost


class Results:
    def __init__(self, path: Path):
        self.path = path
        self.df = pd.read_csv(path) if path.exists() else pd.DataFrame()

    def has(self, condition: str, wording: str, run: int) -> bool:
        if self.df.empty:
            return False
        sel = self.df[
            (self.df["condition"] == condition) & (self.df["wording"] == wording)
            & (self.df["run"] == run)
        ]
        return len(sel) > 0

    def add(self, rows: list[dict]):
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.df.to_csv(self.path, index=False)


def run_phase3(
    judge_model: str,
    out_root: str | Path = REPO / "runs" / "phase3",
    cache_root: str | Path = REPO / "runs" / "phase3" / "cache",
    weekly_cap: float = 195.0,
    floor: float = 10.0,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_root) / "cache.jsonl"
    client = LLMClient(cache_path=cache_path, log_path=Path(cache_root) / "calls.jsonl")
    res = Results(out_root / "phase3_results.csv")

    def budget_left() -> float:
        return weekly_cap - _cache_cost(cache_path)

    rows = devdata.load_rows("evaluation")
    real_csv = devdata.build_dev_split()["evaluation"]
    print(f"[phase3] {len(rows)} evaluation rows; budget left ${budget_left():.2f}",
          flush=True)

    # ---------------- P4: stated tables -> copula ------------------------------
    for wording in WORDINGS:
        for run in RUNS:
            if res.has("P4", wording, run):
                continue
            if budget_left() < floor:
                return _finish(out_root)
            df, *_ = p2.run_p4(
                client, judge_model, rows, real_csv, out_root / "P4",
                wording, seed=1_900_000 + 1_000 * WORDINGS.index(wording) + 97_000 * run,
                tables_dir=REPO / "runs" / "phase3" / "stated_tables" / f"{wording}_r{run}",
                runs=(run,),
            )
            df = df[(df["wording"] == wording) & (df["run"] == run)]
            res.add(df.to_dict("records"))

    # ---------------- B5 --------------------------------------------------------
    for wording in WORDINGS:
        for run in RUNS:
            if res.has("B5", wording, run):
                continue
            if budget_left() < floor:
                return _finish(out_root)
            seed = 1_910_000 + 1_000 * WORDINGS.index(wording) + 97_000 * run
            df = _b5_run(client, judge_model, rows, real_csv, out_root / "B5",
                         wording, seed, run)
            res.add(df.to_dict("records"))

    # ---------------- R3 --------------------------------------------------------
    for wording in WORDINGS:
        for run in RUNS:
            if res.has("R3", wording, run):
                continue
            if budget_left() < floor:
                return _finish(out_root)
            graph = propose_graph(
                client, judge_model, seed=1_920_000 + WORDINGS.index(wording) + 97_000 * run
            )
            (out_root / "R3_graphs").mkdir(exist_ok=True)
            (out_root / "R3_graphs" / f"{wording}_r{run}.json").write_text(json.dumps(graph))
            seed = 1_921_000 + 1_000 * WORDINGS.index(wording) + 97_000 * run
            df = _r3_run(client, judge_model, rows, real_csv, out_root / "R3",
                         wording, seed, run, graph)
            res.add(df.to_dict("records"))

    # ---------------- P3: copula candidates + judge -----------------------------
    for wording in WORDINGS:
        for run in RUNS:
            if res.has("P3", wording, run):
                continue
            if budget_left() < floor:
                return _finish(out_root)
            payload = json.loads(
                (REPO / "runs" / "phase3" / "stated_tables" / f"{wording}_r{run}" /
                 wording / "tables.json").read_text()
            )
            margs = {q: np.array(v) for q, v in payload["row_marginals"].items()}
            pools = []
            for j in range(5):
                sim = sample_stated_copula(
                    margs, np.array(payload["corr"]), rows,
                    np.random.default_rng(
                        1_950_000 + 100 * WORDINGS.index(wording) + 1_000 * run + j),
                )
                pools.append(sim)
            pool = [
                [{q: pools[j].iloc[i][q] for q in output_names()} for j in range(5)]
                for i in range(len(rows))
            ]
            j_probs = p2._judge_pool(
                client, judge_model, rows, pool, wording,
                1_960_000 + 1_000 * WORDINGS.index(wording) + 97_000 * run,
            )
            df = p2.run_judged_pool(
                client, judge_model, rows, real_csv, out_root / "P3", "P3", wording,
                1_961_000 + 1_000 * WORDINGS.index(wording) + 97_000 * run, pool, j_probs,
            )
            df = df[(df["wording"] == wording) & (df["run"] == run)]
            res.add(df.to_dict("records"))

    # ---------------- B4 / B6 ----------------------------------------------------
    from .b4 import generate_b4  # noqa: PLC0415

    for run in RUNS:
        if res.has("B4", "official", run):
            continue
        if budget_left() < floor:
            return _finish(out_root)
        sim = generate_b4(client, judge_model, rows, seed=1_930_000 + 97_000 * run)
        r = p2.score_condition(sim, real_csv, out_root / "B4" / f"run{run}", run)
        res.add([{**r, "condition": "B4", "wording": "official", "run": run}])
        print(f"[phase3] B4 run{run}: agg={r['aggregate']:.3f}", flush=True)

    for run in RUNS:
        if res.has("B6", "official", run):
            continue
        if budget_left() < floor:
            return _finish(out_root)
        df, _ = p2.run_b6(
            client, judge_model, rows, real_csv, out_root / "B6",
            seed=1_940_000 + 97_000 * run,
        )
        df = df[df["run"] == run].copy()
        df["wording"] = "official"
        res.add(df.to_dict("records"))

    # ---------------- references -------------------------------------------------
    fit = devdata.load_rows("fitting")
    marg = fit_marginals(fit)
    cop = fit_copula(fit)
    for run in RUNS:
        rng = np.random.default_rng(3000 + run)
        out = []
        sim1, _ = sample_b1(marg, rows, rng)
        sim2 = sample_b2(cop, rows, rng)
        for name, sim in (("B1", sim1), ("B2", sim2)):
            if res.has(name, "reference", run):
                continue
            r = p2.score_condition(sim, real_csv, out_root / name / f"run{run}", run)
            out.append({**r, "condition": name, "wording": "reference", "run": run})
            print(f"[phase3] {name} run{run}: agg={r['aggregate']:.3f}", flush=True)
        if out:
            res.add(out)

    return _finish(out_root)


def _b5_run(client, model, rows, real_csv, out_root, wording, seed, run):
    """One independent B5 pass (fresh calls: run folded into the seeds)."""
    import random  # noqa: PLC0415

    from ..fields import input_names  # noqa: PLC0415
    from ..llm import run_parallel  # noqa: PLC0415
    from ..rules import state_distribution  # noqa: PLC0415

    outs = output_names()
    in_names = input_names()

    def one_row(i, item):
        _, row = item
        rng = random.Random(seed + 101 * i + 999 * run)
        draws = {}
        for oi, q in enumerate(outs):
            dist = state_distribution(
                client, model, row, q, wording, seed + 101 * i + 7 * oi, in_names
            )
            items = [(v, p) for v, p in dist.probs.items() if p > 0]
            ws = np.array([w for _, w in items])
            draws[q] = str(rng.choices([v for v, _ in items],
                                       weights=(ws / ws.sum()).tolist(), k=1)[0])
        return draws

    t0 = time.time()
    draws_list = run_parallel(one_row, list(rows.iterrows()),
                              max_workers=client.max_workers,
                              desc=f"phase3/B5/{wording}/r{run}")
    sim = rows.copy()
    for q in outs:
        sim[q] = [d[q] for d in draws_list]
    r = p2.score_condition(sim, real_csv, Path(out_root) / wording / f"run{run}", run)
    r.update({"condition": "B5", "wording": wording})
    print(f"[phase3] B5/{wording} run{run}: agg={r['aggregate']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame([r])


def _r3_run(client, model, rows, real_csv, out_root, wording, seed, run, graph):
    import random  # noqa: PLC0415

    from ..fields import input_names  # noqa: PLC0415
    from ..llm import run_parallel  # noqa: PLC0415
    from ..rules import state_distribution  # noqa: PLC0415

    outs = output_names()
    in_names = input_names()
    topo = p2._topo_order(graph)

    def one_row(i, item):
        _, row = item
        given = {}
        for q in topo:
            dist = state_distribution(
                client, model, row, q, wording,
                seed + 211 * i + 7 * outs.index(q) + 100_000, in_names,
                given=dict(given) if given else None,
            )
            items = [(v, p) for v, p in dist.probs.items() if p > 0]
            ws = np.array([w for _, w in items])
            v = str(random.Random(seed + 211 * i + 7 * outs.index(q) + 999 * run).choices(
                [x for x, _ in items], weights=(ws / ws.sum()).tolist(), k=1)[0])
            given[q] = v
        return given

    t0 = time.time()
    draws_list = run_parallel(one_row, list(rows.iterrows()),
                              max_workers=client.max_workers,
                              desc=f"phase3/R3/{wording}/r{run}")
    sim = rows.copy()
    for q in outs:
        sim[q] = [d[q] for d in draws_list]
    r = p2.score_condition(sim, real_csv, Path(out_root) / wording / f"run{run}", run)
    r.update({"condition": "R3", "wording": wording})
    print(f"[phase3] R3/{wording} run{run}: agg={r['aggregate']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame([r])


def _finish(out_root: Path) -> dict:
    df = pd.read_csv(out_root / "phase3_results.csv")
    print(df.groupby(["condition", "wording"])[
        ["type1", "type2", "type3", "aggregate"]].mean().round(3).to_string())
    return {"results": df}

"""Phase 2 orchestrator: all conditions on the 500 pilot rows + Study 3.

B4 is reused from Phase 1 (cached calls, rescored on the pilot rows). B5/R3/
P1-P3 run W1-W3; P4 runs W1-W3 (tables stored per wording); B6 and P5 run
their single/inherit wordings; references B1/B2 are scored on the pilot rows
for comparability.
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
from ..fields import input_group_of, output_names, scored_input_names
from ..llm import LLMClient
from ..references import fit_copula, fit_marginals, sample_b1, sample_b2

REPO = Path(__file__).resolve().parents[2]
WORDINGS = ("W1", "W2", "W3")


def run_phase2(
    judge_model: str,
    small_model: str,
    out_root: str | Path,
    cache_root: str | Path,
    client: LLMClient | None = None,
    rows_name: str = "pilot",
    wordings: tuple[str, ...] = WORDINGS,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = devdata.load_rows(rows_name)
    real_csv = devdata.build_dev_split()[rows_name]
    if client is None:
        client = LLMClient(
            cache_path=Path(cache_root) / "cache.jsonl",
            log_path=Path(cache_root) / "calls.jsonl",
        )

    all_results: list[pd.DataFrame] = []
    t0 = time.time()

    # ---------------- B4 (cached from Phase 1; rescored on pilot rows) -------
    b4_profiles = pd.read_csv(REPO / "runs" / "phase1" / "b4" / "b4_profiles.csv")
    b4_profiles = b4_profiles.drop(columns=[c for c in ("draw",) if c in b4_profiles])
    for run in p2.RUNS:
        r = p2.score_condition(b4_profiles, real_csv, out_root / "B4" / f"run{run}", run)
        r.update({"condition": "B4", "wording": "official"})
        all_results.append(pd.DataFrame([r]))
    print("[phase2] B4 rescored on pilot rows", flush=True)

    # ---------------- B6 -----------------------------------------------------
    df, _ = p2.run_b6(client, judge_model, rows, real_csv, out_root / "B6", seed=888_000)
    df["condition"] = "B6"
    df["wording"] = "official"
    all_results.append(df)

    # ---------------- stated tables (P4) first: reused by P3 -----------------
    p4_result, row_marginals, corr, p4_sims = None, None, None, None
    tables_payload = {}
    for wording in wordings:
        res, marg, cor, sims = p2.run_p4(
            client, judge_model, rows, real_csv, out_root / "P4", wording,
            seed=900_000 + 1_000 * WORDINGS.index(wording),
            tables_dir=REPO / "data" / "stated_tables",
        )
        res["condition"] = "P4"
        all_results.append(res)
        tables_payload[wording] = json.loads(
            (REPO / "data" / "stated_tables" / wording / "tables.json").read_text()
        )

    # ---------------- B5 -----------------------------------------------------
    for wording in wordings:
        all_results.append(p2.run_b5(
            client, judge_model, rows, real_csv, out_root / "B5", wording,
            seed=910_000 + 1_000 * WORDINGS.index(wording),
        ))

    # ---------------- R3 (graph per wording) ---------------------------------
    for wording in wordings:
        graph = propose_graph(
            client, judge_model, seed=920_000 + 1_000 * WORDINGS.index(wording)
        )
        (out_root / "R3_graphs").mkdir(exist_ok=True)
        (out_root / "R3_graphs" / f"{wording}.json").write_text(json.dumps(graph, indent=1))
        res = p2.run_r3(
            client, judge_model, rows, real_csv, out_root / "R3", wording,
            seed=921_000 + 1_000 * WORDINGS.index(wording), graph=graph,
        )
        all_results.append(res)

    # ---------------- candidate pools ----------------------------------------
    own_pool = p2._generate_pool(
        client, judge_model, rows, 5, 930_000, "p1_own_candidate"
    )
    small_pool = p2._generate_pool(
        client, small_model, rows, 10, 940_000, "p2_small_candidate"
    )
    p3_pools: dict[str, list] = {}
    for wording in wordings:
        p3_pools[wording] = []
        payload = tables_payload[wording]
        margs = {q: np.array(v) for q, v in payload["row_marginals"].items()}
        for j in range(5):
            sim = sample_stated_copula(
                margs, np.array(payload["corr"]), rows,
                np.random.default_rng(950_000 + 100 * WORDINGS.index(wording) + j),
            )
            p3_pools[wording].append(sim)

    def pool_from_frame(wording):
        out = []
        for i in range(len(rows)):
            out.append([
                {q: p3_pools[wording][j].iloc[i][q] for q in output_names()}
                for j in range(5)
            ])
        return out

    # ---------------- judged pools -------------------------------------------
    for wording in wordings:
        w_i = WORDINGS.index(wording)
        j_own = p2._judge_pool(client, judge_model, rows, own_pool, wording, 960_000 + 1_000 * w_i)
        j_small = p2._judge_pool(client, judge_model, rows, small_pool, wording, 970_000 + 1_000 * w_i)
        j_p3 = p2._judge_pool(client, judge_model, rows, pool_from_frame(wording), wording, 980_000 + 1_000 * w_i)

        all_results.append(p2.run_judged_pool(
            client, judge_model, rows, real_csv, out_root / "P1", "P1", wording,
            961_000 + 1_000 * w_i, own_pool, j_own,
        ))
        all_results.append(p2.pool_random_contrast(
            rows, own_pool, real_csv, out_root / "P1", "P1-random", wording,
            962_000 + 1_000 * w_i,
        ))
        all_results.append(p2.run_judged_pool(
            client, judge_model, rows, real_csv, out_root / "P2", "P2", wording,
            971_000 + 1_000 * w_i, small_pool, j_small,
        ))
        all_results.append(p2.pool_random_contrast(
            rows, small_pool, real_csv, out_root / "P2", "P2-random", wording,
            972_000 + 1_000 * w_i,
        ))
        all_results.append(p2.run_judged_pool(
            client, judge_model, rows, real_csv, out_root / "P3", "P3", wording,
            981_000 + 1_000 * w_i, pool_from_frame(wording), j_p3,
        ))

        # ---------------- P5 (W1 only; inherits judge wording) ----------------
        if wording == "W1":
            stated = _load_stated_tables(REPO / "data" / "stated_tables" / wording)
            rows_inputs = [
                {k: input_group_of(k, rows.iloc[i][k]) for k in scored_input_names()}
                for i in range(len(rows))
            ]
            res, info = p2.run_p5(
                rows, own_pool, j_own, stated, rows_inputs, real_csv,
                out_root / "P5", wording, 990_000,
            )
            all_results.append(res)
            (out_root / "P5").mkdir(parents=True, exist_ok=True)
            (out_root / "P5" / "s_scale.json").write_text(json.dumps(info))

    # ---------------- references on pilot rows -------------------------------
    fit = devdata.load_rows("fitting")
    marg = fit_marginals(fit)
    cop = fit_copula(fit)
    for run in p2.RUNS:
        rng = np.random.default_rng(2000 + run)
        for name, sampler in (
            ("B1", sample_b1), ("B2", sample_b2),
        ):
            if name == "B1":
                sim, _ = sampler(marg, rows, rng)
            else:
                sim = sampler(cop, rows, rng)
            r = p2.score_condition(sim, real_csv, out_root / name / f"run{run}", run)
            r.update({"condition": name, "wording": "reference"})
            all_results.append(pd.DataFrame([r]))

    results = pd.concat([df for df in all_results], ignore_index=True)
    results.to_csv(out_root / "phase2_results.csv", index=False)
    print(f"[phase2] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return {"results": results}


def _load_stated_tables(wording_dir: Path) -> dict:
    """Stated output × input tables (marginals part of the P4 payload)."""
    payload = json.loads((wording_dir / "tables.json").read_text())
    return payload.get("stated_tables", {})

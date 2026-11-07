"""Phase 1 runner: references B1/B2, Study 3 reference models, preliminary B4.

Everything is fitted on the development fitting part; B1/B2 are scored on the
evaluation part (never on the fitting part, SPEC §5.3). B4 is generated fresh
on the pilot rows (also Phase 2's B4).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import devdata
from ..fields import output_names, scored_input_names
from ..llm import LLMClient
from ..references import fit_copula, fit_marginals, sample_b1, sample_b2
from ..scoring import score_run

REPO = Path(__file__).resolve().parents[2]


def run_references(out_root: str | Path, runs: int = 3) -> dict:
    """Fit B1/B2 on the fitting part; score them on the evaluation part."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fit = devdata.load_rows("fitting")
    ev = devdata.load_rows("evaluation")
    real_csv = devdata.build_dev_split()["evaluation"]

    marg = fit_marginals(fit)
    cop = fit_copula(fit)

    results = []
    for run in range(1, runs + 1):
        rng = np.random.default_rng(1000 + run)
        for name, sampler in (
            ("B1", lambda rows: sample_b1(marg, rows, rng)[0]),
            ("B2", lambda rows: sample_b2(cop, rows, rng)),
        ):
            sim = sampler(ev)
            run_dir = out_root / name / f"run{run}"
            run_dir.mkdir(parents=True, exist_ok=True)
            sim_path = run_dir / "sim_profiles.csv"
            sim.to_csv(sim_path, index=False)
            res = score_run(sim_path, real_csv, run_dir / "score", scoring_seed=run)
            results.append({
                "ref": name, "run": run,
                "type1": res.type_scores.get("type1"),
                "type2": res.type_scores.get("type2"),
                "type3": res.type_scores.get("type3"),
                "aggregate": res.aggregate,
            })
            print(f"[refs] {name} run{run}: agg={res.aggregate:.3f} "
                  f"t1={res.type_scores.get('type1'):.3f} t2={res.type_scores.get('type2'):.3f} "
                  f"t3={res.type_scores.get('type3'):.3f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(out_root / "references_results.csv", index=False)

    # fallback diagnostics for B1
    fb = {}
    for q in output_names():
        used = {"coarse": 0, "gender_band": 0, "band": 0, "global": 0}
        for i in range(len(ev)):
            from ..references import coarse_key, group_marginal  # noqa: PLC0415

            _, lvl = group_marginal(marg, coarse_key(ev.iloc[i]), q)
            used[lvl] += 1
        fb[q] = {k: v / len(ev) for k, v in used.items()}
    (out_root / "b1_fallback_share.json").write_text(json.dumps(fb, indent=1))

    info = {
        "corr_proj_shift": cop.proj_shift,
        "mean_offdiag_corr": float(np.mean(np.abs(cop.corr[np.triu_indices(21, 1)]))),
        "b1_fallback_share": fb,
    }
    (out_root / "reference_info.json").write_text(json.dumps(info, indent=1))
    print(f"[refs] corr |offdiag| mean = {info['mean_offdiag_corr']:.3f}, "
          f"PD projection shift = {cop.proj_shift:.4f}")
    return {"results": df, "info": info}


def run_b4_preliminary(
    model: str,
    out_root: str | Path,
    cache_root: str | Path,
    rows_name: str = "pilot",
    runs: int = 1,
    client: LLMClient | None = None,
) -> dict:
    """Fresh official B4 on the pilot rows; scored with the official scorer."""
    from .b4 import generate_b4  # noqa: PLC0415

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = devdata.load_rows(rows_name)
    real_csv = devdata.build_dev_split()[rows_name]
    if client is None:
        client = LLMClient(
            cache_path=Path(cache_root) / "cache.jsonl",
            log_path=Path(cache_root) / "calls.jsonl",
        )

    t0 = time.time()
    sim = generate_b4(client, model, rows, seed=555_000)
    complete = float(np.mean([
        all(v != "" for v in rec.values()) for rec in sim.to_dict("records")
    ]))
    sim.to_csv(out_root / "b4_profiles.csv", index=False)

    results = []
    for run in range(1, runs + 1):
        res = score_run(
            out_root / "b4_profiles.csv", real_csv,
            out_root / f"score_run{run}", scoring_seed=run,
        )
        results.append({
            "run": run, "type1": res.type_scores.get("type1"),
            "type2": res.type_scores.get("type2"), "type3": res.type_scores.get("type3"),
            "aggregate": res.aggregate, "complete_rate": complete,
        })
        print(f"[b4] run{run}: agg={res.aggregate:.3f} t1={res.type_scores.get('type1'):.3f} "
              f"t2={res.type_scores.get('type2'):.3f} t3={res.type_scores.get('type3'):.3f}",
              flush=True)
    df = pd.DataFrame(results)
    df.to_csv(out_root / "b4_results.csv", index=False)
    return {"results": df, "complete_rate": complete, "elapsed_s": time.time() - t0}

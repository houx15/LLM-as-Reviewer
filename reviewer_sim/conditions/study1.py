"""Study 1 (Phase 0): one question × generators × wordings × rules S0–S7.

Expensive calls (candidates, J, S3, S5 pairs, S6, S7 tables) are made once per
wording and cached; each of the 3 runs redraws every selection with a new run
seed and a new scoring seed (SPEC §4.5).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import devdata, diagnostics, rules
from ..fields import input_group_of, input_names, load_fields, scored_input_names
from ..llm import LLMClient
from ..scoring import score_study1

RUNS = (1, 2, 3)
PROMPT_INPUTS = None  # filled lazily: all 9 inputs in official order


def _prompt_inputs() -> list[str]:
    return input_names()


@dataclass
class Study1Config:
    judge_model: str
    generators: dict[str, str]          # name -> model id
    questions: list[str]
    wordings: tuple[str, ...] = ("W1", "W2", "W3")
    n_candidates: int = 10
    rows_name: str = "pilot"
    seed_base: int = 10_000
    max_rows: int | None = None         # cost control: use only first N rows


def run_study1(
    cfg: Study1Config,
    out_root: str | Path,
    cache_root: str | Path,
    client: LLMClient | None = None,
) -> dict:
    out_root = Path(out_root)
    cache_root = Path(cache_root)
    in_names = scored_input_names()          # 8 scored inputs
    prompt_inputs = _prompt_inputs()         # 9, official order
    fs_all = load_fields()

    rows = devdata.load_rows(cfg.rows_name)
    if cfg.max_rows:
        rows = rows.head(cfg.max_rows).reset_index(drop=True)
    real_csv = devdata.build_dev_split()[cfg.rows_name]
    # real reference restricted to the rows we simulate (profile_ids must match)
    real_rows = rows  # rows carry the real outputs; scoring uses profile_id join

    if client is None:
        client = LLMClient(
            cache_path=cache_root / "cache.jsonl",
            log_path=cache_root / "calls.jsonl",
        )

    group_map = {}
    for iname in in_names:
        for v in rows[iname].dropna().unique():
            group_map[(iname, v)] = input_group_of(iname, v)

    results = []
    judge_stats = []
    t0 = time.time()

    for question in cfg.questions:
        fs = fs_all[question]
        q_i = cfg.questions.index(question)
        for gen_name, gen_model in cfg.generators.items():
            g_i = list(cfg.generators).index(gen_name)
            # ---- candidates: once per question × generator (official wording)
            cand_seed = cfg.seed_base + 1000 * (q_i + 1) + 100 * (g_i + 1)
            candidates = rules.generate_candidate_values(
                client, gen_model, rows, question, cfg.n_candidates,
                seed=cand_seed, input_names=prompt_inputs,
            )
            valid_rate = float(np.mean([c != "" for cs in candidates for c in cs]))
            dup_share = float(np.mean(
                [1 - len({c for c in cs if c != ""}) / max(1, sum(c != "" for c in cs))
                 for cs in candidates]
            ))
            print(f"[study1] candidates done: {question}/{gen_name} "
                  f"valid={valid_rate:.3f} dup={dup_share:.3f}", flush=True)

            for wording in cfg.wordings:
                w_i = cfg.wordings.index(wording)
                w_seed = (cand_seed + 7919 * (w_i + 1)) % 2**31

                # ---- per-row expensive calls (J, S3, S6) in parallel
                def row_calls(i, item):
                    _, row = item
                    distinct = [v for v in dict.fromkeys(candidates[i]) if v != ""]
                    if not distinct:
                        distinct = [fs.answers[0]]
                    j_probs = rules.judge_values(
                        client, cfg.judge_model, row, question, distinct,
                        wording, w_seed + 31 * i, prompt_inputs,
                    )
                    plausible = rules.plausibility_filter(
                        client, cfg.judge_model, row, question, distinct,
                        wording, w_seed + 31 * i + 7, prompt_inputs,
                    )
                    stated = rules.state_distribution(
                        client, cfg.judge_model, row, question, wording,
                        w_seed + 31 * i + 13, prompt_inputs,
                    )
                    ri = rules.RuleInputs(
                        candidates=candidates[i], j_probs=j_probs,
                        plausible=plausible, pairwise={}, stated=stated.probs,
                        s7_dist={},
                    )
                    ent = -sum(p * np.log(p + 1e-12) for p in j_probs.values())
                    return ri, ent, len(distinct)

                from ..llm import run_parallel  # local import to avoid cycle

                row_out = run_parallel(
                    row_calls, list(rows.iterrows()),
                    max_workers=client.max_workers,
                    desc=f"rows/{question}/{gen_name}/{wording}",
                )
                rule_inputs = [ri for ri, _, _ in row_out]
                for ri, ent, n_dist in row_out:
                    judge_stats.append({
                        "question": question, "generator": gen_name,
                        "wording": wording, "j_entropy": ent, "n_distinct": n_dist,
                    })

                # ---- S7: stated tables + log-linear fit (8 scored inputs)
                s7_dists, s7_info = rules.s7_row_distributions(
                    client, cfg.judge_model, question, rows, in_names,
                    wording, w_seed + 123_456,
                )
                for ri, d in zip(rule_inputs, s7_dists, strict=True):
                    ri.s7_dist = d

                # ---- runs: redraw selections (parallel over rows, per-cell rng)
                base_cols = ["profile_id", *prompt_inputs]
                for run in RUNS:
                    run_seed = w_seed + 500 * run

                    def row_draws(i, item):
                        _, row = item
                        ri = rule_inputs[i]
                        out = {}
                        for rule in rules.RULES:
                            rng = rules.cell_rng(run_seed, rule, i)

                            def pair_fetch(x, y, i=i, row=row):
                                return rules.pairwise_probability(
                                    client, cfg.judge_model, row, question,
                                    x, y, wording, w_seed + 61 * i + 17,
                                    prompt_inputs,
                                )
                            out[rule] = rules.apply_rule(rule, ri, rng, pair_fetch=pair_fetch)
                        return out

                    all_draws = run_parallel(
                        row_draws, list(rows.iterrows()),
                        max_workers=client.max_workers,
                        desc=f"draws/{question}/{gen_name}/{wording}/run{run}",
                    )

                    # ---- score every rule with a per-run scoring seed
                    for rule in rules.RULES:
                        sim = rows[base_cols].copy()
                        sim[question] = [d[rule] for d in all_draws]
                        run_dir = out_root / question / gen_name / wording / f"run{run}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        sim_path = run_dir / f"sim_{rule}.csv"
                        sim.to_csv(sim_path, index=False)
                        res = score_study1(
                            question, sim_path, real_csv,
                            run_dir / f"score_{rule}", scoring_seed=run,
                            input_names=in_names,
                        )
                        diag = diagnostics.study1_diagnostics(
                            real_rows, sim, question, in_names, group_map,
                        )
                        results.append({
                            "question": question, "generator": gen_name,
                            "wording": wording, "rule": rule, "run": run,
                            "type1": res.type_scores["type1"],
                            "type2": res.type_scores["type2"],
                            "aggregate": res.aggregate,
                            "spread_ratio": diag["spread_ratio_mean"],
                            "collapse_rate": diag["collapse_rate"],
                            "tv_distance": diag["tv_distance"],
                            "assoc_diff": diag["association_diff_mean"],
                            "missing_rate": diag["missing_rate"],
                            "s7_mad": s7_info.get("mad") if rule == "S7" else None,
                        })
                        print(f"[study1] {question}/{gen_name}/{wording}/{rule} "
                              f"run{run}: agg={res.aggregate:.3f} "
                              f"t1={res.type_scores['type1']:.3f} "
                              f"t2={res.type_scores['type2']:.3f}", flush=True)

    df = pd.DataFrame(results)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "study1_results.csv", index=False)
    pd.DataFrame(judge_stats).to_csv(out_root / "study1_judge_stats.csv", index=False)
    meta = {
        "config": {k: (dict(v) if isinstance(v, dict) else v) for k, v in asdict(cfg).items()},
        "n_rows": len(rows), "elapsed_s": time.time() - t0,
        "valid_rate": valid_rate, "dup_share": dup_share,
        "s7_info": s7_info,
    }
    (out_root / "study1_meta.json").write_text(json.dumps(meta, indent=1))
    return {"results": df, "meta": meta}

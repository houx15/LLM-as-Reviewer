"""Phase 2 (Study 2 pilot): whole profiles across the readout matrix, on the
500 pilot rows, plus the P5 population builder. All conditions write one
sim_profiles CSV per run in the official format and are scored with the
official Types 1-3. Expensive calls are made once per wording and cached;
selections are redrawn per run (SPEC §6.6).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import devdata
from ..copula import collect_output_tables, sample_stated_copula
from ..fields import input_names, load_fields, output_names, scored_input_names
from ..llm import LLMClient, run_parallel
from ..parse import extract_json
from ..profilejudge import judge_profiles, select_random, select_s4
from ..prompts import b6_batch_prompt, official_generation_prompt, r3_graph_prompt
from ..rules import (
    SYSTEM_JSON,
    RuleInputs,
    _validate_answer,
    apply_rule,
    state_distribution,
)
from ..scoring import score_run
from ..tables import output_group_phrase

RUNS = (1, 2, 3)
SCORING_B = 100          # official master setting; tests lower this


def _score_run(sim_csv, real_csv, out_dir, run, b=None):
    from ..scoring import score_run as _sr

    return _sr(sim_csv, real_csv, out_dir, scoring_seed=run, B=b or SCORING_B)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _validate_profile(raw: str, fs) -> dict[str, str]:
    try:
        obj = extract_json(raw)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for q in output_names():
        v = obj.get(q)
        out[q] = _validate_answer(json.dumps({q: v}), fs[q]) if v is not None else ""
    return out


def _complete(rec: dict[str, str]) -> bool:
    return bool(rec) and all(v != "" for v in rec.values())


def score_condition(
    sim: pd.DataFrame,
    real_csv: Path,
    out_dir: Path,
    run: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_path = out_dir / "sim_profiles.csv"
    sim.to_csv(sim_path, index=False)
    res = _score_run(sim_path, real_csv, out_dir / "score", run)
    return {
        "run": run,
        "type1": res.type_scores.get("type1"),
        "type2": res.type_scores.get("type2"),
        "type3": res.type_scores.get("type3"),
        "aggregate": res.aggregate,
    }


# --------------------------------------------------------------------------- #
# B6: population batch generation
# --------------------------------------------------------------------------- #


def run_b6(client, model, rows, real_csv, out_root, seed, batch_size=10):
    from ..references import coarse_key

    out_root = Path(out_root)
    fs = load_fields()
    in_names = input_names()
    # group rows by coarse group; batches of 10 within group
    keys = [coarse_key(r) for _, r in rows.iterrows()]
    order = np.argsort([str(k) for k in keys], kind="stable")
    results = {int(rows.iloc[i]["profile_id"]): None for i in order}

    batches: list[list[int]] = []
    by_group: dict[tuple, list[int]] = {}
    for i in order:
        by_group.setdefault(keys[i], []).append(int(i))
    for _, idxs in by_group.items():
        for s in range(0, len(idxs), batch_size):
            batches.append(idxs[s : s + batch_size])

    def gen_batch(batch, seed_b, split=False):
        input_rows = [
            {k: rows.iloc[i][k] for k in in_names if pd.notna(rows.iloc[i][k])}
            for i in batch
        ]
        prompt, meta = b6_batch_prompt(input_rows)
        res = client.chat(
            [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
            model=model,
            json_mode=True,
            seed=seed_b,
            meta={**meta, "batch": batch, "split": split},
            purpose="b6_batch",
        )
        try:
            arr = json.loads(res.content)
            if isinstance(arr, dict):
                arr = arr.get("profiles", arr.get("individuals", [arr]))
        except json.JSONDecodeError:
            arr = []
        out = []
        for obj in arr if isinstance(arr, list) else []:
            raw = json.dumps(obj)
            out.append(_validate_profile(raw, fs))
        return out

    for b_i, batch in enumerate(batches):
        seed_b = seed + 31 * b_i
        profiles = gen_batch(batch, seed_b)
        if len(profiles) != len(batch) or not all(map(_complete, profiles)):
            # retry once, then split in two (SPEC §6.1)
            profiles2 = gen_batch(batch, seed_b + 7)
            if len(profiles2) == len(batch) and all(map(_complete, profiles2)):
                profiles = profiles2
            elif len(batch) > 1:
                mid = len(batch) // 2
                h1 = gen_batch(batch[:mid], seed_b + 11)
                h2 = gen_batch(batch[mid:], seed_b + 13)
                profiles = h1 + h2
        # assign whatever valid profiles we have, in order; missing stay NaN
        for i, prof in zip(batch, profiles, strict=False):
            if _complete(prof):
                results[int(rows.iloc[i]["profile_id"])] = prof

    base = rows.copy()
    for q in output_names():
        base[q] = [results[int(pid)].get(q, pd.NA) if results[int(pid)] else pd.NA
                   for pid in rows["profile_id"]]
    out = []
    for run in RUNS:
        out.append(score_condition(base, real_csv, out_root / f"run{run}", run))
    return pd.DataFrame(out), base


# --------------------------------------------------------------------------- #
# B5: independent stating per output
# --------------------------------------------------------------------------- #


def run_b5(client, model, rows, real_csv, out_root, wording, seed):
    out_root = Path(out_root) / wording
    outs = output_names()

    def one_row(i, item):
        _, row = item
        draws = {}
        for oi, q in enumerate(outs):
            dist = state_distribution(
                client, model, row, q, wording, seed + 101 * i + 7 * oi, input_names()
            )
            items = [(v, p) for v, p in dist.probs.items() if p > 0]
            draws[q] = _weighted(items, random.Random(seed + 101 * i + 7 * oi + 1000 * run_i))
        return draws

    all_results = []
    for run in RUNS:
        run_i = run
        t0 = time.time()
        draws_list = run_parallel(
            one_row, list(rows.iterrows()),
            max_workers=client.max_workers,
            desc=f"b5/{wording}/run{run}",
        )
        sim = rows.copy()
        for q in outs:
            sim[q] = [d[q] for d in draws_list]
        r = score_condition(sim, real_csv, out_root / f"run{run}", run)
        r.update({"condition": "B5", "wording": wording})
        all_results.append(r)
        print(f"[phase2] B5/{wording} run{run}: agg={r['aggregate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(all_results)


def _weighted(items, rng):
    vals = [v for v, _ in items]
    ws = np.array([w for _, w in items])
    if ws.sum() <= 0:
        return rng.choice(vals)
    return str(rng.choices(vals, weights=(ws / ws.sum()).tolist(), k=1)[0])


# --------------------------------------------------------------------------- #
# R3: joint stating along a proposed graph
# --------------------------------------------------------------------------- #


def propose_graph(client, model, seed, max_attempts=3) -> dict[str, list[str]]:
    prompt, meta = r3_graph_prompt()
    outs = output_names()
    for attempt in range(max_attempts):
        res = client.chat(
            [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
            model=model,
            json_mode=True,
            seed=seed + attempt,
            meta={**meta, "attempt": attempt},
            purpose="r3_graph",
        )
        try:
            obj = extract_json(res.content)
        except Exception:  # noqa: BLE001
            continue
        graph = {q: [str(p) for p in obj.get(q, []) if str(p) in outs and str(p) != q]
                 for q in outs}
        if _is_dag(graph):
            return graph
    return {q: [] for q in outs}  # fallback: independent (= B5 structure)


def _is_dag(graph: dict[str, list[str]]) -> bool:
    try:
        _topo_order(graph)
        return True
    except ValueError:
        return False


def _topo_order(graph: dict[str, list[str]]) -> list[str]:
    indeg = {q: len(set(graph[q])) for q in graph}
    children: dict[str, list[str]] = {q: [] for q in graph}
    for q, ps in graph.items():
        for p in set(ps):
            children[p].append(q)
    queue = [q for q, d in indeg.items() if d == 0]
    order = []
    while queue:
        n = queue.pop()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(graph):
        raise ValueError("cycle")
    return order


def run_r3(client, model, rows, real_csv, out_root, wording, seed, graph):
    out_root = Path(out_root) / wording
    outs = output_names()
    topo = _topo_order(graph)

    def one_row(i, item):
        _, row = item
        given: dict[str, object] = {}
        for q in topo:
            dist = state_distribution(
                client, model, row, q, wording,
                seed + 211 * i + 7 * outs.index(q) + 100_000,
                input_names(), given=dict(given) if given else None,
            )
            v = _weighted(list(dist.probs.items()), random.Random(seed + 211 * i + 7 * outs.index(q) + 999 * run_i))
            given[q] = v
        return given

    all_results = []
    for run in RUNS:
        run_i = run
        t0 = time.time()
        draws_list = run_parallel(
            one_row, list(rows.iterrows()),
            max_workers=client.max_workers,
            desc=f"r3/{wording}/run{run}",
        )
        sim = rows.copy()
        for q in outs:
            sim[q] = [d[q] for d in draws_list]
        r = score_condition(sim, real_csv, out_root / f"run{run}", run)
        r.update({"condition": "R3", "wording": wording})
        all_results.append(r)
        print(f"[phase2] R3/{wording} run{run}: agg={r['aggregate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(all_results)


# --------------------------------------------------------------------------- #
# P4: stated tables -> copula -> profiles (also P3's candidate source)
# --------------------------------------------------------------------------- #


def run_p4(client, model, rows, real_csv, out_root, wording, seed, tables_dir,
           runs=RUNS):
    out_root = Path(out_root) / wording
    cache = Path(tables_dir) / wording / "tables.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        got = collect_output_tables(
            client, model, rows, scored_input_names(), wording, seed
        )
        payload = {
            "row_marginals": {q: got["row_marginals"][q].tolist() for q in output_names()},
            "corr": got["corr"].tolist(),
            "proj_shift": got["proj_shift"],
            "flags": got["flags"],
            "stated_tables": {q: t for q, t in got["marginals"].items()},
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    row_marginals = {q: np.array(v) for q, v in payload["row_marginals"].items()}
    corr = np.array(payload["corr"])

    all_results = []
    sims = {}
    for run in runs:
        rng = np.random.default_rng(seed + run)
        sim = sample_stated_copula(row_marginals, corr, rows, rng)
        sims[run] = sim
        r = score_condition(sim, real_csv, out_root / f"run{run}", run)
        r.update({"condition": "P4", "wording": wording})
        all_results.append(r)
        print(f"[phase2] P4/{wording} run{run}: agg={r['aggregate']:.3f}", flush=True)
    return pd.DataFrame(all_results), row_marginals, corr, sims


# --------------------------------------------------------------------------- #
# P1 / P2 / P3: candidate pools + whole-profile judge + S4
# --------------------------------------------------------------------------- #


def _generate_pool(client, model, rows, n_candidates, seed, purpose):
    """n complete profiles per row (official prompt, retry loop)."""
    fs = load_fields()
    in_names = input_names()

    def one_row(_i, item):
        _, row = item
        inputs = {k: row[k] for k in in_names if pd.notna(row[k])}
        prompt = official_generation_prompt(inputs)
        profiles = []
        for j in range(n_candidates):
            rec = {}
            for attempt in range(4):
                a_seed = seed + 10_000 * int(row["profile_id"]) + j * 100 + attempt
                res = client.chat(
                    [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
                    model=model,
                    json_mode=True,
                    seed=a_seed,
                    meta={"row": int(row["profile_id"]), "j": j, "attempt": attempt},
                    purpose=purpose,
                )
                rec = _validate_profile(res.content, fs)
                if _complete(rec):
                    break
            profiles.append(rec if _complete(rec) else None)
        return profiles

    return run_parallel(one_row, list(rows.iterrows()),
                        max_workers=client.max_workers, desc=purpose)


def _judge_pool(client, judge_model, rows, pools, wording, seed):
    """Averaged judge probabilities for each row's complete candidates."""
    def one_row(i, item):
        _, row = item
        pool = [p for p in pools[i] if p is not None]
        if not pool:
            return []
        probs = judge_profiles(
            client, judge_model, row, pool, wording,
            seed + 501 * i, input_names(),
        )
        return probs

    return run_parallel(one_row, list(rows.iterrows()),
                        max_workers=client.max_workers, desc=f"judge/{wording}")


def run_judged_pool(
    client, judge_model, rows, real_csv, out_root, name, wording, seed,
    pools, j_probs,
):
    """S4 selection over each row's pool (name: 'P1' | 'P2' | 'P3')."""
    outs = output_names()

    def build(run):
        rng = random.Random(seed + 43 * run)
        sim = rows.copy()
        for q in outs:
            sim[q] = pd.NA
        for i in range(len(rows)):
            pool = [p for p in pools[i] if p is not None]
            probs = j_probs[i]
            if not pool:
                continue
            k = select_s4(probs, rng)
            for q in outs:
                sim.iat[i, sim.columns.get_loc(q)] = pool[k][q]
        return sim

    all_results = []
    for run in RUNS:
        sim = build(run)
        r = score_condition(sim, real_csv, Path(out_root) / wording / f"run{run}", run)
        r.update({"condition": name, "wording": wording})
        all_results.append(r)
        print(f"[phase2] {name}/{wording} run{run}: agg={r['aggregate']:.3f}", flush=True)
    return pd.DataFrame(all_results)


def pool_random_contrast(rows, pools, real_csv, out_root, name, wording, seed):
    """Uniform pick among the same pool (P1-random / P2-random)."""
    outs = output_names()
    all_results = []
    for run in RUNS:
        rng = random.Random(seed + 43 * run + 777)
        sim = rows.copy()
        for q in outs:
            sim[q] = pd.NA
        for i in range(len(rows)):
            pool = [p for p in pools[i] if p is not None]
            if not pool:
                continue
            k = select_random(len(pool), rng)
            for q in outs:
                sim.iat[i, sim.columns.get_loc(q)] = pool[k][q]
        r = score_condition(sim, real_csv, Path(out_root) / wording / f"run{run}", run)
        r.update({"condition": name, "wording": wording})
        all_results.append(r)
        print(f"[phase2] {name}/{wording} run{run}: agg={r['aggregate']:.3f}", flush=True)
    return pd.DataFrame(all_results)


# --------------------------------------------------------------------------- #
# P5: the population builder (SPEC §6.5)
# --------------------------------------------------------------------------- #


def _row_tables(prof: dict, keys_in: dict) -> dict[tuple, dict[str, int]]:
    """Which (output, input-group) cells a profile with given inputs touches."""
    touched = {}
    for q, val in prof.items():
        for iname, cat in keys_in.items():
            touched[(q, iname, cat)] = val
    return touched


def run_p5(
    rows, pools, j_probs, stated_tables, rows_inputs, real_csv, out_root,
    wording, seed, lambdas=(0, 1, 5), warmup=50, scale_rows=100,
):
    """Population builder over P1's pool; weights j(c) * exp(-lam * dD / s).

    stated_tables: {output: {input: {category: {answer: prob}}}} (P4's
    output × input tables for this wording).
    rows_inputs: per-row dict input -> group category.
    dD is 0 for the first `warmup` rows; s = median |dD| over the first
    `scale_rows` rows of the lam=0 pass (logged).
    """
    from ..tables import table_distance

    outs = output_names()
    n = len(rows)

    def build_pass(run, lam, s_scale=None, return_picks=False):
        rng_order = random.Random(seed + 97 * run)
        rng_pick = random.Random(seed + 97 * run + 1)
        order = list(range(n))
        rng_order.shuffle(order)

        # accumulate counts per (output, input, category): {answer: n}
        counts: dict[tuple, dict[str, int]] = {}
        added = 0
        dDs = []
        picks = []
        sim = rows.copy()
        for q in outs:
            sim[q] = pd.NA

        for pos, i in enumerate(order):
            pool = [p for p in pools[i] if p is not None]
            probs = j_probs[i]
            if not pool:
                picks.append(-1)
                continue
            keys_in = rows_inputs[i]

            def dd(c):
                # mean TV change over stated tables for the row's input groups
                ds = []
                for q in outs:
                    for iname, cat in keys_in.items():
                        st = stated_tables.get(q, {}).get(iname, {}).get(cat)
                        if st is None:
                            continue
                        cur = counts.get((q, iname, cat), {})
                        n_cur = sum(cur.values())
                        d_before = _tv(st, cur, n_cur)
                        cur2 = dict(cur)
                        cur2[c[q]] = cur2.get(c[q], 0) + 1
                        d_after = _tv(st, cur2, n_cur + 1)
                        ds.append(d_after - d_before)
                return float(np.mean(ds)) if ds else 0.0

            if added < warmup:
                weights = probs
            else:
                # DeltaD computed on every pass (needed for the s scale at lam=0)
                dds = [dd(c) for c in pool]
                dDs.extend(abs(d) for d in dds)
                if lam == 0 or s_scale is None or s_scale <= 0:
                    weights = probs
                else:
                    weights = [
                        p * np.exp(-lam * d / s_scale) for p, d in zip(probs, dds, strict=True)
                    ]
            w = np.array(weights, dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(pool))
            k = int(rng_pick.choices(range(len(pool)), weights=(w / w.sum()).tolist(), k=1)[0])
            picks.append(k)
            chosen = pool[k]
            for q in outs:
                sim.iat[i, sim.columns.get_loc(q)] = chosen[q]
            for q in outs:
                for iname, cat in keys_in.items():
                    key = (q, iname, cat)
                    cur = counts.setdefault(key, {})
                    cur[chosen[q]] = cur.get(chosen[q], 0) + 1
            added += 1
        if return_picks:
            return sim, dDs, picks
        return sim, dDs, None

    # derive the scale s from the lam=0 pass, and record each run's lam=0 picks
    picks0: dict[int, list[int]] = {}
    dDs0_all = []
    for run in RUNS:
        sim0, dDs0, picks = build_pass(run, 0, return_picks=True)
        picks0[run] = picks
        dDs0_all.extend(dDs0)
    s_scale = float(np.median(dDs0_all[: 3 * scale_rows])) if dDs0_all else 1.0

    all_results = []
    for lam in lambdas:
        for run in RUNS:
            sim, dDs, picks = build_pass(
                run, lam, s_scale=s_scale if lam > 0 else None, return_picks=True
            )
            changed = sum(
                1 for a, b in zip(picks, picks0[run], strict=True) if a != b
            ) if lam > 0 else 0
            r = score_condition(
                sim, real_csv, Path(out_root) / wording / f"lam{lam}" / f"run{run}", run
            )
            r.update({
                "condition": "P5", "wording": wording, "lambda": lam,
                "picks_changed_vs_lam0": changed,
            })
            all_results.append(r)
            print(f"[phase2] P5/{wording}/lam{lam} run{run}: agg={r['aggregate']:.3f} "
                  f"changed={changed}", flush=True)
    return pd.DataFrame(all_results), {"s_scale": s_scale}


def _tv(st: dict[str, float], counts: dict[str, int], n: int) -> float:
    if n == 0:
        return 0.5  # maximal distance from any stated distribution
    emp = {k: v / n for k, v in counts.items()}
    keys = set(st) | set(emp)
    return 0.5 * sum(abs(st.get(k, 0.0) - emp.get(k, 0.0)) for k in keys)


# --------------------------------------------------------------------------- #
# P5-view: the judge sees current population tables + stated targets
# --------------------------------------------------------------------------- #


def run_p5_view(
    client, judge_model, rows, pools, j_probs, stated_tables, rows_inputs,
    real_csv, out_root, wording, seed, warmup=50, n_orders=2,
):
    """Sequential builder where the judge sees the population state directly.

    One call per row (two orders averaged); its stated probabilities replace
    j(c) * exp(-lam dD). Warmup rows use P1's judge probabilities."""
    from ..prompts import judge_profiles_with_tables

    outs = output_names()
    n = len(rows)
    all_results = []
    for run in RUNS:
        rng_order = random.Random(seed + 59 * run)
        rng_pick = random.Random(seed + 59 * run + 1)
        order = list(range(n))
        rng_order.shuffle(order)
        counts: dict[tuple, dict[str, int]] = {}
        added = 0
        sim = rows.copy()
        for q in outs:
            sim[q] = pd.NA
        for i in order:
            pool = [p for p in pools[i] if p is not None]
            probs = j_probs[i]
            if not pool:
                continue
            keys_in = rows_inputs[i]
            if added < warmup:
                weights = probs
            else:
                # current empirical table vs stated table, rendered per output
                summary, target = [], []
                for q in outs:
                    for iname, cat in keys_in.items():
                        st = stated_tables.get(q, {}).get(iname, {}).get(cat)
                        if st is None:
                            continue
                        cur = counts.get((q, iname, cat), {})
                        n_cur = sum(cur.values())
                        emp = {k: v / n_cur for k, v in cur.items()} if n_cur else {}
                        summary.append(
                            f"{q} | {iname}={cat} (n={n_cur}): "
                            + ", ".join(f"{k}={emp.get(k, 0):.2f}" for k in st)
                        )
                        target.append(
                            f"{q} | {iname}={cat}: "
                            + ", ".join(f"{k}={p:.2f}" for k, p in st.items())
                        )
                probs_v = None
                if summary:
                    for o in range(n_orders):
                        prompt, meta = judge_profiles_with_tables(
                            {k: rows.iloc[i][k] for k in _prompt_inputs()},
                            pool, "\n".join(summary), "\n".join(target),
                            seed + 59 * run + 100 * o + 7 * i,
                        )
                        res = client.chat(
                            [
                                {"role": "system", "content": SYSTEM_JSON},
                                {"role": "user", "content": prompt},
                            ],
                            model=judge_model,
                            json_mode=True,
                            seed=seed + 59 * run + 100 * o + 7 * i,
                            meta={**meta, "row": int(rows.iloc[i]["profile_id"]),
                                  "order": o},
                            purpose="p5view",
                        )
                        from ..parse import ParseError, parse_distribution, uniform  # noqa: PLC0415

                        labels = list(meta["label_of"].values())
                        try:
                            dist = parse_distribution(res.content, labels)
                        except ParseError:
                            dist = uniform(labels)
                        inv = {lab: int(x) for x, lab in meta["label_of"].items()}
                        v = np.zeros(len(pool))
                        for lab, p in dist.probs.items():
                            v[inv[lab]] += p / n_orders
                        probs_v = v if probs_v is None else probs_v + v
                weights = probs_v if probs_v is not None and probs_v.sum() > 0 else probs
            w = np.array(weights, dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(pool))
            k = int(rng_pick.choices(range(len(pool)), weights=(w / w.sum()).tolist(), k=1)[0])
            chosen = pool[k]
            for q in outs:
                sim.iat[i, sim.columns.get_loc(q)] = chosen[q]
            for q in outs:
                for iname, cat in keys_in.items():
                    cur = counts.setdefault((q, iname, cat), {})
                    cur[chosen[q]] = cur.get(chosen[q], 0) + 1
            added += 1
        r = score_condition(sim, real_csv, Path(out_root) / wording / f"run{run}", run)
        r.update({"condition": "P5-view", "wording": wording})
        all_results.append(r)
        print(f"[phase2] P5-view/{wording} run{run}: agg={r['aggregate']:.3f}", flush=True)
    return pd.DataFrame(all_results)


_PROMPT_INPUTS = None


def _prompt_inputs() -> list[str]:
    global _PROMPT_INPUTS  # noqa: PLW0603
    if _PROMPT_INPUTS is None:
        from ..fields import input_names  # noqa: PLC0415

        _PROMPT_INPUTS = input_names()
    return _PROMPT_INPUTS

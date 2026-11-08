"""Study 3 (SPEC §7): measuring the judge directly, alongside Phase 2.

Six tests on development evaluation-part items; reference models fitted on the
fitting part (data/study3_refs/). Nothing here feeds generation. W1-style
diagnostic prompts are constructed from the official framing.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import devdata
from ..fields import input_names, load_fields, output_names, scored_input_names
from ..llm import LLMClient
from ..parse import ParseError, parse_single_probability
from ..prompts import official_framing
from ..rules import SYSTEM_JSON, _row_inputs
from ..references import coarse_key, fit_marginals, group_marginal
from .. import study3refs

REPO = Path(__file__).resolve().parents[2]
N_ITEMS = 300


def _which_is_real_prompt(row, prof_a: dict, prof_b: dict, seed) -> tuple[str, dict]:
    import random  # noqa: PLC0415

    from ..prompts import label_candidates  # noqa: PLC0415

    fs = [f for f in load_fields().values() if f.role == "y"]
    rng = random.Random(int(seed))
    if rng.random() < 0.5:
        a, b = prof_a, prof_b
        real_is = "A"
    else:
        a, b = prof_b, prof_a
        real_is = "B"
    framing = official_framing(_row_inputs(row, input_names()), fs)
    import json as _json  # noqa: PLC0415

    block = (
        "Profile A: " + _json.dumps(a, ensure_ascii=False) + "\n"
        "Profile B: " + _json.dumps(b, ensure_ascii=False)
    )
    w = (
        "Exactly one of the two profiles below is a real respondent's answers and the "
        "other has exactly one answer replaced by a draw from that answer's overall "
        "distribution. Which profile is the real one? Give the probability that A is "
        "the real one."
    )
    instr = 'Return JSON only: {"A": p}.'
    meta = {"family": "s3_discrimination", "real_is": real_is, "seed": seed}
    return framing + "\n\n" + w + "\n\n" + block + "\n\n" + instr, meta


def _calibration_prompt(row, given: dict, question, value, seed) -> tuple[str, dict]:
    fs = load_fields()
    framing = official_framing({**_row_inputs(row, input_names()), **given}, [fs[question]])
    w = (
        f'For a person with exactly the attributes above, give the probability that '
        f'their answer to "{fs[question].description}" is {value!r}.'
    )
    instr = 'Return JSON only: {"p": prob}.'
    meta = {"family": "s3_calibration", "question": question, "value": value, "seed": seed}
    return framing + "\n\n" + w + "\n\n" + instr, meta


def _population_prompt(
    question_summaries_a: str, question_summaries_b: str, seed
) -> tuple[str, dict]:
    import random  # noqa: PLC0415

    ctx = load_context_str()
    rng = random.Random(int(seed))
    if rng.random() < 0.5:
        a, b, real_is = "real", "synthetic", "A"
    else:
        a, b, real_is = "synthetic", "real", "B"
    w = (
        "Two sets of summary statistics (marginals and cross-tabulations) for one "
        "group of respondents are shown. One set is computed from real survey data; "
        "the other from synthetic data. Which set is the real one? Give the "
        "probability that Set A is the real one, then list up to 5 cells that most "
        "give it away."
    )
    block = (
        f"Set A (computed from {a} data):\n{question_summaries_a}\n\n"
        f"Set B (computed from {b} data):\n{question_summaries_b}"
    )
    instr = (
        'Return JSON only: {"A": p, "cells": ["var=category", ...]} where p is the '
        "probability that Set A is real."
    )
    meta = {"family": "s3_population", "real_is": real_is, "seed": seed}
    return ctx + "\n\n" + w + "\n\n" + block + "\n\n" + instr, meta


def load_context_str() -> str:
    from ..fields import load_context  # noqa: PLC0415

    c = load_context()
    return f"Population: {c['context']} (reference year {c['reference_year']})."


def run_study3(
    client: LLMClient,
    judge_model: str,
    out_root: str | Path,
    b4_profiles: pd.DataFrame,
    seed: int = 700_000,
    n_items: int = N_ITEMS,
) -> dict:
    import random  # noqa: PLC0415

    from ..tables import ask_table_row  # noqa: PLC0415

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    ev = devdata.load_rows("evaluation")
    fit = devdata.load_rows("fitting")
    marg = fit_marginals(fit)

    with open(REPO / "data" / "study3_refs" / "bn.pkl", "rb") as f:
        bn = pickle.load(f)
    with open(REPO / "data" / "study3_refs" / "gbm.pkl", "rb") as f:
        gbm = pickle.load(f)

    outs = output_names()
    results: dict = {}

    # ---- 1. discrimination -------------------------------------------------
    item_rows = ev.iloc[rng.choice(len(ev), size=n_items, replace=False)]
    judge_p_real, gbm_p_real = [], []
    per_output: dict[str, list] = {q: [] for q in outs}
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from ..study3refs import ordinal_code  # noqa: PLC0415

    for i, (_, row) in enumerate(item_rows.iterrows()):
        real_prof = {q: _fmt(row[q]) for q in outs}
        q = outs[int(rng.integers(len(outs)))]
        m, _lvl = group_marginal(marg, coarse_key(row), q)
        vals = list(m)
        if len(vals) < 2 or pd.isna(row[q]):
            continue
        alt_v = str(rng.choice(vals, p=[m[v] for v in vals]))
        if alt_v == real_prof[q]:
            continue
        fake_prof = dict(real_prof)
        fake_prof[q] = alt_v
        # judge: two label orders averaged
        ps = []
        meta = None
        for o in range(2):
            prompt, meta = _which_is_real_prompt(row, real_prof, fake_prof, seed + i * 2 + o)
            res = client.chat(
                [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
                model=judge_model, json_mode=True, seed=seed + i * 2 + o,
                meta={**meta, "item": i, "order": o}, purpose="s3_discrimination",
            )
            try:
                p = parse_single_probability(res.content, "A")
            except (ParseError, ValueError):
                p = 0.5
            if meta["real_is"] == "B":
                p = 1.0 - p
            ps.append(float(p))
        p_real = float(np.clip(np.mean(ps), 1e-6, 1 - 1e-6))
        judge_p_real.append((p_real, q))
        per_output[q].append(p_real)
        # GBM reference on the same item (score = P(real) for both rows)
        alt_row = row.copy()
        alt_row[q] = alt_v
        Xr = ordinal_code(pd.DataFrame([row]))
        Xf = ordinal_code(pd.DataFrame([alt_row]))
        X = pd.concat([Xr, Xf], ignore_index=True).fillna(-1.0)
        X = X.reindex(columns=gbm.feature_names_in_, fill_value=-1.0)
        p_swap = gbm.predict_proba(X)[:, 1]
        gbm_p_real.append((float(1 - p_swap[0]), float(1 - p_swap[1])))

    def _auc_from_p_real(ps):
        if not ps:
            return float("nan")
        y = np.r_[np.ones(len(ps)), np.zeros(len(ps))]
        s = np.r_[ps, [1 - p for p in ps]]
        return float(roc_auc_score(y, s))

    auroc_judge = _auc_from_p_real([p for p, _ in judge_p_real])
    g_pairs = [(r, f) for r, f in gbm_p_real]
    y = np.r_[np.ones(len(g_pairs)), np.zeros(len(g_pairs))]
    s = np.r_[[r for r, _ in g_pairs], [f for _, f in g_pairs]]
    auroc_gbm = float(roc_auc_score(y, s)) if g_pairs else float("nan")
    results["discrimination"] = {
        "judge_auroc": auroc_judge, "gbm_auroc": auroc_gbm,
        "gate": 0.5 + (auroc_gbm - 0.5) / 2, "n_items": len(judge_p_real),
        "pass": auroc_judge >= 0.5 + (auroc_gbm - 0.5) / 2,
        "per_output_auc": {
            q: _auc_from_p_real(ps)
            for q, ps in per_output.items() if len(ps) >= 10
        },
    }

    # ---- 2. calibration ----------------------------------------------------
    import statsmodels.api as sm  # noqa: PLC0415

    xs_j, xs_b, cal_per_output = [], [], {q: [] for q in outs}
    item_rows2 = ev.iloc[rng.choice(len(ev), size=n_items, replace=False)]
    for i, (_, row) in enumerate(item_rows2.iterrows()):
        q = outs[int(rng.integers(len(outs)))]
        cond = [o for o in outs if o != q][:2]
        given = {o: _fmt(row[o]) for o in cond if pd.notna(row[o])}
        if pd.isna(row[q]):
            continue
        value = _fmt(row[q])
        p_j = None
        for o in range(2):
            prompt, meta = _calibration_prompt(
                row, given, q, value, seed + 10_000 + i * 2 + o
            )
            res = client.chat(
                [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
                model=judge_model, json_mode=True, seed=seed + 10_000 + i * 2 + o,
                meta=meta, purpose="s3_calibration",
            )
            try:
                obj = json.loads(res.content)
                v = float(obj.get("p", obj.get("probability", np.nan)))
                p_j = v if o == 0 else (p_j + v) / 2
            except (json.JSONDecodeError, TypeError, ValueError):
                p_j = p_j if p_j is not None else 0.5
        # BN reference
        try:
            evd = {**{k: str(row[k]) for k in scored_input_names() if pd.notna(row[k])},
                   **given}
            # bin the conditioning outputs like the BN's discretization
            if "age" in evd:
                from ..references import age_band  # noqa: PLC0415

                evd["age"] = age_band(float(evd["age"]))
            for o in cond:
                evd[o] = _bn_state(o, evd[o])
            qd = study3refs.bn_query(bn, evd, q)
            p_b = qd.get(_bn_state(q, value), 0.0)
        except Exception:  # noqa: BLE001
            p_b = 0.5
        pj = float(np.clip(p_j, 1e-4, 1 - 1e-4))
        pb = float(np.clip(p_b, 1e-4, 1 - 1e-4))
        xs_j.append(np.log(pj / (1 - pj)))
        xs_b.append(np.log(pb / (1 - pb)))
        cal_per_output[q].append((pj, pb))
    X = sm.add_constant(np.array(xs_b))
    model = sm.OLS(np.array(xs_j), X).fit()
    results["calibration"] = {
        "slope": float(model.params[1]), "r2": float(model.rsquared),
        "n_items": len(xs_j), "pass": 0.5 <= model.params[1] <= 1.5,
    }

    # ---- 3. typicality trap --------------------------------------------------
    # B4's modal answer per coarse group vs the real modal (evaluation part)
    b4 = b4_profiles.copy()
    b4["key"] = [str(coarse_key(r)) for _, r in b4.iterrows()]
    ev2 = ev.copy()
    ev2["key"] = [str(coarse_key(r)) for _, r in ev2.iterrows()]
    trap_items = []
    for key, grp in b4.groupby("key"):
        if key not in set(ev2["key"]):
            continue
        rgrp = ev2[ev2["key"] == key]
        for q in outs:
            b4_modal = grp[q].mode()
            real_modal = rgrp[q].dropna().mode()
            if len(b4_modal) == 0 or len(real_modal) == 0:
                continue
            b4m, rm = str(b4_modal.iloc[0]), str(real_modal.iloc[0])
            if b4m != rm:
                trap_items.append((key, q, b4m, rm))
    from ..rules import pairwise_probability  # noqa: PLC0415

    correct, n_tried = [], 0
    rng_t = np.random.default_rng(7)
    order_idx = rng_t.permutation(len(item_rows))
    for i in order_idx:
        if n_tried >= 200 or not trap_items:
            break
        i = int(i)
        row = item_rows.iloc[i]
        key, q, b4m, rm = trap_items[int(rng_t.integers(len(trap_items)))]
        p_b4 = pairwise_probability(
            client, judge_model, row, q, b4m, rm, "W1",
            seed + 20_000 + i, input_names(),
        )
        n_tried += 1
        correct.append(p_b4 < 0.5)  # correct = judge prefers the real-common answer
    acc = float(np.mean(correct)) if correct else float("nan")
    results["typicality"] = {
        "accuracy": acc, "n_pairs": n_tried, "n_trap_groups": len(trap_items),
        "pass": acc > 0.5 if acc == acc else False,
    }

    # ---- 4. order and wording stability --------------------------------------
    flips_order, flips_wording, n_stab = 0, 0, 0
    from ..rules import judge_values  # noqa: PLC0415

    stab_rows = item_rows.head(100)
    for i, (_, row) in enumerate(stab_rows.iterrows()):
        vals = [f.name for f in load_fields().values() if f.role == "y"]
        q = outs[int(rng.integers(len(outs)))]
        m, _lvl = group_marginal(marg, coarse_key(row), q)
        distinct = list(m)[:3]
        if len(distinct) < 2:
            continue
        j1 = judge_values(client, judge_model, row, q, distinct, "W1", seed + 30_000 + i, input_names(), n_orders=1)
        j2 = judge_values(client, judge_model, row, q, distinct, "W1", seed + 30_000 + i + 1, input_names(), n_orders=1)
        j3 = judge_values(client, judge_model, row, q, distinct, "W2", seed + 30_000 + i + 2, input_names(), n_orders=1)
        a1 = max(j1, key=j1.get)
        a2 = max(j2, key=j2.get)
        a3 = max(j3, key=j3.get)
        flips_order += int(a1 != a2)
        flips_wording += int(a1 != a3)
        n_stab += 1
    results["stability"] = {
        "order_flip_share": flips_order / max(n_stab, 1),
        "wording_flip_share": flips_wording / max(n_stab, 1),
        "n_items": n_stab,
    }

    # ---- 5. coherence ---------------------------------------------------------
    pairs_idx = [(i, j) for i in range(len(outs)) for j in range(i + 1, len(outs))]
    sel = [pairs_idx[k] for k in rng.choice(len(pairs_idx), size=50, replace=False)]
    devs = []
    for k, (i, j) in enumerate(sel):
        qi, qj = outs[i], outs[j]
        fsi, fsj = load_fields()[qi], load_fields()[qj]
        p_ij = ask_table_row(client, judge_model, qj, f"respondents who answered '{fsi.answers[0]}' to '{fsi.description}'", "W1", seed + 40_000 + k)
        p_ji = ask_table_row(client, judge_model, qi, f"respondents who answered '{fsj.answers[0]}' to '{fsj.description}'", "W1", seed + 40_000 + k + 500)
        m_i = ask_table_row(client, judge_model, qi, "all respondents", "W1", seed + 40_000 + k + 1000)
        m_j = ask_table_row(client, judge_model, qj, "all respondents", "W1", seed + 40_000 + k + 1500)
        pij = p_ij[0].probs[fsj.answers[0]]
        pji = p_ji[0].probs[fsi.answers[0]]
        pi = m_i[0].probs[fsi.answers[0]]
        pj = m_j[0].probs[fsj.answers[0]]
        # Bayes: P(X=x|Y=y) P(Y=y) = P(Y=y|X=x) P(X=x)
        devs.append(abs(pij * pj - pji * pi))
    results["coherence"] = {"mean_abs_bayes_deviation": float(np.mean(devs)), "n_pairs": len(devs)}

    # ---- 6. population discrimination ----------------------------------------
    # real evaluation tables vs method synthetic tables, per coarse group
    methods = {
        "B4": b4_profiles,
    }
    p4_dir = REPO / "runs" / "phase2" / "P4" / "W1" / "run1"
    if (p4_dir / "sim_profiles.csv").exists():
        methods["P4"] = pd.read_csv(p4_dir / "sim_profiles.csv")
    b5_dir = REPO / "runs" / "phase2" / "B5" / "W1" / "run1"
    if (b5_dir / "sim_profiles.csv").exists():
        methods["B5"] = pd.read_csv(b5_dir / "sim_profiles.csv")

    groups = [k for k, _ in ev2.groupby("key") if len(ev2[ev2["key"] == k]) >= 30][:20]
    correct_pop, flagged_total = 0, 0
    n_pop = 0
    for mname, mdf in methods.items():
        for g_i, key in enumerate(groups):
            sub_r = ev2[ev2["key"] == key]
            mdf2 = mdf.copy()
            mdf2["key"] = [str(coarse_key(r)) for _, r in mdf2.iterrows()]
            sub_m = mdf2[mdf2["key"] == key]
            if len(sub_m) < 30:
                continue
            summ_r = _group_summary(sub_r)
            summ_m = _group_summary(sub_m)
            prompt, meta = _population_prompt(summ_r, summ_m, seed + 50_000 + g_i)
            res = client.chat(
                [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
                model=judge_model, json_mode=True, seed=seed + 50_000 + g_i,
                meta={**meta, "method": mname}, purpose="s3_population",
            )
            try:
                obj = json.loads(res.content)
                p_a = float(obj.get("A", 0.5))
            except (json.JSONDecodeError, TypeError, ValueError):
                p_a = 0.5
            guess_real = "A" if p_a >= 0.5 else "B"
            correct_pop += int(guess_real == meta["real_is"])
            n_pop += 1
    results["population_discrimination"] = {
        "accuracy": correct_pop / max(n_pop, 1), "n_items": n_pop,
        "methods": list(methods),
    }

    (out_root / "study3_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)
    return results


def _fmt(v) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _bn_state(q: str, v: str) -> str:
    from ..study3refs import bin_age_first_childbirth, bin_mental_health  # noqa: PLC0415

    if q == "mental_health":
        return bin_mental_health(v)
    if q == "age_first_childbirth":
        return bin_age_first_childbirth(v)
    return v


def _group_summary(df: pd.DataFrame, max_vars: int = 12) -> str:
    lines = []
    for q in output_names()[:max_vars]:
        vc = df[q].dropna().value_counts(normalize=True)
        if len(vc) == 0:
            continue
        lines.append(q + ": " + ", ".join(f"{k}={v:.2f}" for k, v in vc.items()))
    return "\n".join(lines)

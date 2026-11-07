"""Gaussian copula from stated tables (P4, P3 candidates; SPEC §6.3).

- Marginals: p(y_k | x) from the log-linear fit to the 8 stated
  output × input tables per output (the S7 machinery).
- Dependence: each stated output × output table becomes one correlation:
  ordinal/numeric pairs get the code-Pearson correlation of the stated joint;
  pairs involving a nominal output get Cramér's V times the sign of the
  code-Pearson [default; documented]. The 21 × 21 matrix is assembled with a
  unit diagonal and projected to the nearest positive-definite matrix; the
  projection distance is reported.
- Sampling: per row, draw z ~ N(0, R), map each coordinate through the inverse
  CDF of that row's stated marginal. Config level order defines the latent
  ordering of nominal categories (same encoding as reference B2).

Stated tables are stored under data/stated_tables/<wording>/ and reused by
P3 and P5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fields import load_fields, output_names
from .loglinear import build_design_matrix, fit_loglinear
from .tables import ask_table_row, output_group_phrase, stated_input_tables


@dataclass
class StatedCopula:
    marginals: np.ndarray          # (n_rows, n_outputs, max_categories) padded
    category_index: dict[str, list[str]]  # output -> ordered categories
    corr: np.ndarray               # (n_outputs, n_outputs)
    proj_shift: float
    table_flags: dict


def _code_pearson(table: pd.DataFrame) -> float:
    """Pearson correlation of integer-coded row/col categories on the stated
    joint table (rows = conditioning output, cols = target output)."""
    ri = np.arange(len(table.index))
    ci = np.arange(len(table.columns))
    p = table.values / table.values.sum()
    xr = (ri[:, None] * np.ones_like(p)).ravel()
    yc = (np.ones_like(p) * ci[None, :]).ravel()
    w = p.ravel()
    if xr.std() == 0 or yc.std() == 0:
        return 0.0
    return float(np.cov(xr, yc, aweights=w)[0, 1] / (xr.std() * yc.std()))


def _cramers_v_from_table(table: pd.DataFrame) -> float:
    obs = table.values
    n = obs.sum()
    if n == 0 or min(obs.shape) < 2:
        return 0.0
    expected = np.outer(obs.sum(1), obs.sum(0)) / n
    chi2 = ((obs - expected) ** 2 / (expected + 1e-12)).sum()
    q = min(obs.shape) - 1
    return float(np.sqrt(chi2 / (n * q)))


def stated_pair_correlation(table: pd.DataFrame, target_nominal: bool, cond_nominal: bool) -> float:
    """One correlation from one stated output × output table."""
    r = _code_pearson(table)
    if target_nominal or cond_nominal:
        v = _cramers_v_from_table(table)
        sign = np.sign(r) if r != 0 else 1.0
        return float(sign * v)
    return r


def collect_output_tables(
    client, model: str, rows, input_names, wording: str, seed: int
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, pd.DataFrame], dict]:
    """Stated output × input tables (log-linear marginals) and output × output
    tables (dependence), all for one wording."""
    outs = output_names()
    fs = load_fields()

    # ---- output × input tables -> per-row marginals
    marginals: dict[str, dict[str, dict[str, float]]] = {}
    flags = {"fallback": 0, "renormalized": 0, "calls": 0}
    from .fields import input_group_of, input_group_values_with_missing

    rows_inputs = [
        {k: input_group_of(k, row[k]) for k in input_names} for _, row in rows.iterrows()
    ]
    cats = {k: input_group_values_with_missing(k, rows) for k in input_names}
    X, features = build_design_matrix(rows_inputs, input_names, cats)

    row_marginals: dict[str, np.ndarray] = {}
    for oi, q in enumerate(outs):
        tables, tflags = stated_input_tables(
            client, model, q, rows, input_names, wording, seed + 10_000 * oi
        )
        for k, v in tflags.items():
            flags[k] = flags.get(k, 0) + v
        answers = fs[q].answers
        memberships = []
        for iname in input_names:
            for cat, dist in tables[iname].items():
                idx = np.array([i for i, r in enumerate(rows_inputs) if r[iname] == cat])
                if len(idx) == 0:
                    continue
                s = np.array([dist[a] for a in answers])
                memberships.append((s, idx, len(idx) / len(rows)))
        fit = fit_loglinear(X, memberships, answers)
        P = fit.predict_proba(X)
        row_marginals[q] = P
        marginals[q] = tables

    # ---- output × output tables -> correlation matrix
    pair_tables: dict[str, pd.DataFrame] = {}
    n = len(outs)
    corr = np.eye(n)
    pair_flags = {"fallback": 0, "calls": 0}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            qi, qj = outs[i], outs[j]
            fj = fs[qj]
            table = np.zeros((len(fs[qi].answers), len(fj.answers)))
            for ci, a in enumerate(fs[qi].answers):
                dist, meta = ask_table_row(
                    client, model, qj,
                    output_group_phrase(qi, a), wording,
                    seed + 100_000 + 1_000 * i + j * 10 + ci,
                )
                pair_flags["calls"] += 1
                pair_flags["fallback"] += int(meta.get("fallback", False))
                for cj, b in enumerate(fj.answers):
                    table[ci, cj] = dist.probs[b]
            t = pd.DataFrame(table, index=fs[qi].answers, columns=fj.answers)
            pair_tables[f"{qi}|{qj}"] = t
            corr[i, j] = stated_pair_correlation(
                t, target_nominal=(fj.type == "nominal"), cond_nominal=(fs[qi].type == "nominal")
            )
    corr = (corr + corr.T) / 2

    w, V = np.linalg.eigh(corr)
    shift = float(np.max(np.maximum(0, 1e-4 - w)))
    w = np.clip(w, 1e-4, None)
    corr_pd = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)

    return {
        "marginals": marginals,
        "row_marginals": row_marginals,
        "pair_tables": pair_tables,
        "corr": corr_pd,
        "proj_shift": shift,
        "flags": {**flags, **pair_flags},
    }


def sample_stated_copula(
    row_marginals: dict[str, np.ndarray],
    corr: np.ndarray,
    rows: pd.DataFrame,
    rng: np.random.Generator,
    category_index: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """One profile per row: z ~ N(0, corr), each coordinate mapped through the
    row's stated marginal quantiles."""
    outs = output_names()
    fs = load_fields()
    if category_index is None:
        category_index = {q: fs[q].answers for q in outs}
    chol = np.linalg.cholesky(corr)
    n = len(rows)
    z = chol @ rng.standard_normal((len(outs), n))  # (21, n)
    u = norm.cdf(z)
    sim = rows.copy()
    for q in outs:
        sim[q] = pd.Series(pd.NA, index=sim.index, dtype="object")
    for j, q in enumerate(outs):
        P = row_marginals[q]  # (n, K)
        cats = category_index[q]
        for i in range(n):
            cum = np.cumsum(P[i])
            k = int(np.searchsorted(cum, u[j, i], side="right"))
            k = min(k, len(cats) - 1)
            sim.iat[i, sim.columns.get_loc(q)] = cats[k]
    return sim

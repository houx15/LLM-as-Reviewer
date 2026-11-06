"""Diagnostics for Study 1 (SPEC §4.5): spread, collapse, TV distance,
signed association difference. Type-level split diagnostics arrive in Phase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _entropy_of(s: pd.Series) -> float:
    v = s.dropna().value_counts(normalize=True)
    return float(entropy(v, base=2)) if len(v) else float("nan")


def _spread(s: pd.Series, numeric: bool) -> float:
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    if numeric:
        return float(s.astype(float).std(ddof=1))
    return _entropy_of(s)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    n = ct.values.sum()
    chi2 = ((ct.values - np.outer(ct.sum(1), ct.sum(0)) / n) ** 2 /
            (np.outer(ct.sum(1), ct.sum(0)) / n)).sum()
    q = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * q)))


def eta_squared(num: pd.Series, cat: pd.Series) -> float:
    df = pd.DataFrame({"num": num, "cat": cat}).dropna()
    if df["cat"].nunique() < 2 or len(df) < 5:
        return float("nan")
    grand = df["num"].astype(float).mean()
    ss_between = sum(
        ((g["num"].astype(float) - grand) ** 2).sum() for _, g in df.groupby("cat")
    )
    ss_total = ((df["num"].astype(float) - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def association(y: pd.Series, x: pd.Series, y_numeric: bool) -> float:
    if y_numeric:
        return eta_squared(y, x)
    return cramers_v(y, x)


def _canon(s: pd.Series, numeric: bool) -> pd.Series:
    """Canonicalize a column's values: numeric -> float ("" to NaN);
    categorical -> stripped strings (NaN/'nan'/'' to NaN)."""
    if numeric:
        return pd.to_numeric(s, errors="coerce")
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def study1_diagnostics(
    real: pd.DataFrame,
    sim: pd.DataFrame,
    question: str,
    input_names: list[str],
    group_map: dict[str, str],   # (input, value) -> group label, from fields.input_group_of
    min_rows: int = 30,
) -> dict:
    """Compare sim answers against real answers on one question.

    ``real``/``sim``: profile_id + inputs + the question column.
    """
    from .fields import load_fields

    fs = load_fields()[question]
    y_num = fs.is_numeric
    real_q = _canon(real[question], y_num)
    sim_q = _canon(sim[question], y_num)

    out: dict = {}

    # marginal TV distance
    rm = real_q.dropna().value_counts(normalize=True)
    sm = sim_q.dropna().value_counts(normalize=True)
    keys = sorted(set(rm.index.astype(str)) | set(sm.index.astype(str)))
    rmd = {str(k): v for k, v in rm.items()}
    smd = {str(k): v for k, v in sm.items()}
    out["tv_distance"] = float(
        0.5 * sum(abs(rmd.get(k, 0.0) - smd.get(k, 0.0)) for k in keys)
    )

    def _grp(iname, v):
        if pd.isna(v):
            return "Missing"
        return group_map.get((iname, v), str(v))

    # within-group spread ratio + collapse, per input
    spread_rows, collapse_groups, total_groups = [], 0, 0
    for iname in input_names:
        real_grp = real[iname].map(lambda v: _grp(iname, v))
        sim_grp = sim[iname].map(lambda v: _grp(iname, v))
        for gname, gidx in real.groupby(real_grp).groups.items():
            if len(gidx) < min_rows:
                continue
            total_groups += 1
            r_vals = real_q.loc[gidx]
            s_vals = sim_q.loc[sim_grp == gname]
            sr = _spread(r_vals, y_num)
            ss = _spread(s_vals, y_num)
            ratio = ss / sr if sr and sr == sr and ss == ss and sr > 1e-12 else float("nan")
            spread_rows.append({"input": iname, "group": gname, "n": len(gidx),
                                "real": sr, "model": ss, "ratio": ratio})
            top = s_vals.value_counts(normalize=True)
            if len(top) and top.iloc[0] > 0.9:
                rtop = r_vals.value_counts(normalize=True)
                if not (len(rtop) and rtop.iloc[0] > 0.9):
                    collapse_groups += 1
    spread_df = pd.DataFrame(spread_rows)
    out["spread_ratio_mean"] = float(spread_df["ratio"].mean())
    out["spread_rows"] = spread_df
    out["collapse_rate"] = collapse_groups / total_groups if total_groups else float("nan")

    # signed association difference per input (model - real)
    assoc = {}
    for iname in input_names:
        a_r = association(real_q, real[iname], y_num)
        a_s = association(sim_q, sim[iname], y_num)
        assoc[iname] = {
            "real": a_r, "model": a_s,
            "diff": (a_s - a_r) if (a_s == a_s and a_r == a_r) else float("nan"),
        }
    out["association"] = assoc
    out["association_diff_mean"] = float(np.nanmean([v["diff"] for v in assoc.values()]))

    # missing rate
    out["missing_rate"] = float(sim_q.isna().mean())
    return out

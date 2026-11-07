"""References B1 and B2 (SPEC §5.3): what real data alone would achieve.

Fitted on the development **fitting part** only; never used by a method under
test. Both produce complete data (no missing), like the LLM methods.

- **B1**: each output sampled from its marginal within the row's coarse group
  (gender × race × age band), with hierarchical fallback when the group has
  fewer than ``min_n`` valid answers for that output: gender × band → band →
  global (all fitted on the fitting part). Fallback level is logged.
- **B2**: Gaussian copula with per-coarse-group marginals (as B1) and a global
  normal-scores correlation matrix estimated on the fitting part (available-case
  per pair, projected to the nearest positive-definite matrix).

Encoding for B2 [default, documented]: every output is treated as ordinal —
categorical outputs as codes 0..k-1 in config level order, numeric outputs as
their integer values, special labels ("No Child") above the numeric maximum.
The copula maps a latent normal through each group's marginal CDF, so the
encoding only fixes the *order* of the cumulative intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fields import AGE_BANDS, load_fields, output_names


def age_band(a) -> str | None:
    if pd.isna(a):
        return None
    for (lo, hi), lab in zip(AGE_BANDS, ["18-34", "35-54", "55+"], strict=True):
        if lo <= a <= hi:
            return lab
    return None


def coarse_key(row) -> tuple[str, str, str]:
    return (str(row["gender"]), str(row["race"]), str(age_band(row["age"])))


def _fallback_keys(key: tuple[str, str, str]) -> list[tuple]:
    """Coarse group -> gender x band -> band -> global, as tuples of filters."""
    g, r, b = key
    return [
        ("coarse", g, r, b),
        ("gender_band", g, None, b),
        ("band", None, None, b),
        ("global", None, None, None),
    ]


@dataclass
class Marginals:
    """Per-output probability vectors at each fallback level."""
    levels: dict[tuple, dict[str, dict[str, float]]]   # level-key -> output -> {value: p}
    level_n: dict[tuple, dict[str, int]] = field(default_factory=dict)  # level-key -> output -> n valid
    fallbacks: dict[str, int] = field(default_factory=dict)  # output -> times a non-coarse level was used
    min_n: int = 30


def fit_marginals(fit: pd.DataFrame, min_n: int = 30) -> Marginals:
    band = fit["age"].map(age_band)
    groups: dict[tuple, pd.DataFrame] = {
        ("global", None, None, None): fit,
    }
    for (g, b), idx in fit.groupby([fit["gender"], band]).groups.items():
        groups[("gender_band", g, None, b)] = fit.loc[idx]
    for b, idx in fit.groupby(band).groups.items():
        groups[("band", None, None, b)] = fit.loc[idx]
    for (g, r, b), idx in fit.groupby([fit["gender"], fit["race"], band]).groups.items():
        groups[("coarse", g, r, b)] = fit.loc[idx]

    levels: dict[tuple, dict[str, dict[str, float]]] = {}
    level_n: dict[tuple, dict[str, int]] = {}
    for key, gdf in groups.items():
        levels[key] = {}
        level_n[key] = {}
        for q in output_names():
            valid = gdf[q].dropna()
            if len(valid) == 0:
                continue
            vc = valid.value_counts()
            levels[key][q] = {_fmt(k): v / len(valid) for k, v in vc.items()}
            level_n[key][q] = len(valid)
    return Marginals(levels=levels, level_n=level_n, min_n=min_n)


def group_marginal(
    m: Marginals, key: tuple[str, str, str], q: str
) -> tuple[dict[str, float], str]:
    """Marginal for q at the most specific level with enough data."""
    for level in _fallback_keys(key):
        marg = m.levels.get(level, {}).get(q)
        n = m.level_n.get(level, {}).get(q, 0)
        if marg and (n >= m.min_n or level[0] == "global"):
            if level[0] != "coarse":
                m.fallbacks[q] = m.fallbacks.get(q, 0) + 1
            return marg, level[0]
    return m.levels.get(("global", None, None, None), {}).get(q, {}), "global"


def _fmt(v) -> str:
    """Format a real value as the canonical answer string (ints without .0)."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def sample_b1(
    m: Marginals, rows: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, int]]:
    sim = rows.copy()
    for q in output_names():
        sim[q] = pd.Series(pd.NA, index=sim.index, dtype="object")
    used_levels: dict[str, int] = {}
    for i in range(len(rows)):
        row = rows.iloc[i]
        key = coarse_key(row)
        for q in output_names():
            marg, lvl = group_marginal(m, key, q)
            used_levels[lvl] = used_levels.get(lvl, 0) + 1
            vals = list(marg)
            col = sim.columns.get_loc(q)
            sim.iat[i, col] = rng.choice(vals, p=[marg[v] for v in vals]) if vals else pd.NA
    return sim, used_levels


# --------------------------------------------------------------------------- #
# B2: Gaussian copula
# --------------------------------------------------------------------------- #


@dataclass
class CopulaRef:
    marginals: Marginals
    corr: np.ndarray              # (21, 21) latent correlation
    outputs: list[str]
    proj_shift: float             # how far the correlation matrix moved when projected


def _normal_scores(s: pd.Series) -> pd.Series:
    """Mid-rank van der Waerden scores (NaN kept)."""
    r = s.rank(method="average")
    n = s.notna().sum()
    u = (r - 0.5) / n
    return pd.Series(norm.ppf(u.clip(1e-6, 1 - 1e-6)), index=s.index)


def fit_copula(fit: pd.DataFrame, min_n: int = 30) -> CopulaRef:
    fs = load_fields()
    outs = output_names()
    marginals = fit_marginals(fit, min_n=min_n)

    # ordinal codes for correlation estimation: config order for categorical,
    # integer value for numeric (special labels above max)
    def code(q: str, s: pd.Series) -> pd.Series:
        f = fs[q]
        if f.allowed is not None:
            idx = {v: i for i, v in enumerate(f.allowed)}
            return s.map(lambda v: idx.get(str(v), np.nan))
        assert f.numeric is not None
        sp = {v: f.numeric.max + 1 + i for i, v in enumerate(f.numeric.special)}

        def _c(v):
            if pd.isna(v):
                return np.nan
            sv = str(v)
            if sv in sp:
                return float(sp[sv])
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        return s.map(_c)

    coded = pd.DataFrame({q: code(q, fit[q]) for q in outs})
    scores = coded.apply(_normal_scores)
    corr = scores.corr().values

    # project to nearest positive-definite matrix (eigenvalue clipping)
    w, V = np.linalg.eigh((corr + corr.T) / 2)
    w_min = 1e-4
    shift = float(np.max(np.maximum(0, w_min - w)))
    w = np.clip(w, w_min, None)
    corr_pd = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)

    return CopulaRef(
        marginals=marginals, corr=corr_pd, outputs=outs, proj_shift=shift
    )


def sample_b2(ref: CopulaRef, rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per row: draw latent normal, map through the coarse-group marginals."""
    sim = rows.copy()
    for q in output_names():
        sim[q] = pd.Series(pd.NA, index=sim.index, dtype="object")
    chol = np.linalg.cholesky(ref.corr)
    for i in range(len(rows)):
        row = rows.iloc[i]
        key = coarse_key(row)
        z = chol @ rng.standard_normal(len(ref.outputs))
        u = norm.cdf(z)
        for j, q in enumerate(ref.outputs):
            marg, _ = group_marginal(ref.marginals, key, q)
            col = sim.columns.get_loc(q)
            if not marg:
                sim.iat[i, col] = pd.NA
                continue
            vals = list(marg)
            cum = np.cumsum([marg[v] for v in vals])
            k = int(np.searchsorted(cum, u[j], side="right"))
            k = min(k, len(vals) - 1)
            sim.iat[i, col] = vals[k]
    return sim

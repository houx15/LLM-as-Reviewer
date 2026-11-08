"""Study 3 reference models, fitted on the development fitting part (SPEC §5.4).

- **Bayesian network** over inputs + outputs (hill-climb, BIC, at most 3
  parents) for the calibration test and the typicality trap. All variables are
  discrete: categorical outputs as their categories, numeric outputs as integer
  values, except ``mental_health`` (0-30, binned 0 / 1-7 / 8-14 / 15-30) and
  ``age_first_childbirth`` ("No Child" / 17-24 / 25-34 / 35-45), binned to keep
  the state space estimable at n=539. Missing answers are kept as an explicit
  "NA" category (missingness is informative in GSS).
- **Gradient-boosting classifier** for the discrimination test: separates real
  profiles from one-output-swapped fakes (inputs one-hot; outputs ordinal
  codes, missing = -1).

Nothing from this module feeds generation.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .fields import load_fields, output_names, scored_input_names

MH_BINS = [-0.5, 0.5, 7.5, 14.5, 30.5]
MH_LABELS = ["0", "1-7", "8-14", "15-30"]
AFC_BINS = [16.0, 24.5, 34.5, 45.5]
AFC_LABELS = ["17-24", "25-34", "35-45"]


def bin_mental_health(v) -> str:
    if pd.isna(v):
        return "NA"
    x = float(v)
    for lab, lo, hi in zip(MH_LABELS, MH_BINS[:-1], MH_BINS[1:], strict=True):
        if lo < x <= hi:
            return lab
    return MH_LABELS[-1]


def bin_age_first_childbirth(v) -> str:
    if isinstance(v, str):
        try:
            x = float(v)
        except ValueError:
            return v  # special label ("No Child")
        v = x
    if pd.isna(v):
        return "NA"
    x = float(v)
    for lab, lo, hi in zip(AFC_LABELS, AFC_BINS[:-1], AFC_BINS[1:], strict=True):
        if lo <= x <= hi:
            return lab
    return AFC_LABELS[-1]


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """All modeled variables as string categories (missing -> 'NA').

    Age is banded (18-34/35-54/55+), as everywhere else in the design; numeric
    outputs are integer-valued strings, with mental_health and
    age_first_childbirth binned (see module docstring).
    """
    from .references import age_band  # noqa: PLC0415

    out = pd.DataFrame(index=df.index)
    fs = load_fields()
    for k in scored_input_names():
        if k == "age":
            out[k] = df[k].map(lambda v: "NA" if pd.isna(v) else age_band(v))
        else:
            out[k] = df[k].map(lambda v: "NA" if pd.isna(v) else str(v))
    for q in output_names():
        f = fs[q]
        if q == "mental_health":
            out[q] = df[q].map(bin_mental_health)
        elif q == "age_first_childbirth":
            out[q] = df[q].map(bin_age_first_childbirth)
        elif f.allowed is not None:
            out[q] = df[q].map(lambda v: "NA" if pd.isna(v) else str(v))
        else:
            out[q] = df[q].map(lambda v: "NA" if pd.isna(v) else str(int(v)))
    return out


def fit_bayesian_network(fit: pd.DataFrame, max_parents: int = 3):
    """Hill-climb + BIC structure, MLE parameters; returns fitted BN."""
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pgmpy.estimators import HillClimbSearch  # noqa: PLC0415
        from pgmpy.models import DiscreteBayesianNetwork  # noqa: PLC0415

        data = discretize(fit)
        hc = HillClimbSearch(data)
        dag = hc.estimate(scoring_method="bic-d", max_indegree=max_parents)
        model = DiscreteBayesianNetwork(dag.edges())
        model.add_nodes_from(data.columns)  # keep isolated nodes (marginal models)
        # Bayesian-parameter fit: MLE alone leaves NaN probabilities at parent
        # configurations unseen in the fitting part.
        from pgmpy.parameter_estimator import DiscreteBayesianEstimator  # noqa: PLC0415

        est = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=10)
        model.fit(data, estimator=est)
    return model
    return model


def bn_query(model, evidence: dict[str, str], target: str) -> dict[str, float]:
    """P(target | evidence) as a dict over target states (inferenence via VE)."""
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pgmpy.inference import VariableElimination  # noqa: PLC0415

        ve = VariableElimination(model)
        q = ve.query([target], evidence=evidence, show_progress=False)
        return {str(s): float(p) for s, p in zip(q.state_names[target], q.values, strict=True)}


# --------------------------------------------------------------------------- #
# Discrimination reference: gradient boosting on real vs swapped
# --------------------------------------------------------------------------- #


def ordinal_code(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs one-hot (age kept numeric), outputs ordinal codes, missing = -1."""
    fs = load_fields()
    parts: list[tuple[str, pd.Series]] = [
        ("age", df["age"].fillna(-1.0).astype(float))
    ]
    for k in scored_input_names():
        if k == "age":
            continue
        vals = df[k].map(lambda v: "NA" if pd.isna(v) else str(v))
        for lev in sorted(vals.unique()):
            parts.append((f"{k}={lev}", (vals == lev).astype(float)))
    for q in output_names():
        f = fs[q]
        if f.allowed is not None:
            idx = {v: i for i, v in enumerate(f.allowed)}
            parts.append((q, df[q].map(lambda v: -1.0 if pd.isna(v) else float(idx.get(str(v), -1)))))
        else:
            assert f.numeric is not None
            sp = {v: f.numeric.max + 1 + i for i, v in enumerate(f.numeric.special)}

            def _code(v, sp=sp):
                if pd.isna(v):
                    return -1.0
                sv = str(v)
                if sv in sp:
                    return float(sp[sv])
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return -1.0

            parts.append((q, df[q].map(_code)))
    return pd.DataFrame({k: pd.Series(v, index=df.index) for k, v in parts})


def make_swapped_negatives(
    fit: pd.DataFrame, rng: np.random.Generator, per_real: int = 1
) -> tuple[pd.DataFrame, np.ndarray]:
    """Real profiles and copies with one output swapped for a coarse-group
    marginal draw; returns (X, y with 0=real 1=swapped). X is encoded over the
    whole frame so one-hot levels align."""
    from .references import coarse_key, fit_marginals, group_marginal  # noqa: PLC0415

    marg = fit_marginals(fit)
    outs = output_names()
    frames: list[pd.DataFrame] = []
    y: list[int] = []
    for i in range(len(fit)):
        row = fit.iloc[i]
        key = coarse_key(row)
        frames.append(row.to_frame().T)
        y.append(0)
        for _ in range(per_real):
            alt = row.copy()
            q = outs[int(rng.integers(len(outs)))]
            m, _lvl = group_marginal(marg, key, q)
            vals = list(m)
            if vals:
                alt[q] = rng.choice(vals, p=[m[v] for v in vals])
            frames.append(alt.to_frame().T)
            y.append(1)
    X = ordinal_code(pd.concat(frames, ignore_index=True))
    return X, np.asarray(y)


def fit_discriminator(fit: pd.DataFrame, seed: int = 0, per_real: int = 2):
    from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    X, y = make_swapped_negatives(fit, rng, per_real=per_real)
    clf = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=200)
    clf.fit(X, y)
    return clf, float((clf.predict(X) == y).mean())

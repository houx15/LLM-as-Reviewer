"""Log-linear fit p(y | x) to stated two-way tables (SPEC §4.4).

A multinomial logit with additive main effects of the 8 input categories,
parameters chosen to minimize the KL divergence between the stated two-way
tables and the tables implied by the model when averaged over the row set's
actual input distribution. Inputs are given to every method, so using their
distribution is allowed; no real outputs are used.

Notation: each row i belongs to one category of each input k, i.e. to one
target table row t = (k, c). The implied distribution for t is the average of
the model's p(y | x_i) over rows in t; the objective is

    L(θ) = Σ_t w_t Σ_y s_t[y] log(s_t[y] / imp_t[y]),   w_t = share of rows in t.

With q_i[y] = Σ_{t ∋ i} (w_t/n_t) s_t[y]/imp_t[y], the gradient is

    ∇L = Xᵀ ( P ⊙ (r − Q) ),   r_i = Σ_y q_i[y] P_i[y],
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class LogLinearFit:
    theta: np.ndarray          # (n_features, n_classes) logits
    feature_names: list[str]   # columns of the design matrix
    classes: list[str]         # y categories
    mad: float                 # mean absolute deviation between stated and implied
    loss: float
    n_iter: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (n, n_features) one-hot with intercept; returns (n, n_classes)."""
        logits = X @ self.theta
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


def build_design_matrix(
    rows_inputs: list[dict[str, str]],
    input_names: list[str],
    input_categories: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """One-hot design with intercept; baseline = first category of each input.

    ``rows_inputs``: per-row dict of input name -> category string (age already
    banded).
    """
    features: list[str] = ["intercept"]
    for name in input_names:
        cats = input_categories[name]
        features += [f"{name}={c}" for c in cats[1:]]
    index = {f: i for i, f in enumerate(features)}
    X = np.zeros((len(rows_inputs), len(features)), dtype=float)
    X[:, 0] = 1.0
    for i, row in enumerate(rows_inputs):
        for name in input_names:
            cats = input_categories[name]
            if row[name] not in cats:
                raise ValueError(f"unknown category {row[name]!r} for {name}")
            if row[name] != cats[0]:
                X[i, index[f"{name}={row[name]}"]] = 1.0
    return X, features


def fit_loglinear(
    X: np.ndarray,
    memberships: list[tuple[np.ndarray, np.ndarray, float]],  # (s_t, idx_t, w_t)
    classes: list[str],
    max_iter: int = 1000,
) -> LogLinearFit:
    """Fit θ by L-BFGS on the weighted KL objective.

    memberships: one entry per (input, category) table row: stated distribution
    ``s_t`` (len(classes),) over row indices ``idx_t`` with row-share weight w_t.
    """
    n, d = X.shape
    k = len(classes)
    memberships = [(s, np.asarray(idx, int), w) for s, idx, w in memberships if len(idx) and w > 0]
    if not memberships:
        raise ValueError("no usable table rows")

    # init: mean stated distribution as intercept
    s_bar = np.mean([s for s, _, _ in memberships], axis=0)
    theta0 = np.zeros((d, k))
    theta0[0, :] = np.log(np.clip(s_bar, 1e-6, None))
    theta0[0, :] -= theta0[0, :].mean()

    def softmax_p(th):
        logits = X @ th
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def loss_grad(flat: np.ndarray):
        th = flat.reshape(d, k)
        P = softmax_p(th)
        Q = np.zeros((n, k))
        for s, idx, w in memberships:
            imp = P[idx].mean(axis=0)
            imp = np.clip(imp, 1e-12, None)
            Q[idx] += (w / len(idx)) * (s / imp)[None, :]
        r = (Q * P).sum(axis=1)
        loss = 0.0
        for s, idx, w in memberships:
            imp = np.clip(P[idx].mean(axis=0), 1e-12, None)
            loss += w * float(s @ (np.log(np.clip(s, 1e-12, None)) - np.log(imp)))
        G = X.T @ (P * (r[:, None] - Q))
        return loss, G.ravel()

    res = minimize(
        loss_grad,
        theta0.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "maxfun": max_iter * 2},
    )
    theta = res.x.reshape(d, k)

    # mean absolute deviation between stated and implied distributions
    P = softmax_p(theta)
    devs = []
    for s, idx, w in memberships:
        devs.append(np.abs(s - P[idx].mean(axis=0)).mean())
    return LogLinearFit(
        theta=theta,
        feature_names=[],  # filled by caller if needed
        classes=classes,
        mad=float(np.mean(devs)),
        loss=float(res.fun),
        n_iter=int(res.nit),
    )

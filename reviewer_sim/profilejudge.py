"""Whole-profile judging (Family 2) and the S4 selection rule on profiles.

Shared by P1, P2, P3 and P5. The judge sees the input row and k complete
profiles labelled A, B, C…; two label orders are run and the probability
vectors averaged (SPEC §3.2). S4 (frequency-corrected) draws a candidate with
probability proportional to the judge's probability alone.
"""

from __future__ import annotations

import random

import numpy as np

from .llm import LLMClient
from .parse import ParseError, parse_distribution, uniform
from .prompts import judge_profiles_prompt
from .rules import SYSTEM_JSON


def judge_profiles(
    client: LLMClient,
    judge_model: str,
    row,                                # pd.Series with the 9 inputs
    profiles: list[dict[str, object]],  # candidate output dicts
    wording: str,
    seed: int,
    input_names: list[str],
    n_orders: int = 2,
    max_attempts: int = 3,
) -> list[float]:
    """Averaged judge probability per candidate (original order)."""
    out = np.zeros(len(profiles))
    for o in range(n_orders):
        dist = None
        meta = None
        for attempt in range(max_attempts):
            a_seed = seed + 89 * o + 733 * attempt
            prompt, meta = judge_profiles_prompt(
                {k: row[k] for k in input_names}, profiles, wording, seed=a_seed
            )
            res = client.chat(
                [
                    {"role": "system", "content": SYSTEM_JSON},
                    {"role": "user", "content": prompt},
                ],
                model=judge_model,
                json_mode=True,
                seed=a_seed,
                meta={**meta, "row": int(row["profile_id"]), "order": o, "attempt": attempt},
                purpose="judge_profiles",
            )
            labels = list(meta["label_of"].values())
            try:
                dist = parse_distribution(res.content, labels, counts_mode=(wording == "W2"))
                break
            except ParseError:
                dist = None
        if dist is None:
            labels_now = list(meta["label_of"].values())
            dist = uniform(labels_now)
        # map label -> original index
        inv = {lab: int(i) for i, lab in meta["label_of"].items()}
        for lab, p in dist.probs.items():
            out[inv[lab]] += p / n_orders
    total = out.sum()
    return [float(p / total) if total > 0 else 1.0 / len(profiles) for p in out]


def select_s4(j_probs: list[float], rng: random.Random) -> int:
    """Draw an index with probability proportional to the judge's probability."""
    w = np.array([max(p, 0.0) for p in j_probs])
    if w.sum() <= 0:
        return int(rng.randrange(len(j_probs)))
    p = w / w.sum()
    return int(rng.choices(range(len(j_probs)), weights=p.tolist(), k=1)[0])


def select_random(n: int, rng: random.Random) -> int:
    return int(rng.randrange(n))

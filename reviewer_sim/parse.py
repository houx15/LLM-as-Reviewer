"""Parse model JSON answers into validated probability distributions.

SPEC §3.2: every required key must be present, values numeric and >= 0; sums
within 0.9–1.1 (W2 counts: 90–110) are renormalized; otherwise retry (handled
by the caller); after retries fail, fall back to uniform and flag it.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass

SUM_TOL = (0.9, 1.1)


class ParseError(Exception):
    pass


def extract_json(text: str) -> dict:
    """Parse JSON, tolerating markdown fences or surrounding prose."""
    if not text or not text.strip():
        raise ParseError("empty response")
    t = text.strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ParseError("no JSON object found") from None
        snippet = t[start : end + 1]
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            # tolerate unquoted numeric keys (invalid JSON, seen in practice)
            try:
                obj = ast.literal_eval(snippet)
            except (ValueError, SyntaxError) as e:
                raise ParseError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    return obj


def _norm_key(k: str) -> str:
    return str(k).strip()


@dataclass
class DistResult:
    probs: dict[str, float]
    renormalized: bool
    fallback: bool
    raw_sum: float


def parse_distribution(
    raw: str,
    keys: list[str],
    *,
    counts_mode: bool = False,
) -> DistResult:
    """Parse ``{key: p}`` for exactly ``keys`` (order-insensitive) and normalize.

    ``counts_mode``: W2 answers counts out of 100; total tolerance is scaled
    to 0.9–1.1 of 100. Keys may be numeric strings (allowed integers); the
    model may return them as JSON numbers, so both are matched.
    """
    obj = extract_json(raw)
    want = {str(k): k for k in keys}
    got: dict[str, float] = {}
    for k, v in obj.items():
        nk = _norm_key(k)
        matched = None
        if nk in want:
            matched = nk
        else:
            # numeric keys: model may write 5 instead of "5", or strip spaces
            for cand in want:
                if cand.strip().lower() == nk.lower():
                    matched = cand
                    break
        if matched is None:
            continue  # ignore extraneous keys (e.g. "explanation")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ParseError(f"value for {matched!r} not numeric: {v!r}")
        fv = float(v)
        if math.isnan(fv) or fv < 0:
            raise ParseError(f"value for {matched!r} negative or NaN")
        got[matched] = got.get(matched, 0.0) + fv

    missing = [k for k in keys if k not in got]
    if missing:
        raise ParseError(f"missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    total = sum(got.values())
    if counts_mode:
        lo, hi = 100 * SUM_TOL[0], 100 * SUM_TOL[1]
    else:
        lo, hi = SUM_TOL
    ok = lo <= total <= hi
    if not ok and not counts_mode:
        # robustness: a model may answer percentages (e.g. 60/40) in W1
        if 100 * SUM_TOL[0] <= total <= 100 * SUM_TOL[1]:
            ok = True
    if not ok:
        raise ParseError(f"sum {total:.4g} outside tolerance [{lo:g}, {hi:g}]")
    probs = {k: got[k] / total for k in keys}
    return DistResult(probs=probs, renormalized=abs(total - 1.0) > 1e-9, fallback=False, raw_sum=total)


def uniform(keys: list[str], raw_sum: float = float("nan")) -> DistResult:
    n = len(keys)
    return DistResult(
        probs={k: 1.0 / n for k in keys},
        renormalized=False,
        fallback=True,
        raw_sum=raw_sum,
    )


def parse_labels(raw: str, keys: list[str], vocabulary: tuple[str, ...]) -> dict[str, str]:
    """Parse ``{label: "plausible"|"implausible"}`` (rule S3 / Family 3)."""
    obj = extract_json(raw)
    out: dict[str, str] = {}
    for k, v in obj.items():
        nk = _norm_key(k)
        val = str(v).strip().lower()
        if val not in vocabulary:
            raise ParseError(f"unexpected verdict for {nk!r}: {v!r}")
        out[nk] = val
    missing = [k for k in keys if k not in out]
    if missing:
        raise ParseError(f"missing labels: {missing}")
    return out


def parse_single_probability(raw: str, key: str = "X", counts_mode: bool = False) -> float:
    """Parse ``{"X": p}`` (Family 4 pairwise) into a probability for X."""
    obj = extract_json(raw)
    if key not in obj:
        raise ParseError(f"missing key {key!r}")
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ParseError(f"value for {key!r} not numeric: {v!r}")
    fv = float(v)
    if fv < 0:
        raise ParseError("negative probability")
    if counts_mode:
        if not (0 <= fv <= 100 * SUM_TOL[1]):
            raise ParseError(f"count {fv} out of range")
        return min(fv / 100.0, 1.0)
    if fv > 1.0 + 1e-9:
        # tolerate percentage-style answers (e.g. 70 for 0.7), not scale errors like 1.5
        if 10.0 <= fv <= 100.0 * SUM_TOL[1]:
            return fv / 100.0
        raise ParseError(f"probability {fv} > 1")
    return fv

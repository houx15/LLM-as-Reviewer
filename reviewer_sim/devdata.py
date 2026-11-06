"""Development data: split, pilot rows, official run-folder CSV format.

Development respondents = GSS 2018 respondents NOT drawn into the benchmark's
evaluation sample (verified: the benchmark sample reproduces the official
``sample(n=1000, random_state=42)``). Split once, 40/60 fitting/evaluation,
seed 2026 (SPEC §3.5). All lists are frozen as CSVs under ``data/dev/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fields import input_names, output_names

REPO_ROOT = Path(__file__).resolve().parents[1]
PARQUET = REPO_ROOT / "data" / "processed" / "gss_2018" / "data.parquet"
DEV_DIR = REPO_ROOT / "data" / "dev"
SPLIT_SEED = 2026
PILOT_SEED = 2026
PILOT_N = 500

ALL_COLUMNS = ["profile_id", *input_names(), *output_names()]


def load_full() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


def _write_official_csv(df: pd.DataFrame, path: Path, keep: list[str] = ()) -> None:
    """Write rows in the official sampled_inputs/sim_profiles format.

    ``keep``: extra bookkeeping columns (e.g. resp_id) preserved in front;
    the scorer only reads configured columns, so extras are harmless.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in ALL_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    front = [c for c in keep if c in out.columns]
    out = out[[*front, *ALL_COLUMNS]]
    out.to_csv(path, index=False)


def build_dev_split(force: bool = False) -> dict[str, Path]:
    """Create (or return, if present) the frozen dev split files."""
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "fitting": DEV_DIR / "dev_fitting.csv",
        "evaluation": DEV_DIR / "dev_evaluation.csv",
        "pilot": DEV_DIR / "pilot_rows.csv",
        "final": REPO_ROOT / "data" / "final_rows.csv",
    }
    if all(p.exists() for p in paths.values()) and not force:
        return paths

    df = load_full()

    dev = df[~df["in_benchmark"]].copy()
    bench = df[df["in_benchmark"]].copy()
    if len(dev) < 1000:
        raise RuntimeError(f"Only {len(dev)} development respondents; need >= 1000 (SPEC §3.5)")

    # final-test rows: the benchmark's own evaluation sample, frozen now (Phase 1 item 1)
    bench = bench.sort_values("benchmark_profile_id").reset_index(drop=True)
    bench["profile_id"] = bench["benchmark_profile_id"].astype(int)
    _write_official_csv(bench, paths["final"], keep=["resp_id"])

    # 40/60 fitting/evaluation split, seed 2026
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(dev))
    n_fit = int(round(0.4 * len(dev)))
    fit = dev.iloc[sorted(perm[:n_fit])].copy().reset_index(drop=True)
    ev = dev.iloc[sorted(perm[n_fit:])].copy().reset_index(drop=True)
    fit["profile_id"] = np.arange(len(fit))
    ev["profile_id"] = np.arange(len(ev))
    _write_official_csv(fit, paths["fitting"], keep=["resp_id"])
    _write_official_csv(ev, paths["evaluation"], keep=["resp_id"])

    # pilot rows: 500 from the evaluation part
    rng2 = np.random.default_rng(PILOT_SEED)
    pilot_idx = sorted(rng2.choice(len(ev), size=min(PILOT_N, len(ev)), replace=False))
    pilot = ev.iloc[pilot_idx].copy().reset_index(drop=True)
    pilot["profile_id"] = np.arange(len(pilot))
    _write_official_csv(pilot, paths["pilot"], keep=["resp_id"])

    print(
        f"dev split written: fitting={len(fit)}, evaluation={len(ev)}, "
        f"pilot={len(pilot)}, final(benchmark)={len(bench)}"
    )
    return paths


def load_rows(name: str) -> pd.DataFrame:
    """Load one of the frozen row lists ('fitting' | 'evaluation' | 'pilot' | 'final')."""
    paths = build_dev_split()
    return pd.read_csv(paths[name])


def real_csv(name: str) -> Path:
    """Path of the real-rows CSV used as scoring reference for a row set."""
    return build_dev_split()[name]


def write_run_csv(
    profiles: pd.DataFrame, run_dir: str | Path, run_id: str
) -> tuple[Path, Path]:
    """Write a condition's run folder: sim_profiles + the row set's real CSV.

    ``profiles`` must contain profile_id + the 21 outputs. The folder layout
    matches the official one so the scorer reads it unchanged.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_path = run_dir / f"sim_profiles_{run_id}.csv"
    sampled_path = run_dir / f"sampled_inputs_{run_id}.csv"
    _write_official_csv(profiles, sim_path)
    return sim_path, sampled_path

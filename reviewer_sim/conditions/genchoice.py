"""Generator choice (SPEC §4.1): 10 full profiles for each of 50 pilot rows
per candidate model, official prompt and validation; diversity report (Table S1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .. import devdata
from ..fields import input_names, load_fields, output_names
from ..llm import LLMClient, run_parallel
from ..prompts import official_generation_prompt
from ..rules import SYSTEM_JSON, _validate_answer

N_ROWS = 50
N_PROFILES = 10


def _generate_profiles(
    client: LLMClient,
    model: str,
    rows: pd.DataFrame,
    n_profiles: int,
    seed: int,
    max_retries: int = 3,
) -> list[list[dict[str, str]]]:
    """n_profiles validated profiles per row; official retry-on-empty loop."""
    fs = load_fields()
    in_names = input_names()

    def one_row(_i, item):
        _, row = item
        inputs = {k: row[k] for k in in_names if pd.notna(row[k])}
        prompt = official_generation_prompt(inputs)
        profiles: list[dict[str, str]] = []
        for j in range(n_profiles):
            rec: dict[str, str] = {}
            for attempt in range(max_retries + 1):
                a_seed = seed + 10_000 * int(row["profile_id"]) + j * 100 + attempt
                res = client.chat(
                    [
                        {"role": "system", "content": SYSTEM_JSON},
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    json_mode=True,
                    seed=a_seed,
                    meta={"row": int(row["profile_id"]), "j": j, "attempt": attempt},
                    purpose="genchoice_profile",
                )
                try:
                    obj = json.loads(res.content)
                except json.JSONDecodeError:
                    obj = {}
                rec = {}
                for q in output_names():
                    v = obj.get(q)
                    rec[q] = _validate_answer(json.dumps({q: v}), fs[q]) if v is not None else ""
                if all(v != "" for v in rec.values()):
                    break  # complete record
            profiles.append(rec)
        return profiles

    return run_parallel(one_row, list(rows.iterrows()),
                        max_workers=client.max_workers, desc=f"genchoice/{model}")


def diversity_metrics(profiles: list[list[dict[str, str]]]) -> dict:
    """Valid rate, duplicate share, mean per-output entropy, Hamming distance."""
    outputs = output_names()
    n_fields = len(outputs)

    valid_all, dup_shares, ents, hams = [], [], [], []
    for row_profiles in profiles:
        keys = sorted({k for p in row_profiles for k in p})
        n_fields = max(1, len(keys))
        # valid-output rate over all fields × candidates
        valid_all += [v != "" for p in row_profiles for v in p.values()]
        # exact duplicate share within the row
        sigs = [json.dumps(p, sort_keys=True) for p in row_profiles]
        dup_shares.append(1 - len(set(sigs)) / len(sigs))
        # per-output entropy across the row's candidates (valid only)
        row_ents = []
        for q in keys:
            vs = [p[q] for p in row_profiles if p.get(q, "") != ""]
            if len(vs) < 2:
                row_ents.append(0.0)
                continue
            counts = pd.Series(vs).value_counts(normalize=True)
            row_ents.append(float(entropy(counts, base=2)))
        ents.append(np.mean(row_ents))
        # pairwise Hamming distance over the fields
        ds = []
        for a in range(len(row_profiles)):
            for b in range(a + 1, len(row_profiles)):
                d = sum(row_profiles[a].get(q, "") != row_profiles[b].get(q, "") for q in keys)
                ds.append(d / n_fields)
        hams.append(np.mean(ds))

    return {
        "valid_rate": float(np.mean(valid_all)),
        "duplicate_share": float(np.mean(dup_shares)),
        "mean_entropy": float(np.mean(ents)),
        "mean_hamming": float(np.mean(hams)),
    }


def run_genchoice(
    models: dict[str, str],
    out_dir: str | Path,
    cache_root: str | Path,
    n_rows: int = N_ROWS,
    n_profiles: int = N_PROFILES,
    seed: int = 777_000,
    client: LLMClient | None = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = devdata.load_rows("pilot").head(n_rows).reset_index(drop=True)
    if client is None:
        client = LLMClient(
            cache_path=Path(cache_root) / "cache.jsonl",
            log_path=Path(cache_root) / "calls.jsonl",
        )

    records = []
    for m_i, (name, model) in enumerate(models.items()):
        profiles = _generate_profiles(client, model, rows, n_profiles, seed + 1000 * m_i)
        metrics = diversity_metrics(profiles)
        records.append({"model_name": name, "model": model, **metrics})
        # store raw profiles for later reuse (P2 candidate pools etc.)
        payload = [
            {"profile_id": int(rows.iloc[i]["profile_id"]), "profiles": ps}
            for i, ps in enumerate(profiles)
        ]
        (out_dir / f"profiles_{name}.json").write_text(json.dumps(payload, ensure_ascii=False))
        print(f"[genchoice] {name}: {metrics}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "table_s1.csv", index=False)
    return df

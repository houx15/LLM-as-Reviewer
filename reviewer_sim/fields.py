"""Field metadata for the SSDataBench GSS 2018 track.

Loads the official content config (``simulation/configs/content/gss_2018.yaml``)
from the vendored SSDataBench copy and exposes names, types, allowed values,
descriptions and input groups exactly as the benchmark defines them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SSD_ROOT = REPO_ROOT / "data" / "raw" / "ssdatabench"
CONTENT_CONFIG = SSD_ROOT / "simulation" / "configs" / "content" / "gss_2018.yaml"
# Verified variable types (nominal / ordinal / numeric) from the previous project's
# processed schema; the official content config marks everything "static".
PROCESSED_SCHEMA = REPO_ROOT / "data" / "processed" / "gss_2018" / "schema.yaml"

# Age bands used for input groups (SPEC §2, [default]).
AGE_BANDS = [(18, 34), (35, 54), (55, 200)]
AGE_BAND_LABELS = ["18-34", "35-54", "55+"]


@dataclass(frozen=True)
class NumericSpec:
    min: int
    max: int
    special: tuple[str, ...] = ()

    @property
    def values(self) -> list[str]:
        """Every allowed answer, integers formatted without decimals."""
        return [str(v) for v in range(self.min, self.max + 1)] + list(self.special)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: str  # "x" or "y"
    type: str  # numeric | ordinal | nominal
    description: str
    allowed: tuple[str, ...] | None  # categorical levels; None for numeric
    numeric: NumericSpec | None
    not_applicable: tuple[str, ...]

    @property
    def is_numeric(self) -> bool:
        return self.numeric is not None

    @property
    def answers(self) -> list[str]:
        """All allowed answers as strings (used as keys of stated distributions)."""
        if self.numeric is not None:
            return self.numeric.values
        assert self.allowed is not None
        return list(self.allowed)


def _parse_allowed(spec: Any) -> tuple[tuple[str, ...] | None, NumericSpec | None]:
    if isinstance(spec, list):
        return tuple(spec), None
    if isinstance(spec, dict) and spec.get("type") == "numeric":
        return None, NumericSpec(
            min=int(spec["min"]),
            max=int(spec["max"]),
            special=tuple(spec.get("special", [])),
        )
    raise ValueError(f"Unsupported allowed spec: {spec!r}")


@lru_cache(maxsize=1)
def load_fields(config_path: str | Path = CONTENT_CONFIG) -> dict[str, FieldSpec]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    types: dict[str, str] = {}
    if PROCESSED_SCHEMA.exists():
        with open(PROCESSED_SCHEMA, encoding="utf-8") as f:
            types = {
                name: v["type"]
                for name, v in yaml.safe_load(f)["variables"].items()
            }
    fields: dict[str, FieldSpec] = {}
    for role, section in (("x", "input_variables"), ("y", "output_variables")):
        for name, vcfg in cfg[section].items():
            allowed, numeric = _parse_allowed(vcfg["allowed"])
            # Output-level "not applicable" categories (Unemployed / No spouse) are
            # listed inside the output's allowed values in the content config.
            na: tuple[str, ...] = ()
            if role == "y" and allowed is not None:
                na = tuple(v for v in allowed if v in ("Unemployed", "No spouse"))
            vtype = "numeric" if numeric is not None else types.get(name, "nominal")
            fields[name] = FieldSpec(
                name=name,
                role=role,
                type=vtype,
                description=vcfg["description"],
                allowed=allowed,
                numeric=numeric,
                not_applicable=na,
            )
    return fields


@lru_cache(maxsize=1)
def load_context() -> dict[str, Any]:
    """Population context, reference year, context string from the official config."""
    with open(CONTENT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "context": cfg["context"].strip(),
        "reference_year": cfg["population_context"]["reference_year"],
    }


def input_names() -> list[str]:
    """Input names in official prompt order (birth_year first, as in the config)."""
    return [f.name for f in load_fields().values() if f.role == "x"]


def output_names() -> list[str]:
    return [f.name for f in load_fields().values() if f.role == "y"]


def scored_input_names() -> list[str]:
    """The 8 scored inputs (birth_year appears in prompts but is not scored)."""
    return [n for n in input_names() if n != "birth_year"]


def age_band(age: float) -> str:
    for (lo, hi), label in zip(AGE_BANDS, AGE_BAND_LABELS, strict=True):
        if lo <= age <= hi:
            return label
    raise ValueError(f"Age out of bands: {age}")


def input_group_values(name: str) -> list[str]:
    """Categories of an input as used by tables / population methods (SPEC §2)."""
    fs = load_fields()[name]
    if name == "age":
        return AGE_BAND_LABELS
    if fs.allowed is not None:
        return list(fs.allowed)
    raise ValueError(f"No categorical groups for numeric input {name}")


def input_group_of(name: str, value: Any) -> str:
    """Map one input value to its group category (NaN -> "Missing").

    Missingness is informative in GSS (e.g. father_education ~28% missing), so
    it is kept as its own category for tables and the log-linear design.
    """
    try:
        if value is None or pd.isna(value):
            return "Missing"
    except (TypeError, ValueError):
        pass
    if name == "age":
        return age_band(float(value))
    return str(value)


def input_group_values_with_missing(name: str, rows: "object") -> list[str]:
    """Config categories, plus 'Missing' when the rows contain NaN values."""
    vals = input_group_values(name)
    if rows[name].isna().any():
        return [*vals, "Missing"]
    return list(vals)

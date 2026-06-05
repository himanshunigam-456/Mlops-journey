"""Capture per-prediction rows into a flat, Parquet-friendly schema.

Each row records what the model saw + what it returned + when. The schema
is intentionally flat (no nested dicts) so it serializes cleanly to Parquet
and feeds straight into Evidently's drift report.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionRow:
    """One scoring event — request → response → timestamp."""

    request_id: str
    features: dict[str, float]
    prediction: int
    probability_default: float
    model_version: str
    scored_at: str = field(
        default_factory=lambda: dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )


def capture_row(
    *,
    request_id: str,
    features: dict[str, float],
    prediction: Any,
    probability_default: Any,
    model_version: Any,
) -> PredictionRow:
    """Build a PredictionRow with `scored_at` defaulted to now (UTC ISO-8601).

    Type-coerces `prediction`, `probability_default`, `model_version` so callers
    can pass numpy scalars, strings, or python primitives interchangeably.
    """
    return PredictionRow(
        request_id=str(request_id),
        features={str(k): float(v) for k, v in features.items()},
        prediction=int(prediction),
        probability_default=float(probability_default),
        model_version=str(model_version),
    )


def to_dataframe(rows: list[PredictionRow]) -> pd.DataFrame:
    """Flatten captured rows into a DataFrame with one column per feature.

    Feature columns get a `feature_` prefix so they're distinguishable from
    the metadata columns (request_id, prediction, etc.) when Evidently scans
    column-level drift.
    """
    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for r in rows:
        flat: dict[str, Any] = {
            "request_id": r.request_id,
            "prediction": r.prediction,
            "probability_default": r.probability_default,
            "model_version": r.model_version,
            "scored_at": r.scored_at,
        }
        flat.update({f"feature_{k}": v for k, v in r.features.items()})
        records.append(flat)
    return pd.DataFrame(records)

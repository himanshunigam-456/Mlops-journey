"""Evidently DataDriftPreset wrapper — input two DataFrames, output typed summary.

We wrap Evidently behind a thin facade because Evidently is pre-1.0 and has
already broken its API once (0.4 → 0.7). Callers depend on our `DriftSummary`,
not Evidently directly — when 0.8 ships and renames things again, we patch
ONE file instead of every consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset


@dataclass(frozen=True)
class DriftSummary:
    """Drift report result, normalized for downstream consumers.

    Fields:
        drift_share:     fraction of columns flagged as drifted, in [0, 1]
        drifted_columns: names of columns flagged
        n_columns:       total columns compared
        html:            full Evidently HTML report (for inline rendering)
    """

    drift_share: float
    drifted_columns: list[str]
    n_columns: int
    html: str


def _extract_p_value(value_field: Any) -> float | None:
    """Pull a p-value out of an Evidently `metric_value` field.

    The field may be a bare number, a dict like `{"value": 0.05}`, or a string
    representation. Returns None if it can't be coerced.
    """
    if isinstance(value_field, int | float):
        return float(value_field)
    if isinstance(value_field, dict):
        for key in ("value", "p_value", "score"):
            if key in value_field:
                try:
                    return float(value_field[key])
                except (TypeError, ValueError):
                    pass
    try:
        return float(value_field)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> DriftSummary:
    """Run Evidently's DataDriftPreset, return a normalized summary + HTML.

    Compares only columns present in both reference and current. Drift is
    detected per-column when the K-S test p-value < threshold (default 0.05).
    """
    common = [c for c in reference.columns if c in current.columns]
    ref = reference[common].copy()
    cur = current[common].copy()

    ref_ds = Dataset.from_pandas(ref)
    cur_ds = Dataset.from_pandas(cur)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    d = result.dict()
    metrics = d.get("metrics", [])

    # First entry is the summary `DriftedColumnsCount` with .value = {count, share}
    drift_share = 0.0
    if metrics:
        summary_val = metrics[0].get("value", {})
        if isinstance(summary_val, dict) and "share" in summary_val:
            try:
                drift_share = float(summary_val["share"])
            except (TypeError, ValueError):
                drift_share = 0.0

    # Remaining entries are per-column `ValueDrift`
    drifted_columns: list[str] = []
    n_columns = 0
    for m in metrics[1:]:
        cfg = m.get("config", {})
        if "ValueDrift" not in str(cfg.get("type", "")):
            continue
        n_columns += 1
        col = cfg.get("column", "?")
        threshold = float(cfg.get("threshold", 0.05))
        p_value = _extract_p_value(m.get("value"))
        if p_value is not None and p_value < threshold:
            drifted_columns.append(str(col))

    html = result.get_html_str(as_iframe=False)

    return DriftSummary(
        drift_share=drift_share,
        drifted_columns=drifted_columns,
        n_columns=n_columns,
        html=html,
    )

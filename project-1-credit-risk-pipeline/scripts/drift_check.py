"""Drift check CLI — the heartbeat of the drift monitoring loop.

Compares the training reference (s3://reports/reference_v{N}.parquet) against
the last N days of production captures, generates an Evidently HTML report,
uploads it to s3://reports/latest.html, and exits non-zero if drift exceeds
threshold so a CI cron can open a GitHub issue.

Usage:
    python scripts/drift_check.py
    python scripts/drift_check.py --threshold 0.3 --days 7
    python scripts/drift_check.py --threshold 0.5  # noisier model, looser bound
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import date, timedelta

import pandas as pd
from credit_risk.drift.reporter import compute_drift_report
from credit_risk.drift.store import (
    REPORTS_BUCKET,
    _s3_client,
    read_predictions_range,
    write_report_html,
)
from credit_risk.serving.model_loader import load_staging_model

# Columns that aren't model inputs or outputs — they describe HOW the row was
# captured, not WHAT it contained. Including them in drift analysis would
# always flag `scored_at` (timestamps vs the literal "REFERENCE" sentinel)
# and `request_id` (always unique by construction).
METADATA_COLUMNS = {"request_id", "scored_at", "model_version"}


def _drop_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in METADATA_COLUMNS if c in df.columns])


def _read_reference(version: str) -> pd.DataFrame:
    """Pull the training-time reference snapshot for this model version."""
    s3 = _s3_client()
    key = f"reference_v{version}.parquet"
    obj = s3.get_object(Bucket=REPORTS_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Fail with exit 1 if drift_share exceeds this fraction (default: 0.3).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=7,
        help="How many past days of production traffic to compare (default: 7).",
    )
    p.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Skip check (exit 0) if production captures < this row count (default: 30).",
    )
    args = p.parse_args()

    bundle = load_staging_model()
    print(f"Model: credit-risk-classifier v{bundle.version}")

    reference = _read_reference(str(bundle.version))
    print(f"Reference: {len(reference):,} rows × {len(reference.columns)} cols")

    end = date.today()
    start = end - timedelta(days=args.days - 1)
    current = read_predictions_range(start, end)
    print(f"Current  ({start} → {end}): {len(current):,} rows")

    if len(current) < args.min_rows:
        print(
            f"\nNot enough production data yet "
            f"(need ≥ {args.min_rows} rows, have {len(current)}). "
            "Skipping drift check."
        )
        return 0

    summary = compute_drift_report(_drop_metadata(reference), _drop_metadata(current))
    pct = summary.drift_share * 100
    print(
        f"\nDrift share: {pct:.1f}%  "
        f"({len(summary.drifted_columns)}/{summary.n_columns} columns drifted)"
    )

    if summary.drifted_columns:
        print("Top drifted columns:")
        for col in summary.drifted_columns[:10]:
            print(f"  • {col}")
        if len(summary.drifted_columns) > 10:
            print(f"  … and {len(summary.drifted_columns) - 10} more")

    write_report_html("latest.html", summary.html)
    print(f"\nReport published → s3://{REPORTS_BUCKET}/latest.html")
    print("View at  http://localhost:8000/drift  or in Streamlit's 'Drift monitoring' tab")

    if summary.drift_share > args.threshold:
        print(f"\n❌ DRIFT THRESHOLD EXCEEDED " f"({pct:.1f}% > {args.threshold * 100:.0f}%)")
        return 1

    print(f"\n✅ Within threshold ({pct:.1f}% ≤ {args.threshold * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
